# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Kernel-ABI wiring for the host-side engram lookup.

The runner gathers zero-padded TP partials straight from the mapped
checkpoint; the pypto-lib engram kernel consumes them as one fp8
``embed_weight [T, 6144]`` payload plus a transposed e8m0
``embed_scale [192, T]`` wire per rank (the raw checkpoint bytes; the
kernel decodes them on device).
These tests pin that boundary with kernel-shaped tables: the
dequantized partial sum must reproduce the full-table gather bit-exactly,
the scale bytes must stay valid e8m0 (exact powers of two on decode), and
the zero padding must survive the wire conversion.

The on-device guard is opt-in: set ``PYPTO_ENGRAM_KERNEL_TP=1|2|4|8`` (and
optionally ``PYPTO_ENGRAM_KERNEL_DEVICES=0,1,...``) to ship the
serving-produced tensors to the engram kernel on real A5 hardware.
"""

from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

import pytest
import torch

from pypto_serving.model.deepseek.engram_hash import (
    EngramHashLayout,
    NgramHasher,
    build_compressed_token_map,
    find_next_prime,
)
from pypto_serving.model.deepseek.engram_loader import (
    ENGRAM_EMBED_BLOCK_SIZE,
    EngramEmbedCheckpoint,
    EngramEmbedLayout,
    EngramEmbedPartial,
    parse_engram_embed_layouts,
)
from pypto_serving.model.deepseek.engram_runner import EngramEmbedSource, EngramModelRunner

# Kernel-shaped geometry: 24 hash columns of 256 dims (the real 6144-wide
# lookup), so the wire tensors need no reshaping adapter. Primes near 997
# keep the bucket search fast.
_KERNEL_MAX_NGRAM = 3
_KERNEL_N_HEADS = 12
_KERNEL_HEAD_DIM = 256
_KERNEL_VOCAB = 997
_LAYER_IDS = [1, 14]
# The synthetic tokenizer folds each id onto itself, so the compressed space
# is the tokenizer vocab.
_TOKENIZER_VOCAB = 512
_SCALE_EXP_LO, _SCALE_EXP_HI = 121, 134  # e8m0 bytes for 2**-6 .. 2**6
_N_HASH_COLS = (_KERNEL_MAX_NGRAM - 1) * _KERNEL_N_HEADS  # 24
_FLAT_WIDTH = _N_HASH_COLS * _KERNEL_HEAD_DIM  # 6144
_SCALE_COLS = _FLAT_WIDTH // ENGRAM_EMBED_BLOCK_SIZE  # 192

_PYPTO_LIB = Path("/home/pyptouser/liuchao/source/pypto-lib")
_ON_DEVICE_TP = os.environ.get("PYPTO_ENGRAM_KERNEL_TP")
# Real-checkpoint validation (e.g. /srv/models/DeepSeek-V4.1-Flash): original
# 384M-row tables, tokenizer-built token map, real wkv / gate weights.
_MODEL_DIR = os.environ.get("PYPTO_ENGRAM_MODEL_DIR")
# Synthetic checkpoints default to four shards; the real-checkpoint tests
# exercise the production TP8 sharding of the 384M-row tables.
_DEFAULT_TP_SIZE = 4
_REAL_TP_SIZE = 8


def _write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    """Write one minimal safetensors shard (header JSON + raw payloads)."""
    header: dict[str, dict[str, object]] = {}
    blob = bytearray()
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [len(blob), len(blob) + len(data)],
        }
        blob += data
    header_bytes = json.dumps(header).encode()
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        handle.write(blob)


def _draw_bucket_sums(layer_ids: list[int]) -> list[int]:
    """Row counts implied by the prime buckets the config geometry draws."""
    bucket_sums, seen = [], set()
    for _ in layer_ids:
        layer_total = 0
        for _ in range(_KERNEL_MAX_NGRAM - 1):
            current = _KERNEL_VOCAB - 1
            for _ in range(_KERNEL_N_HEADS):
                current = find_next_prime(current, seen)
                seen.add(current)
                layer_total += current
        bucket_sums.append(layer_total)
    return bucket_sums


def _write_tokenizer(root: Path, vocab: int) -> None:
    """Write a minimal WordLevel tokenizer whose compressed map is the identity."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    Tokenizer(WordLevel({f"t{i:05d}": i for i in range(vocab)}, unk_token=None)).save(
        str(root / "tokenizer.json")
    )


def _build_kernel_checkpoint(
    root: Path,
) -> tuple[dict[int, tuple[torch.Tensor, torch.Tensor]], dict[str, object]]:
    """Materialize a synthetic original-checkpoint engram directory.

    Payload bytes come from quantizing real values (never an e4m3 NaN) and
    scale bytes stay in ``[_SCALE_EXP_LO, _SCALE_EXP_HI)`` (never an e8m0
    NaN), so the dequantized lookup math stays finite. Returns the raw
    uint8 payloads plus the written config.
    """
    bucket_sums = _draw_bucket_sums(_LAYER_IDS)
    root.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(21)
    payloads: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    tensors: dict[str, tuple[str, list[int], bytes]] = {}
    scale_groups = _KERNEL_HEAD_DIM // ENGRAM_EMBED_BLOCK_SIZE
    for layer_id, num_embeddings in zip(_LAYER_IDS, bucket_sums, strict=True):
        weight = (
            torch.randn(num_embeddings, _KERNEL_HEAD_DIM, generator=generator)
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
        )
        scale = torch.randint(
            _SCALE_EXP_LO,
            _SCALE_EXP_HI,
            (num_embeddings, scale_groups),
            dtype=torch.uint8,
            generator=generator,
        )
        tensors[f"layers.{layer_id}.engram.embed.weight"] = (
            "F8_E4M3",
            [num_embeddings, _KERNEL_HEAD_DIM],
            weight.contiguous().numpy().tobytes(),
        )
        tensors[f"layers.{layer_id}.engram.embed.scale"] = (
            "F8_E8M0",
            [num_embeddings, scale_groups],
            scale.contiguous().numpy().tobytes(),
        )
        payloads[layer_id] = (weight, scale)
    _write_safetensors(root / "model-00001-of-00001.safetensors", tensors)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "model-00001-of-00001.safetensors" for name in tensors}})
    )
    config = {
        "text_config": {
            "engram_layer_ids": _LAYER_IDS,
            "engram_num_embeddings": bucket_sums,
            "engram_max_ngram_size": _KERNEL_MAX_NGRAM,
            "engram_vocab_size": _KERNEL_VOCAB,
            "engram_n_heads": _KERNEL_N_HEADS,
            "engram_head_dim": _KERNEL_HEAD_DIM,
            "engram_compressed_vocab_size": _TOKENIZER_VOCAB,
            "engram_pad_token_id": 2,
        }
    }
    (root / "config.json").write_text(json.dumps(config))
    _write_tokenizer(root, _TOKENIZER_VOCAB)
    return payloads, config


def _kernel_runner(
    checkpoint_dir: Path, *, tp_size: int = _DEFAULT_TP_SIZE
) -> tuple[EngramModelRunner, dict[int, tuple[torch.Tensor, torch.Tensor]]]:
    """Build a standalone engram runner over a kernel-shaped checkpoint."""
    payloads, _ = _build_kernel_checkpoint(checkpoint_dir)
    config = json.loads((checkpoint_dir / "config.json").read_text())
    hash_layout = EngramHashLayout.from_config(config)
    layouts = {
        layer: EngramEmbedLayout(
            hash_layout.num_embeddings[index], _KERNEL_HEAD_DIM, tp_size=tp_size
        )
        for index, layer in enumerate(hash_layout.layer_ids)
    }
    source = EngramEmbedSource(
        checkpoint_dir=str(checkpoint_dir),
        tp_size=tp_size,
        layouts=layouts,
        hash_layout=hash_layout,
    )
    return EngramModelRunner(source), payloads


def _hash_ids_for(
    runner: EngramModelRunner, tokens: int
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    """Hash a fixed token stream into per-layer [tokens, n_hash_cols] ids."""
    hasher = runner.hasher()
    generator = torch.Generator().manual_seed(31)
    input_ids = torch.randint(
        0, _TOKENIZER_VOCAB, (tokens,), dtype=torch.int64, generator=generator
    )
    ids, _ = hasher.hash_tokens(input_ids)
    # [tokens, n_layers, n_hash_cols] -> per-layer [tokens, n_hash_cols].
    return (
        {layer: ids[:, index] for index, layer in enumerate(hasher.layout.layer_ids)},
        input_ids,
    )


def _dequant(partial: EngramEmbedPartial) -> torch.Tensor:
    """fp8 payload x e8m0 scale -> [T, 24, 256] fp32, as the kernel does."""
    payload = partial.weight.float()
    exponents = partial.scale.view(torch.uint8).to(torch.int32) - 127
    scales = torch.pow(2.0, exponents.float())
    return payload * scales.repeat_interleave(ENGRAM_EMBED_BLOCK_SIZE, dim=-1)


def _kernel_wire(
    partials: dict[int, dict[int, EngramEmbedPartial]], layer: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Serving partials -> kernel ABI: embed_weight [TP,T,6144] + embed_scale [TP,192,T]."""
    embed_weight_ranks, embed_scale_ranks = [], []
    for tp_rank in sorted(partials):
        partial = partials[tp_rank][layer]
        tokens = partial.weight.shape[0]
        embed_weight = partial.weight.view(torch.uint8).reshape(tokens, _FLAT_WIDTH)
        embed_weight_ranks.append(embed_weight.view(torch.float8_e4m3fn))
        # Raw checkpoint e8m0 bytes, transposed to [SCALE_COLS, T]; the
        # kernel rebuilds the fp32 scale on device (byte << 23, the byte
        # being the biased fp32 exponent field).
        embed_scale = partial.scale.view(torch.uint8).reshape(tokens, _SCALE_COLS)
        embed_scale_ranks.append(embed_scale.t().contiguous().view(torch.float8_e8m0fnu))
    return torch.stack(embed_weight_ranks), torch.stack(embed_scale_ranks)


_REAL_STATE: dict[int, tuple[EngramEmbedCheckpoint, dict, NgramHasher]] = {}


def _real_state(
    tp_size: int,
) -> tuple[EngramEmbedCheckpoint, dict, NgramHasher]:
    """Real checkpoint: rebuilt geometry, mmap'd tables, tokenizer token map.

    The token map is rebuilt from the checkpoint's raw ``tokenizer.json``
    (exactly what training decoded with); the compressed-id count is a hard
    check of the rebuild. Cached per tp_size.
    """
    if tp_size in _REAL_STATE:
        return _REAL_STATE[tp_size]
    from tokenizers import Tokenizer

    model_dir = Path(_MODEL_DIR)
    config = json.loads((model_dir / "config.json").read_text())
    # Strictly re-derives the 384M-row prime buckets from the declared config.
    hash_layout = EngramHashLayout.from_config(config)
    layouts = parse_engram_embed_layouts(config, tp_size=tp_size)
    checkpoint = EngramEmbedCheckpoint(model_dir)
    backend = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    token_map = build_compressed_token_map(backend)
    hasher = NgramHasher(hash_layout, token_map, pad_token_id=hash_layout.pad_token_id)
    _REAL_STATE[tp_size] = (checkpoint, layouts, hasher)
    return _REAL_STATE[tp_size]


def _real_kernel_weights(
    checkpoint: EngramEmbedCheckpoint, layer: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantize the real wkv (fp8, per-32x32-block e8m0) and fold q/k.

    Returns the kernel's ``wkv_weight`` ``[6144, 25600]`` bf16 and ``weight``
    ``[4, 5120]`` fp32.
    """
    weight_name = f"layers.{layer}.engram.wkv.weight"
    scale_name = f"layers.{layer}.engram.wkv.scale"
    wkv_fp8 = checkpoint._mapped_shard(weight_name).row_view(
        weight_name, row_start=0, row_count=25600, row_bytes=6144,
        columns=6144, dtype="F8_E4M3",
    ).view(torch.float8_e4m3fn)
    wkv_scale = checkpoint._mapped_shard(scale_name).row_view(
        scale_name, row_start=0, row_count=800, row_bytes=192,
        columns=192, dtype="F8_E8M0",
    ).view(torch.uint8)
    scales = torch.pow(2.0, wkv_scale.to(torch.int32).float() - 127)
    blocks = scales.repeat_interleave(32, dim=0).repeat_interleave(32, dim=1)
    wkv = (wkv_fp8.float() * blocks).t().contiguous().to(torch.bfloat16)

    gate = {}
    for kind in ("q_weight", "k_weight"):
        name = f"layers.{layer}.engram.{kind}"
        gate[kind] = checkpoint._mapped_shard(name).row_view(
            name, row_start=0, row_count=4, row_bytes=5120 * 2,
            columns=5120, dtype="BF16",
        )
    weight = gate["q_weight"].float() * gate["k_weight"].float()
    return wkv, weight


def _real_hash_ids(hasher: NgramHasher, tokens: int) -> torch.Tensor:
    """Hash a fixed token stream into per-layer [tokens, n_hash_cols] ids."""
    generator = torch.Generator().manual_seed(31)
    input_ids = torch.randint(
        0, hasher.token_map.numel(), (tokens,), dtype=torch.int64, generator=generator
    )
    ids, _ = hasher.hash_tokens(input_ids)
    return {layer: ids[:, index] for index, layer in enumerate(hasher.layout.layer_ids)}


@pytest.mark.skipif(_MODEL_DIR is None, reason="set PYPTO_ENGRAM_MODEL_DIR to the real checkpoint")
class TestRealCheckpoint:
    def test_token_map_rebuild_matches_compressed_vocab(self):
        _, _, hasher = _real_state(_REAL_TP_SIZE)
        # The tokenizer rebuild must land exactly on the trained 99092 ids.
        assert int(hasher.token_map.max().item()) + 1 == hasher.layout.compressed_vocab_size

    def test_hash_matches_reference_chunked(self):
        _, _, hasher = _real_state(_REAL_TP_SIZE)
        reference_dir = Path("/home/pyptouser/liuchao/source/DeepSeek-V4.1-Flash/inference")
        if str(reference_dir) not in sys.path:
            sys.path.insert(0, str(reference_dir))
        from types import SimpleNamespace

        import numpy as np
        from engram import (
            EngramLayout,
            NgramHashState,
            compute_hash_multipliers as reference_multipliers,
        )

        text = json.loads((Path(_MODEL_DIR) / "config.json").read_text())["text_config"]
        args = SimpleNamespace(
            engram_layer_ids=tuple(text["engram_layer_ids"]),
            engram_num_embeddings=tuple(text["engram_num_embeddings"]),
            engram_max_ngram_size=text["engram_max_ngram_size"],
            engram_vocab_size=text["engram_vocab_size"],
            engram_n_heads=text["engram_n_heads"],
            engram_head_dim=text["engram_head_dim"],
        )
        layout = EngramLayout.from_args(args)
        reference = object.__new__(NgramHashState)
        reference.layout = layout
        reference.pad_id = int(hasher.token_map[2].item())
        reference.primes = torch.tensor(layout.primes)
        flat = [[p for per_ngram in layer for p in per_ngram] for layer in layout.primes]
        reference.offsets = torch.tensor(
            np.array([np.cumsum([0, *sizes[:-1]]) for sizes in flat])
        )
        reference.multipliers = reference_multipliers(
            layout.layer_ids, layout.max_ngram_size, hasher.layout.compressed_vocab_size
        )
        reference.token_map = hasher.token_map
        reference.cache = torch.empty(1, 64, dtype=torch.int64)

        generator = torch.Generator().manual_seed(31)
        input_ids = torch.randint(
            0, hasher.token_map.numel(), (8,), dtype=torch.int64, generator=generator
        )
        history, pieces = None, []
        start = 0
        for chunk in (input_ids[:5], input_ids[5:6], input_ids[6:7], input_ids[7:8]):
            ids, history = hasher.hash_tokens(chunk, history=history)
            expected = reference.forward(chunk[None], start_pos=start)[0]
            assert torch.equal(ids, expected), f"chunk at offset {start}"
            pieces.append(ids)
            start += chunk.numel()
        whole, _ = hasher.hash_tokens(input_ids)
        assert torch.equal(torch.cat(pieces), whole)

    def test_partial_sum_matches_full_gather(self):
        checkpoint, layouts, hasher = _real_state(_REAL_TP_SIZE)
        tp_size = _REAL_TP_SIZE
        layer = hasher.layout.layer_ids[0]
        layout = layouts[layer]
        ids = _real_hash_ids(hasher, tokens=8)[layer]

        weight = checkpoint._mapped_shard(f"layers.{layer}.engram.embed.weight").row_view(
            f"layers.{layer}.engram.embed.weight",
            row_start=0, row_count=layout.num_embeddings,
            row_bytes=layout.head_dim, columns=layout.head_dim, dtype="F8_E4M3",
        )
        scale = checkpoint._mapped_shard(f"layers.{layer}.engram.embed.scale").row_view(
            f"layers.{layer}.engram.embed.scale",
            row_start=0, row_count=layout.num_embeddings,
            row_bytes=layout.scale_groups, columns=layout.scale_groups, dtype="F8_E8M0",
        )
        gathered = weight[ids].view(torch.float8_e4m3fn).float() * torch.pow(
            2.0, scale[ids].view(torch.uint8).to(torch.int32).float() - 127
        ).repeat_interleave(ENGRAM_EMBED_BLOCK_SIZE, dim=-1)
        partial_sum = torch.stack([
            _dequant(checkpoint.gather_partial_rows(layer, layout, ids, tp_rank=rank))
            for rank in range(tp_size)
        ]).sum(dim=0)
        assert torch.equal(partial_sum, gathered)

    def test_wire_from_real_partials(self):
        checkpoint, layouts, hasher = _real_state(_REAL_TP_SIZE)
        tp_size = _REAL_TP_SIZE
        layer = hasher.layout.layer_ids[0]
        ids = _real_hash_ids(hasher, tokens=8)[layer]
        partials = {
            rank: {layer: checkpoint.gather_partial_rows(layer, layouts[layer], ids, tp_rank=rank)}
            for rank in range(tp_size)
        }

        embed_weight, embed_scale = _kernel_wire(partials, layer)

        assert embed_weight.shape == (tp_size, 8, _FLAT_WIDTH)
        assert embed_scale.shape == (tp_size, _SCALE_COLS, 8)
        log2 = torch.log2(embed_scale.float())
        assert torch.equal(log2, log2.round())
        layout = layouts[layer]
        for rank in range(tp_size):
            start, end = layout.row_range(rank)
            hit = (ids >= start) & (ids < end)
            wire = embed_weight[rank].view(torch.uint8).reshape(8, _N_HASH_COLS, -1)
            assert torch.equal(wire[hit], partials[rank][layer].weight.view(torch.uint8)[hit])
            assert bool((wire[~hit] == 0).all())


class TestDequantizedGather:
    def test_partial_sum_matches_full_gather(self, tmp_path):
        runner, payloads = _kernel_runner(tmp_path)
        hash_ids, _ = _hash_ids_for(runner, tokens=8)
        tp_size = runner.source.tp_size

        partials = runner.gather_partials(hash_ids)

        for layer, (weight, scale) in payloads.items():
            ids = hash_ids[layer]
            gathered = weight[ids].view(torch.float8_e4m3fn).float() * torch.pow(
                2.0, scale[ids].to(torch.int32).float() - 127
            ).repeat_interleave(ENGRAM_EMBED_BLOCK_SIZE, dim=-1)
            partial_sum = torch.stack(
                [_dequant(partials[tp_rank][layer]) for tp_rank in range(tp_size)]
            ).sum(dim=0)
            # Every element has exactly one non-zero contributor, so the
            # partial sum is the full gather bit for bit.
            assert torch.equal(partial_sum, gathered)


class TestKernelWire:
    def test_wire_flattens_payload_and_transposes_scales(self, tmp_path):
        runner, _ = _kernel_runner(tmp_path)
        hash_ids, _ = _hash_ids_for(runner, tokens=8)
        tp_size = runner.source.tp_size
        partials = runner.gather_partials(hash_ids)

        embed_weight, embed_scale = _kernel_wire(partials, layer=1)

        assert embed_weight.shape == (tp_size, 8, _FLAT_WIDTH)
        assert embed_weight.dtype == torch.float8_e4m3fn
        assert embed_scale.shape == (tp_size, _SCALE_COLS, 8)
        assert embed_scale.dtype == torch.float8_e8m0fnu
        partial = partials[0][1]
        # Flatten order: (hash column, head dim), matching the kernel layout.
        assert torch.equal(
            embed_weight[0].view(torch.uint8),
            partial.weight.view(torch.uint8).reshape(8, _FLAT_WIDTH),
        )
        # Transposed: wire[g, t] scales the flat columns [32g, 32g + 32);
        # the wire carries the checkpoint's raw e8m0 bytes unchanged.
        expected = partial.scale.view(torch.uint8).reshape(8, _SCALE_COLS)
        assert torch.equal(embed_scale[0].t().view(torch.uint8), expected)

    def test_wire_scales_are_exact_powers_of_two(self, tmp_path):
        runner, _ = _kernel_runner(tmp_path)
        hash_ids, _ = _hash_ids_for(runner, tokens=8)
        partials = runner.gather_partials(hash_ids)

        _, embed_scale = _kernel_wire(partials, layer=1)

        # e8m0 decodes to exact powers of two, so the kernel's byte << 23
        # rebuild recovers the scale value with no rounding.
        log2 = torch.log2(embed_scale.float())
        assert torch.equal(log2, log2.round())

    def test_wire_zero_padding_stays_zero(self, tmp_path):
        runner, _ = _kernel_runner(tmp_path)
        hash_ids, _ = _hash_ids_for(runner, tokens=8)
        partials = runner.gather_partials(hash_ids)
        layouts = runner.source.layouts

        embed_weight, _ = _kernel_wire(partials, layer=1)

        ids = hash_ids[1]
        for tp_rank, per_layer in partials.items():
            start, end = layouts[1].row_range(tp_rank)
            hit = (ids >= start) & (ids < end)
            wire = embed_weight[tp_rank].view(torch.uint8).reshape(8, _N_HASH_COLS, -1)
            assert torch.equal(wire[hit], per_layer[1].weight.view(torch.uint8)[hit])
            assert bool((wire[~hit] == 0).all())

    def test_chunked_hash_chain_feeds_gather(self, tmp_path):
        runner, _ = _kernel_runner(tmp_path)
        hash_ids, input_ids = _hash_ids_for(runner, tokens=8)

        hasher = runner.hasher()
        history, pieces = None, []
        for chunk in (input_ids[:5], input_ids[5:6], input_ids[6:7], input_ids[7:8]):
            ids, history = hasher.hash_tokens(chunk, history=history)
            pieces.append(ids)
        chained = torch.cat(pieces)
        for index, layer in enumerate(hasher.layout.layer_ids):
            assert torch.equal(chained[:, index], hash_ids[layer])

        partials = runner.gather_partials(
            {layer: chained[:, index] for index, layer in enumerate(hasher.layout.layer_ids)}
        )
        assert set(partials) == set(range(runner.source.tp_size))


@pytest.mark.skipif(
    _ON_DEVICE_TP is None,
    reason="set PYPTO_ENGRAM_KERNEL_TP=1|2|4|8 (plus PYPTO_ENGRAM_KERNEL_DEVICES) for A5",
)
def test_kernel_consumes_serving_wire_on_device(tmp_path, monkeypatch):
    """Ship the serving-produced wire tensors to the pypto-lib engram kernel."""
    tp_size = int(_ON_DEVICE_TP)
    device_ids = [
        int(d)
        for d in os.environ.get(
            "PYPTO_ENGRAM_KERNEL_DEVICES", ",".join(str(d) for d in range(tp_size))
        ).split(",")
    ]
    assert len(device_ids) == tp_size
    # The kernel module parses --tp from sys.argv at import time.
    monkeypatch.setattr(sys, "argv", [sys.argv[0], "--tp", str(tp_size)])
    if str(_PYPTO_LIB) not in sys.path:
        sys.path.insert(0, str(_PYPTO_LIB))
    from golden import ScalarSpec, TensorSpec, ratio_allclose, run
    from models.deepseek_v4_1_flash import engram as kernel
    from pypto.ir import DistributedConfig

    if _MODEL_DIR is not None:
        # Real checkpoint: original tables, tokenizer token map, real weights.
        checkpoint, layouts, hasher = _real_state(tp_size)
        layer = hasher.layout.layer_ids[0]
        ids = _real_hash_ids(hasher, tokens=8)[layer]
        partials = {
            rank: {
                layer: checkpoint.gather_partial_rows(
                    layer, layouts[layer], ids, tp_rank=rank
                )
            }
            for rank in range(tp_size)
        }
        embed_weight, embed_scale = _kernel_wire(partials, layer)
        wkv, weight = _real_kernel_weights(checkpoint, layer)
    else:
        runner, _ = _kernel_runner(tmp_path, tp_size=tp_size)
        hash_ids, _ = _hash_ids_for(runner, tokens=8)
        partials = runner.gather_partials(hash_ids)
        embed_weight, embed_scale = _kernel_wire(partials, layer=1)

        generator = torch.Generator().manual_seed(3)
        wkv = (
            torch.randn(kernel.ENGRAM_K, kernel.KV_OUT, generator=generator)
            / (kernel.ENGRAM_K**0.5)
        ).to(torch.bfloat16)
        weight = (torch.rand(kernel.HC_MULT, kernel.D, generator=generator) + 0.5) * (
            torch.rand(kernel.HC_MULT, kernel.D, generator=generator) + 0.5
        )
    x = torch.randn(
        8, kernel.HC_MULT, kernel.D, generator=torch.Generator().manual_seed(3)
    ).to(torch.bfloat16)

    def per_rank(value: torch.Tensor) -> torch.Tensor:
        return value.unsqueeze(0).repeat(tp_size, *([1] * value.ndim))

    specs = [
        TensorSpec(
            "embed_weight", list(embed_weight.shape), torch.float8_e4m3fn,
            init_value=lambda: embed_weight,
        ),
        TensorSpec(
            "embed_scale", list(embed_scale.shape), torch.float8_e8m0fnu,
            init_value=lambda: embed_scale,
        ),
        TensorSpec(
            "wkv_weight", [tp_size, kernel.ENGRAM_K, kernel.KV_OUT], torch.bfloat16,
            init_value=lambda: per_rank(wkv),
        ),
        TensorSpec(
            "weight", [tp_size, kernel.HC_MULT, kernel.D], torch.float32,
            init_value=lambda: per_rank(weight),
        ),
        TensorSpec(
            "x", [tp_size, 8, kernel.HC_MULT, kernel.D], torch.bfloat16,
            init_value=lambda: per_rank(x),
        ),
        TensorSpec("out", [tp_size, 8, kernel.HC_MULT, kernel.D], torch.bfloat16),
        # Persistent-signal epoch: the kernel takes it as a runtime scalar.
        ScalarSpec("epoch", torch.int32, 1, compile_runtime=True),
    ]
    result = run(
        fn=kernel.engram_group,
        specs=specs,
        golden_fn=kernel.golden_engram,
        config={
            "platform": "a5",
            "distributed_config": DistributedConfig(device_ids=device_ids, num_sub_workers=0),
        },
        rtol=1e-3,
        atol=1e-3,
        compare_fn={"out": kernel._precision_compare("out", ratio_allclose(atol=1e-3, rtol=1e-2))},
    )
    assert result.passed, result.error
