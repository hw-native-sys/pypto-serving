# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Serving-side engram embed wiring: executor source build and host lookup.

The tables never land on the device: the executor validates the original
checkpoint (geometry, tensors, tokenizer), and the runner serves the
host-side hasher plus zero-padded TP gathers the kernel later dequantizes and
all-reduces.
"""

from __future__ import annotations

import json
import logging
import struct
from pathlib import Path

import pytest
import torch

from pypto_serving.model.deepseek.engram_hash import (
    EngramHashLayout,
    build_compressed_token_map,
    find_next_prime,
)
from pypto_serving.model.deepseek.engram_loader import (
    ENGRAM_EMBED_BLOCK_SIZE,
    EngramEmbedLayout,
)
from pypto_serving.model.deepseek.engram_runner import EngramEmbedSource
from pypto_serving.model.deepseek_dspark.npu_executor import DeepSeekV4DSparkPyptoExecutor
from pypto_serving.model.deepseek_dspark.npu_runner import (
    DSparkCacheLayout,
    DSparkCompiledKernels,
    DSparkModelRunner,
    DSparkRopeTables,
)

# Small hash geometry so prime search stays fast: 3-gram over 2 heads, four
# buckets per layer drawn from a tiny vocabulary. The synthetic tokenizer
# folds each id onto itself, so the compressed space is the tokenizer vocab.
_HASH_MAX_NGRAM = 3
_HASH_N_HEADS = 2
_HASH_VOCAB = 101
_TOKENIZER_VOCAB = 80


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
    primes, seen = [], set()
    for _ in layer_ids:
        layer_total = 0
        for _ in range(_HASH_MAX_NGRAM - 1):
            current = _HASH_VOCAB - 1
            for _ in range(_HASH_N_HEADS):
                current = find_next_prime(current, seen)
                seen.add(current)
                layer_total += current
        primes.append(layer_total)
    return primes


def _hash_config(layer_ids: list[int], num_embeddings: list[int]) -> dict[str, object]:
    return {
        "text_config": {
            "engram_layer_ids": layer_ids,
            "engram_num_embeddings": num_embeddings,
            "engram_max_ngram_size": _HASH_MAX_NGRAM,
            "engram_vocab_size": _HASH_VOCAB,
            "engram_n_heads": _HASH_N_HEADS,
            "engram_head_dim": 64,
            "engram_compressed_vocab_size": _TOKENIZER_VOCAB,
            "engram_pad_token_id": 2,
        }
    }


def _write_tokenizer(root: Path, vocab: int) -> None:
    """Write a minimal WordLevel tokenizer whose compressed map is the identity."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    Tokenizer(WordLevel({f"t{i:05d}": i for i in range(vocab)}, unk_token=None)).save(
        str(root / "tokenizer.json")
    )


def _build_engram_checkpoint(
    root: Path,
    *,
    layer_ids: list[int] | None = None,
    with_config: bool = True,
    with_tokenizer: bool = True,
) -> tuple[dict[int, tuple[torch.Tensor, torch.Tensor]], dict[str, object]]:
    """Materialize a synthetic original-checkpoint engram directory.

    The declared table rows are the sums of the drawn prime buckets, so the
    config survives the layout rebuild's strict geometry check. Returns the
    raw uint8 payloads plus the written config for mutation in tests.
    """
    layer_ids = layer_ids or [1, 14]
    bucket_sums = _draw_bucket_sums(layer_ids)
    root.mkdir(parents=True, exist_ok=True)
    payloads: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    tensors: dict[str, tuple[str, list[int], bytes]] = {}
    head_dim = 64
    for layer_id, num_embeddings in zip(layer_ids, bucket_sums, strict=True):
        scale_groups = head_dim // ENGRAM_EMBED_BLOCK_SIZE
        weight = torch.randint(0, 256, (num_embeddings, head_dim), dtype=torch.uint8)
        scale = torch.randint(0, 256, (num_embeddings, scale_groups), dtype=torch.uint8)
        tensors[f"layers.{layer_id}.engram.embed.weight"] = (
            "F8_E4M3",
            [num_embeddings, head_dim],
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
    config = _hash_config(layer_ids, bucket_sums)
    if with_config:
        (root / "config.json").write_text(json.dumps(config))
    if with_tokenizer:
        _write_tokenizer(root, _TOKENIZER_VOCAB)
    return payloads, config


def _executor(checkpoint: str | None) -> DeepSeekV4DSparkPyptoExecutor:
    """Build an executor shell carrying only the engram checkpoint setting."""
    executor = DeepSeekV4DSparkPyptoExecutor.__new__(DeepSeekV4DSparkPyptoExecutor)
    executor._engram_checkpoint = checkpoint
    return executor


class TestBuildEngramEmbedSource:
    def test_none_without_checkpoint(self):
        assert _executor(None)._build_engram_embed_source(DSparkCacheLayout(), {}) is None

    def test_none_without_checkpoint_even_when_model_declares_engram(self, tmp_path, caplog):
        # The engram stays opt-in: a V4.1-Flash model without the flag only warns.
        _, config = _build_engram_checkpoint(tmp_path, layer_ids=[1])
        with caplog.at_level(logging.WARNING):
            source = _executor(None)._build_engram_embed_source(DSparkCacheLayout(), config)
        assert source is None
        assert "engram lookup stays off" in caplog.text

    def test_model_without_engram_layers_rejects_checkpoint(self, tmp_path):
        # Plain DeepSeek V4 declares no engram; the flag must not apply to it.
        _build_engram_checkpoint(tmp_path, layer_ids=[1])
        with pytest.raises(ValueError, match="model's config declares none"):
            _executor(str(tmp_path))._build_engram_embed_source(
                DSparkCacheLayout(), {"text_config": {}}
            )

    def test_checkpoint_geometry_mismatch_raises(self, tmp_path):
        # Model declares layers 1 and 14; the checkpoint only carries layer 1.
        _build_engram_checkpoint(tmp_path, layer_ids=[1])
        model_config = _hash_config([1, 14], _draw_bucket_sums([1, 14]))
        with pytest.raises(ValueError, match="does not match the served model"):
            _executor(str(tmp_path))._build_engram_embed_source(DSparkCacheLayout(), model_config)

    def test_builds_validated_source(self, tmp_path):
        _, config = _build_engram_checkpoint(tmp_path, layer_ids=[1, 14])
        source = _executor(str(tmp_path))._build_engram_embed_source(DSparkCacheLayout(), config)
        assert isinstance(source, EngramEmbedSource)
        assert source.checkpoint_dir == str(tmp_path)
        assert source.tp_size == DSparkCacheLayout().tp_size
        assert set(source.layouts) == {1, 14}
        assert isinstance(source.hash_layout, EngramHashLayout)
        assert source.hash_layout.layer_ids == (1, 14)
        assert source.hash_layout.n_hash_cols == (_HASH_MAX_NGRAM - 1) * _HASH_N_HEADS
        assert source.hash_layout.pad_token_id == 2
        # The declared rows are exactly the drawn bucket sums.
        for layer in (1, 14):
            drawn = sum(
                prime
                for per_ngram in source.hash_layout.primes[source.hash_layout.layer_ids.index(layer)]
                for prime in per_ngram
            )
            assert source.layouts[layer].num_embeddings == drawn

    def test_missing_config_raises(self, tmp_path):
        _, config = _build_engram_checkpoint(tmp_path, layer_ids=[1], with_config=False)
        with pytest.raises(FileNotFoundError, match="config.json"):
            _executor(str(tmp_path))._build_engram_embed_source(DSparkCacheLayout(), config)

    def test_config_without_engram_layers_raises(self, tmp_path):
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "config.json").write_text(json.dumps({"text_config": {}}))
        model_config = _hash_config([1], _draw_bucket_sums([1]))
        with pytest.raises(ValueError, match="no engram layers"):
            _executor(str(tmp_path))._build_engram_embed_source(DSparkCacheLayout(), model_config)

    def test_declared_but_missing_tensors_raise(self, tmp_path):
        # The config declares layers 1 and 14 but the safetensors hold only 1.
        _build_engram_checkpoint(tmp_path, layer_ids=[1])
        config = _hash_config([1, 14], _draw_bucket_sums([1, 14]))
        (tmp_path / "config.json").write_text(json.dumps(config))
        with pytest.raises(KeyError, match="layers.14.engram.embed.weight"):
            _executor(str(tmp_path))._build_engram_embed_source(DSparkCacheLayout(), config)

    def test_missing_tokenizer_raises(self, tmp_path):
        _, config = _build_engram_checkpoint(tmp_path, layer_ids=[1], with_tokenizer=False)
        with pytest.raises(FileNotFoundError, match="tokenizer.json"):
            _executor(str(tmp_path))._build_engram_embed_source(DSparkCacheLayout(), config)

    def test_wrong_prime_geometry_raises(self, tmp_path):
        # Declared rows disagree with the buckets the config draws.
        _build_engram_checkpoint(tmp_path, layer_ids=[1])
        config = _hash_config([1], [_draw_bucket_sums([1])[0] + 1])
        (tmp_path / "config.json").write_text(json.dumps(config))
        with pytest.raises(ValueError, match="rebuilt buckets"):
            _executor(str(tmp_path))._build_engram_embed_source(DSparkCacheLayout(), config)


def _engram_runner(
    checkpoint_dir: Path,
) -> tuple[DSparkModelRunner, dict[int, tuple[torch.Tensor, torch.Tensor]]]:
    """Build a minimal DSpark runner over a synthetic engram checkpoint."""
    payloads, _ = _build_engram_checkpoint(checkpoint_dir, layer_ids=[1, 14])
    config = json.loads((checkpoint_dir / "config.json").read_text())
    hash_layout = EngramHashLayout.from_config(config)
    layouts = {
        layer: EngramEmbedLayout(
            hash_layout.num_embeddings[index], 64, tp_size=DSparkCacheLayout().tp_size
        )
        for index, layer in enumerate(hash_layout.layer_ids)
    }
    rows = torch.arange(512 * 64, dtype=torch.float32).reshape(512, 64)
    rope = DSparkRopeTables(
        max_position=512,
        swa_cos=rows.to(torch.bfloat16),
        swa_sin=(rows + 1).to(torch.bfloat16),
        ratio4_cos=(rows + 2).to(torch.bfloat16),
        ratio4_sin=(rows + 3).to(torch.bfloat16),
        ratio128_cos=(rows + 4).to(torch.bfloat16),
        ratio128_sin=(rows + 5).to(torch.bfloat16),
        ratio128_half_cos=rows[:, :32] + 6,
        ratio128_half_sin=rows[:, :32] + 7,
    )
    source = EngramEmbedSource(
        checkpoint_dir=str(checkpoint_dir),
        tp_size=DSparkCacheLayout().tp_size,
        layouts=layouts,
        hash_layout=hash_layout,
    )
    runner = DSparkModelRunner(
        compiled=DSparkCompiledKernels(
            layout=DSparkCacheLayout(),
            model_dir="unused",
            weight_map={},
            weight_store=None,
            compress_ratios=(0,) * 43,
            layer_plan=(),
            kernel_dir="unused",
            rope=rope,
            num_speculative_tokens=0,
            engram_embed=source,
        )
    )
    return runner, payloads


class TestEngramHostLookup:
    def _hash_ids_for(self, runner: DSparkModelRunner, tokens: int) -> dict[int, torch.Tensor]:
        hasher = runner.engram_hasher()
        generator = torch.Generator().manual_seed(31)
        input_ids = torch.randint(0, _TOKENIZER_VOCAB, (tokens,), dtype=torch.int64, generator=generator)
        ids, _ = hasher.hash_tokens(input_ids)
        # [tokens, n_layers, n_hash_cols] -> per-layer [tokens, n_hash_cols].
        return {
            layer: ids[:, index]
            for index, layer in enumerate(hasher.layout.layer_ids)
        }

    def test_hasher_built_from_tokenizer(self, tmp_path):
        from tokenizers import Tokenizer

        runner, _ = _engram_runner(tmp_path)
        hasher = runner.engram_hasher()
        backend = Tokenizer.from_file(str(tmp_path / "tokenizer.json"))
        token_map = build_compressed_token_map(backend)
        assert hasher.pad_id == int(token_map[2])
        assert hasher.layout.n_hash_cols == (_HASH_MAX_NGRAM - 1) * _HASH_N_HEADS
        # Cached: a second access returns the same state, no rebuild.
        assert runner.engram_hasher() is hasher

    def test_partials_cover_the_table_exactly_once(self, tmp_path):
        runner, payloads = _engram_runner(tmp_path)
        hash_ids = self._hash_ids_for(runner, tokens=9)
        tp_size = DSparkCacheLayout().tp_size

        partials = runner.gather_engram_partials(hash_ids)

        assert set(partials) == set(range(tp_size))
        for layer, (weight, scale) in payloads.items():
            rows = hash_ids[layer]
            summed_weight = torch.zeros_like(weight[rows], dtype=torch.uint8)
            summed_scale = torch.zeros_like(scale[rows], dtype=torch.uint8)
            for tp_rank in range(tp_size):
                partial = partials[tp_rank][layer]
                assert partial.weight.shape[-1] == weight.shape[1]
                summed_weight += partial.weight.view(torch.uint8)
                summed_scale += partial.scale.view(torch.uint8)
            # Each hash column is non-zero on exactly one rank, so the byte
            # sums reproduce the direct full-table gather.
            assert torch.equal(summed_weight, weight[rows])
            assert torch.equal(summed_scale, scale[rows])

    def test_partials_zero_pad_outside_their_shard(self, tmp_path):
        runner, payloads = _engram_runner(tmp_path)
        hash_ids = self._hash_ids_for(runner, tokens=7)
        layouts = runner._compiled.engram_embed.layouts

        partials = runner.gather_engram_partials(hash_ids)

        for tp_rank, per_layer in partials.items():
            for layer, partial in per_layer.items():
                start, end = layouts[layer].row_range(tp_rank)
                rows = hash_ids[layer]
                hit = (rows >= start) & (rows < end)
                weight, scale = payloads[layer]
                assert torch.equal(
                    partial.weight.view(torch.uint8)[hit], weight[rows[hit]]
                )
                assert torch.equal(
                    partial.scale.view(torch.uint8)[hit], scale[rows[hit]]
                )
                assert bool((partial.weight.view(torch.uint8)[~hit] == 0).all())
                assert bool((partial.scale.view(torch.uint8)[~hit] == 0).all())

    def test_dp_groups_gather_independently(self, tmp_path):
        """Two DP groups share one hasher and tables but keep their own requests.

        Each group calls gather_engram_partials with its own hash ids; the
        per-rank mapping stays ``tp_rank = rank % tp_size`` and neither
        group's gather may leak rows into the other's partials.
        """
        runner, payloads = _engram_runner(tmp_path)
        tp_size = DSparkCacheLayout().tp_size
        hasher = runner.engram_hasher()
        group_ids = []
        for seed in (41, 43):
            generator = torch.Generator().manual_seed(seed)
            input_ids = torch.randint(
                0, _TOKENIZER_VOCAB, (6,), dtype=torch.int64, generator=generator
            )
            ids, _ = hasher.hash_tokens(input_ids)
            group_ids.append(
                {layer: ids[:, index] for index, layer in enumerate(hasher.layout.layer_ids)}
            )

        # Each DP group runs its own gather over the shared host state.
        group_partials = [runner.gather_engram_partials(ids) for ids in group_ids]

        for ids, partials in zip(group_ids, group_partials):
            assert set(partials) == set(range(tp_size))
            for layer, (weight, _) in payloads.items():
                rows = ids[layer]
                summed_weight = torch.zeros_like(weight[rows], dtype=torch.uint8)
                for tp_rank in range(tp_size):
                    summed_weight += partials[tp_rank][layer].weight.view(torch.uint8)
                assert torch.equal(summed_weight, weight[rows])

        # Re-gathering group 0 after group 1 must return the same bytes: the
        # shared hasher/tables carry no per-group state to clobber. Compared
        # as raw bytes -- the synthetic table holds NaN payload bytes, and a
        # value-wise fp8 equal would be False under NaN != NaN.
        again = runner.gather_engram_partials(group_ids[0])
        for tp_rank in range(tp_size):
            for layer in group_ids[0]:
                assert torch.equal(
                    group_partials[0][tp_rank][layer].weight.view(torch.uint8),
                    again[tp_rank][layer].weight.view(torch.uint8),
                )

    def test_raises_without_source(self, tmp_path):
        runner, _ = _engram_runner(tmp_path)
        runner._compiled.engram_embed = None
        with pytest.raises(RuntimeError, match="without an engram source"):
            runner.engram_hasher()

    def test_unknown_and_missing_layers_raise(self, tmp_path):
        runner, _ = _engram_runner(tmp_path)
        hash_ids = self._hash_ids_for(runner, tokens=3)
        with pytest.raises(KeyError, match="unknown engram layers"):
            runner.gather_engram_partials({99: hash_ids[1]})
        with pytest.raises(KeyError, match="Missing hash ids"):
            runner.gather_engram_partials({1: hash_ids[1]})
