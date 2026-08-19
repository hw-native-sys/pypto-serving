# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# A synthetic on-disk DeepSeekV4 W8A8 checkpoint, for exercising the weight pipeline offline.
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from pypto_serving.model.deepseek.weight_loader import (
    _DEEPSEEK_V4_ATTENTION_OUT,
    _DEEPSEEK_V4_CSA_COMPRESS_RATIO,
    _DEEPSEEK_V4_CSA_INNER_OUT_DIM,
    _DEEPSEEK_V4_CSA_MAIN_OUT_DIM,
    _DEEPSEEK_V4_HADAMARD_IDX_DIM,
    _DEEPSEEK_V4_HCA_COMPRESS_RATIO,
    _DEEPSEEK_V4_HCA_MAIN_OUT_DIM,
    _DEEPSEEK_V4_HEAD_DIM,
    _DEEPSEEK_V4_HIDDEN_SIZE,
    _DEEPSEEK_V4_Q_LORA,
    DeepSeekV4WeightStore,
    deepseek_v4_layer_weight_names,
)

# Suffix -> (shape, dtype), with the shapes the real packer already accepts in
# test_model_components.py. Which of these a given layer needs is NOT decided here: it comes
# from `deepseek_v4_layer_weight_names`, so the fixture follows the checkpoint contract
# instead of duplicating it — and a contract that grows a tensor fails this fixture loudly
# rather than quietly testing the old shape.
_EXPERT_KEY = "ffn.experts.{}."
_SHAPES: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
    "hc_attn_fn": ((1, 4), torch.float32),
    "hc_attn_scale": ((3,), torch.float32),
    "hc_attn_base": ((1,), torch.float32),
    "attn_norm.weight": ((4,), torch.bfloat16),
    "attn.wq_a.weight": ((2, 4), torch.bfloat16),
    "attn.wq_b.weight": ((6, 2), torch.int8),
    "attn.wq_b.scale": ((6,), torch.float32),
    "attn.wkv.weight": ((3, 4), torch.bfloat16),
    "attn.q_norm.weight": ((2,), torch.bfloat16),
    "attn.kv_norm.weight": ((3,), torch.bfloat16),
    "attn.attn_sink": ((2,), torch.float32),
    "attn.wo_a.weight": ((16, 4), torch.bfloat16),
    "attn.wo_b.weight": ((4, 16), torch.int8),
    "attn.wo_b.scale": ((4,), torch.float32),
    "hc_ffn_fn": ((1, 4), torch.float32),
    "hc_ffn_scale": ((3,), torch.float32),
    "hc_ffn_base": ((1,), torch.float32),
    "ffn_norm.weight": ((4,), torch.bfloat16),
    "ffn.gate.weight": ((4, 4), torch.bfloat16),
    "ffn.gate.bias": ((4,), torch.float32),
    # Vocabulary-sized and replicated per rank: the one tensor here that is genuinely
    # large, so the fixture keeps num_hash_layers at 1 by default.
    "ffn.gate.tid2eid": ((129280, 6), torch.int32),
    "ffn.shared_experts.w1.weight": ((2, 4), torch.int8),
    "ffn.shared_experts.w1.scale": ((2,), torch.float32),
    "ffn.shared_experts.w2.weight": ((4, 2), torch.int8),
    "ffn.shared_experts.w2.scale": ((4,), torch.float32),
    "ffn.shared_experts.w3.weight": ((2, 4), torch.int8),
    "ffn.shared_experts.w3.scale": ((2,), torch.float32),
    "attn.compressor.wkv.weight": ((2, 4), torch.bfloat16),
    "attn.compressor.wgate.weight": ((2, 4), torch.bfloat16),
    "attn.compressor.ape": ((4, 2), torch.float32),
    "attn.compressor.norm.weight": ((3,), torch.bfloat16),
    "attn.indexer.wq_b.weight": ((6, 2), torch.int8),
    "attn.indexer.wq_b.scale": ((6,), torch.float32),
    "attn.indexer.weights_proj.weight": ((2, 4), torch.bfloat16),
    "attn.indexer.compressor.wkv.weight": ((2, 4), torch.bfloat16),
    "attn.indexer.compressor.wgate.weight": ((2, 4), torch.bfloat16),
    "attn.indexer.compressor.ape": ((4, 2), torch.float32),
    "attn.indexer.compressor.norm.weight": ((2,), torch.bfloat16),
    f"{_EXPERT_KEY}w1.weight": ((2, 4), torch.int8),
    f"{_EXPERT_KEY}w1.scale": ((2,), torch.float32),
    f"{_EXPERT_KEY}w2.weight": ((4, 2), torch.int8),
    f"{_EXPERT_KEY}w2.scale": ((4,), torch.float32),
    f"{_EXPERT_KEY}w3.weight": ((2, 4), torch.int8),
    f"{_EXPERT_KEY}w3.scale": ((2,), torch.float32),
}
_EXPERT_ID = re.compile(r"ffn\.experts\.\d+\.")

# The compressor/indexer tensors are the exception to "tiny is fine": the packer validates
# the active branch against fixed model dimensions, and fills the inactive branch with a
# zero tensor of those same dimensions. Since every slab is allocated from layer 0's
# template, a synthetic checkpoint whose CSA/HCA tensors are toy-sized cannot match the
# template's placeholder and the stack fails on a shape mismatch. So these follow the
# production constants — imported, not copied, so a dimension change here is a rename away
# rather than a silent divergence. The two entries the packer transposes are stored in
# their pre-transpose orientation.
_KIND_SHAPES: dict[int, dict[str, tuple[tuple[int, ...], torch.dtype]]] = {
    _DEEPSEEK_V4_HCA_COMPRESS_RATIO: {
        "attn.compressor.wkv.weight": ((_DEEPSEEK_V4_HCA_MAIN_OUT_DIM, _DEEPSEEK_V4_HIDDEN_SIZE), torch.bfloat16),
        "attn.compressor.wgate.weight": (
            (_DEEPSEEK_V4_HCA_MAIN_OUT_DIM, _DEEPSEEK_V4_HIDDEN_SIZE),
            torch.bfloat16,
        ),
        "attn.compressor.ape": ((_DEEPSEEK_V4_HCA_COMPRESS_RATIO, _DEEPSEEK_V4_HCA_MAIN_OUT_DIM), torch.float32),
        "attn.compressor.norm.weight": ((_DEEPSEEK_V4_HEAD_DIM,), torch.bfloat16),
    },
    _DEEPSEEK_V4_CSA_COMPRESS_RATIO: {
        "attn.compressor.wkv.weight": ((_DEEPSEEK_V4_CSA_MAIN_OUT_DIM, _DEEPSEEK_V4_HIDDEN_SIZE), torch.bfloat16),
        "attn.compressor.wgate.weight": (
            (_DEEPSEEK_V4_CSA_MAIN_OUT_DIM, _DEEPSEEK_V4_HIDDEN_SIZE),
            torch.bfloat16,
        ),
        "attn.compressor.ape": ((_DEEPSEEK_V4_CSA_COMPRESS_RATIO, _DEEPSEEK_V4_CSA_MAIN_OUT_DIM), torch.float32),
        "attn.compressor.norm.weight": ((_DEEPSEEK_V4_HEAD_DIM,), torch.bfloat16),
        # transposed by the packer, hence (out, in) here
        "attn.indexer.wq_b.weight": ((_DEEPSEEK_V4_ATTENTION_OUT // 4, _DEEPSEEK_V4_Q_LORA), torch.int8),
        "attn.indexer.wq_b.scale": ((_DEEPSEEK_V4_ATTENTION_OUT // 4,), torch.float32),
        # transposed by the packer too
        "attn.indexer.weights_proj.weight": ((64, _DEEPSEEK_V4_HIDDEN_SIZE), torch.bfloat16),
        "attn.indexer.compressor.wkv.weight": (
            (_DEEPSEEK_V4_CSA_INNER_OUT_DIM, _DEEPSEEK_V4_HIDDEN_SIZE),
            torch.bfloat16,
        ),
        "attn.indexer.compressor.wgate.weight": (
            (_DEEPSEEK_V4_CSA_INNER_OUT_DIM, _DEEPSEEK_V4_HIDDEN_SIZE),
            torch.bfloat16,
        ),
        "attn.indexer.compressor.ape": (
            (_DEEPSEEK_V4_CSA_COMPRESS_RATIO, _DEEPSEEK_V4_CSA_INNER_OUT_DIM),
            torch.float32,
        ),
        "attn.indexer.compressor.norm.weight": ((_DEEPSEEK_V4_HADAMARD_IDX_DIM,), torch.bfloat16),
    },
}


def _shape_for(suffix: str, compress_ratio: int) -> tuple[tuple[int, ...], torch.dtype]:
    """Look up a suffix, collapsing the expert index so one entry covers every expert."""
    kind = _KIND_SHAPES.get(int(compress_ratio), {})
    if suffix in kind:
        return kind[suffix]
    key = _EXPERT_ID.sub(_EXPERT_KEY, suffix)
    try:
        return _SHAPES[key]
    except KeyError as exc:  # pragma: no cover - fires only when the contract grows
        raise KeyError(
            f"synthetic DeepSeekV4 checkpoint has no shape for {suffix!r}; the checkpoint "
            "contract gained a tensor and this fixture must gain it too"
        ) from exc


def _filled(shape: tuple[int, ...], dtype: torch.dtype, seed: int) -> torch.Tensor:
    """Deterministic values that differ per layer and per tensor.

    int8 wraps at 127, so the seed is folded into a small offset instead of truncated: two
    layers colliding by accident would make the order-sensitivity test pass while proving
    nothing.
    """
    count = 1
    for dim in shape:
        count *= dim
    if dtype == torch.int8:
        values = (torch.arange(count, dtype=torch.int16) + (seed % 61)) % 127 - 63
        return values.to(torch.int8).reshape(shape)
    if dtype == torch.int32:
        return ((torch.arange(count, dtype=torch.int32) + seed) % 4).reshape(shape)
    return (torch.arange(count, dtype=torch.float32) + float(seed)).to(dtype).reshape(shape)


@dataclass(frozen=True)
class DeepSeekCheckpoint:
    """A synthetic checkpoint on disk plus the knobs its loader needs."""

    model_dir: Path
    weight_map: dict[str, str]
    compress_ratios: tuple[int, ...]
    n_routed_experts: int
    num_hash_layers: int
    ranks: int

    def store(self) -> DeepSeekV4WeightStore:
        """Open a lazy store over this checkpoint."""
        return DeepSeekV4WeightStore(model_dir=self.model_dir, weight_map=self.weight_map)

    def load_stacked(self):
        """Run the real load -> pack -> stack path, bypassing any prepacked sidecar."""
        return self.store().load_stacked_layer_weights(
            ranks=self.ranks,
            n_routed_experts=self.n_routed_experts,
            compress_ratios=self.compress_ratios,
            num_hash_layers=self.num_hash_layers,
            use_prepacked=False,
        )


@pytest.fixture
def deepseek_checkpoint(tmp_path):
    """Write a synthetic DeepSeekV4 checkpoint and return a handle to it.

    One shard per layer, which also makes the store's per-shard grouping observable. The
    default mixes all three attention kinds — ``(0, 4, 128, 4)`` — because the stacked
    layout sends FWD, CSA and HCA weights to three different slabs, and a fixture with a
    single kind could not tell a group-placement regression from a no-op.
    """

    builds = itertools.count()

    def _build(
        *,
        compress_ratios: tuple[int, ...] = (0, 4, 128, 4),
        n_routed_experts: int = 4,
        num_hash_layers: int = 1,
        ranks: int = 2,
        layer_seeds: dict[int, int] | None = None,
    ) -> DeepSeekCheckpoint:
        from safetensors.torch import save_file

        # Each build gets its own directory: a test that compares two checkpoints would
        # otherwise have the second overwrite the first's shards and compare a checkpoint
        # with itself — passing while proving nothing.
        model_dir = tmp_path / f"checkpoint-{next(builds)}"
        model_dir.mkdir()
        seeds = dict(layer_seeds or {})
        weight_map: dict[str, str] = {}
        for layer_id, ratio in enumerate(compress_ratios):
            seed = seeds.get(layer_id, layer_id)
            prefix = f"layers.{layer_id}."
            names = deepseek_v4_layer_weight_names(
                layer_id,
                n_routed_experts=n_routed_experts,
                compress_ratio=int(ratio),
                include_tid2eid=layer_id < num_hash_layers,
                include_gate_bias=layer_id >= num_hash_layers,
            )
            tensors: dict[str, torch.Tensor] = {}
            for index, name in enumerate(names):
                assert name.startswith(prefix), f"unexpected layer tensor name {name!r}"
                shape, dtype = _shape_for(name[len(prefix) :], int(ratio))
                tensors[name] = _filled(shape, dtype, seed * 1000 + index)
            filename = f"layer-{layer_id:05d}.safetensors"
            save_file(tensors, str(model_dir / filename))
            weight_map.update({name: filename for name in tensors})

        return DeepSeekCheckpoint(
            model_dir=model_dir,
            weight_map=weight_map,
            compress_ratios=tuple(compress_ratios),
            n_routed_experts=n_routed_experts,
            num_hash_layers=num_hash_layers,
            ranks=ranks,
        )

    return _build
