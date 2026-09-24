# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DSpark weight packing against the real store pipeline.

This file lives beside the DeepSeek V4 tests on purpose: it consumes the shared
synthetic W8A8 checkpoint fixture (both families read the same checkpoint
contract), wrapping it in a :class:`DSparkWeightStore` to exercise the real
load -> pack -> stack path end to end. The fixtures span at least two NZ column
blocks per weight, so unpacked round-trips verify the actual byte order rather
than coinciding with an unpacked slab.
"""
from __future__ import annotations

import torch

from pypto_serving.model.common.weights.nz import unpack_nz
from pypto_serving.model.deepseek.weight_loader import deepseek_v4_local_expert_ids
from pypto_serving.model.deepseek_dspark.weight_loader import DSparkWeightStore
from pypto_serving.model.deepseek_dspark.weight_spec import DSPARK_TP_SIZE

_RANKS = 4


def test_dspark_stacked_weights_nz_round_trip(deepseek_checkpoint):
    checkpoint = deepseek_checkpoint(
        compress_ratios=(0, 4, 128, 4),
        n_routed_experts=4,
        num_hash_layers=1,
        ranks=_RANKS,
    )
    store = DSparkWeightStore(model_dir=checkpoint.model_dir, weight_map=checkpoint.weight_map)
    weights = store.load_stacked_layer_weights(
        ranks=_RANKS,
        n_routed_experts=4,
        compress_ratios=(0, 4, 128, 4),
        num_hash_layers=1,
    )
    tensors = weights.tensors
    layers = range(4)
    local_groups = 8 // DSPARK_TP_SIZE

    # A missing NZ pass would leave these bytes equal to their own unpacking.
    # Both output projections are covered, in both packed dtypes.
    for name in ("wo_a", "wo_b", "wq_a", "shared_w2"):
        assert not torch.equal(tensors[name], unpack_nz(tensors[name])), name

    # New-layer-axis NZ banks: a layer is an index, and every rank holds the
    # same (replicated) checkpoint tensor once unpacked.
    for name, suffix, transpose in (
        ("wq_a", "attn.wq_a.weight", True),
        ("wq_b", "attn.wq_b.weight", True),
        ("shared_w2", "ffn.shared_experts.w2.weight", False),
    ):
        slab = tensors[name]
        assert slab.shape[1] == 4, name  # the new layer axis, not merged rows
        unpacked = unpack_nz(slab)
        for layer in layers:
            expected = store.load_tensor(f"layers.{layer}.{suffix}")
            if transpose:
                expected = expected.t().contiguous()
            assert torch.equal(unpacked[0, layer], expected), (name, layer)

    # Group-merged NZ banks (wo_a / wo_b): a layer is a group window on axis 1 and
    # each TP rank owns its own two groups.
    for layer in layers:
        wo_a = store.load_tensor(f"layers.{layer}.attn.wo_a.weight").reshape(8, 16, 32)
        wo_b = (
            store.load_tensor(f"layers.{layer}.attn.wo_b.weight")
            .reshape(16, 8, 64)
            .permute(1, 0, 2)
            .contiguous()
        )
        for rank in range(_RANKS):
            tp = rank % DSPARK_TP_SIZE
            groups = slice(tp * local_groups, (tp + 1) * local_groups)
            assert torch.equal(
                unpack_nz(tensors["wo_a"])[rank, layer * local_groups : (layer + 1) * local_groups],
                wo_a[groups],
            ), ("wo_a", rank, layer)
            assert torch.equal(
                unpack_nz(tensors["wo_b"])[rank, layer * local_groups : (layer + 1) * local_groups],
                wo_b[groups],
            ), ("wo_b", rank, layer)

    # Expert-merged NZ banks: a layer is a window on the local-expert axis.
    unpacked = unpack_nz(tensors["routed_w1"])
    for rank in range(_RANKS):
        local_ids = deepseek_v4_local_expert_ids(rank=rank, ranks=_RANKS, n_routed_experts=4)
        for layer in layers:
            for local_index, expert in enumerate(local_ids):
                expected = store.load_tensor(f"layers.{layer}.ffn.experts.{expert}.w1.weight")
                assert torch.equal(unpacked[rank, layer * len(local_ids) + local_index], expected)

    # The DSpark-ND names must stay row-major: comparing without unpacking is the
    # assertion -- an NZ-packed wkv would only match after unpack_nz. wkv merges
    # its layers on the row axis (ND, so a row window is addressable).
    for layer in (0, 2):
        expected = store.load_tensor(f"layers.{layer}.attn.wkv.weight").t().contiguous()
        rows = int(expected.shape[0])
        assert torch.equal(tensors["wkv"][0, layer * rows : (layer + 1) * rows], expected)

    # Prefill reads the o-projection pair ND: its bytes equal the *unpacked*
    # decode bytes, and applying unpack_nz to the prefill side would scramble
    # them -- the inverse goes on the decode side only.
    assert torch.equal(weights.prefill_tensors["wo_a"], unpack_nz(tensors["wo_a"]))
    assert torch.equal(weights.prefill_tensors["wo_b"], unpack_nz(tensors["wo_b"]))

    # Prefill's CSA indexer projection is NZ on a new layer axis while decode's
    # stays the flat ND row window; both unpack to the same per-layer matrices.
    prefill_csa = unpack_nz(weights.prefill_tensors["csa_weights_proj"])
    decode_csa = tensors["csa_weights_proj"]
    assert prefill_csa.shape[1] == 2  # the two ratio-4 layers
    rows = int(decode_csa.shape[1]) // 2
    for layer in range(2):
        assert torch.equal(prefill_csa[0, layer], decode_csa[0, layer * rows : (layer + 1) * rows])

    # Upload-once: the prefill slabs are popped and re-keyed to the kernel arg
    # names, so no `prefill_` rule name survives in either dict -- a survivor
    # would mean the runner uploads that slab a second time under its rule name.
    for name in ("prefill_wo_a", "prefill_wo_b", "prefill_csa_weights_proj"):
        assert name not in tensors, name
        assert name not in weights.prefill_tensors, name
    for name in ("wo_a", "wo_b", "csa_weights_proj"):
        assert name in weights.prefill_tensors, name
