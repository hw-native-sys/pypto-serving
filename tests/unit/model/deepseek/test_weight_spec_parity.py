# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Differential parity: the declarative core rules must reproduce the hand-written pack table.
#
# This is the gate #163 asks for at step 3 — old and new importable side by side, compared
# directly on the same input, rather than a golden file that only says "something changed".
from __future__ import annotations

import torch

from pypto_serving.model.common.weights.packer import pack_layer
from pypto_serving.model.common.weights.spec import LayerContext
from pypto_serving.model.deepseek.weight_loader import pack_deepseek_v4_layer_weights
from pypto_serving.model.deepseek.weight_spec import (
    DEEPSEEK_V4_CORE_LAYER_RULES,
    deepseek_v4_replicate,
)

_RANKS = 2
_EXPERTS = 4


def _raw_for_layer(checkpoint, layer_id: int = 0) -> dict[str, torch.Tensor]:
    """Read one layer's tensors through the real store, not a hand-built dict."""
    return checkpoint.store().load_layer_weights(
        layer_id,
        n_routed_experts=_EXPERTS,
        compress_ratio=int(checkpoint.compress_ratios[layer_id]),
        include_tid2eid=layer_id < checkpoint.num_hash_layers,
        include_gate_bias=layer_id >= checkpoint.num_hash_layers,
    )


def _legacy(raw, layer_id: int = 0, destinations=None):
    return pack_deepseek_v4_layer_weights(
        layer_id,
        raw,
        ranks=_RANKS,
        n_routed_experts=_EXPERTS,
        compress_ratio=0,
        include_tid2eid=True,
        include_gate_bias=False,
        destinations=destinations,
    )


def _from_spec(raw, layer_id: int = 0, destinations=None):
    return pack_layer(
        DEEPSEEK_V4_CORE_LAYER_RULES,
        raw,
        LayerContext(layer_id=layer_id, prefix=f"layers.{layer_id}", ranks=_RANKS),
        policy=deepseek_v4_replicate(_RANKS),
        destinations=destinations,
    )


def test_core_rules_cover_the_hand_written_table_in_order(deepseek_checkpoint):
    """Same names, same order — the order is what the sidecar's offset map is built from."""
    checkpoint = deepseek_checkpoint(compress_ratios=(0,), n_routed_experts=_EXPERTS)
    legacy = _legacy(_raw_for_layer(checkpoint))

    rule_names = [rule.name for rule in DEEPSEEK_V4_CORE_LAYER_RULES]
    legacy_core_order = [name for name in legacy.tensors if name in set(rule_names)]

    assert rule_names == legacy_core_order


def test_spec_reproduces_the_hand_written_table_byte_for_byte(deepseek_checkpoint, fingerprint_tensors):
    """The direct path: fresh buffers on both sides, identical bytes."""
    checkpoint = deepseek_checkpoint(compress_ratios=(0,), n_routed_experts=_EXPERTS)
    raw = _raw_for_layer(checkpoint)

    legacy = _legacy(raw)
    spec = _from_spec(raw)

    covered = {name: legacy.tensors[name] for name in spec}
    assert fingerprint_tensors(covered) == fingerprint_tensors(spec)


def test_spec_matches_the_hand_written_table_on_the_destination_path(
    deepseek_checkpoint, fingerprint_tensors
):
    """The destination path casts inside ``copy_()`` instead of ``.to(dtype)``.

    That difference in form is exactly what #163 flags as a byte-identity trap, so the two
    packers are compared writing into equally-shaped slabs, not only into fresh buffers.
    """
    checkpoint = deepseek_checkpoint(compress_ratios=(0,), n_routed_experts=_EXPERTS)
    raw = _raw_for_layer(checkpoint)

    # The legacy packer writes every name it owns, so it needs a destination for each --
    # router and experts included, not just the core subset this spec covers yet.
    reference = _legacy(raw).tensors
    legacy_slabs = {name: torch.zeros_like(tensor) for name, tensor in reference.items()}
    _legacy(raw, destinations=legacy_slabs)

    spec_slabs = {
        rule.name: torch.zeros_like(reference[rule.name]) for rule in DEEPSEEK_V4_CORE_LAYER_RULES
    }
    _from_spec(raw, destinations=spec_slabs)

    covered = {name: legacy_slabs[name] for name in spec_slabs}
    assert fingerprint_tensors(covered) == fingerprint_tensors(spec_slabs)


def test_destination_and_direct_paths_agree_bit_for_bit(deepseek_checkpoint, fingerprint_tensors):
    """The trap itself, asserted rather than assumed: explicit cast == cast inside copy_()."""
    checkpoint = deepseek_checkpoint(compress_ratios=(0,), n_routed_experts=_EXPERTS)
    raw = _raw_for_layer(checkpoint)

    direct = _from_spec(raw)
    slabs = {name: torch.zeros_like(tensor) for name, tensor in direct.items()}
    _from_spec(raw, destinations=slabs)

    assert fingerprint_tensors(direct) == fingerprint_tensors(slabs)


def test_a_subset_of_destinations_packs_only_that_subset(deepseek_checkpoint):
    """Skipping absent destinations is how a caller stages one group without paying for the rest."""
    checkpoint = deepseek_checkpoint(compress_ratios=(0,), n_routed_experts=_EXPERTS)
    raw = _raw_for_layer(checkpoint)
    template = _from_spec(raw)
    wanted = "attn_norm_w"

    packed = _from_spec(raw, destinations={wanted: torch.zeros_like(template[wanted])})

    assert list(packed) == [wanted]
