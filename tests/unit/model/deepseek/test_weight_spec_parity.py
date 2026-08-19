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

import pytest
import torch

from pypto_serving.model.common.weights.packer import pack_layer
from pypto_serving.model.common.weights.spec import LayerContext
from pypto_serving.model.deepseek.weight_loader import pack_deepseek_v4_layer_weights
from pypto_serving.model.deepseek.weight_spec import (
    DEEPSEEK_V4_CORE_LAYER_RULES,
    DEEPSEEK_V4_LAYER_RULES,
    deepseek_v4_expert_parallel,
    deepseek_v4_factories,
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


def _legacy_full(raw, *, ratio, tid2eid, gate_bias, destinations=None):
    return pack_deepseek_v4_layer_weights(
        0,
        raw,
        ranks=_RANKS,
        n_routed_experts=_EXPERTS,
        compress_ratio=ratio,
        include_tid2eid=tid2eid,
        include_gate_bias=gate_bias,
        destinations=destinations,
    )


def _spec_full(raw, *, ratio, tid2eid, gate_bias, destinations=None):
    return pack_layer(
        DEEPSEEK_V4_LAYER_RULES,
        raw,
        LayerContext(
            layer_id=0,
            prefix="layers.0",
            ranks=_RANKS,
            compress_ratio=ratio,
            n_routed_experts=_EXPERTS,
            include_tid2eid=tid2eid,
            include_gate_bias=gate_bias,
        ),
        policy=deepseek_v4_replicate(_RANKS),
        expert_policy=deepseek_v4_expert_parallel(_RANKS, _EXPERTS),
        factories=deepseek_v4_factories(),
        destinations=destinations,
    )


@pytest.mark.parametrize("ratio", [0, 4, 128])
def test_full_rule_table_reproduces_the_packer_byte_for_byte(
    ratio, deepseek_checkpoint, fingerprint_tensors
):
    """All 49 names, every attention kind, direct path.

    Parametrised over the three kinds because the zero-filled branches are the ones a
    declarative rule can get wrong without changing a single shape.
    """
    checkpoint = deepseek_checkpoint(compress_ratios=(ratio,), n_routed_experts=_EXPERTS, num_hash_layers=1)
    raw = _raw_for_layer(checkpoint)
    flags = {"ratio": ratio, "tid2eid": True, "gate_bias": False}

    legacy = _legacy_full(raw, **flags).tensors
    spec = _spec_full(raw, **flags)

    assert list(legacy) == list(spec), "name order must match the hand-written table"
    assert fingerprint_tensors(legacy) == fingerprint_tensors(spec)


@pytest.mark.parametrize("ratio", [0, 4, 128])
def test_full_rule_table_matches_on_the_destination_path(ratio, deepseek_checkpoint, fingerprint_tensors):
    """Same 49 names written into preallocated slabs instead of fresh buffers."""
    checkpoint = deepseek_checkpoint(compress_ratios=(ratio,), n_routed_experts=_EXPERTS, num_hash_layers=1)
    raw = _raw_for_layer(checkpoint)
    flags = {"ratio": ratio, "tid2eid": True, "gate_bias": False}
    reference = _legacy_full(raw, **flags).tensors

    legacy_slabs = {name: torch.zeros_like(tensor) for name, tensor in reference.items()}
    spec_slabs = {name: torch.zeros_like(tensor) for name, tensor in reference.items()}
    _legacy_full(raw, **flags, destinations=legacy_slabs)
    _spec_full(raw, **flags, destinations=spec_slabs)

    assert fingerprint_tensors(legacy_slabs) == fingerprint_tensors(spec_slabs)


def test_router_placeholder_matches_when_the_layer_carries_the_other_mode(
    deepseek_checkpoint, fingerprint_tensors
):
    """A layer with gate_bias and no tid2eid: the placeholder is the interesting half."""
    checkpoint = deepseek_checkpoint(
        compress_ratios=(0, 0), n_routed_experts=_EXPERTS, num_hash_layers=1
    )
    raw = checkpoint.store().load_layer_weights(
        1, n_routed_experts=_EXPERTS, compress_ratio=0, include_tid2eid=False, include_gate_bias=True
    )
    raw = {name.replace("layers.1", "layers.0", 1): tensor for name, tensor in raw.items()}
    flags = {"ratio": 0, "tid2eid": False, "gate_bias": True}

    legacy = _legacy_full(raw, **flags).tensors
    spec = _spec_full(raw, **flags)

    assert fingerprint_tensors(legacy) == fingerprint_tensors(spec)


def test_a_required_router_source_still_raises_when_absent(deepseek_checkpoint):
    """The requirement flag must not be softened into a silent zero fill."""
    checkpoint = deepseek_checkpoint(compress_ratios=(0,), n_routed_experts=_EXPERTS, num_hash_layers=1)
    raw = dict(_raw_for_layer(checkpoint))
    del raw["layers.0.ffn.gate.tid2eid"]

    with pytest.raises(KeyError, match="tid2eid"):
        _spec_full(raw, ratio=0, tid2eid=True, gate_bias=False)
