# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Differential parity for the slab geometry: the generic stacker must place layers exactly
# where the hand-written helpers did.
#
# Placement is the half of the contract a shape check cannot see. Every slab keeps its shape
# and dtype no matter which layer landed in which slice, so a stacker that got the geometry
# wrong would produce correctly-shaped, correctly-typed, wrong weights — and the prepacked
# sidecars already on disk record the layout it must reproduce.
from __future__ import annotations

import pytest
import torch

from pypto_serving.model.common.weights.stacker import (
    allocate_slabs,
    destinations_for,
    resolve_members,
    stack_layers,
)
from pypto_serving.model.deepseek import weight_loader
from pypto_serving.model.deepseek.weight_spec import deepseek_v4_stack_groups

_RANKS = 2


def _template(names, width=3):
    """One packed layer's worth of tensors: `[ranks, width, ...]`, distinct per name."""
    return {
        name: torch.full((_RANKS, width), float(index), dtype=torch.float32)
        for index, name in enumerate(names)
    }


def _all_names():
    csa = tuple(weight_loader.DEEPSEEK_V4_CSA_STACKED_WEIGHT_NAMES)
    hca = tuple(weight_loader.DEEPSEEK_V4_HCA_STACKED_WEIGHT_NAMES)
    return ("fwd_a", "fwd_b", *csa, *hca), csa, hca


def _legacy_stack(template, compress_ratios, packed_per_layer):
    """Drive the hand-written helpers, exactly as `load_stacked_layer_weights` used to."""
    packed = weight_loader.DeepSeekV4PackedLayerWeights(layer_id=0, tensors=template)
    slabs, fwd_names = weight_loader._allocate_stacked_layer_weights(
        packed, compress_ratios=compress_ratios
    )
    csa_order = 0
    hca_order = 0
    for layer_id, ratio in enumerate(compress_ratios):
        destinations = weight_loader._stacked_layer_destinations(
            slabs,
            packed,
            fwd_names=fwd_names,
            layer_id=layer_id,
            compress_ratio=int(ratio),
            csa_order=csa_order,
            hca_order=hca_order,
        )
        weight_loader._copy_packed_layer(
            weight_loader.DeepSeekV4PackedLayerWeights(
                layer_id=layer_id, tensors=packed_per_layer[layer_id]
            ),
            destinations,
        )
        csa_order += int(int(ratio) == 4)
        hca_order += int(int(ratio) == 128)
    return slabs


def _generic_stack(template, compress_ratios, packed_per_layer):
    def pack_into(layer_id, destinations):
        for name, destination in destinations.items():
            destination.copy_(packed_per_layer[layer_id][name])

    return stack_layers(
        deepseek_v4_stack_groups(compress_ratios),
        template,
        layer_ids=range(len(compress_ratios)),
        pack_into=pack_into,
    )


@pytest.mark.parametrize(
    "compress_ratios",
    [
        (0, 4, 128, 4),
        (0, 0, 0),
        (4, 4, 4),
        (128,),
        (0, 128, 4, 128, 0, 4),
    ],
    ids=["mixed", "no-groups", "csa-only", "hca-only", "interleaved"],
)
def test_generic_stacker_places_layers_where_the_helpers_did(
    compress_ratios, fingerprint_tensors
):
    """Byte-for-byte across five layouts, including the ones where a group is empty."""
    names, _, _ = _all_names()
    template = _template(names)
    # Each layer gets its own values, so a misplaced layer changes the slab's contents.
    per_layer = [
        {name: template[name] + 100.0 * (layer_id + 1) for name in names}
        for layer_id in range(len(compress_ratios))
    ]

    legacy = _legacy_stack(template, compress_ratios, per_layer)
    generic = _generic_stack(template, compress_ratios, per_layer)

    assert list(legacy) == list(generic), "slab order is what the sidecar's offsets follow"
    assert fingerprint_tensors(legacy) == fingerprint_tensors(generic)


def test_an_empty_group_allocates_no_slabs():
    """A model with no CSA layers must not reserve CSA slabs."""
    names, csa, _ = _all_names()
    template = _template(names)

    slabs = allocate_slabs(resolve_members(deepseek_v4_stack_groups((0, 128)), template), template)

    assert not set(csa) & set(slabs)


def test_destinations_are_views_into_the_slabs():
    """Writing a destination must land in the slab: that is what avoids a second full copy."""
    names, _, _ = _all_names()
    template = _template(names)
    groups = resolve_members(deepseek_v4_stack_groups((0, 0)), template)
    slabs = allocate_slabs(groups, template)

    destinations = destinations_for(slabs, groups, template, layer_id=1)
    destinations["fwd_a"].fill_(7.0)

    width = template["fwd_a"].shape[1]
    assert torch.all(slabs["fwd_a"][:, width:] == 7.0)
    assert torch.all(slabs["fwd_a"][:, :width] != 7.0)


def test_a_layer_outside_a_group_gets_no_destination_for_it():
    """A non-CSA layer must not be handed CSA destinations, or it would pack and discard them."""
    names, csa, _ = _all_names()
    template = _template(names)
    groups = resolve_members(deepseek_v4_stack_groups((0, 4)), template)
    slabs = allocate_slabs(groups, template)

    plain = destinations_for(slabs, groups, template, layer_id=0)
    csa_layer = destinations_for(slabs, groups, template, layer_id=1)

    assert not set(csa) & set(plain)
    assert set(csa) <= set(csa_layer)


def test_two_catch_all_groups_are_rejected():
    """The catch-all is what makes a new weight join FWD automatically; two would be ambiguous."""
    from pypto_serving.model.common.weights.stacker import StackGroup

    template = _template(("a", "b"))

    with pytest.raises(ValueError, match="catch-all"):
        resolve_members(
            (StackGroup("x", None, (0,)), StackGroup("y", None, (0,))),
            template,
        )


def test_a_group_naming_an_absent_weight_is_rejected():
    """Silently dropping it would leave a kernel weight unstacked."""
    from pypto_serving.model.common.weights.stacker import StackGroup

    template = _template(("a",))

    with pytest.raises(ValueError, match="ghost"):
        resolve_members((StackGroup("g", ("ghost",), (0,)),), template)
