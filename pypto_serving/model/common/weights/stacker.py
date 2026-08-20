# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Whole-model slab allocation and per-layer placement.

pypto-lib fuses every layer into one kernel, so a layer's weight has no standalone existence
on device: the model is uploaded as a few whole-model slabs, each holding all of its layers
back to back. This module owns that geometry — allocate the slabs from one layer's shapes,
hand each layer a *view* of its own slice, and let the packer write straight into it.

Writing into views is the point. Packing each layer separately and concatenating afterwards
would hold the whole model twice at the peak, which for DeepSeek V4 is ~346 GB it does not
have to pay, so nothing here ever calls ``torch.cat``.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .pipeline import StagingPolicy


@dataclass(frozen=True)
class StackGroup:
    """A set of weights stacked over a subset of the layers.

    ``layer_ids`` is ordered, and a layer's *position* in it — not its id — is the slice the
    layer occupies in the group's slabs. That is what lets a group cover only some layers: a
    model whose layers 1 and 3 are the only ones with a given attention kind stacks them at
    positions 0 and 1, contiguously, with nothing reserved for the layers in between.

    ``members`` may be ``None`` for exactly one group in a set, meaning "every weight no other
    group claims", so a family lists only its special groups and lets the rest fall through.
    """

    id: str
    members: tuple[str, ...] | None
    layer_ids: tuple[int, ...]

    def position_of(self, layer_id: int) -> int | None:
        """Return this layer's slot in the group, or ``None`` if it is not a member."""
        try:
            return self.layer_ids.index(int(layer_id))
        except ValueError:
            return None


def resolve_members(groups: Sequence[StackGroup], template: Mapping[str, torch.Tensor]) -> tuple[StackGroup, ...]:
    """Fill in the one group declared with ``members=None`` from what the others do not claim.

    Order follows the template, which is the order the packer produced — and therefore the
    order the prepacked sidecar's name-to-offset map is built in, so it is contract rather
    than presentation.
    """
    claimed = {name for group in groups if group.members is not None for name in group.members}
    catch_all = [group for group in groups if group.members is None]
    if len(catch_all) > 1:
        raise ValueError(
            f"at most one stack group may be the catch-all, got {[group.id for group in catch_all]}"
        )
    missing = claimed - set(template)
    if missing:
        raise ValueError(f"stack groups name weights the packed layer does not have: {sorted(missing)}")
    return tuple(
        StackGroup(
            id=group.id,
            members=tuple(name for name in template if name not in claimed)
            if group.members is None
            else group.members,
            layer_ids=group.layer_ids,
        )
        for group in groups
    )


def allocate_slabs(
    groups: Sequence[StackGroup],
    template: Mapping[str, torch.Tensor],
    *,
    rank_error: str = "packed weight {name} must have rank >= 2, got {ndim}",
) -> dict[str, torch.Tensor]:
    """Allocate one whole-model slab per weight, sized from *template* and the group's layers.

    A group with no layers allocates nothing: a model that uses none of an attention kind must
    not reserve slabs for it. Stacking is on dim 1 because dim 0 is the rank axis, which
    ``alloc_stacked_tensor`` shards on.
    """
    slabs: dict[str, torch.Tensor] = {}
    for group in groups:
        count = len(group.layer_ids)
        if count == 0:
            continue
        for name in group.members or ():
            source = template[name]
            if source.ndim < 2:
                raise ValueError(rank_error.format(name=name, ndim=source.ndim))
            shape = (int(source.shape[0]), count * int(source.shape[1]), *source.shape[2:])
            slabs[name] = torch.empty(shape, dtype=source.dtype, device="cpu")
    return slabs


def destinations_for(
    slabs: Mapping[str, torch.Tensor],
    groups: Sequence[StackGroup],
    template: Mapping[str, torch.Tensor],
    *,
    layer_id: int,
) -> dict[str, torch.Tensor]:
    """Return this layer's slice of every slab it belongs to, as views into the slabs.

    A weight absent from the result is one this layer does not contribute to, which the packer
    reads as "skip it" rather than "pack it and discard it".
    """
    destinations: dict[str, torch.Tensor] = {}
    for group in groups:
        position = group.position_of(layer_id)
        if position is None:
            continue
        for name in group.members or ():
            width = int(template[name].shape[1])
            destinations[name] = slabs[name][:, position * width : (position + 1) * width]
    return destinations


def copy_packed_layer(
    packed: Mapping[str, torch.Tensor],
    destinations: Mapping[str, torch.Tensor],
    *,
    mismatch_error: str = (
        "packed weight {name} shape/dtype mismatch: source={source}, destination={destination}"
    ),
) -> None:
    """Copy an already-packed layer into its slab slices.

    Needed only for a layer packed *before* the slabs existed — the template layer, whose
    shapes are what sized them. Every later layer is packed into its destination directly and
    never passes through here.
    """
    for name, destination in destinations.items():
        source = packed[name]
        if tuple(source.shape) != tuple(destination.shape) or source.dtype != destination.dtype:
            raise ValueError(
                mismatch_error.format(
                    name=name,
                    source=f"{tuple(source.shape)}/{source.dtype}",
                    destination=f"{tuple(destination.shape)}/{destination.dtype}",
                )
            )
        destination.copy_(source)


def stack_layers(
    groups: Sequence[StackGroup],
    template: Mapping[str, torch.Tensor],
    *,
    layer_ids: Sequence[int],
    pack_into: Callable[[int, Mapping[str, torch.Tensor]], None],
    template_layer_id: int | None = None,
    on_layer_done: Callable[[int], None] | None = None,
    policy: "StagingPolicy | None" = None,
    rank_error: str = "packed weight {name} must have rank >= 2, got {ndim}",
    mismatch_error: str = (
        "packed weight {name} shape/dtype mismatch: source={source}, destination={destination}"
    ),
) -> dict[str, torch.Tensor]:
    """Allocate the slabs and fill them layer by layer, in ``layer_ids`` order.

    ``template`` is one already-packed layer: its shapes size the slabs, and if
    ``template_layer_id`` names it, its tensors are copied into place instead of being packed a
    second time. ``pack_into(layer_id, destinations)`` must write every destination it is
    given; it is the caller's packer, so this module never learns what a weight means.

    ``on_layer_done`` fires once per layer whichever path it took, so progress reporting does
    not silently skip the template layer.

    ``policy`` decides whether layers are staged one at a time or overlapped. Overlapping is
    safe here by construction — each layer writes a disjoint slice of each slab, so no two
    workers touch the same bytes — but it is not always *faster*, which is why it is the
    caller's call: a family whose per-layer packing allocates gigabytes of intermediates pays
    more in peak memory and bandwidth contention than it gains in hidden latency.
    """
    from .pipeline import StagingPolicy, stage_layers  # noqa: PLC0415 -- avoids a cycle

    resolved = resolve_members(groups, template)
    slabs = allocate_slabs(resolved, template, rank_error=rank_error)

    def _stage(layer_id: int) -> None:
        destinations = destinations_for(slabs, resolved, template, layer_id=layer_id)
        if template_layer_id is not None and int(layer_id) == int(template_layer_id):
            copy_packed_layer(template, destinations, mismatch_error=mismatch_error)
        else:
            pack_into(int(layer_id), destinations)

    stage_layers(
        list(layer_ids),
        stage=_stage,
        policy=policy if policy is not None else StagingPolicy(workers=1),
        on_layer_done=on_layer_done,
    )
    return slabs
