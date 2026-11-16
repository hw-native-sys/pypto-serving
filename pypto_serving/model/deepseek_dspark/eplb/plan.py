# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Checked, deterministic source plans for six-tensor expert migration."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .state import Placement, immutable_ints

EXPERT_TENSORS = (
    "routed_w1", "routed_w1_scale", "routed_w3", "routed_w3_scale", "routed_w2", "routed_w2_scale",
)


@dataclass(frozen=True)
class LayerMovePlan:
    layer_id: int
    sources: np.ndarray
    changed: np.ndarray
    payload_bytes: int

    def __post_init__(self) -> None:
        if isinstance(self.layer_id, bool) or not isinstance(self.layer_id, int) or self.layer_id < 0:
            raise ValueError("Invalid migration layer ID")
        sources = immutable_ints(self.sources, name="sources")
        changed = np.asarray(self.changed)
        if changed.dtype != np.bool_ or sources.shape != (*changed.shape, 2) or changed.ndim != 2:
            raise ValueError("Invalid migration source/mask shape")
        if (isinstance(self.payload_bytes, bool) or not isinstance(self.payload_bytes, int)
                or self.payload_bytes <= 0):
            raise ValueError("Migration payload must be positive")
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "changed", np.frombuffer(changed.tobytes(), dtype=np.bool_).reshape(changed.shape))


@dataclass(frozen=True)
class RebalancePlan:
    base_generation: int
    target: Placement
    layers: tuple[LayerMovePlan, ...]
    predicted_gain: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "layers", tuple(self.layers))
        if (isinstance(self.base_generation, bool) or not isinstance(self.base_generation, int)
                or self.base_generation < 0 or self.target.generation != self.base_generation + 1):
            raise ValueError("Migration must advance exactly one generation")
        if not self.layers or len({layer.layer_id for layer in self.layers}) != len(self.layers):
            raise ValueError("Migration must contain distinct changed layers")
        if (isinstance(self.predicted_gain, bool) or not isinstance(self.predicted_gain, (int, float))
                or not math.isfinite(self.predicted_gain) or not 0 < self.predicted_gain <= 1):
            raise ValueError("Invalid predicted migration gain")

    @property
    def payload_bytes(self) -> int:
        return sum(layer.payload_bytes for layer in self.layers)


def layer_move_plan(base: Placement, target: Placement, layer: int, *, expert_bytes: int) -> LayerMovePlan:
    """Prefer local sources, then spread sends over existing remote replicas."""
    if expert_bytes <= 0:
        raise ValueError("Expert payload size must be positive")
    ranks, slots = base.ranks, base.local_slots
    old = base.physical_to_logical[layer].reshape(ranks, slots)
    new = target.physical_to_logical[layer].reshape(ranks, slots)
    changed = old != new
    sources = np.empty((ranks, slots, 2), dtype=np.int64)
    sent = np.zeros(ranks, dtype=np.int64)
    for rank in range(ranks):
        for slot in range(slots):
            if not changed[rank, slot]:
                sources[rank, slot] = rank, slot
                continue
            candidates = np.argwhere(old == new[rank, slot])
            local = candidates[candidates[:, 0] == rank]
            if len(local):
                source_rank, source_slot = local[0]
            else:
                source_rank, source_slot = min(candidates, key=lambda pair: (sent[pair[0]], *pair))
                sent[source_rank] += 1
            sources[rank, slot] = source_rank, source_slot
    return LayerMovePlan(layer, sources, changed, int(changed.sum()) * expert_bytes)


def validate_plan(base: Placement, plan: RebalancePlan, *, expert_bytes: int,
                  max_layers: int, max_migration_bytes: int) -> None:
    """Validate all sources and complete layers before the first device-side write."""
    target = plan.target
    if plan.base_generation != base.generation or target.generation != base.generation + 1:
        raise ValueError("Stale EPLB migration generation")
    if (target.physical_to_logical.shape != base.physical_to_logical.shape
            or target.logical_to_physical.shape != base.logical_to_physical.shape):
        raise ValueError("EPLB cannot change physical capacity at runtime")
    if expert_bytes <= 0 or len(plan.layers) > max_layers or plan.payload_bytes > max_migration_bytes:
        raise ValueError("EPLB migration exceeds its layer or byte budget")
    changed_layers = set(np.flatnonzero(np.any(base.physical_to_logical != target.physical_to_logical, axis=1)))
    if changed_layers != {layer.layer_id for layer in plan.layers}:
        raise ValueError("Migration must cover every changed layer exactly once")
    ranks, slots = base.ranks, base.local_slots
    for layer in plan.layers:
        if not 0 <= layer.layer_id < base.layers or layer.changed.shape != (ranks, slots):
            raise ValueError("Invalid migration layer geometry")
        old = base.physical_to_logical[layer.layer_id].reshape(ranks, slots)
        new = target.physical_to_logical[layer.layer_id].reshape(ranks, slots)
        if not np.array_equal(layer.changed, old != new):
            raise ValueError("Migration changed mask does not match the target")
        if layer.payload_bytes != int(layer.changed.sum()) * expert_bytes:
            raise ValueError("Migration byte accounting mismatch")
        source_ranks = layer.sources[:, :, 0]
        source_slots = layer.sources[:, :, 1]
        if (np.any(source_ranks < 0) or np.any(source_ranks >= ranks)
                or np.any(source_slots < 0) or np.any(source_slots >= slots)):
            raise ValueError("Migration source is outside the resident weight bank")
        if not np.array_equal(old[source_ranks, source_slots], new):
            raise ValueError("Migration source does not contain the target logical expert")
