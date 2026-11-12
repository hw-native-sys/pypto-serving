# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Immutable logical/physical expert placement and dispatch-owned load samples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def immutable_ints(value, *, name: str) -> np.ndarray:
    """Own integer data in a bytes-backed array that cannot be made writable."""
    array = np.asarray(value)
    if array.dtype.kind not in "iu" or (array.dtype.kind == "u" and np.any(array > np.iinfo(np.int64).max)):
        raise ValueError(f"{name} must contain signed 64-bit representable integers")
    return np.frombuffer(array.astype(np.int64).tobytes(), dtype=np.int64).reshape(array.shape)


def _placement_tables(mapping: np.ndarray, experts: int, ranks: int) -> tuple[np.ndarray, np.ndarray]:
    """Build canonical inverse rows with a stable sort over physical locations."""
    layers, physical = mapping.shape
    layer_ids = np.arange(layers)[:, None]
    counts = np.zeros((layers, experts), dtype=np.int64)
    np.add.at(counts, (layer_ids, mapping), 1)
    if np.any(counts > ranks):
        raise ValueError("Replica count exceeds rank count")
    order = np.argsort(mapping, axis=1, kind="stable")
    sorted_experts = np.take_along_axis(mapping, order, axis=1)
    starts = np.cumsum(counts, axis=1) - counts
    ordinal = np.arange(physical)[None, :] - np.take_along_axis(starts, sorted_experts, axis=1)
    inverse = np.full((layers, experts, ranks), -1, dtype=np.int64)
    inverse[layer_ids, sorted_experts, ordinal] = order
    return inverse, counts


@dataclass(frozen=True)
class Placement:
    generation: int
    physical_to_logical: np.ndarray
    logical_to_physical: np.ndarray
    replica_count: np.ndarray

    def __post_init__(self) -> None:
        if isinstance(self.generation, bool) or not isinstance(self.generation, int) or self.generation < 0:
            raise ValueError("Placement generation must be a non-negative integer")
        for name in ("physical_to_logical", "logical_to_physical", "replica_count"):
            object.__setattr__(self, name, immutable_ints(getattr(self, name), name=name))
        mapping = self.physical_to_logical
        inverse = self.logical_to_physical
        counts = self.replica_count
        if mapping.ndim != 2 or inverse.ndim != 3 or counts.ndim != 2:
            raise ValueError("Invalid placement dimensions")
        layers, experts, ranks = inverse.shape
        if not layers or not experts or not ranks or counts.shape != (layers, experts):
            raise ValueError("Invalid placement dimensions")
        if mapping.shape[0] != layers or not mapping.shape[1] or mapping.shape[1] % ranks:
            raise ValueError("Physical expert capacity must be divisible by ranks")
        slots = mapping.shape[1] // ranks
        if slots > experts or np.any(mapping < 0) or np.any(mapping >= experts):
            raise ValueError("Invalid logical expert ID or local capacity")
        local = np.sort(mapping.reshape(layers, ranks, slots), axis=2)
        if np.any(local[:, :, 1:] == local[:, :, :-1]):
            raise ValueError("A rank cannot hold duplicate logical experts")
        expected_inverse, expected_counts = _placement_tables(mapping, experts, ranks)
        if np.any(expected_counts == 0) or not np.array_equal(counts, expected_counts):
            raise ValueError("Every logical expert must have its declared replicas")
        if not np.array_equal(inverse, expected_inverse):
            raise ValueError("Inconsistent logical-to-physical table")

    @property
    def ranks(self) -> int:
        return self.logical_to_physical.shape[2]

    @property
    def local_slots(self) -> int:
        return self.physical_to_logical.shape[1] // self.ranks

    @property
    def logical_experts(self) -> int:
        return self.replica_count.shape[1]

    @property
    def layers(self) -> int:
        return self.physical_to_logical.shape[0]

    @classmethod
    def from_map(cls, mapping, *, logical_experts: int, ranks: int, generation: int = 0) -> Placement:
        mapping = immutable_ints(mapping, name="physical_to_logical")
        if logical_experts <= 0 or ranks <= 0 or mapping.ndim != 2:
            raise ValueError("Invalid placement dimensions")
        if np.any(mapping < 0) or np.any(mapping >= logical_experts):
            raise ValueError("Invalid logical expert ID")
        inverse, counts = _placement_tables(mapping, logical_experts, ranks)
        return cls(generation, mapping, inverse, counts)

    @classmethod
    def initial(cls, *, layers: int, logical_experts: int, ranks: int,
                redundant_experts: int = 0) -> Placement:
        if min(layers, logical_experts, ranks) <= 0 or redundant_experts < 0:
            raise ValueError("Invalid initial placement capacity")
        physical = logical_experts + redundant_experts
        if physical % ranks or physical // ranks > logical_experts:
            raise ValueError("Invalid initial physical expert capacity")
        mapping = np.tile(np.arange(physical, dtype=np.int64) % logical_experts, (layers, 1))
        return cls.from_map(mapping, logical_experts=logical_experts, ranks=ranks)


@dataclass(frozen=True)
class LoadSample:
    dispatch_id: int
    phase: str
    placement: Placement
    local_counts: np.ndarray

    def __post_init__(self) -> None:
        if isinstance(self.dispatch_id, bool) or not isinstance(self.dispatch_id, int) or self.dispatch_id < 0:
            raise ValueError("Load sample dispatch_id must be a non-negative integer")
        if self.phase not in ("prefill", "decode"):
            raise ValueError("Invalid load sample phase")
        counts = immutable_ints(self.local_counts, name="local_counts")
        expected = (self.placement.ranks, self.placement.layers, self.placement.local_slots)
        if counts.shape != expected or np.any(counts < 0):
            raise ValueError("Invalid physical load sample shape or counts")
        if counts.sum(dtype=object) > np.iinfo(np.int64).max:
            raise ValueError("Load sample count overflow")
        object.__setattr__(self, "local_counts", counts)

    def logical_counts(self) -> np.ndarray:
        """Interpret counts using the layout captured at submission, never the latest one."""
        logical = np.zeros_like(self.placement.replica_count)
        for layer in range(self.placement.layers):
            np.add.at(logical[layer], self.placement.physical_to_logical[layer],
                      self.local_counts[:, layer, :].reshape(-1))
        return logical
