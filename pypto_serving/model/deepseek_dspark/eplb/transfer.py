# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Borrow raw expert-bank layer views for the device maintenance program."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

import torch

from .plan import EXPERT_TENSORS, LayerMovePlan
from .state import Placement


@dataclass(frozen=True)
class ExpertBankLayout:
    ranks: int
    layers: int
    local_slots: int
    row_elements: tuple[int, ...]
    element_bytes: tuple[int, ...]

    @property
    def expert_bytes(self) -> int:
        return sum(elements * size for elements, size in zip(self.row_elements, self.element_bytes))

    @property
    def receive_bytes_per_rank(self) -> int:
        return self.local_slots * self.expert_bytes

    @classmethod
    def inspect(cls, banks: Mapping[str, Any], placement: Placement) -> ExpertBankLayout:
        rows = []
        sizes = []
        for name in EXPERT_TENSORS:
            bank = banks[name]
            expected_dtype = torch.float32 if name.endswith("_scale") else torch.int8
            shape = tuple(bank.full_shape)
            if (len(shape) < 3 or shape[:2] != (placement.ranks, placement.layers * placement.local_slots)
                    or tuple(bank.worker_ids) != tuple(range(placement.ranks))
                    or bank.dtype != expected_dtype):
                raise ValueError(f"Invalid resident EPLB weight bank: {name}")
            rows.append(math.prod(shape[2:]))
            sizes.append(torch.empty((), dtype=expected_dtype).element_size())
        if rows[0] != rows[2] or rows[0] != rows[4] or rows[1] != rows[3] or rows[0] != rows[1] * rows[5]:
            raise ValueError("Inconsistent W8A8 expert weight and scale geometry")
        return cls(placement.ranks, placement.layers, placement.local_slots, tuple(rows), tuple(sizes))

    def layer_views(self, worker: Any, banks: Mapping[str, Any], layer: int) -> tuple[Any, ...]:
        """Create borrowed contiguous ND views; retain NZ weight bytes without repacking."""
        from pypto.runtime import StackedDeviceTensor

        if not 0 <= layer < self.layers:
            raise ValueError("EPLB layer outside the resident bank")
        result = []
        for name, elements, size in zip(EXPERT_TENSORS, self.row_elements, self.element_bytes):
            bank = banks[name]
            shape = (self.local_slots, elements)
            byte_offset = layer * self.local_slots * elements * size
            shards = tuple(worker.view_tensor(bank.shards[rank], shape=shape,
                                               byte_offset=byte_offset, worker_id=rank)
                           for rank in range(self.ranks))
            result.append(StackedDeviceTensor(shards, (self.ranks, *shape), tuple(range(self.ranks))))
        return tuple(result)

    def controls(self, move: LayerMovePlan | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Build control data to copy into pre-fork shared buffers; None requests a write-free warmup."""
        if move is None:
            sources = torch.zeros((self.ranks, self.local_slots, 2), dtype=torch.int32)
            changed = torch.zeros((self.ranks, self.local_slots), dtype=torch.int32)
        else:
            if move.changed.shape != (self.ranks, self.local_slots):
                raise ValueError("EPLB migration control geometry mismatch")
            sources = torch.from_numpy(move.sources.astype("int32"))
            changed = torch.from_numpy(move.changed.astype("int32"))
        return (sources.unsqueeze(0).repeat(self.ranks, 1, 1, 1),
                changed.unsqueeze(0).repeat(self.ranks, 1, 1))
