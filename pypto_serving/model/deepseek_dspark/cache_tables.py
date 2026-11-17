# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Slot-local physical mappings; callers retain the existing dispatch/reclaim fence."""

from collections.abc import Sequence

import numpy as np
import torch


class DecodeTableRows:
    """Update a group row and its TP/owner mirrors only when its binding changes."""

    def __init__(self, table: torch.Tensor, *, ring: bool, fillers: Sequence[int],
                 replicas: Sequence[torch.Tensor] = (), owner: torch.Tensor | None = None):
        self.table = table
        self._array = table.numpy()
        self._ring = ring
        self._replicas = tuple(replicas)
        self._owner = owner
        self._bindings: list[tuple | None] = [None] * len(fillers)
        for row, block in enumerate(fillers):
            self.update(row, None, (block,))

    def update(self, row: int, identity: tuple | None, blocks: tuple[int, ...]) -> bool:
        binding = (identity, blocks)
        if self._bindings[row] == binding:
            return False
        depth = self._array.shape[1]
        if not blocks or min(blocks) < 0 or (not self._ring and len(blocks) > depth):
            raise ValueError("invalid physical block row")
        target = self._array[row]
        if self._ring:
            repeats, tail = divmod(depth, len(blocks))
            target[:repeats * len(blocks)].reshape(repeats, len(blocks))[:] = blocks
            target[repeats * len(blocks):] = blocks[:tail]
        else:
            target.fill(-1)
            target[:len(blocks)] = blocks
        for replica in self._replicas:
            replica[row].copy_(self.table[row])
        if self._owner is not None and row < self._owner.shape[0]:
            self._owner[row].copy_(self.table[row])
        self._bindings[row] = binding
        return True


class DrafterTableRows:
    """One slot/bucket's dense lease mapping, independent of token positions."""

    def __init__(self, table: torch.Tensor, patterns: np.ndarray, filler_base: int):
        self._array = table.numpy()
        self._patterns = patterns
        self._filler = filler_base
        ranks, _, batch, _ = table.shape
        self._bindings: list[list[tuple | None]] = [[None] * batch for _ in range(ranks)]
        np.add(patterns[None, :, None, :], filler_base, out=self._array)

    def update(self, rank: int, row: int, identity: tuple | None, base: int) -> bool:
        binding = None if identity is None else (identity, base)
        if self._bindings[rank][row] == binding:
            return False
        np.add(self._patterns, self._filler if identity is None else base,
               out=self._array[rank, :, row, :])
        self._bindings[rank][row] = binding
        return True
