# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host writes to one packed, mutable DSpark prefill embedding slot."""

import torch


class PrefillEmbeddingSlot:
    def __init__(self, *, dirty_rows: int = 0):
        self.used_rows = dirty_rows
        self.dirty = ((0, dirty_rows),) if dirty_rows else ()

    def stage(self, target: torch.Tensor, group_lengths: tuple[int, ...], groups: tuple[int, ...],
              offsets: tuple[int, ...], embeddings: tuple[torch.Tensor, ...]) -> None:
        ranks, rows = target.shape[:2]
        tp = ranks // len(group_lengths)
        flat = target.view(ranks * rows, *target.shape[2:])
        previous = self.dirty
        self.used_rows = max(self.used_rows, ranks * rows)
        # An interrupted staging, submission or copy-back can leave any exposed
        # row dirty. Only a successful forward establishes clean idle groups.
        self.dirty = ((0, self.used_rows),)
        for group, length in enumerate(group_lengths):
            if length == rows:
                continue
            first = group * tp * rows
            # A fully idle TP group has one contiguous padding range.
            for start in range(first, first + tp * rows, rows if length else tp * rows):
                stop = start + (rows if length else tp * rows)
                for lo, hi in previous:
                    begin, end = max(lo, start + length), min(hi, stop)
                    if begin < end:
                        flat[begin:end].zero_()
        for group, offset, embedding in zip(groups, offsets, embeddings, strict=True):
            dest = target[group * tp:(group + 1) * tp, offset:offset + len(embedding)]
            dest.copy_(embedding.unsqueeze(1))

    def completed(self, rows: int, group_lengths: tuple[int, ...], tp: int) -> None:
        # The Lib prefill ABI returns zero x_hc for idle groups. Every row of
        # an active group (including padding) may have been written by the fwd.
        group_rows = tp * rows
        dirty = [(g * group_rows, (g + 1) * group_rows)
                 for g, length in enumerate(group_lengths) if length]
        end = len(group_lengths) * group_rows
        if end < self.used_rows:
            dirty.append((end, self.used_rows))
        self.dirty = tuple(dirty)
