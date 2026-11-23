# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Capture the last128 real CP prefill taps into the native history ring."""

import torch


class NativeHistorySeed:
    def __init__(self):
        self.host = torch.zeros((136, 12288), dtype=torch.bfloat16).share_memory_()
        self.generation = None
        self.positions = {}

    def begin(self, generation):
        if self.generation is not None or type(generation) is not int or generation <= 0:
            raise RuntimeError("history seed requires an idle positive generation")
        self.host.zero_()
        self.positions.clear()
        self.generation = generation

    def capture(self, worker, owner, positions, *, prompt_length, generation):
        if generation != self.generation:
            raise RuntimeError("history capture belongs to a different generation")
        if (not isinstance(positions, torch.Tensor) or positions.dtype != torch.int64
                or positions.device.type != "cpu" or positions.ndim != 2 or positions.shape[0] != 8
                or tuple(owner.shape) != (8, positions.shape[1], 12288)
                or owner.dtype != torch.bfloat16 or owner.worker_ids != tuple(range(8))):
            raise ValueError("history seed requires the actual CP8 position/tap layout")
        if type(prompt_length) is not int or prompt_length <= 0:
            raise ValueError("a positive real prompt length is required")
        first = max(0, prompt_length-128)
        selected = []
        seen = set()
        # Validate the complete capture plan before touching the shared seed.
        for rank, rank_positions in enumerate(positions.tolist()):
            for row, position in enumerate(rank_positions):
                if position == -1:
                    continue
                if not 0 <= position < prompt_length or position in seen:
                    raise ValueError("invalid/duplicate CP tap position")
                seen.add(position)
                if position >= first:
                    selected.append((rank, row, position))
        offset = 0
        row_bytes = 12288*2
        while offset < len(selected):
            rank, row, position = selected[offset]
            count = 1
            while offset+count < len(selected):
                nr, r, p = selected[offset+count]
                if nr != rank or r != row+count or p != position+count or position%128+count >= 128:
                    break
                count += 1
            shard = owner.shards[rank]
            worker.copy_from(self.host.data_ptr()+position%128*row_bytes,
                             shard.data_ptr-shard.byte_offset, count*row_bytes,
                             src_offset=shard.byte_offset+row*row_bytes, worker_id=rank)
            for index in range(count):
                self.positions[(position+index)%128] = position+index
            offset += count

    def ready(self, *, prompt_length, generation):
        expected = {position%128: position for position in range(max(0, prompt_length-128), prompt_length)}
        if generation != self.generation or self.positions != expected:
            raise RuntimeError("history seed does not cover the final real prompt positions")
        if not torch.isfinite(self.host.float()).all():
            raise ValueError("prefill history taps contain nonfinite values")
        return self.host

    def release(self):
        self.generation = None
        self.positions.clear()
