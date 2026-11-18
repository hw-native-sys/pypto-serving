# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded absolute-anchor metadata and prefork storage for native K5 requests."""

import torch
from pypto.runtime import ReadOnlyHostTensor

from .bindings import DTYPES, borrow_view, parameter_rows
from .speculation import validate_native_capacity


class NativeInputs:
    MAX_ANCHORS = 2047

    def __init__(self, metadata):
        rows = parameter_rows(metadata)
        self.banks = {}
        for name, row in rows.items():
            if name.startswith("d_") and name.endswith("_bank"):
                shape = row["shape"]
                if shape[:2] != [4, -1] or any(n <= 0 for n in shape[2:]):
                    raise ValueError("unexpected native metadata bank geometry")
                self.banks[name] = torch.zeros((4, self.MAX_ANCHORS, *shape[2:]),
                                               dtype=DTYPES[row["dtype"]]).share_memory_()
        if len(self.banks) != 10:
            raise ValueError("all ten actual drafter metadata fields are required")
        self.tables = torch.empty((4, 3, 1, 32768), dtype=torch.int32).share_memory_()
        logical = torch.arange(32768, dtype=torch.int32)
        for owner in range(4):
            for layer in range(3):
                self.tables[owner, layer, 0] = owner*48+(logical+7*layer)%6
        self.state = torch.zeros((8, 4), dtype=torch.int64).share_memory_()
        self.zero_history = torch.zeros((136, 12288), dtype=torch.bfloat16).share_memory_()
        self.zero_draft_cache = torch.zeros((3, 512, 32, 1, 512), dtype=torch.bfloat16).share_memory_()

    @property
    def inherited(self):
        return (*self.banks.values(), self.state, ReadOnlyHostTensor(self.tables),
                ReadOnlyHostTensor(self.zero_history), ReadOnlyHostTensor(self.zero_draft_cache))

    def prepare_banks(self, *, prompt_tokens, max_output_tokens, rope_cos, rope_sin):
        capacity = len(rope_cos)
        validate_native_capacity(prompt_tokens=prompt_tokens, max_output_tokens=max_output_tokens,
                                 physical_capacity=capacity)
        count = max_output_tokens-1
        if not 1 <= count <= self.MAX_ANCHORS:
            raise ValueError("no native parent is needed for a prefill-only request")
        if (rope_cos.shape != rope_sin.shape or rope_cos.shape[1:] != (64,)
                or rope_cos.dtype != torch.bfloat16 or rope_sin.dtype != torch.bfloat16):
            raise ValueError("native banks require exact uncompressed BF16 RoPE rows")
        anchors = torch.arange(prompt_tokens-1, prompt_tokens-1+count, dtype=torch.int64)
        context_count = (anchors+1).clamp(max=128)
        first = anchors+1-context_count
        columns = torch.arange(128, dtype=torch.int64)
        local = torch.full((count, 4, 128), -1, dtype=torch.int32)
        local[:, 0] = torch.where(columns[None, :] < context_count[:, None],
                                  first[:, None]+columns[None, :], -1).to(torch.int32)
        local[:, 1:, 0] = 0
        context_slots = torch.full((count, 3, 4, 128), -1, dtype=torch.int64)
        query = torch.zeros((count, 4, 20), dtype=torch.int32)
        query_slots = torch.full((count, 3, 4, 20), -1, dtype=torch.int64)
        for owner in range(4):
            position = local[:, owner].to(torch.int64)
            safe = position.clamp(min=0)
            query[:, owner, :5] = ((anchors if owner == 0 else torch.zeros_like(anchors))[:, None]
                                       +torch.arange(1, 6)[None, :]).to(torch.int32)
            query_position = query[:, owner, :5].to(torch.int64)
            for layer in range(3):
                table = self.tables[owner, layer, 0].to(torch.int64)
                context_slots[:, layer, owner] = torch.where(position >= 0,
                                                             table[safe//32]*32+safe%32, -1)
                query_slots[:, layer, owner, :5] = table[query_position//32]*32+query_position%32
        group_context, group_query = local.reshape(count, 512), query.reshape(count, 80)
        values = {"d_context_group_position_ids_bank": group_context,
                  "d_context_group_slot_mapping_bank": context_slots.reshape(count, 3, 512),
                  "d_query_group_position_ids_bank": group_query,
                  "d_query_group_slot_mapping_bank": query_slots.reshape(count, 3, 80)}
        for suffix, table in (("cos", rope_cos), ("sin", rope_sin)):
            def rope(positions):
                flat = positions.clamp(min=0).to(torch.int64).reshape(-1)
                return table.index_select(0, flat).reshape(*positions.shape, 64)
            values["d_context_group_freqs_"+suffix+"_bank"] = rope(group_context)
            values["d_query_group_freqs_"+suffix+"_bank"] = rope(group_query)
            per_owner = rope(query).permute(1, 0, 2, 3)
            host = self.banks["d_query_freqs_"+suffix+"_bank"]
            host[:, :count].copy_(per_owner)
        for name, value in values.items():
            self.banks[name][:, :count].copy_(value.unsqueeze(0).expand(4, *value.shape))
        return count

    def upload_banks(self, arena, destinations, count):
        if type(count) is not int or not 1 <= count <= self.MAX_ANCHORS:
            raise ValueError("invalid native bank extent")
        result = {}
        for name, host in self.banks.items():
            shape = (4, count, *host.shape[2:])
            view = borrow_view(arena.worker, destinations[name], shape)
            for rank, shard in enumerate(view.shards):
                source = host[rank, :count]
                arena.worker.copy_to(shard.data_ptr, source.data_ptr(), source.numel()*source.element_size(),
                                     worker_id=rank)
            result[name] = view
        return result
