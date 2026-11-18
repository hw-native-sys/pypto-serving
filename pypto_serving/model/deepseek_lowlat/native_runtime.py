# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Own native K5 resources; each decode call submits exactly one parent."""

from math import prod

import torch

from .bindings import DTYPES, parameter_rows
from .history import NativeHistorySeed
from .native_frames import HOST_SHAPES, NativeFrames, allocate_native_hosts
from .native_inputs import NativeInputs
from .speculation import NativeState


STATIC_FILLS = {
    "s_sampling_temperatures": 0, "s_sampling_top_ks": 1, "s_sampling_seeds": 0,
    "s_num_tokens_per_owner": 6, "s_bridge_markov_ids": 0,
    "s_logit_row_indices": 0, "m_logit_row_indices": -1,
    "m_num_sampled": 1, "m_next_prefill_tokens": 0,
}


class NativeRuntime:
    def __init__(self, programs, *, frame_count=1):
        self.programs = programs
        self.rows = parameter_rows(programs.native_metadata[1])
        self.inputs = NativeInputs(programs.native_metadata[1])
        self.history = NativeHistorySeed()
        self.hosts = allocate_native_hosts(frame_count)
        self.common, self.outputs = {}, [{} for _ in range(frame_count)]
        self.frames = None
        self.state = None
        self.next_epoch = 1
        self.static = {}
        for name, fill in STATIC_FILLS.items():
            row = self.rows[name]
            self.static[name] = torch.full(row["shape"], fill, dtype=DTYPES[row["dtype"]]).share_memory_()
        self.static["s_logit_row_indices"][:] = torch.arange(6, dtype=torch.int32)
        self.static["m_logit_row_indices"][:, :5] = torch.arange(5, dtype=torch.int32)

    @property
    def inherited(self):
        return (*self.inputs.inherited, self.history.host, *self.static.values(),
                *(value for frame in self.hosts for value in frame.values()),
                *self.programs.draft_weights.inherited)

    def _shape(self, name):
        shape = self.rows[name]["shape"]
        if name in self.inputs.banks:
            return (4, self.inputs.MAX_ANCHORS, *shape[2:])
        if shape is None or any(size <= 0 for size in shape):
            raise ValueError("native dynamic parameter has no explicit allocation plan: "+name)
        return tuple(shape)

    def planned_extra_bytes(self, target_bindings):
        result = [self.programs.draft_weights.bytes_for_rank(rank) for rank in range(8)]
        for name, row in self.rows.items():
            if name in HOST_SHAPES or name in self.programs.draft_weights.hosts:
                continue
            if (name.startswith("s_") and name not in STATIC_FILLS
                    and name[2:] in target_bindings and row["direction"] != "Out"):
                continue
            shape = self._shape(name)
            size = prod(shape[1:])*torch.empty((), dtype=DTYPES[row["dtype"]]).element_size()
            for rank in range(shape[0]):
                result[rank] += size*(len(self.outputs) if row["direction"] == "Out" else 1)
        return result

    def allocate(self, runner):
        self.arena, self.worker, self.config = runner.arena, runner.worker, runner.run_config
        self.common = self.programs.draft_weights.upload(self.arena)
        aliases = {**runner.weights.owners, **runner.cache.owners, **runner.decode_bindings}
        for name, row in self.rows.items():
            if name in self.common or name in HOST_SHAPES:
                continue
            if row["direction"] == "Out":
                for outputs in self.outputs:
                    outputs[name] = self.arena.allocate(self._shape(name), DTYPES[row["dtype"]])
            elif name.startswith("s_") and name not in STATIC_FILLS and name[2:] in aliases:
                owner = aliases[name[2:]]
                expected = row["shape"]
                if (len(owner.shape) != len(expected) or owner.dtype != DTYPES[row["dtype"]]
                        or any(want >= 0 and got != want for got, want in zip(owner.shape, expected))):
                    raise ValueError("native shared target alias differs from actual metadata: "+name)
                self.common[name] = owner
            else:
                self.common[name] = self.arena.allocate(self._shape(name), DTYPES[row["dtype"]])
        for name, host in self.static.items():
            self.arena.copy(self.common[name], host)
        self.arena.copy(self.common["d_block_tables"], self.inputs.tables)
        # Unknown inputs must never receive a guessed all-zero initializer.
        allowed = (set(self.inputs.banks) | set(STATIC_FILLS)
                   | {"d_block_tables", "d_kv_caches", "s_history_ring", "s_request_state"})
        for name, row in self.rows.items():
            if (row["direction"] != "Out" and name not in self.programs.draft_weights.hosts
                    and not (name.startswith("s_") and name[2:] in aliases) and name not in allowed):
                raise ValueError("native input has no actual source: "+name)

    def start(self, *, context, generation, token, rope_cos, rope_sin):
        if self.frames is not None and not self.frames.idle():
            raise RuntimeError("cannot initialize a request while native frames remain owned")
        if self.state is not None or not 0 < self.next_epoch < 2**31-2048:
            raise RuntimeError("native request still active or communication epoch exhausted")
        if context.max_output_tokens == 1:
            return
        count = self.inputs.prepare_banks(prompt_tokens=context.prompt_tokens,
                                          max_output_tokens=context.max_output_tokens,
                                          rope_cos=rope_cos, rope_sin=rope_sin)
        common = dict(self.common)
        common.update(self.inputs.upload_banks(self.arena, self.common, count))
        for rank, shard in enumerate(self.common["d_kv_caches"].shards):
            host = self.inputs.zero_draft_cache
            self.worker.copy_to(shard.data_ptr, host.data_ptr(), host.numel()*host.element_size(), worker_id=rank)
        for rank, shard in enumerate(self.common["s_history_ring"].shards):
            host = self.inputs.zero_history
            self.worker.copy_to(shard.data_ptr, host.data_ptr(), host.numel()*host.element_size(), worker_id=rank)
        seed = self.history.ready(prompt_length=context.prompt_tokens, generation=generation)
        self.worker.copy_to(self.common["s_history_ring"].shards[0].data_ptr, seed.data_ptr(), seed.numel()*2,
                            worker_id=0)
        state = NativeState(self.next_epoch, context.prompt_tokens-1, context.max_output_tokens-1, token)
        for rank in range(8):
            self.inputs.state[rank] = torch.tensor(tuple(vars(state).values()), dtype=torch.int64)
        self.arena.copy(self.common["s_request_state"], self.inputs.state)
        self.frames = NativeFrames(self.worker, {c: p.compiled for c, p in self.programs.native.items()},
                                   self.programs.native_metadata, common, self.outputs, self.hosts,
                                   self.config)
        self.state = state

    def commit(self, packet):
        self.state = packet.state
        self.next_epoch = packet.state.epoch

    def release(self):
        if self.frames is not None and not self.frames.idle():
            raise RuntimeError("native frame has not retired before request release")
        self.state = None
        self.history.release()
