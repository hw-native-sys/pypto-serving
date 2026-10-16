# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Model-specific transfer semantics; generic PD code only consumes this interface."""

from dataclasses import replace
from math import prod
from typing import Protocol

from .planner import Geometry, rank_mapping
from .protocol import Component, Contract, Prefix, digest
from .worker import WorkerRuntime


class ModelAdapter(Protocol):
    def prefix_spec(self, tokens, cache) -> Prefix: ...
    def contract(self, model, geometry, topology) -> Contract: ...
    def worker_factory(self, **kwargs): ...
    def mapping(self, topology, source_partition, destination_partition): ...


COMPONENTS = (
    Component("ori", "ori", "ori"), Component("hca_cmp", "cmp_c128", "hca_cmp"),
    Component("csa_cmp", "cmp_c4", "csa_cmp"), Component("idx_k", "idx", "index"),
    Component("idx_scale", "idx", "index"), Component("hca_state", "hca_state", "hca_state", True),
    Component("csa_state", "csa_state", "csa_state", True),
    Component("csa_inner_state", "csa_inner_state", "csa_inner_state", True),
)
TENSORS = dict(zip((c.name for c in COMPONENTS), (
    "kv_cache", "hca_cmp_kv", "csa_cmp_kv", "idx_kv_cache", "idx_kv_scale",
    "hca_compress_state", "csa_compress_state", "csa_inner_compress_state")))


class DeepSeekDSpark:
    @staticmethod
    def mapping(topology, source_partition, destination_partition):
        return rank_mapping(*topology, source_partition, destination_partition)

    @staticmethod
    def validate_mapping(topology, placement, mapping):
        if not mapping:
            raise ValueError("missing rank mapping")
        if mapping != rank_mapping(*topology, mapping[0][0] // topology[1], placement.partition):
            raise ValueError("invalid DSpark partition/TP slot mapping")

    def contract(self, model, geometry, topology):
        if {g.name for g in geometry} != set(TENSORS):
            raise ValueError("DSpark requires all eight physical regions")
        geometry = {g.name: g for g in geometry}
        a, b = geometry["idx_k"], geometry["idx_scale"]
        if (a.layers, a.blocks, a.rows) != (b.layers, b.blocks, b.rows):
            raise ValueError("index K/scale geometry mismatch")
        layout = digest(tuple(geometry[c.name].layout for c in COMPONENTS))
        return Contract("deepseek-v4-dspark-k7", model, layout, tuple(topology), COMPONENTS)

    def prefix_spec(self, tokens, cache):
        groups = tuple(dict.fromkeys(c.group for c in COMPONENTS if not c.final))
        hashes = cache.compute_group_block_hashes(list(tokens), group_names=groups)
        return Prefix(len(tokens), cache.group_prefix_cache_alignment,
                      {k: tuple(h.hex() for h in values) for k, values in hashes.items()})

    @staticmethod
    def validate(config):
        if config.temperature != 0.0 or config.top_p != 1.0 or config.top_k is not None:
            raise ValueError("DSpark PD requires greedy sampling")

    @staticmethod
    def worker_factory(**kwargs):
        return DSparkExtension(**kwargs)


ADAPTERS = {"PyptoDeepSeekV4DSparkExecutor": DeepSeekDSpark}


def select_adapter(executor):
    try:
        return ADAPTERS[executor]()
    except KeyError:
        raise ValueError(f"no PD model adapter for {executor}") from None


class DSparkExtension:
    def __init__(self, *, settings, compiled, worker, cache, draft_states, reserve_state,
                 initialize_device_state, progress_channel=None):
        self.compiled, self.worker, self.cache = compiled, worker, cache
        self.draft_states, self.reserve_state = draft_states, reserve_state
        self.initialize_device_state = initialize_device_state
        self.ranks = compiled.layout.ranks
        self.tp = compiled.layout.tp_size
        self.transfer = WorkerRuntime(settings, self, progress_channel)

    def regions(self):
        cache = self.cache()
        return {name: cache[tensor] for name, tensor in TENSORS.items()}

    def geometry(self):
        layers = tuple(p.layer_id for p in self.compiled.layer_plan)
        hca = tuple(p.layer_id for p in self.compiled.layer_plan if p.compress_ratio == 128)
        csa = tuple(p.layer_id for p in self.compiled.layer_plan if p.compress_ratio == 4)
        mappings = dict(ori=layers, hca_cmp=hca, csa_cmp=csa, idx_k=csa, idx_scale=csa,
                        hca_state=hca, csa_state=csa, csa_inner_state=csa)
        result = []
        for name, tensor in self.regions().items():
            shard = tensor.shards[0]
            shape = tuple(shard.shape)
            indices = mappings[name]
            if not indices or len(shape) < 2 or shape[0] % len(indices):
                raise ValueError("invalid DSpark layer/block/token layout")
            item, remainder = divmod(shard.nbytes, prod(shape))
            if remainder or item <= 0:
                raise ValueError("invalid DSpark storage extent")
            result.append(Geometry(name, indices, shape[0] // len(indices), shape[1],
                                   prod(shape[2:]) * item, str(shard.dtype)))
        return tuple(result)

    def worker_options(self):
        return self.transfer.worker_options()

    def worker_ready(self):
        self.transfer.worker_ready()

    def wrap_prepare(self, prepare):
        def build(batch, *, buffer_slot):
            bootstrap = tuple(r for r in batch.initial_request_ids if r not in self.draft_states)
            for i, request in enumerate(batch.request_ids):
                if request not in bootstrap:
                    continue
                length = int(batch.seq_lens[i].item()) - 1
                if length < 1:
                    raise ValueError("adopted request must contain a prompt")
                state = self.reserve_state(request, group=batch.cache_partitions[i],
                                           prompt_len=length, defer_device_clear=True)
                state.committed_count = length
            return replace(prepare(batch, buffer_slot=buffer_slot), bootstrap_request_ids=bootstrap)
        return build

    def wrap_dispatch(self, dispatch):
        def submit(batch, inputs):
            for i, request in enumerate(batch.request_ids):
                if request in batch.initial_request_ids:
                    state = self.draft_states[request]
                    if not state.device_state_initialized:
                        state.current_token_id = int(batch.token_ids[i].reshape(-1)[-1].item())
                        self.initialize_device_state(state)
            return dispatch(batch, inputs)
        return submit

    def set_profile_active(self, active):
        for owner in self.transfer.owners:
            owner.call("profile", active=active)

    def close(self):
        self.transfer.close()
