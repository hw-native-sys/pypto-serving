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

from pypto_serving.tools.profile import profile_span
from .planner import Geometry, rank_mapping
from .protocol import Component, Contract, Prefix, digest
from .worker import WorkerRuntime


class ModelAdapter(Protocol):
    def prefix_spec(self, tokens, cache) -> Prefix: ...
    def contract(self, model, geometry) -> Contract: ...
    def worker_factory(self, **kwargs): ...
    def mapping(self, source_topology, destination_topology, source_partition, destination_partition): ...


CACHE_COMPONENTS = (
    Component("ori", "ori", "ori"), Component("hca_cmp", "cmp_c128", "hca_cmp"),
    Component("csa_cmp", "cmp_c4", "csa_cmp"), Component("idx_k", "idx", "index"),
    Component("idx_scale", "idx", "index"), Component("hca_state", "hca_state", "hca_state", True),
    Component("csa_state", "csa_state", "csa_state", True),
    Component("csa_inner_state", "csa_inner_state", "csa_inner_state", True),
)
COMPONENTS = CACHE_COMPONENTS + (Component("bootstrap_hidden", "bootstrap", "bootstrap", True),)
TENSORS = dict(zip((c.name for c in CACHE_COMPONENTS), (
    "kv_cache", "hca_cmp_kv", "csa_cmp_kv", "idx_kv_cache", "idx_kv_scale",
    "hca_compress_state", "csa_compress_state", "csa_inner_compress_state")))


class DeepSeekDSpark:
    @staticmethod
    def mapping(source_topology, destination_topology, source_partition, destination_partition):
        return rank_mapping(source_topology, destination_topology, source_partition, destination_partition)

    @staticmethod
    def validate_mapping(source_topology, destination_topology, placement, mapping):
        if not mapping:
            raise ValueError("missing rank mapping")
        if mapping != rank_mapping(source_topology, destination_topology,
                                   source_topology.partition(mapping[0][0]), placement.partition):
            raise ValueError("invalid DSpark partition/TP slot mapping")

    def contract(self, model, geometry):
        if {g.name for g in geometry} != {c.name for c in COMPONENTS}:
            raise ValueError("DSpark requires target cache, final state and bootstrap hidden")
        geometry = {g.name: g for g in geometry}
        a, b = geometry["idx_k"], geometry["idx_scale"]
        if (a.layers, a.blocks, a.rows) != (b.layers, b.blocks, b.rows):
            raise ValueError("index K/scale geometry mismatch")
        layout = digest(tuple(geometry[c.name].layout for c in COMPONENTS))
        return Contract("deepseek-v4-dspark-k7", model, layout, COMPONENTS)

    def prefix_spec(self, tokens, cache):
        groups = tuple(dict.fromkeys(c.group for c in COMPONENTS if not c.final))
        hashes = (cache.compute_group_block_hashes(list(tokens), group_names=groups)
                  if cache.enable_prefix_cache else {})
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
    def __init__(self, *, settings, runner, progress_channel=None):
        self.runner, self.compiled = runner, runner._compiled
        self.worker, self.cache = runner._shared_l3_worker, runner._materialize_decode_device_cache
        self.draft_states = runner._drafter_states
        self.ranks = self.compiled.layout.ranks
        self.tp = self.compiled.layout.tp_size
        if self.compiled.prefill_capture is None or (
            runner.speculative and self.compiled.bootstrap_metadata is None
        ):
            raise ValueError("PD requires device prompt-tail capture and Decode bootstrap")
        self.transfer = WorkerRuntime(settings, self, progress_channel)

    def regions(self):
        cache = self.cache()
        return {**{name: cache[tensor] for name, tensor in TENSORS.items()},
                "bootstrap_hidden": self.runner._device_scratch[("prefill_state", "hidden")]}

    def geometry(self):
        layers = tuple(p.layer_id for p in self.compiled.layer_plan)
        hca = tuple(p.layer_id for p in self.compiled.layer_plan if p.compress_ratio == 128)
        csa = tuple(p.layer_id for p in self.compiled.layer_plan if p.compress_ratio == 4)
        mappings = dict(ori=layers, hca_cmp=hca, csa_cmp=csa, idx_k=csa, idx_scale=csa,
                        hca_state=hca, csa_state=csa, csa_inner_state=csa)
        result = []
        regions = self.regions()
        for name in TENSORS:
            tensor = regions[name]
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
        tail = regions["bootstrap_hidden"].shards[0]
        slots, rows, _ = tail.shape
        result.append(Geometry("bootstrap_hidden", (0,), slots, rows,
                               tail.nbytes // (slots * rows), str(tail.dtype)))
        return tuple(result)

    def worker_options(self):
        return self.transfer.worker_options()

    def worker_ready(self):
        self.transfer.worker_ready()

    def wrap_capture(self, capture):
        if self.runner.speculative:
            return capture
        window = self.compiled.layout.sliding_window
        def retain(batch, inputs, **kwargs):
            # P only exports the final prompt tail. Keep the existing all-rank
            # collective intact when any packed request overlaps that window.
            if any(row.chunk_start < row.prompt_len and row.chunk_end > max(0, row.prompt_len - window)
                   for row in inputs.descriptors):
                return capture(batch, inputs, **kwargs)
        return retain

    def wrap_prepare(self, prepare):
        def build(batch, *, buffer_slot):
            bootstrap = tuple(r for r in batch.initial_request_ids
                              if not self.draft_states[r].device_state_initialized)
            return replace(prepare(batch, buffer_slot=buffer_slot), bootstrap_request_ids=bootstrap)
        return build

    def wrap_dispatch(self, dispatch):
        def submit(batch, inputs):
            fresh = [i for i, request in enumerate(batch.request_ids)
                     if request in batch.initial_request_ids
                     and not self.draft_states[request].device_state_initialized]
            if fresh:
                with profile_span("pd.dspark.bootstrap", cat="pd", args={"requests": len(fresh)}):
                    self.bootstrap(batch, fresh)
            return dispatch(batch, inputs)
        return submit

    def handle(self, operation, value):
        if operation == "reserve_state":
            state = self.runner._reserve_drafter_state(value["request"], group=value["partition"],
                prompt_len=value["prompt_len"], defer_device_clear=True)
            state.committed_count = state.prompt_len
            return {"bootstrap": state.lease}
        if operation == "source_state":
            import torch
            state = self.draft_states[value]
            window = self.compiled.layout.sliding_window
            expected = torch.arange(max(0, state.prompt_len - window), state.prompt_len, dtype=torch.int64)
            if state.prefill_tail_positions is None or not torch.equal(state.prefill_tail_positions, expected):
                raise RuntimeError("handoff requires the complete prompt tail")
            return {"bootstrap": state.lease}
        if operation == "discard_state":
            self.runner.release_finished_requests((value,))
            return None
        raise ValueError("unknown model operation")

    def bootstrap(self, batch, fresh):
        """Reuse terminal-prefill bootstrap after the asynchronous final handoff.

        Only hidden rows cross the network. Absolute positions are determined
        by the agreed prompt length; lease generations remain local to D.
        """
        import torch
        from pypto_serving.model.deepseek_dspark.npu_runner import (
            DSPARK_DRAFTER_QUERY_WIDTH, DSPARK_SAMPLED_IDS_PAD, DSparkStepDescriptor,
        )

        runner = self.runner
        if self.compiled.bootstrap_metadata is None:
            raise RuntimeError("PD continuation requires device prefill bootstrap")
        with profile_span("pd.dspark.bootstrap.wait", cat="pd", args={"requests": len(fresh)}):
            runner._wait_for_pending_decode_dispatches()
        requests = [batch.request_ids[i] for i in fresh]
        # Scrub reused draft KV, never the received target cache or hidden tail.
        for request in requests:
            state = self.draft_states[request]
            key = state.group, state.lease
            if key in runner._drafter_dirty_leases:
                runner._clear_drafter_lease(*key)
                runner._drafter_dirty_leases.remove(key)
        runner._grant_prefill_generations(requests)
        positions = runner._device_scratch[("prefill_state", "positions")]
        sampled = runner._prefill_prepared_device["sampled_ids"]
        window = self.compiled.layout.sliding_window
        ordinals = {}
        for index in fresh:
            request = batch.request_ids[index]
            state = self.draft_states[request]
            ordinal = ordinals.get(state.group, 0)
            ordinals[state.group] = ordinal + 1
            length = state.prompt_len
            start = max(0, length - window)
            state.prefill_tail_start, state.prefill_tail_end = start, length
            state.prefill_tail_positions = torch.arange(start, length, dtype=torch.int64)
            state.prefill_descriptor = DSparkStepDescriptor(
                state.lease, state.generation, state.group, length - 1, 1, length, 0, 1, 0, ordinal,
                bootstrap_enabled=(length + DSPARK_DRAFTER_QUERY_WIDTH - 1
                                   < runner._require_rope_tables().max_position))
            ring = torch.full((window,), -1, dtype=torch.int32)
            ring[state.prefill_tail_positions % window] = state.prefill_tail_positions.to(torch.int32)
            token = int(batch.token_ids[index].reshape(-1)[-1].item())
            state.current_token_id = token
            row = torch.full((DSPARK_SAMPLED_IDS_PAD,), token, dtype=torch.int32)
            for rank in range(state.group * self.tp, (state.group + 1) * self.tp):
                for target, source, slot in ((positions, ring, state.lease), (sampled, row, ordinal)):
                    size = source.numel() * source.element_size()
                    self.worker().copy_to(target.shards[rank].data_ptr, source.data_ptr(), size,
                                          dst_offset=slot * size, worker_id=target.worker_ids[rank])
        with profile_span("pd.dspark.bootstrap.execute", cat="pd", args={"requests": len(fresh)}):
            runner._bootstrap_prefill_requests(requests, [0] * len(requests))

    def set_profile_active(self, active):
        for owner in self.transfer.owners:
            owner.call("profile", active=active)

    def close(self):
        self.transfer.close()
