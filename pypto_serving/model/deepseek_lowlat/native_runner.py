# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Synchronous K5 serving with real D/M state and one native cycle per step."""

import json
import logging
import time

import torch
from pypto_serving.config.types import DecodeResult, RequestExecutionContext

from .profiling import operation
from .bindings import parameter_rows
from .native_runtime import NativeRuntime
from .npu_runner import LowlatModelRunner
from .speculation import validate_native_capacity


logger = logging.getLogger(__name__)


class NativeModelRunner(LowlatModelRunner):
    def __init__(self, programs, contract, *, cache_initialization="on-demand"):
        super().__init__(programs, contract, cache_initialization=cache_initialization)
        self.native = NativeRuntime(programs)
        self.request_contexts = {}

    def register_contexts(self, contexts):
        for context in contexts:
            if not isinstance(context, RequestExecutionContext) or context.initial_output_tokens:
                raise ValueError("native requests require an explicit new B1 execution context")
            validate_native_capacity(prompt_tokens=context.prompt_tokens,
                                     max_output_tokens=context.max_output_tokens,
                                     physical_capacity=self.contract.compiled_context)
            if context.request_id in self.request_contexts or self.state is not None:
                raise ValueError("native request registration would interleave generations")
        if len(contexts) > 1:
            raise ValueError("native B1 cannot register multiple requests at once")
        self.request_contexts.update((c.request_id, c) for c in contexts)

    def _prepare_additional_hosts(self):
        return self.native.inherited

    def _additional_compiled(self):
        return [program.compiled for program in self.programs.native.values()]

    def _additional_bytes_per_rank(self):
        aliases = set(parameter_rows(self.programs.single_metadata))
        return max(self.native.planned_extra_bytes(aliases))

    def _allocate_additional_frames(self):
        self.native.allocate(self)

    def _request(self, request_ids, prompt_len=None):
        if self.state is None:
            if len(request_ids) != 1 or request_ids[0] not in self.request_contexts:
                raise ValueError("native request has no admission-resolved execution context")
            context = self.request_contexts[request_ids[0]]
            if prompt_len != context.prompt_tokens:
                raise ValueError("native prompt differs from its registered context")
        return super()._request(request_ids, prompt_len)

    def _on_request_started(self, generation):
        self.native.history.begin(generation)

    def _after_prefill(self, batch, result):
        with self.programs.context():
            from models.deepseek_v4_flash_lowlat.prefill_metadata import prefill_tap_positions

            positions = prefill_tap_positions(batch.chunk_starts[0], batch.chunk_lens[0])
        self.native.history.capture(self.worker, result["target_hidden"], positions,
                                    prompt_length=self.state.prompt_len, generation=self.state.generation)

    @operation("lowlat.native.seed", category="lowlat")
    def finalize_prefill(self, request_ids, sampled_token_ids, sampling_params=None):
        state = self._request(request_ids)
        if state.prefilled_tokens != state.prompt_len or len(sampled_token_ids) != 1:
            raise ValueError("native finalize requires one complete real prefill")
        try:
            self.native.start(context=self.request_contexts[state.request_id], generation=state.generation,
                              token=int(sampled_token_ids[0]), rope_cos=self.rope_cos[0], rope_sin=self.rope_sin[0])
        except Exception:
            self.transfers.poison()
            raise
        super().finalize_prefill(request_ids, sampled_token_ids, sampling_params)

    def run_decode(self, model, batch):
        started = time.perf_counter()
        state = self._request(batch.request_ids)
        position = int(batch.seq_lens.reshape(-1)[0])-1
        token = int(batch.token_ids.reshape(-1)[0])
        state.check_decode(position, token)
        if (batch.constraint_states or any(p.temperature != 0 or p.forbidden_token_ids
                                           for p in batch.sampling_params)
                or self.native.state is None or self.native.state.remaining <= 0):
            raise ValueError("native decode requires a continuing unconstrained greedy request")
        self.transfers.acquire(state.generation, self.programs.single.compiled)
        try:
            tables, counts = self.cache.tables_for(batch.block_ids_by_group[0], input_end=position+6)
            self.transfers.stage_tables(tables, counts)
            prepared = time.perf_counter()
            handle = self.native.frames.submit(0, generation=state.generation, cycles=1)
            state.pending_handles.append(handle)
            self.native.frames.complete(0)
            completed = time.perf_counter()
            packet = self.native.frames.capture(0, generation=state.generation, state=self.native.state)
            result = DecodeResult(None, None, sampled_token_ids=torch.tensor([packet.state.last_token]),
                                  accepted_token_ids=[list(packet.token_ids)],
                                  num_draft_tokens=([5] if len(packet.token_ids) == packet.advances[0] else None))
            # The returned immutable snapshot owns output independently of the shared frame.
            self.native.commit(packet)
            self.native.frames.consume(0)
            self.native.frames.release(0)
        except Exception:
            self.transfers.poison()
            raise
        state.pending_handles.remove(handle)
        state.kv_committed_tokens = packet.state.anchor+1
        state.last_sampled_token = packet.state.last_token
        self.transfers.complete()
        captured = time.perf_counter()
        logger.info("LOWLAT_NATIVE_DECODE_METRIC %s", json.dumps(
            dict(position=position, epoch=packet.controls[0][0], emitted=len(packet.token_ids),
                 advance=packet.advances[0], accepted=packet.accepted[0],
                 prepare_ms=(prepared-started)*1000, submit_wait_ms=(completed-prepared)*1000,
                 capture_ms=(captured-completed)*1000, total_ms=(captured-started)*1000)))
        return result

    def _preflight_input_extent(self):
        return 7

    def preflight(self, record):
        self.register_contexts([RequestExecutionContext("__lowlat_warmup__", 1, 2)])
        return super().preflight(record)

    def release_finished_requests(self, request_ids):
        if self.state is not None and self.state.request_id in request_ids:
            for handle in self.state.pending_handles:
                handle.result()
            self.native.release()
        super().release_finished_requests(request_ids)
        for request_id in request_ids:
            self.request_contexts.pop(request_id, None)
