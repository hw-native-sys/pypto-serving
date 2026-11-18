# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""One/two-cycle native parents with explicit reservations and FIFO capture."""

from dataclasses import dataclass
from concurrent.futures import Future
import json
import logging
import threading
import time

import torch
from pypto_serving.config.decoding import DecodeReservation
from .decoding import NativeDecodeRetirement, encode_retirement
from pypto_serving.config.types import DecodeResult

from .profiling import operation
from .native_runner import NativeModelRunner
from .native_runtime import NativeRuntime
from .worker_options import native_worker_options


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedNativeDecode:
    reservation: DecodeReservation
    cache_generation: int
    pages: tuple


@dataclass(frozen=True)
class PendingNativeDecode:
    prepared: PreparedNativeDecode
    submission: Future
    submitted_at: float
    accepted_depth: int

    @property
    def handle(self):
        return self.submission.result()

    def result(self):
        return self.handle.result()


class NativePipelineRunner(NativeModelRunner):
    supports_async_decode_reclaim = True

    def __init__(self, programs, contract, *, cache_initialization="on-demand"):
        super().__init__(programs, contract, cache_initialization=cache_initialization)
        if set(programs.native) != {1, 2}:
            raise ValueError("native pipeline requires both generated programs")
        self.native = NativeRuntime(programs, frame_count=2)
        self._native_lock = threading.RLock()
        self._accepted = []
        self._seen_reservations = set()
        self._diagnostic_id = 0
        self.launch_depth = 1

    def _worker_factory_context(self):
        return native_worker_options()

    def _preflight_input_extent(self):
        return 9

    def finalize_prefill(self, request_ids, sampled_token_ids, sampling_params=None):
        super().finalize_prefill(request_ids, sampled_token_ids, sampling_params)
        if self.native.state is not None:
            self._publish_tables()

    def _publish_tables(self):
        # Actual B1 scheduler ownership is fixed before the first prefill.
        # No copy or mutable table preparation occurs while a parent is accepted.
        if self._accepted:
            raise RuntimeError("cannot mutate shared native tables with accepted parents")
        self.transfers.acquire(self.state.generation, self.programs.single.compiled)
        try:
            self.transfers.stage_tables(self.cache._tables, self.cache._counts)
        except Exception:
            self.transfers.poison()
            raise
        self.transfers.complete()

    @operation("lowlat.native.prepare", category="lowlat")
    def prepare_decode(self, model, batch, *, buffer_slot=0):
        if len(batch.request_ids) != 1 or len(batch.decode_reservations) != 1:
            raise ValueError("native pipeline requires one explicit reservation")
        reservation = batch.decode_reservations[0]
        from .decoding import validate_reservation

        validate_reservation(reservation)
        if not isinstance(reservation, DecodeReservation):
            raise ValueError("native decode reservation is missing")
        with self._native_lock:
            state = self._request(batch.request_ids)
            context = self.request_contexts[state.request_id]
            if (reservation.request_id != state.request_id
                    or reservation.generation != context.execution_generation
                    or reservation.physical_input_end != context.prompt_tokens+context.max_output_tokens+6
                    or batch.constraint_states or batch.stop_token_masks
                    or any(p.temperature != 0 or p.forbidden_token_ids for p in batch.sampling_params)):
                raise ValueError("native reservation differs from the admitted execution context")
            if len(batch.block_ids_by_group) != 1:
                raise ValueError("native preparation requires real scheduler pages")
            pages = tuple((name, tuple(batch.block_ids_by_group[0][name])) for name in self.cache.specs)
            if any(ids != self.cache._page_ids[name] for name, ids in pages):
                raise ValueError("native prepared pages differ from the pinned physical snapshot")
            return PreparedNativeDecode(reservation, state.generation, pages)

    def prepared_decode_requires_token(self, prepared):
        return False

    @operation("lowlat.native.dispatch", category="lowlat")
    def dispatch_prepared_decode(self, model, batch, prepared):
        if not isinstance(prepared, PreparedNativeDecode):
            raise TypeError("native dispatch requires its prepared immutable reservation")
        with self._native_lock:
            state = self._request(batch.request_ids)
            reservation = prepared.reservation
            context = self.request_contexts[state.request_id]
            if (prepared.cache_generation != state.generation
                    or reservation.generation != context.execution_generation
                    or reservation.reservation_id in self._seen_reservations
                    or self.native.state is None or self.native.state.remaining <= 0
                    or len(self._accepted) >= 2
                    or any(ids != self.cache._page_ids[name] for name, ids in prepared.pages)):
                raise RuntimeError("stale, duplicate or unowned native reservation")
            remaining = self.native.state.remaining
            if self._accepted:
                first = self._accepted[0].prepared.reservation
                if (first.cycles != 2 or reservation.cycles != 2 or remaining < 24
                        or first.frame_id == reservation.frame_id):
                    raise RuntimeError("native successor violates24token/two-cycle/free-frame admission")
                remaining -= first.max_output_tokens
            if (remaining < reservation.remaining_lower_bound
                    or (reservation.cycles == 2 and remaining < 7)):
                raise RuntimeError("native reservation has no conservative remaining budget")
            started = time.perf_counter()
            self.native.frames.reserve(reservation.frame_id, generation=state.generation,
                                       cycles=reservation.cycles)
            pending = PendingNativeDecode(prepared, Future(), started, len(self._accepted)+1)
            self._accepted.append(pending)
            self._seen_reservations.add(reservation.reservation_id)
            state.pending_handles.append(pending)
        # A successor submit can wait for its predecessor inside the runtime.
        # Its reserved frame and submission future retain ownership while the
        # completed predecessor is captured independently on the output lane.
        try:
            handle = self.native.frames.submit_reserved(reservation.frame_id,
                generation=prepared.cache_generation, cycles=reservation.cycles)
        except BaseException as error:
            pending.submission.set_exception(error)
            self.transfers.poison()
            raise
        pending.submission.set_result(handle)
        return pending

    @operation("lowlat.native.retire", category="lowlat")
    def reclaim_prepared_decode(self, pending):
        if not isinstance(pending, PendingNativeDecode):
            raise TypeError("native reclaim requires its own accepted ticket")
        # Blocking publication must not hold the admission lock: a second parent
        # may prepare/accept while the first executes, with launch depth still1.
        index = pending.prepared.reservation.frame_id
        pending.submission.result()
        self.native.frames.complete(index)
        with self._native_lock:
            if not self._accepted or self._accepted[0] is not pending:
                raise RuntimeError("native parent capture must retire in accepted order")
            state = self.state
            if state is None or pending.prepared.cache_generation != state.generation:
                raise RuntimeError("native parent belongs to a released generation")
            try:
                packet = self.native.frames.capture(index, generation=state.generation, state=self.native.state)
                if len(packet.token_ids) > pending.prepared.reservation.max_output_tokens:
                    raise RuntimeError("native output exceeds its explicit reservation")
                # Token tuples own the handoff; no downstream reader retains a shared frame.
                result = DecodeResult(None, None, sampled_token_ids=torch.tensor([packet.state.last_token]),
                    accepted_token_ids=[list(packet.token_ids)], num_draft_tokens=None,
                    retired_reservation_ids=(pending.prepared.reservation.reservation_id,),
                    execution_receipts=(encode_retirement(NativeDecodeRetirement(pending.prepared.reservation.reservation_id,
                        state.request_id, pending.prepared.reservation.generation,
                        packet.controls[0][2]+1, packet.state.anchor+1, packet.advances, packet.accepted,
                        len(packet.token_ids), packet.state.remaining, packet.state.epoch)),))
                self.native.commit(packet)
                self.native.frames.consume(index)
                self.native.frames.release(index)
            except Exception:
                self.transfers.poison()
                raise
            self._accepted.pop(0)
            state.pending_handles.remove(pending)
            state.kv_committed_tokens = packet.state.anchor+1
            state.last_sampled_token = packet.state.last_token
            logger.info("LOWLAT_NATIVE_PARENT_METRIC %s", json.dumps(dict(
                reservation_id=pending.prepared.reservation.reservation_id,
                cycles=len(packet.controls), attempted_drafts=5*len(packet.controls),
                accepted_drafts=sum(packet.accepted), emitted=len(packet.token_ids),
                elapsed_ms=(time.perf_counter()-pending.submitted_at)*1000,
                accepted_parents_remaining=len(self._accepted), launch_depth=getattr(self, "launch_depth", 1))))
            return result

    def run_prepared_decode(self, model, batch, prepared):
        return self.reclaim_prepared_decode(self.dispatch_prepared_decode(model, batch, prepared))

    def run_decode(self, model, batch):
        if batch.decode_reservations and batch.decode_reservations[0] is not None:
            return self.run_prepared_decode(model, batch, self.prepare_decode(model, batch))
        # Warmup and offline diagnostics still execute exactly one bounded parent.
        # Production async commands must carry the scheduler's reservation.
        from dataclasses import replace

        state = self._request(batch.request_ids)
        context = self.request_contexts[state.request_id]
        if self.native.state is None or self.native.state.remaining <= 0:
            raise ValueError("native request has no remaining output")
        position = int(batch.seq_lens.reshape(-1)[0])-1
        state.check_decode(position, int(batch.token_ids.reshape(-1)[0]))
        self.cache.tables_for(batch.block_ids_by_group[0], input_end=position+6)
        self._publish_tables()
        self._diagnostic_id -= 1
        remaining = self.native.state.remaining
        cycles = 2 if remaining >= 7 else 1
        # IDs occupy a disjoint positive range from the scheduler's monotonic IDs.
        reservation = DecodeReservation(2**62+self._diagnostic_id, state.request_id,
            context.execution_generation or 1, cycles, min(6*cycles, remaining),
            context.prompt_tokens+context.max_output_tokens+6, remaining, 0)
        if not context.execution_generation:
            self.request_contexts[state.request_id] = replace(context, execution_generation=1)
        batch = replace(batch, decode_reservations=(reservation,))
        return self.run_prepared_decode(model, batch, self.prepare_decode(model, batch))

    def release_finished_requests(self, request_ids):
        # Both capture and discard accepted successors, even after an EOS/abort.
        # No replacement is submitted during this bounded drain.
        if self.state is not None and self.state.request_id in request_ids:
            while self._accepted:
                self.reclaim_prepared_decode(self._accepted[0])
        result = super().release_finished_requests(request_ids)
        if self.state is None:
            self._seen_reservations.clear()
        return result
