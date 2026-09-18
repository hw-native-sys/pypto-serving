# Copyright (c) PyPTO Contributors.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""One-request external Router transaction for fixed 1P1D."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import suppress
import time
import uuid

from pypto_serving.serving.pd.http_api import (
    AbortHandoffHTTP,
    AuthorizeRouteHTTP,
    DecodeStreamFrame,
    ExecutePrefillHTTP,
    HandoffHTTP,
    PlacementReservation,
    PrefillHandoffResult,
    PrepareRequestHTTP,
    PreparedRequest,
    ReservePlacementHTTP,
    capability_compatibility_digest,
)
from pypto_serving.serving.pd.protocol import (
    DecodeOutputWire,
    HandoffKey,
)
from pypto_serving.serving.pd.admission import FairHandoffAdmission
from pypto_serving.serving.pd.metrics import PDMetrics

from .config import RouterConfig
from .directory import WorkerDirectory
from .journal import RouterJournal
from .recovery import (
    FixedPairRecoveryController,
    FixedPairRuntimeManager,
    RecoveryPhase,
    ReplayableRequest,
    RuntimeGeneration,
)


class RouterCoordinator:
    """Router-owned route identity, intent, output, and failure decision."""

    def __init__(
        self,
        config: RouterConfig,
        directory: WorkerDirectory,
        journal: RouterJournal,
    ) -> None:
        self.config = config
        self.directory = directory
        self.journal = journal
        self.admission = FairHandoffAdmission(
            config.max_active_handoffs,
            config.max_pending_handoffs,
        )
        self.metrics = PDMetrics("pd-router", config.run_id)
        self.recovery = FixedPairRecoveryController(
            generation=1,
            control_incarnation=config.control_incarnation,
        )
        self._replayable: dict[str, ReplayableRequest] = {}

    async def reconcile_startup(self) -> None:
        """Query both owners, then durably fence interrupted routes for review."""
        unresolved = self.journal.unresolved
        if not unresolved:
            return
        pair = await self.directory.select()
        for key in unresolved:
            results = await asyncio.gather(
                pair.prefill_client.post("/internal/pd/query", HandoffHTTP(key)),
                pair.decode_client.post("/internal/pd/query", HandoffHTTP(key)),
                return_exceptions=True,
            )
            summary = "_".join(
                type(result).__name__ if isinstance(result, BaseException) else "STATUS"
                for result in results
            )
            self.journal.append("RECOVERY_REQUIRED", key, error_code=summary[:128])
        self.recovery.require("INTERRUPTED_ROUTER_JOURNAL", unresolved)

    async def generate(
        self,
        request_kind: str,
        request_json: bytes,
        request_id: str,
        *,
        publish_immediately: bool = True,
    ) -> AsyncGenerator[DecodeOutputWire, None]:
        started_ns = time.monotonic_ns()
        previous_output_ns: int | None = None
        self.recovery.assert_admission()
        replayable = ReplayableRequest(request_id, request_kind, request_json)
        self._replayable[request_id] = replayable
        self.metrics.increment("requests.total")
        final = None
        try:
            async with self.admission.admit():
                self.metrics.set_gauge("handoffs.active", self.admission.active)
                self.metrics.set_gauge("handoffs.queued", self.admission.queued)
                self.metrics.set_peak_gauge(
                    "handoffs.active_peak", self.admission.active
                )
                self.metrics.set_peak_gauge(
                    "handoffs.queued_peak", self.admission.queued
                )
                stream = self._generate_admitted(
                    request_kind,
                    request_json,
                    request_id,
                )
                async for output in stream:
                    now_ns = time.monotonic_ns()
                    if previous_output_ns is None:
                        self.metrics.observe_ns("request.ttft", now_ns - started_ns)
                    else:
                        self.metrics.observe_ns(
                            "request.inter_output",
                            now_ns - previous_output_ns,
                        )
                    previous_output_ns = now_ns
                    if publish_immediately and not replayable.output_published:
                        replayable = ReplayableRequest(
                            request_id,
                            request_kind,
                            request_json,
                            output_published=True,
                        )
                        self._replayable[request_id] = replayable
                    final = output
                    yield output
                if final is None or not final.finished:
                    raise RuntimeError("Router request ended without a terminal output")
                self.metrics.increment("requests.completed")
                self.metrics.record_terminal(
                    request_id=request_id,
                    state="COMPLETED",
                    prompt_tokens=final.prompt_tokens,
                    completion_tokens=final.completion_tokens,
                    token_ids=final.token_ids,
                )
        except BaseException as exc:
            self.metrics.increment("requests.failed")
            self.metrics.record_terminal(
                request_id=request_id,
                state="FAILED",
                error_code=type(exc).__name__,
            )
            if self.recovery.phase is RecoveryPhase.RUNNING:
                self._replayable.pop(request_id, None)
            raise
        finally:
            if "stream" in locals():
                with suppress(Exception):
                    await stream.aclose()
            self.metrics.observe_ns(
                "request.e2e",
                time.monotonic_ns() - started_ns,
            )
            # This runs after FairHandoffAdmission.__aexit__, so a quiescent
            # Router reports zero rather than the final request's stale slot.
            self.metrics.set_gauge("handoffs.active", self.admission.active)
            self.metrics.set_gauge("handoffs.queued", self.admission.queued)
            if final is not None and final.finished:
                self._replayable.pop(request_id, None)

    async def recover_fixed_pair(
        self,
        manager: FixedPairRuntimeManager,
    ) -> RuntimeGeneration:
        """Run the crash-only barrier and re-prefill unpublished requests.

        A deployment supervisor supplies exact process retirement/restart.  The
        Router owns request eligibility and new-generation route identities.
        """

        async def replay(request: ReplayableRequest) -> None:
            self.directory.control_incarnation = (
                self.recovery.current.control_incarnation
            )
            final = None
            async for output in self._generate_admitted(
                request.request_kind,
                request.request_json,
                request.request_id,
            ):
                final = output
            if final is None or not final.finished:
                raise RuntimeError("re-prefill replay ended without terminal Decode output")
            self._replayable.pop(request.request_id, None)

        generation = await self.recovery.recover(
            manager,
            tuple(self._replayable.values()),
            replay,
        )
        self.directory.control_incarnation = generation.control_incarnation
        self._replayable = {
            request_id: request
            for request_id, request in self._replayable.items()
            if not request.output_published
        }
        return generation

    async def _generate_admitted(
        self,
        request_kind: str,
        request_json: bytes,
        request_id: str,
    ) -> AsyncGenerator[DecodeOutputWire, None]:
        if not request_json or len(request_json) > self.config.max_request_bytes:
            raise ValueError("public request exceeds the Router body limit")
        pair = await self.directory.select()
        prepared = await pair.prefill_client.post(
            "/internal/pd/prepare",
            PrepareRequestHTTP(request_id, request_kind, request_json),
            PreparedRequest,
        )
        key = HandoffKey(
            request_id=request_id,
            handoff_id=uuid.uuid4().hex,
            data_generation=self.recovery.current.data_generation,
            route_epoch=self.config.route_epoch,
            control_incarnation=self.recovery.current.control_incarnation,
        )
        self.journal.append("HANDOFF_CREATED", key)
        reservation: PlacementReservation | None = None
        execute_task: asyncio.Task | None = None
        next_frame: asyncio.Task | None = None
        decode_stream = None
        terminal = False
        try:
            reservation = await pair.decode_client.post(
                "/internal/pd/reserve",
                ReservePlacementHTTP(
                    key=key,
                    prepared_request_id=prepared.prepared_request_id,
                    prepared_digest=prepared.prepared_digest,
                    prompt_token_count=len(prepared.continuation.prompt_token_ids),
                    max_new_tokens=prepared.continuation.max_new_tokens,
                    layout_fingerprint=pair.prefill.capabilities.layout_fingerprint,
                    prefill_node_id=pair.prefill.node_id,
                    prefill_endpoint_generation=pair.prefill.endpoint_generation,
                ),
                PlacementReservation,
            )
            if (
                reservation.key != key
                or reservation.prepared_request_id != prepared.prepared_request_id
                or reservation.prepared_digest != prepared.prepared_digest
                or reservation.decode_node_id != pair.decode.node_id
                or reservation.decode_endpoint_generation
                != pair.decode.endpoint_generation
                or not reservation.reservation_id
                or not reservation.reservation_capability
            ):
                raise RuntimeError("D returned a reservation for another route")
            self.journal.append("HANDOFF_RESERVED", key)
            compatibility_digest = capability_compatibility_digest(
                pair.prefill.capabilities
            )
            await pair.decode_client.post(
                "/internal/pd/authorize",
                AuthorizeRouteHTTP(
                    key=key,
                    prepared_request_id=prepared.prepared_request_id,
                    prepared_digest=prepared.prepared_digest,
                    reservation_id=reservation.reservation_id,
                    reservation_capability=reservation.reservation_capability,
                    compatibility_digest=compatibility_digest,
                    prefill_node_id=pair.prefill.node_id,
                    prefill_endpoint_generation=pair.prefill.endpoint_generation,
                    decode_node_id=pair.decode.node_id,
                    decode_endpoint_generation=pair.decode.endpoint_generation,
                ),
            )
            self.journal.append("ROUTE_AUTHORIZED", key)

            decode_stream = pair.decode_client.stream_decode(
                "/internal/pd/await-decode",
                HandoffHTTP(key),
            )
            first = await anext(decode_stream)
            if first.event != "waiting":
                raise RuntimeError("D did not establish output waiting before Prefill")

            execute_task = asyncio.create_task(
                pair.prefill_client.post(
                    "/internal/pd/execute",
                    ExecutePrefillHTTP(
                        key=key,
                        prepared_request_id=prepared.prepared_request_id,
                        prepared_digest=prepared.prepared_digest,
                        reservation_id=reservation.reservation_id,
                        partition=reservation.partition,
                        block_ids_by_group=reservation.block_ids_by_group,
                        reservation_capability=reservation.reservation_capability,
                        compatibility_digest=compatibility_digest,
                        prefill_node_id=pair.prefill.node_id,
                        prefill_endpoint_generation=pair.prefill.endpoint_generation,
                        decode_node_id=pair.decode.node_id,
                        decode_control_host=reservation.decode_control_host,
                        decode_control_port=reservation.decode_control_port,
                        decode_endpoint_generation=reservation.decode_endpoint_generation,
                    ),
                    PrefillHandoffResult,
                )
            )
            expected_sequence = 1
            prefill_result: PrefillHandoffResult | None = None
            next_frame = asyncio.create_task(anext(decode_stream))
            while True:
                wait_set = {next_frame}
                if execute_task is not None and not execute_task.done():
                    wait_set.add(execute_task)
                done, _ = await asyncio.wait(wait_set, return_when=asyncio.FIRST_COMPLETED)
                if execute_task in done:
                    prefill_result = execute_task.result()
                    execute_task = None
                    if prefill_result.key != key or prefill_result.reservation_id != reservation.reservation_id:
                        raise RuntimeError("P returned a mismatched handoff result")
                    self.journal.append("PREFILL_READY", key)
                    if next_frame not in done:
                        continue
                if next_frame not in done:
                    continue
                try:
                    frame = next_frame.result()
                except StopAsyncIteration as exc:
                    raise EOFError("D Decode stream ended without a terminal output") from exc
                if frame.event == "error":
                    raise RuntimeError(f"D Decode failed: {frame.error_code}")
                output = self._validate_output(frame, key, expected_sequence)
                expected_sequence += 1
                yield output
                if frame.event == "finished":
                    if execute_task is not None:
                        prefill_result = await execute_task
                        execute_task = None
                    if prefill_result is None:
                        raise RuntimeError("D completed before P reported ReadyAck")
                    terminal = True
                    self.journal.append("HANDOFF_COMPLETED", key)
                    return
                next_frame = asyncio.create_task(anext(decode_stream))
        except BaseException as exc:
            self.journal.append(
                "RECOVERY_REQUIRED" if reservation is not None else "HANDOFF_FAILED",
                key,
                error_code=type(exc).__name__,
            )
            if reservation is not None:
                self.recovery.require(type(exc).__name__, (key,))
            await self._abort_pair(pair, key, type(exc).__name__)
            raise
        finally:
            if next_frame is not None and not next_frame.done():
                next_frame.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await next_frame
            if execute_task is not None and not execute_task.done():
                execute_task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await execute_task
            if decode_stream is not None:
                with suppress(Exception):
                    await decode_stream.aclose()
            if not terminal:
                # The journal entry above is the durable recovery decision. No
                # timeout-based allocator release is performed in the Router.
                pass

    @staticmethod
    def _validate_output(
        frame: DecodeStreamFrame,
        key: HandoffKey,
        expected_sequence: int,
    ) -> DecodeOutputWire:
        output = frame.output
        if output is None or output.key != key:
            raise RuntimeError("D returned an output for another handoff")
        if output.output_sequence != expected_sequence:
            raise RuntimeError("D Decode output sequence is stale or discontinuous")
        if (frame.event == "finished") != output.finished:
            raise RuntimeError("D Decode terminal frame and output disagree")
        return output

    @staticmethod
    async def _abort_pair(pair, key: HandoffKey, reason: str) -> None:
        payload = AbortHandoffHTTP(key=key, reason=reason[:128])
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    pair.prefill_client.post("/internal/pd/abort", payload),
                    pair.decode_client.post("/internal/pd/abort", payload),
                    return_exceptions=True,
                ),
                timeout=15,
            )
        except TimeoutError:
            # The durable recovery decision is already recorded.  Abort is a
            # convergence hint and must not keep the public request open for
            # the full node HTTP timeout when a peer is unhealthy.
            return
