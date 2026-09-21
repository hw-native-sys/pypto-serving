# Copyright (c) PyPTO Contributors.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""Router-owned request transaction across a compatible P/D node pool."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import suppress
import time
import uuid

import anyio

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
    ReservePlacementResult,
    capability_compatibility_digest,
)
from pypto_serving.serving.pd.protocol import (
    DecodeOutputWire,
    HandoffKey,
    HandoffStatus,
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
        runtime_manager: FixedPairRuntimeManager | None = None,
    ) -> None:
        self.config = config
        self.directory = directory
        self.journal = journal
        self.metrics = PDMetrics("pd-router", config.run_id)
        self.admission = FairHandoffAdmission(
            config.max_active_handoffs,
            config.max_pending_handoffs,
            state_observer=self._observe_admission,
        )
        self.recovery = FixedPairRecoveryController(
            generation=config.data_generation,
            control_incarnation=config.control_incarnation,
        )
        self.runtime_manager = runtime_manager
        self._replayable: dict[str, ReplayableRequest] = {}
        self._replay_results: dict[str, tuple[DecodeOutputWire, ...]] = {}
        self._automatic_recovery_task: asyncio.Task[RuntimeGeneration] | None = None

    def _observe_admission(self, active: int, queued: int) -> None:
        self.metrics.set_gauge("handoffs.active", active)
        self.metrics.set_gauge("handoffs.queued", queued)
        self.metrics.set_peak_gauge("handoffs.active_peak", active)
        self.metrics.set_peak_gauge("handoffs.queued_peak", queued)

    async def reconcile_startup(self) -> None:
        """Query the journaled owners, then durably fence interrupted routes."""
        unresolved = self.journal.unresolved_bindings
        if not unresolved:
            return
        for binding in unresolved:
            try:
                pair = await self.directory.resolve_binding(
                    binding.prefill_node_id,
                    binding.decode_node_id,
                    binding.prefill_endpoint_generation,
                    binding.decode_endpoint_generation,
                )
                results = await asyncio.gather(
                    pair.prefill_client.post(
                        "/internal/pd/query", HandoffHTTP(binding.key)
                    ),
                    pair.decode_client.post(
                        "/internal/pd/query", HandoffHTTP(binding.key)
                    ),
                    return_exceptions=True,
                )
                summary = "_".join(
                    type(result).__name__
                    if isinstance(result, BaseException)
                    else "STATUS"
                    for result in results
                )
            except (RuntimeError, ValueError) as exc:
                summary = f"BINDING_{type(exc).__name__}"
            self.journal.append(
                "RECOVERY_REQUIRED",
                binding.key,
                error_code=summary[:128],
                prefill_node_id=binding.prefill_node_id,
                decode_node_id=binding.decode_node_id,
                prefill_endpoint_generation=binding.prefill_endpoint_generation,
                decode_endpoint_generation=binding.decode_endpoint_generation,
            )
        self.recovery.require(
            "INTERRUPTED_ROUTER_JOURNAL",
            tuple(binding.key for binding in unresolved),
        )

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
            if (
                self.runtime_manager is not None
                and not replayable.output_published
                and not isinstance(exc, (GeneratorExit, asyncio.CancelledError))
                and (
                    self.recovery.phase is RecoveryPhase.RECOVERY_REQUIRED
                    or request_id in self._replay_results
                )
            ):
                try:
                    if request_id not in self._replay_results:
                        await self._ensure_automatic_recovery()
                    replayed = self._replay_results.pop(request_id)
                    for output in replayed:
                        now_ns = time.monotonic_ns()
                        if previous_output_ns is None:
                            self.metrics.observe_ns(
                                "request.ttft", now_ns - started_ns
                            )
                        else:
                            self.metrics.observe_ns(
                                "request.inter_output",
                                now_ns - previous_output_ns,
                            )
                        previous_output_ns = now_ns
                        final = output
                        yield output
                    if final is None or not final.finished:
                        raise RuntimeError(
                            "automatic re-prefill ended without terminal output"
                        )
                    self.metrics.increment("requests.completed")
                    self.metrics.record_terminal(
                        request_id=request_id,
                        state="COMPLETED",
                        prompt_tokens=final.prompt_tokens,
                        completion_tokens=final.completion_tokens,
                        token_ids=final.token_ids,
                    )
                    return
                except BaseException as recovery_exc:
                    self.metrics.increment("requests.failed")
                    self.metrics.record_terminal(
                        request_id=request_id,
                        state="FAILED",
                        error_code=type(recovery_exc).__name__,
                    )
                    raise recovery_exc from exc
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
            outputs = []
            async for output in self._generate_admitted(
                request.request_kind,
                request.request_json,
                request.request_id,
            ):
                outputs.append(output)
            final = outputs[-1] if outputs else None
            if final is None or not final.finished:
                raise RuntimeError("re-prefill replay ended without terminal Decode output")
            self._replay_results[request.request_id] = tuple(outputs)
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

    async def _ensure_automatic_recovery(self) -> RuntimeGeneration:
        """Join the single runtime recovery task for all affected requests."""
        if self.runtime_manager is None:
            raise RuntimeError("automatic recovery has no runtime manager")
        task = self._automatic_recovery_task
        if task is None or task.done():
            task = asyncio.create_task(
                self.recover_fixed_pair(self.runtime_manager)
            )
            self._automatic_recovery_task = task
        try:
            return await asyncio.shield(task)
        finally:
            if task.done() and self._automatic_recovery_task is task:
                self._automatic_recovery_task = None

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
        self.journal.append(
            "HANDOFF_CREATED",
            key,
            prefill_node_id=pair.prefill.node_id,
            decode_node_id=pair.decode.node_id,
            prefill_endpoint_generation=pair.prefill.endpoint_generation,
            decode_endpoint_generation=pair.decode.endpoint_generation,
        )
        reservation: PlacementReservation | None = None
        execute_task: asyncio.Task | None = None
        next_frame: asyncio.Task | None = None
        decode_stream = None
        terminal = False
        native_may_have_started = False
        transfer_completed = False
        try:
            rejected_decode_nodes: set[str] = set()
            while True:
                self.journal.append(
                    "RESERVATION_ATTEMPT",
                    key,
                    prefill_node_id=pair.prefill.node_id,
                    decode_node_id=pair.decode.node_id,
                    prefill_endpoint_generation=pair.prefill.endpoint_generation,
                    decode_endpoint_generation=pair.decode.endpoint_generation,
                )
                outcome = await pair.decode_client.post(
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
                        prefix_match_spec=prepared.prefix_match_spec,
                    ),
                    ReservePlacementResult,
                )
                if outcome.rejection is None:
                    reservation = outcome.reservation
                    break
                rejection = outcome.rejection
                if (
                    rejection.key != key
                    or rejection.decode_node_id != pair.decode.node_id
                    or rejection.decode_endpoint_generation
                    != pair.decode.endpoint_generation
                ):
                    raise RuntimeError("D returned a rejection for another route")
                self.journal.append(
                    "RESERVATION_REJECTED",
                    key,
                    error_code=rejection.reason,
                    prefill_node_id=pair.prefill.node_id,
                    decode_node_id=pair.decode.node_id,
                    prefill_endpoint_generation=pair.prefill.endpoint_generation,
                    decode_endpoint_generation=pair.decode.endpoint_generation,
                )
                if not rejection.retryable:
                    raise RuntimeError(
                        f"D deterministically rejected reservation: {rejection.reason}"
                    )
                rejected_decode_nodes.add(pair.decode.node_id)
                next_pair = await self.directory.select_decode(
                    pair.prefill.node_id,
                    excluded_decode_node_ids=frozenset(rejected_decode_nodes),
                )
                if (
                    next_pair.prefill.endpoint_generation
                    != pair.prefill.endpoint_generation
                ):
                    raise RuntimeError("P endpoint changed while retrying D reservation")
                pair = next_pair
            if reservation is None:
                raise RuntimeError("D returned an empty reservation outcome")
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
            self.journal.append(
                "HANDOFF_RESERVED",
                key,
                prefill_node_id=pair.prefill.node_id,
                decode_node_id=pair.decode.node_id,
                prefill_endpoint_generation=pair.prefill.endpoint_generation,
                decode_endpoint_generation=pair.decode.endpoint_generation,
            )
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
            self.journal.append(
                "ROUTE_AUTHORIZED",
                key,
                prefill_node_id=pair.prefill.node_id,
                decode_node_id=pair.decode.node_id,
                prefill_endpoint_generation=pair.prefill.endpoint_generation,
                decode_endpoint_generation=pair.decode.endpoint_generation,
            )

            decode_stream = pair.decode_client.stream_decode(
                "/internal/pd/await-decode",
                HandoffHTTP(key),
            )
            first = await anext(decode_stream)
            if first.event != "waiting":
                raise RuntimeError("D did not establish output waiting before Prefill")

            native_may_have_started = True
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
                        prefix_hit_tokens=reservation.prefix_hit_tokens,
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
                    transfer_completed = True
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
                # Any Decode output is downstream of a committed final
                # manifest, even if the Prefill HTTP reply races behind it.
                transfer_completed = True
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
            deterministic_abort = not native_may_have_started or (
                isinstance(exc, (GeneratorExit, asyncio.CancelledError))
                and transfer_completed
            )
            # Starlette runs StreamingResponse bodies inside an AnyIO cancel
            # scope.  Once the client disconnects, every checkpoint remains
            # cancelled unless cleanup is explicitly shielded.  The P/D abort
            # acknowledgements are part of the safety decision, not optional
            # response work, so complete this bounded convergence before
            # propagating the original cancellation.
            with anyio.CancelScope(shield=True):
                abort_confirmed = await self._abort_pair(
                    pair,
                    key,
                    type(exc).__name__,
                    deterministic=deterministic_abort,
                )
                recovery_required = reservation is not None and (
                    not deterministic_abort or not abort_confirmed
                )
                self.journal.append(
                    "RECOVERY_REQUIRED" if recovery_required else "HANDOFF_FAILED",
                    key,
                    error_code=type(exc).__name__,
                )
                if recovery_required:
                    self.recovery.require(type(exc).__name__, (key,))
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
    async def _abort_pair(
        pair,
        key: HandoffKey,
        reason: str,
        *,
        deterministic: bool,
    ) -> bool:
        payload = AbortHandoffHTTP(
            key=key,
            reason=reason[:128],
            deterministic=deterministic,
        )
        try:
            results = await asyncio.wait_for(
                asyncio.gather(
                    pair.prefill_client.post(
                        "/internal/pd/abort", payload, HandoffStatus
                    ),
                    pair.decode_client.post(
                        "/internal/pd/abort", payload, HandoffStatus
                    ),
                    return_exceptions=True,
                ),
                timeout=15,
            )
        except TimeoutError:
            # The durable recovery decision is already recorded.  Abort is a
            # convergence hint and must not keep the public request open for
            # the full node HTTP timeout when a peer is unhealthy.
            return False
        if not deterministic or any(
            isinstance(result, BaseException) for result in results
        ):
            return False
        safe_terminal = {"ABORTED", "FAILED", "RELEASED", "COMPLETED"}
        return all(
            isinstance(result, HandoffStatus) and result.state in safe_terminal
            for result in results
        )
