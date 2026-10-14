# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Router routing/admission; node-owned handoff and public response passthrough."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import aclosing, suppress
import time
import uuid

import anyio
import msgspec

from pypto_serving.serving.pd.config import PD_SCHEMA_VERSION
from pypto_serving.serving.pd.generation import generation_body, public_document, request_fingerprint
from pypto_serving.serving.pd.http_api import AbortHandoffHTTP, HandoffHTTP
from pypto_serving.serving.pd.protocol import GenerationContext, HandoffKey, HandoffStatus, PrefillResult
from pypto_serving.serving.pd.admission import FairHandoffAdmission
from pypto_serving.serving.pd.observability import PDMetrics

from .client import PublicResult
from .config import RouterConfig
from .directory import WorkerDirectory
from .journal import RouterJournal
from .recovery import (
    FixedPairRecoveryController, FixedPairRuntimeManager, RecoveryPhase,
    ReplayableRequest, RuntimeGeneration,
)


class SourceRejected(RuntimeError):
    def __init__(self, result: PrefillResult):
        super().__init__(f"Prefill rejected safely: {result.error_message or result.error_code}")
        self.result = result


class DecodeStreamError(RuntimeError):
    def __init__(self, frame: bytes):
        super().__init__("Decode returned a public SSE error")
        self.frame = frame


def sse_document(frame: bytes):
    data = b"\n".join(line[5:].lstrip() for line in frame.splitlines() if line.startswith(b"data:"))
    if data == b"[DONE]":
        return None
    value = msgspec.json.decode(data)
    if not isinstance(value, dict):
        raise ValueError("Decode SSE data is not an object")
    return value


def terminal_document(value) -> bool:
    return value is None or value.get("usage") is not None or any(
        choice.get("finish_reason") is not None for choice in value.get("choices", ())
    )


class RouterCoordinator:
    def __init__(self, config: RouterConfig, directory: WorkerDirectory, journal: RouterJournal,
                 runtime_manager: FixedPairRuntimeManager | None = None) -> None:
        self.config, self.directory, self.journal = config, directory, journal
        self.metrics = PDMetrics("pd-router", config.run_id)
        self.admission = FairHandoffAdmission(config.max_active_handoffs, config.max_pending_handoffs,
                                              state_observer=self._observe_admission)
        self.requests = FairHandoffAdmission(
            config.max_pending_handoffs, config.max_pending_handoffs,
            state_observer=lambda active, queued: self.metrics.set_gauge("requests.active", active),
        )
        self.recovery = FixedPairRecoveryController(config.data_generation, config.control_incarnation)
        self.runtime_manager = runtime_manager
        self._replayable: dict[str, ReplayableRequest] = {}
        self._replay_results: dict[str, tuple[bytes | PublicResult, ...]] = {}
        self._automatic_recovery_task: asyncio.Task | None = None

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
        self, request_kind: str, request_json: bytes, request_id: str, *,
        publish_immediately: bool = True, _signal_ready: bool = False,
    ) -> AsyncGenerator[bytes | PublicResult | None, None]:
        started = time.monotonic_ns()
        previous = None
        terminal = []
        usage = {}
        signalled = False
        replayable = ReplayableRequest(request_id, request_kind, request_json)
        self.recovery.assert_admission()
        self._replayable[request_id] = replayable
        self.metrics.increment("requests.total")

        async def consume(stream):
            nonlocal previous, replayable, usage, signalled
            async with aclosing(stream):
                async for output in stream:
                    if output is None:
                        signalled = True
                        yield None
                        continue
                    value = (msgspec.json.decode(output.body) if isinstance(output, PublicResult)
                             else sse_document(output))
                    if value and value.get("usage") is not None:
                        usage = value["usage"]
                    if isinstance(output, PublicResult) or terminal_document(value):
                        terminal.append(output)
                        continue
                    now = time.monotonic_ns()
                    self.metrics.observe_ns("request.ttft" if previous is None else "request.inter_output",
                                            now - (started if previous is None else previous))
                    previous = now
                    if publish_immediately:
                        replayable = ReplayableRequest(request_id, request_kind, request_json, True)
                        self._replayable[request_id] = replayable
                    yield output

        try:
            async with self.requests.admit(), self.admission.admit() as release_handoff:
                consumed = consume(self._generate_admitted(
                    request_kind, request_json, request_id, signal_ready=_signal_ready, on_ready=release_handoff,
                ))
                async with aclosing(consumed):
                    async for output in consumed:
                        yield output
                if not terminal:
                    raise RuntimeError("Decode response has no terminal result")
        except BaseException as exc:
            if (self.runtime_manager is not None and not replayable.output_published
                    and not isinstance(exc, (GeneratorExit, asyncio.CancelledError))
                    and self.recovery.phase is RecoveryPhase.RECOVERY_REQUIRED):
                await self._ensure_automatic_recovery()
                terminal.clear()
                if _signal_ready and not signalled:
                    yield None
                async def replay():
                    for output in self._replay_results.pop(request_id):
                        yield output
                async with aclosing(consume(replay())) as consumed:
                    async for output in consumed:
                        yield output
            else:
                self.metrics.increment("requests.failed")
                self.metrics.record_terminal(request_id=request_id, state="FAILED", error_code=type(exc).__name__)
                if self.recovery.phase is RecoveryPhase.RUNNING:
                    self._replayable.pop(request_id, None)
                raise
        finally:
            self.metrics.observe_ns("request.e2e", time.monotonic_ns() - started)
        self.metrics.increment("requests.completed")
        self.metrics.record_terminal(
            request_id=request_id, state="COMPLETED",
            prompt_tokens=usage.get("prompt_tokens", 0), completion_tokens=usage.get("completion_tokens", 0),
        )
        self._replayable.pop(request_id, None)
        # Release sockets, durable handoff intent and both budgets before
        # finish/usage/[DONE] reaches a client that may immediately disconnect.
        for output in terminal:
            yield output

    async def start_stream(self, request_kind, request_json, request_id):
        """Wait for D to accept P's prepared input, not for P's cleanup."""
        stream = self.generate(request_kind, request_json, request_id, _signal_ready=True)
        try:
            if await anext(stream) is not None:
                raise RuntimeError("Router stream missed the Ready boundary")
        except BaseException:
            await stream.aclose()
            raise
        return stream

    async def recover_fixed_pair(self, manager: FixedPairRuntimeManager) -> RuntimeGeneration:
        async def replay(request: ReplayableRequest):
            self.directory.control_incarnation = self.recovery.current.control_incarnation
            self._replay_results[request.request_id] = tuple(
                [output async for output in self._generate_admitted(
                    request.request_kind, request.request_json, request.request_id,
                ) if output is not None]
            )
        generation = await self.recovery.recover(manager, tuple(self._replayable.values()), replay)
        self.directory.control_incarnation = generation.control_incarnation
        return generation

    async def _ensure_automatic_recovery(self):
        task = self._automatic_recovery_task
        if task is None or task.done():
            task = asyncio.create_task(self.recover_fixed_pair(self.runtime_manager))
            self._automatic_recovery_task = task
        return await asyncio.shield(task)

    async def _generate_admitted(self, kind, raw, request_id, *, signal_ready=False, on_ready=None):
        if not raw or len(raw) > self.config.max_request_bytes:
            raise ValueError("public request exceeds the Router body limit")
        public = public_document(raw)
        if "pd_transfer" in public:
            raise ValueError("pd_transfer is reserved for Router assignments")
        pair = await self.directory.select()
        excluded = set()
        while True:
            try:
                async with aclosing(self._attempt(pair, kind, raw, request_id, signal_ready, on_ready)) as attempt:
                    async for output in attempt:
                        yield output
                return
            except SourceRejected as exc:
                if not exc.result.retryable:
                    if exc.result.error_code in ("ValueError", "ValidationError", "TypeError", "INVALID_DECODE_SEED"):
                        raise ValueError(str(exc)) from exc
                    raise
                if exc.result.state != "NOT_STARTED":
                    raise RuntimeError("retry requires a no-compute proof") from exc
                excluded.add(pair.decode.node_id)
                next_pair = await self.directory.select_decode(
                    pair.prefill.node_id, excluded_decode_node_ids=frozenset(excluded),
                )
                if next_pair.prefill.endpoint_generation != pair.prefill.endpoint_generation:
                    raise RuntimeError("P incarnation changed during target reselection") from exc
                pair = next_pair

    async def _prefill(self, client, path, body):
        response = await client.open_generation(path, body)
        try:
            value = await response.json()
            result = msgspec.convert(value.get("pd_transfer"), type=PrefillResult)
            if response.status >= 400 and result.state not in ("NOT_STARTED", "FAILED"):
                raise RuntimeError("P failed without a bound safe-failure result")
            return result
        finally:
            await response.aclose()

    async def _attempt(self, pair, kind, raw, request_id, signal_ready, on_ready):
        key = HandoffKey(request_id, uuid.uuid4().hex, self.recovery.current.data_generation,
                         self.config.route_epoch, self.recovery.current.control_incarnation)
        context = GenerationContext(
            key, request_fingerprint(kind, public_document(raw)),
            pair.prefill.node_id, pair.prefill.endpoint_generation,
            pair.decode.node_id, pair.decode.endpoint_generation,
            pair.decode.control_host, pair.decode.control_port, PD_SCHEMA_VERSION,
        )
        body = generation_body(raw, kind, context)
        path = "/v1/chat/completions" if kind == "chat" else "/v1/completions"
        self.journal.append(
            "HANDOFF_CREATED", key, prefill_node_id=pair.prefill.node_id, decode_node_id=pair.decode.node_id,
            prefill_endpoint_generation=pair.prefill.endpoint_generation,
            decode_endpoint_generation=pair.decode.endpoint_generation,
        )
        source = destination = pending_read = None
        response = None
        frames = None
        ready = safe_failure = decode_started = False

        async def confirm_source():
            nonlocal ready, safe_failure
            result = await self._prefill(pair.prefill_client, path, body)
            if result.key != key or result.request_fingerprint != context.request_fingerprint:
                raise RuntimeError("P returned a mismatched generation result")
            if result.state in ("NOT_STARTED", "FAILED") and result.error_code:
                if result.state == "FAILED" and result.retryable:
                    raise RuntimeError("a post-compute failure cannot be a no-compute retry")
                safe_failure = True
                raise SourceRejected(result)
            if result.state != "READY" or result.error_code:
                raise RuntimeError("P did not confirm a ready handoff")
            ready = True
            self.journal.append("PREFILL_READY", key)
            if on_ready is not None:
                await on_ready()

        async def supervised(read):
            # Observe P failure even if D is waiting for its next output.
            # A successful P reply independently releases the handoff budget.
            nonlocal pending_read
            pending_read = asyncio.ensure_future(read)
            done, _ = await asyncio.wait({source, pending_read}, return_when=asyncio.FIRST_COMPLETED)
            if source in done:
                await source
            result = await asyncio.shield(pending_read)
            pending_read = None
            return result

        try:
            # These are the only generation HTTP calls. Physical reservation,
            # normalized metadata and chunk/commit messages never visit Router.
            source = asyncio.create_task(confirm_source())
            destination = asyncio.create_task(pair.decode_client.open_generation(path, body))
            response = await supervised(destination)
            if response.status >= 400:
                # Prefer a bound no-compute rejection over a secondary D error.
                await source
                raise RuntimeError(f"D failed with HTTP {response.status}")
            # D sends headers only after accepting P's normalized continuation.
            # This proves one-time preparation succeeded. Source-pin retirement
            # may still be pending, but must not delay already available output.
            if signal_ready:
                yield None
            streaming = public_document(raw).get("stream", False)
            if streaming:
                if "text/event-stream" not in response.headers.get("content-type", ""):
                    raise ValueError("D did not return a standard SSE response")
                finished = used = False
                frames = response.frames()
                while True:
                    frame = await supervised(anext(frames))
                    value = sse_document(frame)
                    if value is None:
                        if not finished or not used:
                            raise EOFError("D completed SSE without finish and usage")
                        yield frame
                        break
                    if "error" in value:
                        # D can reject the seed just before P returns its bound
                        # safe-failure proof. Do not mistake that response race
                        # for uncertain native writes; timeout remains uncertain.
                        with suppress(TimeoutError):
                            await asyncio.wait_for(asyncio.shield(source), timeout=15)
                        raise DecodeStreamError(frame)
                    if value.get("id") != request_id:
                        raise ValueError("D returned SSE for another request")
                    decode_started = True
                    if value.get("usage") is not None:
                        if used:
                            raise ValueError("D duplicated terminal usage")
                        used = True
                    for choice in value.get("choices", ()):
                        if choice.get("finish_reason") is not None:
                            if finished:
                                raise ValueError("D duplicated finish")
                            finished = True
                        elif finished:
                            raise ValueError("D returned content after finish")
                    yield frame
            else:
                result = await supervised(response.result())
                value = msgspec.json.decode(result.body)
                if (value.get("id") != request_id or not value.get("choices")
                        or value["choices"][0].get("finish_reason") is None or not value.get("usage")):
                    raise ValueError("D returned an invalid public JSON result")
                yield result
            # Keep terminal publication, sockets and ownership tied to P's
            # matching READY even when D finishes first.
            await source
            self.journal.append("HANDOFF_COMPLETED", key)
        except BaseException as exc:
            cancelled = isinstance(exc, (GeneratorExit, asyncio.CancelledError))
            with anyio.CancelScope(shield=True):
                if cancelled and decode_started and not ready:
                    # D output can precede P's source-pin cleanup. Give that
                    # already-started confirmation a bounded chance to finish;
                    # client cancellation alone never proves native retirement.
                    with suppress(Exception):
                        await asyncio.wait_for(asyncio.shield(source), timeout=15)
                deterministic = safe_failure or (ready and cancelled)
                confirmed = await self._abort_pair(pair, key, type(exc).__name__, deterministic=deterministic)
                unsafe = not deterministic or not confirmed
                self.journal.append("RECOVERY_REQUIRED" if unsafe else "HANDOFF_FAILED",
                                    key, error_code=type(exc).__name__)
                if unsafe:
                    self.recovery.require(type(exc).__name__, (key,))
                if isinstance(exc, SourceRejected) and unsafe:
                    raise RuntimeError("no-compute abort was not confirmed") from exc
            raise
        finally:
            with anyio.CancelScope(shield=True):
                # Abort must precede draining a pending HTTP open/read. D may
                # need that abort to unblock the response thread itself.
                tasks = {task for task in (source, destination, pending_read) if task is not None}
                for task in tasks:
                    if task is not None and not task.done():
                        task.cancel()
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                if response is None and destination is not None and not destination.cancelled():
                    with suppress(Exception):
                        response = destination.result()
                if frames is not None:
                    await frames.aclose()
                if response is not None:
                    await response.aclose()

    @staticmethod
    async def _abort_pair(pair, key: HandoffKey, reason: str, *, deterministic: bool) -> bool:
        payload = AbortHandoffHTTP(key, reason[:128], deterministic)
        try:
            results = await asyncio.wait_for(asyncio.gather(
                pair.prefill_client.post("/internal/pd/abort", payload, HandoffStatus),
                pair.decode_client.post("/internal/pd/abort", payload, HandoffStatus),
                return_exceptions=True,
            ), timeout=15)
        except TimeoutError:
            return False
        return deterministic and all(
            isinstance(result, HandoffStatus) and result.key == key
            and result.state in {"ABORTED", "FAILED", "RELEASED", "COMPLETED"}
            for result in results
        )
