# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Connect replica-local agent waiting to the native model scheduler."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field, replace
from typing import Generic, Iterator

from pypto_serving.serving.agent_capacity import ReplicaKVLoad
from pypto_serving.serving.agent_identity import AgentIdentity
from pypto_serving.serving.agent_policy import ProgressTTLStrategy
from pypto_serving.serving.agent_policy_state import (
    AdmissionDisposition,
    DiagnosticValue,
    ProgramLifecycle,
    ProgramRef,
    ProgramState,
    ProgramStatus,
    ProgramView,
    SchedulerObservabilityConfig,
    SchedulingEvent,
    SchedulingEventKind,
    SchedulingSnapshot,
    SharedPrefixAttribution,
    StrategyDiagnostic,
    TokenObservationSource,
    TransitionController,
    TransitionKind,
    TransitionRequest,
    TransitionResult,
)
from pypto_serving.serving.agent_progress import ProgressTTLFactors
from pypto_serving.serving.agent_request_queue import AgentRequestQueue, QueuedRequest, RetainedRequestT
from pypto_serving.serving.agent_sessions import AgentSession, AgentSessionRegistry

logger = logging.getLogger(__name__)


@dataclass
class _Activity:
    prefix_anchor: float
    last_finished: float | None = None
    inflight: set[str] = field(default_factory=set)
    latest: str = ""
    latest_finished: bool = False
    terminal: bool = False

    def register(self, request_id: str) -> None:
        self.inflight.add(request_id)
        self.latest = request_id
        self.latest_finished = False
        self.terminal = False


@dataclass
class _Attempt:
    session: AgentSession
    prompt: int
    arrived: float
    previous_context: int
    return_gap: float
    tracks_prefix: bool
    queue_wait: float = 0.0
    generated: int = 0
    first_output: float | None = None
    cached_prefix_tokens: int | None = None
    shared_prefix_tokens: int | None = None
    hbm_cached_prefix_tokens: int | None = None
    hbm_hit_ref_zero_tokens: int | None = None

    def observe_prefix(
        self,
        cached: int,
        shared: int | None,
        local_hit: int | None,
        reclaimable_hit: int | None,
    ) -> None:
        if shared is not None:
            self.cached_prefix_tokens = cached
            self.shared_prefix_tokens = shared
        if local_hit is not None:
            self.hbm_cached_prefix_tokens = min(self.prompt, max(0, local_hit))
        if reclaimable_hit is not None:
            self.hbm_hit_ref_zero_tokens = min(self.hbm_cached_prefix_tokens or 0, max(0, reclaimable_hit))

    def usage(self, now: float, total: int) -> dict[str, DiagnosticValue]:
        prefix = self.session.tokens.shared_prefix_tokens
        cached = prefix if self.cached_prefix_tokens is None else self.cached_prefix_tokens
        return {
            "total_tokens": max(0, total),
            "previous_context_tokens": self.previous_context,
            "prompt_tokens": self.prompt,
            "completion_tokens": max(self.generated, total - self.prompt, 0),
            "request_latency_seconds": max(0.0, now - self.arrived),
            "decode_seconds": None if self.first_output is None else max(0.0, now - self.first_output),
            "inter_request_gap_seconds": self.return_gap,
            "scheduler_queue_seconds": self.queue_wait,
            "cached_prefix_tokens": cached,
            "shared_prefix_tokens": prefix
            if self.shared_prefix_tokens is None
            else self.shared_prefix_tokens,
            "hbm_cached_prefix_tokens": cached
            if self.hbm_cached_prefix_tokens is None
            else self.hbm_cached_prefix_tokens,
            "hbm_hit_ref_zero_tokens": self.hbm_hit_ref_zero_tokens or 0,
        }


class AgentReplicaScheduler(Generic[RetainedRequestT]):
    """Own the wait queue, request usage, and live sessions of one PyPTO replica."""

    def __init__(
        self,
        strategy: ProgressTTLStrategy,
        strategy_factors: ProgressTTLFactors,
        replica_load: ReplicaKVLoad,
        schedule_interval_seconds: float = 1.0,
        observability: SchedulerObservabilityConfig | None = None,
    ) -> None:
        if not math.isfinite(schedule_interval_seconds) or schedule_interval_seconds <= 0:
            raise ValueError("schedule interval must be positive and finite")
        self.strategy = strategy
        self.strategy_factors = strategy_factors
        self.replica_load = replica_load
        self._replica_id = replica_load.replica_id
        self.registry = AgentSessionRegistry()
        self.request_pool: AgentRequestQueue[RetainedRequestT] = AgentRequestQueue()
        self._request_bindings: dict[str, _Attempt] = {}
        self._activity: dict[ProgramRef, _Activity] = {}
        self._schedule_interval_seconds = schedule_interval_seconds
        self._periodic_at = 0.0
        self._dirty = False
        self._observability = observability or SchedulerObservabilityConfig()
        self._diagnostics_at = 0.0
        self._refresh_timers()

    @property
    def retained_request_count(self) -> int:
        return len(self.request_pool)

    @property
    def retained_request_ids(self) -> tuple[str, ...]:
        return tuple(item.request_id for item in self.request_pool.entries)

    def _prefix_waiters(self) -> Iterator[tuple[QueuedRequest[RetainedRequestT], float | None]]:
        for item in self.request_pool.entries:
            attempt = self._request_bindings.get(item.request_id)
            if attempt is None or not attempt.tracks_prefix:
                continue
            session = attempt.session
            if session.state is ProgramState.PAUSED and session.status is ProgramStatus.REASONING:
                yield item, session.tokens.shared_prefix_fresh_until_monotonic_s

    def prefix_cache_candidates(self) -> tuple[tuple[str, RetainedRequestT], ...]:
        now = time.monotonic()
        return tuple(
            (item.request_id, item.retained_request)
            for item, refresh_at in self._prefix_waiters()
            if refresh_at is None or refresh_at <= now
        )

    def next_prefix_cache_refresh_at(self) -> float | None:
        now = time.monotonic()
        return min(
            (deadline for _, deadline in self._prefix_waiters() if deadline is not None and deadline > now),
            default=None,
        )

    def _refresh_timers(self) -> None:
        self._deadline = self.strategy.next_check_at(self.strategy_factors)
        self._lightweight_at = self.strategy.next_lightweight_check_at(self.strategy_factors)

    def needs_schedule_cycle(self, now_monotonic_s: float) -> bool:
        if self._deadline is not None and self._deadline <= now_monotonic_s:
            return True
        return now_monotonic_s >= self._periodic_at and (
            self._dirty or self.registry.has_programs or bool(self.request_pool)
        )

    def run_due_lightweight_checks(self, now_monotonic_s: float) -> None:
        if self._lightweight_at is not None and self._lightweight_at <= now_monotonic_s:
            self.strategy.handle_lightweight_check(self.strategy_factors, now_monotonic_s)
            self._lightweight_at = self.strategy.next_lightweight_check_at(self.strategy_factors)

    def _set_load(self, load: ReplicaKVLoad) -> None:
        if load.rank != self.replica_load.rank:
            raise ValueError("a replica cannot adopt another rank's KV load")
        self.replica_load = load

    def _snapshot(self, *, except_waiter: ProgramRef | None = None) -> SchedulingSnapshot:
        load = self.replica_load
        views = self.registry.views()
        return SchedulingSnapshot(
            observed_at_monotonic_s=time.monotonic(),
            backend_id=self._replica_id,
            total_kv_tokens=load.capacity.total_tokens,
            programs=views,
            waiting_programs=self._waiting(views, except_waiter),
            native_used_kv_tokens=load.capacity.used_tokens,
            native_waiting_kv_tokens=load.waiting_tokens,
            uncached_free_head_kv_tokens=load.uncached_free_head_tokens,
            request_pool_waiting_requests=self.request_pool.waiting_count,
        )

    @staticmethod
    def _waiting(views: tuple[ProgramView, ...], exclude: ProgramRef | None = None) -> tuple[ProgramRef, ...]:
        return tuple(view.ref for view in views if view.state is ProgramState.PAUSED and view.ref != exclude)

    def _after_placement(
        self,
        snapshot: SchedulingSnapshot,
        controller: TransitionController,
    ) -> SchedulingSnapshot:
        changes = controller.take_changed_sessions()
        if not changes:
            return snapshot
        views = []
        for view in snapshot.programs:
            if view.ref in changes:
                live = self.registry.get(view.ref)
                if live is None:
                    continue
                view = replace(
                    view, state=live.state, backend_id=live.backend_id, marked_for_pause=live.marked_for_pause
                )
            views.append(view)
        return replace(snapshot, programs=tuple(views), waiting_programs=self._waiting(tuple(views)))

    def _publish(
        self,
        session: AgentSession,
        kind: SchedulingEventKind,
        reason: str,
        *,
        before: ProgramState | None = None,
        prior_status: ProgramStatus | None = None,
        **facts: DiagnosticValue,
    ) -> None:
        event = SchedulingEvent(
            occurred_at_monotonic_s=time.monotonic(),
            program=session.ref,
            kind=kind,
            reason=reason,
            fields=facts,
            previous_state=before if before is not None else session.state,
            current_state=session.state,
            previous_status=prior_status if prior_status is not None else session.status,
            current_status=session.status,
        )
        self.strategy.handle_scheduling_event(self.strategy_factors, event)
        if self._observability.enabled:
            logger.info(
                "AgentInfer event kind=%s reason=%s program=%s facts=%s",
                kind.value,
                reason,
                session.ref,
                facts,
            )
            for diagnostic in self.strategy.event_diagnostics(event, self.strategy_factors):
                self._write_diagnostic(diagnostic)
        self._refresh_timers()

    @staticmethod
    def _write_diagnostic(item: StrategyDiagnostic) -> None:
        logger.info(
            "AgentInfer diagnostic name=%s reason=%s %s",
            item.name,
            item.reason,
            " ".join(f"{key}={value}" for key, value in item.fields),
        )

    def _transition(self, action: TransitionRequest) -> TransitionResult:
        session = self.registry.get(action.program)
        if session is None:
            return TransitionResult(action.kind, action.program, False, "stale_program")
        before = session.state
        kind = action.kind
        rejected = None
        if kind in (TransitionKind.ADMIT, TransitionKind.RESUME, TransitionKind.QUEUE):
            if before is not ProgramState.PAUSED:
                rejected = f"{kind.value}_requires_paused"
            elif kind is not TransitionKind.QUEUE:
                session.state = ProgramState.ACTIVE
                session.backend_id = action.backend_id
                session.marked_for_pause = False
                for item in self.request_pool.entries:
                    if item.program == session.ref:
                        self.request_pool.grant(item.request_id, self._replica_id)
        elif kind is TransitionKind.PAUSE:
            if before is not ProgramState.ACTIVE or session.status is not ProgramStatus.ACTING:
                rejected = "pause_requires_active_acting"
            else:
                session.state = ProgramState.PAUSED
                session.backend_id = None
                session.marked_for_pause = False
                activity = self._activity[session.ref]
                began = time.monotonic()
                deadline = session.tokens.shared_prefix_fresh_until_monotonic_s
                if deadline is not None:
                    session.tokens = replace(
                        session.tokens,
                        shared_prefix_fresh_until_monotonic_s=(
                            began + max(0.0, deadline - activity.prefix_anchor)
                        ),
                    )
                activity.prefix_anchor = began
        elif kind is TransitionKind.MARK_FOR_PAUSE:
            if before is not ProgramState.ACTIVE or session.status is not ProgramStatus.REASONING:
                rejected = "mark_requires_active_reasoning"
            elif session.marked_for_pause:
                rejected = "already_marked"
            else:
                session.marked_for_pause = True
        elif kind is TransitionKind.RELEASE:
            if self._activity[session.ref].inflight:
                rejected = "release_requires_idle"
            else:
                self.registry.release(session.ref)
                del self._activity[session.ref]
        else:
            rejected = "unsupported_transition"
        if rejected is not None:
            return TransitionResult(kind, session.ref, False, rejected)
        self._publish(
            session,
            kind.event,
            action.reason,
            before=before,
            backend_id=action.backend_id,
            marked_for_pause=session.marked_for_pause,
        )
        return TransitionResult(kind, session.ref, True, action.reason)

    def on_request_arrival(
        self,
        request_id: str,
        metadata: AgentIdentity,
        prompt_tokens: int,
        replica_load: ReplicaKVLoad,
        retained_request: RetainedRequestT,
        *,
        native_fallback: bool = False,
    ) -> bool:
        if request_id in self._request_bindings:
            raise ValueError(f"duplicate AgentInfer request: {request_id}")
        self._dirty = True
        self._set_load(replica_load)
        session, created = self.registry.materialize(metadata)
        prior_status = self.registry.mark_request_started(session.ref)
        now = time.monotonic()
        if created:
            self._activity[session.ref] = _Activity(prefix_anchor=now)
        activity = self._activity[session.ref]
        context = session.tokens.estimated_context_tokens
        prompt = max(0, prompt_tokens)
        self._request_bindings[request_id] = _Attempt(
            session,
            prompt,
            now,
            context,
            max(0.0, now - activity.last_finished) if activity.last_finished is not None else 0.0,
            session.state is ProgramState.PAUSED,
        )
        activity.register(request_id)
        session.tokens = replace(session.tokens, estimated_context_tokens=max(context, prompt))
        self.request_pool.append(QueuedRequest(request_id, session.ref, now, retained_request))
        if prior_status is not session.status:
            self._publish(
                session,
                SchedulingEventKind.PROGRAM_STATUS_CHANGED,
                "request_started",
                prior_status=prior_status,
            )
        if native_fallback:
            self.admit_native_fallback(request_id, reason="native_queue_full")
        else:
            result = self.strategy.handle_admission(
                self._snapshot(except_waiter=session.ref if created else None),
                self.strategy_factors,
                TransitionController(self._transition),
                session.ref,
            )
            # The policy may grant privilege after the transition refreshes the timers.
            self._lightweight_at = self.strategy.next_lightweight_check_at(self.strategy_factors)
            if self._observability.enabled:
                logger.info("AgentInfer admission request=%s result=%s", request_id, result)
            if result.disposition is AdmissionDisposition.QUEUED:
                return False
        self.request_pool.grant(request_id, self._replica_id)
        taken = self.request_pool.take(request_id)
        if taken is None:
            raise RuntimeError("admitted request has no queue entry")
        self._note_dispatched(taken[0])
        return True

    def admit_native_fallback(self, request_id: str, *, reason: str) -> bool:
        item = self.request_pool.waiting(request_id)
        if item is None:
            return False
        session = self.registry.require(item.program)
        if session.state is ProgramState.PAUSED:
            result = self._transition(
                TransitionRequest(TransitionKind.RESUME, session.ref, reason, self._replica_id)
            )
            if not result.applied:
                raise RuntimeError(f"cannot release {request_id} to native queue: {result.reason}")
        else:
            self.request_pool.grant(request_id, self._replica_id)
        self._dirty = True
        return True

    def schedule_cycle(self, replica_load: ReplicaKVLoad) -> None:
        self._set_load(replica_load)
        now = time.monotonic()
        if not self.needs_schedule_cycle(now):
            return
        snapshots = self._snapshot()
        if self._observability.enabled and self._diagnostics_at <= now:
            for item in self.strategy.diagnostics(snapshots, self.strategy_factors):
                self._write_diagnostic(item)
            self._diagnostics_at = now + self._observability.log_interval_seconds
        changes = TransitionController(self._transition)
        self.strategy.schedule_resume(snapshots, self.strategy_factors, changes)
        after_resume = self._after_placement(snapshots, changes)
        self.strategy.repair_capacity(after_resume, self.strategy_factors, changes)
        if self._deadline is not None and self._deadline <= now:
            self.strategy.handle_scheduled_check(
                self._after_placement(after_resume, changes), self.strategy_factors, changes
            )
        self._periodic_at = now + self._schedule_interval_seconds
        self._dirty = False
        self._refresh_timers()

    def _note_dispatched(self, entry: QueuedRequest[RetainedRequestT]) -> None:
        attempt = self._request_bindings.get(entry.request_id)
        if attempt is not None:
            attempt.queue_wait = max(0.0, time.monotonic() - attempt.arrived)

    def consume_admitted_requests(self) -> tuple[tuple[QueuedRequest[RetainedRequestT], str], ...]:
        batch = self.request_pool.drain()
        for entry, _ in batch:
            self._note_dispatched(entry)
        return batch

    def on_stream_output(self, request_id: str, total_output_tokens: int) -> None:
        attempt = self._request_bindings.get(request_id)
        if attempt is not None and total_output_tokens > attempt.generated:
            attempt.generated = total_output_tokens
            if attempt.first_output is None:
                attempt.first_output = time.monotonic()

    def on_prefix_cache_observation(
        self,
        request_id: str,
        cached_prefix_tokens: int,
        shared_prefix_tokens: int | None = None,
        hbm_cached_prefix_tokens: int | None = None,
        hbm_hit_ref_zero_tokens: int | None = None,
    ) -> None:
        attempt = self._request_bindings.get(request_id)
        if attempt is None:
            return
        cached = min(attempt.prompt, max(0, cached_prefix_tokens))
        shared = None if shared_prefix_tokens is None else min(cached, max(0, shared_prefix_tokens))
        attempt.observe_prefix(cached, shared, hbm_cached_prefix_tokens, hbm_hit_ref_zero_tokens)
        if not attempt.tracks_prefix:
            return
        session = attempt.session
        refresh_at = session.tokens.shared_prefix_fresh_until_monotonic_s
        if refresh_at is not None and refresh_at > time.monotonic():
            return
        snapshot = self._snapshot()
        view = snapshot.program(session.ref)
        if view is None:
            return
        keep = self.strategy.shared_prefix_freshness_seconds(snapshot, self.strategy_factors, view)
        anchor = self._activity[session.ref].prefix_anchor
        session.tokens = replace(
            session.tokens,
            shared_prefix_tokens=cached if shared is None else shared,
            shared_prefix_fresh_until_monotonic_s=None if keep is None else anchor + max(0.0, keep),
            shared_prefix_attribution=(
                SharedPrefixAttribution.PARTIAL if shared is None else SharedPrefixAttribution.EXACT
            ),
            source=TokenObservationSource.MIXED,
        )

    def on_request_completion(self, request_id: str, total_tokens: int) -> None:
        attempt = self._request_bindings.pop(request_id, None)
        if attempt is None:
            return
        self._dirty = True
        session = attempt.session
        activity = self._activity[session.ref]
        activity.inflight.remove(request_id)
        now = time.monotonic()
        activity.last_finished = now
        if activity.latest == request_id:
            activity.latest_finished = True
        prior_status = self.registry.record_completion(session.ref, total_tokens=total_tokens)
        if activity.inflight:
            session.status = ProgramStatus.REASONING
        active = sum(view.state is ProgramState.ACTIVE for view in self.registry.views())
        facts = attempt.usage(now, total_tokens)
        facts.update(active_programs=active, waiting_programs=len(self._activity) - active)
        self._publish(
            session,
            SchedulingEventKind.REQUEST_FINISHED,
            "request_completed",
            prior_status=prior_status,
            **facts,
        )
        if self._release_terminal(session, activity):
            return
        if session.marked_for_pause and session.status is ProgramStatus.ACTING:
            self._transition(TransitionRequest(TransitionKind.PAUSE, session.ref, "marked_pause_completion"))
            return
        self.strategy.handle_request_completion(
            self._snapshot(), self.strategy_factors, TransitionController(self._transition), session.ref
        )

    def _release_terminal(self, session: AgentSession, activity: _Activity) -> bool:
        if activity.inflight or not activity.terminal or session.status is not ProgramStatus.ACTING:
            return False
        result = self._transition(
            TransitionRequest(TransitionKind.RELEASE, session.ref, "api_terminal_response")
        )
        self._dirty |= result.applied
        return result.applied

    def on_response_completion(self, program_id: str, request_id: str, lifecycle: ProgramLifecycle) -> bool:
        if lifecycle is not ProgramLifecycle.TERMINAL:
            return False
        ref = self.registry.current_ref(program_id)
        if ref is None:
            return False
        activity = self._activity[ref]
        if activity.latest != request_id or not activity.latest_finished:
            return False
        activity.terminal = True
        return self._release_terminal(self.registry.require(ref), activity)

    def cancel_request(self, request_id: str) -> QueuedRequest[RetainedRequestT] | None:
        queued = self.request_pool.cancel(request_id)
        attempt = self._request_bindings.pop(request_id, None)
        if queued is not None or attempt is not None:
            self._dirty = True
        if attempt is None:
            return queued
        session = attempt.session
        activity = self._activity[session.ref]
        activity.inflight.discard(request_id)
        if not activity.inflight:
            prior = session.status
            session.status = ProgramStatus.ACTING
            if prior is not session.status:
                self._publish(
                    session,
                    SchedulingEventKind.PROGRAM_STATUS_CHANGED,
                    "request_cancelled",
                    prior_status=prior,
                )
            self._transition(TransitionRequest(TransitionKind.RELEASE, session.ref, "request_cancelled"))
        return queued
