# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Choose when to issue, pause, or release work for an agent session."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import Callable

from pypto_serving.serving.agentinfer.agent_policy_config import (
    ProgressTTLConfig,
)
from pypto_serving.serving.agentinfer.agent_policy_state import (
    AdmissionDisposition,
    AdmissionOutcome,
    ProgramRef,
    ProgramState,
    ProgramStatus,
    ProgramView,
    SchedulingEvent,
    SchedulingEventKind,
    SchedulingSnapshot,
    TransitionController,
    TransitionResult,
)
from pypto_serving.serving.agentinfer.agent_progress import ProgressTTLFactors, ProgressTTLProgramFactors
from pypto_serving.serving.agentinfer.agent_recovery import RecoveryBudget, recovery_order
from pypto_serving.serving.agentinfer.agent_ttl import AgentTTLStatistics
from pypto_serving.serving.agentinfer.agent_workload import AgentModelCosts, AgentRound, prefix_refresh_interval

logger = logging.getLogger(__name__)

# Bound synchronous subset search work on the engine's event loop.
_CAPACITY_FIT_SEARCH_STEPS = 4096


@dataclass(frozen=True)
class _ForceResumeTimeoutEstimate:
    """Frozen work-ahead estimate used to arm one force-resume deadline."""

    timeout_seconds: float
    active_remaining_rounds: float
    pool_remaining_rounds: float
    request_throughput_per_second: float


class ProgressTTLStrategy:
    """Concrete Progress-TTL admission, resume, capacity-repair, and deadline policy."""

    def __init__(
        self,
        config: ProgressTTLConfig,
        *,
        capacity_cost: Callable[[int], int] | None = None,
    ) -> None:
        self.config = config
        self._costs = AgentModelCosts(
            prefill_fixed=config.ttl_prefill_model_intercept_seconds,
            prefill_per_1000=config.ttl_prefill_model_linear_seconds_per_1k_tokens,
            prefill_per_1000_squared=config.ttl_prefill_model_quadratic_seconds_per_1k_tokens_squared,
            prefill_interference=1.0 - config.ttl_decode_throughput_alpha,
            decode_fixed=config.decode_step_fixed_seconds,
            decode_per_session=config.decode_step_seconds_per_request,
            decode_per_token=config.decode_step_seconds_per_context_token,
        )
        self._capacity_cost = capacity_cost or (lambda tokens: tokens)

    def handle_scheduling_event(self, factors: ProgressTTLFactors, event: SchedulingEvent) -> None:
        match event.kind:
            case SchedulingEventKind.REQUEST_FINISHED:
                self.on_request_finished(factors, event)
            case SchedulingEventKind.REQUEST_PENDING:
                self.on_request_pending(factors, event)
            case SchedulingEventKind.REQUEST_ADMITTED:
                self.on_program_admitted(factors, event)
            case SchedulingEventKind.PROGRAM_STATUS_CHANGED:
                self.on_program_status_changed(factors, event)
            case SchedulingEventKind.PROGRAM_PAUSED:
                self.on_program_paused(factors, event)
            case SchedulingEventKind.PROGRAM_MARKED_FOR_PAUSE:
                self.on_program_marked_for_pause(factors, event)
            case SchedulingEventKind.PROGRAM_RESUMED:
                self.on_program_resumed(factors, event)
            case SchedulingEventKind.PROGRAM_RELEASED:
                self.on_program_released(factors, event)

    def shared_prefix_freshness_seconds(
        self,
        snapshot: SchedulingSnapshot,
        strategy_factors: ProgressTTLFactors,
        program: ProgramView,
    ) -> float:
        """Estimate the lifetime of one shared-prefix observation from KV-pool turnover."""
        running = sum(item.status is ProgramStatus.REASONING for item in self._active_programs(snapshot))
        return prefix_refresh_interval(
            strategy_factors.global_factors,
            running=running,
            pool_tokens=snapshot.total_kv_tokens,
            warmup=self.config.shared_prefix_freshness_warmup_seconds,
            turnovers=self.config.shared_prefix_freshness_kv_turnovers,
        )

    def handle_admission(
        self,
        snapshot: SchedulingSnapshot,
        factors: ProgressTTLFactors,
        changes: TransitionController,
        candidate: ProgramRef,
    ) -> AdmissionOutcome:
        session = snapshot.program(candidate)
        if session is None:
            raise ValueError("missing session in admission snapshot")
        self._reconcile_privileges(snapshot, factors)
        needed = self._required_tokens(session)
        active = tuple(self._active_programs(snapshot))
        if session in active:
            return AdmissionOutcome(AdmissionDisposition.ADMITTED, "already_active", required_tokens=needed)
        donor = self._privileged_relationship_source(snapshot, factors, session)
        if donor is not None:
            return self._admit_by_privilege_handoff(snapshot, factors, changes, session, donor, needed)

        headroom = self._remaining_tokens(snapshot, self.config.resume_capacity_ratio)
        reserve = self._continuous_growth_reserve_tokens(active, factors, candidate=session)
        deficit = needed + reserve - headroom if headroom is not None else 0
        returning = candidate in snapshot.waiting_programs
        reason, admit = self._admission_decision(
            snapshot, factors, session, active, needed, headroom, deficit, returning
        )
        if not admit:
            return self._queue(
                snapshot,
                factors,
                changes,
                candidate,
                reason,
                needed,
                reserve,
                deficit if reason == "insufficient_capacity" else 0,
            )
        promote = not snapshot.waiting_programs and self._can_promote(snapshot, factors, session)
        apply = changes.resume if returning else changes.admit
        self._require_applied(apply(candidate, reason=reason, backend_id=snapshot.backend_id))
        if promote:
            self._set_privileged(
                factors,
                candidate,
                True,
                reason="direct_admission_promote"
                if reason == "direct_capacity"
                else "batch_gain_admission_promote",
                now_monotonic_s=snapshot.observed_at_monotonic_s,
            )
        return AdmissionOutcome(
            AdmissionDisposition.ADMITTED, reason, needed, reserve, transitioned_programs=(candidate,)
        )

    def _admission_decision(
        self,
        snapshot: SchedulingSnapshot,
        factors: ProgressTTLFactors,
        session: ProgramView,
        active: tuple[ProgramView, ...],
        needed: int,
        headroom: int | None,
        deficit: int,
        returning: bool,
    ) -> tuple[str, bool]:
        if returning:
            fits = headroom is not None and deficit <= 0
            if snapshot.request_pool_waiting_requests == 1 and fits:
                return "progress_ttl_arrival_resume", True
            return "already_waiting", False
        if snapshot.waiting_programs:
            return "waiting_queue_precedence", False
        if headroom is None:
            return "capacity_unknown", False
        if deficit <= 0:
            return "direct_capacity", True
        if needed <= headroom and self._batch_gain_covers_recovery(active, factors, session, headroom):
            return "batch_gain_over_recovery_cost", True
        return "insufficient_capacity", False

    def schedule_resume(
        self,
        snapshot: SchedulingSnapshot,
        strategy_factors: ProgressTTLFactors,
        transitions: TransitionController,
    ) -> None:
        """Resume paused reasoning Programs under the current capacity policy."""
        self._reconcile_privileges(snapshot, strategy_factors)
        remaining = self._remaining_tokens(snapshot, self.config.resume_capacity_ratio)
        active = self._active_programs(snapshot)
        waiting = sorted(
            self._waiting_for_resume(snapshot),
            key=lambda program: self._resume_key(program, strategy_factors, snapshot.observed_at_monotonic_s),
        )
        starts = [
            (when, program.ref.program_id, program.ref.generation, program.ref)
            for program in waiting
            if (
                when := self._program_factors(
                    strategy_factors, program.ref
                ).request_wait_started_at_monotonic_s
            )
            is not None
        ]
        oldest = min(starts)[-1] if starts else None
        reserved = (snapshot.total_kv_tokens or 0) - int(
            (snapshot.total_kv_tokens or 0) * self.config.resume_capacity_ratio
        )
        budget = RecoveryBudget(
            reserved_tokens=reserved,
            relaxed_after=self.config.resume_wait_relax_seconds,
            relaxed_growth_fraction=self.config.resume_wait_growth_reserve_fraction,
            relaxed_headroom=self.config.resume_wait_extra_headroom_tokens,
            keep_cached_blocks=self.config.resume_wait_preserve_cached_blocks,
            uncached_tokens=(
                None
                if snapshot.uncached_free_head_kv_tokens is None
                else snapshot.uncached_free_head_kv_tokens - (snapshot.native_waiting_kv_tokens or 0)
            ),
        )
        now = snapshot.observed_at_monotonic_s
        for program in waiting:
            factors = self._program_factors(strategy_factors, program.ref)
            required = self._required_tokens(program)
            if self._force_resume_due(program, strategy_factors, now):
                reason = "progress_ttl_force_resume_timeout"
                wait_started = factors.request_wait_started_at_monotonic_s
                waited = now - wait_started if wait_started is not None else 0.0
                physical_remaining = remaining + reserved if remaining is not None else None
                logger.warning(
                    "Progress-TTL force resume program=%s generation=%d waited_seconds=%.1f "
                    "required_tokens=%d remaining_tokens=%s physical_remaining_tokens=%s",
                    program.ref.program_id,
                    program.ref.generation,
                    waited,
                    required,
                    remaining,
                    physical_remaining,
                )
            elif factors.is_privileged:
                reason = "progress_ttl_privileged_resume"
            elif remaining is None:
                continue
            else:
                reason, remaining = self._resume_with_budget(
                    program,
                    snapshot,
                    strategy_factors,
                    transitions,
                    active,
                    remaining,
                    required,
                    budget if program.ref == oldest else None,
                )
                if reason is None:
                    continue
            self._require_applied(
                transitions.resume(program.ref, reason=reason, backend_id=snapshot.backend_id)
            )
            active.append(program)
            if remaining is not None:
                remaining -= required

    def _resume_with_budget(
        self,
        program: ProgramView,
        snapshot: SchedulingSnapshot,
        factors: ProgressTTLFactors,
        transitions: TransitionController,
        active: list[ProgramView],
        remaining: int,
        required: int,
        relaxed: RecoveryBudget | None,
    ) -> tuple[str | None, int]:
        active_views = tuple(active)
        growth = self._continuous_growth_reserve_tokens(active_views, factors, candidate=program)
        if relaxed is not None:
            started = self._program_factors(factors, program.ref).request_wait_started_at_monotonic_s
            if relaxed.allows_relaxed(
                waited=snapshot.observed_at_monotonic_s - started if started is not None else None,
                needed=required,
                growth=growth,
                remaining=remaining,
                acting_tokens=lambda: sum(
                    self._required_tokens(item) for item in active if item.status is ProgramStatus.ACTING
                ),
            ):
                logger.info(
                    "Progress-TTL wait capacity relax program=%s generation=%d required_tokens=%d "
                    "effective_remaining_tokens=%d physical_remaining_tokens=%d growth_reserve_tokens=%d",
                    program.ref.program_id,
                    program.ref.generation,
                    required,
                    remaining,
                    remaining + relaxed.reserved_tokens,
                    growth,
                )
                return "progress_ttl_wait_capacity_relax", remaining
        deficit = required + growth - remaining
        if deficit <= 0:
            return "progress_ttl_resume", remaining
        if required <= remaining and self._batch_gain_covers_recovery(
            active_views, factors, program, remaining
        ):
            return "progress_ttl_resume_batch_gain", remaining
        if not self.config.resume_reclaim_acting_programs:
            return None, remaining

        layers: tuple[list[ProgramView], list[ProgramView]] = ([], [])
        for session in active:
            if session.status is not ProgramStatus.ACTING:
                continue
            state = self._program_factors(factors, session.ref)
            if not state.is_evictable_after_min_rounds or state.is_privileged:
                continue
            layer = int(state.segment_served_rounds < self.config.target_max_segment_rounds)
            layers[layer].append(session)
        selected = self._select_layered_capacity_fit(layers, deficit)
        if selected is None:
            return None, remaining
        for session in selected:
            self._require_applied(transitions.pause(session.ref, reason="progress_ttl_resume_reclaim"))
            active.remove(session)
            remaining += self._required_tokens(session)
        growth = self._continuous_growth_reserve_tokens(tuple(active), factors, candidate=program)
        return ("progress_ttl_resume" if required + growth <= remaining else None), remaining

    def repair_capacity(
        self,
        snapshot: SchedulingSnapshot,
        state: ProgressTTLFactors,
        actions: TransitionController,
    ) -> None:
        self._reconcile_privileges(snapshot, state)
        self._promote_progress_privileges(snapshot, state)
        available = self._remaining_tokens(snapshot, self.config.pause_capacity_ratio)
        if available is None:
            return
        decode_budget = math.ceil(max(0, state.global_factors.avg_completion_tokens))
        unmarked = []
        for session in self._active_programs(snapshot):
            reasoning = session.status is ProgramStatus.REASONING
            if reasoning:
                available -= decode_budget
            if not session.marked_for_pause:
                unmarked.append(session)
            elif reasoning:
                available += self._required_tokens(session) + decode_budget
        if available >= 0:
            return
        ordinary = [item for item in unmarked if not self._program_factors(state, item.ref).is_privileged]
        ordinary.sort(
            key=lambda item: (
                -self._pause_priority_score(item, state, snapshot.observed_at_monotonic_s),
                -self._required_tokens(item),
                item.ref,
            )
        )
        for victim in ordinary:
            available += self._yield(victim, actions, decode_budget)
            if available >= 0:
                return
        while available < 0:
            victim = self._demote_privileged_for_capacity(unmarked, state)
            if victim is None:
                return
            available += self._yield(victim, actions, decode_budget)

    def _yield(self, session: ProgramView, actions: TransitionController, expected_decode: int) -> int:
        idle = session.status is ProgramStatus.ACTING
        request = actions.pause if idle else actions.mark_for_pause
        self._require_applied(request(session.ref, reason="progress_ttl_capacity_repair"))
        return self._required_tokens(session) + (0 if idle else expected_decode)

    def handle_request_completion(
        self,
        snapshot: SchedulingSnapshot,
        state: ProgressTTLFactors,
        actions: TransitionController,
        completed: ProgramRef,
    ) -> None:
        session = snapshot.program(completed)
        if session is None or session.status is not ProgramStatus.ACTING:
            return
        if session not in self._active_programs(snapshot):
            return
        stats = self._program_factors(state, completed)
        if stats.is_privileged or stats.segment_served_rounds < self.config.target_max_segment_rounds:
            return
        contender = min(
            self._waiting_for_resume(snapshot),
            default=None,
            key=lambda item: self._resume_key(item, state, snapshot.observed_at_monotonic_s),
        )
        already_yielding = sum(
            self._required_tokens(item) for item in self._active_programs(snapshot) if item.marked_for_pause
        )
        if contender is not None and self._required_tokens(contender) > already_yielding:
            self._require_applied(actions.pause(completed, reason="progress_ttl_max_round_queue_pressure"))

    def next_check_at(self, state: ProgressTTLFactors) -> float | None:
        checks = []
        for _, timing in state.program_factors:
            if timing.release_deadline_monotonic_s is not None:
                checks.append(timing.release_deadline_monotonic_s)
            if timing.acting_since_monotonic_s is not None:
                checks.append(timing.acting_since_monotonic_s + self.config.ttl_max_seconds)
            if timing.ttl_deadline_monotonic_s is not None:
                checks.append(timing.ttl_deadline_monotonic_s)
        return min(checks, default=None)

    def next_lightweight_check_at(self, state: ProgressTTLFactors) -> float | None:
        expiries = (
            timing.privilege_deadline_monotonic_s
            for _, timing in state.program_factors
            if timing.is_privileged
        )
        return min((at for at in expiries if at is not None), default=None)

    def handle_lightweight_check(self, state: ProgressTTLFactors, now_monotonic_s: float) -> None:
        for ref, lease in state.program_factors:
            at = lease.privilege_deadline_monotonic_s
            if lease.is_privileged and at is not None and at <= now_monotonic_s:
                self._set_privileged(state, ref, False, reason="privilege_ttl_expired")
                state.update(ref, privilege_ttl_expired=True)

    def handle_scheduled_check(
        self,
        snapshot: SchedulingSnapshot,
        state: ProgressTTLFactors,
        actions: TransitionController,
    ) -> None:
        self._reconcile_privileges(snapshot, state)
        now = snapshot.observed_at_monotonic_s
        successors = set()
        for idle in (session for session in snapshot.programs if session.status is ProgramStatus.ACTING):
            if idle.state is not ProgramState.ACTIVE or idle.ref in successors:
                continue
            reservation = self._program_factors(state, idle.ref)
            if not self._acting_ttl_due(idle, reservation, state, now):
                continue
            successor = self._expire_reservation(
                snapshot, state, actions, idle, reservation.is_privileged
            )
            if successor is not None:
                successors.add(successor)
        for waiting in snapshot.programs:
            if waiting.state is ProgramState.PAUSED and waiting.status is ProgramStatus.ACTING:
                release_at = self._program_factors(state, waiting.ref).release_deadline_monotonic_s
                if release_at is not None and release_at <= now:
                    self._require_applied(
                        actions.release(waiting.ref, reason="progress_ttl_paused_retention_expired")
                    )

    def _acting_ttl_due(
        self,
        session: ProgramView,
        timing: ProgressTTLProgramFactors,
        state: ProgressTTLFactors,
        now: float,
    ) -> bool:
        started, armed = timing.acting_since_monotonic_s, timing.ttl_deadline_monotonic_s
        if started is None or (armed is not None and armed <= now):
            return True
        prompt = timing.last_request_prompt_tokens or session.tokens.estimated_context_tokens
        return started + self._ttl_seconds(state, prompt, timing.segment_share_tokens) <= now

    def _expire_reservation(
        self,
        snapshot: SchedulingSnapshot,
        state: ProgressTTLFactors,
        actions: TransitionController,
        expired: ProgramView,
        owns_lease: bool,
    ) -> ProgramRef | None:
        successor = self._same_task_ttl_privilege_target(snapshot, expired) if owns_lease else None
        self._require_applied(actions.pause(expired.ref, reason="progress_ttl_expired"))
        if successor is None:
            return None
        self._set_privileged(state, expired.ref, False, reason="ttl_handoff_source")
        if successor.state is ProgramState.PAUSED:
            self._require_applied(
                actions.resume(
                    successor.ref, backend_id=snapshot.backend_id, reason="progress_ttl_privilege_ttl_handoff"
                )
            )
        self._set_privileged(
            state, successor.ref, True, reason="ttl_handoff", now_monotonic_s=snapshot.observed_at_monotonic_s
        )
        return successor.ref

    def on_request_pending(self, state: ProgressTTLFactors, event: SchedulingEvent) -> None:
        prior = self._program_factors(state, event.program)
        now = event.occurred_at_monotonic_s
        updates = dict(
            acting_since_monotonic_s=None, ttl_deadline_monotonic_s=None, release_deadline_monotonic_s=None
        )
        if prior.wait_started_at_monotonic_s is None:
            updates["wait_started_at_monotonic_s"] = now
        if prior.request_wait_started_at_monotonic_s is None:
            timeout = self.config.force_resume_timeout_max_seconds
            updates.update(
                request_wait_started_at_monotonic_s=now,
                force_resume_timeout_seconds=timeout,
                force_resume_deadline_monotonic_s=now + timeout,
            )
        state.update(event.program, **updates)

    def on_program_admitted(self, state: ProgressTTLFactors, event: SchedulingEvent) -> None:
        self._reset_segment(state, event)

    def on_program_resumed(self, state: ProgressTTLFactors, event: SchedulingEvent) -> None:
        self._reset_segment(
            state, event, last_resume_at_monotonic_s=event.occurred_at_monotonic_s, resume_reason=event.reason
        )

    def on_program_status_changed(self, state: ProgressTTLFactors, event: SchedulingEvent) -> None:
        if event.current_status is ProgramStatus.REASONING:
            self._program_factors(state, event.program)
            state.update(event.program, acting_since_monotonic_s=None, ttl_deadline_monotonic_s=None)

    def on_request_finished(self, strategy_factors: ProgressTTLFactors, event: SchedulingEvent) -> None:
        if event.program is None:
            return
        sample = AgentRound.from_metrics(event.field, event.occurred_at_monotonic_s)
        prior = self._program_factors(strategy_factors, event.program)
        count = int(sample.maintains_context)
        streak = prior.rounds_since_ttl_pause + count
        sample.collect(
            strategy_factors.global_factors, first=prior.segment_first_request_pending, uninterrupted=streak
        )
        if prior.previous_ttl_seconds is not None and sample.gap > 0:
            self._track_return(strategy_factors.global_factors, sample, prior.previous_ttl_seconds)
        guess = self._ttl_seconds(strategy_factors, sample.prompt, sample.shared)
        rounds = prior.segment_served_rounds + count
        idle = sample.finished if event.current_status is ProgramStatus.ACTING else None
        strategy_factors.set_program_factors(
            event.program,
            replace(
                prior,
                segment_served_rounds=rounds,
                segment_prompt_tokens=prior.segment_prompt_tokens + sample.prompt,
                segment_completion_tokens=prior.segment_completion_tokens + sample.generated,
                segment_first_request_pending=False,
                last_request_prompt_tokens=sample.prompt,
                last_request_context_tokens=sample.total,
                lifetime_generated_tokens=prior.lifetime_generated_tokens + sample.generated,
                segment_share_tokens=sample.shared,
                rounds_since_ttl_pause=streak,
                previous_ttl_seconds=guess,
                ttl_expiry_observed=False,
                is_evictable_after_min_rounds=rounds >= self._protected_min_segment_rounds(strategy_factors),
                acting_since_monotonic_s=idle,
                ttl_deadline_monotonic_s=None if idle is None else idle + guess,
                release_deadline_monotonic_s=None,
                last_inter_request_gap_seconds=sample.gap,
                last_request_finished_at_monotonic_s=sample.finished,
            ),
        )

    def _track_return(self, stats: AgentTTLStatistics, sample: AgentRound, previous_ttl: float) -> None:
        ready = stats.continuity_window_complete
        stats.observe_continuity(
            interval_seconds=sample.gap,
            cache_miss_impact_seconds=self._costs.reload_cost(sample.prompt, sample.shared),
            assigned_ttl_seconds=previous_ttl,
        )
        if not ready and stats.continuity_window_complete:
            stats.recalibrate_continuity_window(
                minimum_seconds=self.config.ttl_min_seconds,
                maximum_seconds=self.config.ttl_max_seconds,
                impact_ratio=self.config.ttl_max_cache_miss_impact_ratio,
            )

    def on_program_paused(self, state: ProgressTTLFactors, event: SchedulingEvent) -> None:
        record = self._program_factors(state, event.program)
        now = event.occurred_at_monotonic_s
        expiry = event.reason == "progress_ttl_expired"
        changes = dict(
            last_segment_served_rounds=record.segment_served_rounds,
            last_segment_prompt_tokens=record.segment_prompt_tokens,
            last_segment_completion_tokens=record.segment_completion_tokens,
            segment_first_request_pending=True,
            segment_started_at_monotonic_s=None,
            wait_started_at_monotonic_s=now,
            ttl_expiry_observed=False,
            release_deadline_monotonic_s=now + self.config.paused_program_ttl_seconds,
            last_pause_at_monotonic_s=now,
            pause_reason=event.reason,
        )
        if expiry:
            state.global_factors.update_ttl_pause(served_rounds=record.segment_served_rounds)
            changes["rounds_since_ttl_pause"] = 0
        else:
            changes.update(
                is_privileged=False,
                privilege_deadline_monotonic_s=None,
                privilege_reason=f"{event.reason}_pause_clear",
            )
        self._reset_segment(state, event, **changes)

    def on_program_marked_for_pause(self, state: ProgressTTLFactors, event: SchedulingEvent) -> None:
        self._program_factors(state, event.program)
        state.update(event.program, ttl_deadline_monotonic_s=None)

    def on_program_released(self, state: ProgressTTLFactors, event: SchedulingEvent) -> None:
        state.remove_program_factors(event.program)

    def _reset_segment(self, state: ProgressTTLFactors, event: SchedulingEvent, **overrides) -> None:
        self._program_factors(state, event.program)
        defaults = ProgressTTLProgramFactors()
        reset = {
            key: getattr(defaults, key)
            for key in (
                "segment_served_rounds",
                "segment_prompt_tokens",
                "segment_completion_tokens",
                "is_evictable_after_min_rounds",
                "wait_started_at_monotonic_s",
                "request_wait_started_at_monotonic_s",
                "force_resume_timeout_seconds",
                "force_resume_deadline_monotonic_s",
                "force_resume_active_remaining_rounds",
                "force_resume_pool_remaining_rounds",
                "force_resume_request_throughput_per_second",
                "acting_since_monotonic_s",
                "ttl_deadline_monotonic_s",
                "release_deadline_monotonic_s",
            )
        }
        reset["segment_started_at_monotonic_s"] = event.occurred_at_monotonic_s
        reset.update(overrides)
        state.update(event.program, **reset)

    def _resume_key(
        self, program: ProgramView, state: ProgressTTLFactors, now: float
    ) -> tuple[int, float, str, int]:
        """Order forced and privileged waiters first, then prefer recently completed sessions."""
        factors = self._program_factors(state, program.ref)
        priority, time_key = recovery_order(
            forced=self._force_resume_due(program, state, now),
            privileged=factors.is_privileged,
            reasoning=program.status is ProgramStatus.REASONING,
            last_finished=factors.last_request_finished_at_monotonic_s,
        )
        return priority, time_key, program.ref.program_id, program.ref.generation

    def _force_resume_due(self, program: ProgramView, state: ProgressTTLFactors, now: float) -> bool:
        """Return whether a retained reasoning request reached its frozen adaptive deadline."""
        deadline = self._program_factors(state, program.ref).force_resume_deadline_monotonic_s
        return program.status is ProgramStatus.REASONING and deadline is not None and now >= deadline

    def _dynamic_force_resume_timeout(
        self,
        snapshot: SchedulingSnapshot,
        state: ProgressTTLFactors,
        candidate: ProgramView,
    ) -> _ForceResumeTimeoutEstimate:
        stats = state.global_factors
        rate = stats.request_throughput_per_second
        minimum, maximum = (
            self.config.force_resume_timeout_min_seconds,
            self.config.force_resume_timeout_max_seconds,
        )
        if not stats.request_window_complete or rate <= 0:
            return _ForceResumeTimeoutEstimate(maximum, 0.0, 0.0, rate)
        target = self._target_growth_rounds(state)
        now = snapshot.observed_at_monotonic_s
        priority = self._resume_key(candidate, state, now)
        running_work = 0.0
        for item in self._active_programs(snapshot):
            floor = float(item.status is ProgramStatus.REASONING)
            running_work += max(floor, target - self._program_factors(state, item.ref).segment_served_rounds)
        queued_work = sum(
            max(1.0, target - self._program_factors(state, item.ref).segment_served_rounds)
            for item in self._waiting_for_resume(snapshot)
            if self._resume_key(item, state, now) < priority
        )
        delay = self.config.force_resume_timeout_scale * (running_work + queued_work) / rate
        return _ForceResumeTimeoutEstimate(max(minimum, min(maximum, delay)), running_work, queued_work, rate)

    @staticmethod
    def _waiting_for_resume(snapshot: SchedulingSnapshot) -> Iterator[ProgramView]:
        ordered = map(snapshot.program, snapshot.waiting_programs)
        return (
            view
            for view in ordered
            if view is not None
            and view.status is ProgramStatus.REASONING
            and view.state is ProgramState.PAUSED
        )

    @staticmethod
    def _active_programs(snapshot: SchedulingSnapshot) -> list[ProgramView]:
        return [
            view
            for view in snapshot.programs
            if view.backend_id == snapshot.backend_id and view.state is ProgramState.ACTIVE
        ]

    def _privileged_program_limit(self, snapshot: SchedulingSnapshot) -> int:
        return max(1, (snapshot.total_kv_tokens or 0) // (2 * self.config.privileged_max_context_tokens))

    def _set_privileged(
        self,
        state: ProgressTTLFactors,
        ref: ProgramRef,
        enabled: bool,
        *,
        reason: str,
        now_monotonic_s: float | None = None,
    ) -> None:
        previous = self._program_factors(state, ref)
        if (previous.is_privileged, previous.privilege_reason) == (enabled, reason):
            return
        deadline = None
        if enabled:
            if now_monotonic_s is None:
                raise ValueError("granting privilege requires the current replica time")
            deadline = now_monotonic_s + self.config.privileged_ttl_seconds
        state.update(
            ref,
            privilege_reason=reason,
            is_privileged=enabled,
            privilege_deadline_monotonic_s=deadline,
            privilege_ttl_expired=previous.privilege_ttl_expired and not enabled,
        )
        logger.info("Progress-TTL privilege program=%s enabled=%s reason=%s", ref, enabled, reason)

    def _reconcile_privileges(self, snapshot: SchedulingSnapshot, state: ProgressTTLFactors) -> None:
        placed = {p.ref for p in self._active_programs(snapshot)}
        now = snapshot.observed_at_monotonic_s
        limit = self._privileged_program_limit(snapshot)
        valid = []
        for ref, lease in state.program_factors:
            if not lease.is_privileged:
                continue
            session = snapshot.program(ref)
            in_queue = (
                session is not None
                and session.state is ProgramState.PAUSED
                and session.status is ProgramStatus.REASONING
            )
            deadline = lease.privilege_deadline_monotonic_s
            if (
                session is None
                or session.task_id is None
                or not (ref in placed or in_queue)
                or (deadline is not None and deadline <= now)
            ):
                self._set_privileged(state, ref, False, reason="privilege_reconcile_clear")
                continue
            valid.append(
                (-lease.segment_served_rounds, -self._required_tokens(session), ref, session.task_id)
            )
        owners = set()
        for _, _, ref, task in sorted(valid):
            if task in owners or len(owners) >= limit:
                self._set_privileged(state, ref, False, reason="privilege_limit_clear")
            else:
                owners.add(task)

    def _can_promote(
        self, snapshot: SchedulingSnapshot, state: ProgressTTLFactors, candidate: ProgramView
    ) -> bool:
        if candidate.task_id is None or self._program_factors(state, candidate.ref).privilege_ttl_expired:
            return False
        used = 0
        for session in self._active_programs(snapshot):
            if not self._program_factors(state, session.ref).is_privileged:
                continue
            if candidate.task_id == session.task_id and candidate.ref != session.ref:
                return False
            used += 1
        return used < self._privileged_program_limit(snapshot)

    def _progress_key(
        self, state: ProgressTTLFactors, session: ProgramView
    ) -> tuple[int, int, int, ProgramRef]:
        progress = self._program_factors(state, session.ref).lifetime_generated_tokens
        return progress, session.step_count, self._required_tokens(session), session.ref

    def _promote_progress_privileges(self, snapshot: SchedulingSnapshot, state: ProgressTTLFactors) -> None:
        sessions = self._active_programs(snapshot)
        for session in sorted(sessions, key=lambda item: self._progress_key(state, item), reverse=True):
            if not self._program_factors(state, session.ref).is_privileged and self._can_promote(
                snapshot, state, session
            ):
                self._set_privileged(
                    state,
                    session.ref,
                    True,
                    reason="scheduled_progress_promote",
                    now_monotonic_s=snapshot.observed_at_monotonic_s,
                )

    def _privileged_relationship_source(
        self,
        snapshot: SchedulingSnapshot,
        state: ProgressTTLFactors,
        candidate: ProgramView,
    ) -> ProgramView | None:
        if candidate.task_id is None:
            return None
        parent = candidate.parent_program_id
        ident = candidate.ref.program_id
        donors = []
        for session in self._active_programs(snapshot):
            if (
                session.task_id != candidate.task_id
                or session.ref == candidate.ref
                or session.marked_for_pause
            ):
                continue
            if (
                session.status is not ProgramStatus.ACTING
                or not self._program_factors(state, session.ref).is_privileged
            ):
                continue
            is_parent = session.ref.program_id == parent
            if (
                is_parent
                or session.parent_program_id == ident
                or (parent is not None and session.parent_program_id == parent)
            ):
                donors.append((not is_parent, session.ref, session))
        return min(donors)[-1] if donors else None

    def _admit_by_privilege_handoff(
        self,
        snapshot: SchedulingSnapshot,
        state: ProgressTTLFactors,
        transitions: TransitionController,
        candidate: ProgramView,
        source: ProgramView,
        required_tokens: int,
    ) -> AdmissionOutcome:
        self._require_applied(transitions.pause(source.ref, reason="progress_ttl_privilege_source_handoff"))
        self._set_privileged(state, source.ref, False, reason="relationship_handoff_source")
        returning = (
            candidate.step_count
            or self._program_factors(state, candidate.ref).last_pause_at_monotonic_s is not None
        )
        handoff = transitions.resume if returning else transitions.admit
        self._require_applied(
            handoff(
                candidate.ref,
                backend_id=snapshot.backend_id,
                reason="progress_ttl_privilege_relationship_resume"
                if returning
                else "progress_ttl_privilege_relationship_admission",
            )
        )
        self._set_privileged(
            state,
            candidate.ref,
            True,
            reason="relationship_handoff_target",
            now_monotonic_s=snapshot.observed_at_monotonic_s,
        )
        retained = tuple(p for p in self._active_programs(snapshot) if p.ref != source.ref)
        reserve = self._continuous_growth_reserve_tokens(retained, state, candidate=candidate)
        return AdmissionOutcome(
            AdmissionDisposition.ADMITTED,
            "privileged_relationship_handoff",
            required_tokens,
            reserve,
            transitioned_programs=(source.ref, candidate.ref),
        )

    @staticmethod
    def _same_task_ttl_privilege_target(
        snapshot: SchedulingSnapshot, source: ProgramView
    ) -> ProgramView | None:
        if source.task_id is None:
            return None
        priorities = {
            (ProgramState.ACTIVE, ProgramStatus.REASONING): 0,
            (ProgramState.PAUSED, ProgramStatus.REASONING): 1,
            (ProgramState.ACTIVE, ProgramStatus.ACTING): 2,
        }
        peers = []
        for session in snapshot.programs:
            rank = priorities.get((session.state, session.status))
            if rank is not None and session.ref != source.ref and session.task_id == source.task_id:
                started = session.wait_started_at_monotonic_s
                peers.append(
                    (rank, started is None, math.inf if started is None else started, session.ref, session)
                )
        return min(peers)[-1] if peers else None

    def _demote_privileged_for_capacity(
        self, active: list[ProgramView], state: ProgressTTLFactors
    ) -> ProgramView | None:
        chosen = min(
            (p for p in active if self._program_factors(state, p.ref).is_privileged),
            key=lambda p: self._progress_key(state, p),
            default=None,
        )
        if chosen is not None:
            self._set_privileged(state, chosen.ref, False, reason="capacity_demote")
        return chosen

    def _continuous_growth_reserve_tokens(
        self,
        running: tuple[ProgramView, ...],
        state: ProgressTTLFactors,
        *,
        candidate: ProgramView,
    ) -> int:
        target = self._target_growth_rounds(state)
        outstanding = target
        for session in running:
            if session.ref != candidate.ref:
                outstanding += max(
                    0.0, target - self._program_factors(state, session.ref).segment_served_rounds
                )
        return math.ceil(outstanding * self._capacity_growth_per_round(state))

    def _capacity_growth_per_round(self, state: ProgressTTLFactors) -> float:
        configured = self.config.fixed_input_token_growth_per_round
        return (
            float(configured)
            if self.config.use_fixed_input_token_growth
            else max(0.0, state.global_factors.avg_input_token_growth_per_round)
        )

    def _protected_min_segment_rounds(self, state: ProgressTTLFactors) -> int:
        return int(self._target_growth_rounds(state))

    def _target_growth_rounds(self, state: ProgressTTLFactors) -> float:
        stats = state.global_factors
        learned = stats.estimated_continuity_rounds() if stats.request_window_complete else 0
        return float(min(learned, self.config.target_max_segment_rounds))

    def _remaining_tokens(self, snapshot: SchedulingSnapshot, capacity_ratio: float) -> int | None:
        total = snapshot.total_kv_tokens
        if total is None:
            return None
        allocated = snapshot.native_used_kv_tokens
        billed = self._active_programs(snapshot)
        if allocated is not None:
            billed = [session for session in billed if session.status is ProgramStatus.ACTING]
        committed = sum(self._required_tokens(session) for session in billed)
        if allocated is not None:
            committed += allocated + (snapshot.native_waiting_kv_tokens or 0)
        return int(total * capacity_ratio) - committed

    def _required_tokens(self, session: ProgramView) -> int:
        context = session.tokens
        private = max(0, context.estimated_context_tokens - context.shared_prefix_tokens)
        return self._capacity_cost(private + self.config.decode_buffer_tokens)

    def _pause_priority_score(self, session: ProgramView, state: ProgressTTLFactors, now: float) -> float:
        tenure = self._program_factors(state, session.ref)
        start = tenure.segment_started_at_monotonic_s
        lived = 0.0 if start is None else max(0.0, now - start)
        tokens = tenure.last_request_prompt_tokens if session.status is ProgramStatus.ACTING else 0
        normal = max(state.global_factors.avg_prompt_tokens, 1e-6)
        relative = (tokens or session.tokens.estimated_context_tokens) / normal
        return lived * max(0.5, 1 / math.sqrt(max(relative, 1e-6)))

    def _batch_gain_covers_recovery(
        self,
        active: tuple[ProgramView, ...],
        state: ProgressTTLFactors,
        candidate: ProgramView,
        available: int,
    ) -> bool:
        if not self.config.enable_batch_gain_admission:
            return False
        peers = [
            item.tokens
            for item in active
            if item.ref != candidate.ref and item.status is ProgramStatus.REASONING
        ]
        count = len(peers)
        if not count:
            return False
        total = sum(tokens.estimated_context_tokens for tokens in peers)
        adding = candidate.tokens.estimated_context_tokens
        base = self._costs.decode_rate(count, total)
        updated = self._costs.decode_rate(count + 1, total + adding)
        gain = state.global_factors.avg_decode_seconds * max(0.0, updated / base - 1.0)
        reload = self._costs.reload_cost(adding, candidate.tokens.shared_prefix_tokens)
        return gain >= reload + self._continuity_loss_seconds(active, state, candidate, available)

    def _continuity_loss_seconds(
        self,
        active: tuple[ProgramView, ...],
        state: ProgressTTLFactors,
        candidate: ProgramView,
        available: int,
    ) -> float:
        growth, target = self._capacity_growth_per_round(state), self._target_growth_rounds(state)
        if growth <= 0 or target <= 0:
            return 0.0
        outstanding = 0.0
        cost = 0.0
        count = 0
        for session in active:
            if session.ref == candidate.ref:
                continue
            remaining = target - self._program_factors(state, session.ref).segment_served_rounds
            if remaining <= 0:
                continue
            outstanding += remaining
            count += 1
            context = session.tokens
            cost += self._costs.reload_cost(context.estimated_context_tokens, context.shared_prefix_tokens)
        if not count or cost <= 0:
            return 0.0
        before = min(outstanding, max(0.0, available) / growth)
        after = min(
            outstanding, max(0.0, available - self._required_tokens(candidate) - growth * target) / growth
        )
        if before <= after:
            return 0.0
        return math.inf if after <= 0 else (before - after) / after * (cost / count)

    def _ttl_seconds(self, state: ProgressTTLFactors, prompt: int, shared: int) -> float:
        return state.global_factors.estimate_ttl(
            impact_seconds=self._costs.reload_cost(prompt, shared),
            minimum_seconds=self.config.ttl_min_seconds,
            maximum_seconds=self.config.ttl_max_seconds,
            impact_ratio=self.config.ttl_max_cache_miss_impact_ratio,
        ).ttl_seconds

    def _select_capacity_fit(
        self,
        sessions: list[ProgramView],
        deficit: int,
        *,
        per_program_relief_tokens: int = 0,
    ) -> tuple[ProgramView, ...] | None:
        if deficit <= 0:
            return ()
        ordered = sorted(sessions, key=lambda session: session.ref)
        weights = [self._required_tokens(session) + per_program_relief_tokens for session in ordered]
        indices = [i for i, weight in enumerate(weights) if weight > 0]
        total = sum(weights[i] for i in indices)
        if total < deficit:
            return None

        # For each unreached total, retain the shortest cover and break ties by session ID.
        partial: dict[int, tuple[int, ...]] = {0: ()}
        best: tuple[int, int, tuple[int, ...]] | None = None
        budget = _CAPACITY_FIT_SEARCH_STEPS
        for i in indices:
            budget -= len(partial)
            if budget < 0:
                break
            for subtotal, chosen in list(partial.items()):
                freed = subtotal + weights[i]
                key = chosen + (i,)
                if freed >= deficit:
                    rank = (freed, len(key), key)
                    if best is None or rank < best:
                        best = rank
                else:
                    previous = partial.get(freed)
                    if previous is None or (len(key), key) < (len(previous), previous):
                        partial[freed] = key
        else:
            return tuple(ordered[i] for i in best[2]) if best is not None else None

        # With the search budget exhausted, shed unnecessary victims in decreasing size order.
        cover = []
        for i in sorted(indices, key=lambda i: (-weights[i], i)):
            if total - weights[i] >= deficit:
                total -= weights[i]
            else:
                cover.append(i)
        fallback = (total, len(cover), tuple(sorted(cover)))
        chosen = fallback if best is None else min(fallback, best)
        return tuple(ordered[i] for i in chosen[2])

    def _select_layered_capacity_fit(
        self,
        tiers: tuple[list[ProgramView], ...],
        deficit: int,
    ) -> tuple[ProgramView, ...] | None:
        victims = []
        for tier in tiers:
            cover = self._select_capacity_fit(tier, deficit)
            if cover is not None:
                return tuple(victims) + cover
            for session in tier:
                victims.append(session)
                deficit -= self._required_tokens(session)
        return None

    @staticmethod
    def _program_factors(state: ProgressTTLFactors, ref: ProgramRef) -> ProgressTTLProgramFactors:
        return state.get_or_create(ref)

    @staticmethod
    def _require_applied(result: TransitionResult) -> None:
        if not result.applied:
            raise RuntimeError(
                f"Progress-TTL could not {result.kind.value} {result.program}: {result.reason}"
            )

    @staticmethod
    def _transition_applied(result: TransitionResult) -> bool:
        if not result.applied:
            logger.warning(
                "Progress-TTL action=%s session=%s reason=%s",
                result.kind.value,
                result.program,
                result.reason,
            )
        return result.applied

    def _queue(
        self,
        snapshot: SchedulingSnapshot,
        state: ProgressTTLFactors,
        actions: TransitionController,
        ref: ProgramRef,
        reason: str,
        required: int,
        reserve: int = 0,
        deficit: int = 0,
    ) -> AdmissionOutcome:
        prior = self._program_factors(state, ref)
        if prior.request_wait_started_at_monotonic_s is None:
            view = snapshot.program(ref)
            if view is None:
                raise ValueError("missing queued session in admission snapshot")
            estimate = self._dynamic_force_resume_timeout(snapshot, state, view)
            now = snapshot.observed_at_monotonic_s
            started = prior.wait_started_at_monotonic_s
            state.update(
                ref,
                wait_started_at_monotonic_s=now if started is None else started,
                request_wait_started_at_monotonic_s=now,
                force_resume_timeout_seconds=estimate.timeout_seconds,
                force_resume_deadline_monotonic_s=now + estimate.timeout_seconds,
                force_resume_active_remaining_rounds=estimate.active_remaining_rounds,
                force_resume_pool_remaining_rounds=estimate.pool_remaining_rounds,
                force_resume_request_throughput_per_second=estimate.request_throughput_per_second,
            )
        result = actions.queue(ref, reason=reason)
        if not result.applied:
            state.set_program_factors(ref, prior)
        self._require_applied(result)
        return AdmissionOutcome(AdmissionDisposition.QUEUED, reason, required, reserve, deficit, (ref,))
