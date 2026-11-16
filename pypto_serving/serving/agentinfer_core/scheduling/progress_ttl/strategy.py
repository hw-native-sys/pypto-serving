# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
# Modified by PyPTO Serving: internal imports use the bundled scheduling package.
"""Progress-TTL strategy over the shared AgentInfer scheduling contracts.

The strategy guarantees short consecutive service segments, retains acting Programs only while avoided cold-prefill
cost justifies it, resumes pending reasoning before idle acting work, and repairs capacity by segment-tenure priority.
It owns decision factors only and requests every core Program-state mutation through ``TransitionController``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from typing import Callable

from pypto_serving.serving.agentinfer_core.scheduling.admission_outcome import AdmissionDisposition, AdmissionOutcome
from pypto_serving.serving.agentinfer_core.scheduling.domain import ProgramRef, ProgramState, ProgramStatus, ProgramView
from pypto_serving.serving.agentinfer_core.scheduling.events import SchedulingEvent, SchedulingEventKind, StrategyDiagnostic
from pypto_serving.serving.agentinfer_core.scheduling.factors import StrategyFactors
from pypto_serving.serving.agentinfer_core.scheduling.progress_ttl.config import ProgressTTLConfig, ProgressTTLMode, ProgressTTLResumeOrder
from pypto_serving.serving.agentinfer_core.scheduling.progress_ttl.factors import ProgressTTLProgramFactors
from pypto_serving.serving.agentinfer_core.scheduling.progress_ttl.rolling_stats import ProgressTTLGlobalFactors, TTLEstimate
from pypto_serving.serving.agentinfer_core.scheduling.snapshot import SchedulingSnapshot
from pypto_serving.serving.agentinfer_core.scheduling.strategy import SchedulingStrategy
from pypto_serving.serving.agentinfer_core.scheduling.transitions import TransitionController, TransitionResult

ProgressTTLFactors = StrategyFactors[ProgressTTLGlobalFactors, ProgressTTLProgramFactors]
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ForceResumeTimeoutEstimate:
    """Frozen work-ahead estimate used to arm one force-resume deadline."""

    timeout_seconds: float
    active_remaining_rounds: float
    pool_remaining_rounds: float
    request_throughput_per_second: float


class ProgressTTLStrategy(SchedulingStrategy[ProgressTTLGlobalFactors, ProgressTTLProgramFactors]):
    """Concrete Progress-TTL admission, resume, capacity-repair, and deadline policy."""

    def __init__(
        self, config: ProgressTTLConfig, *, capacity_cost: Callable[[int], int] | None = None,
    ) -> None:
        self.config = config
        self._capacity_cost = capacity_cost or (lambda tokens: tokens)

    def diagnostics(
        self,
        snapshot: SchedulingSnapshot,
        strategy_factors: ProgressTTLFactors,
    ) -> tuple[StrategyDiagnostic, ...]:
        """Describe the current Program mix and capacity projection without changing policy state."""
        active = self._active_programs(snapshot)
        paused = [program for program in snapshot.programs if program.state is ProgramState.PAUSED]
        remaining_rounds = sum(
            max(
                0.0,
                self._target_growth_rounds(strategy_factors)
                - float(self._effective_segment_rounds(strategy_factors, program.ref)),
            )
            for program in active
        )
        rolling_growth = max(0.0, strategy_factors.global_factors.avg_input_token_growth_per_round)
        capacity_growth = self._capacity_growth_per_round(strategy_factors)
        projected_reserve = math.ceil(capacity_growth * remaining_rounds)
        effective_capacity = (
            int(snapshot.total_kv_tokens * self.config.resume_capacity_ratio)
            if snapshot.total_kv_tokens is not None
            else None
        )
        acting_reserved_tokens = sum(
            self._capacity_tokens(program) for program in active if program.status is ProgramStatus.ACTING
        )
        if snapshot.native_used_kv_tokens is None:
            used_tokens = sum(self._capacity_tokens(program) for program in active)
            capacity_source = "program_estimate"
        else:
            used_tokens = (
                snapshot.native_used_kv_tokens + acting_reserved_tokens + (snapshot.native_waiting_kv_tokens or 0)
            )
            capacity_source = "native_usage_plus_acting_plus_waiting"
        global_factors = strategy_factors.global_factors
        return (
            StrategyDiagnostic(
                "progress_ttl_state",
                "periodic_sample",
                (
                    ("active", len(active)),
                    ("active_reasoning", sum(program.status is ProgramStatus.REASONING for program in active)),
                    ("active_acting", sum(program.status is ProgramStatus.ACTING for program in active)),
                    ("paused", len(paused)),
                    ("paused_reasoning", sum(program.status is ProgramStatus.REASONING for program in paused)),
                    ("paused_acting", sum(program.status is ProgramStatus.ACTING for program in paused)),
                    ("waiting_programs", len(snapshot.waiting_programs)),
                    ("capacity_source", capacity_source),
                    ("used_tokens", used_tokens),
                    ("native_used_tokens", snapshot.native_used_kv_tokens),
                    ("acting_reserved_tokens", acting_reserved_tokens),
                    ("native_waiting_tokens", snapshot.native_waiting_kv_tokens),
                    ("uncached_free_head_tokens", snapshot.uncached_free_head_kv_tokens),
                    ("total_kv_tokens", snapshot.total_kv_tokens),
                    ("resume_effective_capacity", effective_capacity),
                    ("rolling_growth", rolling_growth),
                    ("capacity_growth", capacity_growth),
                    ("rolling_cache_churn_tokens", global_factors.avg_cache_churn_tokens_per_round),
                    ("rolling_cached_prefix_tokens", global_factors.avg_cached_prefix_tokens),
                    ("rolling_uncached_prompt_tokens", global_factors.avg_uncached_prompt_tokens),
                    ("rolling_completion_tokens", global_factors.avg_completion_tokens),
                    ("rolling_request_throughput_per_second", global_factors.request_throughput_per_second),
                    ("mode", self.config.mode.value),
                    ("transitions_enabled", self._transitions_enabled(strategy_factors)),
                    ("request_window_samples", global_factors.request_sample_count),
                    ("request_window_size", global_factors.request_window_size),
                    ("continuity_window_samples", global_factors.continuity_sample_count),
                    ("continuity_window_complete", global_factors.continuity_window_complete),
                    ("continuity_utility_seconds", global_factors.continuity_utility_seconds),
                    ("continuity_enabled", global_factors.continuity_enabled),
                    ("remaining_growth_rounds", remaining_rounds),
                    ("projected_active_reserve", projected_reserve),
                ),
            ),
        )

    def event_diagnostics(
        self,
        event: SchedulingEvent,
        strategy_factors: ProgressTTLFactors,
    ) -> tuple[StrategyDiagnostic, ...]:
        """Describe a frozen force-resume deadline or newly armed acting TTL after its committed event."""
        if event.program is None:
            return ()
        sidecar = self._program_factors(strategy_factors, event.program)
        if event.kind is SchedulingEventKind.REQUEST_PENDING:
            return (
                StrategyDiagnostic(
                    "progress_ttl_force_resume_armed",
                    event.reason,
                    (
                        ("program", event.program.program_id),
                        ("generation", event.program.generation),
                        ("timeout_seconds", sidecar.force_resume_timeout_seconds),
                        ("deadline", sidecar.force_resume_deadline_monotonic_s),
                        ("active_remaining_rounds", sidecar.force_resume_active_remaining_rounds),
                        ("pool_remaining_rounds", sidecar.force_resume_pool_remaining_rounds),
                        (
                            "request_throughput_per_second",
                            sidecar.force_resume_request_throughput_per_second,
                        ),
                    ),
                ),
            )
        if event.kind is not SchedulingEventKind.REQUEST_FINISHED:
            return ()
        if (
            event.current_status is not ProgramStatus.ACTING
            or sidecar.acting_since_monotonic_s is None
            or sidecar.ttl_deadline_monotonic_s is None
        ):
            return ()
        return (
            StrategyDiagnostic(
                "progress_ttl_armed",
                "request_finished",
                (
                    ("program", event.program.program_id),
                    ("generation", event.program.generation),
                    ("total_tokens", event.field("total_tokens")),
                    ("prompt_tokens", event.field("prompt_tokens")),
                    ("shared_prefix_tokens", sidecar.segment_share_tokens),
                    (
                        "cache_miss_impact_seconds",
                        self._cache_recovery_impact_seconds(
                            strategy_factors,
                            sidecar.last_request_prompt_tokens,
                            sidecar.segment_share_tokens,
                        ),
                    ),
                    ("acting_since", sidecar.acting_since_monotonic_s),
                    ("ttl_seconds", sidecar.ttl_deadline_monotonic_s - sidecar.acting_since_monotonic_s),
                    ("ttl_deadline", sidecar.ttl_deadline_monotonic_s),
                    ("continuity_window_samples", strategy_factors.global_factors.continuity_sample_count),
                    ("continuity_window_complete", strategy_factors.global_factors.continuity_window_complete),
                    ("continuity_utility_seconds", strategy_factors.global_factors.continuity_utility_seconds),
                    ("continuity_enabled", strategy_factors.global_factors.continuity_enabled),
                    ("mode", self.config.mode.value),
                    ("is_privileged", sidecar.is_privileged),
                ),
            ),
        )

    def shared_prefix_freshness_seconds(
        self,
        snapshot: SchedulingSnapshot,
        strategy_factors: ProgressTTLFactors,
        program: ProgramView,
    ) -> float:
        """Estimate the lifetime of one shared-prefix observation from KV-pool turnover."""
        del program
        global_factors = strategy_factors.global_factors
        if not global_factors.request_window_complete:
            return self.config.shared_prefix_freshness_warmup_seconds
        active_reasoning = sum(
            view.state is ProgramState.ACTIVE
            and view.status is ProgramStatus.REASONING
            and view.backend_id == snapshot.backend_id
            for view in snapshot.programs
        )
        if (
            active_reasoning <= 0
            or snapshot.total_kv_tokens is None
            or global_factors.avg_cache_churn_tokens_per_round <= 0
        ):
            return self.config.shared_prefix_freshness_warmup_seconds
        return (
            self.config.shared_prefix_freshness_kv_turnovers
            * snapshot.total_kv_tokens
            * global_factors.avg_request_latency_seconds
            / active_reasoning
            / global_factors.avg_cache_churn_tokens_per_round
        )

    def handle_admission(
        self,
        snapshot: SchedulingSnapshot,
        strategy_factors: ProgressTTLFactors,
        transitions: TransitionController,
        candidate: ProgramRef,
    ) -> AdmissionOutcome:
        """Admit new work only when its context and minimum-segment growth reserve both fit."""
        view = snapshot.program(candidate)
        if view is None:
            raise ValueError("admission candidate is absent from the scheduling snapshot")
        if not self._transitions_enabled(strategy_factors):
            return self._admit_without_pt(transitions, view, snapshot.backend_id)
        self._reconcile_privileges(snapshot, strategy_factors)
        required = self._required_tokens(view)
        if view.state is ProgramState.ACTIVE and view.backend_id == snapshot.backend_id:
            return AdmissionOutcome(AdmissionDisposition.ADMITTED, "already_active", required_tokens=required)
        relationship_source = self._privileged_relationship_source(snapshot, strategy_factors, view)
        if relationship_source is not None:
            return self._admit_by_privilege_handoff(
                snapshot,
                strategy_factors,
                transitions,
                view,
                relationship_source,
                required,
            )
        candidate_privileged = not snapshot.waiting_programs and self._can_promote(
            snapshot,
            strategy_factors,
            view,
        )
        reserve = self._continuous_growth_reserve_tokens(
            tuple(self._active_programs(snapshot)),
            strategy_factors,
            candidate=view,
        )
        if candidate in snapshot.waiting_programs:
            if snapshot.request_pool_waiting_requests == 1:
                remaining = self._remaining_tokens(snapshot, self.config.resume_capacity_ratio)
                required_with_reserve = required + reserve
                if remaining is not None and remaining >= required_with_reserve:
                    result = transitions.resume(
                        candidate,
                        reason="progress_ttl_arrival_resume",
                        backend_id=snapshot.backend_id,
                    )
                    self._require_applied(result)
                    return AdmissionOutcome(
                        AdmissionDisposition.ADMITTED,
                        "progress_ttl_arrival_resume",
                        required_tokens=required,
                        reserve_tokens=reserve,
                        transitioned_programs=(candidate,),
                    )
            return self._queue(
                snapshot,
                strategy_factors,
                transitions,
                candidate,
                "already_waiting",
                required,
                reserve,
            )
        if snapshot.waiting_programs:
            return self._queue(
                snapshot,
                strategy_factors,
                transitions,
                candidate,
                "waiting_queue_precedence",
                required,
                reserve,
            )
        remaining = self._remaining_tokens(snapshot, self.config.resume_capacity_ratio)
        if remaining is None:
            return self._queue(
                snapshot,
                strategy_factors,
                transitions,
                candidate,
                "capacity_unknown",
                required,
                reserve,
            )
        required_with_reserve = required + reserve
        if remaining < required_with_reserve:
            if remaining >= required and self._batch_gain_covers_recovery(
                tuple(self._active_programs(snapshot)),
                strategy_factors,
                view,
                remaining,
            ):
                result = transitions.admit(
                    candidate,
                    reason="batch_gain_over_recovery_cost",
                    backend_id=snapshot.backend_id,
                )
                self._require_applied(result)
                if candidate_privileged:
                    self._set_privileged(
                        strategy_factors,
                        view.ref,
                        True,
                        reason="batch_gain_admission_promote",
                        now_monotonic_s=snapshot.observed_at_monotonic_s,
                    )
                return AdmissionOutcome(
                    AdmissionDisposition.ADMITTED,
                    "batch_gain_over_recovery_cost",
                    required_tokens=required,
                    reserve_tokens=reserve,
                    transitioned_programs=(candidate,),
                )
            return self._queue(
                snapshot,
                strategy_factors,
                transitions,
                candidate,
                "insufficient_capacity",
                required,
                reserve,
                required_with_reserve - remaining,
            )
        result = transitions.admit(candidate, reason="direct_capacity", backend_id=snapshot.backend_id)
        self._require_applied(result)
        if candidate_privileged:
            self._set_privileged(
                strategy_factors,
                view.ref,
                True,
                reason="direct_admission_promote",
                now_monotonic_s=snapshot.observed_at_monotonic_s,
            )
        return AdmissionOutcome(
            AdmissionDisposition.ADMITTED,
            "direct_capacity",
            required_tokens=required,
            reserve_tokens=reserve,
            transitioned_programs=(candidate,),
        )

    def schedule_resume(
        self,
        snapshot: SchedulingSnapshot,
        strategy_factors: ProgressTTLFactors,
        transitions: TransitionController,
    ) -> None:
        """Resume paused reasoning Programs while off, otherwise score-order reasoning Programs under capacity."""
        if not self._transitions_enabled(strategy_factors):
            for program in snapshot.programs:
                # Keep paused acting Programs paused: their retention and TTL
                # state must survive an auto on -> off transition.  A later
                # request changes such a Program to reasoning and the off-mode
                # admission path then activates it directly.
                if program.state is not ProgramState.PAUSED or program.status is not ProgramStatus.REASONING:
                    continue
                self._transition_applied(
                    transitions.resume(program.ref, reason="progress_ttl_off_resume", backend_id=snapshot.backend_id)
                )
            return
        self._reconcile_privileges(snapshot, strategy_factors)
        remaining = self._remaining_tokens(snapshot, self.config.resume_capacity_ratio)
        active = self._active_programs(snapshot)
        waiting = [
            program
            for ref in snapshot.waiting_programs
            if (program := snapshot.program(ref)) is not None
            and program.state is ProgramState.PAUSED
            and program.status is ProgramStatus.REASONING
        ]
        waiting.sort(key=lambda program: self._resume_key(program, strategy_factors, snapshot.observed_at_monotonic_s))
        oldest_waiting = min(
            (
                program for program in waiting
                if self._program_factors(strategy_factors, program.ref).request_wait_started_at_monotonic_s is not None
            ),
            key=lambda program: (
                self._program_factors(strategy_factors, program.ref).request_wait_started_at_monotonic_s,
                program.ref.program_id,
                program.ref.generation,
            ),
            default=None,
        )
        victims = (
            [
                program
                for program in snapshot.programs
                if program.state is ProgramState.ACTIVE
                and program.status is ProgramStatus.ACTING
                and program.backend_id == snapshot.backend_id
                and self._program_factors(strategy_factors, program.ref).is_evictable_after_min_rounds
                and not self._program_factors(strategy_factors, program.ref).is_privileged
            ]
            if self.config.resume_reclaim_acting_programs
            else []
        )
        used_victims: set[ProgramRef] = set()
        for program in waiting:
            sidecar = self._program_factors(strategy_factors, program.ref)
            required = self._required_tokens(program)
            if self._force_resume_due(program, strategy_factors, snapshot.observed_at_monotonic_s):
                request_wait_started = sidecar.request_wait_started_at_monotonic_s
                waited_seconds = (
                    snapshot.observed_at_monotonic_s - request_wait_started if request_wait_started is not None else 0.0
                )
                physical_remaining = self._physical_remaining_tokens(snapshot, remaining)
                growth_reserve = self._continuous_growth_reserve_tokens(
                    tuple(active), strategy_factors, candidate=program
                )
                batch_gain = (
                    remaining is not None
                    and remaining >= required
                    and self._batch_gain_covers_recovery(tuple(active), strategy_factors, program, remaining)
                )
                logger.warning(
                    "Progress-TTL force resume program=%s generation=%d waited_seconds=%.1f "
                    "required_tokens=%d remaining_tokens=%s physical_remaining_tokens=%s "
                    "growth_reserve_tokens=%d batch_gain_covers_recovery=%s",
                    program.ref.program_id,
                    program.ref.generation,
                    waited_seconds,
                    required,
                    remaining,
                    physical_remaining,
                    growth_reserve,
                    batch_gain,
                )
                self._require_applied(
                    transitions.resume(
                        program.ref,
                        reason="progress_ttl_force_resume_timeout",
                        backend_id=snapshot.backend_id,
                    )
                )
                if remaining is not None:
                    remaining -= required
                active.append(program)
                continue
            if sidecar.is_privileged:
                self._require_applied(
                    transitions.resume(
                        program.ref,
                        reason="progress_ttl_privileged_resume",
                        backend_id=snapshot.backend_id,
                    )
                )
                if remaining is not None:
                    remaining -= required
                active.append(program)
                continue
            if remaining is None:
                continue
            wait_started = sidecar.request_wait_started_at_monotonic_s
            if (
                self.config.resume_wait_relax_seconds is not None
                and oldest_waiting is not None
                and program.ref == oldest_waiting.ref
                and wait_started is not None
                and snapshot.observed_at_monotonic_s - wait_started >= self.config.resume_wait_relax_seconds
                and (physical_remaining := self._physical_remaining_tokens(snapshot, remaining)) is not None
                and physical_remaining >= required
            ):
                growth_reserve = self._continuous_growth_reserve_tokens(
                    tuple(active), strategy_factors, candidate=program
                )
                protected_growth = math.ceil(
                    growth_reserve * self.config.resume_wait_growth_reserve_fraction
                )
                uncached_free_head = snapshot.uncached_free_head_kv_tokens
                cache_safe_remaining = (
                    None if uncached_free_head is None else
                    uncached_free_head - (snapshot.native_waiting_kv_tokens or 0)
                    - sum(self._capacity_tokens(item) for item in active if item.status is ProgramStatus.ACTING)
                )
                if (physical_remaining >= required + protected_growth
                        + self.config.resume_wait_extra_headroom_tokens
                        and (not self.config.resume_wait_preserve_cached_blocks
                             or (cache_safe_remaining is not None and cache_safe_remaining >= required))):
                    self._require_applied(
                        transitions.resume(
                            program.ref,
                            reason="progress_ttl_wait_capacity_relax",
                            backend_id=snapshot.backend_id,
                        )
                    )
                    logger.info(
                        "Progress-TTL wait capacity relax program=%s generation=%d at_monotonic=%.6f "
                        "waited_seconds=%.3f required_tokens=%d effective_remaining_tokens=%d "
                        "physical_remaining_tokens=%d growth_reserve_tokens=%d "
                        "protected_growth_tokens=%d extra_headroom_tokens=%d cache_safe_remaining_tokens=%s "
                        "uncovered_reserve_tokens=%d "
                        "native_used_kv_tokens=%s native_waiting_kv_tokens=%s active_count=%d",
                        program.ref.program_id,
                        program.ref.generation,
                        snapshot.observed_at_monotonic_s,
                        snapshot.observed_at_monotonic_s - wait_started,
                        required,
                        remaining,
                        physical_remaining,
                        growth_reserve,
                        protected_growth,
                        self.config.resume_wait_extra_headroom_tokens,
                        cache_safe_remaining,
                        max(0, required + growth_reserve - remaining),
                        snapshot.native_used_kv_tokens,
                        snapshot.native_waiting_kv_tokens,
                        len(active),
                    )
                    remaining -= required
                    active.append(program)
                    continue
            reserve = self._continuous_growth_reserve_tokens(tuple(active), strategy_factors, candidate=program)
            required_with_reserve = required + reserve
            if remaining < required_with_reserve:
                if remaining >= required and self._batch_gain_covers_recovery(
                    tuple(active),
                    strategy_factors,
                    program,
                    remaining,
                ):
                    self._require_applied(
                        transitions.resume(
                            program.ref,
                            reason="progress_ttl_resume_batch_gain",
                            backend_id=snapshot.backend_id,
                        )
                    )
                    remaining -= required
                    active.append(program)
                    continue
                deficit = required_with_reserve - remaining
                available = [victim for victim in victims if victim.ref not in used_victims]
                selected = self._select_layered_capacity_fit(
                    (
                        [
                            victim
                            for victim in available
                            if self._program_factors(strategy_factors, victim.ref).segment_served_rounds
                            >= self.config.target_max_segment_rounds
                        ],
                        [
                            victim
                            for victim in available
                            if self._program_factors(strategy_factors, victim.ref).segment_served_rounds
                            < self.config.target_max_segment_rounds
                        ],
                    ),
                    deficit,
                )
                if selected is None:
                    continue
                selected_refs = {victim.ref for victim in selected}
                for victim in selected:
                    self._require_applied(transitions.pause(victim.ref, reason="progress_ttl_resume_reclaim"))
                    used_victims.add(victim.ref)
                    remaining += self._capacity_tokens(victim)
                active = [item for item in active if item.ref not in selected_refs]
                reserve = self._continuous_growth_reserve_tokens(tuple(active), strategy_factors, candidate=program)
                required_with_reserve = required + reserve
                if remaining < required_with_reserve:
                    continue
            self._require_applied(
                transitions.resume(program.ref, reason="progress_ttl_resume", backend_id=snapshot.backend_id)
            )
            remaining -= required
            active.append(program)

    def repair_capacity(
        self,
        snapshot: SchedulingSnapshot,
        strategy_factors: ProgressTTLFactors,
        transitions: TransitionController,
    ) -> None:
        """Restore pause headroom by segment-tenure priority across both Program statuses."""
        if not self._transitions_enabled(strategy_factors):
            return
        self._reconcile_privileges(snapshot, strategy_factors)
        self._promote_progress_privileges(snapshot, strategy_factors)
        all_active = self._active_programs(snapshot)
        active = [program for program in all_active if not program.marked_for_pause]
        remaining = self._remaining_tokens(snapshot, self.config.pause_capacity_ratio)
        required_headroom = self._capacity_repair_headroom_tokens(all_active, strategy_factors)
        future_pause_relief_tokens = sum(
            self._capacity_tokens(program) + self._expected_completion_tokens(strategy_factors)
            for program in all_active
            if program.status is ProgramStatus.REASONING and program.marked_for_pause
        )
        if remaining is None or remaining + future_pause_relief_tokens >= required_headroom:
            return
        deficit = required_headroom - remaining - future_pause_relief_tokens
        candidates = [
            program for program in active if not self._program_factors(strategy_factors, program.ref).is_privileged
        ]
        while deficit > 0:
            if not candidates:
                demoted = self._demote_privileged_for_capacity(active, strategy_factors)
                if demoted is None:
                    return
                candidates.append(demoted)
            ordered = sorted(
                candidates,
                key=lambda program: (
                    -self._pause_priority_score(
                        program,
                        strategy_factors,
                        snapshot.observed_at_monotonic_s,
                    ),
                    -self._capacity_tokens(program),
                    program.ref.program_id,
                    program.ref.generation,
                ),
            )
            for victim in ordered:
                if victim.status is ProgramStatus.ACTING:
                    result = transitions.pause(victim.ref, reason="progress_ttl_capacity_repair")
                else:
                    result = transitions.mark_for_pause(victim.ref, reason="progress_ttl_capacity_repair")
                self._require_applied(result)
                candidates.remove(victim)
                victim_relief = self._capacity_tokens(victim)
                if victim.status is ProgramStatus.REASONING:
                    victim_relief += self._expected_completion_tokens(strategy_factors)
                deficit -= victim_relief
                if deficit <= 0:
                    return

    def handle_request_completion(
        self,
        snapshot: SchedulingSnapshot,
        strategy_factors: ProgressTTLFactors,
        transitions: TransitionController,
        completed: ProgramRef,
    ) -> None:
        """Yield a maximum-round ordinary Program only while Progress-TTL transitions are enabled."""
        if not self._transitions_enabled(strategy_factors):
            return
        program = snapshot.program(completed)
        if (
            program is None
            or program.state is not ProgramState.ACTIVE
            or program.status is not ProgramStatus.ACTING
            or program.backend_id != snapshot.backend_id
            or self._program_factors(strategy_factors, completed).is_privileged
            or self._effective_segment_rounds(strategy_factors, completed) < self.config.target_max_segment_rounds
        ):
            return
        waiting = [
            candidate
            for ref in snapshot.waiting_programs
            if (candidate := snapshot.program(ref)) is not None
            and candidate.state is ProgramState.PAUSED
            and candidate.status is ProgramStatus.REASONING
        ]
        if not waiting:
            return
        top = min(
            waiting,
            key=lambda candidate: self._resume_key(
                candidate,
                strategy_factors,
                snapshot.observed_at_monotonic_s,
            ),
        )
        future_paused = sum(
            self._capacity_tokens(candidate)
            for candidate in snapshot.programs
            if candidate.state is ProgramState.ACTIVE
            and candidate.backend_id == snapshot.backend_id
            and candidate.marked_for_pause
        )
        if self._required_tokens(top) <= future_paused:
            return
        self._require_applied(transitions.pause(completed, reason="progress_ttl_max_round_queue_pressure"))

    def next_check_at(self, strategy_factors: ProgressTTLFactors) -> float | None:
        """Return the earliest live acting TTL deadline."""
        deadlines = (
            deadline
            for _, state in strategy_factors.program_factors
            for deadline in (
                (
                    state.ttl_deadline_monotonic_s
                    if self._transitions_enabled(strategy_factors) or not state.ttl_expiry_observed
                    else None
                ),
                (
                    state.acting_since_monotonic_s + self.config.ttl_max_seconds
                    if state.acting_since_monotonic_s is not None
                    and (self._transitions_enabled(strategy_factors) or not state.ttl_expiry_observed)
                    else None
                ),
                state.release_deadline_monotonic_s,
            )
            if deadline is not None
        )
        return min(deadlines, default=None)

    def next_lightweight_check_at(self, strategy_factors: ProgressTTLFactors) -> float | None:
        """Return the earliest privilege-only expiry without waking the full policy cycle."""
        return min(
            (
                sidecar.privilege_deadline_monotonic_s
                for _, sidecar in strategy_factors.program_factors
                if sidecar.is_privileged and sidecar.privilege_deadline_monotonic_s is not None
            ),
            default=None,
        )

    def handle_lightweight_check(
        self,
        strategy_factors: ProgressTTLFactors,
        now_monotonic_s: float,
    ) -> None:
        """Expire bounded privilege sidecars without resume, repair, or TTL transitions."""
        for ref, sidecar in tuple(strategy_factors.program_factors):
            deadline = sidecar.privilege_deadline_monotonic_s
            if sidecar.is_privileged and deadline is not None and deadline <= now_monotonic_s:
                self._set_privileged(strategy_factors, ref, False, reason="privilege_ttl_expired")
                current = self._program_factors(strategy_factors, ref)
                strategy_factors.set_program_factors(ref, replace(current, privilege_ttl_expired=True))

    def handle_scheduled_check(
        self,
        snapshot: SchedulingSnapshot,
        strategy_factors: ProgressTTLFactors,
        transitions: TransitionController,
    ) -> None:
        """Pause stale acting Programs and release paused Programs after their retention TTL."""
        self._reconcile_privileges(snapshot, strategy_factors)
        now = snapshot.observed_at_monotonic_s
        protected_targets: set[ProgramRef] = set()
        for program in snapshot.programs:
            sidecar = self._program_factors(strategy_factors, program.ref)
            fallback_deadline = (
                sidecar.acting_since_monotonic_s
                + self._ttl_seconds(
                    strategy_factors,
                    sidecar.last_request_prompt_tokens or program.tokens.estimated_context_tokens,
                    sidecar.segment_share_tokens,
                )
                if sidecar.acting_since_monotonic_s is not None
                else None
            )
            acting_deadline_due = (
                sidecar.ttl_deadline_monotonic_s is not None and sidecar.ttl_deadline_monotonic_s <= now
            ) or (fallback_deadline is not None and fallback_deadline <= now)
            invalid_acting_clock = (
                program.state is ProgramState.ACTIVE
                and program.status is ProgramStatus.ACTING
                and sidecar.acting_since_monotonic_s is None
            )
            if (
                (acting_deadline_due or invalid_acting_clock)
                and program.state is ProgramState.ACTIVE
                and program.status is ProgramStatus.ACTING
                and program.ref not in protected_targets
            ):
                if not self._transitions_enabled(strategy_factors):
                    if not sidecar.ttl_expiry_observed:
                        self._record_theoretical_ttl_expiry(strategy_factors, program.ref, now)
                    protected_targets.add(program.ref)
                    continue
                target = self._same_task_ttl_privilege_target(snapshot, program) if sidecar.is_privileged else None
                self._require_applied(transitions.pause(program.ref, reason="progress_ttl_expired"))
                if target is None:
                    continue
                self._set_privileged(strategy_factors, program.ref, False, reason="ttl_handoff_source")
                if target.state is ProgramState.PAUSED:
                    self._require_applied(
                        transitions.resume(
                            target.ref,
                            reason="progress_ttl_privilege_ttl_handoff",
                            backend_id=snapshot.backend_id,
                        )
                    )
                self._set_privileged(
                    strategy_factors,
                    target.ref,
                    True,
                    reason="ttl_handoff",
                    now_monotonic_s=snapshot.observed_at_monotonic_s,
                )
                protected_targets.add(target.ref)
        for program in snapshot.programs:
            sidecar = self._program_factors(strategy_factors, program.ref)
            if (
                program.state is ProgramState.PAUSED
                and program.status is ProgramStatus.ACTING
                and sidecar.release_deadline_monotonic_s is not None
                and sidecar.release_deadline_monotonic_s <= now
            ):
                self._require_applied(transitions.release(program.ref, reason="progress_ttl_paused_retention_expired"))

    def on_request_pending(self, strategy_factors: ProgressTTLFactors, event: SchedulingEvent) -> None:
        """Freeze one request's adaptive force deadline and invalidate an acting TTL."""
        if event.program is None:
            return
        current = self._program_factors(strategy_factors, event.program)
        request_wait_started = current.request_wait_started_at_monotonic_s
        timeout_seconds = current.force_resume_timeout_seconds
        deadline = current.force_resume_deadline_monotonic_s
        if request_wait_started is None:
            request_wait_started = event.occurred_at_monotonic_s
            timeout_seconds = self.config.force_resume_timeout_max_seconds
            deadline = request_wait_started + timeout_seconds
        strategy_factors.set_program_factors(
            event.program,
            replace(
                current,
                wait_started_at_monotonic_s=(
                    current.wait_started_at_monotonic_s
                    if current.wait_started_at_monotonic_s is not None
                    else event.occurred_at_monotonic_s
                ),
                request_wait_started_at_monotonic_s=request_wait_started,
                force_resume_timeout_seconds=timeout_seconds,
                force_resume_deadline_monotonic_s=deadline,
                acting_since_monotonic_s=None,
                ttl_deadline_monotonic_s=None,
                release_deadline_monotonic_s=None,
            ),
        )

    def on_program_admitted(self, strategy_factors: ProgressTTLFactors, event: SchedulingEvent) -> None:
        """Start a fresh service segment after initial admission."""
        self._start_segment(strategy_factors, event, resume=False)

    def on_program_resumed(self, strategy_factors: ProgressTTLFactors, event: SchedulingEvent) -> None:
        """Start a fresh service segment after resume without counting a round yet."""
        self._start_segment(strategy_factors, event, resume=True)

    def on_program_status_changed(self, strategy_factors: ProgressTTLFactors, event: SchedulingEvent) -> None:
        """Invalidate TTL as soon as another reasoning request arrives."""
        if event.program is None or event.current_status is not ProgramStatus.REASONING:
            return
        current = self._program_factors(strategy_factors, event.program)
        strategy_factors.set_program_factors(
            event.program,
            replace(current, acting_since_monotonic_s=None, ttl_deadline_monotonic_s=None),
        )

    def on_request_finished(self, strategy_factors: ProgressTTLFactors, event: SchedulingEvent) -> None:
        """Count a completed round and arm TTL only after all overlapping reasoning is done."""
        if event.program is None:
            return
        prompt_tokens = self._event_int(event.field("prompt_tokens"))
        completion_tokens = self._event_int(event.field("completion_tokens"))
        total_tokens = self._event_int(event.field("total_tokens")) or prompt_tokens + completion_tokens
        previous_context_tokens = self._event_int(event.field("previous_context_tokens"))
        context_growth_tokens = max(0, total_tokens - previous_context_tokens)
        cached_prefix_tokens = min(
            prompt_tokens,
            self._event_int(event.field("cached_prefix_tokens")),
        )
        shared_prefix_tokens = min(
            cached_prefix_tokens,
            self._event_int(event.field("shared_prefix_tokens")),
        )
        hbm_cached_observation = self._event_optional_float(event.field("hbm_cached_prefix_tokens"))
        hbm_cached_prefix_tokens = min(
            prompt_tokens,
            cached_prefix_tokens if hbm_cached_observation is None else int(hbm_cached_observation),
        )
        hbm_hit_ref_zero_tokens = min(
            hbm_cached_prefix_tokens,
            self._event_int(event.field("hbm_hit_ref_zero_tokens")),
        )
        current = self._program_factors(strategy_factors, event.program)
        segment_hbm_hit_ref_zero_tokens = hbm_hit_ref_zero_tokens if current.segment_first_request_pending else 0
        cache_churn_tokens = (
            max(0, prompt_tokens - hbm_cached_prefix_tokens) + segment_hbm_hit_ref_zero_tokens + completion_tokens
        )
        continuous_round_completed = (
            previous_context_tokens <= 0 or hbm_cached_prefix_tokens > 0.9 * previous_context_tokens
        )
        rounds_since_ttl_pause = current.rounds_since_ttl_pause + int(continuous_round_completed)
        global_factors = strategy_factors.global_factors
        inter_request_gap_seconds = self._event_float(event.field("inter_request_gap_seconds"))
        request_latency_seconds = self._event_float(event.field("request_latency_seconds"))
        global_factors.observe_request_interval(
            started_at_monotonic_s=max(0.0, event.occurred_at_monotonic_s - request_latency_seconds),
            finished_at_monotonic_s=event.occurred_at_monotonic_s,
        )
        global_factors.update_request(
            prompt_tokens=prompt_tokens,
            cached_prefix_tokens=cached_prefix_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            request_latency_seconds=request_latency_seconds,
            decode_seconds=self._event_optional_float(event.field("decode_seconds")),
            active_programs=self._event_int(event.field("active_programs")),
            waiting_programs=self._event_int(event.field("waiting_programs")),
            input_token_growth=context_growth_tokens,
            cache_churn_tokens=cache_churn_tokens,
            inter_request_gap_seconds=inter_request_gap_seconds,
            scheduler_queue_seconds=self._event_float(event.field("scheduler_queue_seconds")),
            rounds_since_ttl_pause=rounds_since_ttl_pause,
        )
        cache_miss_impact_seconds = self._cache_recovery_impact_seconds(
            strategy_factors,
            prompt_tokens,
            shared_prefix_tokens,
        )
        if current.previous_ttl_seconds is not None and inter_request_gap_seconds > 0:
            continuity_window_was_complete = global_factors.continuity_window_complete
            global_factors.observe_continuity(
                interval_seconds=inter_request_gap_seconds,
                cache_miss_impact_seconds=cache_miss_impact_seconds,
                assigned_ttl_seconds=current.previous_ttl_seconds,
            )
            if not continuity_window_was_complete and global_factors.continuity_window_complete:
                global_factors.recalibrate_continuity_window(
                    minimum_seconds=self.config.ttl_min_seconds,
                    maximum_seconds=self.config.ttl_max_seconds,
                    impact_ratio=self.config.ttl_max_cache_miss_impact_ratio,
                )
            global_factors.update_continuity_mode(
                enable_threshold_seconds=self.config.auto_enable_utility_seconds,
                disable_threshold_seconds=self.config.auto_disable_utility_seconds,
            )
        ttl_estimate = self._ttl_estimate(strategy_factors, prompt_tokens, shared_prefix_tokens)
        rounds = current.segment_served_rounds + int(continuous_round_completed)
        ttl_deadline = None
        acting_since = None
        if event.current_status is ProgramStatus.ACTING:
            acting_since = event.occurred_at_monotonic_s
            ttl_deadline = event.occurred_at_monotonic_s + ttl_estimate.ttl_seconds
        strategy_factors.set_program_factors(
            event.program,
            replace(
                current,
                segment_served_rounds=rounds,
                segment_prompt_tokens=current.segment_prompt_tokens + prompt_tokens,
                segment_completion_tokens=current.segment_completion_tokens + completion_tokens,
                segment_first_request_pending=False,
                last_request_prompt_tokens=prompt_tokens,
                last_request_context_tokens=total_tokens,
                lifetime_generated_tokens=current.lifetime_generated_tokens + completion_tokens,
                segment_share_tokens=shared_prefix_tokens,
                rounds_since_ttl_pause=rounds_since_ttl_pause,
                previous_ttl_seconds=ttl_estimate.ttl_seconds,
                ttl_expiry_observed=False,
                is_evictable_after_min_rounds=rounds >= self._protected_min_segment_rounds(strategy_factors),
                acting_since_monotonic_s=acting_since,
                ttl_deadline_monotonic_s=ttl_deadline,
                release_deadline_monotonic_s=None,
                last_inter_request_gap_seconds=inter_request_gap_seconds,
                last_request_finished_at_monotonic_s=event.occurred_at_monotonic_s,
            ),
        )

    def on_program_paused(self, strategy_factors: ProgressTTLFactors, event: SchedulingEvent) -> None:
        """Freeze the completed service segment for later resume scoring."""
        if event.program is None:
            return
        current = self._program_factors(strategy_factors, event.program)
        if event.reason == "progress_ttl_expired":
            strategy_factors.global_factors.update_ttl_pause(served_rounds=current.segment_served_rounds)
        strategy_factors.set_program_factors(
            event.program,
            replace(
                current,
                segment_served_rounds=0,
                segment_started_at_monotonic_s=None,
                segment_prompt_tokens=0,
                segment_completion_tokens=0,
                segment_first_request_pending=True,
                last_segment_served_rounds=current.segment_served_rounds,
                last_segment_prompt_tokens=current.segment_prompt_tokens,
                last_segment_completion_tokens=current.segment_completion_tokens,
                is_evictable_after_min_rounds=False,
                wait_started_at_monotonic_s=event.occurred_at_monotonic_s,
                request_wait_started_at_monotonic_s=None,
                force_resume_timeout_seconds=None,
                force_resume_deadline_monotonic_s=None,
                force_resume_active_remaining_rounds=0.0,
                force_resume_pool_remaining_rounds=0.0,
                force_resume_request_throughput_per_second=0.0,
                acting_since_monotonic_s=None,
                ttl_deadline_monotonic_s=None,
                ttl_expiry_observed=False,
                release_deadline_monotonic_s=(event.occurred_at_monotonic_s + self.config.paused_program_ttl_seconds),
                last_pause_at_monotonic_s=event.occurred_at_monotonic_s,
                pause_reason=event.reason,
                rounds_since_ttl_pause=(
                    0 if event.reason == "progress_ttl_expired" else current.rounds_since_ttl_pause
                ),
                is_privileged=current.is_privileged if event.reason == "progress_ttl_expired" else False,
                privilege_reason=(
                    current.privilege_reason
                    if event.reason == "progress_ttl_expired"
                    else f"{event.reason}_pause_clear"
                ),
                privilege_deadline_monotonic_s=(
                    current.privilege_deadline_monotonic_s if event.reason == "progress_ttl_expired" else None
                ),
            ),
        )

    def on_program_marked_for_pause(self, strategy_factors: ProgressTTLFactors, event: SchedulingEvent) -> None:
        """Invalidate an acting deadline when a reasoning Program is committed to pause."""
        if event.program is None:
            return
        current = self._program_factors(strategy_factors, event.program)
        strategy_factors.set_program_factors(event.program, replace(current, ttl_deadline_monotonic_s=None))

    def on_program_released(self, strategy_factors: ProgressTTLFactors, event: SchedulingEvent) -> None:
        """Remove terminal sidecars and their deadlines."""
        if event.program is not None:
            strategy_factors.remove_program_factors(event.program)

    def _start_segment(self, strategy_factors: ProgressTTLFactors, event: SchedulingEvent, *, resume: bool) -> None:
        if event.program is None:
            return
        current = self._program_factors(strategy_factors, event.program)
        strategy_factors.set_program_factors(
            event.program,
            replace(
                current,
                segment_served_rounds=0,
                segment_started_at_monotonic_s=event.occurred_at_monotonic_s,
                segment_prompt_tokens=0,
                segment_completion_tokens=0,
                is_evictable_after_min_rounds=False,
                wait_started_at_monotonic_s=None,
                request_wait_started_at_monotonic_s=None,
                force_resume_timeout_seconds=None,
                force_resume_deadline_monotonic_s=None,
                force_resume_active_remaining_rounds=0.0,
                force_resume_pool_remaining_rounds=0.0,
                force_resume_request_throughput_per_second=0.0,
                acting_since_monotonic_s=None,
                ttl_deadline_monotonic_s=None,
                release_deadline_monotonic_s=None,
                last_resume_at_monotonic_s=(
                    event.occurred_at_monotonic_s if resume else current.last_resume_at_monotonic_s
                ),
                resume_reason=event.reason if resume else current.resume_reason,
            ),
        )

    def _resume_key(self, program: ProgramView, state: ProgressTTLFactors, now: float) -> tuple[int, float, str, int]:
        """Order forced and privileged waiters first, then apply the configured ordinary ordering."""
        sidecar = self._program_factors(state, program.ref)
        if self._force_resume_due(program, state, now):
            tier = -2
        elif sidecar.is_privileged:
            tier = -1
        elif program.status is ProgramStatus.REASONING and sidecar.last_request_finished_at_monotonic_s is not None:
            tier = 0
        elif program.status is ProgramStatus.REASONING:
            tier = 1
        else:
            tier = 2
        if self.config.resume_order is ProgressTTLResumeOrder.FCFS:
            order_key = (
                sidecar.request_wait_started_at_monotonic_s
                if sidecar.request_wait_started_at_monotonic_s is not None
                else math.inf
            )
        else:
            order_key = (
                -sidecar.last_request_finished_at_monotonic_s
                if sidecar.last_request_finished_at_monotonic_s is not None
                else 0.0
            )
        return tier, order_key, program.ref.program_id, program.ref.generation

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
        """Convert work ahead at queue entry into one bounded, immutable force-resume timeout."""
        stats = state.global_factors
        request_throughput = stats.request_throughput_per_second
        if not stats.request_window_complete or request_throughput <= 0:
            return _ForceResumeTimeoutEstimate(
                timeout_seconds=self.config.force_resume_timeout_max_seconds,
                active_remaining_rounds=0.0,
                pool_remaining_rounds=0.0,
                request_throughput_per_second=request_throughput,
            )
        target_rounds = self._target_growth_rounds(state)
        active_remaining_rounds = sum(
            max(
                1.0 if program.status is ProgramStatus.REASONING else 0.0,
                target_rounds - float(self._effective_segment_rounds(state, program.ref)),
            )
            for program in self._active_programs(snapshot)
        )
        candidate_key = self._resume_key(candidate, state, snapshot.observed_at_monotonic_s)
        pool_remaining_rounds = sum(
            max(
                1.0,
                target_rounds - float(self._effective_segment_rounds(state, program.ref)),
            )
            for ref in snapshot.waiting_programs
            if (program := snapshot.program(ref)) is not None
            and program.state is ProgramState.PAUSED
            and program.status is ProgramStatus.REASONING
            and self._resume_key(program, state, snapshot.observed_at_monotonic_s) < candidate_key
        )
        raw_timeout = (
            self.config.force_resume_timeout_scale
            * (active_remaining_rounds + pool_remaining_rounds)
            / request_throughput
        )
        timeout = min(
            self.config.force_resume_timeout_max_seconds,
            max(self.config.force_resume_timeout_min_seconds, raw_timeout),
        )
        return _ForceResumeTimeoutEstimate(
            timeout_seconds=timeout,
            active_remaining_rounds=active_remaining_rounds,
            pool_remaining_rounds=pool_remaining_rounds,
            request_throughput_per_second=request_throughput,
        )

    def _active_programs(self, snapshot: SchedulingSnapshot) -> list[ProgramView]:
        """Return Programs currently consuming logical capacity on this backend."""
        return [
            program
            for program in snapshot.programs
            if program.state is ProgramState.ACTIVE and program.backend_id == snapshot.backend_id
        ]

    def _privilege_enabled(self) -> bool:
        """Return whether bounded workflow privilege is available."""
        return True

    def _privileged_program_limit(self, snapshot: SchedulingSnapshot) -> int:
        """Reserve at most half of the backend's full-context slots for privileged Programs."""
        if snapshot.total_kv_tokens is None or snapshot.total_kv_tokens <= 0:
            return 1
        full_context_slots = snapshot.total_kv_tokens // self.config.privileged_max_context_tokens
        return max(1, full_context_slots // 2)

    def _reconcile_privileges(self, snapshot: SchedulingSnapshot, state: ProgressTTLFactors) -> None:
        """Clear stale, duplicate-task, and over-limit privilege sidecars from fresh runtime facts."""
        privileged: list[ProgramView] = []
        for ref, sidecar in state.program_factors:
            if not sidecar.is_privileged:
                continue
            program = snapshot.program(ref)
            if (
                not self._privilege_enabled()
                or program is None
                or (
                    program.state is not ProgramState.ACTIVE
                    and not (program.state is ProgramState.PAUSED and program.status is ProgramStatus.REASONING)
                )
                or (program.state is ProgramState.ACTIVE and program.backend_id != snapshot.backend_id)
                or program.task_id is None
                or (
                    sidecar.privilege_deadline_monotonic_s is not None
                    and sidecar.privilege_deadline_monotonic_s <= snapshot.observed_at_monotonic_s
                )
            ):
                self._set_privileged(state, ref, False, reason="privilege_reconcile_clear")
                continue
            privileged.append(program)
        privileged.sort(
            key=lambda program: (
                -self._program_factors(state, program.ref).segment_served_rounds,
                -self._capacity_tokens(program),
                program.ref.program_id,
                program.ref.generation,
            )
        )
        retained_tasks: set[str] = set()
        retained_count = 0
        limit = self._privileged_program_limit(snapshot)
        for program in privileged:
            task_id = program.task_id
            if task_id is None or task_id in retained_tasks or retained_count >= limit:
                self._set_privileged(state, program.ref, False, reason="privilege_limit_clear")
                continue
            retained_tasks.add(task_id)
            retained_count += 1

    def _can_promote(self, snapshot: SchedulingSnapshot, state: ProgressTTLFactors, candidate: ProgramView) -> bool:
        """Return whether a candidate can occupy a backend and task privilege slot."""
        if (
            not self._privilege_enabled()
            or candidate.task_id is None
            or self._program_factors(state, candidate.ref).privilege_ttl_expired
        ):
            return False
        privileged = [
            program
            for program in snapshot.programs
            if self._program_factors(state, program.ref).is_privileged
            and program.state is ProgramState.ACTIVE
            and program.backend_id == snapshot.backend_id
        ]
        if any(program.task_id == candidate.task_id and program.ref != candidate.ref for program in privileged):
            return False
        return len(privileged) < self._privileged_program_limit(snapshot)

    def _set_privileged(
        self,
        state: ProgressTTLFactors,
        ref: ProgramRef,
        enabled: bool,
        *,
        reason: str,
        now_monotonic_s: float | None = None,
    ) -> None:
        """Update one privilege bit using the caller's deterministic decision clock."""
        current = self._program_factors(state, ref)
        if current.is_privileged is enabled and current.privilege_reason == reason:
            return
        if enabled and now_monotonic_s is None:
            raise ValueError("now_monotonic_s is required when enabling privilege")
        state.set_program_factors(
            ref,
            replace(
                current,
                is_privileged=enabled,
                privilege_reason=reason,
                privilege_deadline_monotonic_s=(
                    now_monotonic_s + self.config.privileged_ttl_seconds
                    if now_monotonic_s is not None and enabled
                    else None
                ),
                privilege_ttl_expired=False if enabled else current.privilege_ttl_expired,
            ),
        )
        logger.info(
            "Progress-TTL privilege %s program=%s generation=%d reason=%s",
            "enabled" if enabled else "disabled",
            ref.program_id,
            ref.generation,
            reason,
        )

    def _promote_progress_privileges(self, snapshot: SchedulingSnapshot, state: ProgressTTLFactors) -> None:
        """Fill available slots from active Programs with the most lifetime generated tokens."""
        if not self._privilege_enabled():
            return
        while True:
            candidates = [
                program
                for program in snapshot.programs
                if program.state is ProgramState.ACTIVE
                and program.backend_id == snapshot.backend_id
                and not self._program_factors(state, program.ref).is_privileged
                and not self._program_factors(state, program.ref).privilege_ttl_expired
                and self._can_promote(snapshot, state, program)
            ]
            if not candidates:
                return
            selected = max(
                candidates,
                key=lambda program: (
                    self._program_factors(state, program.ref).lifetime_generated_tokens,
                    program.step_count,
                    self._capacity_tokens(program),
                    program.ref.program_id,
                    program.ref.generation,
                ),
            )
            self._set_privileged(
                state,
                selected.ref,
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
        """Return an explicitly related privileged active-acting source in historical priority order."""
        if not self._privilege_enabled() or candidate.task_id is None:
            return None

        def relation_tier(source: ProgramView) -> int | None:
            if candidate.parent_program_id == source.ref.program_id:
                return 0
            if candidate.parent_program_id is not None and source.parent_program_id == candidate.parent_program_id:
                return 1
            if source.parent_program_id == candidate.ref.program_id:
                return 1
            return None

        sources: list[tuple[int, ProgramView]] = []
        for source in snapshot.programs:
            tier = relation_tier(source)
            if (
                tier is not None
                and source.ref != candidate.ref
                and source.task_id == candidate.task_id
                and source.state is ProgramState.ACTIVE
                and source.status is ProgramStatus.ACTING
                and source.backend_id == snapshot.backend_id
                and not source.marked_for_pause
                and self._program_factors(state, source.ref).is_privileged
            ):
                sources.append((tier, source))
        if not sources:
            return None
        return min(sources, key=lambda item: (item[0], item[1].ref.program_id, item[1].ref.generation))[1]

    def _admit_by_privilege_handoff(
        self,
        snapshot: SchedulingSnapshot,
        state: ProgressTTLFactors,
        transitions: TransitionController,
        candidate: ProgramView,
        source: ProgramView,
        required_tokens: int,
    ) -> AdmissionOutcome:
        """Transfer a task's service privilege by pausing its safe acting source.

        This bounded priority path intentionally bypasses the ordinary capacity watermark. The next capacity-repair
        cycle restores headroom by pausing ordinary Programs first, or by demoting a privileged Program if necessary.
        """
        self._require_applied(transitions.pause(source.ref, reason="progress_ttl_privilege_source_handoff"))
        self._set_privileged(state, source.ref, False, reason="relationship_handoff_source")
        candidate_state = self._program_factors(state, candidate.ref)
        if candidate.step_count > 0 or candidate_state.last_pause_at_monotonic_s is not None:
            target_result = transitions.resume(
                candidate.ref,
                reason="progress_ttl_privilege_relationship_resume",
                backend_id=snapshot.backend_id,
            )
        else:
            target_result = transitions.admit(
                candidate.ref,
                reason="progress_ttl_privilege_relationship_admission",
                backend_id=snapshot.backend_id,
            )
        self._require_applied(target_result)
        self._set_privileged(
            state,
            candidate.ref,
            True,
            reason="relationship_handoff_target",
            now_monotonic_s=snapshot.observed_at_monotonic_s,
        )
        active_after_handoff = tuple(
            program for program in self._active_programs(snapshot) if program.ref not in {source.ref, candidate.ref}
        )
        reserve = self._continuous_growth_reserve_tokens(
            active_after_handoff,
            state,
            candidate=candidate,
        )
        return AdmissionOutcome(
            AdmissionDisposition.ADMITTED,
            "privileged_relationship_handoff",
            required_tokens=required_tokens,
            reserve_tokens=reserve,
            transitioned_programs=(source.ref, candidate.ref),
        )

    def _same_task_ttl_privilege_target(
        self,
        snapshot: SchedulingSnapshot,
        source: ProgramView,
    ) -> ProgramView | None:
        """Choose active reasoning, paused reasoning, then active acting as a TTL privilege target."""
        if source.task_id is None:
            return None
        candidates: list[tuple[int, ProgramView]] = []
        for target in snapshot.programs:
            if target.ref == source.ref or target.task_id != source.task_id:
                continue
            if target.state is ProgramState.ACTIVE and target.status is ProgramStatus.REASONING:
                candidates.append((0, target))
            elif target.state is ProgramState.PAUSED and target.status is ProgramStatus.REASONING:
                candidates.append((1, target))
            elif target.state is ProgramState.ACTIVE and target.status is ProgramStatus.ACTING:
                candidates.append((2, target))
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda item: (
                item[0],
                item[1].wait_started_at_monotonic_s is None,
                (item[1].wait_started_at_monotonic_s if item[1].wait_started_at_monotonic_s is not None else math.inf),
                item[1].ref.program_id,
                item[1].ref.generation,
            ),
        )[1]

    def _demote_privileged_for_capacity(
        self,
        active: list[ProgramView],
        state: ProgressTTLFactors,
    ) -> ProgramView | None:
        """Demote the least-progressed privilege only when ordinary victims cannot restore safety."""
        candidates = [program for program in active if self._program_factors(state, program.ref).is_privileged]
        if not candidates:
            return None
        demoted = min(
            candidates,
            key=lambda program: (
                self._program_factors(state, program.ref).lifetime_generated_tokens,
                program.step_count,
                self._capacity_tokens(program),
                program.ref.program_id,
                program.ref.generation,
            ),
        )
        self._set_privileged(state, demoted.ref, False, reason="capacity_demote")
        return demoted

    def _continuous_growth_reserve_tokens(
        self,
        active_programs: tuple[ProgramView, ...],
        strategy_factors: ProgressTTLFactors,
        *,
        candidate: ProgramView,
    ) -> int:
        """Reserve context growth until every post-activation Program reaches the workload-derived target.

        The candidate starts a fresh service segment and therefore contributes all configured minimum rounds. Active
        Programs contribute only their remaining rounds. Admission and resume call this same projection so neither
        path can fill capacity with current contexts while omitting the near-term growth required by the policy.
        """
        remaining_rounds = self._target_growth_rounds(strategy_factors)
        for program in active_programs:
            if program.ref == candidate.ref:
                continue
            served = self._program_factors(strategy_factors, program.ref).segment_served_rounds
            target = self._target_growth_rounds(strategy_factors)
            remaining_rounds += max(0.0, target - float(served))
        growth = self._capacity_growth_per_round(strategy_factors)
        return math.ceil(growth * remaining_rounds)

    def _capacity_growth_per_round(self, state: ProgressTTLFactors) -> float:
        """Return fixed deployment growth when enabled, otherwise the rolling workload estimate."""
        if self.config.use_fixed_input_token_growth:
            return float(self.config.fixed_input_token_growth_per_round)
        return max(0.0, state.global_factors.avg_input_token_growth_per_round)

    def _transitions_enabled(self, state: ProgressTTLFactors) -> bool:
        """Return whether Progress-TTL may apply lifecycle transitions in the selected mode."""
        if self.config.mode is ProgressTTLMode.ON:
            return True
        if self.config.mode is ProgressTTLMode.OFF:
            return False
        return state.global_factors.continuity_enabled

    def _admit_without_pt(
        self,
        transitions: TransitionController,
        candidate: ProgramView,
        backend_id: str,
    ) -> AdmissionOutcome:
        """Admit directly while Progress-TTL transitions are disabled."""
        required = self._required_tokens(candidate)
        if candidate.state is ProgramState.ACTIVE and candidate.backend_id == backend_id:
            return AdmissionOutcome(
                AdmissionDisposition.ADMITTED,
                "progress_ttl_off_already_active",
                required_tokens=required,
            )
        result = transitions.admit(candidate.ref, reason="progress_ttl_off_direct", backend_id=backend_id)
        if not result.applied:
            return AdmissionOutcome(
                AdmissionDisposition.QUEUED,
                "progress_ttl_off_direct_transition_rejected",
                required_tokens=required,
            )
        return AdmissionOutcome(
            AdmissionDisposition.ADMITTED,
            "progress_ttl_off_direct",
            required_tokens=required,
            transitioned_programs=(candidate.ref,),
        )

    def _record_theoretical_ttl_expiry(
        self,
        state: ProgressTTLFactors,
        ref: ProgramRef,
        now_monotonic_s: float,
    ) -> None:
        """Record an off-mode TTL boundary without changing the Program state."""
        current = self._program_factors(state, ref)
        state.set_program_factors(
            ref,
            replace(
                current,
                rounds_since_ttl_pause=0,
                ttl_expiry_observed=True,
                last_pause_at_monotonic_s=now_monotonic_s,
                pause_reason="progress_ttl_theoretical_expired",
            ),
        )

    def _protected_min_segment_rounds(self, state: ProgressTTLFactors) -> int:
        """Return the workload-derived protection target."""
        return int(self._target_growth_rounds(state))

    def _target_growth_rounds(self, state: ProgressTTLFactors) -> float:
        """Return the workload-derived growth target bounded by the segment limit."""
        if not state.global_factors.request_window_complete:
            return 0.0
        return float(
            min(
                self.config.target_max_segment_rounds,
                state.global_factors.estimated_continuity_rounds(),
            )
        )

    def _remaining_tokens(self, snapshot: SchedulingSnapshot, capacity_ratio: float) -> int | None:
        if snapshot.total_kv_tokens is None:
            return None
        effective_capacity = int(snapshot.total_kv_tokens * capacity_ratio)
        if snapshot.native_used_kv_tokens is None:
            used = sum(
                self._capacity_tokens(program)
                for program in snapshot.programs
                if program.state is ProgramState.ACTIVE and program.backend_id == snapshot.backend_id
            )
        else:
            acting_reserved = sum(
                self._capacity_tokens(program)
                for program in snapshot.programs
                if program.state is ProgramState.ACTIVE
                and program.status is ProgramStatus.ACTING
                and program.backend_id == snapshot.backend_id
            )
            used = snapshot.native_used_kv_tokens + acting_reserved + (snapshot.native_waiting_kv_tokens or 0)
        return effective_capacity - used

    def _physical_remaining_tokens(self, snapshot: SchedulingSnapshot, remaining: int | None) -> int | None:
        if remaining is None or snapshot.total_kv_tokens is None:
            return None
        reserved = snapshot.total_kv_tokens - int(snapshot.total_kv_tokens * self.config.resume_capacity_ratio)
        return remaining + reserved

    def _capacity_repair_headroom_tokens(
        self,
        active_programs: list[ProgramView],
        strategy_factors: ProgressTTLFactors,
    ) -> int:
        """Reserve only expected decode growth for requests that are currently reasoning."""
        reasoning_count = sum(program.status is ProgramStatus.REASONING for program in active_programs)
        return reasoning_count * self._expected_completion_tokens(strategy_factors)

    @staticmethod
    def _expected_completion_tokens(strategy_factors: ProgressTTLFactors) -> int:
        """Return the rolling completion-token estimate used by capacity repair."""
        return math.ceil(max(0.0, strategy_factors.global_factors.avg_completion_tokens))

    def _required_tokens(self, program: ProgramView) -> int:
        return self._capacity_cost(
            max(0, program.tokens.estimated_context_tokens - program.tokens.shared_prefix_tokens)
            + self.config.decode_buffer_tokens
        )

    def _effective_segment_rounds(self, state: ProgressTTLFactors, ref: ProgramRef) -> int:
        """Return current segment rounds used by the policy's soft progress boundaries."""
        return self._program_factors(state, ref).segment_served_rounds

    def _capacity_tokens(self, program: ProgramView) -> int:
        return self._required_tokens(program)

    def _pause_priority_score(
        self,
        program: ProgramView,
        state: ProgressTTLFactors,
        now_monotonic_s: float,
    ) -> float:
        """Return segment tenure scaled by current input reload cost for capacity repair."""
        sidecar = self._program_factors(state, program.ref)
        segment_started_at = sidecar.segment_started_at_monotonic_s
        duration = max(0.0, now_monotonic_s - segment_started_at) if segment_started_at is not None else 0.0
        input_tokens = program.tokens.estimated_context_tokens
        if program.status is ProgramStatus.ACTING and sidecar.last_request_prompt_tokens:
            input_tokens = sidecar.last_request_prompt_tokens
        return duration * self._resource_scale(input_tokens, state.global_factors.avg_prompt_tokens)

    def _cache_miss_impact_seconds(self, prompt_tokens: int, shared_prefix_tokens: int) -> float:
        """Estimate cold-prefill and decode-interference cost for private input KV."""
        uncached = max(0, prompt_tokens - shared_prefix_tokens)
        cold_prefill = self._cold_prefill_seconds(uncached)
        decode_interference = (1.0 - self.config.ttl_decode_throughput_alpha) * cold_prefill
        return cold_prefill + decode_interference

    def _cache_recovery_impact_seconds(
        self,
        state: ProgressTTLFactors,
        prompt_tokens: int,
        shared_prefix_tokens: int,
    ) -> float:
        """Estimate the no-offloading recovery impact from private uncached tokens."""
        return self._cache_miss_impact_seconds(prompt_tokens, shared_prefix_tokens)

    def _cold_prefill_seconds(self, uncached_tokens: int) -> float:
        """Estimate cold-prefill time from deployment-calibrated quadratic coefficients."""
        if uncached_tokens <= 0:
            return 0.0
        tokens_per_1k = uncached_tokens / 1000.0
        return (
            self.config.ttl_prefill_model_intercept_seconds
            + self.config.ttl_prefill_model_linear_seconds_per_1k_tokens * tokens_per_1k
            + self.config.ttl_prefill_model_quadratic_seconds_per_1k_tokens_squared * tokens_per_1k**2
        )

    def _batch_gain_covers_recovery(
        self,
        active_programs: tuple[ProgramView, ...],
        state: ProgressTTLFactors,
        candidate: ProgramView,
        remaining_tokens: int,
    ) -> bool:
        """Return whether batch gain covers candidate recovery and lost active continuity."""
        if not self.config.enable_batch_gain_admission:
            return False
        running = [
            program
            for program in active_programs
            if program.status is ProgramStatus.REASONING and program.ref != candidate.ref
        ]
        batch_size = len(running)
        if batch_size == 0:
            return False
        total_context = sum(program.tokens.estimated_context_tokens for program in running)
        candidate_context = candidate.tokens.estimated_context_tokens
        throughput_before = self._decode_throughput(batch_size, total_context)
        throughput_after = self._decode_throughput(batch_size + 1, total_context + candidate_context)
        relative_gain = max(0.0, throughput_after / throughput_before - 1.0)
        batch_gain_seconds = state.global_factors.avg_decode_seconds * relative_gain
        recovery_cost = self._cache_recovery_impact_seconds(
            state,
            candidate.tokens.estimated_context_tokens,
            candidate.tokens.shared_prefix_tokens,
        )
        continuity_loss = self._continuity_loss_seconds(
            active_programs,
            state,
            candidate,
            remaining_tokens,
        )
        return batch_gain_seconds >= recovery_cost + continuity_loss

    def _continuity_loss_seconds(
        self,
        active_programs: tuple[ProgramView, ...],
        state: ProgressTTLFactors,
        candidate: ProgramView,
        remaining_tokens: int,
    ) -> float:
        """Estimate recovery cost exposed by reducing active Programs' protected growth rounds."""
        growth = self._capacity_growth_per_round(state)
        target_rounds = self._target_growth_rounds(state)
        if growth <= 0 or target_rounds <= 0:
            return 0.0
        protected: list[tuple[ProgramView, float]] = []
        for program in active_programs:
            if program.ref == candidate.ref:
                continue
            served_rounds = float(self._program_factors(state, program.ref).segment_served_rounds)
            remaining_rounds = max(0.0, target_rounds - served_rounds)
            if remaining_rounds > 0:
                protected.append((program, remaining_rounds))
        total_protected_rounds = sum(rounds for _, rounds in protected)
        if total_protected_rounds <= 0:
            return 0.0
        protected_before = min(
            total_protected_rounds,
            max(0.0, remaining_tokens) / growth,
        )
        candidate_growth_tokens = growth * target_rounds
        protected_after = min(
            total_protected_rounds,
            max(
                0.0,
                remaining_tokens - self._required_tokens(candidate) - candidate_growth_tokens,
            )
            / growth,
        )
        lost_rounds = max(0.0, protected_before - protected_after)
        if lost_rounds <= 0:
            return 0.0
        recovery_costs = tuple(
            self._cache_recovery_impact_seconds(
                state,
                program.tokens.estimated_context_tokens,
                program.tokens.shared_prefix_tokens,
            )
            for program, _ in protected
        )
        average_recovery_cost = sum(recovery_costs) / len(recovery_costs)
        if average_recovery_cost <= 0:
            return 0.0
        if protected_after <= 0:
            return math.inf
        return lost_rounds / protected_after * average_recovery_cost

    def _decode_throughput(self, batch_size: int, total_context_tokens: int) -> float:
        """Evaluate the configured deployment-level decode throughput surface."""
        decode_step_seconds = (
            self.config.decode_step_fixed_seconds
            + self.config.decode_step_seconds_per_request * batch_size
            + self.config.decode_step_seconds_per_context_token * total_context_tokens
        )
        return batch_size / decode_step_seconds

    def _ttl_seconds(
        self,
        state: ProgressTTLFactors,
        prompt_tokens: int,
        shared_prefix_tokens: int,
    ) -> float:
        """Return the adaptive, impact-bounded acting TTL."""
        return self._ttl_estimate(state, prompt_tokens, shared_prefix_tokens).ttl_seconds

    def _ttl_estimate(
        self,
        state: ProgressTTLFactors,
        prompt_tokens: int,
        shared_prefix_tokens: int,
    ) -> TTLEstimate:
        """Return a fitted TTL and disable candidates with negative expected utility."""
        impact_seconds = self._cache_recovery_impact_seconds(state, prompt_tokens, shared_prefix_tokens)
        return state.global_factors.estimate_ttl(
            impact_seconds=impact_seconds,
            minimum_seconds=self.config.ttl_min_seconds,
            maximum_seconds=self.config.ttl_max_seconds,
            impact_ratio=self.config.ttl_max_cache_miss_impact_ratio,
        )

    def _select_capacity_fit(
        self,
        programs: list[ProgramView],
        deficit: int,
        *,
        per_program_relief_tokens: int = 0,
    ) -> tuple[ProgramView, ...] | None:
        if deficit <= 0:
            return ()
        reachable: dict[int, tuple[ProgramView, ...]] = {0: ()}
        best: tuple[int, tuple[ProgramView, ...]] | None = None
        for program in sorted(programs, key=lambda item: (item.ref.program_id, item.ref.generation)):
            tokens = self._capacity_tokens(program) + per_program_relief_tokens
            additions: dict[int, tuple[ProgramView, ...]] = {}
            for freed, selected in tuple(reachable.items()):
                new_freed = freed + tokens
                candidate = (*selected, program)
                if new_freed >= deficit:
                    key = (new_freed, len(candidate), tuple(item.ref for item in candidate))
                    if best is None or key < (best[0], len(best[1]), tuple(item.ref for item in best[1])):
                        best = (new_freed, candidate)
                else:
                    existing = additions.get(new_freed, reachable.get(new_freed))
                    candidate_key = (len(candidate), tuple(item.ref for item in candidate))
                    if existing is None or candidate_key < (len(existing), tuple(item.ref for item in existing)):
                        additions[new_freed] = candidate
            reachable.update(additions)
        return best[1] if best is not None else None

    def _select_layered_capacity_fit(
        self,
        layers: tuple[list[ProgramView], ...],
        deficit: int,
    ) -> tuple[ProgramView, ...] | None:
        """Prefer older progress tiers while minimizing over-release within each tier."""
        selected: list[ProgramView] = []
        remaining = deficit
        for layer in layers:
            fit = self._select_capacity_fit(layer, remaining)
            if fit is not None:
                return (*selected, *fit)
            selected.extend(layer)
            remaining -= sum(self._capacity_tokens(program) for program in layer)
            if remaining <= 0:
                return tuple(selected)
        return None

    @staticmethod
    def _program_factors(state: ProgressTTLFactors, ref: ProgramRef) -> ProgressTTLProgramFactors:
        current = state.for_program(ref)
        if current is None:
            current = ProgressTTLProgramFactors()
            state.set_program_factors(ref, current)
        return current

    @staticmethod
    def _resource_scale(tokens: int, average_tokens: float) -> float:
        ratio = max(float(tokens) / max(average_tokens, 1e-6), 1e-6)
        return max(0.5, 1.0 / math.sqrt(ratio))

    @staticmethod
    def _event_int(value: object) -> int:
        if isinstance(value, bool):
            return 0
        if isinstance(value, int):
            return max(0, value)
        if isinstance(value, float) and math.isfinite(value):
            return max(0, int(value))
        return 0

    @staticmethod
    def _event_float(value: object) -> float:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return 0.0
        result = float(value)
        return max(0.0, result) if math.isfinite(result) else 0.0

    @staticmethod
    def _event_optional_float(value: object) -> float | None:
        if value is None or not isinstance(value, (int, float)) or isinstance(value, bool):
            return None
        result = float(value)
        return max(0.0, result) if math.isfinite(result) else None

    @staticmethod
    def _require_applied(result: TransitionResult) -> None:
        if not result.applied:
            raise RuntimeError(f"Progress-TTL transition rejected: {result.kind.value}: {result.reason}")

    @staticmethod
    def _transition_applied(result: TransitionResult) -> bool:
        """Return whether a state change succeeded and warn instead of terminating the host on rejection."""
        if result.applied:
            return True
        logger.warning(
            "Progress-TTL transition rejected kind=%s program=%s generation=%d reason=%s",
            result.kind.value,
            result.program.program_id,
            result.program.generation,
            result.reason,
        )
        return False

    def _queue(
        self,
        snapshot: SchedulingSnapshot,
        strategy_factors: ProgressTTLFactors,
        transitions: TransitionController,
        candidate: ProgramRef,
        reason: str,
        required_tokens: int,
        reserve_tokens: int = 0,
        deficit_tokens: int = 0,
    ) -> AdmissionOutcome:
        current = self._program_factors(strategy_factors, candidate)
        previous = current
        if current.request_wait_started_at_monotonic_s is None:
            candidate_view = snapshot.program(candidate)
            if candidate_view is None:
                raise ValueError("queued candidate is absent from the scheduling snapshot")
            estimate = self._dynamic_force_resume_timeout(
                snapshot,
                strategy_factors,
                candidate_view,
            )
            wait_started = snapshot.observed_at_monotonic_s
            strategy_factors.set_program_factors(
                candidate,
                replace(
                    current,
                    wait_started_at_monotonic_s=(
                        current.wait_started_at_monotonic_s
                        if current.wait_started_at_monotonic_s is not None
                        else wait_started
                    ),
                    request_wait_started_at_monotonic_s=wait_started,
                    force_resume_timeout_seconds=estimate.timeout_seconds,
                    force_resume_deadline_monotonic_s=wait_started + estimate.timeout_seconds,
                    force_resume_active_remaining_rounds=estimate.active_remaining_rounds,
                    force_resume_pool_remaining_rounds=estimate.pool_remaining_rounds,
                    force_resume_request_throughput_per_second=estimate.request_throughput_per_second,
                ),
            )
        result = transitions.queue(candidate, reason=reason)
        if not result.applied:
            strategy_factors.set_program_factors(candidate, previous)
        self._require_applied(result)
        return AdmissionOutcome(
            AdmissionDisposition.QUEUED,
            reason,
            required_tokens=required_tokens,
            reserve_tokens=reserve_tokens,
            deficit_tokens=deficit_tokens,
            transitioned_programs=(candidate,),
        )
