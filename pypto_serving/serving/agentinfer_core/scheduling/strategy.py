# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
# Modified by PyPTO Serving: internal imports use the bundled scheduling package.
"""Scheduling policy extension point and lifecycle callback dispatcher.

``SchedulingStrategy`` is the package's intentional abstract surface. For each decision, the caller holds its
scheduling lock and passes one immutable ``SchedulingSnapshot``, one policy-owned ``StrategyFactors``, and one lock-scoped
``TransitionController`` directly to the Hook. A concrete policy computes a complete admission, post-completion yield,
resume, or capacity-repair plan from that fixed snapshot and updates its local capacity projection after each applied
transition; it does not refresh runtime state during the Hook. Committed runtime events use a separate single callback
path so strategy accounting is not applied twice.

The base class defines no score, capacity threshold, host runtime I/O, mutable Program operation, or concrete policy fields.
It also exposes ``next_check_at()`` and ``handle_scheduled_check()`` as an optional timer contract: the caller may
start a new locked Hook with a fresh snapshot at the requested monotonic deadline, rather than scheduling a future
transition from stale decision factors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Generic

from pypto_serving.serving.agentinfer_core.scheduling.admission_outcome import AdmissionOutcome
from pypto_serving.serving.agentinfer_core.scheduling.domain import ProgramRef, ProgramView
from pypto_serving.serving.agentinfer_core.scheduling.events import SchedulingEvent, SchedulingEventKind, StrategyDiagnostic
from pypto_serving.serving.agentinfer_core.scheduling.factors import StrategyFactors, StrategyGlobalFactorsT, StrategyProgramFactorsT
from pypto_serving.serving.agentinfer_core.scheduling.snapshot import SchedulingSnapshot
from pypto_serving.serving.agentinfer_core.scheduling.transitions import TransitionController


class SchedulingStrategy(Generic[StrategyGlobalFactorsT, StrategyProgramFactorsT], ABC):
    """Policy that consumes runtime facts and maintained factors under the caller's scheduling lock."""

    _EVENT_CALLBACKS: ClassVar[dict[SchedulingEventKind, str]] = {
        SchedulingEventKind.REQUEST_PENDING: "on_request_pending",
        SchedulingEventKind.REQUEST_ADMITTED: "on_program_admitted",
        SchedulingEventKind.REQUEST_FINISHED: "on_request_finished",
        SchedulingEventKind.PROGRAM_STATUS_CHANGED: "on_program_status_changed",
        SchedulingEventKind.PROGRAM_PAUSED: "on_program_paused",
        SchedulingEventKind.PROGRAM_MARKED_FOR_PAUSE: "on_program_marked_for_pause",
        SchedulingEventKind.PROGRAM_RESUMED: "on_program_resumed",
        SchedulingEventKind.PROGRAM_RELEASED: "on_program_released",
    }

    @abstractmethod
    def handle_admission(
        self,
        snapshot: SchedulingSnapshot,
        strategy_factors: StrategyFactors[StrategyGlobalFactorsT, StrategyProgramFactorsT],
        transitions: TransitionController,
        candidate: ProgramRef,
    ) -> AdmissionOutcome:
        """Admit or queue one candidate, optionally applying a fixed victim plan.

        Args:
            snapshot: Fixed backend, Program, and waiting-pool facts for this locked Hook.
            strategy_factors: Policy-owned decision factors protected by the caller's scheduling lock.
            transitions: Lock-scoped capability for caller-owned core state changes.
            candidate: Exact Program generation requesting admission.
        """

    @abstractmethod
    def schedule_resume(
        self,
        snapshot: SchedulingSnapshot,
        strategy_factors: StrategyFactors[StrategyGlobalFactorsT, StrategyProgramFactorsT],
        transitions: TransitionController,
    ) -> None:
        """Select and apply a fixed resume batch from one snapshot.

        Args:
            snapshot: Fixed backend, Program, and waiting-pool facts for this locked Hook.
            strategy_factors: Policy-owned decision factors protected by the caller's scheduling lock.
            transitions: Lock-scoped capability for caller-owned core state changes.
        """

    @abstractmethod
    def repair_capacity(
        self,
        snapshot: SchedulingSnapshot,
        strategy_factors: StrategyFactors[StrategyGlobalFactorsT, StrategyProgramFactorsT],
        transitions: TransitionController,
    ) -> None:
        """Select and apply a fixed pause or mark-for-pause plan.

        Args:
            snapshot: Fixed backend, Program, and waiting-pool facts for this locked Hook.
            strategy_factors: Policy-owned decision factors protected by the caller's scheduling lock.
            transitions: Lock-scoped capability for caller-owned core state changes.
        """

    def handle_request_completion(
        self,
        snapshot: SchedulingSnapshot,
        strategy_factors: StrategyFactors[StrategyGlobalFactorsT, StrategyProgramFactorsT],
        transitions: TransitionController,
        completed: ProgramRef,
    ) -> None:
        """Optionally yield the Program after its committed request accounting.

        The caller invokes this Hook with fresh facts after dispatching ``REQUEST_FINISHED``. A delayed pause already
        committed while the request was reasoning takes precedence, so the runtime skips this Hook when it can apply
        that pause immediately. The default policy does not yield on request completion.

        Args:
            snapshot: Fresh runtime facts after final token and status accounting.
            strategy_factors: Updated policy decision factors protected by the caller's scheduling lock.
            transitions: Lock-scoped capability for caller-owned core state changes.
            completed: Exact Program generation whose request just completed.
        """

    def on_schedule_cycle_start(
        self,
        snapshot: SchedulingSnapshot,
        strategy_factors: StrategyFactors[StrategyGlobalFactorsT, StrategyProgramFactorsT],
    ) -> None:
        """Reconcile idempotent decision factors before resume and capacity repair.

        Args:
            snapshot: Fixed runtime facts for the new scheduling cycle.
            strategy_factors: Policy-owned decision factors protected by the caller's scheduling lock.
        """

    def diagnostics(
        self,
        snapshot: SchedulingSnapshot,
        strategy_factors: StrategyFactors[StrategyGlobalFactorsT, StrategyProgramFactorsT],
    ) -> tuple[StrategyDiagnostic, ...]:
        """Return opt-in, observation-only calculations for one periodic sample.

        Args:
            snapshot: Fixed runtime facts for the sampled scheduling boundary.
            strategy_factors: Policy-owned decision factors protected by the caller's scheduling lock.
        """
        return ()

    def event_diagnostics(
        self,
        event: SchedulingEvent,
        strategy_factors: StrategyFactors[StrategyGlobalFactorsT, StrategyProgramFactorsT],
    ) -> tuple[StrategyDiagnostic, ...]:
        """Return opt-in calculations derived after one committed event.

        Args:
            event: Committed runtime fact already applied to the strategy factors.
            strategy_factors: Updated policy-owned factors protected by the caller's scheduling lock.
        """
        return ()

    def next_check_at(
        self,
        strategy_factors: StrategyFactors[StrategyGlobalFactorsT, StrategyProgramFactorsT],
    ) -> float | None:
        """Return the earliest absolute monotonic deadline for a new scheduling Hook.

        The default policy needs no deadline. A concrete strategy may inspect its current factors and return a timestamp,
        but the caller remains responsible for timer management and for constructing a fresh locked snapshot when the
        deadline expires.

        Args:
            strategy_factors: Current decision factors, including updates from the completed Hook.
        """
        return None

    def handle_scheduled_check(
        self,
        snapshot: SchedulingSnapshot,
        strategy_factors: StrategyFactors[StrategyGlobalFactorsT, StrategyProgramFactorsT],
        transitions: TransitionController,
    ) -> None:
        """Revalidate an expired policy deadline and request any resulting transitions.

        The caller invokes this optional Hook only after acquiring the shared scheduling lock and constructing fresh
        inputs. A stale or superseded timer is valid and should produce no transition. The default policy requests no
        timed checks, so this implementation is a no-op.

        Args:
            snapshot: Fresh backend, Program, and waiting-pool facts observed when the timer fired.
            strategy_factors: Current decision factors protected by the caller's scheduling lock.
            transitions: Lock-scoped capability for caller-owned core state changes.
        """

    def next_lightweight_check_at(
        self,
        strategy_factors: StrategyFactors[StrategyGlobalFactorsT, StrategyProgramFactorsT],
    ) -> float | None:
        """Return the earliest factor-only deadline that needs no full scheduling cycle."""
        return None

    def handle_lightweight_check(
        self,
        strategy_factors: StrategyFactors[StrategyGlobalFactorsT, StrategyProgramFactorsT],
        now_monotonic_s: float,
    ) -> None:
        """Apply due factor-only expiry without constructing a scheduling snapshot."""

    def shared_prefix_freshness_seconds(
        self,
        snapshot: SchedulingSnapshot,
        strategy_factors: StrategyFactors[StrategyGlobalFactorsT, StrategyProgramFactorsT],
        program: ProgramView,
    ) -> float | None:
        """Return the retention period for one observed shared-prefix boundary.

        ``None`` preserves the generic runtime's historical behavior: every eligible request or periodic probe may
        refresh its observation. A non-negative duration avoids repeated native prefix lookups and prevents residual
        self KV from being mistaken for a newly shared Program prefix before the duration expires.
        """
        return None

    def handle_scheduling_event(
        self,
        strategy_factors: StrategyFactors[StrategyGlobalFactorsT, StrategyProgramFactorsT],
        event: SchedulingEvent,
    ) -> None:
        """Dispatch one committed runtime event through exactly one callback path.

        Args:
            strategy_factors: Policy-owned decision factors receiving lifecycle updates.
            event: Committed runtime fact to dispatch.
        """
        callback_name = self._EVENT_CALLBACKS.get(event.kind)
        if callback_name is not None:
            getattr(self, callback_name)(strategy_factors, event)

    def on_request_pending(
        self,
        strategy_factors: StrategyFactors[StrategyGlobalFactorsT, StrategyProgramFactorsT],
        event: SchedulingEvent,
    ) -> None:
        """Initialize waiting-related factors after a request enters the Program pool."""

    def on_program_admitted(
        self,
        strategy_factors: StrategyFactors[StrategyGlobalFactorsT, StrategyProgramFactorsT],
        event: SchedulingEvent,
    ) -> None:
        """Update decision factors after a Program is admitted."""

    def on_request_finished(
        self,
        strategy_factors: StrategyFactors[StrategyGlobalFactorsT, StrategyProgramFactorsT],
        event: SchedulingEvent,
    ) -> None:
        """Update policy accounting from trustworthy final request usage."""

    def on_program_status_changed(
        self,
        strategy_factors: StrategyFactors[StrategyGlobalFactorsT, StrategyProgramFactorsT],
        event: SchedulingEvent,
    ) -> None:
        """Update decision factors after reasoning/acting status changes."""

    def on_program_paused(
        self,
        strategy_factors: StrategyFactors[StrategyGlobalFactorsT, StrategyProgramFactorsT],
        event: SchedulingEvent,
    ) -> None:
        """Clear or transfer decision factors after a committed pause."""

    def on_program_resumed(
        self,
        strategy_factors: StrategyFactors[StrategyGlobalFactorsT, StrategyProgramFactorsT],
        event: SchedulingEvent,
    ) -> None:
        """Initialize decision factors for a newly resumed service segment."""

    def on_program_marked_for_pause(
        self,
        strategy_factors: StrategyFactors[StrategyGlobalFactorsT, StrategyProgramFactorsT],
        event: SchedulingEvent,
    ) -> None:
        """Update decision factors after delayed pause is committed."""

    def on_program_released(
        self,
        strategy_factors: StrategyFactors[StrategyGlobalFactorsT, StrategyProgramFactorsT],
        event: SchedulingEvent,
    ) -> None:
        """Remove per-Program decision factors after terminal release."""
