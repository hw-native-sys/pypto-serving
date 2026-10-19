# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
# Modified by PyPTO Serving: internal imports use the bundled scheduling package.
"""Mutable Program records owned by the unified Scheduler runtime.

``RuntimeProgram`` is the sole mutable representation of one live Program generation. It stores integration and
lifecycle facts needed to build the immutable ``ProgramView`` consumed by scheduling strategies. Decision factors,
waiting primitives, locks, and backend telemetry are deliberately excluded; they belong to the strategy or host
Adapter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from pypto_serving.serving.agentinfer_core.scheduling.domain import (
    ProgramRef,
    ProgramState,
    ProgramStatus,
    ProgramTokenObservation,
    ProgramView,
)


@dataclass
class RuntimeProgram:
    """Mutable core state for one exact logical Program incarnation.

    Args:
        ref: Generation-safe Program identity.
        state: Program-level scheduling eligibility.
        status: Current model-service or acting status.
        tokens: Latest host or backend token observation.
        backend_id: Backend currently assigned to an active Program.
        task_id: Optional task grouping used for task-aware scheduling.
        parent_program_id: Optional stable parent Program id.
        blocks_parent: Whether this Program prevents its parent from progressing.
        blocked_on_child: Whether at least one live child blocks this Program.
        expected_resume: Lifecycle hint used by future physical KV management.
        agent_role: Optional upstream role used for diagnostics only.
        spawn_reason: Optional open-ended upstream agent type or spawn reason.
        step_count: Successfully completed model requests.
        wait_started_at_monotonic_s: Start of the current Scheduler waiting interval.
        marked_for_pause: Whether completion of an in-flight request must pause the Program.
        release_after_inflight: Whether an overlapping completed request observed terminal lifecycle state.
        blocking_children: Exact live child generations currently blocking this Program.
    """

    ref: ProgramRef
    state: ProgramState
    status: ProgramStatus
    tokens: ProgramTokenObservation
    backend_id: str | None = None
    task_id: str | None = None
    parent_program_id: str | None = None
    blocks_parent: bool = False
    blocked_on_child: bool = False
    expected_resume: bool = True
    agent_role: str | None = None
    spawn_reason: str | None = None
    step_count: int = 0
    wait_started_at_monotonic_s: float | None = None
    marked_for_pause: bool = False
    release_after_inflight: bool = False
    blocking_children: set[ProgramRef] = field(default_factory=set)

    def __post_init__(self) -> None:
        """Reject invalid counters and incomplete blocking relationships."""
        if self.step_count < 0:
            raise ValueError("program step_count must be non-negative")
        if self.wait_started_at_monotonic_s is not None and (
            not math.isfinite(self.wait_started_at_monotonic_s) or self.wait_started_at_monotonic_s < 0
        ):
            raise ValueError("program wait start must be finite and non-negative")
        if self.blocks_parent and self.parent_program_id is None:
            raise ValueError("blocks_parent requires parent_program_id")

    def to_view(self) -> ProgramView:
        """Return the immutable scheduler-visible projection of this Program."""
        return ProgramView(
            ref=self.ref,
            state=self.state,
            status=self.status,
            tokens=self.tokens,
            backend_id=self.backend_id,
            task_id=self.task_id,
            parent_program_id=self.parent_program_id,
            blocked_on_child=self.blocked_on_child,
            expected_resume=self.expected_resume,
            step_count=self.step_count,
            wait_started_at_monotonic_s=self.wait_started_at_monotonic_s,
            marked_for_pause=self.marked_for_pause,
        )
