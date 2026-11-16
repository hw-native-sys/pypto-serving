# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
# Modified by PyPTO Serving: internal imports use the bundled scheduling package.
"""Environment-neutral program facts for AgentInfer scheduling strategies.

This module defines the stable vocabulary shared by Router Runtime and every scheduling policy: generation-safe
Program identity, Router-owned forwarding state, reasoning/acting status, token observations, and immutable
``ProgramView`` values. These objects describe facts only; they contain no mutable waiting primitive, Router lock,
capacity formula, policy score, or strategy-specific decision factors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class ProgramState(str, Enum):
    """Whether the scheduling runtime may forward work for a program."""

    ACTIVE = "active"
    PAUSED = "paused"
    TERMINATED = "terminated"


class ProgramStatus(str, Enum):
    """Whether a live program is using model service or acting outside it."""

    REASONING = "reasoning"
    ACTING = "acting"


class SharedPrefixAttribution(str, Enum):
    """Quality of per-program accounting for physically shared prefix tokens."""

    EXACT = "exact"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


class TokenObservationSource(str, Enum):
    """Source that produced a program token observation."""

    ROUTER_ESTIMATE = "router_estimate"
    BACKEND_USAGE = "backend_usage"
    MIXED = "mixed"


@dataclass(frozen=True, order=True)
class ProgramRef:
    """Generation-safe identifier for one logical program incarnation.

    Args:
        program_id: Stable external program identifier.
        generation: Registry-wide creation sequence, including rematerialized external identifiers.
    """

    program_id: str
    generation: int

    def __post_init__(self) -> None:
        """Reject empty identifiers and negative generations."""
        if not self.program_id:
            raise ValueError("program_id must not be empty")
        if self.generation < 0:
            raise ValueError("program generation must be non-negative")


@dataclass(frozen=True)
class ProgramTokenObservation:
    """Estimated and optional physical token footprint for one program.

    Args:
        estimated_context_tokens: Router-side estimate of current context tokens.
        shared_prefix_tokens: Prefix tokens observed as already cached for this Program's current segment.
        shared_prefix_fresh_until_monotonic_s: Deadline before which the runtime retains this shared-prefix observation.
        actual_resident_tokens: Optional backend-observed resident KV tokens.
        actual_allocated_blocks: Optional backend-observed allocated KV blocks.
        block_size_tokens: Token count represented by one observed KV block.
        shared_prefix_attribution: Whether shared-prefix ownership is exact, partial, or unknown.
        source: Integration that produced the strongest observation.
        observed_at_monotonic_s: Monotonic timestamp associated with the observation.
    """

    estimated_context_tokens: int
    shared_prefix_tokens: int = 0
    shared_prefix_fresh_until_monotonic_s: float | None = None
    actual_resident_tokens: int | None = None
    actual_allocated_blocks: int | None = None
    block_size_tokens: int | None = None
    shared_prefix_attribution: SharedPrefixAttribution = SharedPrefixAttribution.UNKNOWN
    source: TokenObservationSource = TokenObservationSource.ROUTER_ESTIMATE
    observed_at_monotonic_s: float | None = None

    def __post_init__(self) -> None:
        """Validate non-negative counts and a positive supplied KV block size."""
        counts = (
            self.estimated_context_tokens,
            self.shared_prefix_tokens,
            self.actual_resident_tokens,
            self.actual_allocated_blocks,
        )
        if any(value is not None and value < 0 for value in counts):
            raise ValueError("program token observations must be non-negative")
        if self.block_size_tokens is not None and self.block_size_tokens <= 0:
            raise ValueError("block_size_tokens must be positive when supplied")
        if self.observed_at_monotonic_s is not None and (
            not math.isfinite(self.observed_at_monotonic_s) or self.observed_at_monotonic_s < 0
        ):
            raise ValueError("observation timestamp must be finite and non-negative")
        if self.shared_prefix_fresh_until_monotonic_s is not None and (
            not math.isfinite(self.shared_prefix_fresh_until_monotonic_s)
            or self.shared_prefix_fresh_until_monotonic_s < 0
        ):
            raise ValueError("shared-prefix freshness deadline must be finite and non-negative")


@dataclass(frozen=True)
class ProgramView:
    """Immutable scheduler-visible facts for one program.

    Args:
        ref: Generation-safe program reference.
        state: Core forwarding lifecycle state owned by the runtime.
        status: Model-service activity status owned by the runtime.
        tokens: Estimated and optional physical token observations.
        backend_id: Backend currently owning the active program, if any.
        task_id: Optional task grouping used by task-aware strategies.
        parent_program_id: Optional parent relationship supplied by an adapter.
        blocked_on_child: Whether the program is known to wait on a blocking child.
        expected_resume: Whether the upstream lifecycle expects later requests.
        step_count: Successfully completed model-service steps.
        wait_started_at_monotonic_s: Start of the current scheduling wait interval.
        marked_for_pause: Whether an in-flight reasoning request will pause on completion.
    """

    ref: ProgramRef
    state: ProgramState
    status: ProgramStatus
    tokens: ProgramTokenObservation
    backend_id: str | None = None
    task_id: str | None = None
    parent_program_id: str | None = None
    blocked_on_child: bool = False
    expected_resume: bool = True
    step_count: int = 0
    wait_started_at_monotonic_s: float | None = None
    marked_for_pause: bool = False

    def __post_init__(self) -> None:
        """Validate counters and monotonic timestamps exposed to strategies."""
        if self.step_count < 0:
            raise ValueError("program step_count must be non-negative")
        if self.wait_started_at_monotonic_s is not None and (
            not math.isfinite(self.wait_started_at_monotonic_s) or self.wait_started_at_monotonic_s < 0
        ):
            raise ValueError("wait_started_at_monotonic_s must be finite and non-negative")
