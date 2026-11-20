# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
# Modified by PyPTO Serving: internal imports use the bundled scheduling package.
"""Synchronous forwarding-gate outcome returned by the admission Hook.

Admission is the only scheduling Hook whose caller needs an immediate aggregate disposition: the host runtime must either
open the request's forwarding gate or keep it in the waiting pool. Resume and capacity-repair Hooks express their work
through individual transition results, committed scheduling events, and strategy diagnostics, so they do not require
parallel outcome types. ``AdmissionOutcome`` also carries the capacity inputs and exact Program transitions needed to
explain how the Hook reached its final forwarding decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pypto_serving.serving.agentinfer_core.scheduling.domain import ProgramRef


class AdmissionDisposition(str, Enum):
    """Final program-level disposition of one admission Hook."""

    ADMITTED = "admitted"
    QUEUED = "queued"


@dataclass(frozen=True)
class AdmissionOutcome:
    """Summary returned after an admission Hook completes its transitions.

    Args:
        disposition: Whether the host runtime may forward the candidate or must keep it queued.
        reason: Stable machine-readable reason for the outcome.
        required_tokens: Candidate capacity used by the strategy.
        reserve_tokens: Additional policy headroom used by the strategy.
        deficit_tokens: Capacity deficit observed before any victim transitions.
        transitioned_programs: Programs changed while reaching this outcome.
    """

    disposition: AdmissionDisposition
    reason: str
    required_tokens: int = 0
    reserve_tokens: int = 0
    deficit_tokens: int = 0
    transitioned_programs: tuple[ProgramRef, ...] = ()

    def __post_init__(self) -> None:
        """Validate outcome reasons and non-negative capacity quantities."""
        object.__setattr__(self, "transitioned_programs", tuple(self.transitioned_programs))
        if not self.reason:
            raise ValueError("admission outcome reason must not be empty")
        if min(self.required_tokens, self.reserve_tokens, self.deficit_tokens) < 0:
            raise ValueError("admission token quantities must be non-negative")
