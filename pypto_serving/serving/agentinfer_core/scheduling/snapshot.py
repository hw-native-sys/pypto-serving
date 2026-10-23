# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
# Modified by PyPTO Serving: internal imports use the bundled scheduling package.
"""Immutable scheduling snapshots passed to stateful strategy Hooks.

``SchedulingSnapshot`` is the complete read model used for one admission, resume, or capacity-repair decision. It
combines the latest backend KV-token capacity with immutable Program views, waiting order, and decision time while
exposing no live host runtime objects. Decision factors are passed separately under the scheduling lock because they may
change after an accepted transition. The snapshot stores only source facts: remaining capacity, reserve, reclaimable
tokens, score, and other policy-derived values must be computed by the concrete strategy.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from pypto_serving.serving.agentinfer_core.scheduling.domain import ProgramRef, ProgramView


@dataclass(frozen=True)
class SchedulingSnapshot:
    """Complete immutable input observed by one strategy Hook invocation.

    Args:
        observed_at_monotonic_s: Monotonic timestamp used by waiting and decay formulas.
        backend_id: Stable identifier of the configured backend.
        total_kv_tokens: Current GPU KV token capacity reported by that backend.
        native_used_kv_tokens: Optional vLLM block-pool use for native running reasoning requests observed by Adapter.
        native_waiting_kv_tokens: Optional logical token estimate for native vLLM waiting requests.
        request_pool_waiting_requests: Requests currently retained before downstream admission.
        programs: Scheduler-visible facts for all live programs.
        waiting_programs: Ordered program references in the program-level waiting pool.
    """

    observed_at_monotonic_s: float
    backend_id: str
    total_kv_tokens: int | None
    programs: tuple[ProgramView, ...]
    waiting_programs: tuple[ProgramRef, ...]
    native_used_kv_tokens: int | None = None
    native_waiting_kv_tokens: int | None = None
    request_pool_waiting_requests: int = 0
    _programs_by_ref: Mapping[ProgramRef, ProgramView] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate time, capacity, unique Program refs, and waiting-pool membership."""
        object.__setattr__(self, "programs", tuple(self.programs))
        object.__setattr__(self, "waiting_programs", tuple(self.waiting_programs))
        if not math.isfinite(self.observed_at_monotonic_s) or self.observed_at_monotonic_s < 0:
            raise ValueError("snapshot timestamp must be finite and non-negative")
        if not self.backend_id:
            raise ValueError("snapshot backend_id must not be empty")
        if self.total_kv_tokens is not None and self.total_kv_tokens < 0:
            raise ValueError("snapshot total_kv_tokens must be non-negative")
        if self.native_used_kv_tokens is not None and self.native_used_kv_tokens < 0:
            raise ValueError("snapshot native_used_kv_tokens must be non-negative")
        if self.native_waiting_kv_tokens is not None and self.native_waiting_kv_tokens < 0:
            raise ValueError("snapshot native_waiting_kv_tokens must be non-negative")
        if self.request_pool_waiting_requests < 0:
            raise ValueError("snapshot request_pool_waiting_requests must be non-negative")
        if (
            self.total_kv_tokens is not None
            and self.native_used_kv_tokens is not None
            and self.native_used_kv_tokens > self.total_kv_tokens
        ):
            raise ValueError("snapshot native_used_kv_tokens must not exceed total_kv_tokens")
        programs_by_ref: dict[ProgramRef, ProgramView] = {}
        for program in self.programs:
            if program.ref in programs_by_ref:
                raise ValueError("snapshot contains duplicate program references")
            programs_by_ref[program.ref] = program
        if any(ref not in programs_by_ref for ref in self.waiting_programs):
            raise ValueError("waiting pool references a program absent from the snapshot")
        if len(self.waiting_programs) != len(set(self.waiting_programs)):
            raise ValueError("waiting pool contains duplicate program references")
        object.__setattr__(self, "_programs_by_ref", MappingProxyType(programs_by_ref))

    def program(self, ref: ProgramRef) -> ProgramView | None:
        """Return an exact program generation from this snapshot, if present.

        Args:
            ref: Generation-safe program reference to look up.
        """
        return self._programs_by_ref.get(ref)
