# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
# Modified by PyPTO Serving: internal imports use the bundled scheduling package.
"""Generation-safe Program identity, task membership, and blocking-edge registry.

``ProgramRegistry`` owns live ``RuntimeProgram`` records and the indexes required to materialize immutable scheduling
views. It deliberately does not own a waiting queue, forwarding gate, backend capacity, strategy state, or lock.
The caller serializes mutations with the Scheduler mutation boundary, while ``ProgramRef`` protects each mutation
from stale responses and reused external ids.
"""

from __future__ import annotations

import time
from dataclasses import replace

from pypto_serving.serving.agentinfer_core.scheduling.domain import (
    ProgramRef,
    ProgramState,
    ProgramStatus,
    ProgramTokenObservation,
    ProgramView,
)
from pypto_serving.serving.agentinfer_core.scheduling.identity import AgentIdentity
from pypto_serving.serving.agentinfer_core.scheduling.program_runtime import RuntimeProgram


class StaleProgramReferenceError(LookupError):
    """Raised when a mutation targets a missing or superseded Program generation."""


class ProgramRegistry:
    """Mutable owner of live Program records and task relationships.

    The registry is intentionally synchronous. The hosting ``ProgramScheduler`` Adapter serializes registry
    mutation, snapshot construction, strategy sidecars, and state transitions under one mutation boundary, so adding
    an internal lock or version would duplicate synchronization without improving generation safety.
    """

    def __init__(self) -> None:
        self._programs: dict[str, RuntimeProgram] = {}
        self._next_generation = 0
        self._task_programs: dict[str, set[ProgramRef]] = {}
        self._children_by_parent_id: dict[str, set[ProgramRef]] = {}

    def __bool__(self) -> bool:
        """Return whether the registry owns any live Program generation."""
        return bool(self._programs)

    @property
    def has_programs(self) -> bool:
        """Return in O(1) whether any live Program generation is registered."""
        return bool(self._programs)

    def materialize(
        self,
        metadata: AgentIdentity,
        *,
        initial_state: ProgramState = ProgramState.PAUSED,
    ) -> tuple[RuntimeProgram, bool]:
        """Return the current Program or create its next generation.

        Args:
            metadata: Normalized identity and lifecycle hints from the endpoint adapter.
            initial_state: Forwarding state used only for a newly created generation.

        Returns:
            The live Program and whether this call created a new generation.
        """
        current = self._programs.get(metadata.program_id)
        if current is not None:
            self._apply_metadata(current, metadata)
            return current, False
        generation = self._next_generation
        self._next_generation += 1
        program = RuntimeProgram(
            ref=ProgramRef(metadata.program_id, generation),
            state=initial_state,
            status=ProgramStatus.REASONING,
            tokens=ProgramTokenObservation(estimated_context_tokens=0),
            task_id=metadata.task_id,
            parent_program_id=metadata.parent_program_id,
            blocks_parent=metadata.blocks_parent,
            expected_resume=metadata.expected_resume,
            agent_role=metadata.agent_role,
            spawn_reason=metadata.spawn_reason,
        )
        self._programs[metadata.program_id] = program
        self._attach_task(program)
        self._attach_parent_edge(program)
        self._refresh_parent(metadata.program_id)
        return program, True

    def current_ref(self, program_id: str) -> ProgramRef | None:
        """Return the current exact generation for a stable Program id."""
        program = self._programs.get(program_id)
        return program.ref if program is not None else None

    def get(self, ref: ProgramRef) -> RuntimeProgram | None:
        """Return a Program only if ``ref`` still names its live generation."""
        program = self._programs.get(ref.program_id)
        return program if program is not None and program.ref == ref else None

    def require(self, ref: ProgramRef) -> RuntimeProgram:
        """Return an exact live Program or reject a stale generation."""
        program = self.get(ref)
        if program is None:
            raise StaleProgramReferenceError(f"Program reference is stale: {ref.program_id}@{ref.generation}")
        return program

    def views(self) -> tuple[ProgramView, ...]:
        """Return deterministic immutable views for all live Programs."""
        return tuple(self._programs[program_id].to_view() for program_id in sorted(self._programs))

    def task_programs(self, task_id: str) -> tuple[ProgramRef, ...]:
        """Return current Program generations belonging to one task."""
        refs = self._task_programs.get(task_id, set())
        return tuple(sorted(ref for ref in refs if self.get(ref) is not None))

    def children(self, parent: ProgramRef) -> tuple[ProgramRef, ...]:
        """Return current blocking children for one exact live parent."""
        self.require(parent)
        refs = self._children_by_parent_id.get(parent.program_id, set())
        return tuple(sorted(ref for ref in refs if self._is_live_child_of(ref, parent.program_id)))

    def mark_request_started(self, ref: ProgramRef) -> ProgramStatus:
        """Set a Program to reasoning and return its previous status."""
        program = self.require(ref)
        previous = program.status
        program.status = ProgramStatus.REASONING
        return previous

    def update_tokens(self, ref: ProgramRef, tokens: ProgramTokenObservation) -> None:
        """Replace the token observation for one exact Program generation."""
        self.require(ref).tokens = tokens

    def record_completion(
        self,
        ref: ProgramRef,
        *,
        total_tokens: int | None,
        observed_at_monotonic_s: float | None = None,
    ) -> ProgramStatus:
        """Apply trustworthy final usage and return the previous status.

        Args:
            ref: Exact Program generation that owned the request.
            total_tokens: Final prompt plus completion tokens, when supplied by the backend.
            observed_at_monotonic_s: Deterministic timestamp override used by tests.
        """
        program = self.require(ref)
        previous_status = program.status
        previous = program.tokens
        program.tokens = replace(
            previous,
            estimated_context_tokens=previous.estimated_context_tokens
            if total_tokens is None
            else max(0, total_tokens),
            observed_at_monotonic_s=(time.monotonic() if observed_at_monotonic_s is None else observed_at_monotonic_s),
        )
        program.step_count += 1
        program.status = ProgramStatus.ACTING
        return previous_status

    def release(self, ref: ProgramRef) -> RuntimeProgram | None:
        """Remove one exact Program and all task and dependency indexes idempotently."""
        program = self.get(ref)
        if program is None:
            return None
        self._detach_task(program)
        self._detach_parent_edge(program)
        child_refs = self._children_by_parent_id.pop(ref.program_id, set())
        for child_ref in child_refs:
            child = self.get(child_ref)
            if child is None or child.parent_program_id != ref.program_id:
                continue
            child.parent_program_id = None
            child.blocks_parent = False
        program.blocking_children.clear()
        program.blocked_on_child = False
        program.state = ProgramState.TERMINATED
        program.backend_id = None
        self._programs.pop(ref.program_id)
        return program

    def _apply_metadata(self, program: RuntimeProgram, metadata: AgentIdentity) -> None:
        """Apply compatible hints without changing stable Program or task identity."""
        if metadata.task_id is not None:
            if program.task_id is not None and program.task_id != metadata.task_id:
                raise ValueError("task_id cannot change within one Program generation")
            if program.task_id is None:
                program.task_id = metadata.task_id
                self._attach_task(program)
        relation_changed = False
        if metadata.parent_program_id != program.parent_program_id:
            self._detach_parent_edge(program)
            program.parent_program_id = metadata.parent_program_id
            relation_changed = True
        if metadata.blocks_parent != program.blocks_parent:
            if not metadata.blocks_parent:
                self._detach_parent_edge(program)
            program.blocks_parent = metadata.blocks_parent
            relation_changed = True
        if program.blocks_parent and program.parent_program_id is None:
            raise ValueError("blocks_parent requires parent_program_id")
        if relation_changed:
            self._attach_parent_edge(program)
        program.expected_resume = metadata.expected_resume
        if metadata.agent_role is not None:
            program.agent_role = metadata.agent_role
        if metadata.spawn_reason is not None:
            program.spawn_reason = metadata.spawn_reason

    def _attach_task(self, program: RuntimeProgram) -> None:
        if program.task_id is not None:
            self._task_programs.setdefault(program.task_id, set()).add(program.ref)

    def _detach_task(self, program: RuntimeProgram) -> None:
        if program.task_id is None:
            return
        refs = self._task_programs.get(program.task_id)
        if refs is None:
            return
        refs.discard(program.ref)
        if not refs:
            self._task_programs.pop(program.task_id, None)

    def _attach_parent_edge(self, child: RuntimeProgram) -> None:
        if not child.blocks_parent or child.parent_program_id is None:
            return
        self._children_by_parent_id.setdefault(child.parent_program_id, set()).add(child.ref)
        self._refresh_parent(child.parent_program_id)

    def _detach_parent_edge(self, child: RuntimeProgram) -> None:
        parent_id = child.parent_program_id
        if parent_id is None:
            return
        refs = self._children_by_parent_id.get(parent_id)
        if refs is not None:
            refs.discard(child.ref)
            if not refs:
                self._children_by_parent_id.pop(parent_id, None)
        self._refresh_parent(parent_id)

    def _refresh_parent(self, parent_id: str) -> None:
        parent = self._programs.get(parent_id)
        if parent is None:
            return
        refs = self._children_by_parent_id.get(parent_id, set())
        live = {ref for ref in refs if self._is_live_child_of(ref, parent_id)}
        parent.blocking_children = live
        parent.blocked_on_child = bool(live)

    def _is_live_child_of(self, child_ref: ProgramRef, parent_id: str) -> bool:
        child = self.get(child_ref)
        return child is not None and child.blocks_parent and child.parent_program_id == parent_id
