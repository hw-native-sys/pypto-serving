# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Track live agent sessions within a serving replica."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from itertools import count

from pypto_serving.serving.agentinfer.agent_identity import AgentIdentity
from pypto_serving.serving.agentinfer.agent_policy_state import (
    ProgramRef,
    ProgramState,
    ProgramStatus,
    ProgramTokenObservation,
    ProgramView,
)


@dataclass
class AgentSession:
    ref: ProgramRef
    tokens: ProgramTokenObservation
    state: ProgramState = ProgramState.PAUSED
    status: ProgramStatus = ProgramStatus.REASONING
    backend_id: str | None = None
    task_id: str | None = None
    parent_program_id: str | None = None
    blocks_parent: bool = False
    blocked_on_child: bool = False
    expected_resume: bool = True
    step_count: int = 0
    marked_for_pause: bool = False

    def to_view(self) -> ProgramView:
        return ProgramView(**{name: getattr(self, name) for name in _VIEW_FIELDS})


_VIEW_FIELDS = tuple(
    name for name in ProgramView.__dataclass_fields__ if name != "wait_started_at_monotonic_s"
)


class AgentSessionRegistry:
    """Own live sessions and update parent wait state when child links change."""

    def __init__(self) -> None:
        self._by_id: dict[str, AgentSession] = {}
        self._generation = count()
        self._blockers: dict[str, set[ProgramRef]] = {}

    @property
    def has_programs(self) -> bool:
        return bool(self._by_id)

    def current_ref(self, program_id: str) -> ProgramRef | None:
        session = self._by_id.get(program_id)
        return session.ref if session is not None else None

    def get(self, ref: ProgramRef) -> AgentSession | None:
        session = self._by_id.get(ref.program_id)
        if session is None or session.ref != ref:
            return None
        return session

    def require(self, ref: ProgramRef) -> AgentSession:
        session = self.get(ref)
        if session is None:
            raise LookupError(f"Agent session {ref.program_id}@{ref.generation} is no longer live")
        return session

    def views(self) -> tuple[ProgramView, ...]:
        return tuple(session.to_view() for _, session in sorted(self._by_id.items()))

    def materialize(self, identity: AgentIdentity) -> tuple[AgentSession, bool]:
        session = self._by_id.get(identity.program_id)
        created = session is None
        if session is None:
            session = AgentSession(
                ref=ProgramRef(identity.program_id, next(self._generation)),
                tokens=ProgramTokenObservation(estimated_context_tokens=0),
            )
            self._by_id[identity.program_id] = session
            self._sync_blockers(identity.program_id)
        if identity.task_id is not None:
            if session.task_id is not None and session.task_id != identity.task_id:
                raise ValueError("task_id cannot change within one Program generation")
            session.task_id = identity.task_id
        previous_parent = session.parent_program_id if session.blocks_parent else None
        session.parent_program_id = identity.parent_program_id
        session.blocks_parent = identity.blocks_parent
        session.expected_resume = identity.expected_resume
        parent = identity.parent_program_id if identity.blocks_parent else None
        if previous_parent != parent:
            if previous_parent is not None:
                self._remove_blocker(previous_parent, session.ref)
            if parent is not None:
                self._blockers.setdefault(parent, set()).add(session.ref)
                self._sync_blockers(parent)
        return session, created

    def mark_request_started(self, ref: ProgramRef) -> ProgramStatus:
        return self._set_phase(self.require(ref), ProgramStatus.REASONING)

    def record_completion(self, ref: ProgramRef, *, total_tokens: int) -> ProgramStatus:
        session = self.require(ref)
        session.tokens = replace(
            session.tokens,
            estimated_context_tokens=max(0, total_tokens),
            observed_at_monotonic_s=time.monotonic(),
        )
        session.step_count += 1
        return self._set_phase(session, ProgramStatus.ACTING)

    def release(self, ref: ProgramRef) -> AgentSession | None:
        session = self.get(ref)
        if session is None:
            return None

        del self._by_id[ref.program_id]
        if session.blocks_parent and session.parent_program_id is not None:
            self._remove_blocker(session.parent_program_id, ref)
        for child_ref in self._blockers.pop(ref.program_id, ()):
            child = self.require(child_ref)
            child.parent_program_id = None
            child.blocks_parent = False

        session.blocked_on_child = False
        session.state = ProgramState.TERMINATED
        session.backend_id = None
        return session

    @staticmethod
    def _set_phase(session: AgentSession, phase: ProgramStatus) -> ProgramStatus:
        before = session.status
        session.status = phase
        return before

    def _remove_blocker(self, parent: str, child: ProgramRef) -> None:
        blockers = self._blockers[parent]
        blockers.remove(child)
        if not blockers:
            del self._blockers[parent]
        self._sync_blockers(parent)

    def _sync_blockers(self, parent: str) -> None:
        session = self._by_id.get(parent)
        if session is not None:
            session.blocked_on_child = parent in self._blockers
