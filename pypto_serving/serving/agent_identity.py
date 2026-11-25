# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Translate serving request metadata into an admission session."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass


class MetadataError(ValueError):
    """Invalid metadata for an agent session."""


@dataclass(frozen=True)
class AgentIdentity:
    program_id: str
    task_id: str | None = None
    session_id: str | None = None
    agent_id: str | None = None
    parent_program_id: str | None = None
    blocks_parent: bool = False
    expected_resume: bool = True
    agent_role: str | None = None
    spawn_reason: str | None = None
    request_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("program_id", "task_id", "session_id", "agent_id", "parent_program_id", "request_id"):
            value = getattr(self, name)
            if value is not None and not value.strip():
                raise MetadataError(f"{name} must not be empty")
        if self.parent_program_id is None:
            if self.blocks_parent:
                raise MetadataError("blocks_parent requires parent_program_id")
        elif self.parent_program_id == self.program_id:
            raise MetadataError("parent_program_id must differ from program_id")


_CONTEXT_LIMIT = 16 * 1024
_CLIENT_HEADERS = {
    "x-claude-code-session-id": {
        "agent_id": "x-claude-code-agent-id",
        "parent_agent_id": "x-claude-code-parent-agent-id",
    },
    "x-opencode-session-id": {},
    "x-opencode-session": {},
    "x-deepseek-harness-session-id": {},
    "session-id": {"agent_id": "thread-id", "parent_agent_id": "x-codex-parent-thread-id"},
    "x-session-id": {},
}


def parse_agent_identity(
    *,
    headers: Mapping[str, str],
    vllm_xargs: dict | None = None,
    agent_hint: dict | None = None,
) -> AgentIdentity | None:
    """Apply metadata in request priority order."""
    context = _read_context(vllm_xargs)
    if context is None and agent_hint is not None:
        return _from_hint(agent_hint)
    return _from_context(context or {}, _client_fields(headers))


def _read_context(xargs: dict | None) -> dict | None:
    if xargs is None:
        return None
    if not isinstance(xargs, dict):
        raise MetadataError("vllm_xargs must be an object")
    context = xargs.get("agentic_context")
    if context is None:
        return None
    if isinstance(context, str):
        try:
            if len(context.encode("utf-8")) > _CONTEXT_LIMIT:
                raise ValueError("metadata exceeds size limit")
            context = json.loads(context)
        except (ValueError, RecursionError) as exc:
            raise MetadataError("vllm_xargs.agentic_context must contain a bounded JSON object") from exc
    if not isinstance(context, dict):
        raise MetadataError("vllm_xargs.agentic_context must be an object")
    return context


def _client_fields(headers: Mapping[str, str]) -> dict[str, str]:
    values = {name.lower(): value.strip() for name, value in headers.items() if value.strip()}
    for session_header, related in _CLIENT_HEADERS.items():
        if session_header in values:
            fields = {field: values[name] for field, name in related.items() if name in values}
            fields["session_id"] = values[session_header]
            return fields
    return {}


def _from_hint(hint: dict) -> AgentIdentity:
    if not isinstance(hint, dict):
        raise MetadataError("agent_hint must be an object")
    session = _text(hint, "session_id")
    if session is None:
        raise MetadataError("agent_hint.session_id is required")
    parent = _text(hint, "parent_session_id")
    blocking = _flag(hint, "blocks_parent", default=True)
    if parent is None and blocking:
        raise MetadataError("blocking agent_hint requires parent_session_id")
    return AgentIdentity(
        program_id=session,
        session_id=session,
        parent_program_id=parent,
        blocks_parent=blocking,
        expected_resume=_flag(hint, "expected_resume", default=not blocking),
    )


def _from_context(body: dict, client: dict[str, str]) -> AgentIdentity | None:
    session = _text(body, "session_id") or client.get("session_id")
    explicit_program = _text(body, "program_id")
    task = _text(body, "task_id")
    if task is None and (not explicit_program or "task_id" not in body):
        task = session

    member = _text(body, "agent_id") or client.get("agent_id")
    member = _member_name(task, member or ("lead" if task is not None else None), "agent_id")
    role = _text(body, "agent_role")
    if role is None and member is not None:
        role = "lead" if member == "lead" else "subagent"
    program = explicit_program
    if program is None:
        if task is None or member is None:
            return None
        program = _in_task(task, member)

    parent = _text(body, "parent_program_id", "parent_id")
    parent_member = _text(body, "parent_agent_id") or client.get("parent_agent_id")
    parent_member = _member_name(task, parent_member, "parent_agent_id")
    is_subagent = role == "subagent"
    if parent is None:
        parent_member = parent_member or ("lead" if is_subagent else None)
        if parent_member is not None:
            namespace = program.removesuffix(f":{member}") if member else program
            if namespace != program:
                parent = f"{namespace}:{parent_member}"
            elif task is not None:
                parent = _in_task(task, parent_member)
    blocking = _flag(body, "blocks_parent", "blocking_parent", default=is_subagent)
    return AgentIdentity(
        program_id=program,
        task_id=task,
        session_id=session,
        agent_id=member,
        parent_program_id=parent,
        blocks_parent=blocking,
        agent_role=role,
        expected_resume=_flag(body, "expected_resume", default=not (blocking and is_subagent)),
        spawn_reason=_text(body, "spawn_reason", "subagent_type", "agent_type"),
        request_id=_text(body, "request_id"),
    )


def _member_name(task: str | None, value: str | None, field: str) -> str | None:
    if task is None or value is None:
        return value
    member = value.removeprefix(f"{task}:")
    if not member:
        raise MetadataError(f"{field} must include an agent component")
    return member


def _in_task(task: str, member: str) -> str:
    return member if member.startswith(f"{task}:") else f"{task}:{member}"


def _text(data: dict, *fields: str) -> str | None:
    for name in fields:
        value = data.get(name)
        if value is not None:
            if type(value) not in (str, int):
                raise MetadataError(f"{name} must be a string-compatible identifier")
            value = str(value).strip()
            if not value:
                raise MetadataError(f"{name} must not be empty")
            return value
    return None


def _flag(data: dict, *fields: str, default: bool) -> bool:
    for name in fields:
        value = data.get(name)
        if value is not None:
            if not isinstance(value, bool):
                raise MetadataError(f"{name} must be a boolean")
            return value
    return default
