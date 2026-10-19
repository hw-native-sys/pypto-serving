# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
# Modified by PyPTO Serving: internal imports use the bundled scheduling package.
"""Normalize supported request metadata into one host-neutral Agent identity.

The module is the shared metadata boundary for AgentCache and AgentRouter. It parses explicit AgentInfer fields and
known framework aliases, but performs no HTTP I/O, request forwarding, Program mutation, or scheduling decision.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject = dict[str, JsonValue]
JsonMapping = Mapping[str, JsonValue]

_MAX_AGENTIC_CONTEXT_BYTES = 16 * 1024


class MetadataError(ValueError):
    """Raised when a supported identity input is malformed."""


@dataclass(frozen=True)
class AgentIdentity:
    """Canonical Program identity and workflow hints produced by a host boundary.

    Args:
        program_id: Stable serving-side key for one logical Program.
        task_id: Optional task group shared by related Programs.
        session_id: Optional upstream session identity.
        agent_id: Optional logical agent identity within the task.
        parent_program_id: Optional stable parent Program key.
        blocks_parent: Whether this Program blocks the parent from progressing.
        expected_resume: Whether another request is expected after this round.
        agent_role: Optional explicit or inferred lead/subagent role.
        spawn_reason: Optional open-ended agent type or spawn reason.
        request_id: Optional upstream correlation id; not part of Program identity.
    """

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
        """Reject empty identities and incomplete blocking edges."""
        identifiers = {
            "program_id": self.program_id,
            "task_id": self.task_id,
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "parent_program_id": self.parent_program_id,
            "request_id": self.request_id,
        }
        for name, value in identifiers.items():
            if value is not None and not value.strip():
                raise MetadataError(f"{name} must not be empty")
        if self.parent_program_id == self.program_id:
            raise MetadataError("parent_program_id must differ from program_id")
        if self.blocks_parent and self.parent_program_id is None:
            raise MetadataError("blocks_parent requires parent_program_id")

    def as_context(self) -> dict[str, str | bool]:
        """Return a JSON-compatible canonical context for transport through host extension fields."""
        values: dict[str, str | bool | None] = {
            "program_id": self.program_id,
            "task_id": self.task_id,
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "parent_program_id": self.parent_program_id,
            "blocks_parent": self.blocks_parent,
            "expected_resume": self.expected_resume,
            "agent_role": self.agent_role,
            "spawn_reason": self.spawn_reason,
            "request_id": self.request_id,
        }
        return {key: value for key, value in values.items() if value is not None}


_FRAMEWORK_HEADER_SCHEMAS = (
    ("x-claude-code-session-id", "x-claude-code-agent-id", "x-claude-code-parent-agent-id"),
    ("session-id", "thread-id", None),
    ("x-session-id", None, None),
)


def parse_agent_identity(
    *,
    vllm_xargs: JsonValue = None,
    agent_hint: JsonValue = None,
    headers: Mapping[str, str],
) -> AgentIdentity | None:
    """Normalize explicit AgentInfer hints or infer a supported framework identity.

    Args:
        vllm_xargs: Optional vLLM extension object containing canonical ``agentic_context``.
        agent_hint: Optional session-oriented framework hint object.
        headers: Upstream headers matched case-insensitively.

    Returns:
        Canonical identity, or ``None`` when no supported Program identity exists.

    Raises:
        MetadataError: Metadata in a supported input is malformed.
    """
    context = _vllm_agentic_context(vllm_xargs)
    if context is None:
        agent_hint_identity = _agent_hint_identity(agent_hint)
        if agent_hint_identity is not None:
            return agent_hint_identity
    context = context or {}
    normalized_headers = _normalized_headers(headers)
    header_session_id, header_agent_id, header_parent_agent_id = _framework_identity_headers(normalized_headers)

    explicit_program_id = _context_string(context, "program_id")
    body_task_id = _context_string(context, "task_id")
    body_session_id = _context_string(context, "session_id")
    session_id = body_session_id or header_session_id
    task_id = (
        body_task_id
        if body_task_id is not None or (explicit_program_id is not None and "task_id" in context)
        else session_id
    )

    agent_id = _context_string(context, "agent_id") or header_agent_id
    if agent_id is None and task_id is not None:
        agent_id = "lead"
    agent_id = _agent_component(task_id, agent_id, "agent_id")
    agent_role = _context_string(context, "agent_role")
    if agent_role is None and agent_id is not None:
        agent_role = "lead" if agent_id == "lead" else "subagent"

    program_id = explicit_program_id or _program_id(task_id, agent_id)
    if program_id is None:
        return None

    parent_program_id = _context_string(context, "parent_program_id", "parent_id")
    parent_agent_id = _context_string(context, "parent_agent_id") or header_parent_agent_id
    parent_agent_id = _agent_component(task_id, parent_agent_id, "parent_agent_id")
    if parent_program_id is None and parent_agent_id is not None:
        parent_program_id = _related_program_id(program_id, agent_id, parent_agent_id, task_id)
    if parent_program_id is None and agent_role == "subagent":
        parent_program_id = _related_program_id(program_id, agent_id, "lead", task_id)

    blocks_parent = _context_bool(context, "blocks_parent", "blocking_parent")
    if blocks_parent is None:
        blocks_parent = agent_role == "subagent"

    expected_resume = _context_bool(context, "expected_resume")
    if expected_resume is None:
        expected_resume = not (blocks_parent and agent_role == "subagent")

    return AgentIdentity(
        program_id=program_id,
        task_id=task_id,
        session_id=session_id,
        agent_id=agent_id,
        parent_program_id=parent_program_id,
        blocks_parent=blocks_parent,
        expected_resume=expected_resume,
        agent_role=agent_role,
        spawn_reason=_context_string(context, "spawn_reason", "subagent_type", "agent_type"),
        request_id=_context_string(context, "request_id"),
    )


def encode_agent_identity(identity: AgentIdentity) -> str:
    """Serialize one resolved identity for transport through vLLM scalar extension fields."""
    context: dict[str, JsonValue] = identity.as_context()
    context["task_id"] = identity.task_id
    return json.dumps(context, separators=(",", ":"))


def _agent_hint_identity(raw: JsonValue) -> AgentIdentity | None:
    """Map the session-oriented ``agent_hint`` body extension without inventing a task identity."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise MetadataError("agent_hint must be an object")
    session_id = _context_string(raw, "session_id")
    if session_id is None:
        raise MetadataError("agent_hint.session_id is required")
    parent_program_id = _context_string(raw, "parent_session_id")
    blocks_parent = _context_bool(raw, "blocks_parent")
    if blocks_parent is None:
        blocks_parent = True
    if blocks_parent and parent_program_id is None:
        raise MetadataError("blocking agent_hint requires parent_session_id")
    expected_resume = _context_bool(raw, "expected_resume")
    if expected_resume is None:
        expected_resume = not blocks_parent
    return AgentIdentity(
        program_id=session_id,
        task_id=None,
        session_id=session_id,
        parent_program_id=parent_program_id,
        blocks_parent=blocks_parent,
        expected_resume=expected_resume,
    )


def _vllm_agentic_context(raw_xargs: JsonValue) -> JsonObject | None:
    """Decode the sole canonical body contract at ``vllm_xargs.agentic_context``."""
    if raw_xargs is None:
        return None
    if not isinstance(raw_xargs, dict):
        raise MetadataError("vllm_xargs must be an object")
    raw = raw_xargs.get("agentic_context")
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            encoded_size = len(raw.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise MetadataError("vllm_xargs.agentic_context must contain a bounded JSON object") from exc
        if encoded_size > _MAX_AGENTIC_CONTEXT_BYTES:
            raise MetadataError("vllm_xargs.agentic_context must contain a bounded JSON object")
        try:
            raw = cast(JsonValue, json.loads(raw))
        except (ValueError, RecursionError) as exc:
            raise MetadataError("vllm_xargs.agentic_context must contain a bounded JSON object") from exc
    if not isinstance(raw, dict):
        raise MetadataError("vllm_xargs.agentic_context must be an object")
    return dict(raw)


def _normalized_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Normalize non-empty request headers for deterministic lookup."""
    return {key.lower(): value.strip() for key, value in headers.items() if value.strip()}


def _framework_identity_headers(headers: Mapping[str, str]) -> tuple[str | None, str | None, str | None]:
    """Select one framework by session header and return only that framework's identity fields."""
    for session_key, agent_key, parent_agent_key in _FRAMEWORK_HEADER_SCHEMAS:
        session_id = headers.get(session_key)
        if session_id is None:
            continue
        agent_id = headers.get(agent_key) if agent_key is not None else None
        parent_agent_id = headers.get(parent_agent_key) if parent_agent_key is not None else None
        return session_id, agent_id, parent_agent_id
    return None, None, None


def _context_string(context: JsonMapping, *keys: str) -> str | None:
    """Return the first supplied string-compatible metadata value."""
    for key in keys:
        if key not in context:
            continue
        value = _optional_string(context[key], key)
        if value is not None:
            return value
    return None


def _context_bool(context: JsonMapping, *keys: str) -> bool | None:
    """Return the first supplied boolean without truthiness coercion."""
    for key in keys:
        if key not in context:
            continue
        value = context[key]
        if value is None:
            continue
        if not isinstance(value, bool):
            raise MetadataError(f"{key} must be a boolean")
        return value
    return None


def _agent_component(task_id: str | None, agent_id: str | None, location: str) -> str | None:
    """Remove one exact task qualification before role and relationship inference."""
    if task_id is None or agent_id is None:
        return agent_id
    prefix = f"{task_id}:"
    if not agent_id.startswith(prefix):
        return agent_id
    component = agent_id[len(prefix) :]
    if not component:
        raise MetadataError(f"{location} must include an agent component")
    return component


def _program_id(task_id: str | None, agent_id: str | None) -> str | None:
    """Build a canonical Program id without duplicating an existing task prefix."""
    if task_id is None or agent_id is None:
        return None
    prefix = f"{task_id}:"
    return agent_id if agent_id.startswith(prefix) else f"{prefix}{agent_id}"


def _related_program_id(
    program_id: str,
    agent_id: str | None,
    related_agent_id: str,
    task_id: str | None,
) -> str | None:
    """Preserve an explicit Program namespace when deriving a related agent id."""
    suffix = f":{agent_id}" if agent_id is not None else None
    if suffix is not None and program_id.endswith(suffix):
        return f"{program_id[: -len(suffix)]}:{related_agent_id}"
    return _program_id(task_id, related_agent_id)


def _optional_string(value: JsonValue, location: str) -> str | None:
    """Normalize one optional string-compatible identifier."""
    if value is None:
        return None
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        raise MetadataError(f"{location} must be a string-compatible identifier")
    normalized = str(value).strip()
    if not normalized:
        raise MetadataError(f"{location} must not be empty")
    return normalized
