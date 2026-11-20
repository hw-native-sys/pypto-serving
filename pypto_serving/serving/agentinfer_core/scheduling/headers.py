# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
# Modified by PyPTO Serving: internal imports use the bundled scheduling package.

"""Shared Claude request header parsing and Router forwarding policy."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

CLAUDE_SESSION_HEADER = "X-Claude-Code-Session-Id"
CLAUDE_AGENT_HEADER = "X-Claude-Code-Agent-Id"
DSH_SESSION_HEADER = "X-DeepSeek-Harness-Session-Id"


@dataclass(frozen=True)
class AgentRequestIdentity:
    """Represent agent actor identity observed at the benchmark boundary."""

    session_id: str | None
    actor_id: str
    actor_role: Literal["lead", "subagent", "unknown"]


def _value(headers: Mapping[str, str], name: str) -> str | None:
    wanted = name.lower()
    value = next((value for key, value in headers.items() if key.lower() == wanted), None)
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def parse_agent_identity(headers: Mapping[str, str]) -> AgentRequestIdentity:
    """Parse supported agent identity headers for benchmark tracing."""

    session_id = _value(headers, CLAUDE_SESSION_HEADER)
    agent_id = _value(headers, CLAUDE_AGENT_HEADER)
    if agent_id:
        return AgentRequestIdentity(session_id, agent_id, "subagent")
    if session_id:
        return AgentRequestIdentity(session_id, "lead", "lead")
    dsh_session_id = _value(headers, DSH_SESSION_HEADER)
    if dsh_session_id:
        # Degraded-path fallback: DSH emits this header on subagent sessions
        # too, but the header alone cannot distinguish roles. Canonical
        # vllm_xargs.agentic_context lineage normally wins before this runs.
        return AgentRequestIdentity(dsh_session_id, "lead", "lead")
    return AgentRequestIdentity(None, "unknown", "unknown")


def select_router_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Select non-empty Claude and DSH identity headers for upstream forwarding."""

    selected = {}
    for name in (CLAUDE_SESSION_HEADER, CLAUDE_AGENT_HEADER, DSH_SESSION_HEADER):
        if value := _value(headers, name):
            selected[name] = value
    return selected
