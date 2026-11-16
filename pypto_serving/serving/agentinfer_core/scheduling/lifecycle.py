# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
# Modified by PyPTO Serving: internal imports use the bundled scheduling package.
"""Protocol-neutral response lifecycle facts accepted by the Scheduler.

API adapters translate wire-specific terminal markers into this closed vocabulary. Scheduling policies never parse
OpenAI finish reasons, Anthropic stop reasons, response bodies, or transport success values.
"""

from __future__ import annotations

from enum import Enum


class ProgramLifecycle(str, Enum):
    """Whether one completed response proves that its Program will continue."""

    UNKNOWN = "unknown"
    CONTINUE = "continue"
    TERMINAL = "terminal"
