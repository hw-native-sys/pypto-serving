# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Model-independent output parser contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Sequence


@dataclass(frozen=True)
class OutputParserSpec:
    """Normalized, request-local contract for Serving output parsing."""

    parser_id: str
    initial_state: Literal["content", "reasoning"]
    include_reasoning: bool = True
    tool_choice: Literal["none", "auto", "required"] = "none"
    tool_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.parser_id not in ("deepseek_v4", "deepseek_v41", "qwen3"):
            raise ValueError(f"unsupported output parser {self.parser_id!r}")
        if self.initial_state not in ("content", "reasoning"):
            raise ValueError("output parser initial_state must be content or reasoning")
        if self.tool_choice not in ("none", "auto", "required"):
            raise ValueError("output parser tool_choice must be none, auto, or required")
        if self.tool_choice != "none" and not self.tool_names:
            raise ValueError("enabled tool choice requires tool names")


@dataclass(frozen=True)
class ParsedToolCall:
    """Model-independent function call; incomplete calls are never repaired."""

    id: str
    name: str
    arguments: str
    complete: bool = True


@dataclass(frozen=True)
class ToolCallDelta:
    """Append-only updates to one call, identified by its request-local index."""

    index: int
    id: str | None = None
    name: str | None = None
    arguments: str = ""


@dataclass(frozen=True)
class ParsedDelta:
    """New semantic text released by one incremental parser feed."""

    reasoning: str = ""
    content: str = ""
    tool_call_deltas: tuple[ToolCallDelta, ...] = ()
    # Populated at finalization only, like the final TokenOutput snapshot.
    tool_calls: tuple[ParsedToolCall, ...] = ()


@dataclass(frozen=True)
class ParsedOutput:
    """Complete semantic output for one non-streaming generation request."""

    reasoning: str = ""
    content: str = ""
    tool_calls: tuple[ParsedToolCall, ...] = ()


class OutputParser(Protocol):
    def feed(
        self,
        delta_text: str,
        delta_token_ids: Sequence[int],
    ) -> ParsedDelta: ...

    def finish(self, *, truncated: bool = False) -> ParsedDelta: ...

    def parse_complete(
        self,
        text: str,
        token_ids: Sequence[int],
        *,
        truncated: bool = False,
    ) -> ParsedOutput: ...
