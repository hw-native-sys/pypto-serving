# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Parse Qwen3's token-delimited JSON function calls."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import replace
from typing import Sequence

from pypto_serving.serving.reasoning.base import (
    DROP_TERMINAL,
    END_TERMINAL,
    START_TERMINAL,
    Segment,
    TextSegment,
    TokenReasoningParser,
)
from pypto_serving.serving.reasoning.types import OutputParserSpec, ParsedDelta, ParsedToolCall, ToolCallDelta


TOOL_START = "<tool_call>"
TOOL_END = "</tool_call>"
_FUNCTION_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}")


class Qwen3ToolParser(TokenReasoningParser):
    """Hold each JSON object until its closing terminal, then publish one call."""

    def __init__(self, tokenizer, spec: OutputParserSpec) -> None:
        super().__init__(tokenizer, spec)
        self._publish_tools = spec.tool_choice != "none"
        self._reset()

    def _extra_terminals(self, vocab: dict[str, int]) -> dict[int, str]:
        terminals = {}
        for text, kind in ((TOOL_START, "tool_start"), (TOOL_END, "tool_end")):
            token_id = self._require_terminal(vocab, text)
            terminals[token_id] = kind
        if len(terminals) != 2:
            raise ValueError("Qwen3 tool terminals must use distinct token IDs")
        return terminals

    def _reset(self) -> None:
        super()._reset()
        self._inside_tool = False
        self._body_parts: list[str] = []
        self._in_string = False
        self._escaped = False
        self._calls: list[ParsedToolCall] = []
        self._has_content = False
        self._after_tool = False
        self._pending_whitespace = ""

    def _consume(self, segments: Sequence[Segment]) -> ParsedDelta:
        reasoning: list[str] = []
        content: list[str] = []
        deltas: list[ToolCallDelta] = []
        for segment in segments:
            if isinstance(segment, TextSegment):
                if self._inside_tool:
                    self._append_body(segment.text)
                elif self._state == "reasoning":
                    if self._include_reasoning:
                        reasoning.append(segment.text)
                else:
                    # Wait for the next terminal before deciding whether whitespace separates calls.
                    if (self._after_tool or not self._has_content) and segment.text.isspace():
                        self._pending_whitespace += segment.text
                    else:
                        content.append(self._pending_whitespace + segment.text)
                        self._pending_whitespace = ""
                        self._after_tool = False
                        self._has_content = True
                continue

            if self._inside_tool:
                if segment.kind == "tool_end" and not self._in_string:
                    self._close_tool(deltas)
                else:
                    self._append_body(segment.text)
            elif segment.kind == "tool_start":
                self._inside_tool = True
                self._state = "content"
                self._after_tool = False
                self._pending_whitespace = ""
            elif segment.kind == "tool_end":
                raise ValueError("unexpected Qwen3 tool-call end")
            elif segment.kind == START_TERMINAL:
                content.append(self._pending_whitespace)
                self._pending_whitespace = ""
                self._after_tool = False
                self._state = "reasoning"
            elif segment.kind == END_TERMINAL:
                content.append(self._pending_whitespace)
                self._pending_whitespace = ""
                self._after_tool = False
                self._state = "content"
            elif segment.kind != DROP_TERMINAL:
                raise ValueError(f"unexpected Qwen3 terminal {segment.kind!r}")
        return ParsedDelta(
            reasoning="".join(reasoning), content="".join(content),
            tool_call_deltas=tuple(deltas),
        )

    def _append_body(self, text: str) -> None:
        self._body_parts.append(text)
        for char in text:
            if self._escaped:
                self._escaped = False
            elif char == "\\" and self._in_string:
                self._escaped = True
            elif char == '"':
                self._in_string = not self._in_string

    def _close_tool(self, deltas: list[ToolCallDelta]) -> None:
        try:
            payload = json.loads("".join(self._body_parts))
            if not isinstance(payload, dict):
                raise ValueError("Qwen3 tool call must be a JSON object")
            name, arguments = payload.get("name"), payload.get("arguments")
            if (
                not isinstance(name, str)
                or not _FUNCTION_NAME.fullmatch(name)
                or not isinstance(arguments, dict)
            ):
                raise ValueError("Qwen3 tool call requires a name and object arguments")
            encoded = json.dumps(arguments, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        except (json.JSONDecodeError, RecursionError) as exc:
            raise ValueError(f"invalid Qwen3 tool call: {exc}") from exc
        if self._publish_tools:
            call = ParsedToolCall(id="call_" + uuid.uuid4().hex, name=name, arguments=encoded)
            deltas.append(ToolCallDelta(
                index=len(self._calls), id=call.id, name=name, arguments=encoded,
            ))
            self._calls.append(call)
        self._inside_tool = False
        self._after_tool = True
        self._body_parts.clear()
        self._in_string = False
        self._escaped = False

    def finish(self, *, truncated: bool = False) -> ParsedDelta:
        tail = super().finish(truncated=truncated)
        if self._inside_tool and not truncated:
            raise ValueError("generation ended inside an incomplete Qwen3 tool call")
        content = tail.content + ("" if self._after_tool else self._pending_whitespace)
        self._pending_whitespace = ""
        self._after_tool = False
        return replace(tail, content=content, tool_calls=tuple(self._calls))
