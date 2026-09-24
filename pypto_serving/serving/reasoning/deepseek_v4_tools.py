# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek V4 DSML parsing on the existing token-aligned reasoning stream."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field, replace
from typing import Iterable, Sequence

from .parser import (
    DeepSeekV4ReasoningParser,
    OutputParserSpec,
    ParsedDelta,
    ParsedToolCall,
    ToolCallDelta,
    _DROP_TERMINAL,
    _END_TERMINAL,
    _Segment,
    _START_TERMINAL,
    _TerminalSegment,
    _TextSegment,
)

_DSML = "\uff5cDSML\uff5c"
TOOL_START = f"<{_DSML}tool_calls>"
TOOL_END = f"</{_DSML}tool_calls>"
_PARAM_END = f"</{_DSML}parameter>"
_INVOKE_END = f"</{_DSML}invoke>"
_INVOKE_RE = re.compile(rf'<{_DSML}invoke\s+name="([A-Za-z0-9_-]{{1,64}})"\s*>')
_PARAM_RE = re.compile(rf'<{_DSML}parameter\s+name="([^"<>]*)"\s+string="(true|false)"\s*>')


@dataclass
class _Call:
    name: str
    id: str = field(default_factory=lambda: "call_" + uuid.uuid4().hex)
    parts: list[str] = field(default_factory=list)
    keys: set[str] = field(default_factory=set)
    complete: bool = False


class DeepSeekV4ToolParser(DeepSeekV4ReasoningParser):
    """Split reasoning, content and DSML calls without rescanning past arguments."""

    def __init__(self, tokenizer, spec: OutputParserSpec) -> None:
        super().__init__(tokenizer, spec)
        self._publish_tools = spec.tool_choice != "none"
        self._reset()

    def _extra_terminals(self, vocab: dict[str, int]) -> dict[int, str]:
        # Checkpoints may expose whole root tags or only the inner DSML sentinel
        # as special tokens. As in vLLM, use token boundaries where available and
        # bounded text matching for roots that are not individual vocab entries.
        terminals = {
            token_id: "dsml_text" for text, token_id in vocab.items()
            if text == _DSML or text.startswith((f"<{_DSML}", f"</{_DSML}"))
        }
        self._text_roots = {}
        for text, kind in ((TOOL_START, "tool_start"), (TOOL_END, "tool_end")):
            if text in vocab:
                terminals[vocab[text]] = kind
            else:
                self._text_roots[text] = kind
        self._root_pattern = (
            re.compile("|".join(re.escape(text) for text in self._text_roots))
            if self._text_roots else None
        )
        return terminals

    def _reset(self) -> None:
        super()._reset()
        self._inside_tools = False
        self._tool_state = "between_calls"
        self._header_parts: list[str] = []
        self._body_tail = ""
        self._value_parts: list[str] = []
        self._string_value = False
        self._calls: list[_Call] = []
        self._root_tail = ""

    def _consume(self, segments: Sequence[_Segment]) -> ParsedDelta:
        return self._consume_segments(self._split_text_roots(segments))

    def _split_text_roots(self, segments: Iterable[_Segment]) -> Iterable[_Segment]:
        for segment in segments:
            if self._root_pattern is None:
                yield segment
                continue
            if isinstance(segment, _TextSegment) or segment.kind == "dsml_text":
                text = self._root_tail + segment.text
                self._root_tail = ""
                offset = 0
                for match in self._root_pattern.finditer(text):
                    if match.start() > offset:
                        yield _TextSegment(text[offset:match.start()])
                    yield _TerminalSegment(self._text_roots[match.group()], match.group())
                    offset = match.end()
                keep = 0
                for root in self._text_roots:
                    for size in range(min(len(root) - 1, len(text) - offset), keep, -1):
                        if text.endswith(root[:size]):
                            keep = size
                            break
                if len(text) - keep > offset:
                    yield _TextSegment(text[offset:len(text) - keep])
                self._root_tail = text[-keep:] if keep else ""
            else:
                # Never join a textual tag across a token-ID control boundary.
                if self._root_tail:
                    yield _TextSegment(self._root_tail)
                    self._root_tail = ""
                yield segment

    def _consume_segments(self, segments: Iterable[_Segment]) -> ParsedDelta:
        reasoning: list[str] = []
        content: list[str] = []
        deltas: list[ToolCallDelta] = []
        for segment in segments:
            if isinstance(segment, _TextSegment):
                if self._inside_tools:
                    self._consume_tool_text(segment.text, deltas)
                elif self._state == "reasoning":
                    if self._include_reasoning:
                        reasoning.append(segment.text)
                else:
                    content.append(segment.text)
                continue
            if segment.kind == _DROP_TERMINAL:
                continue
            if self._inside_tools and self._tool_state == "value":
                # A think/root token inside a value is data, not a phase switch.
                self._consume_tool_text(segment.text, deltas)
            elif segment.kind == "tool_start":
                if self._inside_tools:
                    raise ValueError("nested DSML tool_calls block")
                self._state = "content"
                self._inside_tools = True
                self._tool_state = "between_calls"
            elif segment.kind == "tool_end":
                if not self._inside_tools or self._header_parts:
                    raise ValueError("unexpected DSML tool_calls end")
                if self._tool_state == "between_params":
                    # V4 also permits TOOL_END to close the last invoke.
                    self._close_call(deltas)
                self._inside_tools = False
            elif self._inside_tools:
                self._consume_tool_text(segment.text, deltas)
            elif segment.kind == _START_TERMINAL:
                self._state = "reasoning"
            elif segment.kind == _END_TERMINAL:
                self._state = "content"
        return ParsedDelta(
            reasoning="".join(reasoning), content="".join(content),
            tool_call_deltas=self._coalesce(deltas),
        )

    def _consume_tool_text(self, text: str, deltas: list[ToolCallDelta]) -> None:
        offset = 0
        while offset < len(text):
            if self._tool_state == "value":
                # Join only the short boundary, not the entire remaining input
                # once per parameter (which would be quadratic for a large feed).
                if self._body_tail:
                    prefix = text[offset:offset + len(_PARAM_END)]
                    body = self._body_tail + prefix
                    old_tail_length = len(self._body_tail)
                    end = body.find(_PARAM_END)
                    if end >= 0:
                        self._body_tail = ""
                        self._append_value(body[:end], deltas)
                        self._close_value(deltas)
                        offset += end + len(_PARAM_END) - old_tail_length
                    else:
                        keep = self._closing_prefix_length(body)
                        self._body_tail = body[-keep:] if keep else ""
                        self._append_value(body[:-keep] if keep else body, deltas)
                        offset += len(prefix)
                    continue
                end = text.find(_PARAM_END, offset)
                if end < 0:
                    keep = self._closing_prefix_length(text[max(offset, len(text) - len(_PARAM_END)):])
                    self._body_tail = text[-keep:] if keep else ""
                    self._append_value(text[offset:len(text) - keep], deltas)
                    return
                self._append_value(text[offset:end], deltas)
                self._close_value(deltas)
                offset = end + len(_PARAM_END)
                continue

            if not self._header_parts:
                while offset < len(text) and text[offset].isspace():
                    offset += 1
                if offset == len(text):
                    return
                if text[offset] != "<":
                    raise ValueError("unexpected text in DSML tool_calls block")
            end = text.find(">", offset)
            if end < 0:
                self._header_parts.append(text[offset:])
                return
            self._header_parts.append(text[offset:end + 1])
            header = "".join(self._header_parts)
            self._header_parts.clear()
            self._consume_header(header, deltas)
            offset = end + 1

    @staticmethod
    def _closing_prefix_length(text: str) -> int:
        for size in range(min(len(text), len(_PARAM_END) - 1), 0, -1):
            if text.endswith(_PARAM_END[:size]):
                return size
        return 0

    def _consume_header(self, header: str, deltas: list[ToolCallDelta]) -> None:
        if self._tool_state == "between_calls":
            match = _INVOKE_RE.fullmatch(header)
            if match is None:
                raise ValueError("expected a DSML invoke header")
            name = match.group(1)
            call = _Call(name=name)
            self._calls.append(call)
            if self._publish_tools:
                deltas.append(ToolCallDelta(index=len(self._calls) - 1, id=call.id, name=name))
            self._append_arguments("{", deltas)
            self._tool_state = "between_params"
            return
        if header == _INVOKE_END:
            self._close_call(deltas)
            return
        match = _PARAM_RE.fullmatch(header)
        if match is None:
            raise ValueError("expected a DSML parameter or invoke end")
        key, is_string = match.groups()
        call = self._calls[-1]
        if key in call.keys:
            raise ValueError(f"duplicate DSML parameter {key!r}")
        prefix = "," if call.keys else ""
        call.keys.add(key)
        self._string_value = is_string == "true"
        self._value_parts.clear()
        self._tool_state = "value"
        self._append_arguments(
            prefix + json.dumps(key, ensure_ascii=False) + ":" + ('"' if self._string_value else ""),
            deltas,
        )

    def _append_value(self, text: str, deltas: list[ToolCallDelta]) -> None:
        if not text:
            return
        if self._string_value:
            self._append_arguments(json.dumps(text, ensure_ascii=False)[1:-1], deltas)
        else:
            self._value_parts.append(text)

    def _close_value(self, deltas: list[ToolCallDelta]) -> None:
        if self._string_value:
            self._append_arguments('"', deltas)
        else:
            try:
                value = json.loads("".join(self._value_parts))
                encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
            except RecursionError as exc:
                raise ValueError("DSML parameter JSON exceeds the nesting limit") from exc
            self._append_arguments(encoded, deltas)
            self._value_parts.clear()
        self._tool_state = "between_params"

    def _append_arguments(self, text: str, deltas: list[ToolCallDelta]) -> None:
        if not text:
            return
        self._calls[-1].parts.append(text)
        if self._publish_tools:
            deltas.append(ToolCallDelta(index=len(self._calls) - 1, arguments=text))

    def _close_call(self, deltas: list[ToolCallDelta]) -> None:
        self._append_arguments("}", deltas)
        self._calls[-1].complete = True
        self._tool_state = "between_calls"

    @staticmethod
    def _coalesce(deltas: list[ToolCallDelta]) -> tuple[ToolCallDelta, ...]:
        heads: dict[int, ToolCallDelta] = {}
        parts: dict[int, list[str]] = {}
        for delta in deltas:
            heads.setdefault(delta.index, delta)
            parts.setdefault(delta.index, []).append(delta.arguments)
        return tuple(replace(head, arguments="".join(parts[index])) for index, head in heads.items())

    def finish(self, *, truncated: bool = False) -> ParsedDelta:
        tail = super().finish(truncated=truncated)
        if self._root_tail:
            pending = self._root_tail
            self._root_tail = ""
            rest = self._consume_segments((_TextSegment(pending),))
            tail = ParsedDelta(
                reasoning=tail.reasoning + rest.reasoning,
                content=tail.content + rest.content,
                tool_call_deltas=self._coalesce([*tail.tool_call_deltas, *rest.tool_call_deltas]),
            )
        if self._inside_tools and not truncated:
            raise ValueError("generation ended inside an incomplete DSML tool call")
        calls = tuple(
            ParsedToolCall(id=call.id, name=call.name, arguments="".join(call.parts), complete=call.complete)
            for call in self._calls
        ) if self._publish_tools else ()
        return replace(tail, tool_calls=calls)
