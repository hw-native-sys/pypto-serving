# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Request-local parsing of generated tokens into public semantic channels."""

from __future__ import annotations

from typing import Sequence

from .base import END_TERMINAL, START_TERMINAL, Segment, TextSegment, TokenReasoningParser
from .types import (
    OutputParser,
    OutputParserSpec,
    ParsedDelta,
    ParsedOutput,
    ParsedToolCall,
    ToolCallDelta,
)


def supports_tool_calls(parser_id: str | None) -> bool:
    """Report model parser capabilities before starting a request."""
    return parser_id in ("deepseek_v4", "qwen3")


class DeepSeekV4ReasoningParser(TokenReasoningParser):
    """Incrementally split DeepSeek V4 reasoning from answer content."""

    def _consume(self, segments: Sequence[Segment]) -> ParsedDelta:
        reasoning_parts: list[str] = []
        content_parts: list[str] = []

        for segment in segments:
            if isinstance(segment, TextSegment):
                if not segment.text:
                    continue
                if self._state == "reasoning":
                    if self._include_reasoning:
                        reasoning_parts.append(segment.text)
                else:
                    content_parts.append(segment.text)
                continue

            if segment.kind == START_TERMINAL:
                # Duplicate starts are absorbed, as in vLLM.
                self._state = "reasoning"
            elif segment.kind == END_TERMINAL:
                # A bare end in content is absorbed; reasoning ends otherwise.
                if self._state == "reasoning":
                    self._state = "content"

        return ParsedDelta(
            reasoning="".join(reasoning_parts),
            content="".join(content_parts),
        )


def create_output_parser(spec: OutputParserSpec | None, tokenizer) -> OutputParser | None:
    """Create one request-local parser from a normalized Serving contract."""
    if spec is None:
        return None
    if spec.parser_id == "deepseek_v41":
        return DeepSeekV4ReasoningParser(tokenizer, spec)
    if spec.parser_id == "deepseek_v4":
        if spec.tool_names:
            from .deepseek_v4_tools import DeepSeekV4ToolParser

            return DeepSeekV4ToolParser(tokenizer, spec)
        return DeepSeekV4ReasoningParser(tokenizer, spec)
    if spec.parser_id == "qwen3":
        from pypto_serving.model.qwen.tool_parser import Qwen3ToolParser

        return Qwen3ToolParser(tokenizer, spec)
    raise ValueError(f"unsupported output parser {spec.parser_id!r}")


__all__ = (
    "DeepSeekV4ReasoningParser",
    "OutputParser",
    "OutputParserSpec",
    "ParsedDelta",
    "ParsedOutput",
    "ParsedToolCall",
    "ToolCallDelta",
    "create_output_parser",
    "supports_tool_calls",
)
