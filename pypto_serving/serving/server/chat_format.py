# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pure chat response formatting shared by ordinary Serving and the PD Router."""

from __future__ import annotations

from pypto_serving.serving.reasoning import ParsedToolCall, ToolCallDelta

from .api_types import (
    ChatDelta,
    ChatMessage,
    DeltaFunctionCall,
    DeltaToolCall,
    FunctionCall,
    ToolCall,
)


def map_finish_reason(reason: str) -> str:
    return {
        "FINISHED_EOS": "stop",
        "FINISHED_LENGTH": "length",
        "FINISHED_STOP": "stop",
        "FINISHED_ABORTED": "aborted",
        "error": "error",
    }.get(reason, "stop")


def chat_finish_reason(output) -> str:
    if (
        output.finish_reason in ("FINISHED_EOS", "FINISHED_STOP")
        and output.tool_calls
        and all(call.complete for call in output.tool_calls)
    ):
        return "tool_calls"
    return map_finish_reason(output.finish_reason)


def chat_message(
    text: str,
    reasoning: str,
    tool_calls: tuple[ParsedToolCall, ...],
    *,
    parallel_tool_calls: bool = True,
) -> ChatMessage:
    selected = tool_calls if parallel_tool_calls else tool_calls[:1]
    return ChatMessage(
        role="assistant",
        content=text or (None if selected else ""),
        reasoning=reasoning or None,
        tool_calls=[
            ToolCall(
                id=call.id,
                function=FunctionCall(name=call.name, arguments=call.arguments),
            )
            for call in selected
        ] or None,
    )


def chat_delta(
    text_delta: str,
    reasoning_delta: str,
    tool_call_deltas: tuple[ToolCallDelta, ...],
    *,
    parallel_tool_calls: bool = True,
) -> ChatDelta:
    selected = [
        DeltaToolCall(
            index=delta.index,
            id=delta.id,
            type="function" if delta.id else None,
            function=DeltaFunctionCall(
                name=delta.name,
                arguments=delta.arguments or None,
            ),
        )
        for delta in tool_call_deltas
        if parallel_tool_calls or delta.index == 0
    ]
    return ChatDelta(
        role="assistant",
        content=text_delta or (None if selected else ""),
        reasoning=reasoning_delta or None,
        tool_calls=selected or None,
    )
