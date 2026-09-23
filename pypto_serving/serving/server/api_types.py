# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Lightweight public HTTP models shared by ordinary Serving and the PD Router."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field, model_serializer


ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]


class CompletionRequest(BaseModel):
    model: str = ""
    prompt: str | list[int] = ""
    # Omitted sampling fields use the server's GenerateConfig defaults.
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    stop: list[str] | None = None
    stream: bool = False


class FunctionDefinition(BaseModel):
    name: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    description: str | None = None
    parameters: dict = Field(default_factory=lambda: {"type": "object", "properties": {}})
    strict: bool | None = None


class ChatTool(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition


class FunctionCall(BaseModel):
    name: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    arguments: str


class ToolCall(BaseModel):
    id: str = Field(min_length=1)
    type: Literal["function"] = "function"
    function: FunctionCall


class ChatMessage(BaseModel):
    role: str
    content: str | None = None
    reasoning: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        data = handler(self)
        for key in ("tool_calls", "tool_call_id"):
            if not data.get(key):
                data.pop(key, None)
        return data


class DeltaFunctionCall(BaseModel):
    name: str | None = None
    arguments: str | None = None


class DeltaToolCall(BaseModel):
    index: int
    id: str | None = None
    type: Literal["function"] | None = None
    function: DeltaFunctionCall

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        data = {key: value for key, value in handler(self).items() if value is not None}
        data["function"] = {key: value for key, value in data["function"].items() if value is not None}
        return data


class ChatDelta(ChatMessage):
    tool_calls: list[DeltaToolCall] | None = None


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage]
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    stop: list[str] | None = None
    stream: bool = False
    reasoning_effort: ReasoningEffort | None = None
    chat_template_kwargs: dict | None = None
    include_reasoning: bool = True
    tools: list[ChatTool] | None = None
    tool_choice: str | dict | None = None
    parallel_tool_calls: bool = True


class CompletionChoice(BaseModel):
    index: int = 0
    text: str = ""
    finish_reason: str | None = None


class ResponseUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: ResponseUsage | None = None


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage | None = None
    delta: ChatDelta | None = None
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ResponseUsage | None = None


def validate_chat_request(request: ChatCompletionRequest) -> str:
    """Validate tool semantics before an HTTP stream starts."""
    if "tools" in (request.chat_template_kwargs or {}):
        raise ValueError("tools must be supplied as a top-level request field")
    choice = request.tool_choice
    if choice is None:
        choice = "auto" if request.tools else "none"
    if choice not in ("none", "auto"):
        raise ValueError("only tool_choice 'auto' and 'none' are supported; constrained tool choice is unavailable")
    if choice == "auto" and not request.tools:
        raise ValueError("tool_choice 'auto' requires non-empty tools")
    names = []
    for tool in request.tools or ():
        if tool.function.strict:
            raise ValueError("strict tool schemas require constrained decoding, which is not supported")
        names.append(tool.function.name)
    if len(set(names)) != len(names):
        raise ValueError("tool function names must be unique")
    for message in request.messages:
        calls = message.tool_calls or ()
        if calls and message.role != "assistant":
            raise ValueError("only assistant messages can contain tool_calls")
        if message.content is None and not (message.role == "assistant" and calls):
            raise ValueError("message content must be text, or null for an assistant tool call")
        if message.role == "tool" and not message.tool_call_id:
            raise ValueError("tool messages require tool_call_id")
        if len({call.id for call in calls}) != len(calls):
            raise ValueError("assistant tool call IDs must be unique")
        for call in calls:
            if not isinstance(json.loads(call.function.arguments), dict):
                raise ValueError("tool call arguments must encode an object")
    return choice
