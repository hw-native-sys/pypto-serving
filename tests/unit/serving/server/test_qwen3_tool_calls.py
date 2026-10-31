# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3 tool calls through the Chat HTTP and engine delivery path."""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from pypto_serving.serving.server.server import ChatCompletionRequest
from pypto_serving.model.qwen.tool_parser import TOOL_END, TOOL_START
from tests.unit.model.qwen.test_tool_parser import QwenToolTokenizer
from tests.unit.serving.server.test_tool_calls import TOOLS, _assert_released, _collect_chat, _replay_server


class _QwenChatTokenizer(QwenToolTokenizer):
    output_parser_id = "qwen3"
    eos_token_id = QwenToolTokenizer.vocab["<eos>"]
    bos_token_id = None

    def __init__(self):
        self.prompts = []

    def apply_chat_template(self, messages, **kwargs):
        self.prompts.append((messages, kwargs))
        return "Question"


@pytest.mark.parametrize("stream", [False, True])
def test_direct_delivery_suppresses_whitespace_between_calls(stream):
    call = TOOL_START + '{"name":"lookup","arguments":{}}' + TOOL_END
    server = _replay_server(("plan</think>\n\n" + call + "\n" + call + "<eos>", "FINISHED_EOS"),
                            chunk_size=1, tokenizer=_QwenChatTokenizer())
    request = ChatCompletionRequest.model_validate({
        "messages": [{"role": "user", "content": "Question"}],
        "tools": TOOLS, "stream": stream, "reasoning_effort": "high",
    })

    async def run():
        response = await server._chat_completions(request)
        if not stream:
            message = json.loads(response.body)["choices"][0]["message"]
            assert message["content"] is None
            assert message["reasoning"] == "plan"
            assert len(message["tool_calls"]) == 2
        else:
            events = [
                json.loads(chunk[6:]) async for chunk in response.body_iterator
                if chunk != "data: [DONE]\n\n"
            ]
            choices = [event["choices"][0] for event in events if event.get("choices")]
            assert "".join(choice["delta"]["content"] or "" for choice in choices) == ""
            assert "".join(choice["delta"]["reasoning"] or "" for choice in choices) == "plan"
            assert sum(len(choice["delta"].get("tool_calls") or ()) for choice in choices) == 2
            assert choices[-1]["finish_reason"] == "tool_calls"

    asyncio.run(run())
    _assert_released(server, 1)


@pytest.mark.parametrize("stream", [False, True])
def test_direct_delivery_omits_trailing_tool_whitespace(stream):
    call = TOOL_START + '{"name":"lookup","arguments":{}}' + TOOL_END
    server = _replay_server((call + "\n<eos>", "FINISHED_EOS"),
                            chunk_size=1, tokenizer=_QwenChatTokenizer())
    request = ChatCompletionRequest.model_validate({
        "messages": [{"role": "user", "content": "Question"}],
        "tools": TOOLS, "stream": stream,
    })

    async def run():
        response = await server._chat_completions(request)
        if not stream:
            choice = json.loads(response.body)["choices"][0]
            assert choice["message"]["content"] is None
            assert len(choice["message"]["tool_calls"]) == 1
            assert choice["finish_reason"] == "tool_calls"
        else:
            events = [
                json.loads(chunk[6:]) async for chunk in response.body_iterator
                if chunk != "data: [DONE]\n\n"
            ]
            choices = [event["choices"][0] for event in events if event.get("choices")]
            assert "".join(choice["delta"]["content"] or "" for choice in choices) == ""
            assert sum(len(choice["delta"].get("tool_calls") or ()) for choice in choices) == 1
            assert choices[-1]["finish_reason"] == "tool_calls"

    asyncio.run(run())
    _assert_released(server, 1)


@pytest.mark.parametrize("stream", [False, True])
def test_direct_delivery_rejects_invalid_function_name(stream):
    call = TOOL_START + '{"name":"weather.get","arguments":{}}' + TOOL_END + "<eos>"
    server = _replay_server((call, "FINISHED_EOS"), tokenizer=_QwenChatTokenizer())
    request = ChatCompletionRequest.model_validate({
        "messages": [{"role": "user", "content": "Question"}],
        "tools": TOOLS, "stream": stream,
    })

    async def run():
        if not stream:
            with pytest.raises(ValueError, match="Qwen3 tool call requires"):
                await server._chat_completions(request)
        else:
            response = await server._chat_completions(request)
            events = [
                json.loads(chunk[6:]) async for chunk in response.body_iterator
                if chunk != "data: [DONE]\n\n"
            ]
            assert any("Qwen3 tool call requires" in event.get("error", {}).get("message", "")
                       for event in events)
            assert not any(event.get("usage") for event in events)

    asyncio.run(run())
    _assert_released(server, 1)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 7, 100000])
def test_tool_roundtrip_and_followup_answer(stream, chunk_size):
    call = TOOL_START + '{"name":"lookup","arguments":{"city":"杭州"}}' + TOOL_END + "<eos>"
    server = _replay_server(
        (call, "FINISHED_EOS"), ("Done<eos>", "FINISHED_EOS"),
        chunk_size=chunk_size, tokenizer=_QwenChatTokenizer(),
    )
    messages = [{"role": "user", "content": "Find city"}]
    with TestClient(server.app) as client:
        first, reason, _ = _collect_chat(client.post("/v1/chat/completions", json={
            "messages": messages, "tools": TOOLS, "stream": stream,
        }), stream)
        assert reason == "tool_calls"
        assert first["content"] is None
        assert json.loads(first["tool_calls"][0]["function"]["arguments"]) == {"city": "杭州"}
        messages += [first, {"role": "tool", "tool_call_id": first["tool_calls"][0]["id"], "content": "Found"}]
        second, reason, _ = _collect_chat(client.post("/v1/chat/completions", json={
            "messages": messages, "tools": TOOLS, "stream": stream,
        }), stream)
    assert second["content"] == "Done"
    assert reason == "stop"
    assert server.engine.tokenizer.prompts[1][0][1]["tool_calls"][0]["id"] == first["tool_calls"][0]["id"]
    _assert_released(server, 2)


@pytest.mark.parametrize("stream", [False, True])
def test_tool_free_qwen_chat_keeps_existing_unparsed_output(stream):
    raw = "<think>hidden</think>Answer<eos>"
    server = _replay_server((raw, "FINISHED_EOS"), tokenizer=_QwenChatTokenizer())
    with TestClient(server.app) as client:
        message, reason, _ = _collect_chat(client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Question"}], "stream": stream,
        }), stream)
    assert message["content"] == "hiddenAnswer"
    assert reason == "stop"
    _assert_released(server, 1)
