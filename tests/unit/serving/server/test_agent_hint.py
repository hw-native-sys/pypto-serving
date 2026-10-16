# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate optional Agent session identity without changing Chat responses."""

import pytest
from fastapi.testclient import TestClient

from pypto_serving.config.types import GenerateConfig
from pypto_serving.serving.engine.async_engine import TokenOutput
from pypto_serving.serving.reasoning import ParsedToolCall
from pypto_serving.serving.server.server import ServingServer


class _Tokenizer:
    @staticmethod
    def apply_chat_template(messages, **kwargs):
        return " ".join(message["content"] or "" for message in messages)


class _Engine:
    tokenizer = _Tokenizer()

    def __init__(self, finish_reason="FINISHED_EOS", tool_calls=()):
        self.calls = []
        self.lifecycle = []
        self.finish_reason = finish_reason
        self.tool_calls = tool_calls

    def observe_agent_lifecycle(self, session_id, lifecycle):
        self.lifecycle.append((session_id, lifecycle))
        return lifecycle == "terminal"

    async def add_request(self, request_id, prompt, config, **kwargs):
        self.calls.append((request_id, prompt, kwargs))
        yield TokenOutput(
            text="OK", finished=True, finish_reason=self.finish_reason, tool_calls=self.tool_calls,
            prompt_tokens=4, completion_tokens=1,
        )


def test_text_action_rounds_share_identity_and_plain_chat_stays_unchanged():
    engine = _Engine()
    server = ServingServer(engine, "test-model", GenerateConfig())
    with TestClient(server.app) as client:
        first = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Read the file"}],
            "agent_hint": {"session_id": "run-123"},
        })
        second = client.post("/v1/chat/completions", json={
            "messages": [
                {"role": "user", "content": "Read the file"},
                {"role": "assistant", "content": "cat input.txt"},
                {"role": "user", "content": "Tool output: hello"},
            ],
            "agent_hint": {"session_id": "run-123"},
        })
        other = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Another run"}],
            "agent_hint": {"session_id": "run-456"},
        })
        plain = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Ordinary request"}],
        })

    assert all(response.status_code == 200 for response in (first, second, other, plain))
    assert len({response.json()["id"] for response in (first, second, other, plain)}) == 4
    assert all(response.json()["choices"][0]["message"]["content"] == "OK"
               for response in (first, second, other, plain))
    assert [call[2].get("agent_session_id") for call in engine.calls] == [
        "run-123", "run-123", "run-456", None,
    ]
    assert "agent_session_id" not in engine.calls[-1][2]
    assert all("agent_hint" not in response.json() for response in (first, second, other, plain))


def test_streaming_chat_carries_session_identity():
    engine = _Engine()
    server = ServingServer(engine, "test-model", GenerateConfig())
    with TestClient(server.app) as client:
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Stream"}],
            "stream": True,
            "agent_hint": {"session_id": "stream-1"},
        })
    assert response.status_code == 200
    assert "data: [DONE]" in response.text
    assert engine.calls[0][2]["agent_session_id"] == "stream-1"


@pytest.mark.parametrize("hint", [
    {},
    {"session_id": ""},
    {"session_id": "with space"},
    {"session_id": "a" * 129},
    {"session_id": 42},
    {"session_id": "valid", "unexpected": True},
    {"session_id": "valid", "lifecycle_mode": "terminal"},
])
def test_invalid_agent_hint_is_rejected_before_generation(hint):
    engine = _Engine()
    server = ServingServer(engine, "test-model", GenerateConfig())
    with TestClient(server.app) as client:
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Hello"}],
            "agent_hint": hint,
        })
    assert response.status_code == 422
    assert engine.calls == []


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("mode, expected", [("explicit", []), ("chat", [("session-a", "terminal")])])
def test_response_stop_releases_only_opted_in_chat_lifecycle(stream, mode, expected):
    engine = _Engine()
    server = ServingServer(engine, "test-model", GenerateConfig())
    with TestClient(server.app) as client:
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Hello"}], "stream": stream,
            "agent_hint": {"session_id": "session-a", "lifecycle_mode": mode},
        })
    assert response.status_code == 200
    assert engine.lifecycle == expected


def test_text_agent_can_confirm_terminal_lifecycle_after_verification():
    engine = _Engine()
    server = ServingServer(engine, "test-model", GenerateConfig())
    with TestClient(server.app) as client:
        continued = client.post("/v1/agent/lifecycle", json={
            "session_id": "session-a", "lifecycle": "continue",
        })
        terminal = client.post("/v1/agent/lifecycle", json={
            "session_id": "session-a", "lifecycle": "terminal",
        })
        invalid = client.post("/v1/agent/lifecycle", json={
            "session_id": "session-a", "lifecycle": "finished",
        })
    assert continued.json() == {"released": False}
    assert terminal.json() == {"released": True}
    assert invalid.status_code == 422
    assert engine.lifecycle == [("session-a", "continue"), ("session-a", "terminal")]


@pytest.mark.parametrize("stream", [False, True])
def test_truncated_chat_response_never_proves_task_completion(stream):
    engine = _Engine(finish_reason="FINISHED_LENGTH")
    server = ServingServer(engine, "test-model", GenerateConfig())
    with TestClient(server.app) as client:
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Hello"}], "stream": stream,
            "agent_hint": {"session_id": "session-a", "lifecycle_mode": "chat"},
        })
    assert response.status_code == 200
    assert engine.lifecycle == []


@pytest.mark.parametrize("stream", [False, True])
def test_parsed_tool_call_reports_continue_instead_of_terminal(stream):
    call = ParsedToolCall(id="call-1", name="read_file", arguments='{"path":"x"}', complete=True)
    engine = _Engine(tool_calls=(call,))
    server = ServingServer(engine, "test-model", GenerateConfig())
    with TestClient(server.app) as client:
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Read x"}], "stream": stream,
            "agent_hint": {"session_id": "session-a", "lifecycle_mode": "chat"},
        })
    assert response.status_code == 200
    assert engine.lifecycle == [("session-a", "continue")]
