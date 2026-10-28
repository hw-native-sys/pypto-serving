# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate optional Agent session identity without changing Chat responses."""

from types import SimpleNamespace

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

    def observe_agent_lifecycle(self, session_id, request_id, lifecycle):
        self.lifecycle.append((session_id, request_id, lifecycle))
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


@pytest.mark.parametrize("stream", [False, True])
def test_completion_headers_enter_agent_admission(stream):
    engine = _Engine()
    server = ServingServer(engine, "test-model", GenerateConfig())
    with TestClient(server.app) as client:
        tagged = client.post("/v1/completions", headers={"X-Session-Id": "text-agent"}, json={
            "prompt": "Reply OK.", "stream": stream,
        })
        plain = client.post("/v1/completions", json={"prompt": "Reply OK.", "stream": stream})
    assert tagged.status_code == plain.status_code == 200
    assert engine.calls[0][2]["agent_session_id"] == "text-agent:lead"
    assert engine.calls[0][2]["agent_identity"].program_id == "text-agent:lead"
    assert "agent_session_id" not in engine.calls[1][2]


def test_disabled_agentinfer_does_not_parse_framework_headers():
    engine = _Engine()
    engine.config = SimpleNamespace(agentinfer_mode="disabled")
    server = ServingServer(engine, "test-model", GenerateConfig())
    with TestClient(server.app) as client:
        chat = client.post("/v1/chat/completions", headers={"X-Session-Id": "invalid session"}, json={
            "messages": [{"role": "user", "content": "Hello"}],
        })
        completion = client.post("/v1/completions", headers={"X-Session-Id": "invalid session"}, json={
            "prompt": "Hello",
        })
    assert chat.status_code == completion.status_code == 200
    assert all("agent_identity" not in call[2] for call in engine.calls)


@pytest.mark.parametrize("headers, expected", [
    ({"X-Session-ID": "run-123"}, "run-123:lead"),
    ({"Session-ID": "run-123", "Thread-ID": "thread-1"}, "run-123:thread-1"),
    ({"X-Claude-Code-Session-Id": "run-123"}, "run-123:lead"),
    ({"X-DeepSeek-Harness-Session-Id": "run-123"}, "run-123:lead"),
    ({"X-OpenCode-Session-Id": "run-123"}, "run-123:lead"),
    ({"X-OpenCode-Session": "run-123"}, "run-123:lead"),
    ({"X-Claude-Code-Session-Id": "run-123", "X-Claude-Code-Agent-Id": "worker-1"},
     "run-123:worker-1"),
])
@pytest.mark.parametrize("stream", [False, True])
def test_framework_headers_enter_agent_admission(headers, expected, stream):
    engine = _Engine()
    server = ServingServer(engine, "test-model", GenerateConfig())
    with TestClient(server.app) as client:
        response = client.post("/v1/chat/completions", headers=headers, json={
            "messages": [{"role": "user", "content": "Hello"}], "stream": stream,
        })
    assert response.status_code == 200
    assert engine.calls[0][2]["agent_session_id"] == expected


def test_framework_headers_keep_rounds_together_and_isolate_subagents():
    engine = _Engine()
    server = ServingServer(engine, "test-model", GenerateConfig())
    with TestClient(server.app) as client:
        for headers in (
            {"X-Claude-Code-Session-Id": "run-a"},
            {"X-Claude-Code-Session-Id": "run-a"},
            {"X-Claude-Code-Session-Id": "run-a", "X-Claude-Code-Agent-Id": "worker-1"},
            {"X-Claude-Code-Session-Id": "run-b"},
        ):
            response = client.post("/v1/chat/completions", headers=headers, json={
                "messages": [{"role": "user", "content": "Hello"}],
            })
            assert response.status_code == 200
    assert [call[2]["agent_session_id"] for call in engine.calls] == [
        "run-a:lead", "run-a:lead", "run-a:worker-1", "run-b:lead",
    ]


def test_opencode_native_session_header_precedes_generic_alias():
    engine = _Engine()
    server = ServingServer(engine, "test-model", GenerateConfig())
    with TestClient(server.app) as client:
        response = client.post("/v1/chat/completions", headers={
            "X-OpenCode-Session-Id": "opencode-run",
            "X-Session-Id": "generic-run",
        }, json={"messages": [{"role": "user", "content": "Hello"}]})
    assert response.status_code == 200
    assert engine.calls[0][2]["agent_session_id"] == "opencode-run:lead"


def test_native_agent_headers_preserve_parent_lineage():
    engine = _Engine()
    server = ServingServer(engine, "test-model", GenerateConfig())
    with TestClient(server.app) as client:
        claude = client.post("/v1/chat/completions", headers={
            "X-Claude-Code-Session-Id": "run-a",
            "X-Claude-Code-Agent-Id": "worker-1",
            "X-Claude-Code-Parent-Agent-Id": "lead",
        }, json={"messages": [{"role": "user", "content": "Hello"}]})
        codex = client.post("/v1/chat/completions", headers={
            "Session-Id": "run-b", "Thread-Id": "worker-2",
            "X-Codex-Parent-Thread-Id": "lead",
        }, json={"messages": [{"role": "user", "content": "Hello"}]})
    assert claude.status_code == codex.status_code == 200
    claude_identity = engine.calls[0][2]["agent_identity"]
    codex_identity = engine.calls[1][2]["agent_identity"]
    assert (claude_identity.program_id, claude_identity.parent_program_id, claude_identity.blocks_parent) == (
        "run-a:worker-1", "run-a:lead", True,
    )
    assert (codex_identity.program_id, codex_identity.parent_program_id, codex_identity.blocks_parent) == (
        "run-b:worker-2", "run-b:lead", True,
    )


def test_deepseek_harness_compaction_is_not_mistaken_for_agent_progress():
    engine = _Engine()
    server = ServingServer(engine, "test-model", GenerateConfig())
    with TestClient(server.app) as client:
        response = client.post("/v1/chat/completions", headers={
            "X-DeepSeek-Harness-Session-Id": "run-a",
            "X-DeepSeek-Harness-Compact": "1",
        }, json={"messages": [{"role": "user", "content": "Summarize"}]})
    assert response.status_code == 200
    assert "agent_session_id" not in engine.calls[0][2]


def test_canonical_agent_context_precedes_hint_and_headers():
    engine = _Engine()
    server = ServingServer(engine, "test-model", GenerateConfig())
    context = {
        "program_id": "task-1:worker", "task_id": "task-1", "session_id": "agent-1",
        "agent_id": "worker", "parent_program_id": "task-1:lead",
        "blocks_parent": True, "expected_resume": False,
    }
    with TestClient(server.app) as client:
        response = client.post("/v1/chat/completions", headers={"X-Session-Id": "header-run"}, json={
            "messages": [{"role": "user", "content": "Hello"}],
            "agent_hint": {"session_id": "hint-run"},
            "vllm_xargs": {"agentic_context": context},
        })
    assert response.status_code == 200
    assert engine.calls[0][2]["agent_session_id"] == "task-1:worker"
    identity = engine.calls[0][2]["agent_identity"]
    assert identity.parent_program_id == "task-1:lead"
    assert identity.blocks_parent is True
    assert identity.expected_resume is False


def test_canonical_agent_component_can_use_header_session():
    engine = _Engine()
    server = ServingServer(engine, "test-model", GenerateConfig())
    with TestClient(server.app) as client:
        response = client.post("/v1/chat/completions", headers={"X-Session-Id": "run-a"}, json={
            "messages": [{"role": "user", "content": "Hello"}],
            "vllm_xargs": {"agentic_context": {"agent_id": "worker-1"}},
        })
    assert response.status_code == 200
    identity = engine.calls[0][2]["agent_identity"]
    assert identity.program_id == "run-a:worker-1"
    assert identity.parent_program_id == "run-a:lead"


def test_jiuwen_parent_session_is_accepted_without_extra_adapter():
    engine = _Engine()
    server = ServingServer(engine, "test-model", GenerateConfig())
    with TestClient(server.app) as client:
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Hello"}],
            "agent_hint": {"session_id": "worker-1", "parent_session_id": "root-1"},
        })
    assert response.status_code == 200
    identity = engine.calls[0][2]["agent_identity"]
    assert (identity.program_id, identity.parent_program_id, identity.blocks_parent) == (
        "worker-1", "root-1", True,
    )


def test_explicit_hint_precedes_headers_and_unrecognized_headers_use_native_path():
    engine = _Engine()
    server = ServingServer(engine, "test-model", GenerateConfig())
    with TestClient(server.app) as client:
        explicit = client.post("/v1/chat/completions", headers={"X-Session-ID": "header-run"}, json={
            "messages": [{"role": "user", "content": "Hello"}],
            "agent_hint": {"session_id": "explicit-run"},
        })
        plain = client.post("/v1/chat/completions", headers={"X-Other-Session": "unknown"}, json={
            "messages": [{"role": "user", "content": "Hello"}],
        })
    assert explicit.status_code == plain.status_code == 200
    assert engine.calls[0][2]["agent_session_id"] == "explicit-run"
    assert "agent_session_id" not in engine.calls[1][2]


@pytest.mark.parametrize("headers", [
    {"X-Session-ID": "invalid session"},
    {"X-Session-ID": "a" * 128},
])
def test_invalid_framework_identity_is_rejected_before_generation(headers):
    engine = _Engine()
    server = ServingServer(engine, "test-model", GenerateConfig())
    with TestClient(server.app) as client:
        response = client.post("/v1/chat/completions", headers=headers, json={
            "messages": [{"role": "user", "content": "Hello"}],
        })
    assert response.status_code == 422
    assert engine.calls == []


@pytest.mark.parametrize("body", [
    {"vllm_xargs": {"agentic_context": "not-json"}},
    {"vllm_xargs": {"agentic_context": {"program_id": "invalid id"}}},
    {"agent_hint": {"session_id": "worker", "blocks_parent": True}},
])
def test_invalid_canonical_or_parent_identity_is_rejected(body):
    engine = _Engine()
    server = ServingServer(engine, "test-model", GenerateConfig())
    with TestClient(server.app) as client:
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Hello"}], **body,
        })
    assert response.status_code == 422
    assert engine.calls == []


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
@pytest.mark.parametrize("mode", ["explicit", "chat"])
def test_response_stop_releases_only_opted_in_chat_lifecycle(stream, mode):
    engine = _Engine()
    server = ServingServer(engine, "test-model", GenerateConfig())
    with TestClient(server.app) as client:
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Hello"}], "stream": stream,
            "agent_hint": {"session_id": "session-a", "lifecycle_mode": mode},
        })
    assert response.status_code == 200
    expected = [("session-a", engine.calls[0][0], "terminal")] if mode == "chat" else []
    assert engine.lifecycle == expected


def test_text_agent_can_confirm_terminal_lifecycle_after_verification():
    engine = _Engine()
    server = ServingServer(engine, "test-model", GenerateConfig())
    with TestClient(server.app) as client:
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Hello"}],
            "agent_hint": {"session_id": "session-a"},
        })
        request_id = response.json()["id"]
        continued = client.post("/v1/agent/lifecycle", json={
            "session_id": "session-a", "request_id": request_id, "lifecycle": "continue",
        })
        terminal = client.post("/v1/agent/lifecycle", json={
            "session_id": "session-a", "request_id": request_id, "lifecycle": "terminal",
        })
        invalid = client.post("/v1/agent/lifecycle", json={
            "session_id": "session-a", "request_id": request_id, "lifecycle": "finished",
        })
        missing_request = client.post("/v1/agent/lifecycle", json={
            "session_id": "session-a", "lifecycle": "terminal",
        })
    assert continued.json() == {"released": False}
    assert terminal.json() == {"released": True}
    assert invalid.status_code == 422
    assert missing_request.status_code == 422
    assert engine.lifecycle == [("session-a", request_id, "continue"), ("session-a", request_id, "terminal")]


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
    assert engine.lifecycle == [("session-a", engine.calls[0][0], "continue")]
