# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate optional Agent session identity without changing Chat responses."""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from pypto_serving.config.types import GenerateConfig
from pypto_serving.serving.agentinfer.agent_identity import parse_agent_identity
from pypto_serving.serving.agentinfer.agentinfer_admission import AgentInferAdmission
from pypto_serving.serving.engine.async_engine import AsyncLLMEngine, EngineConfig, TokenOutput
from pypto_serving.serving.memory.kv_cache import KvCacheManager
from pypto_serving.serving.reasoning import ParsedToolCall
from pypto_serving.serving.sched.scheduler import Request, Scheduler, SchedulerConfig
from pypto_serving.serving.server.server import ServingServer


class _Tokenizer:
    eos_token_id = 0
    bos_token_id = 1

    @staticmethod
    def encode(text):
        return [1] * len(text.split())

    @staticmethod
    def apply_chat_template(messages, **kwargs):
        return " ".join(message["content"] or "" for message in messages)


class _Engine:
    tokenizer = _Tokenizer()

    def __init__(self):
        self.calls = []
        self.lifecycle = []
        self.finish_reason = "FINISHED_EOS"
        self.tool_calls = ()

    def observe_agent_lifecycle(self, session_id, request_id, lifecycle):
        self.lifecycle.append((session_id, request_id, lifecycle))
        return lifecycle == "terminal"

    async def add_request(self, request_id, prompt, config, **kwargs):
        self.calls.append((request_id, prompt, kwargs))
        yield TokenOutput(
            text="OK", finished=True, finish_reason=self.finish_reason, tool_calls=self.tool_calls,
            prompt_tokens=4, completion_tokens=1,
        )


@pytest.fixture
def engine():
    return _Engine()


@pytest.fixture
def client(engine):
    server = ServingServer(engine, "test-model", GenerateConfig())
    with TestClient(server.app) as client:
        yield client


@pytest.fixture
def enforce_engine():
    engine = AsyncLLMEngine(EngineConfig(agentinfer_mode="enforce"), tokenizer=_Tokenizer())
    manager = KvCacheManager(num_blocks=64, block_size=16)
    admission = AgentInferAdmission(0, Scheduler(SchedulerConfig(), manager), manager)
    engine.single_core()._agentinfer_admission = admission
    return engine, admission


def _chat(client, *, headers=None, messages=None, **body):
    if messages is None:
        messages = [{"role": "user", "content": "Hello"}]
    return client.post("/v1/chat/completions", headers=headers, json={"messages": messages, **body})


def test_text_action_rounds_share_identity_and_plain_chat_stays_unchanged(client, engine):
    first_turn = {"role": "user", "content": "Read the file"}
    next_turn = [
        first_turn,
        {"role": "assistant", "content": "cat input.txt"},
        {"role": "user", "content": "Tool output: hello"},
    ]
    responses = [
        _chat(client, messages=[first_turn], agent_hint={"session_id": "run-123"}),
        _chat(client, messages=next_turn, agent_hint={"session_id": "run-123"}),
        _chat(client, agent_hint={"session_id": "run-456"}),
        _chat(client, headers={"X-Other-Session": "unknown"}),
    ]
    assert all(response.status_code == 200 for response in responses)
    payloads = [response.json() for response in responses]
    assert len({payload["id"] for payload in payloads}) == len(payloads)
    assert all(payload["choices"][0]["message"]["content"] == "OK" for payload in payloads)
    assert [kwargs.get("agent_session_id") for _, _, kwargs in engine.calls] == [
        "run-123", "run-123", "run-456", None,
    ]
    assert "agent_session_id" not in engine.calls[-1][2]
    assert all("agent_hint" not in payload for payload in payloads)
    assert engine.lifecycle == []


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("metadata, session", [
    ({"headers": {"X-Session-Id": "run-123"}}, "run-123:lead"),
    ({"agent_hint": {"session_id": "run-123"}}, "run-123"),
])
def test_chat_carries_session_identity(client, engine, stream, metadata, session):
    response = _chat(client, stream=stream, **metadata)
    assert response.status_code == 200
    if stream:
        assert "data: [DONE]" in response.text
    assert engine.calls[0][2]["agent_session_id"] == session


@pytest.mark.parametrize("stream", [False, True])
def test_completion_headers_enter_agent_admission(client, engine, stream):
    body = {"prompt": "Reply OK.", "stream": stream}
    tagged = client.post("/v1/completions", headers={"X-Session-Id": "text-agent"}, json=body)
    plain = client.post("/v1/completions", json=body)
    assert tagged.status_code == plain.status_code == 200
    assert engine.calls[0][2]["agent_session_id"] == "text-agent:lead"
    assert engine.calls[0][2]["agent_identity"].program_id == "text-agent:lead"
    assert "agent_session_id" not in engine.calls[1][2]


def test_disabled_agentinfer_does_not_parse_framework_headers(client, engine):
    engine.config = SimpleNamespace(agentinfer_mode="disabled")
    headers = {"X-Session-Id": "invalid session"}
    chat = _chat(client, headers=headers)
    completion = client.post("/v1/completions", headers=headers, json={"prompt": "Hello"})
    assert chat.status_code == completion.status_code == 200
    assert all("agent_identity" not in kwargs for _, _, kwargs in engine.calls)


@pytest.mark.parametrize("headers, program", [
    ({"X-Session-ID": "run-123"}, "run-123:lead"),
    ({"Session-ID": "run-123", "Thread-ID": "thread-1"}, "run-123:thread-1"),
    ({"X-Claude-Code-Session-Id": "run-123"}, "run-123:lead"),
    ({"X-DeepSeek-Harness-Session-Id": "run-123"}, "run-123:lead"),
    ({"X-OpenCode-Session-Id": "run-123", "X-Session-Id": "generic-run"}, "run-123:lead"),
    ({"X-OpenCode-Session": "run-123"}, "run-123:lead"),
    ({"X-Claude-Code-Session-Id": "run-123", "X-Claude-Code-Agent-Id": "worker-1"}, "run-123:worker-1"),
])
def test_framework_headers_map_to_programs(headers, program):
    assert parse_agent_identity(headers=headers).program_id == program




def test_deepseek_harness_compaction_is_not_mistaken_for_agent_progress(client, engine):
    response = _chat(client, headers={"X-DeepSeek-Harness-Session-Id": "run-a", "X-DeepSeek-Harness-Compact": "1"})
    assert response.status_code == 200
    assert "agent_session_id" not in engine.calls[0][2]


def test_canonical_agent_context_precedes_hint_and_headers(client, engine):
    context = {
        "program_id": "task-1:worker", "task_id": "task-1", "session_id": "agent-1",
        "agent_id": "worker", "parent_program_id": "task-1:lead",
        "blocks_parent": True, "expected_resume": False,
    }
    response = _chat(
        client, headers={"X-Session-Id": "header-run"}, agent_hint={"session_id": "hint-run"},
        vllm_xargs={"agentic_context": context},
    )
    assert response.status_code == 200
    kwargs = engine.calls[0][2]
    assert kwargs["agent_session_id"] == "task-1:worker"
    identity = kwargs["agent_identity"]
    assert identity.parent_program_id == "task-1:lead"
    assert identity.blocks_parent is True
    assert identity.expected_resume is False




def test_explicit_hint_preserves_parent_and_precedes_header(client, engine):
    response = _chat(
        client, headers={"X-Session-Id": "header-run"},
        agent_hint={"session_id": "worker-1", "parent_session_id": "root-1"},
    )
    assert response.status_code == 200
    kwargs = engine.calls[0][2]
    assert kwargs["agent_session_id"] == "worker-1"
    identity = kwargs["agent_identity"]
    assert (identity.program_id, identity.parent_program_id, identity.blocks_parent) == ("worker-1", "root-1", True)


@pytest.mark.parametrize("metadata", [
    {"headers": {"X-Session-ID": "invalid session"}},
    {"headers": {"X-Session-ID": "a" * 128}},
    {"vllm_xargs": {"agentic_context": "not-json"}},
    {"vllm_xargs": {"agentic_context": {"program_id": "invalid id"}}},
    {"agent_hint": {"session_id": "worker", "blocks_parent": True}},
    {"agent_hint": {}},
    {"agent_hint": {"session_id": ""}},
    {"agent_hint": {"session_id": "with space"}},
    {"agent_hint": {"session_id": "a" * 129}},
    {"agent_hint": {"session_id": 42}},
    {"agent_hint": {"session_id": "valid", "unexpected": True}},
    {"agent_hint": {"session_id": "valid", "lifecycle_mode": "terminal"}},
])
def test_invalid_agent_identity_is_rejected_before_generation(client, engine, metadata):
    assert _chat(client, **metadata).status_code == 422
    assert engine.calls == []


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("finish_reason, mode, tool_calls, expected", [
    ("FINISHED_EOS", "explicit", (), None),
    ("FINISHED_EOS", "chat", (), "terminal"),
    ("FINISHED_LENGTH", "chat", (), None),
    ("FINISHED_EOS", "chat", (ParsedToolCall(id="call-1", name="read_file", arguments="{}", complete=True),),
     "continue"),
], ids=["explicit", "stop", "length", "tool-call"])
def test_chat_response_lifecycle(client, engine, stream, finish_reason, mode, tool_calls, expected):
    engine.finish_reason = finish_reason
    engine.tool_calls = tool_calls
    response = _chat(client, stream=stream, agent_hint={"session_id": "session-a", "lifecycle_mode": mode})
    assert response.status_code == 200
    assert engine.lifecycle == ([("session-a", engine.calls[0][0], expected)] if expected else [])


def test_invalid_agent_lifecycle_is_rejected(client, engine):
    invalid = client.post("/v1/agent/lifecycle", json={
        "session_id": "session-a", "request_id": "request-a", "lifecycle": "finished",
    })
    missing_request = client.post("/v1/agent/lifecycle", json={"session_id": "session-a", "lifecycle": "terminal"})
    assert invalid.status_code == missing_request.status_code == 422
    assert engine.lifecycle == []


@pytest.mark.parametrize("stream", [False, True])
def test_lifecycle_endpoint_reaches_real_engine_and_admission(stream, monkeypatch, enforce_engine):
    engine, admission = enforce_engine
    core = engine.single_core()

    async def complete_without_device(request_id, prompt, config, *, on_queued, prompt_token_ids, **kwargs):
        request = Request(request_id, prompt_token_ids, max_new_tokens=1)
        assert admission.arrive(request, kwargs["agent_identity"])
        on_queued()
        admission.complete(request)
        yield TokenOutput(
            text="OK", finished=True, finish_reason="FINISHED_LENGTH",
            prompt_tokens=len(prompt_token_ids), completion_tokens=1,
        )

    # Keep both engine layers and admission real; substitute only model execution.
    monkeypatch.setattr(core, "add_request", complete_without_device)
    server = ServingServer(engine, "test-model", GenerateConfig())
    with TestClient(server.app) as client:
        for lifecycle in ("continue", "terminal"):
            response = _chat(client, stream=stream, agent_hint={"session_id": "session-a"})
            assert response.status_code == 200
            assert admission.policy.registry.has_programs
            if stream:
                chunks = [json.loads(line[6:]) for line in response.text.splitlines()
                          if line.startswith("data: {")]
                request_id = chunks[0]["id"]
                assert any(choice.get("finish_reason") == "length"
                           for chunk in chunks for choice in chunk["choices"])
                assert "data: [DONE]" in response.text
            else:
                request_id = response.json()["id"]
                assert response.json()["choices"][0]["finish_reason"] == "length"
            notification = client.post("/v1/agent/lifecycle", json={
                "session_id": "session-a", "request_id": request_id, "lifecycle": lifecycle,
            })
            assert notification.status_code == 200
            assert notification.json() == {"released": lifecycle == "terminal"}
            assert admission.policy.registry.has_programs is (lifecycle == "continue")
    assert engine._request_to_replica == {}


@pytest.mark.parametrize("completion_order", [(0, 1), (1, 0)])
def test_automatic_chat_terminal_releases_after_all_overlapping_responses(completion_order, enforce_engine):
    engine, admission = enforce_engine
    server = ServingServer(engine, "test-model", GenerateConfig())
    requests = [Request(f"request-{index}", [1, 2], max_new_tokens=1) for index in range(2)]
    for request in requests:
        assert admission.arrive(request, "session-a")
    output = TokenOutput(text="OK", finished=True, finish_reason="FINISHED_EOS")

    for position, index in enumerate(completion_order):
        request = requests[index]
        admission.complete(request)
        server._observe_chat_lifecycle("session-a", request.request_id, "chat", output)
        assert admission.policy.registry.has_programs is (position == 0)
    assert admission._sessions == {}
