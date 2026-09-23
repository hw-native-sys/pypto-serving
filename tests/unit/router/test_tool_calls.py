# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Public PD Router tool-call shape without loading a model or device worker."""

import json

import pytest
from fastapi.testclient import TestClient

from pypto_serving.router.app import create_router_app
from pypto_serving.router.config import RouterConfig
from pypto_serving.router.coordinator import RouterCoordinator
from pypto_serving.serving.pd.protocol import DecodeOutputWire, HandoffKey
from pypto_serving.serving.reasoning import ParsedToolCall, ToolCallDelta


TOOLS = [{"type": "function", "function": {"name": "lookup"}}]


def _config(tmp_path):
    return RouterConfig(
        prefill_urls=("http://127.0.0.1:19101",),
        decode_urls=("http://127.0.0.1:19102",),
        run_id="tool-test",
        policy="round_robin",
        provider="mooncake",
        journal_path=str(tmp_path / "router.jsonl"),
        log_dir=str(tmp_path / "logs"),
    )


def _wire(request_id, *, sequence, finished=False, deltas=(), calls=()):
    return DecodeOutputWire(
        key=HandoffKey(request_id, "handoff", 1, 1, 1),
        token_id=101,
        text="",
        reasoning="Need data",
        finished=finished,
        finish_reason="FINISHED_EOS" if finished else "",
        prompt_tokens=4,
        completion_tokens=sequence,
        token_ids=(101, 102, 103) if finished else (),
        output_sequence=sequence,
        reasoning_delta="Need data" if sequence == 1 else "",
        tool_call_deltas=deltas,
        tool_calls=calls,
    )


def _sse_events(response):
    assert response.status_code == 200
    lines = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
    assert lines[-1] == "[DONE]"
    return [json.loads(line) for line in lines[:-1]]


def test_pd_router_formats_tool_calls_and_roundtrips_history(tmp_path, monkeypatch):
    seen = []

    async def no_reconcile(self):
        return None

    async def fake_generate(self, request_kind, request_json, request_id, publish_immediately=True):
        assert request_kind == "chat"
        seen.append(json.loads(request_json))
        calls = (ParsedToolCall("call-1", "lookup", '{"city":"杭州"}'),)
        if publish_immediately:
            yield _wire(request_id, sequence=1, deltas=(ToolCallDelta(0, "call-1", "lookup"),))
            yield _wire(request_id, sequence=2, deltas=(ToolCallDelta(0, arguments='{"city":'),))
            yield _wire(request_id, sequence=3, finished=True,
                        deltas=(ToolCallDelta(0, arguments='"杭州"}'),), calls=calls)
        else:
            yield _wire(request_id, sequence=3, finished=True, calls=calls)

    monkeypatch.setattr(RouterCoordinator, "reconcile_startup", no_reconcile)
    monkeypatch.setattr(RouterCoordinator, "generate", fake_generate)
    app = create_router_app(_config(tmp_path))
    payload = {
        "model": "model",
        "messages": [{"role": "user", "content": "Find a city"}],
        "tools": TOOLS,
    }
    with TestClient(app) as client:
        nonstream = client.post("/v1/chat/completions", json=payload)
        stream = client.post("/v1/chat/completions", json={
            **payload,
            "messages": [
                *payload["messages"],
                {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "prior", "type": "function",
                    "function": {"name": "lookup", "arguments": '{"city":"北京"}'},
                }]},
                {"role": "tool", "tool_call_id": "prior", "content": "Found"},
            ],
            "stream": True,
        })

    assert nonstream.status_code == 200
    choice = nonstream.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    assert choice["message"]["tool_calls"][0]["function"] == {
        "name": "lookup", "arguments": '{"city":"杭州"}',
    }
    assert seen[1]["messages"][1]["tool_calls"][0]["id"] == "prior"
    assert seen[1]["messages"][2]["tool_call_id"] == "prior"

    events = _sse_events(stream)
    chunks = [event["choices"][0] for event in events if event["choices"]]
    start = chunks[0]["delta"]["tool_calls"][0]
    assert start["id"] == "call-1"
    assert start["function"]["name"] == "lookup"
    fragments = [chunk["delta"]["tool_calls"][0]["function"].get("arguments", "") for chunk in chunks]
    assert "".join(fragments) == '{"city":"杭州"}'
    assert chunks[-1]["finish_reason"] == "tool_calls"
    assert events[-1]["usage"]["completion_tokens"] == 3


def test_pd_router_rejects_unsupported_tool_choice_before_stream(tmp_path, monkeypatch):
    async def no_reconcile(self):
        return None

    async def unexpected_generate(self, *_args, **_kwargs):
        raise AssertionError("request should fail before generation")
        yield

    monkeypatch.setattr(RouterCoordinator, "reconcile_startup", no_reconcile)
    monkeypatch.setattr(RouterCoordinator, "generate", unexpected_generate)
    with TestClient(create_router_app(_config(tmp_path))) as client:
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Find a city"}],
            "tools": TOOLS,
            "tool_choice": "required",
            "stream": True,
        })
    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")


@pytest.mark.parametrize("stream", [False, True])
def test_pd_router_respects_parallel_tool_calls_false(tmp_path, monkeypatch, stream):
    async def no_reconcile(self):
        return None

    async def fake_generate(self, request_kind, request_json, request_id, publish_immediately=True):
        assert request_kind == "chat"
        calls = (
            ParsedToolCall("call-1", "lookup", '{"city":"杭州"}'),
            ParsedToolCall("call-2", "lookup", '{"city":"北京"}'),
        )
        deltas = (
            ToolCallDelta(0, "call-1", "lookup", '{"city":"杭州"}'),
            ToolCallDelta(1, "call-2", "lookup", '{"city":"北京"}'),
        )
        yield _wire(request_id, sequence=1, finished=True, deltas=deltas, calls=calls)

    monkeypatch.setattr(RouterCoordinator, "reconcile_startup", no_reconcile)
    monkeypatch.setattr(RouterCoordinator, "generate", fake_generate)
    with TestClient(create_router_app(_config(tmp_path))) as client:
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Find cities"}],
            "tools": TOOLS,
            "parallel_tool_calls": False,
            "stream": stream,
        })

    if stream:
        choice = _sse_events(response)[0]["choices"][0]
        assert [call["index"] for call in choice["delta"]["tool_calls"]] == [0]
    else:
        assert response.status_code == 200
        choice = response.json()["choices"][0]
        assert [call["id"] for call in choice["message"]["tool_calls"]] == ["call-1"]
    assert choice["finish_reason"] == "tool_calls"


def test_pd_router_stream_reports_decode_parser_failure_as_sse(tmp_path, monkeypatch):
    async def no_reconcile(self):
        return None

    async def failing_generate(self, *_args, **_kwargs):
        raise RuntimeError("D Decode failed: invalid tool call")
        yield

    monkeypatch.setattr(RouterCoordinator, "reconcile_startup", no_reconcile)
    monkeypatch.setattr(RouterCoordinator, "generate", failing_generate)
    with TestClient(create_router_app(_config(tmp_path))) as client:
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Find a city"}],
            "tools": TOOLS,
            "stream": True,
        })
    events = _sse_events(response)
    assert events == [{"error": {
        "message": "D Decode failed: invalid tool call",
        "type": "pd_decode_error",
        "code": 503,
    }}]
