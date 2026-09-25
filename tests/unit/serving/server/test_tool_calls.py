# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise the public tool-call contract without loading a model or worker."""

import asyncio
import json
from collections import deque
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from pypto_serving.config.types import GenerateConfig
from pypto_serving.model.tokenizer import DeepSeekV4TokenizerAdapter
from pypto_serving.serving.engine.async_engine import AsyncLLMEngine, EngineConfig, ReplicaEngineCore
from pypto_serving.serving.reasoning.deepseek_v4_tools import TOOL_END, TOOL_START
from pypto_serving.serving.sched.scheduler import RequestOutput, RequestStatus, SchedulerOutput
from pypto_serving.serving.server.server import ChatCompletionRequest, ServingServer
from tests.unit.serving.reasoning.test_deepseek_v4_tools import SplitRootTokenizer, ToolTokenizer, invoke


TOOLS = [{"type": "function", "function": {
    "name": "lookup", "description": "Look up a place",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
}}]


def _server():
    engine = SimpleNamespace(tokenizer=DeepSeekV4TokenizerAdapter(tokenizer=object()))
    return ServingServer(engine, "test", GenerateConfig())


def _request(**kwargs):
    return ChatCompletionRequest(messages=[{"role": "user", "content": "Question"}], **kwargs)


def test_tools_default_to_auto_and_freeze_only_names():
    server = _server()
    request = _request(tools=TOOLS)
    spec = server._output_parser_spec(request)
    assert spec.tool_choice == "auto"
    assert spec.tool_names == ("lookup",)
    assert server._output_parser_spec(_request()).tool_choice == "none"
    assert server._output_parser_spec(_request(tools=TOOLS, tool_choice="none")).tool_choice == "none"
    assert "strict" not in request.tools[0].model_dump(exclude_none=True)["function"]


@pytest.mark.parametrize("extra", [
    {"tools": TOOLS, "tool_choice": "required"},
    {"tools": TOOLS, "tool_choice": {"type": "function", "function": {"name": "lookup"}}},
    {"tools": [{"type": "function", "function": {"name": "lookup", "strict": True}}]},
    {"tools": TOOLS * 2},
    {"tool_choice": "auto"},
    {"chat_template_kwargs": {"tools": TOOLS}},
])
@pytest.mark.parametrize("stream", [False, True])
def test_unsupported_tools_fail_before_generation_or_sse_headers(extra, stream):
    with TestClient(_server().app) as client:
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Question"}], "stream": stream, **extra,
        })
    assert response.status_code == 400
    assert response.json()["object"] == "error"
    assert response.headers["content-type"].startswith("application/json")


def test_tool_request_requires_model_capability():
    server = _server()
    server.engine.tokenizer = SimpleNamespace(output_parser_id=None)
    with pytest.raises(ValueError, match="no tool-call parser"):
        server._output_parser_spec(_request(tools=TOOLS))


def test_chat_template_preserves_tool_history_and_reasoning():
    request = ChatCompletionRequest(messages=[
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": None, "reasoning": "Need data", "tool_calls": [
            {"id": "one", "type": "function", "function": {"name": "lookup", "arguments": '{"city":"杭州"}'}},
        ]},
        {"role": "tool", "tool_call_id": "one", "content": "Found"},
    ], reasoning_effort="high")
    server = _server()
    spec = server._output_parser_spec(request)
    assert spec.initial_state == "reasoning"
    prompt = server._apply_chat_template(request.messages, reasoning_effort=request.reasoning_effort)
    assert "Need data</think>" in prompt
    assert '<｜DSML｜invoke name="lookup">' in prompt
    assert "<tool_result>Found</tool_result>" in prompt


@pytest.mark.parametrize("message", [
    {"role": "user", "content": None},
    {"role": "tool", "content": "No ID"},
    {"role": "assistant", "tool_calls": [
        {"id": "one", "function": {"name": "lookup", "arguments": "[]"}},
    ]},
    {"role": "assistant", "tool_calls": [
        {"id": "one", "function": {"name": "lookup", "arguments": "broken"}},
    ]},
])
def test_invalid_tool_history_is_a_request_error(message):
    server = _server()
    with pytest.raises(ValueError):
        server._output_parser_spec(ChatCompletionRequest(messages=[message]))


@pytest.mark.parametrize("value,result", [
    ("before</｜DSML｜parameter>after", "Found"),
    ({"nested": "</｜DSML｜parameter>"}, "Found"),
    ("City", "before</tool_result>after"),
])
@pytest.mark.parametrize("stream", [False, True])
def test_tool_history_delimiters_fail_before_generation_or_sse_headers(value, result, stream):
    with TestClient(_server().app) as client:
        response = client.post("/v1/chat/completions", json={
            "messages": [
                {"role": "assistant", "tool_calls": [
                    {"id": "one", "type": "function", "function": {
                        "name": "lookup", "arguments": json.dumps({"value": value}),
                    }},
                ]},
                {"role": "tool", "tool_call_id": "one", "content": result},
            ],
            "stream": stream,
        })
    assert response.status_code == 400
    assert response.json()["object"] == "error"
    assert response.headers["content-type"].startswith("application/json")


class _ChatTokenizer(ToolTokenizer):
    output_parser_id = "deepseek_v4"
    eos_token_id = ToolTokenizer.vocab["<eos>"]
    bos_token_id = None

    def __init__(self):
        self.prompts = []

    def apply_chat_template(self, messages, **kwargs):
        prompt = DeepSeekV4TokenizerAdapter(self).apply_chat_template(messages, **kwargs)
        self.prompts.append(prompt)
        return prompt


class _SplitChatTokenizer(_ChatTokenizer):
    specials = SplitRootTokenizer.specials
    vocab = SplitRootTokenizer.vocab
    all_special_ids = SplitRootTokenizer.all_special_ids
    eos_token_id = vocab["<eos>"]


class _ReplayScheduler:
    """Replace model execution only; use real engine parsing and delivery."""

    def __init__(self, core, scripts, chunk_size):
        self.core = core
        self.scripts = deque(scripts)
        self.chunk_size = chunk_size
        self.requests = {}
        self.replays = {}
        self.aborted = []
        self.constraint_specs = []

    def add_request(self, request):
        self.constraint_specs.append(request.constraint_spec)
        text, reason = self.scripts.popleft()
        ids = self.core.tokenizer.encode(text) if isinstance(text, str) else text
        chunks = deque(ids[i:i + self.chunk_size] for i in range(0, len(ids), self.chunk_size))
        self.requests[request.request_id] = request
        self.replays[request.request_id] = (chunks, reason)
        asyncio.get_running_loop().call_soon(self._step, request.request_id)

    def _step(self, request_id):
        if request_id not in self.requests:
            return
        chunks, _ = self.replays[request_id]
        self.core._process_step_output(SchedulerOutput(scheduled_requests=[]), {request_id: chunks.popleft()})
        # Simulate a worker draining its release queue before the HTTP consumer
        # resumes. A late duplicate release cannot hide behind list membership.
        self.core.drain_frees()
        if request_id in self.requests:
            asyncio.get_running_loop().call_soon(self._step, request_id)

    def update_from_output(self, scheduler_output, new_tokens):
        outputs = []
        for request_id, ids in new_tokens.items():
            request = self.requests[request_id]
            request.output_token_ids.extend(ids)
            chunks, reason = self.replays[request_id]
            finished = not chunks
            if finished:
                self.requests.pop(request_id)
                self.replays.pop(request_id)
                request.status = RequestStatus[reason]
            outputs.append(RequestOutput(
                request_id=request_id, new_token_id=ids[-1], finished=finished,
                finish_reason=reason if finished else "",
            ))
        return outputs

    def abort_request(self, request_id):
        self.aborted.append(request_id)
        self.requests.pop(request_id, None)
        self.replays.pop(request_id, None)


class _ReplayCore(ReplicaEngineCore):
    def __init__(self, *, tokenizer, config, scripts, chunk_size):
        # Deliberately never construct/start a worker or device cache.
        self.tokenizer = tokenizer
        self.config = config
        self._request_contexts = {}
        self._pending_free_ids = []
        self.freed = []
        self.scheduler = _ReplayScheduler(self, scripts, chunk_size)

    def _record_scheduler_stats(self, output=None):
        pass

    def drain_frees(self):
        self.freed.extend(self._pending_free_ids)
        self._pending_free_ids.clear()


def _replay_server(*scripts, chunk_size=7, tokenizer=None):
    engine = AsyncLLMEngine(
        EngineConfig(model_id="test"), tokenizer or _ChatTokenizer(),
        core_factory=lambda **kwargs: _ReplayCore(**kwargs, scripts=scripts, chunk_size=chunk_size),
    )
    return ServingServer(engine, "test", GenerateConfig(max_new_tokens=2048))


def _enable_fake_constraints(server, monkeypatch):
    server.engine.config.executor_cls = "PyptoDeepSeekV4DSparkExecutor"
    monkeypatch.setattr(
        "pypto_serving.serving.server.server.importlib.util.find_spec",
        lambda name: object() if name == "xgrammar" else None,
    )

    async def preflight(spec):
        assert spec.provider_id == "xgrammar"

    monkeypatch.setattr(server, "_preflight_constraint", preflight)


def _sse_events(response):
    assert response.status_code == 200
    lines = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
    assert lines[-1] == "[DONE]"
    return [json.loads(line) for line in lines[:-1]]


def _collect_chat(response, stream):
    if not stream:
        assert response.status_code == 200, response.text
        body = response.json()
        return body["choices"][0]["message"], body["choices"][0]["finish_reason"], body["usage"]
    events = _sse_events(response)
    assert not any("error" in event for event in events), events
    calls = {}
    content, reasoning = [], []
    reasons = []
    usages = []
    for event in events:
        if not event["choices"]:
            usages.append(event["usage"])
            continue
        choice = event["choices"][0]
        delta = choice["delta"]
        content.append(delta["content"] or "")
        reasoning.append(delta["reasoning"] or "")
        if choice["finish_reason"]:
            reasons.append(choice["finish_reason"])
        for update in delta.get("tool_calls", []):
            index = update["index"]
            if index not in calls:
                assert set(update) == {"index", "id", "type", "function"}
                assert update["function"]["name"]
                calls[index] = {"id": update["id"], "type": update["type"], "function": {
                    "name": update["function"]["name"], "arguments": "",
                }}
            else:
                assert "id" not in update and "type" not in update
                assert "name" not in update["function"]
            calls[index]["function"]["arguments"] += update["function"].get("arguments", "")
    assert len(reasons) == len(usages) == 1
    message = {"role": "assistant", "content": "".join(content) or (None if calls else ""),
               "reasoning": "".join(reasoning) or None}
    if calls:
        assert sorted(calls) == list(range(len(calls)))
        message["tool_calls"] = list(calls.values())
    return message, reasons[0], usages[0]


def _assert_released(server, count):
    core = server.engine._cores[0]
    core.drain_frees()
    assert len(core.freed) == len(set(core.freed)) == count
    assert not core._request_contexts
    assert not core.scheduler.requests
    assert not server.engine._request_to_replica
    assert server.engine._route_extra_load == [0]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 7, 100000])
@pytest.mark.parametrize("tokenizer_class", [_ChatTokenizer, _SplitChatTokenizer])
def test_http_tools_roundtrip_runs_real_output_delivery(stream, chunk_size, tokenizer_class):
    call_text = "Need data</think>" + TOOL_START + invoke(city="杭州") + TOOL_END + "<eos>"
    second_call = "Need more data" + TOOL_START + invoke(city="北京") + TOOL_END + "<eos>"
    answer = "Now known</think>Two results.<eos>"
    server = _replay_server(
        (call_text, "FINISHED_EOS"), (second_call, "FINISHED_EOS"), (answer, "FINISHED_EOS"),
        chunk_size=chunk_size, tokenizer=tokenizer_class(),
    )
    messages = [{"role": "user", "content": "Compare two cities"}]
    with TestClient(server.app) as client:
        for city in ("杭州", "北京"):
            response = client.post("/v1/chat/completions", json={
                "messages": messages, "tools": TOOLS, "stream": stream, "reasoning_effort": "high",
            })
            message, reason, usage = _collect_chat(response, stream)
            assert reason == "tool_calls"
            assert message["content"] is None
            assert message["reasoning"] == ("Need data" if city == "杭州" else "Need more data")
            call = message["tool_calls"][0]
            assert json.loads(call["function"]["arguments"]) == {"city": city}
            assert usage["completion_tokens"] > 0
            assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
            # The client, never Serving, executes the function and sends the result.
            result = {"city": city, "found": True}
            messages.extend([message, {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)}])
        response = client.post("/v1/chat/completions", json={
            "messages": messages, "stream": stream, "reasoning_effort": "high",
        })
        message, reason, _ = _collect_chat(response, stream)
        assert message["content"] == "Two results."
        assert message["reasoning"] == "Now known"
        assert "tool_calls" not in message
        assert reason == "stop"
    assert "Need data</think>" in server.engine.tokenizer.prompts[1]
    assert "Need more data</think>" in server.engine.tokenizer.prompts[2]
    assert server.engine.tokenizer.prompts[2].count("<tool_result>") == 2
    _assert_released(server, 3)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("choice", ["auto", "none"])
def test_http_tool_modes_and_parallel_filter(stream, parallel, choice):
    text = "reason</think>before" + TOOL_START + invoke(city="one") + invoke(city="two") + TOOL_END + "after"
    server = _replay_server((text, "FINISHED_EOS"))
    with TestClient(server.app) as client:
        message, reason, _ = _collect_chat(client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Question"}], "stream": stream, "tools": TOOLS,
            "tool_choice": choice, "parallel_tool_calls": parallel,
            "reasoning_effort": "high", "include_reasoning": False,
        }), stream)
    assert message["content"] == "beforeafter"
    assert message["reasoning"] is None
    if choice == "none":
        assert "tool_calls" not in message
        assert reason == "stop"
    else:
        assert len(message["tool_calls"]) == (2 if parallel else 1)
        assert reason == "tool_calls"
    _assert_released(server, 1)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("complete", [False, True])
@pytest.mark.parametrize("reason", ["FINISHED_LENGTH", "FINISHED_ABORTED"])
def test_http_truncation_never_claims_tool_success(stream, complete, reason):
    text = TOOL_START + invoke(city="unfinished") + TOOL_END
    if not complete:
        text = text[:text.index("unfinished") + 3]
    server = _replay_server((text, reason))
    with TestClient(server.app) as client:
        message, finish_reason, _ = _collect_chat(client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Question"}], "stream": stream, "tools": TOOLS,
        }), stream)
    assert finish_reason == ("length" if reason == "FINISHED_LENGTH" else "aborted")
    arguments = message["tool_calls"][0]["function"]["arguments"]
    if complete:
        assert json.loads(arguments) == {"city": "unfinished"}
    else:
        assert arguments == '{"city":"unf'
        with pytest.raises(ValueError):
            json.loads(arguments)
    _assert_released(server, 1)


@pytest.mark.parametrize("stream", [False, True])
def test_model_tool_name_outside_request_is_returned_for_client_validation(stream):
    text = TOOL_START + invoke("read_file", path="notes.txt") + TOOL_END + "<eos>"
    server = _replay_server((text, "FINISHED_EOS"), chunk_size=7)
    with TestClient(server.app) as client:
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Read notes"}],
            "tools": TOOLS,
            "stream": stream,
        })
        message, reason, _ = _collect_chat(response, stream)
    assert reason == "tool_calls"
    assert len(message["tool_calls"]) == 1
    assert message["tool_calls"][0]["function"]["name"] == "read_file"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"path": "notes.txt"}
    _assert_released(server, 1)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(("invalid", "error"), [
    (TOOL_START + invoke(city="one"), "incomplete DSML"),
    (TOOL_START + invoke(value=123).replace(">123</", ">broken</") + TOOL_END, "Expecting value"),
])
def test_parser_error_is_request_local_and_next_request_succeeds(stream, invalid, error):
    server = _replay_server((invalid, "FINISHED_EOS"), ("Healthy<eos>", "FINISHED_EOS"))
    payload = {"messages": [{"role": "user", "content": "Question"}], "tools": TOOLS, "stream": stream}
    with TestClient(server.app) as client:
        response = client.post("/v1/chat/completions", json=payload)
        if stream:
            events = _sse_events(response)
            assert error in events[-1]["error"]["message"]
            assert not any(event.get("usage") for event in events)
        else:
            assert response.status_code == 400
            assert error in response.json()["message"]
        message, reason, _ = _collect_chat(client.post("/v1/chat/completions", json=payload), stream)
        assert message["content"] == "Healthy"
        assert "tool_calls" not in message and reason == "stop"
    _assert_released(server, 2)


@pytest.mark.parametrize("constrained", [False, True])
def test_closing_http_stream_releases_active_tool_request_once(monkeypatch, constrained):
    async def check():
        text = TOOL_START + invoke(city="x" * 10000) + TOOL_END
        server = _replay_server((text, "FINISHED_EOS"), chunk_size=7)
        if constrained:
            _enable_fake_constraints(server, monkeypatch)
        tools = [{"type": "function", "function": {
            **TOOLS[0]["function"], "strict": True,
        }}] if constrained else TOOLS
        request = _request(tools=tools, stream=True,
                           tool_choice="required" if constrained else "auto")
        response = await server._chat_completions(request)
        chunks = response.body_iterator
        async for chunk in chunks:
            event = json.loads(chunk[6:])
            if event["choices"][0]["delta"].get("tool_calls"):
                break
        await chunks.aclose()
        _assert_released(server, 1)
        assert len(server.engine._cores[0].scheduler.aborted) == 1
        assert bool(server.engine._cores[0].scheduler.constraint_specs[0]) is constrained

    asyncio.run(check())


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 100000])
def test_constrained_chat_protocol_reasoning_to_parallel_tools(monkeypatch, stream, chunk_size):
    text = "Need both</think>" + TOOL_START + invoke(city="杭州") + invoke(city="北京") + TOOL_END + "<eos>"
    server = _replay_server((text, "FINISHED_EOS"), chunk_size=chunk_size)
    _enable_fake_constraints(server, monkeypatch)
    strict_tool = {"type": "function", "function": {
        **TOOLS[0]["function"], "strict": True,
        "parameters": {
            "type": "object", "properties": {"city": {"type": "string"}},
            "required": ["city"], "additionalProperties": False,
        },
    }}
    with TestClient(server.app) as client:
        message, reason, _ = _collect_chat(client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Look up two cities"}],
            "tools": [strict_tool], "tool_choice": "required",
            "parallel_tool_calls": True, "reasoning_effort": "high", "stream": stream,
        }), stream)
    assert message["reasoning"] == "Need both"
    assert message["content"] is None
    assert reason == "tool_calls"
    assert [json.loads(call["function"]["arguments"])["city"] for call in message["tool_calls"]] == [
        "杭州", "北京",
    ]
    assert server.engine._cores[0].scheduler.constraint_specs[0].tool_choice == "required"
    _assert_released(server, 1)


@pytest.mark.parametrize("stream", [False, True])
def test_constrained_chat_length_truncation_keeps_partial_call(monkeypatch, stream):
    text = "Need a lookup</think>" + TOOL_START + invoke(city="unfinished")
    text = text[:text.index("unfinished") + 3]
    server = _replay_server((text, "FINISHED_LENGTH"), chunk_size=3)
    _enable_fake_constraints(server, monkeypatch)
    strict_tool = {"type": "function", "function": {**TOOLS[0]["function"], "strict": True}}
    with TestClient(server.app) as client:
        message, reason, _ = _collect_chat(client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Look up a city"}],
            "tools": [strict_tool], "tool_choice": "required",
            "reasoning_effort": "high", "stream": stream,
        }), stream)
    assert message["reasoning"] == "Need a lookup"
    assert reason == "length"
    assert message["tool_calls"][0]["function"]["arguments"] == '{"city":"unf'
    assert server.engine._cores[0].scheduler.constraint_specs[0] is not None
    _assert_released(server, 1)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("tokenizer_class", [_ChatTokenizer, _SplitChatTokenizer])
def test_real_delivery_holds_utf8_and_distinguishes_literal_markers(stream, tokenizer_class):
    tokenizer = tokenizer_class()
    literal = "</think>secret"
    ids = [1000 + ord(char) for char in literal]
    ids += [tokenizer.vocab["</think>"], 700, 701]
    ids += tokenizer.encode(TOOL_START + invoke(city="杭州") + TOOL_END + "<eos>")
    # First K7-like burst ends in an unfinished multi-token character.
    server = _replay_server((ids, "FINISHED_EOS"), chunk_size=len(literal) + 2, tokenizer=tokenizer)
    with TestClient(server.app) as client:
        message, reason, _ = _collect_chat(client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Question"}], "tools": TOOLS,
            "reasoning_effort": "high", "stream": stream,
        }), stream)
    assert message["reasoning"] == literal
    assert message["content"] == "好"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"city": "杭州"}
    assert reason == "tool_calls"
    _assert_released(server, 1)


@pytest.mark.parametrize("stream", [False, True])
def test_completions_remains_unparsed(stream):
    text = "reason</think>" + TOOL_START + invoke(city="one") + TOOL_END
    server = _replay_server((text, "FINISHED_LENGTH"))
    expected = server.engine.tokenizer.decode(server.engine.tokenizer.encode(text))
    with TestClient(server.app) as client:
        response = client.post("/v1/completions", json={"prompt": "Question", "stream": stream})
    if stream:
        events = _sse_events(response)
        actual = "".join(event["choices"][0]["text"] for event in events if event["choices"])
    else:
        assert response.status_code == 200
        actual = response.json()["choices"][0]["text"]
    assert actual == expected
