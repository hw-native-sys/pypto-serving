# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import asyncio
import json
from types import SimpleNamespace


from pypto_serving.config.types import GenerateConfig
from pypto_serving.serving.server.responses import stream_outputs
from pypto_serving.serving.server.chat_format import chat_message
from pypto_serving.serving.pd.config import PD_SCHEMA_VERSION, PDRole
from pypto_serving.serving.pd.generation import generation_body, public_document, request_fingerprint
from pypto_serving.serving.pd.protocol import GenerationContext, HandoffKey, PrefillResult
from pypto_serving.serving.pd.integration import PDApplication
from pypto_serving.serving.server.server import (
    ChatCompletionRequest,
    ChatMessage,
    ServingServer,
)


class _ExternalConfig:
    enabled = True
    role = PDRole.PREFILL
    node_id = "p"
    generation = route_epoch = control_incarnation = 1


class _Engine:
    pd_service = SimpleNamespace()


def test_pd_node_exposes_public_generation_and_only_operational_internal_api() -> None:
    server = _pd_server(_Engine(), "model", GenerateConfig())
    paths = {route.path for route in server.app.routes}
    assert "/health" in paths
    assert "/internal/pd/descriptor" in paths
    assert "/internal/pd/prepare" not in paths
    assert "/internal/pd/await-decode" not in paths
    assert "/v1/completions" in paths
    assert "/v1/chat/completions" in paths


def _pd_server(engine, model, config):
    def install(server):
        engine.pd_routes = PDApplication.install_routes(
            SimpleNamespace(service=engine.pd_service), server, config=_ExternalConfig()
        )
    return ServingServer(engine, model, config, route_factory=install)


def _body(raw, request_id):
    context = GenerationContext(
        HandoffKey(request_id, "handoff-" + request_id, 1, 1, 1),
        request_fingerprint("chat", public_document(raw)), "p", 1, "d", 1,
        "127.0.0.1", 12345, PD_SCHEMA_VERSION,
    )
    return generation_body(raw, "chat", context)


class _BodyRequest:
    def __init__(self, body: bytes) -> None:
        self._body = body

    async def body(self) -> bytes:
        return self._body


class _ChatTokenizer:
    output_parser_id = "deepseek_v4"

    def __init__(self) -> None:
        self.template_calls = []

    def apply_chat_template(self, messages, **kwargs) -> str:
        self.template_calls.append((messages, kwargs))
        return "templated chat prompt"


class _PDService:
    def __init__(self) -> None:
        self.prepare_calls = []

    def register_generation(self, context):
        return SimpleNamespace(context=context)

    def descriptor(self):
        return SimpleNamespace(capabilities=SimpleNamespace(model_revision="model"))

    def prepare_input(self, *args, **kwargs):
        self.prepare_calls.append((args, kwargs))
        return object()

    async def prefill(self, context, prepared):
        return PrefillResult(context.key, context.request_fingerprint, "READY")



class _ChatEngine(_Engine):
    def __init__(self) -> None:
        self.tokenizer = _ChatTokenizer()
        self.pd_service = _PDService()
        self.tokenize_calls = []

    def resolve_prompt_tokens(self, prompt: str, tokens=None) -> tuple[int, ...]:
        self.tokenize_calls.append(prompt)
        return (101, 202, 303)


def test_pd_chat_streaming_prepare_tokenizes_templated_prompt() -> None:
    engine = _ChatEngine()
    server = _pd_server(
        engine,
        "model",
        GenerateConfig(ignore_eos=False),
    )
    public = ChatCompletionRequest(
        model="model",
        messages=[ChatMessage(role="user", content="紫禁城")],
        max_tokens=8,
        stream=True,
    )
    wire = _body(public.model_dump_json().encode(), "chat-request")

    response = asyncio.run(engine.pd_routes._chat(_BodyRequest(wire)))

    assert response.status_code == 200
    assert engine.tokenize_calls == ["templated chat prompt"]
    args, kwargs = engine.pd_service.prepare_calls[0]
    prompt, config, prompt_token_ids = args
    assert prompt == "templated chat prompt"
    assert config.max_new_tokens == 8
    assert config.stream is True
    assert config.ignore_eos is False
    assert prompt_token_ids == (101, 202, 303)
    spec = kwargs["output_parser_spec"]
    assert spec.parser_id == "deepseek_v4"
    assert spec.initial_state == "content"


def test_pd_chat_prepare_preserves_include_reasoning_policy() -> None:
    engine = _ChatEngine()
    server = _pd_server(engine, "model", GenerateConfig())
    public = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="question")],
        chat_template_kwargs={"enable_thinking": True},
        include_reasoning=False,
    )
    wire = _body(public.model_dump_json().encode(), "chat-hidden-reasoning")

    asyncio.run(engine.pd_routes._chat(_BodyRequest(wire)))

    spec = engine.pd_service.prepare_calls[0][1]["output_parser_spec"]
    assert spec.initial_state == "reasoning"
    assert spec.include_reasoning is False


def test_pd_chat_prepare_preserves_tools_and_tool_history() -> None:
    engine = _ChatEngine()
    _pd_server(engine, "model", GenerateConfig())
    public = ChatCompletionRequest.model_validate({
        "messages": [
            {"role": "user", "content": "Find a city"},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call-1", "type": "function",
                "function": {"name": "lookup", "arguments": '{"city":"杭州"}'},
            }]},
            {"role": "tool", "tool_call_id": "call-1", "content": "Found"},
        ],
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
    })
    wire = _body(public.model_dump_json().encode(), "chat-tools")

    asyncio.run(engine.pd_routes._chat(_BodyRequest(wire)))

    messages, template_kwargs = engine.tokenizer.template_calls[0]
    assert messages[1]["content"] is None
    assert messages[1]["tool_calls"][0]["function"]["name"] == "lookup"
    assert messages[2]["tool_call_id"] == "call-1"
    assert template_kwargs["tools"][0]["function"]["name"] == "lookup"
    spec = engine.pd_service.prepare_calls[0][1]["output_parser_spec"]
    assert spec.tool_choice == "auto"
    assert spec.tool_names == ("lookup",)


class _RouterCoordinator:
    def __init__(self, outputs) -> None:
        self.outputs = outputs

    async def generate(self, request_kind, raw, request_id):
        assert request_kind == "chat"
        for output in self.outputs:
            yield output


def _decode_sse(chunks: list[bytes]) -> list[dict]:
    decoded = []
    for chunk in chunks:
        payload = chunk.removeprefix("data: ").strip()
        if payload == "[DONE]":
            continue
        decoded.append(json.loads(payload))
    return decoded


def test_router_streams_independent_reasoning_and_content_deltas() -> None:
    outputs = [
        SimpleNamespace(
            reasoning="plan",
            text="",
            reasoning_delta="plan",
            text_delta="",
            tool_call_deltas=(),
            tool_calls=(),
            finished=False,
            finish_reason="",
            prompt_tokens=3,
            completion_tokens=1,
        ),
        SimpleNamespace(
            reasoning="plan more",
            text="answer",
            reasoning_delta=" more",
            text_delta="answer",
            tool_call_deltas=(),
            tool_calls=(),
            finished=False,
            finish_reason="",
            prompt_tokens=3,
            completion_tokens=2,
        ),
        SimpleNamespace(
            reasoning="plan more",
            text="answer",
            reasoning_delta="",
            text_delta="",
            tool_call_deltas=(),
            tool_calls=(),
            finished=True,
            finish_reason="FINISHED_LENGTH",
            prompt_tokens=3,
            completion_tokens=3,
        ),
    ]

    async def collect():
        return [
            chunk
            async for chunk in stream_outputs(
                _RouterCoordinator(outputs).generate("chat", b"{}", "request"), "request", "model",
                kind="chat", parsed=True,
            )
        ]

    events = _decode_sse(asyncio.run(collect()))
    deltas = [event["choices"][0]["delta"] for event in events if event["choices"]]
    assert [delta["reasoning"] for delta in deltas] == ["plan", " more", None]
    assert [delta["content"] for delta in deltas] == ["", "answer", ""]


def test_router_non_streaming_message_preserves_both_semantic_channels() -> None:
    message = chat_message("public answer", "private reasoning", ())

    assert message.model_dump(exclude_none=True) == {
        "role": "assistant",
        "reasoning": "private reasoning",
        "content": "public answer",
    }


def test_router_streams_authoritative_deltas_when_cumulative_fields_change() -> None:
    outputs = [
        SimpleNamespace(
            reasoning="stable prefix",
            text="",
            reasoning_delta="stable prefix",
            text_delta="",
            tool_call_deltas=(),
            tool_calls=(),
            finished=False,
            finish_reason="",
            prompt_tokens=1,
            completion_tokens=1,
        ),
        SimpleNamespace(
            reasoning="different",
            text="answer",
            reasoning_delta="",
            text_delta="answer",
            tool_call_deltas=(),
            tool_calls=(),
            finished=True,
            finish_reason="FINISHED_LENGTH",
            prompt_tokens=1,
            completion_tokens=2,
        ),
    ]

    async def collect():
        return [
            chunk
            async for chunk in stream_outputs(
                _RouterCoordinator(outputs).generate("chat", b"{}", "request"), "request", "model",
                kind="chat", parsed=True,
            )
        ]

    events = _decode_sse(asyncio.run(collect()))
    deltas = [event["choices"][0]["delta"] for event in events if event["choices"]]
    assert [delta["reasoning"] for delta in deltas] == ["stable prefix", None]
    assert [delta["content"] for delta in deltas] == ["", "answer"]
