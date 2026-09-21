# Copyright (c) PyPTO Contributors.
# Licensed under CANN Open Software License Agreement Version 2.0.

import asyncio
import json
from types import SimpleNamespace

import pytest

from pypto_serving.config.types import GenerateConfig
from pypto_serving.router.app import _chat_message, _stream_chat
from pypto_serving.serving.pd.http_api import PrepareRequestHTTP, encode_json
from pypto_serving.serving.server.server import (
    ChatCompletionRequest,
    ChatMessage,
    ServingServer,
)


class _ExternalConfig:
    enabled = True


class _Engine:
    config = SimpleNamespace(pd_config=_ExternalConfig())
    pd_health_error = ""


def test_external_node_exposes_internal_api_but_not_public_generation() -> None:
    server = ServingServer(_Engine(), "model", GenerateConfig())
    paths = {route.path for route in server.app.routes}
    assert "/health" in paths
    assert "/internal/pd/descriptor" in paths
    assert "/internal/pd/prepare" in paths
    assert "/internal/pd/await-decode" in paths
    assert "/v1/completions" not in paths
    assert "/v1/chat/completions" not in paths


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

    def prepare_request(self, *args, **kwargs):
        self.prepare_calls.append((args, kwargs))
        return {"status": "prepared"}


class _ChatEngine(_Engine):
    def __init__(self) -> None:
        self.tokenizer = _ChatTokenizer()
        self.pd_service = _PDService()
        self.tokenize_calls = []

    def _tokenize_prompt(self, prompt: str) -> list[int]:
        self.tokenize_calls.append(prompt)
        return [101, 202, 303]


def test_pd_chat_streaming_prepare_tokenizes_templated_prompt() -> None:
    engine = _ChatEngine()
    server = ServingServer(
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
    wire = encode_json(
        PrepareRequestHTTP(
            request_id="chat-request",
            request_kind="chat",
            request_json=public.model_dump_json().encode(),
        )
    )

    response = asyncio.run(server._pd_prepare(_BodyRequest(wire)))

    assert response.status_code == 200
    assert engine.tokenize_calls == ["templated chat prompt"]
    args, kwargs = engine.pd_service.prepare_calls[0]
    request_id, prompt, config, prompt_token_ids = args
    assert request_id == "chat-request"
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
    server = ServingServer(engine, "model", GenerateConfig())
    public = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="question")],
        chat_template_kwargs={"enable_thinking": True},
        include_reasoning=False,
    )
    wire = encode_json(
        PrepareRequestHTTP(
            request_id="chat-hidden-reasoning",
            request_kind="chat",
            request_json=public.model_dump_json().encode(),
        )
    )

    asyncio.run(server._pd_prepare(_BodyRequest(wire)))

    spec = engine.pd_service.prepare_calls[0][1]["output_parser_spec"]
    assert spec.initial_state == "reasoning"
    assert spec.include_reasoning is False


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
        payload = chunk.removeprefix(b"data: ").strip()
        if payload == b"[DONE]":
            continue
        decoded.append(json.loads(payload))
    return decoded


def test_router_streams_independent_cumulative_reasoning_and_content() -> None:
    outputs = [
        SimpleNamespace(
            reasoning="plan",
            text="",
            finished=False,
            finish_reason="",
            prompt_tokens=3,
            completion_tokens=1,
        ),
        SimpleNamespace(
            reasoning="plan more",
            text="answer",
            finished=False,
            finish_reason="",
            prompt_tokens=3,
            completion_tokens=2,
        ),
        SimpleNamespace(
            reasoning="plan more",
            text="answer",
            finished=True,
            finish_reason="FINISHED_LENGTH",
            prompt_tokens=3,
            completion_tokens=3,
        ),
    ]

    async def collect():
        return [
            chunk
            async for chunk in _stream_chat(
                _RouterCoordinator(outputs), b"{}", "request", "model"
            )
        ]

    events = _decode_sse(asyncio.run(collect()))
    deltas = [event["choices"][0]["delta"] for event in events if event["choices"]]
    assert [delta["reasoning"] for delta in deltas] == ["plan", " more", None]
    assert [delta["content"] for delta in deltas] == ["", "answer", ""]


def test_router_non_streaming_message_preserves_both_semantic_channels() -> None:
    message = _chat_message(
        SimpleNamespace(reasoning="private reasoning", text="public answer")
    )

    assert message == {
        "role": "assistant",
        "reasoning": "private reasoning",
        "content": "public answer",
    }


def test_router_rejects_replay_that_changes_published_reasoning_prefix() -> None:
    outputs = [
        SimpleNamespace(
            reasoning="stable prefix",
            text="",
            finished=False,
            finish_reason="",
            prompt_tokens=1,
            completion_tokens=1,
        ),
        SimpleNamespace(
            reasoning="different",
            text="",
            finished=True,
            finish_reason="FINISHED_LENGTH",
            prompt_tokens=1,
            completion_tokens=2,
        ),
    ]

    async def collect():
        return [
            chunk
            async for chunk in _stream_chat(
                _RouterCoordinator(outputs), b"{}", "request", "model"
            )
        ]

    with pytest.raises(RuntimeError, match="reasoning output changed"):
        asyncio.run(collect())
