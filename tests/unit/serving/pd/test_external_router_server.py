# Copyright (c) PyPTO Contributors.
# Licensed under CANN Open Software License Agreement Version 2.0.

import asyncio
from types import SimpleNamespace

from pypto_serving.config.types import GenerateConfig
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
    def __init__(self) -> None:
        self.template_calls = []

    def apply_chat_template(self, messages, **kwargs) -> str:
        self.template_calls.append((messages, kwargs))
        return "templated chat prompt"


class _PDService:
    def __init__(self) -> None:
        self.prepare_calls = []

    def prepare_request(self, *args):
        self.prepare_calls.append(args)
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
    request_id, prompt, config, prompt_token_ids = engine.pd_service.prepare_calls[0]
    assert request_id == "chat-request"
    assert prompt == "templated chat prompt"
    assert config.max_new_tokens == 8
    assert config.stream is True
    assert config.ignore_eos is False
    assert prompt_token_ids == (101, 202, 303)
