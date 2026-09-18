# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""HTTP-layer guards for OpenAI-style token ``usage`` reporting.

The vLLM ``bench serve`` parser reads ``usage.completion_tokens`` from the
terminal SSE chunk (empty ``choices``) that OpenAI emits under
``stream_options.include_usage``. These tests drive ``ServingServer`` with a
fake engine so the usage accounting is exercised without a model or NPU.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from pypto_serving.config.types import GenerateConfig
from pypto_serving.serving.engine.async_engine import TokenOutput
from pypto_serving.serving.pd.metrics import token_ids_sha256
from pypto_serving.serving.reasoning import OutputParserSpec
from pypto_serving.serving.server.server import (
    ChatCompletionRequest,
    ChatMessage,
    CompletionRequest,
    ServingServer,
)

# ---------------------------------------------------------------------------
# Fake engine
# ---------------------------------------------------------------------------


class _FakeEngine:
    """Minimal engine stub: yields caller-supplied TokenOutputs."""

    def __init__(self, outputs: list[TokenOutput]) -> None:
        self._outputs = outputs
        self.calls = []

    class _FakeTokenizer:
        def encode(self, text: str) -> list[int]:
            return [1] * max(1, len(text.split()))

        def apply_chat_template(self, messages, **kwargs):
            return " ".join(m["content"] for m in messages)

    tokenizer = _FakeTokenizer()

    async def add_request(self, request_id, prompt, config, **kwargs):
        self.calls.append((request_id, prompt, kwargs))
        for out in self._outputs:
            yield out


class _RejectingEngine(_FakeEngine):
    async def add_request(self, request_id, prompt, config, **kwargs):
        raise ValueError("uncached prompt length 5 exceeds the single-dispatch limit")
        yield  # pragma: no cover


def _make_server(outputs: list[TokenOutput]) -> ServingServer:
    return ServingServer(async_engine=_FakeEngine(outputs), model_id="test-model", generate_config=GenerateConfig())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_sse(raw: bytes) -> list[dict]:
    """Parse ``data: ...\n\n`` blocks; skip [DONE]."""
    chunks = []
    for line in raw.decode().splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            continue
        chunks.append(json.loads(payload))
    return chunks


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method_name", "route_request"),
    [
        ("_completions", CompletionRequest(prompt="cached prefix", stream=False)),
        (
            "_chat_completions",
            ChatCompletionRequest(
                messages=[ChatMessage(role="user", content="cached prefix")],
                stream=False,
            ),
        ),
    ],
)
def test_non_streaming_scheduler_rejection_reaches_http_400_handler(method_name, route_request):
    server = ServingServer(
        async_engine=_RejectingEngine([]),
        model_id="test-model",
        generate_config=GenerateConfig(),
    )
    message = "uncached prompt length 5 exceeds the single-dispatch limit"

    with pytest.raises(ValueError, match="uncached prompt length 5"):
        asyncio.run(getattr(server, method_name)(route_request))

    handler = server.app.exception_handlers[ValueError]
    response = asyncio.run(handler(None, ValueError(message)))

    assert response.status_code == 400
    assert json.loads(response.body)["message"] == message


def test_stream_completion_terminal_usage_chunk():
    """Final SSE chunk must have empty choices and correct usage counts."""
    outputs = [
        TokenOutput(text="Hello", token_id=100, prompt_tokens=5, completion_tokens=1),
        TokenOutput(text="Hello world", token_id=101, prompt_tokens=5, completion_tokens=2),
        TokenOutput(
            text="Hello world!",
            token_id=102,
            finished=True,
            finish_reason="FINISHED_EOS",
            prompt_tokens=5,
            completion_tokens=3,
        ),
    ]
    server = _make_server(outputs)

    from pypto_serving.config.types import GenerateConfig
    async def collect():
        chunks = []
        async for data in server._stream_completion(
            "req-0", "hello prompt here go", GenerateConfig(max_new_tokens=3), "test-model"
        ):
            chunks.append(data)
        return chunks

    raw = b"".join(c.encode() for c in asyncio.run(collect()))
    parsed = _parse_sse(raw)

    # At least 3 delta chunks + 1 usage chunk
    assert len(parsed) >= 4, f"Expected ≥4 chunks, got {len(parsed)}: {parsed}"

    usage_chunks = [c for c in parsed if not c["choices"]]
    assert len(usage_chunks) == 1, f"Expected exactly 1 usage chunk, got: {usage_chunks}"

    u = usage_chunks[0]["usage"]
    assert u["prompt_tokens"] == 5
    assert u["completion_tokens"] == 3
    assert u["total_tokens"] == 8

    # Intermediate chunks must NOT carry usage
    delta_chunks = [c for c in parsed if c["choices"]]
    for c in delta_chunks:
        assert c.get("usage") is None, f"Intermediate chunk has usage: {c}"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("chat", [False, True])
def test_http_eos_uses_standard_stop_finish_reason(stream, chat):
    server = _make_server([
        TokenOutput(text="Done", finished=True, finish_reason="FINISHED_EOS",
                    prompt_tokens=2, completion_tokens=1),
    ])
    path = "/v1/chat/completions" if chat else "/v1/completions"
    payload = {"messages": [{"role": "user", "content": "Go"}]} if chat else {"prompt": "Go"}
    with TestClient(server.app) as client:
        response = client.post(path, json={**payload, "stream": stream})
    assert response.status_code == 200
    choices = _parse_sse(response.content) if stream else [response.json()]
    assert [item["choices"][0]["finish_reason"] for item in choices if item["choices"]][-1] == "stop"


def test_completion_accepts_exact_prompt_token_ids() -> None:
    engine = _FakeEngine(
        [
            TokenOutput(
                text="ok",
                token_ids=(77,),
                finished=True,
                finish_reason="FINISHED_LENGTH",
                prompt_tokens=4,
                completion_tokens=1,
            )
        ]
    )
    server = ServingServer(
        async_engine=engine,
        model_id="test-model",
        generate_config=GenerateConfig(max_new_tokens=1),
    )
    response = asyncio.run(
        server._completions(CompletionRequest(prompt=[100, 100, 100, 100]))
    )
    assert response.status_code == 200
    assert engine.calls[0][1] == ""
    assert engine.calls[0][2]["prompt_token_ids"] == (100, 100, 100, 100)
    assert response.headers["x-pypto-token-ids-sha256"] == token_ids_sha256((77,))


def test_chat_serializes_reasoning_and_freezes_parser_spec() -> None:
    engine = _FakeEngine(
        [
            TokenOutput(
                text="最终答案",
                reasoning="先分析",
                token_ids=(90, 1, 91, 2),
                finished=True,
                finish_reason="FINISHED_LENGTH",
                prompt_tokens=4,
                completion_tokens=4,
            )
        ]
    )

    class _DeepSeekTokenizer(_FakeEngine._FakeTokenizer):
        output_parser_id = "deepseek_v4"

    engine.tokenizer = _DeepSeekTokenizer()
    server = ServingServer(engine, "test-model", GenerateConfig(max_new_tokens=4))
    response = asyncio.run(
        server._chat_completions(
            ChatCompletionRequest(
                messages=[ChatMessage(role="user", content="问题")],
                chat_template_kwargs={"enable_thinking": True},
            )
        )
    )

    body = json.loads(response.body)
    message = body["choices"][0]["message"]
    assert message["reasoning"] == "先分析"
    assert message["content"] == "最终答案"
    spec = engine.calls[0][2]["output_parser_spec"]
    assert spec.parser_id == "deepseek_v4"
    assert spec.initial_state == "reasoning"


def test_chat_stream_emits_independent_reasoning_and_content_deltas() -> None:
    engine = _FakeEngine(
        [
            TokenOutput(reasoning_delta="先"),
            TokenOutput(reasoning_delta="分析"),
            TokenOutput(
                reasoning="先分析",
                text="答案",
                text_delta="答案",
                finished=True,
                finish_reason="FINISHED_EOS",
                prompt_tokens=2,
                completion_tokens=3,
            ),
        ]
    )
    server = ServingServer(engine, "test-model", GenerateConfig())

    async def collect():
        return [
            chunk
            async for chunk in server._stream_chat_completion(
                "chat-0",
                "prompt",
                GenerateConfig(stream=True),
                "test-model",
                output_parser_spec=OutputParserSpec(
                    "deepseek_v4",
                    "reasoning",
                ),
            )
        ]

    parsed = _parse_sse(b"".join(chunk.encode() for chunk in asyncio.run(collect())))
    deltas = [chunk["choices"][0]["delta"] for chunk in parsed if chunk["choices"]]
    assert [delta["reasoning"] for delta in deltas] == ["先", "分析", None]
    assert [delta["content"] for delta in deltas] == ["", "", "答案"]


def test_explicit_request_fields_clear_server_generate_defaults():
    server = ServingServer(
        async_engine=_FakeEngine([]),
        model_id="test-model",
        generate_config=GenerateConfig(
            max_new_tokens=11,
            temperature=0.9,
            top_p=0.7,
            top_k=5,
            stop=("END",),
            stream=True,
        ),
    )

    cleared = server._resolve_generate_config(
        CompletionRequest(prompt="x", stop=[], top_k=None)
    )
    assert cleared.stop == ()
    assert cleared.top_k is None
    # Fields the request omitted keep the server defaults.
    assert cleared.temperature == 0.9
    assert cleared.max_new_tokens == 11
    assert cleared.top_p == 0.7
    assert cleared.stream is True

    omitted = server._resolve_generate_config(CompletionRequest(prompt="x"))
    assert omitted.stop == ("END",)
    assert omitted.top_k == 5
    assert omitted.stream is True
