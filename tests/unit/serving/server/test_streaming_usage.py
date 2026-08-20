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

from pypto_serving.config.types import GenerateConfig
from pypto_serving.serving.engine.async_engine import TokenOutput
from pypto_serving.serving.server.server import (
    ServingServer,
)

# ---------------------------------------------------------------------------
# Fake engine
# ---------------------------------------------------------------------------


class _FakeEngine:
    """Minimal engine stub: yields caller-supplied TokenOutputs."""

    def __init__(self, outputs: list[TokenOutput]) -> None:
        self._outputs = outputs
        self.requests = []
        self.tokenizer = self._FakeTokenizer()

    class _FakeTokenizer:
        def __init__(self) -> None:
            self.chat_template_call = None

        def encode(self, text: str) -> list[int]:
            return [1] * max(1, len(text.split()))

        def apply_chat_template(self, messages, chat_template_kwargs=None):
            self.chat_template_call = (messages, chat_template_kwargs)
            return " ".join(m["content"] for m in messages)

    async def add_request(self, request_id, prompt, config, **kwargs):
        self.requests.append((request_id, prompt, config))
        for out in self._outputs:
            yield out


def _make_server(outputs: list[TokenOutput]) -> ServingServer:
    return ServingServer(async_engine=_FakeEngine(outputs), model_id="test-model")


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


def _collect_stream(stream) -> bytes:
    async def collect():
        return [chunk async for chunk in stream]

    return b"".join(chunk.encode() for chunk in asyncio.run(collect()))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


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

    raw = _collect_stream(
        server._stream_completion(
            "req-0", "hello prompt here go", GenerateConfig(max_new_tokens=3), "test-model"
        )
    )
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