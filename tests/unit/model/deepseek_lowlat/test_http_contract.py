# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import asyncio
from types import SimpleNamespace

import httpx
import pytest

from pypto_serving.config.types import GenerateConfig
from pypto_serving.model.deepseek_lowlat.contract import LOWLAT_EXECUTOR
from pypto_serving.serving.engine.async_engine import EngineConfig
from pypto_serving.serving.server.server import ServingServer


class AdmissionOnlyEngine:
    def __init__(self):
        self.config = EngineConfig(
            executor_cls=LOWLAT_EXECUTOR, max_num_running_reqs=1, enable_prefix_cache=False
        )
        self.tokenizer = SimpleNamespace(output_parser_id=None)
        self.dispatched = False

    def validate_generate_config(self, config, **kwargs):
        self.config.validate_generate_config(config, **kwargs)

    async def add_request(self, *args, **kwargs):
        self.dispatched = True
        raise AssertionError("invalid input reached inference")
        yield


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/v1/completions", {"prompt": "hi", "temperature": 0.5, "stream": True}),
        ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}], "stream": True}),
        (
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "hi"}], "min_tokens": 1, "stream": True},
        ),
        ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "top_p": 0.5}),
    ],
)
def test_http_rejects_before_sse_headers_or_device_dispatch(path, payload):
    engine = AdmissionOnlyEngine()
    server = ServingServer(engine, "lowlat", GenerateConfig())

    async def request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            return await client.post(path, json=payload)

    response = asyncio.run(request())
    assert response.status_code == 400, response.text
    assert response.headers["content-type"].startswith("application/json")
    assert engine.dispatched is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("prompt_tokens,outputs", [(16315, 64), (7, 2049)])
def test_native_budget_rejection_precedes_sse_and_inference(stream, prompt_tokens, outputs):
    from types import SimpleNamespace
    from pypto_serving.config.types import RuntimeConfig
    from pypto_serving.serving.engine.async_engine import AsyncLLMEngine

    engine = AsyncLLMEngine.__new__(AsyncLLMEngine)
    engine.config = EngineConfig(executor_cls=LOWLAT_EXECUTOR, max_num_running_reqs=1,
                                 enable_prefix_cache=False,
                                 runtime_config=RuntimeConfig(max_seq_len=16384, num_speculative_tokens=5))
    engine.tokenizer = SimpleNamespace(encode=lambda text: [260]*prompt_tokens, bos_token_id=None)
    engine.dispatched = False
    async def dispatch(*args, **kwargs):
        engine.dispatched = True
        raise AssertionError("invalid native budget reached serving")
        yield
    engine.add_request = dispatch
    server = ServingServer(engine, "lowlat", GenerateConfig())
    async def request():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app),
                                     base_url="http://test", trust_env=False) as client:
            return await client.post("/v1/completions", json=dict(prompt="x", max_tokens=outputs,
                                                                  temperature=0, stream=stream))
    response = asyncio.run(request())
    assert response.status_code == 400 and response.headers["content-type"].startswith("application/json")
    assert engine.dispatched is False
