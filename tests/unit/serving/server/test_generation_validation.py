# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import json

import pytest
from fastapi.testclient import TestClient

from pypto_serving.config.types import GenerateConfig
from pypto_serving.serving.engine.async_engine import TokenOutput
from pypto_serving.serving.server.server import ServingServer


class _Engine:
    def __init__(self):
        self.configs = []
        self.template_calls = 0
        self.tokenizer = self

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls += 1
        return "Go"

    async def add_request(self, request_id, prompt, config, **kwargs):
        self.configs.append(config)
        yield TokenOutput(
            text="Done", finished=True, finish_reason="FINISHED_EOS",
            prompt_tokens=1, completion_tokens=1,
        )


@pytest.fixture(params=[False, True], ids=["completion", "chat"])
def endpoint(request):
    if request.param:
        return "/v1/chat/completions", {"messages": [{"role": "user", "content": "Go"}]}
    return "/v1/completions", {"prompt": "Go"}


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("fields,status,field", [
    ({"temperature": -0.1}, 400, "temperature"),
    ({"temperature": None}, 400, "temperature"),
    ({"temperature": float("inf")}, 400, "temperature"),
    ({"temperature": float("nan")}, 400, "temperature"),
    ({"top_p": 0}, 400, "top_p"),
    ({"top_p": 1.1}, 400, "top_p"),
    ({"top_p": None}, 400, "top_p"),
    ({"top_k": 0}, 400, "top_k"),
    ({"seed": -1}, 400, "seed"),
    ({"max_tokens": 0}, 400, "max_new_tokens"),
    ({"max_tokens": 1.5}, 422, "max_tokens"),
    ({"stop": [1]}, 422, "stop"),
    ({"stop": "END"}, 422, "stop"),
])
def test_http_rejects_invalid_generation_before_work(endpoint, stream, fields, status, field):
    engine = _Engine()
    server = ServingServer(engine, "test", GenerateConfig())
    path, payload = endpoint
    with TestClient(server.app) as client:
        # Exercise the server's non-finite check even though these values are
        # outside standard JSON and are rejected by httpx's json= encoder.
        response = client.post(
            path, content=json.dumps({**payload, "stream": stream, **fields}),
            headers={"Content-Type": "application/json"},
        )
    assert response.status_code == status
    assert response.headers["content-type"].startswith("application/json")
    assert field in response.text
    assert not engine.configs
    assert engine.template_calls == 0


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("fields,status,field", [
    ({"min_tokens": 3, "max_completion_tokens": 2}, 400, "min_tokens"),
    ({"min_tokens": -1}, 400, "min_tokens"),
    ({"max_completion_tokens": 0}, 400, "max_new_tokens"),
])
def test_chat_specific_fields_use_shared_validation(stream, fields, status, field):
    engine = _Engine()
    server = ServingServer(engine, "test", GenerateConfig())
    with TestClient(server.app) as client:
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Go"}], "stream": stream, **fields,
        })
    assert response.status_code == status
    assert field in response.text
    assert not engine.configs
    assert engine.template_calls == 0


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("clear", [False, True])
def test_http_generation_defaults_and_explicit_overrides(endpoint, stream, clear):
    engine = _Engine()
    server = ServingServer(engine, "test", GenerateConfig(
        max_new_tokens=11, temperature=0.7, top_p=0.8, top_k=5, seed=2, stop=("END",),
    ))
    path, payload = endpoint
    fields = {"temperature": 0, "top_p": 1, "top_k": None, "seed": None, "stop": []} if clear else {}
    with TestClient(server.app) as client:
        response = client.post(path, json={**payload, "stream": stream, **fields})
    assert response.status_code == 200
    assert len(engine.configs) == 1
    config = engine.configs[0]
    assert config.max_new_tokens == 11
    assert config.stream is stream
    assert (config.temperature, config.top_p, config.top_k, config.seed, config.stop) == (
        (0, 1, None, None, ()) if clear else (0.7, 0.8, 5, 2, ("END",))
    )


def test_chat_token_alias_precedence_is_preserved():
    engine = _Engine()
    server = ServingServer(engine, "test", GenerateConfig())
    with TestClient(server.app) as client:
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Go"}],
            "max_tokens": 0, "max_completion_tokens": 2,
        })
    assert response.status_code == 200
    assert engine.configs[0].max_new_tokens == 2
