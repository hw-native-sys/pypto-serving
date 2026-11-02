# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Decode computation errors use HTTP/SSE errors without poisoning completed transfers."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import msgspec
import pytest

from pypto_serving.config.types import GenerateConfig
from pypto_serving.pd.integration import Server
from pypto_serving.pd.protocol import Binding, Continuation
from pypto_serving.pd.service import ExecutionError
from pypto_serving.serving.engine.async_engine import TokenOutput
from tests.unit.pd.test_coordination import Peers, assert_idle, config, request
from tests.unit.pd.test_service import service


def ready(payload, *, finish_reason="error", source_done=True, worker_ack=True):
    s, p, d = service()
    binding = Binding.create(p, d, payload)
    entry = s.entry(binding, create=True)
    entry.continuation = Continuation(tuple(range(128)), {"max_new_tokens": 8}, None, None)
    entry.first_token, entry.end = 7, 128
    entry.reservation = s.cache.reserve_group_cache(binding.request, binding.request, 136,
                                                   prompt_token_count=128)
    s.cache.authorize_group_cache_write(binding.request)
    s.cache.commit_group_cache(binding.request)
    entry.prepared.set()
    entry.transferred.set()
    if source_done:
        entry.source_done.set()
    entry.state_task = asyncio.create_task(asyncio.sleep(0))
    case = SimpleNamespace(service=s, prefill=p, entry=entry, events=[], discards=[], closed=False,
                           release=asyncio.Event())
    case.release.set()
    s.observe = lambda name, entry, **fields: case.events.append((name, fields))

    async def worker(operation, value):
        assert operation == "discard_state" and value == binding.request
        case.discards.append(value)
    s.worker = worker

    async def generate(**kwargs):
        assert kwargs["worker_release_ack"] is entry.worker_release_ack
        s.cache.adopt_group_cache(binding.request)
        try:
            yield TokenOutput(text="partial", text_delta="partial")
            await case.release.wait()
            s.cache.release_group_cache(binding.request)
            if worker_ack:
                entry.worker_release_ack.set()
            yield TokenOutput(text="partial", finished=True, finish_reason=finish_reason)
        finally:
            case.closed = True
    s.core.add_prefilled_request = generate
    s.core.tokenizer = SimpleNamespace(eos_token_id=2)
    s.core.config = SimpleNamespace(model_id="model")
    s.core.metrics = None
    s.core.start_profile = s.core.stop_profile = lambda: None
    return case


@pytest.mark.parametrize("source_done", [False, True])
@pytest.mark.parametrize("worker_ack", [False, True])
def test_error_terminal_keeps_sender_stop_and_worker_release_rules(source_done, worker_ack):
    async def run():
        case = ready({}, source_done=source_done, worker_ack=worker_ack)
        s, entry = case.service, case.entry
        s.register_http(entry.binding, {})
        outputs = s.outputs(entry.binding.request)
        assert (await anext(outputs)).text == "partial"
        case.release.clear()
        pending = asyncio.create_task(anext(outputs))
        await asyncio.sleep(0)
        assert not pending.done() and entry.error is None and not case.closed
        assert entry.state_task is not None and not entry.worker_release_ack.is_set()
        case.release.set()
        with pytest.raises(ExecutionError) as error:
            await pending
        assert entry.error is error.value and error.value.request_id == entry.binding.request
        assert case.closed and not s.gated and s.is_ready()
        assert ("request.error", {"error": "ExecutionError", "uncertain": False}) in case.events
        assert case.discards == ([] if worker_ack else [entry.binding.request])
        assert entry.worker_release_ack.is_set() and entry.state_task is None
        assert s.cache.group_cache_reservation(entry.binding.request) is None
        if not source_done:
            assert entry.binding.request in s.requests and not entry.source_done.is_set()
            assert not any(name == "finished" for name, _ in case.events)
            await s.control(case.prefill, (), "source_done", {"binding": msgspec.to_builtins(entry.binding)})
        assert not s.requests and entry.binding.request in s.terminal
        finished = [fields for name, fields in case.events if name == "finished"]
        assert len(finished) == 1 and finished[0]["error"] is error.value
    asyncio.run(run())


@pytest.mark.parametrize("kind", ["completion", "chat"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("failed", [False, True])
def test_pd_http_reports_compute_error_and_preserves_normal_output(kind, stream, failed):
    async def run():
        payload = dict(stream=stream, max_tokens=8)
        if kind == "completion":
            payload["prompt"] = "hello"
        else:
            payload["messages"] = [{"role": "user", "content": "hello"}]
        case = ready(payload, finish_reason="error" if failed else "FINISHED_LENGTH")
        s, entry = case.service, case.entry
        server = Server(s, s.core, GenerateConfig())
        headers = {"x-pd-context": json.dumps(dict(binding=msgspec.to_builtins(entry.binding),
                                                  peer=msgspec.to_builtins(case.prefill)))}
        path = "/v1/completions" if kind == "completion" else "/v1/chat/completions"
        async with httpx.AsyncClient(transport=httpx.ASGITransport(server.app), base_url="http://decode") as client:
            response = await client.post(path, json=payload, headers=headers)
        assert response.status_code == (500 if failed and not stream else 200)
        if not stream:
            body = response.json()
            if failed:
                assert body == dict(object="error", code="pd_execution_error", message=str(entry.error),
                                    request_id=entry.binding.request)
            else:
                assert body["choices"][0]["finish_reason"] == "length"
        else:
            assert response.headers["content-type"].startswith("text/event-stream")
            frames = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
            assert frames[-1] == "[DONE]" and frames.count("[DONE]") == 1
            chunks = [json.loads(frame) for frame in frames[:-1]]
            errors = [chunk["error"] for chunk in chunks if "error" in chunk]
            if failed:
                assert errors == [dict(type="pd_execution_error", message=str(entry.error))]
                assert all(choice["finish_reason"] is None for chunk in chunks for choice in chunk.get("choices", []))
            else:
                assert not errors and any(choice["finish_reason"] == "length"
                    for chunk in chunks for choice in chunk.get("choices", []))
        assert case.closed and not s.requests and not s.gated
        assert s.cache.group_cache_reservation(entry.binding.request) is None
    asyncio.run(run())


def test_router_forwards_decode_json_execution_error_without_rewriting():
    async def run():
        cfg = config()
        peers = Peers(cfg)
        error = ExecutionError("request")
        response = await Server.execution_error(None, error)
        peers.errors["d0"] = httpx.Response(response.status_code, content=response.body,
                                             headers=dict(response.headers))
        router = peers.router(cfg)
        async with router.app.router.lifespan_context(router.app):
            forwarded = await router.proxy.forward(request({}), "/v1/completions")
            assert forwarded.status_code == 500 and forwarded.body == response.body
            assert forwarded.headers["x-pypto-error-source"] == "upstream"
            assert_idle(router)
    asyncio.run(run())
