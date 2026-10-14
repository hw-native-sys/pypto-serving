# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Public Router -> actual node API -> control/Connector -> output on CPU."""

import asyncio
from types import SimpleNamespace

import msgspec
import pytest

from pypto_serving.config.types import GenerateConfig
from pypto_serving.router.client import PublicResult
from pypto_serving.router.recovery import RecoveryPhase
from pypto_serving.serving.pd.config import PDRole
from pypto_serving.serving.pd.http_api import PDHTTPRoutes, decode_json, encode_json
from pypto_serving.serving.pd.service import PDServingService
from pypto_serving.serving.server.server import ServingServer

from tests.unit.serving.pd.test_service import _FakeCore, _free_port
from tests.unit.serving.pd.test_service_routing import _external_config
from .test_coordinator import _coordinator, sse_document


class _APIResponse:
    def __init__(self, status, headers, body):
        self.status, self.headers, self.body = status, headers, body
    async def json(self):
        return msgspec.json.decode(self.body)
    async def result(self):
        return PublicResult(self.body, self.headers)
    async def frames(self):
        for frame in self.body.split(b"\n\n"):
            if frame:
                yield frame + b"\n\n"
    async def aclose(self):
        return None


class _APIClient:
    """Minimal test-only ASGI transport; no extra HTTP dependency."""
    def __init__(self, app):
        self.app, self.calls = app, []
    async def _request(self, method, path, body=b""):
        sent, received = [], False
        async def receive():
            nonlocal received
            if not received:
                received = True
                return {"type": "http.request", "body": body, "more_body": False}
            await asyncio.Future()
        async def send(message):
            sent.append(message)
        scope = {
            "type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
            "http_version": "1.1", "scheme": "http", "method": method, "path": path,
            "raw_path": path.encode(), "query_string": b"", "root_path": "",
            "headers": [(b"content-type", b"application/json")],
            "client": ("127.0.0.1", 12345), "server": ("127.0.0.1", 80),
        }
        await self.app(scope, receive, send)
        start = next(message for message in sent if message["type"] == "http.response.start")
        headers = {name.decode().lower(): value.decode() for name, value in start["headers"]}
        payload = b"".join(message.get("body", b"") for message in sent if message["type"] == "http.response.body")
        return _APIResponse(start["status"], headers, payload)
    async def get(self, path, type_):
        response = await self._request("GET", path)
        assert response.status == 200
        return decode_json(response.body, type_)
    async def post(self, path, payload, type_=None):
        response = await self._request("POST", path, encode_json(payload))
        assert response.status == 200
        return decode_json(response.body, type_) if type_ else None
    async def open_generation(self, path, body):
        self.calls.append(path)
        return await self._request("POST", path, body)


def _app(service, normalize, *, preflight=None):
    def tokens(prompt, ids=None):
        if not normalize:
            raise AssertionError("D must not tokenize")
        return tuple(ids) if ids is not None else (1, 2, 3)
    def template(messages, **kwargs):
        if not normalize:
            raise AssertionError("D must not template")
        return "prepared prompt"
    engine = SimpleNamespace(
        resolve_prompt_tokens=tokens,
        config=SimpleNamespace(executor_cls="PyptoDeepSeekV4DSparkExecutor"),
        tokenizer=SimpleNamespace(apply_chat_template=template,
                                  output_parser_id="deepseek_v4" if preflight is not None else None),
    )
    def install(server):
        PDHTTPRoutes(server.app, service, service.config,
                     model_id=server.model_id,
                     prepare_completion=server.prepare_completion, prepare_chat=server.prepare_chat,
                     preflight_constraint=preflight or server._preflight_constraint,
                     resolve_prompt_tokens=server.resolve_prompt_tokens,
                     start_profile=server.start_profile, stop_profile=server.stop_profile)
    return ServingServer(engine, "served-alias", GenerateConfig(max_new_tokens=2), route_factory=install).app


@pytest.mark.parametrize("kind", ("completion", "chat"))
@pytest.mark.parametrize("stream", (False, True))
def test_node_public_request_drives_one_real_host_handoff_without_d_normalization(tmp_path, kind, stream):
    async def exercise():
        port = _free_port()
        p_service = PDServingService(_FakeCore(PDRole.PREFILL), _external_config(PDRole.PREFILL, port))
        d_service = PDServingService(_FakeCore(PDRole.DECODE), _external_config(PDRole.DECODE, port))
        await asyncio.gather(p_service.start(), d_service.start())
        p_client, d_client = _APIClient(_app(p_service, True)), _APIClient(_app(d_service, False))
        coordinator, journal = _coordinator(tmp_path, p_client, d_client)
        coordinator.directory.run_id = "run"
        public = {"prompt": [1, 2, 3]} if kind == "completion" else {
            "messages": [{"role": "user", "content": "hello"}],
        }
        raw = msgspec.json.encode({**public, "stream": stream, "max_tokens": 2})
        try:
            output = [out async for out in coordinator.generate(kind, raw, "request")]
            assert len(p_client.calls) == len(d_client.calls) == 1
            assert p_service.core.transfer_calls == 1 and p_service.core.source_released
            assert not p_service._requests and not d_service._requests
            assert not d_service.core.kv_cache_manager.group_cache_reservations
            assert not journal.unresolved
            if stream:
                assert output[-1] == b"data: [DONE]\n\n"
                values = [sse_document(frame) for frame in output[:-1]]
                assert values[-1]["usage"]["completion_tokens"] == 2
            else:
                value = msgspec.json.decode(output[-1].body)
                assert value["usage"]["completion_tokens"] == 2
                assert value["model"] == "served-alias"
                assert "pd_transfer" not in value
        finally:
            journal.close()
            await asyncio.gather(p_service.close(), d_service.close())
    asyncio.run(asyncio.wait_for(exercise(), 15))

def test_p_preflight_error_has_bound_no_compute_proof_and_retires_d_waiter(tmp_path):
    async def exercise():
        port = _free_port()
        p_service = PDServingService(_FakeCore(PDRole.PREFILL), _external_config(PDRole.PREFILL, port))
        d_service = PDServingService(_FakeCore(PDRole.DECODE), _external_config(PDRole.DECODE, port))
        await asyncio.gather(p_service.start(), d_service.start())
        p_client, d_client = _APIClient(_app(p_service, True)), _APIClient(_app(d_service, False))
        coordinator, journal = _coordinator(tmp_path, p_client, d_client)
        try:
            with pytest.raises(ValueError):
                await coordinator.start_stream("completion", b'{"prompt":"hi","stream":true,"temperature":0.9}', "request")
            assert p_service.core.transfer_calls == 0
            assert not p_service._requests and not d_service._requests
            assert not d_service.core.kv_cache_manager.group_cache_reservations
            assert not journal.unresolved
        finally:
            journal.close()
            await asyncio.gather(p_service.close(), d_service.close())
    asyncio.run(asyncio.wait_for(exercise(), 15))


@pytest.mark.parametrize("stream", (False, True))
def test_invalid_decode_seed_is_a_safe_failure_and_next_request_succeeds(tmp_path, monkeypatch, stream):
    monkeypatch.setattr("pypto_serving.serving.pd.service.importlib.util.find_spec", lambda _: object())
    async def exercise():
        port = _free_port()
        p_service = PDServingService(_FakeCore(PDRole.PREFILL), _external_config(PDRole.PREFILL, port))
        d_service = PDServingService(_FakeCore(PDRole.DECODE), _external_config(PDRole.DECODE, port))
        original = p_service.core.add_request
        def prefill(*args, constraint_spec=None, **kwargs):
            return original(*args, **kwargs)
        p_service.core.add_request = prefill
        admissions = []
        original_adopt = d_service.core.add_adopted_handoff
        async def adopt(**kwargs):
            admissions.append(kwargs["request_id"])
            async for output in original_adopt(**kwargs):
                yield output
        d_service.core.add_adopted_handoff = adopt
        async def preflight(spec, initial_tokens=()):
            if initial_tokens:
                raise ValueError("invalid seed")
        p_client, d_client = _APIClient(_app(p_service, True, preflight=preflight)), _APIClient(
            _app(d_service, False, preflight=preflight))
        await asyncio.gather(p_service.start(), d_service.start())
        coordinator, journal = _coordinator(tmp_path, p_client, d_client)
        tools = [{"type": "function", "function": {"name": "lookup", "strict": True,
                  "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}}]
        raw = msgspec.json.encode({"messages": [{"role": "user", "content": "lookup"}],
                                  "tools": tools, "tool_choice": "required", "stream": stream})
        try:
            with pytest.raises(ValueError, match="INVALID_DECODE_SEED"):
                [out async for out in coordinator.generate("chat", raw, "bad-seed")]
            assert coordinator.recovery.phase is RecoveryPhase.RUNNING
            key = next(key for key in p_service._retired if key.request_id == "bad-seed")
            assert p_service.query_handoff(key).state == "FAILED"
            assert p_service.core.kv_cache_manager.group_request_partition("bad-seed") is None
            assert not admissions
            assert not p_service._requests and not d_service._requests
            assert not d_service.core.kv_cache_manager.group_cache_reservations
            assert not journal.unresolved
            output = [out async for out in coordinator.generate(
                "completion", b'{"prompt":[1,2,3],"max_tokens":2}', "next-request")]
            assert msgspec.json.decode(output[-1].body)["usage"]["completion_tokens"] == 2
            assert admissions == ["next-request"]
            assert not p_service._requests and not d_service._requests
            assert not p_service._active_keys and not d_service._active_keys
            assert p_service._admission.count == d_service._admission.count == 0
        finally:
            journal.close()
            await asyncio.gather(p_service.close(), d_service.close())
    asyncio.run(asyncio.wait_for(exercise(), 15))
