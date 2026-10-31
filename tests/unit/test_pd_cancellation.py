# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Cancellation must survive HTTP/control reordering without authorizing native reuse."""

import asyncio
from types import SimpleNamespace

import msgspec
import pytest

from pypto_serving.pd.config import Config, Endpoint
from pypto_serving.pd.protocol import Binding, Component, Contract, Node, Topology


def service(role):
    from pypto_serving.pd.service import Service
    config = Config((Endpoint("p", 8111, "p"),), (Endpoint("d", 8112, "d"),),
                    capacity=1, handoff_timeout=1, request_timeout=2, log_enabled=False)
    contract = Contract("test", "model", "layout", (Component("kv", "kv", "kv"),))
    p, d = (Node(name, kind, "boot-" + name, contract, "registry-" + name, Topology.for_tp(1, 1))
            for name, kind in (("p", "prefill"), ("d", "decode")))
    cache = SimpleNamespace(group_cache_reservation=lambda _: None)
    core = SimpleNamespace(kv_cache_manager=cache, scheduler=SimpleNamespace(requests={}),
                           is_ready=lambda: True)
    result = Service(config, config.endpoint(role), role, None, core, None)
    result.node = p if role == "prefill" else d
    return result, p, d


async def abort(service, binding):
    from pypto_serving.pd.integration import Server
    async def body():
        return msgspec.to_builtins(binding)
    return await Server.abort(SimpleNamespace(service=service), SimpleNamespace(json=body))


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_abort_before_entry_rejects_late_http_and_control(role):
    async def run():
        s, p, d = service(role)
        payload = {"prompt": "hello"}
        binding = Binding.create(p, d, payload)
        assert (await abort(s, binding)).status_code == 200
        assert (await abort(s, binding)).status_code == 200
        assert not s.requests and s.terminal[binding.request] == binding
        with pytest.raises(ValueError, match="terminal"):
            s.register_http(binding, payload)
        if role == "decode":
            with pytest.raises(ValueError, match="terminal"):
                await s.control(p, (), "begin", {"binding": msgspec.to_builtins(binding)})
        assert not s.requests and not s.gated and not s.pins.active
    asyncio.run(run())


def test_abort_after_begin_wakes_waiter_and_rejects_late_http():
    async def run():
        s, p, d = service("decode")
        payload = {"prompt": "hello"}
        binding = Binding.create(p, d, payload)
        begin = asyncio.create_task(s.control(p, (), "begin", {"binding": msgspec.to_builtins(binding)}))
        await asyncio.sleep(0)
        assert binding.request in s.requests
        await abort(s, binding)
        with pytest.raises(ValueError, match="cancelled"):
            await asyncio.wait_for(begin, 1)
        with pytest.raises(ValueError, match="terminal"):
            s.register_http(binding, payload)
        assert not s.requests
    asyncio.run(run())


def test_cancelled_binding_cannot_replace_active_or_terminal_identity():
    async def run():
        s, p, d = service("prefill")
        payload = {"prompt": "active"}
        binding = Binding.create(p, d, payload)
        entry = s.register_http(binding, payload)
        changed = msgspec.structs.replace(binding, input_digest="changed")
        with pytest.raises(ValueError, match="binding mismatch"):
            await abort(s, changed)
        assert not entry.cancelled
        await abort(s, binding)
        with pytest.raises(ValueError, match="cancelled"):
            s.register_http(binding, payload)
        s.finish(entry)
        with pytest.raises(ValueError, match="binding mismatch"):
            await abort(s, changed)
        stale = msgspec.structs.replace(binding, prefill=("p", "old-boot", p.registry))
        with pytest.raises(ValueError, match="stale"):
            await abort(s, stale)
        assert s.terminal[binding.request] == binding
    asyncio.run(run())


def test_tombstones_are_bounded_and_do_not_consume_active_capacity():
    async def run():
        s, p, d = service("prefill")
        entry = s.register_http(Binding.create(p, d, {}), {})
        bindings = [Binding.create(p, d, {}) for _ in range(s.config.capacity * 8 + 1)]
        for binding in bindings:
            await abort(s, binding)
        assert len(s.terminal) == s.config.capacity * 8
        assert bindings[0].request not in s.terminal
        assert list(s.requests.values()) == [entry] and not entry.cancelled
        await abort(s, bindings[1])
        assert next(reversed(s.terminal)) == bindings[1].request
    asyncio.run(run())


def test_confirmed_control_abort_also_remembers_unseen_binding():
    async def run():
        s, p, d = service("decode")
        binding = Binding.create(p, d, {})
        payload = {"binding": msgspec.to_builtins(binding), "stopped": True}
        assert await s.control(p, (), "abort", payload) == {}
        assert await s.control(p, (), "abort", payload) == {}
        with pytest.raises(ValueError, match="terminal"):
            s.register_http(binding, {})
        assert not s.requests and not s.gated
    asyncio.run(run())


def test_http_abort_keeps_authorized_inflight_write_until_native_stop():
    async def run():
        s, p, d = service("decode")
        binding = Binding.create(p, d, {})
        entry = s.register_http(binding, {})
        reservation = SimpleNamespace(write_authorized=True, prompt_token_count=128)
        s.cache.group_cache_reservation = lambda _: reservation
        entry.pending = object()
        await abort(s, binding)
        assert entry.cancelled and not entry.source_done.is_set()
        assert s.requests[binding.request] is entry and binding.request not in s.terminal
    asyncio.run(run())
