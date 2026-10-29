# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Admission and read-only status must preserve the existing resource lifecycle."""

import asyncio
import json
from types import SimpleNamespace

import msgspec
import pytest

from pypto_serving.pd.protocol import Binding, Completion, Plan, digest
from tests.unit.test_pd_cancellation import service


@pytest.mark.parametrize("failure", ["none", "initializing", "loop", "worker", "gated", "closing"])
def test_readiness_is_shared_by_health_descriptor_and_admission(failure):
    from pypto_serving.pd.integration import Server
    from pypto_serving.serving.engine.async_engine import ReplicaEngineCore

    async def run():
        s, p, d = service("decode")
        state = SimpleNamespace(_running=True,
            _loop_task=SimpleNamespace(done=lambda: failure == "loop"),
            _worker_process=SimpleNamespace(is_alive=lambda: failure != "worker"))
        s.core.is_ready = lambda: ReplicaEngineCore.is_ready(state)
        s.gated, s.closing = failure == "gated", failure == "closing"
        if failure == "initializing":
            s.node = None
        server = SimpleNamespace(service=s)
        expected = failure == "none"
        assert s.is_ready() is expected
        assert (await Server._health(server)).status_code == (200 if expected else 503)
        response = await Server.descriptor(server)
        if failure == "initializing":
            assert response.status_code == 503
        else:
            assert json.loads(response.body)["healthy"] is expected
        binding = Binding.create(p, d, {})
        if expected:
            assert s.register_http(binding, {}).binding == binding
        else:
            with pytest.raises(RuntimeError, match="unavailable"):
                s.register_http(binding, {})
            assert not s.requests
    asyncio.run(run())


def test_pd_policy_cannot_select_a_pair_outside_filtered_candidates():
    from pypto_serving.router.routing import RoundRobin
    from tests.unit.router.test_pd_coordination import Peers, config, request, assert_idle

    async def run():
        cfg = config(p=2)
        peers = Peers(cfg)
        router = peers.router(cfg)
        async with router.app.router.lifespan_context(router.app):
            router.registry.set_draining("p0", True)
            def choose(candidates):
                assert all(pair is None or pair[0].node_id == "p1" for pair in candidates)
                return (peers.nodes["p0"], peers.nodes["d0"]), "invalid_choice"
            router.proxy.policy = SimpleNamespace(select=choose)
            with pytest.raises(RuntimeError, match="ineligible"):
                router.proxy._admit({})
            assert_idle(router)
            assert router.registry.total_routed() == 0
            router.proxy.policy = RoundRobin()
            assert (await router.proxy.forward(request({}), "/v1/completions")).status_code == 200
            assert router.registry.total_routed() == 1
            assert_idle(router)
    asyncio.run(run())


@pytest.mark.parametrize("sink", ["enabled", "disabled", "broken"])
def test_native_observation_preserves_result_and_explicit_owner_context(monkeypatch, sink):
    import logging
    import threading
    from pypto_serving.pd.protocol import Copy
    from pypto_serving.pd.transfer import OwnerService
    from pypto_serving.tools import observe

    calls, traces, replies = [], [], []
    observation = dict(request="request", plan_digest="plan", source_rank=0, destination_rank=1,
                       prefill=("p", "p-boot", "p-registry"), decode=("d", "d-boot", "d-registry"))
    commands = iter([
        dict(boot="p-boot", sequence=1, operation="write", peer="d-boot", observation=observation,
             copies=[msgspec.to_builtins(Copy("kv", 0, 1, 0, 0, 64))]),
        dict(boot="p-boot", sequence=2, operation="stop")])
    def duration(*args, **kwargs):
        if sink == "broken":
            raise OSError("profiler unavailable")
        if sink == "enabled":
            traces.append(kwargs)
    def info(*args, **kwargs):
        if sink == "broken":
            raise OSError("log unavailable")
    monkeypatch.setattr(observe, "profile_duration", duration)
    monkeypatch.setenv("MC_TRANSFER_TIMEOUT", "30")
    monkeypatch.setattr(logging.getLogger("pypto_serving.pd.transfer"), "info", info)
    monkeypatch.setattr("pypto_serving.pd.transfer._receive", lambda channel: next(commands))
    monkeypatch.setattr("pypto_serving.pd.transfer._send", lambda channel, value: replies.append(value))
    owner = OwnerService.__new__(OwnerService)
    owner.acl = SimpleNamespace(aclrtSetCurrentContext=lambda _: 0)
    owner.acl_context, owner.channel, owner.tokens, owner.stopped = None, None, [], False
    owner.started = threading.Event()
    owner.owner = SimpleNamespace(boot="p-boot", provider="mooncake", hostname="host",
        provider_factory=lambda *_: SimpleNamespace(write=lambda peer, copies: calls.append((peer, copies))))
    owner._run()
    assert len(calls) == 1 and calls[0][0] == "d-boot"
    assert all("error" not in reply for reply in replies) and owner.stopped
    if sink == "enabled":
        assert traces[0]["args"] == {**observation, "bytes": 64, "segments": 1, "outcome": "completed"}
        assert traces[0]["dur_us"] >= 0
    else:
        assert not traces


@pytest.mark.parametrize("failure", ["engine", "gated", "closing"])
def test_unavailable_node_accepts_existing_completion_and_stop_but_no_new_work(failure):
    async def run():
        s, p, d = service("decode")
        binding = Binding.create(p, d, {})
        entry = s.register_http(binding, {})
        plan = Plan(binding, "continuation", 0, 0, 128, False, ((0, 0),), (), ((0, "kv"),))
        entry.pending = plan
        s.core.is_ready = lambda: failure != "engine"
        s.gated, s.closing = failure == "gated", failure == "closing"
        payload = {"binding": msgspec.to_builtins(binding)}
        with pytest.raises(RuntimeError, match="unavailable"):
            s.register_http(Binding.create(p, d, {}), {})
        with pytest.raises(RuntimeError, match="unavailable"):
            await s.control(p, (), "begin", payload)
        with pytest.raises(RuntimeError, match="unavailable"):
            await s.control(p, (), "plan", payload)
        bad = Completion("other-plan", plan.units)
        with pytest.raises(ValueError, match="completion"):
            await s.control(p, (), "complete", dict(payload, completion=msgspec.to_builtins(bad)))
        assert entry.pending is plan
        await s.control(p, (), "complete",
                        dict(payload, completion=msgspec.to_builtins(Completion(digest(plan), plan.units))))
        assert entry.pending is None and entry.end == 128 and entry.chunk == 1
        await s.control(p, (), "source_done", payload)
        assert entry.source_done.is_set()
        await s.control(p, (), "abort", dict(payload, stopped=True))
        assert entry.cancelled
        s.finish(entry)
        assert binding.request not in s.requests
    asyncio.run(run())


def test_control_cleanup_cannot_create_request_or_accept_stale_binding():
    async def run():
        s, p, d = service("decode")
        binding = Binding.create(p, d, {})
        for operation in ("complete", "source_done", "plan", "unknown"):
            with pytest.raises(ValueError, match="unknown PD request"):
                await s.control(p, (), operation, {"binding": msgspec.to_builtins(binding)})
            assert not s.requests
        s.register_http(binding, {})
        s.gated = True
        stale = msgspec.structs.replace(binding, decode=("d", "old", d.registry))
        with pytest.raises(ValueError, match="stale"):
            await s.control(p, (), "source_done", {"binding": msgspec.to_builtins(stale)})
        assert not s.requests[binding.request].source_done.is_set()
    asyncio.run(run())


def cache_service(role="decode"):
    from pypto_serving.config.types import KVCacheGroupSpec, KVCacheSpec
    from pypto_serving.serving.memory.kv_cache import KvCacheManager
    s, p, d = service(role)
    cache = KvCacheManager(enable_prefix_cache=True)
    cache.init_groups((KVCacheGroupSpec("kv", (0,), KVCacheSpec(32, 64), 32, num_blocks=64),),
                      max_batch_size=2)
    s.cache = s.pins.cache = cache
    binding = Binding.create(p, d, {})
    entry = s.register_http(binding, {})
    return s, p, binding, entry


def test_query_reads_current_lease_and_preserves_prefix_after_release():
    s, _, binding, entry = cache_service()
    rid = binding.request
    hashes = s.cache.compute_group_block_hashes(list(range(128)))
    initial = s.cache.reserve_group_cache(rid, rid, 256, prompt_token_count=128,
                                         prefix_block_hashes=hashes, shareable_group_names=("kv",))
    entry.reservation = initial
    def check(expected):
        current = s.cache.group_cache_reservation(rid)
        references = tuple(block.ref_cnt for pool in s.cache._group_pools.values() for block in pool.blocks)
        result = s.status(rid)
        assert result["reservation"]["state"] == expected
        assert s.cache.group_cache_reservation(rid) is current
        assert references == tuple(block.ref_cnt for pool in s.cache._group_pools.values() for block in pool.blocks)
        assert initial.state.value == "CONSTRUCTING"
        assert result["worker"]["state_release_confirmed"] is False
        assert "block_ids" not in json.dumps(result) and "address" not in json.dumps(result)
        return result
    check("CONSTRUCTING")
    s.cache.authorize_group_cache_write(rid)
    s.cache.commit_group_cache(rid)
    result = check("READY")
    result["reservation"]["published_block_counts"]["kv"] = -1
    assert s.cache.group_cache_reservation(rid).published_block_counts["kv"] == 4
    s.cache.adopt_group_cache(rid)
    check("IN_USE")
    s.cache.release_group_cache(rid)
    check("RELEASED")
    _, hit, _ = s.cache.acquire_group_prefix_blocks("reuse", hashes, max_cache_hit_tokens=128)
    assert hit == 128
    s.cache.release_all_group_requests("reuse")
    s.cache.retire_group_cache_reservation(rid)
    s.finish(entry)
    result = s.status(rid)
    assert result["terminal_binding_present"] and not result["entry_present"]
    assert result["reservation"] == {"present": False, "state": None}
    assert result["worker"] is None
    assert s.status("never-seen")["reservation"] == {"present": False, "state": None}


def test_query_during_unknown_and_http_cancel_is_read_only_and_does_not_claim_release():
    from pypto_serving.pd.transfer import TransferError
    async def run():
        s, p, binding, entry = cache_service()
        rid = binding.request
        entry.reservation = s.cache.reserve_group_cache(rid, rid, 256, prompt_token_count=128)
        s.cache.authorize_group_cache_write(rid)
        entry.pending = Plan(binding, "continuation", 0, 0, 128, True, ((0, 0),), (), ((0, "kv"),))
        s.fail(entry, TransferError("native result unknown", uncertain=True))
        assert entry.transferred.is_set() and entry.prepared.is_set()
        current = s.cache.group_cache_reservation(rid)
        await s.abort(entry)
        result = s.status(rid)
        assert result["reservation"]["state"] == "QUARANTINED"
        assert result["handoff"]["uncertain"] and result["handoff"]["confirmed_end"] == 0
        assert result["handoff"]["pending_plan"]["plan_digest"] == digest(entry.pending)
        assert result["execution"]["cancelled"] and not result["execution"]["source_stop_confirmed"]
        assert s.cache.group_cache_reservation(rid) is current
        with pytest.raises(RuntimeError, match="native writer"):
            s.cache.release_group_cache(rid)
        await s.control(p, (), "abort", {"binding": msgspec.to_builtins(binding), "stopped": True})
        assert s.status(rid)["reservation"] == {"present": False, "state": None}
    asyncio.run(run())


def test_source_pin_is_separate_from_destination_lease_state():
    s, _, binding, _ = cache_service("prefill")
    tables = s.cache.ensure_group_blocks(binding.request, 128, partition=0)
    s.pins.retain(SimpleNamespace(request=binding.request, blocks=tables, partition=0))
    result = s.status(binding.request)
    assert result["source_pin"] and result["reservation"] is None
    assert result["worker"]["state_release_confirmed"] is None
    s.pins.release(binding.request)
    assert s.status(binding.request)["source_pin"] is False
    s.cache.release_all_group_requests(binding.request)


def test_internal_query_remains_available_when_generation_admission_is_closed():
    import httpx
    from pypto_serving.config.types import GenerateConfig
    from pypto_serving.pd.integration import Server
    async def run():
        s, _, binding, _ = cache_service()
        s.gated = s.closing = True
        engine = SimpleNamespace(config=SimpleNamespace(model_id="model"), tokenizer=None, metrics=None,
                                 start_profile=None, stop_profile=None)
        server = Server(s, engine, GenerateConfig())
        before = dict(s.requests)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app),
                                     base_url="http://node") as client:
            result = await client.get("/internal/pd/requests/" + binding.request)
            assert result.status_code == 200 and result.json()["entry_present"]
            assert result.json()["node"]["boot"] == s.node.boot
            assert (await client.get("/health")).status_code == 503
            assert (await client.get("/internal/pd/descriptor")).json()["healthy"] is False
            rejected = await client.post("/v1/completions", json={"prompt": "new"})
            assert rejected.status_code == 503 and rejected.json()["code"] == "pd_node_unavailable"
            assert (await client.get("/internal/pd/requests/missing")).json()["worker"] is None
        assert s.requests == before
    asyncio.run(run())
