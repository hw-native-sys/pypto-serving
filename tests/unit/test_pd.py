# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Critical PD contracts and lifecycle behavior, using shared small Host fakes."""

import asyncio
from dataclasses import asdict, make_dataclass, replace
import json
from types import SimpleNamespace

import msgspec
import pytest

from pypto_serving.pd.config import Config, Endpoint
from pypto_serving.pd.protocol import Binding, Component, Contract, Node, decode, receive
from pypto_serving.pd.routing import Directory
from pypto_serving.pd.session import PeerPool


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_application_lifespan(monkeypatch, role):
    from pypto_serving.pd import integration as module
    runtime = make_dataclass("Runtime", ["num_speculative_tokens", "speculative_prefix_cache_replay_tokens"])(7, 128)
    config = make_dataclass("Engine", ["runtime_config", "executor_kwargs", "async_scheduling",
                                      "executor_cls", "model_id"])(
        runtime, {}, True, "PyptoDeepSeekV4DSparkExecutor", "model")
    engine = SimpleNamespace(config=config, tokenizer=None, metrics=None,
        start_profile=None, stop_profile=None,
        single_core=lambda: SimpleNamespace(config=config, kv_cache_manager=caches()))
    captured = []
    monkeypatch.setattr(module, "AsyncLLMEngine", lambda active, *a, **kw: captured.append(active) or engine)
    events = []
    async def start(self):
        events.append("start")
    async def stop(self):
        events.append("stop")
    monkeypatch.setattr(module.Application, "start", start)
    monkeypatch.setattr(module.Application, "stop", stop)
    deployment = Config((Endpoint("p", 8101),), (Endpoint("d", 8102),))
    app = module.Application(config, deployment, role, getattr(deployment, role)[0], None, None)
    assert captured[0].runtime_config.num_speculative_tokens == (0 if role == "prefill" else 7)
    assert captured[0].runtime_config.speculative_prefix_cache_replay_tokens == 128
    assert captured[0].executor_kwargs["capture_prefill_tail"] == (role == "prefill")
    assert captured[0].async_scheduling == (role == "decode")
    async def check():
        async with app.app.router.lifespan_context(app.app):
            assert events == ["start"]
        assert events == ["start", "stop"]
    asyncio.run(check())
    if app.channel is not None:
        app.channel.close()
        app.progress.channel.close()


@pytest.mark.parametrize("capture,bootstrap", [(None, object()), (object(), None)])
def test_pd_rejects_missing_device_bootstrap_before_creating_owners(capture, bootstrap):
    from pypto_serving.pd.adapter import DSparkExtension
    runner = SimpleNamespace(speculative=True, _drafter_states={}, _shared_l3_worker=None,
        _materialize_decode_device_cache=None, _compiled=SimpleNamespace(
            layout=SimpleNamespace(ranks=16, tp_size=4), prefill_capture=capture,
            bootstrap_metadata=bootstrap))
    with pytest.raises(ValueError, match="device prompt-tail capture"):
        DSparkExtension(settings=None, runner=runner)


@pytest.mark.parametrize("speculative", [False, True])
@pytest.mark.parametrize("chunks,needed", [
    ([(0, 128, 512)], False),
    ([(256, 128, 512)], False),
    ([(384, 128, 512)], True),
    ([(0, 1, 1)], True),
    ([(0, 127, 127)], True),
    ([(0, 128, 129)], True),
    ([(8192, 1, 8193)], True),
    ([(7936, 128, 8193)], False),
    ([(8064, 128, 8193)], True),
    ([(0, 128, 512), (8064, 128, 8193)], True),
    ([(0, 128, 512), (7936, 128, 8193)], False),
])
def test_pd_capture_skips_only_whole_batches_outside_the_final_window(speculative, chunks, needed):
    from pypto_serving.pd.adapter import DSparkExtension
    extension = DSparkExtension.__new__(DSparkExtension)
    extension.runner = SimpleNamespace(speculative=speculative)
    extension.compiled = SimpleNamespace(layout=SimpleNamespace(sliding_window=128))
    calls = []
    def capture(*args, **kwargs):
        calls.append((args, kwargs))
    wrapped = extension.wrap_capture(capture)
    inputs = SimpleNamespace(descriptors=tuple(SimpleNamespace(
        chunk_start=start, chunk_end=start + length, prompt_len=prompt) for start, length, prompt in chunks))
    batch, hidden = object(), object()
    wrapped(batch, inputs, host_hidden=hidden, descriptor_slot=1)
    assert calls == ([((batch, inputs), dict(host_hidden=hidden, descriptor_slot=1))]
                     if speculative or needed else [])
    if speculative:
        assert wrapped is capture  # D keeps normal K7 capture/bootstrap semantics.


def caches(enabled=True, slots=3):
    from pypto_serving.config.types import KVCacheGroupSpec, KVCacheSpec
    from pypto_serving.serving.memory.kv_cache import KvCacheManager
    specs = tuple(KVCacheGroupSpec(name=name, layer_indices=(0,),
        spec=KVCacheSpec(tokens, tokens // ratio * 64, ratio), max_blocks_per_seq=blocks,
        num_partitions=4, sliding_window=window) for name, tokens, ratio, blocks, window in (
        ("ori", 32, 1, 32, 128), ("cmp_c128", 128, 128, 16, None),
        ("cmp_c4", 128, 4, 16, None), ("idx", 128, 4, 16, None),
        ("hca_state", 8, 1, 128, 128), ("csa_state", 2, 1, 512, 8),
        ("csa_inner_state", 2, 1, 512, 8)))
    cache = KvCacheManager(enable_prefix_cache=enabled)
    cache.init_groups(specs, max_batch_size=slots, primary_num_blocks=32 * slots)
    return cache


def geometries(cache):
    from pypto_serving.pd.adapter import CACHE_COMPONENTS
    from pypto_serving.pd.planner import Geometry
    groups = {g.name: g for g in cache.group_specs}
    return tuple(Geometry(c.name, (0,), cache.group_num_blocks(c.group) + 4,
        groups[c.group].spec.storage_block_size, 64, "uint8") for c in CACHE_COMPONENTS) + (
            Geometry("bootstrap_hidden", (0,), 64, 128, 64, "uint8"),)


def warm(cache, tokens, length, partition=0):
    if length:
        cache.ensure_group_blocks("warm", length, partition=partition)
        cache.cache_group_blocks("warm", cache.compute_group_block_hashes(tokens), length, {})
        cache.release_all_group_requests("warm")


@pytest.mark.parametrize("p_hit,d_hit", [(0, 0), (0, 256), (256, 0), (128, 256),
                                       (256, 128), (256, 256), (0, 512), (384, 512)])
@pytest.mark.parametrize("outcome", ["unknown", "complete", "seed_accept", "seed_reject", "cancel", "early_cancel"])
def test_prefix_mapping_completion_and_quarantine(p_hit, d_hit, outcome):
    from types import SimpleNamespace
    from pypto_serving.config.types import GenerateConfig
    from pypto_serving.pd.adapter import DeepSeekDSpark
    from pypto_serving.pd.integration import Chunk
    from pypto_serving.pd.planner import plan_chunk
    from pypto_serving.pd.protocol import Completion, Continuation, Placement, digest
    from pypto_serving.pd.service import Service
    from pypto_serving.pd.transfer import TransferError
    from pypto_serving.serving.memory.reservation import GroupReservationState
    async def run():
        p_cache, d_cache = caches(), caches()
        tokens = tuple(range(512))
        warm(d_cache, tokens, d_hit)
        adapter = DeepSeekDSpark()
        source, destination = geometries(p_cache), geometries(d_cache)
        contract = adapter.contract("weights", source, (16, 4))
        p = Node("p", "prefill", "p-boot", contract, "p-registry")
        d = Node("d", "decode", "d-boot", contract, "d-registry")
        config = Config((Endpoint("p", 8001, "p"),), (Endpoint("d", 8002, "d"),))
        state = set()
        async def worker(operation, payload):
            value = msgspec.msgpack.decode(payload)
            if operation == "reserve_state":
                state.add(value["request"])
                result = {"bootstrap": 5}
            else:
                assert operation == "discard_state"
                state.discard(value)
                result = None
            return msgspec.msgpack.encode(result)
        core = SimpleNamespace(kv_cache_manager=d_cache, call_worker=worker, config=SimpleNamespace(
            runtime_config=SimpleNamespace(max_seq_len=1024)), scheduler=SimpleNamespace(requests={}))
        service = Service(config, config.decode[0], "decode", adapter, core, None)
        service.node, service.geometry = d, destination
        binding = Binding.create(p, d, {"prompt": "test"})
        entry = service.register_http(binding, {"prompt": "test"})
        continuation = Continuation(tokens, asdict(GenerateConfig(max_new_tokens=8)), None, None,
                                    adapter.prefix_spec(tokens, d_cache))
        if outcome.startswith("seed_"):
            from pypto_serving.serving.constraints import ConstraintSpec
            spec = ConstraintSpec("xgrammar", "deepseek_v4", ({"type": "function", "function": {
                "name": "lookup", "parameters": {"type": "object"}}},), "required", False, True)
            continuation = msgspec.structs.replace(continuation, constraint=spec.to_wire())
            async def reject(spec, initial):
                assert initial == (17,)
                if outcome == "seed_reject":
                    raise ValueError("rejected seed")
            service.preflight = reject
        placement = await service.control(p, source, "begin", dict(binding=binding, continuation=continuation))
        assert isinstance(placement, Placement) and placement.hit == d_hit
        if outcome == "early_cancel":
            entry.response_closed = True
            await service.abort(entry)
            assert not service.requests and d_cache.group_cache_reservation(binding.request) is None
            assert not state
            assert await service.control(p, source, "abort", dict(binding=binding, stopped=True)) == {}
            return
        tables = {k: tuple(v) for k, v in p_cache.ensure_group_blocks("source", 512, partition=1).items()}
        chunk = Chunk(binding.request, 0, p_hit, 512, True, 17, tables, 1, {"bootstrap": 3})
        plan = plan_chunk(binding, chunk, placement, contract, p_cache.group_specs, source, destination,
                          p_hit, adapter.mapping(contract.topology, 1, placement.partition), digest(continuation))
        service.validate_plan(entry, plan, source, p_hit)
        tail = next(c for c in plan.copies if c.component == "bootstrap_hidden")
        assert tail.source_offset == 3 * source[-1].page_bytes
        assert tail.destination_offset == 5 * destination[-1].page_bytes
        with pytest.raises(ValueError):
            service.validate_plan(entry, msgspec.structs.replace(plan, copies=tuple(
                c for c in plan.copies if c.component != "bootstrap_hidden")), source, p_hit)
        with pytest.raises(ValueError, match="write aliases"):
            service.validate_plan(entry, msgspec.structs.replace(plan, copies=(
                msgspec.structs.replace(tail, destination_offset=6 * destination[-1].page_bytes),
                *(c for c in plan.copies if c is not tail))), source, p_hit)
        with pytest.raises(ValueError):
            service.validate_plan(entry, msgspec.structs.replace(plan, copies=tuple(
                c for c in plan.copies if c.component != plan.copies[0].component)), source, p_hit)
        with pytest.raises(ValueError):
            service.validate_plan(entry, msgspec.structs.replace(plan, continuation_digest="wrong"), source, p_hit)
        if any(c.component == "idx_k" for c in plan.copies):
            with pytest.raises(ValueError):
                service.validate_plan(entry, msgspec.structs.replace(plan,
                    copies=tuple(c for c in plan.copies if c.component != "idx_scale")), source, p_hit)
        assert plan.mapping == tuple((4 + slot, slot) for slot in range(4))
        assert {c.component for c in plan.copies} >= {"hca_state", "csa_state", "csa_inner_state"}
        if d_hit == 512:
            assert {c.component for c in plan.copies} == {
                "hca_state", "csa_state", "csa_inner_state", "bootstrap_hidden"}
        completion = Completion(digest(plan), plan.units)
        completion.check(plan)
        with pytest.raises(ValueError):
            msgspec.structs.replace(completion, completed=completion.completed[:-1]).check(plan)
        await service.control(p, source, "plan", dict(binding=binding, plan=plan, p_hit=p_hit))
        if outcome != "unknown":
            if outcome == "cancel":
                await service.abort(entry)
                assert d_cache.group_cache_reservation(binding.request).write_authorized
            reply = await service.control(p, source, "complete", dict(binding=binding, completion=completion))
            if outcome == "seed_reject":
                assert reply == {"error": "rejected seed"} and not service.gated
            else:
                assert entry.transferred.is_set() and not service.gated
            entry.response_closed = True
            await service.control(p, source, "abort", dict(binding=binding, stopped=True))
            assert not service.requests and d_cache.group_cache_reservation(binding.request) is None
            assert not state
            assert await service.control(p, source, "abort", dict(binding=binding, stopped=True)) == {}
            with pytest.raises(ValueError):
                await service.control(p, source, "abort", dict(
                    binding=msgspec.structs.replace(binding, input_digest="wrong"), stopped=True))
            return
        service.fail(entry, TransferError("uncertain", uncertain=True))
        reservation = d_cache.group_cache_reservation(binding.request)
        assert reservation.state == GroupReservationState.QUARANTINED
        assert reservation.quarantined_block_ids_by_group == reservation.writable_block_ids_by_group
        for group, shared in reservation.shared_block_ids_by_group.items():
            assert not set(shared) & set(reservation.quarantined_block_ids_by_group[group])
        with pytest.raises(RuntimeError):
            d_cache.release_group_cache(binding.request)
        assert service.gated
        assert state == {binding.request}
    asyncio.run(run())


def test_mooncake_owner_bounds_and_uncertain_retirement():
    from pypto_serving.pd.protocol import Copy
    from pypto_serving.pd.transfer import Mooncake, Owner, Region, TransferError
    class Engine:
        fail = False
        writes = 0
        def initialize(self, *args): return 0
        def get_rpc_port(self): return 1234
        def register_memory(self, *args): return 0
        def unregister_memory(self, *args): return 0
        def batch_transfer_sync_write(self, *args):
            self.writes += 1
            return 1 if self.fail else 0
    engine = Engine()
    provider = Mooncake("host", lambda: engine)
    provider.register("local", 0, "kv", 4096, 1024)
    provider.install((Region("d0", 1, "kv", 1024, {"endpoint": "d0:1", "address": 8192}),
                      Region("d1", 2, "kv", 1024, {"endpoint": "d1:1", "address": 12288})))
    for peer, rank in (("d0", 1), ("d1", 2)):
        provider.write(peer, (Copy("kv", 0, rank, 0, 0, 64),))
    assert engine.writes == 2
    with pytest.raises(ValueError):
        provider.write("d0", (Copy("kv", 0, 1, 1000, 0, 64),))
    assert engine.writes == 2
    engine.fail = True
    with pytest.raises(TransferError) as error:
        provider.write("d0", (Copy("kv", 0, 1, 0, 0, 64),))
    assert error.value.uncertain
    with pytest.raises(TransferError):
        provider.release()
    assert provider.local  # UNKNOWN is not permission to unregister.
    owner = Owner("boot", 0, "host")
    owner.child.close()
    try:
        with pytest.raises(TransferError) as error:
            owner.call("write")
        assert owner.uncertain and error.value.uncertain
    finally:
        owner.parent.close()


def node(name="p0", role="prefill", **kwargs):
    return Node(name, role, f"boot-{name}", Contract("model", "weights", "layout", (4, 2),
                (Component("kv", "kv", "kv"),)), f"registry-{name}", **kwargs)


def test_config(tmp_path):
    import logging
    from pypto_serving.pd.config import log_context
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"runtime": {"prefill": [{"host": "p", "port": 8001}],
                                           "decode": [{"host": "d", "port": 8002}]}}))
    config = Config.load(path)
    assert config.endpoint("prefill").control_port == 18001
    assert config.endpoint("decode").url == "http://d:8002"
    assert config.run_id != Config.load(path).run_id
    config = replace(config, log_root=str(tmp_path / "logs"))
    for role in ("router", "prefill", "decode"):
        with log_context(config, role):
            logging.getLogger("pypto_serving.pd").info("test message")
    assert {p.parent.name for p in (tmp_path / "logs").rglob("*.log")} == {"router", "p", "d"}
    disabled = tmp_path / "disabled"
    with log_context(replace(config, log_enabled=False, log_root=str(disabled)), "router"):
        assert not disabled.exists()
    for override in ({"provider": "other"}, {"policy": "other"}, {"capacity": 0},
                     {"handoff_timeout": float("nan")}, {"run_id": "../escape"}):
        with pytest.raises(ValueError):
            replace(config, **override)


def test_provider_import_preflight(monkeypatch):
    from types import SimpleNamespace
    from pypto_serving.pd.transfer import Mooncake, provider_class
    assert provider_class("mooncake") is Mooncake
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: SimpleNamespace(returncode=0))
    Mooncake.check_available()
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: SimpleNamespace(returncode=1, stderr="missing native library"))
    with pytest.raises(ValueError, match="missing native library"):
        Mooncake.check_available()


def test_held_source_request_cannot_be_preempted():
    from pypto_serving.serving.memory.kv_cache import KvCacheManager
    from pypto_serving.serving.sched.scheduler import Request, Scheduler, SchedulerConfig, SchedulerOutput
    scheduler = Scheduler(SchedulerConfig(), KvCacheManager(num_blocks=16, block_size=32))
    held = Request("held", [1, 2], 4, scheduling_held=True)
    scheduler.running.append(held)
    assert scheduler._preempt_lowest_priority(exclude=Request("other", [1], 4), scheduled_req_ids=set(),
                                 num_scheduled_tokens={}, output=SchedulerOutput()) is None
    assert list(scheduler.running) == [held]


def test_pool_policy_and_binding():
    nodes = [node("p0"), node("p1"), node("d0", "decode"), node("d1", "decode", prefix=False)]
    directory = Directory([Endpoint(n.node_id, 8000, node_id=n.node_id) for n in nodes])
    for n in nodes:
        directory.update(n)
    pairs = [directory.select() for _ in range(4)]
    assert {(p.node_id, d.node_id) for p, d in pairs} == {
        (p, d) for p in ("p0", "p1") for d in ("d0", "d1")}
    binding = Binding.create(*pairs[0], {"prompt": "hello"})
    binding.check(pairs[0][0], "prefill")
    with pytest.raises(ValueError):
        binding.check(msgspec.structs.replace(pairs[0][0], boot="new"), "prefill")
    assert not pairs[0][0].compatible(msgspec.structs.replace(pairs[0][1], version=99))


def test_peer_pool_preserves_inflight():
    async def run():
        class Peer:
            closed = False
            async def close(self):
                self.closed = True
        async def connect():
            return Peer()
        pool = PeerPool(3)
        async with pool.use(("d0", "old", "r"), connect) as old:
            async with pool.use(("d1", "boot", "r"), connect) as other:
                await pool.retire(("d0", "old", "r"))
                assert not old.closed and not other.closed
                with pytest.raises(ValueError):
                    async with pool.use(("d0", "old", "r"), connect):
                        pass
                async with pool.use(("d0", "new", "r"), connect) as new:
                    assert not old.closed and not new.closed
        assert old.closed and not other.closed
        await pool.close()
        assert other.closed
    asyncio.run(run())


def test_peer_pool_reconnects_closed_sessions_and_retires_old_boots():
    async def run():
        class Peer:
            closed = False
            async def close(self):
                self.closed = True
        peers = []
        async def connect():
            peer = Peer()
            peers.append(peer)
            return peer
        pool = PeerPool(2)
        key = ("d0", "boot-0", "r")
        async with pool.use(key, connect) as first:
            pass
        first.closed = True  # Reader task failed between requests.
        async with pool.use(key, connect) as second:
            assert second is not first
            second.closed = True  # Failure during a request is evicted on exit.
        assert not pool.entries
        for boot in range(8):
            async with pool.use(("d0", f"boot-{boot}", "r"), connect):
                assert len(pool.entries) == 1
        assert all(peer.closed for peer in peers[:-1])
        await pool.close()
        assert all(peer.closed for peer in peers)
    asyncio.run(run())


def test_peer_pool_evicts_failed_connect():
    async def run():
        pool = PeerPool(1)
        async def fail():
            raise ConnectionError("disconnected")
        for _ in range(2):
            with pytest.raises(ConnectionError, match="disconnected"):
                async with pool.use(("d0", "boot", "r"), fail):
                    pytest.fail("failed connect must not yield")
            assert not pool.entries
        await pool.close()
    asyncio.run(run())


def test_abort_before_generation_post_releases_capacity_and_rejects_late_post():
    from pypto_serving.pd.service import Service
    async def run():
        config = Config((Endpoint("p", 8001, "p"),), (Endpoint("d", 8002, "d"),), capacity=1)
        service = Service(config, config.decode[0], "decode", None,
                          SimpleNamespace(kv_cache_manager=caches()), None)
        p, service.node = node("p"), node("d", "decode")
        payload = {"prompt": "hello"}
        binding = Binding.create(p, service.node, payload)
        pending = asyncio.create_task(service.control(p, (), "begin", dict(binding=binding)))
        await asyncio.sleep(0)
        assert len(service.requests) == 1
        await service.control(p, (), "abort", dict(binding=binding, stopped=True))
        with pytest.raises(ValueError, match="cancelled before generation POST"):
            await asyncio.wait_for(pending, 1)
        assert not service.requests
        assert not service.gated
        assert await service.control(p, (), "abort", dict(binding=binding, stopped=True)) == {}
        with pytest.raises(ValueError, match="already terminal"):
            service.register_http(binding, payload)
        service.register_http(Binding.create(p, service.node, payload), payload)
        assert len(service.requests) == 1
    asyncio.run(run())


@pytest.mark.parametrize("reserve_fails", [False, True])
def test_cancel_during_state_reservation_drains_and_discards(reserve_fails):
    from pypto_serving.config.types import GenerateConfig
    from pypto_serving.pd.adapter import DeepSeekDSpark
    from pypto_serving.pd.protocol import Continuation, digest
    from pypto_serving.pd.service import Service

    async def run():
        cache = caches()
        geometry = geometries(cache)
        adapter = DeepSeekDSpark()
        config = Config((Endpoint("p", 8001, "p"),), (Endpoint("d", 8002, "d"),))
        entered, release = asyncio.Event(), asyncio.Event()
        allocated = set()
        async def worker(operation, payload):
            value = msgspec.msgpack.decode(payload)
            if operation == "reserve_state":
                entered.set()
                await release.wait()
                allocated.add(value["request"])
                if reserve_fails:
                    raise RuntimeError("lost reserve reply")
                result = {"bootstrap": 5}
            else:
                assert operation == "discard_state"
                allocated.discard(value)
                result = None
            return msgspec.msgpack.encode(result)
        core = SimpleNamespace(kv_cache_manager=cache, call_worker=worker,
            config=SimpleNamespace(runtime_config=SimpleNamespace(max_seq_len=1024)),
            scheduler=SimpleNamespace(requests={}))
        service = Service(config, config.decode[0], "decode", adapter, core, None)
        contract = adapter.contract("weights", geometry, (16, 4))
        p = Node("p", "prefill", "pb", contract, "pr")
        service.node = Node("d", "decode", "db", contract, "dr")
        binding = Binding.create(p, service.node, {"prompt": "test"})
        entry = service.register_http(binding, {"prompt": "test"})
        tokens = tuple(range(256))
        continuation = Continuation(tokens, asdict(GenerateConfig(max_new_tokens=8)), None, None,
                                    adapter.prefix_spec(tokens, cache))
        entry.response_closed = True
        pending = asyncio.create_task(service.control(p, geometry, "begin", dict(
            binding=binding, continuation=continuation)))
        await entered.wait()
        aborted = asyncio.create_task(service.abort(entry))
        await asyncio.sleep(0)
        assert not aborted.done()
        release.set()
        with pytest.raises((ValueError, RuntimeError)):
            await pending
        await aborted
        assert not allocated and not service.requests and entry.state_task is None
        assert cache.group_cache_reservation(binding.request) is None
        assert not service.gated
        assert digest(continuation) == entry.normalized_digest
    asyncio.run(run())


def test_confirmed_abort_of_unadmitted_request_does_not_need_capacity():
    from pypto_serving.pd.service import Service
    async def run():
        config = Config((Endpoint("p", 8001, "p"),), (Endpoint("d", 8002, "d"),), capacity=1)
        service = Service(config, config.decode[0], "decode", None,
                          SimpleNamespace(kv_cache_manager=caches()), None)
        p, service.node = node("p"), node("d", "decode")
        original = Binding.create(p, service.node, {"prompt": "active"})
        service.register_http(original, {"prompt": "active"})
        rejected = Binding.create(p, service.node, {"prompt": "rejected"})
        assert await service.control(p, (), "abort", dict(binding=rejected, stopped=True)) == {}
        assert set(service.requests) == {original.request} and not service.gated
    asyncio.run(run())


@pytest.mark.parametrize("cancel_cleanup", [False, True])
def test_terminal_output_waits_for_owned_retirement(cancel_cleanup):
    from pypto_serving.config.types import GenerateConfig
    from pypto_serving.pd.service import Request, Service
    from pypto_serving.serving.engine.async_engine import TokenOutput
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        states = {"request"}
        core_closed = False
        async def worker(operation, payload):
            assert operation == "discard_state" and core_closed
            entered.set()
            await release.wait()
            states.discard(msgspec.msgpack.decode(payload))
            return msgspec.msgpack.encode(None)
        async def generate(**kwargs):
            nonlocal core_closed
            try:
                yield TokenOutput(token_id=7, text="answer", finished=True)
            finally:
                core_closed = True
        core = SimpleNamespace(kv_cache_manager=caches(), call_worker=worker,
            tokenizer=SimpleNamespace(eos_token_id=2), add_prefilled_request=generate)
        config = Config((Endpoint("p", 8001, "p"),), (Endpoint("d", 8002, "d"),), capacity=1)
        service = Service(config, config.decode[0], "decode", None, core, None)
        entry = Request(SimpleNamespace(request="request"))
        entry.continuation = SimpleNamespace(config=asdict(GenerateConfig()), tokens=(1,),
                                              parser=None, constraint=None)
        entry.first_token = 7
        entry.transferred.set()
        entry.source_done.set()
        entry.state_task = asyncio.create_task(asyncio.sleep(0, result={"bootstrap": 0}))
        service.requests["request"] = entry
        outputs = service.outputs("request")
        pending = asyncio.create_task(anext(outputs))
        try:
            await asyncio.wait_for(entered.wait(), 1)
        except asyncio.TimeoutError:
            release.set()
            await outputs.aclose()
            raise
        assert not pending.done() and service.requests and states
        if cancel_cleanup:
            pending.cancel()
            await asyncio.sleep(0)
            assert not entry.task.cancelled()
        release.set()
        if cancel_cleanup:
            with pytest.raises(asyncio.CancelledError):
                await pending
        else:
            assert (await pending).finished
        await outputs.aclose()
        await entry.task
        assert not states and not service.requests and entry.state_task is None
        assert core_closed
    asyncio.run(run())


def test_router_connection_budget_scales_with_all_node_pools(monkeypatch):
    from pypto_serving.pd import router as module
    captured = {}
    def client(**kwargs):
        captured.update(kwargs)
    monkeypatch.setattr(module.httpx, "AsyncClient", client)
    config = Config(tuple(Endpoint(f"p{i}", 8001) for i in range(3)),
                    tuple(Endpoint(f"d{i}", 8002) for i in range(5)), capacity=16)
    module.Router(config)
    nodes = len(config.prefill) + len(config.decode)
    active = config.capacity * len(config.decode)
    # Streaming D requests and P handoffs must leave room for full-pool
    # refreshes and P/D cancellation even when every D is saturated.
    assert captured["limits"].max_connections >= active * (nodes + 3) + config.capacity * len(config.prefill) + nodes


@pytest.mark.parametrize("tokens", [(), (1, -1, 0), (-(2**63), 2**63 - 1), tuple(range(1024))])
def test_prefix_token_hash_preserves_wire_format(tokens):
    import hashlib
    import struct
    from pypto_serving.serving.memory.kv_cache import hash_block_tokens
    previous = b"parent"
    expected = hashlib.sha256()
    expected.update(previous)
    expected.update(struct.pack(">I", len(tokens)))
    for token in tokens:
        expected.update(struct.pack(">q", token))
    assert hash_block_tokens(previous, tokens) == expected.digest()


def test_wire_rejects_oversize_and_wrong_type():
    async def run():
        reader = asyncio.StreamReader()
        reader.feed_data(b"\xff\xff\xff\xff")
        with pytest.raises(ValueError):
            await receive(reader)
    asyncio.run(run())
    with pytest.raises(msgspec.ValidationError):
        decode({"request": 4}, Binding)


def test_router_cancel_while_waiting_for_handoff_releases_request_slot():
    from pypto_serving.pd.router import Router
    async def run():
        router = Router(Config((Endpoint("p", 8001),), (Endpoint("d", 8002),), capacity=1))
        class Request:
            async def json(self): return {"prompt": "hello"}
        await router.handoffs.acquire()
        pending = asyncio.create_task(router.generate(Request()))
        await asyncio.sleep(0)
        assert router.requests.locked()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert not router.requests.locked()
        router.handoffs.release()
        await router.client.aclose()
    asyncio.run(run())


def test_cancelled_control_call_drains_reply_before_reusing_slot():
    from pypto_serving.pd.protocol import send
    from pypto_serving.pd.session import Session
    async def run():
        seen, release, server_done = asyncio.Event(), asyncio.Event(), asyncio.Event()
        async def serve(reader, writer):
            try:
                first = await receive(reader)
                seen.set()
                await release.wait()
                await send(writer, dict(id=first["id"], result="first"))
                second = await receive(reader)
                await send(writer, dict(id=second["id"], result="second"))
                await reader.read()
            finally:
                writer.close()
                await writer.wait_closed()
                server_done.set()
        listener = await asyncio.start_server(serve, "127.0.0.1", 0)
        reader, writer = await asyncio.open_connection("127.0.0.1", listener.sockets[0].getsockname()[1])
        session = Session(reader, writer, capacity=1)
        first = asyncio.create_task(session.call("first", {}))
        await asyncio.wait_for(seen.wait(), 1)
        first.cancel()
        with pytest.raises(asyncio.CancelledError): await first
        second = asyncio.create_task(session.call("second", {}))
        await asyncio.sleep(.01)
        assert not second.done() and len(session.pending) == 1
        release.set()
        assert await asyncio.wait_for(second, 1) == "second"
        await session.close()
        await asyncio.wait_for(server_done.wait(), 1)
        listener.close()
        await listener.wait_closed()
    asyncio.run(run())


def test_completion_pipe_batches_waiters_and_drains_cancellation():
    import multiprocessing
    from pypto_serving.pd.protocol import Completion
    from pypto_serving.pd.worker import Progress, ProgressServer
    async def run():
        parent, child = multiprocessing.Pipe()
        finished, batches = set(), []
        def poll(jobs):
            batches.append(jobs)
            return [dict(completion=Completion(j, ())) if j in finished else None for j in jobs]
        server = ProgressServer(child, poll)
        progress = Progress(parent, timeout=1)
        a = asyncio.create_task(progress.wait("a"))
        b = asyncio.create_task(progress.wait("b"))
        await asyncio.sleep(.02)
        a.cancel()
        with pytest.raises(asyncio.CancelledError): await a
        finished.update(("a", "b"))
        assert (await b).plan_digest == "b"
        await progress.close()
        server.close()
        assert any(set(batch) == {"a", "b"} for batch in batches) and not progress.pending
    asyncio.run(run())


@pytest.mark.parametrize("offset", [250, 32])
def test_all_rank_validation_precedes_parallel_native_submission(monkeypatch, offset):
    from concurrent.futures import wait
    import threading
    from types import SimpleNamespace
    from pypto_serving.pd.protocol import Copy, Plan, digest
    from pypto_serving.pd.transfer import Region
    from pypto_serving.pd.worker import WorkerRuntime, WorkerSettings
    barrier, release = threading.Barrier(2), threading.Event()
    writes = []
    class Owner:
        retained = ()
        def __init__(self, boot, rank, *args): self.rank = rank
        def call(self, operation, **kwargs):
            if operation == "write":
                writes.append(self.rank)
                barrier.wait(1)
                assert release.wait(1)
        def close(self): pass
    monkeypatch.setattr("pypto_serving.pd.worker.Owner", Owner)
    runtime = WorkerRuntime(WorkerSettings("p", "host", "mooncake", 2, 2),
                            SimpleNamespace(ranks=2, tp=1, regions=lambda: {"kv": None}))
    local = dict(geometry=(), topology=(2, 1), regions=tuple(Region("p", r, "kv", 256, {"buffer": r}) for r in range(2)))
    remote = dict(geometry=(), topology=(2, 1), regions=tuple(Region("d", r, "kv", 256, {"buffer": r}) for r in range(2)))
    runtime.registry = local
    runtime.install(msgspec.to_builtins(remote))
    binding = Binding("request", ("p", "p", digest(local)), ("d", "d", digest(remote)), "input")
    plan = Plan(binding, "normalized", 0, 0, 1, True, ((0, 0), (1, 1)),
                (Copy("kv", 0, 0, 64, 64, 64), Copy("kv", 1, 1, 0, 0, 64),
                 Copy("kv", 0, 0, 0, 0, 64)), ((0, "kv"), (1, "kv")), 7)
    with pytest.raises(ValueError):
        runtime.submit(msgspec.structs.replace(plan, copies=(*plan.copies,
            msgspec.structs.replace(plan.copies[0], destination_offset=offset))))
    assert not writes and not runtime.jobs
    job = runtime.submit(plan)
    assert runtime.poll((job,)) == [None]
    release.set()
    assert not wait(runtime.jobs[job][1], timeout=2).not_done
    runtime.poll((job,))[0]["completion"].check(plan)
    assert set(writes) == {0, 1}
    runtime.handle("forget", msgspec.msgpack.encode(job))
    runtime.close()


def test_disabled_mode_and_router_have_no_native_pd_imports():
    import subprocess
    import sys
    for module, absent in (("pypto_serving.serving.server.server", ("pypto_serving.pd", "mooncake", "xgrammar")),
                           ("pypto_serving.pd.router", ("torch", "mooncake", "pypto_serving.pd.adapter"))):
        script = f"import {module}; import sys; assert not any(n in sys.modules for n in {absent!r})"
        subprocess.run([sys.executable, "-c", script], check=True, timeout=20)


@pytest.mark.parametrize("prefix", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("kind,stream", [("completion", False), ("completion", True),
                                        ("chat", False), ("chat", True)])
@pytest.mark.parametrize("unknown", [False, True])
@pytest.mark.parametrize("max_tokens", [1, 3])
def test_generation_handoff_overlaps_compute_and_source_cleanup(prefix, kind, stream, unknown, max_tokens):
    from types import SimpleNamespace
    import httpx
    from pypto_serving.config.types import GenerateConfig, RuntimeConfig
    from pypto_serving.pd.adapter import DeepSeekDSpark
    from pypto_serving.pd.integration import Chunk, Server
    from pypto_serving.pd.protocol import Completion, Plan, digest
    from pypto_serving.pd.router import create_app
    from pypto_serving.pd.service import Service
    from pypto_serving.serving.engine.async_engine import TokenOutput
    from pypto_serving.serving.sched.scheduler import RequestStatus, Scheduler, SchedulerConfig

    async def run():
        events, jobs = [], {}
        minimum = 2 if kind == "chat" and max_tokens > 1 else 0
        ignore_eos = stream if kind == "chat" else True
        config = Config((Endpoint("p", 8111, "p"),), (Endpoint("d", 8112, "d"),),
                        handoff_timeout=2, request_timeout=5, log_enabled=False)
        adapter = DeepSeekDSpark()
        tokens = tuple(range(256))
        allow_cleanup, second_compute = asyncio.Event(), asyncio.Event()

        class Tokenizer:
            eos_token_id = 2
            def apply_chat_template(self, *args, **kwargs): return "templated prompt"

        class Core:
            def __init__(self, role, enabled):
                self.role, self.kv_cache_manager = role, caches(enabled)
                self.states = set()
                self.config = SimpleNamespace(runtime_config=RuntimeConfig(max_seq_len=1024))
                self.tokenizer = Tokenizer()
                self.resumed, self.finished = asyncio.Event(), asyncio.Event()
                if role == "decode":
                    self.scheduler = Scheduler(SchedulerConfig(max_seq_len=1024), self.kv_cache_manager)
                else:
                    self.scheduler = SimpleNamespace(resume_request=lambda _: self.resumed.set())
            async def call_worker(self, operation, payload):
                value = msgspec.msgpack.decode(payload) if payload else None
                if operation == "submit":
                    plan = decode(value, Plan)
                    jobs[digest(plan)] = plan
                    events.append(("submit", plan.chunk))
                    result = dict(job=digest(plan))
                elif operation == "forget":
                    del jobs[value]
                    result = None
                elif operation == "reserve_state":
                    self.states.add(value["request"])
                    result = {"bootstrap": 5}
                elif operation == "source_state":
                    result = {"bootstrap": 3}
                elif operation == "discard_state":
                    self.states.discard(value)
                    result = None
                else:
                    result = None
                return msgspec.msgpack.encode(result)
            async def add_request(self, request, prompt, generate, **kwargs):
                assert (generate.max_new_tokens, generate.min_tokens, generate.ignore_eos) == (
                    max_tokens, minimum, ignore_eos)
                cache = self.kv_cache_manager
                hashes = cache.compute_group_block_hashes(tokens)
                _, start, _ = cache.acquire_group_prefix_blocks(request, hashes, max_cache_hit_tokens=128,
                                                                 partition=1)
                number = 0
                while start < len(tokens):
                    end = min(start + 128, len(tokens))
                    tables = cache.ensure_group_blocks(request, end, partition=1)
                    events.append(("compute", number))
                    if number:
                        second_compute.set()
                    self.resumed.clear()
                    self.service.chunks[request].put_nowait(Chunk(request, number, start, end,
                        end == len(tokens), 7 if end == len(tokens) else None,
                        {k: tuple(v) for k, v in tables.items()}, 1))
                    if end == len(tokens):
                        await self.finished.wait()
                        await allow_cleanup.wait()
                        break
                    await self.resumed.wait()
                    number, start = number + 1, end
                cache.release_all_group_requests(request)
                events.append(("source-clean", 0))
                yield TokenOutput(finished=True)
            def finish_prefilled_request(self, request):
                self.finished.set()
            async def abort_request(self, request):
                self.finished.set()
                self.resumed.set()
                allow_cleanup.set()
            async def add_prefilled_request(self, **kwargs):
                assert kwargs["min_tokens"] == minimum
                assert kwargs["ignore_eos"] is ignore_eos
                assert kwargs["eos_token_id"] == Tokenizer.eos_token_id
                request, first = self.scheduler.admit_prefilled(**{k: v for k, v in kwargs.items()
                    if k not in ("stream", "output_parser_spec")})
                assert (request.min_tokens, request.ignore_eos) == (minimum, ignore_eos)
                assert first.num_output_tokens == 1
                events.append(("decode-first", 0))
                allow_cleanup.set()
                yield TokenOutput(token_id=first.new_token_id, text="answer", text_delta="answer",
                    finished=first.finished, finish_reason=first.finish_reason,
                    prompt_tokens=len(tokens), completion_tokens=1)
                if not first.finished:
                    request.output_token_ids.extend([8, 9])
                    self.scheduler.finish_request(request.request_id, RequestStatus.FINISHED_LENGTH)
                    yield TokenOutput(token_id=9, text="answer!", text_delta="!", finished=True,
                        finish_reason="FINISHED_LENGTH", prompt_tokens=len(tokens), completion_tokens=3,
                        token_ids=(7, 8, 9))

        class Progress:
            async def wait(self, job):
                plan = jobs[job]
                events.append(("native", plan.chunk))
                if not plan.final:
                    await asyncio.wait_for(second_compute.wait(), 1)
                    events.append(("overlap-proven", plan.chunk))
                if unknown:
                    from pypto_serving.pd.transfer import TransferError
                    raise TransferError("unknown native completion", uncertain=True)
                return Completion(job, plan.units)

        p_core, d_core = Core("prefill", prefix[0]), Core("decode", prefix[1])
        warm(p_core.kv_cache_manager, tokens, 128, partition=1)
        warm(d_core.kv_cache_manager, tokens, 256)
        p = Service(config, config.prefill[0], "prefill", adapter, p_core, Progress())
        d = Service(config, config.decode[0], "decode", adapter, d_core, None)
        for service, name in ((p, "p"), (d, "d")):
            geometry = geometries(service.cache)
            service.geometry = geometry
            service.registry = dict(geometry=geometry, regions=(), topology=(16, 4))
            service.node = Node(name, service.role, name + "-boot",
                adapter.contract("test-model", geometry, (16, 4)), digest(service.registry), prefix=service.cache.enable_prefix_cache)
            service.core.service = service

        class Peer:
            geometry = d.geometry
            closed = False
            async def call(self, operation, payload):
                value = await d.control(p.node, p.geometry, operation,
                                       msgspec.msgpack.decode(msgspec.msgpack.encode(payload)))
                return msgspec.msgpack.decode(msgspec.msgpack.encode(value))
            async def close(self): self.closed = True
        async def connect(peer): return Peer()
        p.connect = connect
        def engine(core):
            return SimpleNamespace(config=SimpleNamespace(model_id="test-model", executor_cls="test"),
                tokenizer=core.tokenizer, metrics=None, _tokenize_prompt=lambda _: tokens,
                start_profile=None, stop_profile=None)
        servers = {"p": Server(p, engine(p_core), GenerateConfig(max_new_tokens=3)),
                   "d": Server(d, engine(d_core), GenerateConfig(max_new_tokens=3))}
        transports = {name: httpx.ASGITransport(app=server.app, raise_app_exceptions=False)
                      for name, server in servers.items()}
        async def transport(request):
            return await transports[request.url.host].handle_async_request(request)
        app = create_app(config, httpx.MockTransport(transport))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://router") as client:
            body = {"model": "test-model", "stream": stream, "max_tokens": max_tokens}
            body.update({"prompt": "hello"} if kind == "completion" else {
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": max_tokens + 1, "max_completion_tokens": max_tokens,
                "min_tokens": minimum, "ignore_eos": ignore_eos})
            path = "/v1/completions" if kind == "completion" else "/v1/chat/completions"
            rejected = await asyncio.wait_for(client.post(path, json={**body, "temperature": 0.5}), 4)
            assert rejected.status_code in (400, 422) and "text/event-stream" not in rejected.headers["content-type"]
            assert not events and not jobs and not p.requests and not d.requests
            response = await asyncio.wait_for(client.post(path, json=body), 4)
            if unknown:
                assert response.status_code >= 400 or '"error"' in response.text
                assert p.gated and d.gated and p.pins.active and jobs
                assert not any(name == "decode-first" for name, _ in events)
                assert p.requests and d.requests and p.chunks
                assert d_core.states == set(d.requests)
                with pytest.raises(RuntimeError):
                    await p.close()
                await p.peers.close()
                await app.state.router.client.aclose()
                return
            assert response.status_code == 200, response.text
            if stream:
                assert response.text.endswith("data: [DONE]\n\n") and '"error"' not in response.text
                assert "answer" in response.text
            else:
                assert response.json()["usage"]["completion_tokens"] == max_tokens
        assert not p.requests and not d.requests and not p.pins.active and not jobs
        assert not d_core.states
        assert not p.gated and not d.gated
        assert events.index(("decode-first", 0)) < events.index(("source-clean", 0))
        if not prefix[0]:
            assert ("overlap-proven", 0) in events
        await p.peers.close()
        await app.state.router.client.aclose()
    asyncio.run(run())
