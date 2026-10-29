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

import msgspec
import pytest

from pypto_serving.pd.config import Config, Endpoint
from pypto_serving.pd.protocol import Binding, Component, Contract, Node, Topology, decode
from pypto_serving.pd.coordinator import PDCoordinator
from pypto_serving.router.app import create_router_app
from pypto_serving.router.config import ReplicaSpec, RouterConfig
from pypto_serving.router.routing import ReplicaRegistry, SessionDirectory


def router_config(config):
    return RouterConfig(replicas=tuple(
        ReplicaSpec(e.node_id, e.host, e.port, role=role)
        for role in ("prefill", "decode") for e in getattr(config, role)),
        pd_options={"capacity": config.capacity, "handoff_timeout": config.handoff_timeout,
                    "request_timeout": config.request_timeout}, observability={"enabled": False})


def caches(enabled=True, slots=3, partitions=4):
    from pypto_serving.config.types import KVCacheGroupSpec, KVCacheSpec
    from pypto_serving.serving.memory.kv_cache import KvCacheManager
    specs = tuple(KVCacheGroupSpec(name=name, layer_indices=(0,),
        spec=KVCacheSpec(tokens, tokens // ratio * 64, ratio), max_blocks_per_seq=blocks,
        num_partitions=partitions, sliding_window=window) for name, tokens, ratio, blocks, window in (
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
    topology = kwargs.pop("topology", Topology.for_tp(4, 2))
    return Node(name, role, f"boot-{name}", Contract("model", "weights", "layout",
                (Component("kv", "kv", "kv"),)), f"registry-{name}", topology, **kwargs)


def test_pool_policy_and_binding():
    nodes = [node("p0"), node("p1"), node("d0", "decode"), node("d1", "decode", prefix=False)]
    config = RouterConfig(replicas=tuple(ReplicaSpec(n.node_id, n.node_id, 8000, role=n.role) for n in nodes))
    registry = ReplicaRegistry(config, SessionDirectory(600))
    coordinator = PDCoordinator(Config.from_router(config), registry, None, None)
    for n in nodes:
        coordinator.nodes[n.node_id] = n
        registry.set_ready(n.node_id, True)
    pairs = [coordinator.policy.select(coordinator.pairs())[0] for _ in range(4)]
    assert {(p.node_id, d.node_id) for p, d in pairs} == {
        (p, d) for p in ("p0", "p1") for d in ("d0", "d1")}
    binding = Binding.create(*pairs[0], {"prompt": "hello"})
    binding.check(pairs[0][0], "prefill")
    with pytest.raises(ValueError):
        binding.check(msgspec.structs.replace(pairs[0][0], boot="new"), "prefill")
    assert not pairs[0][0].compatible(msgspec.structs.replace(pairs[0][1], version=99))


def test_disabled_mode_and_router_have_no_native_pd_imports():
    import subprocess
    import sys
    for module, absent in (("pypto_serving.serving.server.server", ("pypto_serving.pd", "mooncake", "xgrammar")),
                           ("pypto_serving.router.app", ("pypto_serving.pd", "msgspec", "torch", "mooncake")),
                           ("pypto_serving.pd.coordinator", ("torch", "mooncake", "pypto_serving.pd.adapter"))):
        script = f"import {module}; import sys; assert not any(n in sys.modules for n in {absent!r})"
        subprocess.run([sys.executable, "-c", script], check=True, timeout=20)


@pytest.mark.parametrize("prefix", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("kind,stream", [("completion", False), ("completion", True),
                                        ("chat", False), ("chat", True)])
@pytest.mark.parametrize("unknown", [False, True])
@pytest.mark.parametrize("max_tokens", [1, 3])
@pytest.mark.parametrize("p_tp,d_tp", [(4, 4), (4, 1), (1, 4)])
def test_generation_handoff_overlaps_compute_and_source_cleanup(prefix, kind, stream, unknown, max_tokens,
                                                               p_tp, d_tp):
    from types import SimpleNamespace
    import httpx
    from pypto_serving.config.types import GenerateConfig, RuntimeConfig
    from pypto_serving.pd.adapter import DeepSeekDSpark
    from pypto_serving.pd.integration import Chunk, Server
    from pypto_serving.pd.protocol import Completion, Plan, digest
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
            def is_ready(self): return True

            def __init__(self, role, enabled):
                self.role = role
                self.kv_cache_manager = caches(enabled, partitions=16 // (p_tp if role == "prefill" else d_tp))
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
                    if k not in ("stream", "output_parser_spec", "worker_release_ack")})
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
            topology = Topology.for_tp(16, p_tp if name == "p" else d_tp)
            service.registry = dict(geometry=geometry, regions=(), topology=topology)
            service.node = Node(name, service.role, name + "-boot",
                adapter.contract("test-model", geometry), digest(service.registry), topology,
                prefix=service.cache.enable_prefix_cache)
            service.core.service = service

        class Peer:
            geometry = d.geometry
            closed = False
            async def call(self, operation, payload):
                wire = msgspec.msgpack.decode(msgspec.msgpack.encode(payload))
                if operation == "plan":
                    plan = decode(wire["plan"], Plan)
                    invalid = msgspec.structs.replace(plan, units=plan.units[:-1])
                    with pytest.raises(ValueError, match="completion units"):
                        await d.control(p.node, p.geometry, operation,
                                        {**wire, "plan": msgspec.to_builtins(invalid)})
                    if plan.final:
                        invalid = msgspec.structs.replace(plan, copies=tuple(
                            copy for copy in plan.copies if copy.component != "bootstrap_hidden"))
                        with pytest.raises(ValueError, match="required destination pages"):
                            await d.control(p.node, p.geometry, operation,
                                            {**wire, "plan": msgspec.to_builtins(invalid)})
                value = await d.control(p.node, p.geometry, operation, wire)
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
        app = create_router_app(router_config(config), client_factory=lambda _: httpx.AsyncClient(
            transport=httpx.MockTransport(transport)))
        async with app.router.lifespan_context(app):
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

    asyncio.run(run())
