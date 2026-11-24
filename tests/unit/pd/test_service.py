# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Essential PD transfer, Prefix and lease safety using existing cache fixtures."""

import asyncio
import math
import socket
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
from types import SimpleNamespace
from dataclasses import dataclass, replace
from unittest.mock import AsyncMock, Mock

import msgspec
import pytest

from pypto_serving.config.types import KVCacheGroupSpec, KVCacheSpec
from pypto_serving.model.deepseek_dspark.cache_offload import CacheOffloadExtension
from pypto_serving.pd.worker import Progress, ProgressServer, WorkerRuntime, WorkerSettings
from pypto_serving.pd.adapter import CACHE_COMPONENTS, COMPONENTS, DSparkExtension, DeepSeekDSpark
from pypto_serving.pd.config import Config, Endpoint
from pypto_serving.pd.integration import Chunk
from pypto_serving.pd.planner import Geometry, plan_chunk, rank_mapping
from pypto_serving.pd.protocol import Binding, CapacityExhausted, Component, Completion, Continuation, Contract, Copy, Node, Placement, Plan, Topology, digest
from pypto_serving.pd.service import Request, Service
from pypto_serving.pd.session import Session
from pypto_serving.pd.transfer import Mooncake, Region, TransferError, TransferWorker, _serve, host_copy
from pypto_serving.serving.memory.kv_cache import KvCacheManager
from pypto_serving.serving.memory.tiering.simpler import serve_copy
from tests.unit.serving.test_grouped_prefix_cache import GEOMETRY


def test_hixl_link_setup_covers_all_legal_rank_pairs():
    runtime = WorkerRuntime.__new__(WorkerRuntime)
    runtime.settings = SimpleNamespace(provider="hixl", peer_capacity=1)
    runtime.model = SimpleNamespace(ranks=8, tp=4, regions=lambda: {"kv": None})
    runtime.peers, runtime.peer_fingerprints, runtime.peer_regions = {}, {}, {}
    runtime.peer_identities = {}
    installed = {}

    def install(rank, operation, *, regions, peer):
        assert operation == "install"
        assert peer == identity
        assert rank not in installed
        installed[rank] = {region.rank for region in regions}

    runtime.owners = tuple(SimpleNamespace(call=lambda op, rank=rank, **kwargs: install(rank, op, **kwargs))
                           for rank in range(8))
    source = Topology.for_tp(8, 4)
    target = Topology.for_tp(8, 2)
    descriptor = msgspec.to_builtins(dict(topology=target, regions=[
        Region("decode-boot", rank, "kv", 4096, {}) for rank in range(target.ranks)
    ]))
    identity = ("d", "decode-boot", digest(descriptor))
    with ThreadPoolExecutor(max_workers=8) as runtime.pool:
        runtime._install(descriptor, identity)
        runtime._install(descriptor, identity)
    expected = {rank: set() for rank in range(8)}
    for p in range(len(source.groups)):
        for d in range(len(target.groups)):
            for source_rank, destination_rank in rank_mapping(source, target, p, d):
                expected[source_rank].add(destination_rank)
    assert installed == expected


def cache(length, partitions=1, enabled=True, ori_blocks=9, slots=4, state_pages=None):
    state_pages = state_pages or {}
    groups = tuple(KVCacheGroupSpec(name, (0,), KVCacheSpec(page, page // ratio, ratio),
        ori_blocks if name == "ori" else (math.ceil((length + 512) / page) if window is None else state_pages.get(name, blocks)),
        num_blocks=max(slots * state_pages.get(name, blocks) if window is not None else slots * math.ceil((length + 512) / page), 32)
        if name != "ori" else slots * ori_blocks,
        num_partitions=partitions, sliding_window=window)
        for name, page, ratio, blocks, window in GEOMETRY)
    result = KvCacheManager(enable_prefix_cache=enabled)
    result.init_groups(groups, max_batch_size=2)
    return result

def geometries(cache):
    groups = {g.name: g for g in cache.group_specs}
    return tuple(Geometry(c.name, (0,), cache.group_num_blocks(c.group) + 4,
        groups[c.group].spec.storage_block_size, 64, "uint8") for c in CACHE_COMPONENTS) + (
            Geometry("bootstrap_hidden", (0,), 64, 128, 64, "uint8"),)


def warm(cache, tokens, length, partition=0, group_names=None):
    if length:
        cache.ensure_group_blocks("warm", length, partition=partition)
        cache.cache_group_blocks("warm", cache.compute_group_block_hashes(tokens, group_names=group_names), length, {})
        cache.release_all_group_requests("warm")

def handoff(length, p_hit=0, d_hit=0, p_tp=4, d_tp=1, ori_blocks=9, enabled=True, compact=False):
    tokens = (1000000,) + tuple(range(1, length))
    names = ("hca_state", "csa_state", "csa_inner_state")
    p = cache(length, 16 // p_tp, enabled, ori_blocks, state_pages=dict(zip(names, (145, 517, 517))) if compact else None)
    d = cache(length, 16 // d_tp, enabled, ori_blocks, state_pages=dict(zip(names, (35, 77, 77))) if compact else None)
    warm(p, tokens, p_hit)
    warm(d, tokens, d_hit, group_names=("cmp_c128", "cmp_c4", "idx"))
    adapter = DeepSeekDSpark()
    pg, dg = geometries(p), geometries(d)
    nodes = tuple(Node(name, role, name + "-boot", adapter.contract("model", geometry), name + "-registry",
                       Topology.for_tp(16, tp)) for name, role, geometry, tp in
                  (("p", "prefill", pg, p_tp), ("d", "decode", dg, d_tp)))
    binding = Binding.create(*nodes, {})
    request = binding.request
    source_request = request + "-source"
    prefix = adapter.prefix_spec(tokens, d)
    reservation = d.reserve_group_cache(request, request, length + 512, prompt_token_count=length,
        prefix_block_hashes={k: [bytes.fromhex(h) for h in values] for k, values in prefix.hashes.items()},
        shareable_group_names=tuple(prefix.hashes))
    core = SimpleNamespace(kv_cache_manager=d, is_ready=lambda: True)
    config = SimpleNamespace(prefill=(0,), decode=(0,), capacity=4)
    destination = Service(config, SimpleNamespace(node_id="d"), "decode", adapter, core, None)
    destination.node, destination.geometry = nodes[1], dg
    entry = destination.register_http(binding, {})
    entry.continuation = Continuation(tokens, {"max_new_tokens": 512}, None, None)
    entry.normalized_digest = digest(entry.continuation)
    entry.prefix, entry.reservation, entry.state_slots = prefix, reservation, {"bootstrap": 1}
    placement = Placement(request, reservation.prefix_hit_tokens, reservation.partition,
                          reservation.block_ids_by_group, entry.state_slots)
    hashes = p.compute_group_block_hashes(list(tokens))
    _, actual_p_hit, _ = p.acquire_group_prefix_blocks(source_request, hashes,
        max_cache_hit_tokens=p_hit, partition=0)
    assert actual_p_hit == p_hit
    plans = []
    for number, start in enumerate(range(p_hit, length, 128)):
        end = min(start + 128, length)
        tables = p.ensure_group_blocks(source_request, end, partition=0)
        chunk = Chunk(request, number, start, end, end == length, 7 if end == length else None,
                      {k: tuple(v) for k, v in tables.items()}, 0, {"bootstrap": 0})
        plan = plan_chunk(binding, chunk, placement, nodes[0].contract, p.group_specs, pg, dg, p_hit,
            rank_mapping(nodes[0].topology, nodes[1].topology, 0, reservation.partition), entry.normalized_digest,
            adapter.state_ranges(p.group_specs, end) if chunk.final else {})
        destination.validate_plan(entry, plan, pg, p_hit, nodes[0].topology)
        entry.chunk, entry.end = number + 1, end
        plans.append(plan)
    return SimpleNamespace(p=p, d=d, destination=destination, entry=entry, plans=plans, tokens=tokens,
                           source_geometry=pg, nodes=nodes, p_hit=p_hit, source_request=source_request)


def test_rdma_connects_only_requested_pairs_for_the_installed_member():
    case = handoff(128, p_tp=2, d_tp=2)
    plan = case.plans[-1]
    p, d = case.nodes
    runtime = WorkerRuntime.__new__(WorkerRuntime)
    runtime.settings = SimpleNamespace(provider="hixl", role="prefill", boot=p.boot)
    runtime.lock, runtime.uncertain, runtime.accepting = threading.Lock(), False, True
    runtime.peer_identities = {d.boot: d.key}
    runtime.peers = {d.boot: tuple(Region(d.boot, rank, "kv", 4096, {}) for rank in range(d.topology.ranks))}
    connected = {}

    def connect(rank, operation, *, peer, ranks):
        assert operation == "connect_ranks"
        assert peer == d.key
        connected[rank] = tuple(ranks)

    runtime.owners = tuple(SimpleNamespace(call=lambda op, rank=rank, **kwargs: connect(rank, op, **kwargs))
                           for rank in range(p.topology.ranks))
    request = dict(peer=d.key, rank_pairs=plan.mapping)
    with ThreadPoolExecutor(max_workers=2) as runtime.pool:
        for rejected, reason in ((dict(peer=("stale", *d.key[1:])), "identity"),
                                 (dict(rank_pairs=((0, 1000),)), "rank pairs")):
            result = msgspec.msgpack.decode(runtime.handle("connect_ranks", msgspec.msgpack.encode(
                {**request, **rejected})))
            assert reason in result["error"] and not result["uncertain"]
        assert not connected
        runtime.handle("connect_ranks", msgspec.msgpack.encode(request))
    assert connected == {source: (destination,) for source, destination in plan.mapping}


def test_hixl_receiver_prepares_only_ranks_in_the_authorized_plan():
    async def run():
        case = handoff(128, p_tp=2, d_tp=2)
        service, entry, plan = case.destination, case.entry, case.plans[-1]
        service.config.provider = "hixl"
        entry.pending, entry.pending_digest = plan, digest(plan)
        calls = []

        async def worker(operation, ranks):
            calls.append((operation, ranks))

        service.worker = worker
        payload = {"binding": msgspec.to_builtins(plan.binding), "plan_digest": "wrong-digest"}
        with pytest.raises(ValueError, match="authorized plan"):
            await service.control(case.nodes[0], case.source_geometry, "prepare_hixl_plan", payload)
        assert not calls
        payload["plan_digest"] = entry.pending_digest
        await service.control(case.nodes[0], case.source_geometry, "prepare_hixl_plan", payload)
        assert calls == [("prepare_link", sorted({copy.destination_rank for copy in plan.copies}))]

    asyncio.run(run())


@asynccontextmanager
async def hixl_control_pair(case):
    destination = case.destination
    disconnected = asyncio.Event()

    async def accept(reader, writer):
        await destination._accept(reader, writer)
        disconnected.set()

    listener = await asyncio.start_server(accept, "127.0.0.1", 0)
    port = listener.sockets[0].getsockname()[1]
    p_endpoint = Endpoint("127.0.0.1", 20001, "p")
    d_endpoint = Endpoint("127.0.0.1", 20002, "d", control_port=port)
    config = Config((p_endpoint,), (d_endpoint,), provider="hixl", handoff_timeout=2)
    destination.config, destination.endpoint = config, d_endpoint
    destination.core.scheduler = SimpleNamespace(requests={})
    prefill = Service(config, p_endpoint, "prefill", None,
                      SimpleNamespace(kv_cache_manager=SimpleNamespace(tiering=None)), None)
    for service, node, geometry in zip((prefill, destination), case.nodes,
                                      (case.source_geometry, destination.geometry)):
        service.registry = msgspec.to_builtins(dict(topology=node.topology, geometry=geometry))
        service.node = msgspec.structs.replace(node, provider="hixl", registry=digest(service.registry))
    case.entry.binding = msgspec.structs.replace(case.entry.binding, prefill=prefill.node.key,
                                                decode=destination.node.key)
    async with listener:
        yield prefill, destination, disconnected


@pytest.mark.parametrize("ipc_disconnect", [False, True])
def test_hixl_prepare_uncertainty_survives_worker_and_control_round_trips(ipc_disconnect):
    async def run():
        case = handoff(128)
        async with hixl_control_pair(case) as (prefill, destination, disconnected):
            runtime = WorkerRuntime.__new__(WorkerRuntime)
            runtime.settings = SimpleNamespace(provider="hixl", role="decode")
            runtime.lock, runtime.uncertain, runtime.accepting = threading.Lock(), False, True
            calls = []

            def listen(operation):
                assert operation == "prepare_link"
                raise RuntimeError("listener failed after registering device memory")

            runtime.owners = tuple(SimpleNamespace(call=listen) for _ in range(destination.node.topology.ranks))

            async def call_worker(operation, payload):
                ranks = msgspec.msgpack.decode(payload)
                calls.append(ranks)
                if ipc_disconnect and ranks != [-1]:
                    raise ConnectionError("listener outcome unknown")
                return runtime.handle(operation, payload)

            destination.core.call_worker = call_worker
            entry = case.entry
            entry.pending = msgspec.structs.replace(case.plans[-1], binding=entry.binding)
            entry.pending_digest = digest(entry.pending)
            other = destination.register_http(Binding.create(prefill.node, destination.node, {}), {})
            other.pending = msgspec.structs.replace(entry.pending, binding=other.binding)
            other.pending_digest = digest(other.pending)
            other_request = other.binding.request
            other.reservation = case.d.reserve_group_cache(other_request, other_request, 256,
                                                            prompt_token_count=128)
            case.d.authorize_group_cache_write(entry.binding.request)
            case.d.authorize_group_cache_write(other_request)
            reader, writer, _ = await prefill._connect_peer(destination.node)
            session = Session(reader, writer)
            try:
                await session.call("open_link", destination.node.registry)
                payload = dict(binding=entry.binding, plan_digest="wrong-digest")
                with ThreadPoolExecutor() as runtime.pool:
                    with pytest.raises(TransferError, match="destination ranks") as bad_rank:
                        await destination.worker("prepare_link", [-1])
                    assert not bad_rank.value.uncertain

                    with pytest.raises(TransferError, match="authorized plan") as rejected:
                        await session.call("prepare_hixl_plan", payload)
                    assert not rejected.value.uncertain and destination.is_ready()

                    payload["plan_digest"] = entry.pending_digest
                    with pytest.raises(TransferError, match="listener") as failed:
                        await session.call("prepare_hixl_plan", payload)
                    with pytest.raises(TransferError, match="node is unavailable") as blocked:
                        await session.call("prepare_hixl_plan",
                                           dict(binding=other.binding, plan_digest=other.pending_digest))
                assert failed.value.uncertain and not blocked.value.uncertain
                assert runtime.uncertain is not ipc_disconnect
                assert len(calls) == 2
                assert not destination.is_ready()
                assert entry.error is None
                prefill.fail(Request(entry.binding), failed.value)
                assert not prefill.gated
                await session.call("abort", dict(binding=entry.binding, stopped=True))
                assert case.d.group_cache_reservation(entry.binding.request) is None
                await destination.abort(other)
                assert case.d.group_cache_reservation(other_request) is not None
            finally:
                await session.close()
                await asyncio.wait_for(disconnected.wait(), 2)
            assert case.d.group_cache_reservation(other_request).state.value == "QUARANTINED"

    asyncio.run(run())


@pytest.mark.parametrize("lost_ack", ["open_link", "retire_link"])
def test_hixl_retirement_recovers_after_lost_ack(monkeypatch, lost_ack):
    from pypto_serving.pd import service as service_module

    async def run():
        case = handoff(128)
        async with hixl_control_pair(case) as (prefill, destination, disconnected):
            actions = []
            original_send = service_module.send
            dropped = False

            async def send(writer, message):
                nonlocal dropped
                already_retired = prefill.node.key in destination.hixl_retired
                if not dropped and "result" in message and (lost_ack == "open_link" or already_retired):
                    dropped = True
                    writer.close()
                    return
                await original_send(writer, message)

            async def worker(operation, value=None):
                actions.append(operation)

            def bind_domain(session, peer, registry):
                session.transfer_domain = SimpleNamespace(close=lambda: prefill.retire_hixl_session(peer, session))

            monkeypatch.setattr(service_module, "send", send)
            prefill.worker = worker
            prefill.bind_transfer_domain = bind_domain
            if lost_ack == "open_link":
                case.entry.response_closed = True
                with pytest.raises(ConnectionError, match="session lost"):
                    await prefill.connect(destination.node)
                await asyncio.wait_for(disconnected.wait(), 2)
                assert case.entry.source_done.is_set() and not destination.requests
                assert destination.is_ready()
                assert actions == []
            else:
                case.entry.source_done.set()
                destination.finish(case.entry)
                await prefill.connect(destination.node)
                assert actions == ["install"]

            await prefill.close()
            assert destination.hixl_retired == {prefill.node.key}
            assert dropped
            assert actions[-2:] == ["close_peer", "release"]
            assert actions.count("install") == (lost_ack == "retire_link")

    asyncio.run(run())


def test_hixl_retirement_does_not_reconnect_to_a_different_decode_boot():
    async def run():
        case = handoff(128)
        async with hixl_control_pair(case) as (prefill, destination, disconnected):
            reader, writer, _ = await prefill._connect_peer(destination.node)
            session = Session(reader, writer)
            await session.close()
            await asyncio.wait_for(disconnected.wait(), 2)
            peer = destination.node
            destination.node = msgspec.structs.replace(peer, boot="different")

            async def worker(operation, key):
                assert (operation, key) == ("close_peer", peer.key)

            prefill.worker = worker
            with pytest.raises(ValueError, match="identity/contract changed"):
                await prefill.retire_hixl_session(peer, session)
            assert not destination.hixl_retired

    asyncio.run(run())


def test_hixl_retirement_after_disconnect_rejects_unknown_authorized_writes():
    async def run():
        case = handoff(128)
        async with hixl_control_pair(case) as (prefill, destination, disconnected):
            async def connect():
                reader, writer, _ = await prefill._connect_peer(destination.node)
                return Session(reader, writer)

            session = await connect()
            await session.call("open_link", destination.node.registry)
            destination.cache.authorize_group_cache_write(case.entry.binding.request)
            await session.close()
            await asyncio.wait_for(disconnected.wait(), 2)
            assert destination.gated and not case.entry.source_done.is_set()

            session = await connect()
            try:
                with pytest.raises(TransferError, match="unconfirmed handoffs") as failed:
                    await session.call("retire_link", destination.node.registry)
                assert not failed.value.uncertain
                assert not destination.hixl_retired
            finally:
                await session.close()

    asyncio.run(run())


def test_rdma_probe_accepts_one_source_rank_replicated_to_two_decode_ranks():
    from examples.pd_rdma_probe import CheckedCache, GUARDS, PAGE_BYTES, SOURCE_PAGES, source

    buffers = [bytearray(GUARDS), bytearray(GUARDS)]
    owners = tuple(SimpleNamespace(rank=rank, read=lambda memory=memory: bytes(memory))
                   for rank, memory in enumerate(buffers))
    cache = CheckedCache(owners, tp=2)
    request = "probe"
    lease = cache.reserve_group_cache(request, request, len(SOURCE_PAGES), prompt_token_count=len(SOURCE_PAGES))
    cache.authorize_group_cache_write(request)
    mapping = rank_mapping(Topology.for_tp(1, 1), Topology.for_tp(2, 2), 0, lease.partition)
    cache.source_ranks[request] = {target: rank for rank, target in mapping}
    cache.seeds[request] = 1
    for rank, target in mapping:
        contents = source(rank, cache.seeds[request])
        for src, dst in zip(SOURCE_PAGES, lease.block_ids_by_group["kv"]):
            src, dst = src * PAGE_BYTES, dst * PAGE_BYTES
            buffers[target][dst:dst + PAGE_BYTES] = contents[src:src + PAGE_BYTES]
    assert cache.commit_group_cache(request).state.value == "READY"


def test_hixl_prefill_links_are_scoped_to_decode_boot_and_rank(monkeypatch):
    native = pytest.importorskip("simpler.l4_pd_rdma")
    Hixl = native.HixlTransferOwner
    d1_key, d2_key = ("d1", "d1-boot", "registry1"), ("d2", "d2-boot", "registry2")
    servers = {}
    writes = []

    class Link:
        def __init__(self, device, host, port, *, peer=None):
            self.key = host, port
            self.regions = {}

        def register(self, name, base, length):
            assert name not in self.regions
            self.regions[name] = base, length

        def listen(self):
            assert self.key not in servers
            servers[self.key] = self

        def connect(self):
            self.remote = servers[self.key]

        def remote_region(self, name):
            return self.remote.regions[name]

        def batch_put(self, segments):
            writes.append((self.key, segments))

        def close(self):
            pass

    monkeypatch.setattr(native, "HixlDeviceLink", Link)
    monkeypatch.setenv("PYPTO_HIXL_DEVICE_ENDPOINTS", '[["192.0.2.1", 0]]')
    monkeypatch.setenv("PYPTO_HIXL_PORT", "23001")
    d1 = Hixl("127.0.0.1", "decode")
    d1_region = d1.register(d1_key[1], 0, "kv", 4096, 256)
    d1.prepare()
    d1.prepare()
    monkeypatch.setenv("PYPTO_HIXL_PORT", "23002")
    d2 = Hixl("127.0.0.1", "decode")
    d2_region = d2.register(d2_key[1], 0, "kv", 8192, 256)
    d2.prepare()
    p1, p2 = (Hixl("127.0.0.1", "prefill") for _ in range(2))
    p1.register("p1", 0, "kv", 12288, 256)
    p2.register("p2", 0, "kv", 16384, 256)
    p1.install((d1_region,), peer=d1_key)
    p1.install((d2_region,), peer=d2_key)
    p2.install((d1_region,), peer=d1_key)
    p1.connect(d1_key, [0])
    p1.connect(d2_key, [0])
    p2.connect(d1_key, [0])
    p1.put_sync_host(d1_key, (host_copy(Copy("kv", 0, 0, 0, 16, 8)),))
    p1.close_peer(d1_key)
    p2.put_sync_host(d1_key, (host_copy(Copy("kv", 0, 0, 0, 48, 8)),))
    p1.put_sync_host(d2_key, (host_copy(Copy("kv", 0, 0, 0, 64, 8)),))
    assert writes == [
        (("127.0.0.1", 23001), [(12288, 4112, 8)]),
        (("127.0.0.1", 23001), [(16384, 4144, 8)]),
        (("127.0.0.1", 23002), [(12288, 8256, 8)]),
    ]
    assert servers["127.0.0.1", 23001].regions == {"kv": (4096, 256)}
    with pytest.raises(ValueError, match="installed link"):
        p2.put_sync_host(d2_key, (host_copy(Copy("kv", 0, 0, 0, 64, 8)),))
    for peer in (p1, p2, d1, d2):
        peer.close()


def test_hixl_retirement_is_independent_for_each_prefill():
    async def run():
        case = handoff(128)
        p1, d = case.nodes
        p2 = msgspec.structs.replace(p1, node_id="p2", boot="p2-boot")
        restarted = msgspec.structs.replace(p2, boot="new-boot")
        service = case.destination
        for peer in (p1, p2):
            await service._hixl_request(peer, case.source_geometry, "open_link", d.registry)
        with pytest.raises(ValueError, match="boot mismatch"):
            await service._hixl_request(restarted, case.source_geometry, "open_link", d.registry)
        p2_entry = service.register_http(Binding.create(p2, d, {}), {})
        with pytest.raises(ValueError, match="active or unconfirmed"):
            await service._hixl_request(p2, case.source_geometry, "retire_link", d.registry)
        case.entry.source_done.set()
        await service._hixl_request(p1, case.source_geometry, "retire_link", d.registry)
        with pytest.raises(ValueError, match="retired"):
            service.register_http(Binding.create(p1, d, {}), {})
        with pytest.raises(ValueError, match="not active"):
            await service._hixl_request(p1, case.source_geometry, "begin", {})
        service.finish(case.entry)
        p2_entry.source_done.set()
        service.finish(p2_entry)
        with pytest.raises(RuntimeError, match="stop Prefill senders"):
            await service.close()
        await service._hixl_request(p2, case.source_geometry, "retire_link", d.registry)
        assert service.hixl_retired == {p1.key, p2.key}

    asyncio.run(run())


def test_hixl_prefill_retires_all_decode_peers_even_if_one_ack_fails():
    async def run():
        config = Config((Endpoint("127.0.0.1", 20001),),
                        (Endpoint("127.0.0.1", 20002), Endpoint("127.0.0.1", 20003)), provider="hixl")
        service = Service(config, config.prefill[0], "prefill", None,
                          SimpleNamespace(kv_cache_manager=SimpleNamespace(tiering=None)), None)
        actions = []

        async def worker(op, value=None):
            actions.append((op, value))

        async def retire(op, registry):
            assert op == "retire_link"
            assert any(op == "close_peer" and key[2] == registry for op, key in actions)
            actions.append((registry, None))
            if registry == "registry1":
                raise ConnectionError("no acknowledgement")

        async def close():
            actions.append(("session_closed", None))

        async def reconnect(peer):
            assert peer.registry == "registry1"
            raise ConnectionError("still disconnected")

        service.worker = worker
        service._connect_peer = reconnect
        for i, node in enumerate(config.decode, 1):
            peer = Node(node.node_id, "decode", str(i), None, f"registry{i}", Topology.for_tp(1, 1))
            session = SimpleNamespace(call=retire, close=close, closed=False)
            session.transfer_domain = SimpleNamespace(
                close=lambda peer=peer, session=session: service.retire_hixl_session(peer, session))
            service.hixl_sessions[node.node_id] = peer.key, session
        with pytest.raises(RuntimeError, match="every Decode peer"):
            await service.close()
        assert {op for op, _ in actions} == {"close_peer", "registry1", "registry2", "session_closed"}

    asyncio.run(run())


@pytest.mark.parametrize("p_tp,d_tp,p_hit,d_hit", [(4, 4, 0, 8192), (4, 1, 8192, 4096)])
@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("length", [8205, 8320])
def test_suffix_handoff_preserves_shared_prefix_and_final_state(p_tp, d_tp, p_hit, d_hit, compact, length):
    case = handoff(length, p_hit, d_hit, p_tp, d_tp, compact=compact)
    final = case.plans[-1]
    assert final.valid_from == {"ori": max(0, (length - 1) // 128 * 128 - 128)}
    assert {"hca_state", "csa_state", "csa_inner_state", "bootstrap_hidden"} <= {c.component for c in final.copies}
    assert all(c.component != "ori" for plan in case.plans[:-1] for c in plan.copies)
    geometry = {g.name: g for g in geometries(case.d)}
    groups = {g.name: g for g in case.d.group_specs}
    components = {c.name: c for c in COMPONENTS}
    for plan in case.plans:
        Completion(digest(plan), plan.units).check(digest(plan), plan.units)
        for copy in plan.copies:
            group = components[copy.component].group
            if group.endswith("state"):
                source, target = case.p._group_pools[group], case.d._group_pools[group]
                a = tuple(source.local_block_id(block) if block is not None else -1
                          for block in source.request_blocks[case.source_request])
                b = case.entry.reservation.block_ids_by_group[group]
                page = groups[group].spec.token_capacity
                first = max(0, (length - 1) // 128 * 128 - groups[group].sliding_window) // page
                logical = {a[q % len(a)]: q for q in range(first, math.ceil(length / page))}
                sg = next(g for g in case.source_geometry if g.name == copy.component)
                dg = geometry[copy.component]
                for offset in range(0, copy.length, sg.page_bytes):
                    sp = (copy.source_offset + offset) // sg.page_bytes % sg.blocks
                    dp = (copy.destination_offset + offset) // dg.page_bytes % dg.blocks
                    assert dp == b[logical[sp] % len(b)]
            if group not in ("cmp_c128", "cmp_c4", "idx"):
                continue
            geo = geometry[copy.component]
            writable = set(case.entry.reservation.block_ids_by_group[group][d_hit // groups[group].spec.token_capacity:])
            assert all((copy.destination_offset + offset) // geo.page_bytes % geo.blocks in writable
                       for offset in range(0, copy.length, geo.page_bytes))
    with pytest.raises(ValueError):
        Completion(digest(final), final.units[:-1]).check(digest(final), final.units)


def test_owner_rejects_out_of_bounds_and_retains_unknown_registration():
    class Engine:
        fail = False
        def initialize(self, *args): return 0
        def get_rpc_port(self): return 1234
        def register_memory(self, *args): return 0
        def unregister_memory(self, *args): return 0
        def batch_transfer_sync_write(self, *args): return int(self.fail)
    engine = Engine()
    provider = Mooncake("host", lambda: engine)
    provider.register("local", 0, "kv", 4096, 1024)
    provider.install((Region("peer", 1, "kv", 1024, {"endpoint": "peer:1", "address": 8192}),))
    provider.write("peer", (Copy("kv", 0, 1, 0, 0, 64),))
    with pytest.raises(ValueError):
        provider.write("peer", (Copy("kv", 0, 1, 1000, 0, 64),))
    engine.fail = True
    with pytest.raises(TransferError) as error:
        provider.write("peer", (Copy("kv", 0, 1, 0, 0, 64),))
    assert error.value.uncertain
    with pytest.raises(TransferError):
        provider.release()
    assert provider.local


@pytest.mark.parametrize("location", ["", ", location='cpu'"])
def test_mooncake_preflight_requires_location_without_initializing(tmp_path, monkeypatch, location):
    package = tmp_path / "mooncake"
    package.mkdir()
    (package / "__init__.py").touch()
    (package / "engine.py").write_text(
        "class TransferEngine:\n"
        "    def __init__(self): raise AssertionError('unexpected initialization')\n"
        f"    def register_memory(self, address, extent{location}): raise AssertionError('unexpected registration')\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    if location:
        Mooncake.check_available()
    else:
        with pytest.raises(ValueError, match="Mooncake #2191"):
            Mooncake.check_available()


def test_pd_unmap_retries_after_provider_disconnect(monkeypatch):
    from simpler import buffer_export
    from pypto_serving.pd import transfer

    engine, context, channel = Mock(), Mock(), Mock()
    first, second = (Mock(base=4096, descriptor=SimpleNamespace(nbytes=64)) for _ in range(2))
    second.close.side_effect = [RuntimeError("incomplete unmap"), None]
    calls = Mock()
    calls.attach_mock(engine.release, "disconnect")
    calls.attach_mock(first.close, "close_first")
    calls.attach_mock(second.close, "close_second")
    monkeypatch.setenv("MC_TRANSFER_TIMEOUT", "")
    monkeypatch.setattr(transfer, "configure_ports", Mock())
    monkeypatch.setattr(transfer, "configure_profiler", Mock())
    monkeypatch.setattr(transfer, "DeviceContext", Mock(return_value=context))
    monkeypatch.setattr(buffer_export, "BufferExport", SimpleNamespace)
    monkeypatch.setattr(buffer_export, "BufferMapping", Mock(side_effect=[first, second]))
    commands = [dict(operation="register", exported={"device_id": 0}, extent=64, name=name)
                for name in ("kv", "state")] + [dict(operation=op) for op in ("release", "stop")]
    monkeypatch.setattr(transfer, "_receive", Mock(side_effect=[
        dict(boot="boot", sequence=i, **cmd) for i, cmd in enumerate(commands, 1)]))
    send = Mock()
    monkeypatch.setattr(transfer, "_send", send)
    transfer._serve(channel, "boot", 0, 0, "host", "test", lambda *_: engine, None)
    assert [c[0] for c in calls.mock_calls] == ["disconnect", "close_first", "close_second", "close_second"]
    assert all("error" not in c.args[1] for c in send.call_args_list)
    context.close.assert_called_once()


def test_probe_failure_retires_all_unregistered_owners(monkeypatch):
    from pypto_serving.pd import transfer

    processes = [Mock(pid=100 + rank, exitcode=0) for rank in range(2)]
    processes[0].is_alive.return_value = False
    processes[1].is_alive.side_effect = [True, False]
    monkeypatch.setattr(transfer.multiprocessing, "get_context",
                        Mock(return_value=SimpleNamespace(Process=Mock(side_effect=processes))))
    probe = Mock(side_effect=[100, TransferError("probe unknown", uncertain=True)])
    monkeypatch.setattr(transfer.TransferWorker, "ready", probe)
    rpc = Mock(side_effect=AssertionError("startup rollback must not require an RPC"))
    monkeypatch.setattr(transfer.TransferWorker, "call", rpc)
    model = SimpleNamespace(ranks=2, worker=lambda: SimpleNamespace(dc=SimpleNamespace(device_ids=(0, 1))))
    runtime = WorkerRuntime(WorkerSettings("boot", "host", "mooncake", 2, 2), model)
    with pytest.raises(TransferError, match="probe unknown"):
        runtime.worker_ready()
    assert all(owner.process is None and owner.parent.fileno() == -1 for owner in runtime.owners)
    processes[0].terminate.assert_not_called()
    processes[1].terminate.assert_called_once()
    rpc.assert_not_called()
    runtime.close()


def test_hixl_and_ddr_start_distinct_consumers_for_the_same_device(monkeypatch):
    from pypto_serving.pd import transfer, worker

    get_context = Mock()
    monkeypatch.setattr(transfer.multiprocessing, "get_context", get_context)
    monkeypatch.setattr(TransferWorker, "ready", lambda self: None)
    monkeypatch.setattr(worker, "provider_class", lambda name: SimpleNamespace(endpoints=lambda: (8,)))
    runtime = SimpleNamespace(dc=SimpleNamespace(device_ids=(8,)))
    runner = SimpleNamespace(_shared_l3_worker=lambda: runtime)
    model = SimpleNamespace(ranks=1, worker=lambda: runtime)
    pd = WorkerRuntime(WorkerSettings("boot", "host", "hixl", 1, 2, role="decode"), model)
    endpoint = Mock()
    extension = CacheOffloadExtension(runner, (endpoint,), Mock(), Mock(), previous=lambda **_: SimpleNamespace(
        worker_ready=pd.worker_ready, allocation_options=DSparkExtension.allocation_options))
    try:
        extension.worker_ready()
        pd_process, ddr_process = [call.kwargs for call in get_context.return_value.Process.call_args_list]
        assert (pd_process["name"], pd_process["target"], pd_process["args"][3:6]) == (
            "pd-transfer-0", _serve, (8, "host", "hixl"))
        assert (ddr_process["name"], ddr_process["target"], ddr_process["args"][0]) == (
            "cache-copy-0", serve_copy, 8)
        assert pd_process["args"][6].keywords == dict(role="decode", rank=0)
        assert extension.allocation_options("cache") == {"exportable": True}
        assert extension.allocation_options("prefill_state", "hidden") == {"exportable": True}
        endpoint.call.assert_called_once_with("ready")
    finally:
        for owner in pd.owners:
            owner.parent.close()
            owner.child.close()
        pd.pool.shutdown()


def test_vmm_owner_unmaps_after_unregistering_and_before_releasing_export(monkeypatch):
    events = []

    @dataclass
    class Export:
        device_id: int

    class Mapping:
        def __init__(self, exported):
            assert exported.device_id == 8
            self.base = 8192
            self.descriptor = SimpleNamespace(nbytes=64)
            events.append("import")

        def close(self):
            events.append("unmap")
            if events.count("unmap") == 1:
                raise RuntimeError("retry unmap")

    monkeypatch.setitem(sys.modules, "simpler.buffer_export", SimpleNamespace(
        BufferExport=Export, BufferExportError=RuntimeError, BufferImportError=RuntimeError, BufferMapping=Mapping))

    class Runtime:
        def export_buffer(self, tensor, *, worker_id):
            assert tensor.nbytes == 64 and worker_id == 0
            events.append("export")
            return Export(8)

        def release_export(self, exported):
            assert exported == Export(8)
            events.append("release_export")

    class Provider:
        def register(self, boot, rank, name, address, extent):
            assert address == 8192 and extent == 64
            events.append("register")
            return Region(boot, rank, name, extent, {})

        def close(self):
            events.append("unregister")

    monkeypatch.setattr("pypto_serving.pd.transfer.DeviceContext", lambda device: Mock())
    owner = TransferWorker("boot", 0, "host", "hixl", role="decode")
    owner.process = SimpleNamespace(pid=1234)
    thread = threading.Thread(target=_serve, args=(owner.child, owner.boot, owner.rank, 8,
        owner.hostname, owner.provider, lambda *_: Provider(), owner.profile_config), daemon=True)
    thread.start()
    try:
        assert owner.register(Runtime(), SimpleNamespace(nbytes=64), "kv") == Region("boot", 0, "kv", 64, {})
        owner.release()
        owner.call("stop")
        assert events == ["export", "import", "register", "unregister", "unmap", "unmap", "release_export"]
        assert not owner.retained
    finally:
        owner.parent.close()
        thread.join(2)
        owner.child.close()
    assert not thread.is_alive()


def service():
    config = Config((Endpoint("p", 8111, "p"),), (Endpoint("d", 8112, "d"),), log_enabled=False)
    contract = Contract("test", "model", "layout", (Component("kv", "kv", "kv"),))
    p, d = (Node(name, role, "boot-" + name, contract, "registry-" + name, Topology.for_tp(1, 1))
            for name, role in (("p", "prefill"), ("d", "decode")))
    cache = KvCacheManager(enable_prefix_cache=True)
    cache.init_groups((KVCacheGroupSpec("kv", (0,), KVCacheSpec(32, 64), 32, num_blocks=64),), max_batch_size=2)
    core = SimpleNamespace(kv_cache_manager=cache, scheduler=SimpleNamespace(requests={}), is_ready=lambda: True)
    result = Service(config, config.endpoint("decode"), "decode", None, core, None)
    result.node = d
    return result, p, d


@pytest.mark.parametrize("cancelled", [False, True])
def test_prefill_handshake_is_bounded(monkeypatch, cancelled):
    async def run():
        s, p, d = service()
        s.config = replace(s.config, handoff_timeout=60 if cancelled else .02)
        s.node, s.role, s.adapter = p, "prefill", Mock()
        reader = asyncio.StreamReader()
        sent = asyncio.Event()
        writer = Mock(drain=AsyncMock(side_effect=sent.set))
        monkeypatch.setattr(asyncio, "open_connection", AsyncMock(return_value=(reader, writer)))
        entry = s.register_http(Binding.create(p, d, {}), {})
        entry.task = asyncio.create_task(s.prefill(entry, Continuation((1,), {}, None, None), d))
        try:
            if cancelled:
                await asyncio.wait_for(sent.wait(), 2)
                await s.abort(entry)
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(asyncio.shield(entry.task), 2)
            else:
                with pytest.raises(asyncio.TimeoutError):
                    await entry.task
            assert entry.continuation is None and not s.pins.active
        finally:
            await s.peers.close()
        writer.transport.abort.assert_called_once()
    asyncio.run(run())


def test_active_handoff_capacity_uses_the_retryable_error():
    s, p, d = service()
    s.config = replace(s.config, capacity=1)
    s.entry(Binding.create(p, d, {}), create=True)
    with pytest.raises(CapacityExhausted):
        s.entry(Binding.create(p, d, {}), create=True)
    assert len(s.requests) == 1


def test_status_reads_current_lease_without_changing_cache():
    s, p, d = service()
    binding = Binding.create(p, d, {})
    entry = s.register_http(binding, {})
    initial = s.cache.reserve_group_cache(binding.request, binding.request, 256, prompt_token_count=128)
    entry.reservation = initial
    s.cache.authorize_group_cache_write(binding.request)
    s.cache.commit_group_cache(binding.request)
    current = s.cache.group_cache_reservation(binding.request)
    references = tuple(b.ref_cnt for pool in s.cache._group_pools.values() for b in pool.blocks)
    assert s.status(binding.request)["reservation"]["state"] == "READY"
    assert initial.state.value == "CONSTRUCTING" and s.cache.group_cache_reservation(binding.request) is current
    assert references == tuple(b.ref_cnt for pool in s.cache._group_pools.values() for b in pool.blocks)
    assert s.status("missing")["reservation"] == {"present": False, "state": None}


def test_cancel_and_unknown_cannot_release_unconfirmed_native_writes():
    async def run():
        s, p, d = service()
        cancelled = Binding.create(p, d, {})
        await s.cancel_binding(cancelled)
        with pytest.raises(ValueError, match="terminal"):
            s.register_http(cancelled, {})
        binding = Binding.create(p, d, {})
        entry = s.register_http(binding, {})
        entry.reservation = s.cache.reserve_group_cache(binding.request, binding.request, 256, prompt_token_count=128)
        s.cache.authorize_group_cache_write(binding.request)
        entry.pending = Plan(binding, "continuation", 0, 0, 128, True, ((0, 0),), (), ((0, "kv"),))
        entry.pending_digest = digest(entry.pending)
        s.fail(entry, TransferError("native result unknown", uncertain=True))
        await s.abort(entry)
        status = s.status(binding.request)
        assert status["reservation"]["state"] == "QUARANTINED" and status["handoff"]["uncertain"]
        assert not status["execution"]["source_stop_confirmed"] and status["handoff"]["confirmed_end"] == 0
        with pytest.raises(RuntimeError, match="native writer"):
            s.cache.release_group_cache(binding.request)
        await s.control(p, (), "abort", {"binding": msgspec.to_builtins(binding), "stopped": True})
        assert s.status(binding.request)["reservation"] == {"present": False, "state": None}
    asyncio.run(run())


def test_final_completion_retires_cancelled_decode():
    async def run():
        s, p, d = service()
        binding = Binding.create(p, d, {})
        entry = s.register_http(binding, {})
        entry.reservation = s.cache.reserve_group_cache(binding.request, binding.request, 256, prompt_token_count=128)
        s.cache.authorize_group_cache_write(binding.request)
        entry.pending = Plan(binding, "continuation", 0, 0, 128, True, ((0, 0),),
                             (Copy("kv", 0, 0, 0, 0, 64),), ((0, "kv"),))
        entry.pending_digest = digest(entry.pending)
        entry.state_task = asyncio.create_task(asyncio.sleep(0))
        s.worker = AsyncMock()
        await s.close_output(entry, terminal=False)
        assert entry.cancelled and entry.task.done() and not entry.source_done.is_set()
        assert s.cache.group_cache_reservation(binding.request) is not None
        payload = dict(binding=msgspec.to_builtins(binding),
                       completion=msgspec.to_builtins(Completion(entry.pending_digest, ())))
        with pytest.raises(ValueError, match="mismatched transfer completion"):
            await s.control(p, (), "complete", payload)
        s.worker.assert_not_called()
        payload["completion"] = msgspec.to_builtins(Completion(entry.pending_digest, entry.pending.units))
        assert await s.control(p, (), "complete", payload) == {}
        assert entry.source_done.is_set() and not s.requests
        assert s.cache.group_cache_reservation(binding.request) is None
        s.worker.assert_awaited_once_with("discard_state", binding.request)
    asyncio.run(run())


def test_prefill_failure_after_first_chunk_retires_decode_reservation():
    async def run():
        case = handoff(256)
        destination, entry, first = case.destination, case.entry, case.plans[0]
        entry.chunk = entry.end = 0
        entry.response_closed = True
        entry.state_task = asyncio.create_task(asyncio.sleep(0))
        destination.core.scheduler = SimpleNamespace(requests={})
        destination.worker = AsyncMock()
        config = service()[0].config
        request = entry.binding.request
        completed = asyncio.Event()
        tables = case.p.ensure_group_blocks(case.source_request, first.end, partition=0)
        chunk = Chunk(request, 0, 0, first.end, False, None,
                      {name: tuple(ids) for name, ids in tables.items()}, 0, {"bootstrap": 0})

        async def prefill(*args, **kwargs):
            source.pins.retain(chunk)
            await source.chunks[request].put(chunk)
            yield None
            await asyncio.wait_for(completed.wait(), 2)
            raise RuntimeError("second prefill chunk failed")

        core = SimpleNamespace(kv_cache_manager=case.p, is_ready=lambda: True,
                               add_request=prefill, abort_request=AsyncMock())
        progress = SimpleNamespace(wait=AsyncMock(return_value=Completion(digest(first), first.units)))
        source = Service(config, config.endpoint("prefill"), "prefill", destination.adapter, core, progress)
        source.node, source.geometry = case.nodes[0], case.source_geometry
        source.worker = AsyncMock(return_value="job")

        async def rpc(operation, payload):
            if operation == "begin":
                reservation = entry.reservation
                return msgspec.to_builtins(Placement(request, 0, reservation.partition,
                                           reservation.block_ids_by_group, entry.state_slots))
            reply = await destination.control(source.node, source.geometry, operation, msgspec.to_builtins(payload))
            if operation == "complete":
                assert entry.end == first.end and entry.end < len(entry.continuation.tokens)
                assert case.d.group_cache_reservation(request).write_authorized
                completed.set()
            return reply

        @asynccontextmanager
        async def peer(*args):
            yield SimpleNamespace(call=rpc, geometry=destination.geometry)

        source.peers.use = peer
        sending = source.register_http(entry.binding, {})
        with pytest.raises(RuntimeError, match="second prefill chunk failed"):
            await asyncio.wait_for(source.prefill(sending, entry.continuation, destination.node), 3)
        assert entry.source_done.is_set() and case.d.group_cache_reservation(request) is None
        assert not destination.requests and not source.requests and not source.pins.active
        destination.worker.assert_awaited_once_with("discard_state", request)
    asyncio.run(run())


@pytest.mark.parametrize('shutdown', [None, 'drain', 'timeout'])
def test_cancelled_submit_ack_is_observed_until_native_completion(shutdown):
    async def run():
        plan = handoff(128).plans[-1]
        job, native = digest(plan), Future()
        runtime = WorkerRuntime.__new__(WorkerRuntime)
        runtime.lock, runtime.jobs, runtime.uncertain = threading.RLock(), {}, False
        parent, child = socket.socketpair()
        accepted, reply = threading.Event(), threading.Event()
        def handle(operation, payload):
            if operation == 'submit':
                runtime.jobs[job] = (plan, (native,))
                accepted.set()
                assert reply.wait(2)
                return msgspec.msgpack.encode(dict(job=job))
            return msgspec.msgpack.encode(runtime.poll(msgspec.msgpack.decode(payload)))
        server, progress = ProgressServer(child, handle, 2), Progress(parent, 2, 1)
        task = asyncio.create_task(progress.call('submit', b'plan'))
        closing = None
        try:
            assert await asyncio.to_thread(accepted.wait, 2)
            task.cancel()
            if shutdown == 'drain':
                closing = asyncio.create_task(progress.close())
                await asyncio.sleep(0)
                with pytest.raises(TransferError, match='closing'):
                    await progress.call('submit', b'rejected')
            reply.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert job in runtime.jobs and job in progress.pending
            if shutdown == 'timeout':
                progress.timeout = .05
                with pytest.raises(TransferError, match='drain deadline') as error:
                    await progress.close()
                assert error.value.uncertain and job in runtime.jobs
                return
            if closing is not None:
                assert not closing.done()
            native.set_result(None)
            for _ in range(100):
                if not runtime.jobs and not progress.pending:
                    break
                await asyncio.sleep(.01)
            assert not runtime.jobs and not progress.pending
            if closing is not None:
                await closing
        finally:
            reply.set()
            await progress.close()
            server.close()
    asyncio.run(run())
