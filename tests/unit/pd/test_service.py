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
import threading
from types import SimpleNamespace

import msgspec
import pytest

from pypto_serving.config.types import KVCacheGroupSpec, KVCacheSpec
from pypto_serving.serving.engine.async_engine import TokenOutput
from pypto_serving.pd.service import ExecutionError
from pypto_serving.pd.worker import Progress, ProgressServer
from pypto_serving.pd.adapter import CACHE_COMPONENTS, COMPONENTS, DeepSeekDSpark
from pypto_serving.pd.config import Config, Endpoint
from pypto_serving.pd.integration import Chunk
from pypto_serving.pd.planner import Geometry, plan_chunk, rank_mapping
from pypto_serving.pd.protocol import Binding, Component, Completion, Continuation, Contract, Copy, Node, Placement, Plan, Topology, digest
from pypto_serving.pd.service import Service
from pypto_serving.pd.transfer import Mooncake, Region, TransferError
from pypto_serving.serving.memory.kv_cache import KvCacheManager
from tests.unit.serving.test_grouped_prefix_cache import GEOMETRY

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


@pytest.mark.parametrize("p_tp,d_tp", [(4, 4), (4, 1)])
@pytest.mark.parametrize("length", [128, 8192, 8205, 8320])
def test_history_hit_does_not_require_or_skip_rolling_and_final_state(p_tp, d_tp, length):
    hit = length // 128 * 128
    case = handoff(length, d_hit=hit, p_tp=p_tp, d_tp=d_tp, compact=True)
    assert case.entry.reservation.prefix_hit_tokens == hit
    assert set(case.entry.prefix.hashes) == {"cmp_c128", "cmp_c4", "idx"}
    assert not case.entry.reservation.shared_block_ids_by_group["ori"]
    assert not case.d._group_pools["ori"].hash_to_block
    final = case.plans[-1]
    assert {"ori", "hca_state", "csa_state", "csa_inner_state", "bootstrap_hidden"} <= {
        c.component for c in final.copies}
    if hit == length:
        assert not {"hca_cmp", "csa_cmp", "idx_k", "idx_scale"} & {c.component for c in final.copies}


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


@pytest.mark.parametrize("damage", ["overlap", "hole", "shared"])
def test_plan_coverage_rejects_invalid_suffix_writes(damage):
    case = handoff(8205, d_hit=4096, compact=True)
    plan = case.plans[-1]
    case.entry.chunk, case.entry.end = plan.chunk, plan.start
    copies = list(plan.copies)
    if damage == "overlap":
        copies.append(copies[0])
    elif damage == "hole":
        copies.pop()
    else:
        index = next(i for i, c in enumerate(copies) if c.component == "csa_cmp")
        copy = copies[index]
        page = next(g for g in case.destination.geometry if g.name == copy.component).page_bytes
        copies[index] = msgspec.structs.replace(copy, destination_offset=
            case.entry.reservation.shared_block_ids_by_group["cmp_c4"][0] * page)
    with pytest.raises(ValueError):
        case.destination.validate_plan(case.entry, msgspec.structs.replace(plan, copies=tuple(copies)),
                                       case.source_geometry, 0, case.nodes[0].topology)


@pytest.mark.parametrize("lost_ack", [False, True])
def test_progress_runs_during_compute_and_rejects_a_lost_ack(lost_ack):
    async def run():
        parent, child = socket.socketpair()
        release, accepted = threading.Event(), threading.Event()
        compute = threading.Thread(target=release.wait)
        compute.start()
        def handle(operation, payload):
            accepted.set()
            if lost_ack:
                raise EOFError("accepted command lost its reply")
            return payload
        server = ProgressServer(child, handle, 1)
        progress = Progress(parent, 1, 1)
        try:
            if lost_ack:
                with pytest.raises(TransferError) as error:
                    await progress.call("submit", b"work")
                assert accepted.is_set() and error.value.uncertain
                with pytest.raises(TransferError):
                    await progress.call("submit", b"late")
            else:
                assert await progress.call("submit", b"work") == b"work" and compute.is_alive()
        finally:
            await progress.close()
            server.close()
            release.set()
            compute.join(2)
        assert not server.thread.is_alive() and not progress.calls
    asyncio.run(run())


def test_decode_compute_error_keeps_the_normal_output_cleanup():
    async def run():
        s, p, d = service()
        binding = Binding.create(p, d, {})
        entry = s.entry(binding, create=True)
        entry.continuation = Continuation((1, 2), {"max_new_tokens": 8}, None, None)
        entry.first_token = 7
        entry.transferred.set()
        s.core.tokenizer = SimpleNamespace(eos_token_id=2)
        closed = []
        async def generate(**kwargs):
            yield TokenOutput(text="", finished=True, finish_reason="error")
        async def close(current, terminal):
            closed.append((current, terminal))
        s.core.add_prefilled_request, s.close_output = generate, close
        with pytest.raises(ExecutionError):
            _ = [output async for output in s.outputs(binding.request)]
        assert isinstance(entry.error, ExecutionError) and not s.gated
        assert closed == [(entry, False)]
    asyncio.run(run())



def test_decode_full_reservation_rejects_before_state_initialization():
    from pypto_serving.pd.protocol import CapacityExhausted, Prefix
    async def run():
        s, p, d = service()
        s.core.config = SimpleNamespace(max_num_running_reqs=1, runtime_config=SimpleNamespace(max_seq_len=4096))
        s.adapter = SimpleNamespace(validate=lambda _: None,
            prefix_spec=lambda tokens, cache: Prefix(len(tokens), 32, {}))
        first, second = [s.register_http(Binding.create(p, d, {}), {}) for _ in range(2)]
        rid = first.binding.request
        first.reservation = s.cache.reserve_group_cache(rid, rid, 256, prompt_token_count=128)
        continuation = Continuation(tuple(range(128)), {"max_new_tokens": 16}, None, None)
        with pytest.raises(CapacityExhausted):
            await s.control(p, (), "begin", dict(binding=msgspec.to_builtins(second.binding),
                continuation=msgspec.to_builtins(continuation)))
        assert second.reservation is None and second.transferred.is_set() and not s.gated
        s.cache.release_group_cache(rid, confirmed_stopped=True)
    asyncio.run(run())
