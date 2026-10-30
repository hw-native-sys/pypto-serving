# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Transfer coverage, publication and completion ownership without device execution."""

import asyncio
from concurrent.futures import Future
import math
from types import SimpleNamespace
import threading

import msgspec
import pytest

from pypto_serving.config.types import KVCacheGroupSpec, KVCacheSpec
from pypto_serving.pd.adapter import COMPONENTS, DeepSeekDSpark
from pypto_serving.pd.integration import Chunk
from pypto_serving.pd.planner import plan_chunk, rank_mapping
from pypto_serving.pd.protocol import Binding, Completion, Continuation, Node, Placement, Plan, Topology, digest
from pypto_serving.pd.service import Service
from pypto_serving.pd.transfer import TransferError
from pypto_serving.pd.worker import Progress, ProgressServer, WorkerRuntime
from pypto_serving.serving.memory.kv_cache import KvCacheManager
from tests.unit.serving.test_grouped_prefix_cache import GEOMETRY, TARGET, query
from tests.unit.test_pd import geometries, warm
from tests.unit.test_pd_cancellation import service


def cache(length, partitions=1, enabled=True, ori_blocks=9, slots=4):
    groups = tuple(KVCacheGroupSpec(name, (0,), KVCacheSpec(page, page // ratio, ratio),
        ori_blocks if name == "ori" else (math.ceil((length + 512) / page) if window is None else blocks),
        num_blocks=max(slots * blocks if window is not None else slots * math.ceil((length + 512) / page), 32)
        if name != "ori" else slots * ori_blocks,
        num_partitions=partitions, sliding_window=window)
        for name, page, ratio, blocks, window in GEOMETRY)
    result = KvCacheManager(enable_prefix_cache=enabled)
    result.init_groups(groups, max_batch_size=2)
    return result


def handoff(length, p_hit=0, d_hit=0, p_tp=4, d_tp=1, ori_blocks=9, enabled=True, pools=None, group=0):
    tokens = (1000000 + group,) + tuple(range(1, length))
    p, d = pools if pools is not None else (cache(length, 16 // p_tp, enabled, ori_blocks),
                                           cache(length, 16 // d_tp, enabled, ori_blocks))
    warm(p, tokens, p_hit)
    warm(d, tokens, d_hit)
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
            rank_mapping(nodes[0].topology, nodes[1].topology, 0, reservation.partition), entry.normalized_digest)
        destination.validate_plan(entry, plan, pg, p_hit, nodes[0].topology)
        entry.chunk, entry.end = number + 1, end
        plans.append(plan)
    return SimpleNamespace(p=p, d=d, destination=destination, entry=entry, plans=plans, tokens=tokens,
                           source_geometry=pg, nodes=nodes, p_hit=p_hit, source_request=source_request)


def assert_page_mapping(case):
    """Independent per-page oracle for the final physical snapshots."""
    length = len(case.tokens)
    source = case.p.ensure_group_blocks(case.source_request, length, partition=0)
    target = case.entry.reservation.block_ids_by_group
    groups = {g.name: g for g in case.p.group_specs}
    geo = {g.name: g for g in case.source_geometry}
    wanted = set()
    hit = case.entry.reservation.prefix_hit_tokens
    for sr, dr in case.plans[-1].mapping:
        for component in COMPONENTS:
            g = geo[component.name]
            if component.name == "bootstrap_hidden":
                wanted.add((component.name, sr, dr, 0, g.page_bytes))
                continue
            group = groups[component.group]
            page = group.spec.token_capacity
            if component.final:
                stop = math.ceil(length / page)
                indices = range(max(0, stop - len(source[group.name])), stop)
            elif group.sliding_window is not None:
                indices = range(case.plans[-1].valid_from[group.name] // page, math.ceil(length / page))
            else:
                stop = length // page + (length % page >= group.spec.compress_ratio)
                indices = range(hit // page, stop)
            for logical in indices:
                sb = source[group.name][logical % len(source[group.name])
                                        if group.sliding_window is not None else logical]
                db = target[group.name][logical % len(target[group.name])
                                        if group.sliding_window is not None else logical]
                for layer in range(len(g.layers)):
                    wanted.add((component.name, sr, dr, (layer * g.blocks + sb) * g.page_bytes,
                                (layer * g.blocks + db) * g.page_bytes))
    actual = []
    for plan in case.plans:
        for copy in plan.copies:
            size = geo[copy.component].page_bytes
            actual.extend((copy.component, copy.source_rank, copy.destination_rank,
                           copy.source_offset + offset, copy.destination_offset + offset)
                          for offset in range(0, copy.length, size))
    assert len(actual) == len(set(actual)) and set(actual) == wanted


@pytest.mark.parametrize("length", [31, 128, 129, 8192, 8205, 8319, 8320])
@pytest.mark.parametrize("p_tp,d_tp", [(4, 4), (4, 1), (1, 4)])
def test_final_rolling_tail_preserves_decode_and_queryable_checkpoint(length, p_tp, d_tp):
    case = handoff(length, p_tp=p_tp, d_tp=d_tp)
    assert_page_mapping(case)
    final = case.plans[-1]
    assert all(not any(c.component == "ori" for c in plan.copies) for plan in case.plans[:-1])
    boundary = max(0, length - 1) // 128 * 128
    assert final.valid_from == {"ori": max(0, boundary - 128)}
    geometry = {g.name: g for g in case.source_geometry}
    # Expanding the contiguous descriptors must cover every required layer/page exactly once.
    for plan in case.plans:
        occupied = set()
        for copy in plan.copies:
            page = geometry[copy.component].page_bytes
            assert copy.length % page == 0
            for offset in range(0, copy.length, page):
                key = copy.destination_rank, copy.component, copy.destination_offset + offset
                assert key not in occupied
                occupied.add(key)
        assert all(c.component not in ("hca_state", "csa_state", "csa_inner_state", "bootstrap_hidden")
                   for c in plan.copies) or plan.final
    manager = case.d
    manager.authorize_group_cache_write(final.binding.request)
    manager.commit_group_cache(final.binding.request, valid_from=final.valid_from)
    manager.release_group_cache(final.binding.request, confirmed_stopped=True)
    manager.retire_group_cache_reservation(final.binding.request)
    if boundary:
        assert query(manager, list(case.tokens), TARGET) == boundary
    for name, pool in manager._group_pools.items():
        if name == "ori":
            hashes = manager.compute_group_block_hashes(list(case.tokens), group_names=(name,))[name]
            assert all((case.entry.reservation.partition, hashes[i]) not in pool.hash_to_block
                       for i in range(final.valid_from[name] // 32))


@pytest.mark.parametrize("p_hit,d_hit", [(0, 8192), (4096, 8192), (8192, 4096), (4096, 4096)])
def test_independent_prefixes_keep_shared_pages_read_only_and_final_state(p_hit, d_hit):
    case = handoff(8192 + 128, p_hit, d_hit)
    assert_page_mapping(case)
    final = case.plans[-1]
    expected = {"hca_state", "csa_state", "csa_inner_state", "bootstrap_hidden"}
    assert expected <= {copy.component for copy in final.copies}
    case.entry.chunk, case.entry.end = final.chunk, final.start
    for change in ({"ori": 0}, {"ori": final.end}, {"ori": True}, {}):
        if change != final.valid_from:
            with pytest.raises(ValueError, match="rolling publication"):
                case.destination.validate_plan(case.entry, msgspec.structs.replace(final, valid_from=change),
                                               case.source_geometry, case.p_hit, case.nodes[0].topology)


def test_small_ring_never_publishes_a_checkpoint_outside_its_resident_tail():
    case = handoff(8319, ori_blocks=7)
    final = case.plans[-1]
    assert final.valid_from == {"ori": 8096}
    case.d.authorize_group_cache_write(final.binding.request)
    case.d.commit_group_cache(final.binding.request, valid_from=final.valid_from)
    case.d.release_group_cache(final.binding.request, confirmed_stopped=True)
    case.d.retire_group_cache_reservation(final.binding.request)
    assert query(case.d, list(case.tokens), TARGET) == 0


def test_eight_group_growth_reuses_received_cache_without_overwriting_shared_prefix():
    pools = cache(8448, 4, slots=12), cache(8448, 16, slots=12)
    for round_number, length in enumerate((8192, 8320, 8448)):
        for group in range(8):
            case = handoff(length, pools=pools, group=group)
            assert_page_mapping(case)
            reservation = case.entry.reservation
            assert reservation.prefix_hit_tokens == (length - 128 if round_number else 0)
            final = case.plans[-1]
            case.d.authorize_group_cache_write(final.binding.request)
            case.d.commit_group_cache(final.binding.request, valid_from=final.valid_from)
            case.d.release_group_cache(final.binding.request, confirmed_stopped=True)
            case.d.retire_group_cache_reservation(final.binding.request)
            case.p.release_all_group_requests(case.source_request)


def test_prefix_disabled_still_transfers_the_decode_tail_without_native_state_shortcuts():
    case = handoff(8205, enabled=False)
    assert_page_mapping(case)
    assert case.entry.reservation.prefix_hit_tokens == 0
    assert case.entry.prefix.hashes == {}
    assert {"hca_state", "csa_state", "csa_inner_state", "bootstrap_hidden"} <= {
        c.component for c in case.plans[-1].copies}


def test_continuation_has_no_transmitted_prefix_hashes_and_rejects_old_schema():
    value = Continuation(tuple(range(128)), {"max_new_tokens": 512}, None, None)
    wire = msgspec.msgpack.encode(value)
    decoded = msgspec.msgpack.decode(wire)
    assert set(decoded) == {"tokens", "config", "parser", "constraint"}
    with pytest.raises(msgspec.ValidationError, match="unknown field"):
        msgspec.convert({**decoded, "prefix": {}}, type=Continuation)


def test_copy_compaction_preserves_discontinuous_page_mapping():
    from pypto_serving.pd.planner import Geometry
    from pypto_serving.pd.protocol import Component, Contract
    group = KVCacheGroupSpec("kv", (0, 1, 2), KVCacheSpec(32, 192), 8)
    geometry = (Geometry("kv", (0, 1, 2), 32, 32, 2, "uint8"),)
    contract = Contract("test", "model", "layout", (Component("kv", "kv", "kv"),))
    topology = Topology.for_tp(1, 1)
    p, d = (Node(name, role, name, contract, name, topology)
            for name, role in (("p", "prefill"), ("d", "decode")))
    binding = Binding.create(p, d, {})
    source, target = (5, 6, 2, 9), (8, 9, 3, 11)
    chunk = Chunk(binding.request, 0, 0, 128, True, 7, {"kv": source}, 0)
    plan = plan_chunk(binding, chunk, Placement(binding.request, 0, 0, {"kv": target}),
                      contract, (group,), geometry, geometry, 0, ((0, 0),), "continuation")
    pairs = {(copy.source_offset + delta, copy.destination_offset + delta)
             for copy in plan.copies for delta in range(0, copy.length, 64)}
    assert pairs == {((layer * 32 + sb) * 64, (layer * 32 + db) * 64)
                     for layer in range(3) for sb, db in zip(source, target)}
    assert len(plan.copies) == 9  # Only 5->8 and 6->9 can merge within each layer.


def runtime(job, futures):
    result = WorkerRuntime.__new__(WorkerRuntime)
    result.lock, result.jobs, result.uncertain = threading.Lock(), {job: (PLAN, tuple(futures))}, False
    return result


PLAN = Plan(Binding("request", ("p", "p", "p"), ("d", "d", "d"), "input"), "continuation",
            0, 0, 128, False, ((0, 0),), (), ((0, "kv"),))


@pytest.mark.parametrize("failure", [None, False, True])
def test_completion_channel_consumes_only_terminal_jobs_and_keeps_uncertainty(failure):
    future = Future()
    worker = runtime("job", (future,))
    assert worker.poll(("job",)) == [None] and "job" in worker.jobs
    if failure is None:
        future.set_result(None)
    else:
        future.set_exception(TransferError("native failure", uncertain=failure))
    value, = worker.poll(("job",))
    assert not worker.jobs and worker.uncertain is (failure is True)
    if failure is None:
        assert value["completion"] == Completion("job", PLAN.units)
    else:
        assert value["uncertain"] is failure


def test_progress_takes_multiple_native_completions_without_compute_control_commands():
    import multiprocessing
    async def run():
        parent, child = multiprocessing.Pipe()
        futures = [Future(), Future()]
        worker = runtime("one", futures[:1])
        worker.jobs["two"] = PLAN, tuple(futures[1:])
        server = ProgressServer(child, worker.poll)
        progress = Progress(parent, 2)
        try:
            tasks = [asyncio.create_task(progress.wait(job)) for job in ("one", "two")]
            await asyncio.sleep(.01)
            assert len(worker.jobs) == 2
            futures[0].set_result(None)
            futures[1].set_result(None)
            assert [value.plan_digest for value in await asyncio.gather(*tasks)] == ["one", "two"]
            assert not worker.jobs
        finally:
            await progress.close()
            server.close()
    asyncio.run(run())


def test_lost_terminal_reply_cannot_release_source_snapshot_or_owner_registration():
    async def run():
        s, p, d = service("prefill")
        entry = s.register_http(Binding.create(p, d, {}), {})
        future = Future()
        future.set_result(None)
        worker = runtime("job", (future,))
        assert worker.poll(("job",))[0] is not None and not worker.jobs
        # The reply was lost after taking metadata. Its source pages are still owned.
        s.pins.active[entry.binding.request] = ({"kv": (1,)}, 0)
        s.fail(entry, TransferError("lost progress reply", uncertain=True))
        assert s.gated and s.pins.active
        with pytest.raises(RuntimeError, match="active/uncertain"):
            await s.close()
    asyncio.run(run())
