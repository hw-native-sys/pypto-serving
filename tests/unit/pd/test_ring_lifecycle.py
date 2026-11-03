# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unequal-ring handoffs reuse the normal Prefix, allocator and Decode pipeline."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from pypto_serving.model.deepseek_dspark.npu_runner import build_dspark_cache_group_specs
from pypto_serving.pd.adapter import DeepSeekDSpark
from pypto_serving.pd.integration import Chunk
from pypto_serving.pd.planner import SourcePins, plan_chunk, rank_mapping
from pypto_serving.pd.protocol import Binding, Continuation, Node, Placement, Topology
from pypto_serving.pd.service import Service
from pypto_serving.serving.memory.kv_cache import KvCacheManager
from pypto_serving.serving.sched.scheduler import Request, Scheduler, SchedulerConfig
from tests.unit.pd.test_service import geometries
from tests.unit.pd.test_state_handoff import assert_logical_contents
from tests.unit.serving.test_group_block_snapshot import assert_free


def ring_cache(width, partitions, enabled):
    groups = build_dspark_cache_group_specs(43, (0, 0) + (4,) * 21 + (128,) * 20,
        max_seq_len=655360, max_prefill_tokens=width, partitions=partitions)
    cache = KvCacheManager(enable_prefix_cache=enabled)
    cache.init_groups(tuple(replace(g, num_blocks=(g.max_blocks_per_seq if g.sliding_window is None
        else 2 * g.max_blocks_per_seq + 16 * g.sliding_window // g.spec.token_capacity)) for g in groups),
        max_batch_size=8)
    return cache


@pytest.mark.parametrize("destination_width", [8192, 2048])
@pytest.mark.parametrize("d_tp", [1, 4])
@pytest.mark.parametrize("p_prefix,d_prefix", [(True, True), (True, False), (False, True), (False, False)])
def test_eight_groups_replay_independent_prefixes_then_decode_with_two_snapshots(
    destination_width, d_tp, p_prefix, d_prefix,
):
    p, d = ring_cache(8192, 4, p_prefix), ring_cache(destination_width, 16 // d_tp, d_prefix)
    ps = Scheduler(SchedulerConfig(max_num_running_reqs=8, max_num_scheduled_tokens=8192,
        max_prefill_tokens_per_request=8192, long_prefill_token_threshold=8192,
        max_seq_len=655360, speculative_prefix_cache_replay_tokens=128,
        enable_prefix_cache=p_prefix, stop_after_prefill=True), p)
    ds = Scheduler(SchedulerConfig(max_num_running_reqs=8, max_num_scheduled_tokens=destination_width,
        max_seq_len=655360, num_speculative_tokens=7, async_scheduling=True,
        speculative_prefix_cache_replay_tokens=128, enable_prefix_cache=d_prefix), d)
    adapter, pins = DeepSeekDSpark(), SourcePins(p)
    pg, dg = geometries(p), geometries(d)
    pn = Node("p", "prefill", "p-boot", adapter.contract("model", pg), "p-registry", Topology.for_tp(16, 4))
    dn = Node("d", "decode", "d-boot", adapter.contract("model", dg), "d-registry", Topology.for_tp(16, d_tp))
    service = Service(SimpleNamespace(prefill=(0,), decode=(0,), capacity=16),
        SimpleNamespace(node_id="d"), "decode", adapter, SimpleNamespace(kv_cache_manager=d), None)
    service.node, service.geometry = dn, dg
    depth_two = 0
    for turn, length in enumerate((8205, 8205, 8717)):
        requests = []
        for group in range(8):
            tokens = [1000000 + group] + list(range(1, length))
            binding = Binding.create(pn, dn, {})
            rid = binding.request
            prefix = adapter.prefix_spec(tokens, d)
            reservation = d.reserve_group_cache(rid, rid, length + 131072, prompt_token_count=length,
                prefix_block_hashes={k: [bytes.fromhex(h) for h in values] for k, values in prefix.hashes.items()},
                shareable_group_names=tuple(prefix.hashes))
            assert reservation.prefix_hit_tokens == (8192 if turn and d_prefix else 0)
            assert all(not reservation.shared_block_ids_by_group[name]
                       for name in ("hca_state", "csa_state", "csa_inner_state"))
            placement = Placement(rid, reservation.prefix_hit_tokens, reservation.partition,
                                  reservation.block_ids_by_group, {"bootstrap": 1})
            entry = SimpleNamespace(binding=binding, reservation=reservation, pending=None, mapping=(),
                continuation=Continuation(tuple(tokens), {"max_new_tokens": 136}, None, None),
                normalized_digest="continuation", chunk=0, end=0, state_slots={"bootstrap": 1})
            p._next_group_partition = group % 4
            request = Request(rid, tokens, max_new_tokens=512)
            ps.add_request(request)
            p_hit = None
            for number in range(8):
                output = ps.schedule()
                step, = output.scheduled_requests
                if p_hit is None:
                    p_hit = step.num_computed_tokens
                    if turn == 1:
                        assert p_hit == (8064 if p_prefix else 0)
                    elif turn == 2 and p_prefix:
                        assert 8064 <= p_hit <= 8192
                    else:
                        assert p_hit == 0
                end = step.num_computed_tokens + step.num_new_tokens
                final = end == length
                chunk = Chunk(rid, number, step.num_computed_tokens, end, final, 11 if final else None,
                    {k: tuple(v) for k, v in step.block_ids_by_group.items()}, step.cache_partition,
                    {"bootstrap": 0})
                pins.retain(chunk)
                ps.update_from_output(output, {rid: [11]} if final else {})
                plan = plan_chunk(binding, chunk, placement, pn.contract, p.group_specs, pg, dg, p_hit,
                    rank_mapping(pn.topology, dn.topology, chunk.partition, placement.partition),
                    entry.normalized_digest, adapter.state_ranges(p.group_specs, end) if final else {})
                service.validate_plan(entry, plan, pg, p_hit, pn.topology)
                assert_logical_contents(SimpleNamespace(groups=p.group_specs, geometry=pg, service=service,
                    chunk=chunk, placement=placement, p_hit=p_hit), plan)
                if plan.copies or plan.final:
                    d.authorize_group_cache_write(rid)
                pins.release(rid)  # Fake native completion, before the next source chunk.
                entry.chunk, entry.end = number + 1, end
                if final:
                    d.commit_group_cache(rid, valid_from=plan.valid_from)
                    ps.finish_prefilled_request(rid)
                    break
            else:
                pytest.fail("bounded Prefill failed to complete")
            # D hashes KV only; newly received request-local state is never
            # published or mistaken for a reusable shared Prefix page.
            assert all(not d._group_pools[name].hash_to_block
                       for name in ("hca_state", "csa_state", "csa_inner_state"))
            decoded, _ = ds.admit_prefilled(reservation_id=rid, request_id=rid,
                prompt_token_ids=tokens, first_token=11, max_new_tokens=136)
            requests.append(decoded)
        pending = []
        for _ in range(128):
            output = ds.schedule(allow_reject_stalled=not pending)
            assert not output.rejected_requests and not output.preempted_requests
            if output.scheduled_requests:
                ds.advance_after_schedule(output)
                pending.append(output)
            if len(pending) == 2:
                depth_two += 1
            if pending and (len(pending) == 2 or not output.scheduled_requests):
                confirmed = pending.pop(0)
                ds.update_from_output(confirmed, {s.request.request_id: [11] * 8
                                                 for s in confirmed.scheduled_requests})
            if all(r.status.is_finished for r in requests):
                for confirmed in pending:
                    ds.update_from_output(confirmed, {})
                break
        else:
            pytest.fail("bounded external Decode failed to drain")
        assert all(len(r.output_token_ids) == 136 for r in requests)
        for request in requests:
            d.retire_group_cache_reservation(request.request_id)
        assert_free(p)
        assert_free(d)
    assert depth_two > 0 and not pins.active
    # Both deterministic cancellation and UNKNOWN preserve healthy shared KV.
    for uncertain in (False, True):
        rid = f"failed-{uncertain}"
        tokens = [1000000] + list(range(1, 9229))
        prefix = adapter.prefix_spec(tokens, d)
        reservation = d.reserve_group_cache(rid, rid, len(tokens) + 131072,
            prompt_token_count=len(tokens), prefix_block_hashes={k: [bytes.fromhex(h) for h in values]
                for k, values in prefix.hashes.items()}, shareable_group_names=tuple(prefix.hashes))
        if uncertain:
            d.authorize_group_cache_write(rid)
            quarantined = d.quarantine_group_cache(rid)
            for name, pages in reservation.shared_block_ids_by_group.items():
                assert set(pages).isdisjoint(quarantined.quarantined_block_ids_by_group[name])
            with pytest.raises(RuntimeError, match="native writer"):
                d.release_group_cache(rid)
        d.release_group_cache(rid, confirmed_stopped=uncertain)
        d.retire_group_cache_reservation(rid)
        assert_free(d)
    assert not d._reservations._group_reservations
