# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Real scheduler, allocator, IPC and page-copy ownership regressions."""

import pytest

from pypto_serving.serving.memory.protocol import CachePublicationProof
from pypto_serving.model.deepseek_lowlat.execution_policy import (
    LowlatRequestContext as RequestExecutionContext,
)
from pypto_serving.model.deepseek_lowlat.execution_policy import (
    LowlatPrefillMetadata,
    LowlatRequestMetadata,
    decode_request_metadata,
)
from pypto_serving.model.deepseek_lowlat.scheduling import LowlatSchedulingPolicy
import msgspec
from pypto_serving.model.deepseek_lowlat.capacity import LowlatCapacityPlan
from pypto_serving.model.deepseek_lowlat.prefix import checkpoint_proof
from pypto_serving.serving.memory.kv_cache import KvCacheManager
from pypto_serving.serving.sched.scheduler import Scheduler, SchedulerConfig, Request
from pypto_serving.serving.server.ipc import (
    NewRequestData,
    PrefillRequest,
    StepCommand,
    StepResult,
    encode_command,
    decode_command,
    encode_result,
    decode_result,
)


def setup(speculative=5, asynchronous=True):
    groups = LowlatCapacityPlan(16384, 512, "rolling", True).cache_specs()
    manager = KvCacheManager(enable_prefix_cache=True)
    manager.init_groups(groups, max_batch_size=1)
    scheduler = Scheduler(
        SchedulerConfig(
            max_num_running_reqs=1,
            max_num_scheduled_tokens=512,
            long_prefill_token_threshold=512,
            max_prefill_tokens_per_request=512,
            max_seq_len=16384,
            num_speculative_tokens=speculative,
            speculative_prefix_cache_replay_tokens=128,
            enable_prefix_cache=True,
            async_scheduling=asynchronous,
            execution_policy=LowlatSchedulingPolicy("model-A"),
        ),
        manager,
    )
    return manager, scheduler


def finish_prompt(manager, scheduler, rid, tokens):
    request = Request(rid, list(tokens), max_new_tokens=1, temperature=0.0)
    scheduler.add_request(request)
    starts = []
    maps = []
    for _ in range(100):
        output = scheduler.schedule()
        assert len(output.scheduled_requests) == 1
        step = output.scheduled_requests[0]
        starts.append(step.num_computed_tokens)
        maps.append(step.block_ids_by_group)
        if scheduler.config.async_scheduling:
            scheduler.advance_after_schedule(output)
            # Confirmed prefix prefill cannot enqueue a successor before restore/copy settles.
            assert not scheduler.schedule().scheduled_requests
        if getattr(step.execution_plan, "checkpoint", None):
            plan = getattr(step.execution_plan, "checkpoint", None)
            proof = checkpoint_proof(plan)
            manager.publish_group_page_proof(
                proof,
                request_id=rid,
                generation=request.execution_generation,
                mapping_version=1,
                snapshot=plan.destination_tables,
                partition=0,
                block_hashes=request.group_block_hashes,
            )
            manager.release_prefix_checkpoint(rid)
            request.execution_state.checkpoint_complete = True
        end = step.num_computed_tokens + step.num_new_tokens
        scheduler.update_from_output(output, {rid: [19]} if end == len(tokens) else {})
        if request.status.is_finished:
            break
    else:
        pytest.fail("Prefix request made no bounded progress")
    assert all(table == maps[0] for table in maps)
    assert all(block.ref_cnt == 0 for pool in manager._group_pools.values() for block in pool.blocks)
    return request, starts


@pytest.mark.parametrize("speculative,asynchronous", [(5, True)])
@pytest.mark.parametrize("length", [127, 129, 513, 1536, 8192])
def test_cold_hot_checkpoint_restore_with_stable_exclusive_rings(speculative, asynchronous, length):
    manager, scheduler = setup(speculative, asynchronous)
    tokens = list(range(length))
    cold, starts = finish_prompt(manager, scheduler, "cold", tokens)
    boundary = max(0, length - 128) // 128 * 128
    assert starts[0] == 0
    if boundary:
        assert boundary in [s for s in starts[1:]]
    hot, hot_starts = finish_prompt(manager, scheduler, "hot", tokens)
    assert hot_starts[0] == boundary
    if boundary:
        assert hot.execution_state.prefix_resume.hit_tokens == boundary
        assert length - boundary >= 128
        source = {p.page_id for p in hot.execution_state.prefix_resume.pages if p.group == "hca_state"}
        assert not source.intersection(hot.allocated_group_block_ids.get("hca_state", ()))


def test_resume_and_completion_proofs_survive_real_ipc_roundtrip():
    manager, scheduler = setup()
    tokens = list(range(1536))
    finish_prompt(manager, scheduler, "cold", tokens)
    request = Request("hot", tokens, max_new_tokens=64, temperature=0.0)
    scheduler.add_request(request)
    output = scheduler.schedule()
    step = output.scheduled_requests[0]
    new = NewRequestData(
        "hot",
        tokens,
        0.0,
        1.0,
        None,
        max_output_tokens=64,
        execution_generation=request.execution_generation,
        model_extension=msgspec.msgpack.encode(
            LowlatRequestMetadata(request.execution_state.prefix_resume)
        ),
    )
    prefill = PrefillRequest(
        "hot",
        tokens[step.num_computed_tokens :],
        step.num_computed_tokens,
        [],
        step.block_ids_by_group,
        0,
        msgspec.msgpack.encode(
            LowlatPrefillMetadata(manager.group_page_identities(step.block_ids_by_group, 0))
        ),
    )
    cmd = StepCommand([new], [prefill], [], [])
    recovered = decode_command(encode_command(cmd))
    assert recovered == cmd
    descriptor = decode_request_metadata(recovered.new_requests[0].model_extension).prefix_resume
    assert descriptor == request.execution_state.prefix_resume
    context = RequestExecutionContext(
        "hot", 1536, 64, execution_generation=request.execution_generation, prefix_resume=descriptor
    )
    assert context.prefix_resume.hit_tokens == 1408
    # Proof types are preserved by msgspec, rather than accepted as opaque dicts.
    plan = manager.reserve_prefix_checkpoint(
        "hot", generation=request.execution_generation, boundary=1408, snapshot=step.block_ids_by_group
    )
    proof = checkpoint_proof(plan)
    result = decode_result(encode_result(StepResult({}, cache_publications=(proof,))))
    assert isinstance(result.cache_publications[0], CachePublicationProof)
    assert result.cache_publications == (proof,)
    scheduler.abort_request("hot")
    scheduler.discard_scheduled_request(step)
    assert all(b.ref_cnt == 0 for p in manager._group_pools.values() for b in p.blocks)


def test_aborted_restore_releases_pins_and_checkpoint_reservations():
    manager, scheduler = setup()
    tokens = list(range(1536))
    finish_prompt(manager, scheduler, "cold", tokens)
    request = Request("abort", tokens, max_new_tokens=64, temperature=0.0)
    scheduler.add_request(request)
    output = scheduler.schedule()
    assert manager._prefix_resume_pins
    scheduler.abort_request("abort")
    assert not manager._prefix_resume_pins
    assert any(b.ref_cnt > 0 for p in manager._group_pools.values() for b in p.blocks)
    scheduler.discard_scheduled_request(output.scheduled_requests[0])
    assert all(b.ref_cnt == 0 for p in manager._group_pools.values() for b in p.blocks)
    _, starts = finish_prompt(manager, scheduler, "next", tokens)
    assert starts[0] == 1408


def test_registered_page_copies_preserve_logical_tail_across_permuted_ring_ids():
    import ctypes
    from contextlib import contextmanager
    from math import prod
    from types import SimpleNamespace
    import torch

    from pypto_serving.serving.memory.protocol import CachePageIdentity
    from pypto_serving.model.deepseek_lowlat.cache import LowlatCachePool, TABLE_GROUPS, TENSOR_GROUPS
    from pypto_serving.model.deepseek_lowlat.kernel_loader import isolated_profile
    from pypto_serving.model.deepseek_lowlat.prefix import PrefixPageCopies

    class Worker:
        def __init__(self):
            self.memory = {}
            self.calls = []
            self.allocations = set()

        def copy_from(self, destination, source, nbytes, *, src_offset=0, worker_id):
            assert source in self.allocations, "copy_from requires the allocation base"
            ctypes.memset(destination, self.memory[source + src_offset, nbytes], nbytes)
            self.calls.append(("read", worker_id, nbytes))

        def copy_to(self, destination, source, nbytes, *, dst_offset=0, worker_id):
            assert destination in self.allocations, "copy_to requires the allocation base"
            data = ctypes.string_at(source, nbytes)
            assert data == bytes([data[0]]) * nbytes
            self.memory[destination + dst_offset, nbytes] = data[0]
            self.calls.append(("write", worker_id, nbytes))

    @contextmanager
    def context():
        with isolated_profile():
            yield

    rows = {"params": [dict(name=n, shape=[8, 1, 4096], dtype="int32", direction="In") for n in TABLE_GROUPS]}
    programs = SimpleNamespace(single_metadata=rows, context=context)
    worker = Worker()
    specs = LowlatCapacityPlan(16384, 512, "rolling", True).cache_specs()
    cache = LowlatCachePool(programs, SimpleNamespace(worker=worker), specs)
    cache.preserve_contents = True
    source, destination = {}, {}
    for name, spec in cache.specs.items():
        count = spec.max_blocks_per_seq if spec.sliding_window else 12
        source[name] = tuple(reversed(range(count)))
        destination[name] = tuple(range(count, 2 * count)) if spec.sliding_window else source[name]
    with context():
        from models.deepseek_v4_flash_lowlat.prefill_cache import CACHE_LAYOUTS

        layouts = dict(CACHE_LAYOUTS)
    ranges = cache.checkpoint_ranges(1408)
    expected = {}
    for tensor_index, (tensor, (layers, rows, _, tail, dtype)) in enumerate(layouts.items()):
        group = TENSOR_GROUPS[tensor]
        spec = cache.specs[group]
        shards = [
            SimpleNamespace(data_ptr=(rank + 1) * 10**14 + (tensor_index + 1) * 10**10, byte_offset=64)
            for rank in range(8)
        ]
        worker.allocations.update(shard.data_ptr - shard.byte_offset for shard in shards)
        cache.owners[tensor] = SimpleNamespace(shards=shards)
        if not spec.sliding_window:
            continue
        interval = next(r for r in ranges if r.group == group)
        nbytes = rows * prod(tail) * torch.empty((), dtype=dtype).element_size()
        for rank, shard in enumerate(shards):
            for layer in range(layers):
                for logical in range(interval.start, interval.end):
                    value = (logical * 13 + layer * 7 + rank * 5 + tensor_index) % 251 + 1
                    old = source[group][logical % len(source[group])]
                    new = destination[group][logical % len(destination[group])]
                    base = shard.data_ptr + layer * spec.num_blocks * nbytes
                    worker.memory[base + old * nbytes, nbytes] = value
                    expected[base + new * nbytes, nbytes] = value
    copier = PrefixPageCopies()
    copier.copy(cache, source=source, destination=destination, ranges=ranges)
    assert all(worker.memory[key] == value for key, value in expected.items())
    assert copier.bytes_copied == 17842176 * 8
    assert sum(kind == "read" for kind, _, _ in worker.calls) == sum(
        kind == "write" for kind, _, _ in worker.calls
    )
    # Retained contents survive request release; a new allocation epoch invalidates them.
    identity = CachePageIdentity("ori", 0, destination["ori"][10 % len(destination["ori"])], 1)
    cache.page_epochs[identity.group, identity.page_id] = 1
    cache.adopt_page_identities((identity,), retained=True)
    cache.release()
    cache.adopt_page_identities((identity,), retained=True)
    cache.adopt_page_identities((replace_identity(identity, allocation_epoch=2),))
    with pytest.raises(ValueError, match="content/epoch"):
        cache.adopt_page_identities((identity,), retained=True)


def replace_identity(identity, **fields):
    from dataclasses import replace

    return replace(identity, **fields)


@pytest.mark.parametrize("asynchronous", [True])
def test_checkpoint_destination_lives_until_whole_step_ack_after_abort(asynchronous):
    manager, scheduler = setup(5, asynchronous)
    request = Request("cold", list(range(1536)), max_new_tokens=64, temperature=0.0)
    scheduler.add_request(request)
    for _ in range(3):
        output = scheduler.schedule()
        step = output.scheduled_requests[0]
        if asynchronous:
            scheduler.advance_after_schedule(output)
        if getattr(step.execution_plan, "checkpoint", None) is not None:
            break
        scheduler.update_from_output(output, {})
    assert getattr(step.execution_plan, "checkpoint", None) is not None
    plan = getattr(step.execution_plan, "checkpoint", None)
    scheduler.abort_request("cold")
    for page in plan.destination_pages:
        assert manager._group_pools[page.group].block_from_local_id(0, page.page_id).ref_cnt > 0
    assert all(not p.hash_to_block for p in manager._group_pools.values())
    scheduler.update_from_output(output, {})
    assert all(b.ref_cnt == 0 for p in manager._group_pools.values() for b in p.blocks)
    assert all(not p.hash_to_block for p in manager._group_pools.values())
