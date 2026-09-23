# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise the production worker's rank/lease selection without a device."""

from concurrent.futures import Future
from types import SimpleNamespace

import msgspec
import pytest

from pypto_serving.model.deepseek_dspark.pd_adapter import DSV4_DSPARK_K7_ADAPTER, DSparkPDWorker
from pypto_serving.serving.pd.protocol import ChunkManifest, HandoffKey, RankMapping
from pypto_serving.serving.pd.worker import PDWorkerRuntime
from pypto_serving.serving.pd.worker_api import InstallPeerRequest, TransferChunkRequest
from pypto_serving.transfer.types import CompletionCertainty, OwnerRef, RegionLease

from .helpers import make_cache_manager, make_registry
from .test_service import _bundle


class _Bridge:
    def __init__(self, rank):
        self.owner = OwnerRef("run", rank, 1, 1, "P")
        self.process_handle = object()
        self.installed = []

    def install_destinations(self, entries):
        self.installed.append(entries)


class _Agent:
    def __init__(self, owner, provider, **_kwargs):
        self.owner = owner
        self.tasks = []
        self.futures = []

    def submit(self, task, fence, *, timeout):
        assert fence.wait(timeout)
        self.tasks.append(task)
        future = Future()
        self.futures.append(future)
        return future


@pytest.fixture
def worker(monkeypatch):
    manager = make_cache_manager()
    registry = make_registry(manager)
    bundle = _bundle(manager)
    model = SimpleNamespace(
        ranks=16,
        component_ids=tuple(component.component_id for component in registry.components),
        registry=lambda rank, **kwargs: registry,
        copies=DSparkPDWorker.copies,
        rank_mapping=DSV4_DSPARK_K7_ADAPTER.rank_mapping,
    )
    config = SimpleNamespace(run_id="run", model_revision=registry.model_revision, max_active_handoffs=4)
    runtime = PDWorkerRuntime(config, model)
    runtime._pd_owner_bridges = tuple(_Bridge(rank) for rank in range(16))
    runtime._pd_owner_leases = tuple({
        component.component_id: RegionLease(bridge.owner, component.component_id, 1, component.extent)
        for component in registry.components
    } for bridge in runtime._pd_owner_bridges)
    monkeypatch.setattr(runtime, "_prepare_pd_registry", lambda: bundle)
    monkeypatch.setattr("pypto_serving.transfer.agent.TransferAgent", _Agent)
    request = InstallPeerRequest(registry.fingerprint, registry.layout_fingerprint, bundle.ranks)
    runtime._install_pd_peer(request)
    runtime._install_pd_peer(request)
    return runtime, manager, registry


def _request(manager, registry, source_partition, destination_partition):
    source = manager.ensure_group_blocks("P", 128, partition=source_partition)
    destination = manager.ensure_group_blocks("D", 128, partition=destination_partition)
    mapping = DSV4_DSPARK_K7_ADAPTER.rank_mapping((16, 4), source_partition, destination_partition)
    key = HandoffKey("request", "handoff", 1, 1, 1)
    plan = DSV4_DSPARK_K7_ADAPTER.make_planner(registry, manager.group_specs).plan_chunk(
        key, chunk_id=0, start_token=0, end_token=128, final=False, rank_mapping=mapping,
        source_blocks_by_rank={pair.source_rank_id: source for pair in mapping},
        destination_blocks_by_rank={pair.destination_rank_id: destination for pair in mapping},
    )
    manifest = ChunkManifest(
        key=key, chunk_id=0, start_token=0, end_token=128, final=False,
        rank_mapping=mapping, manifest_hash=plan.manifest_hash,
        expected_units=plan.expected_units, copies_by_destination_rank=plan.copies_by_destination_rank,
    )
    return TransferChunkRequest(manifest, registry.fingerprint)


@pytest.mark.parametrize("source_partition,destination_partition", ((0, 0), (1, 2), (3, 0), (0, 3)))
def test_worker_submits_all_actual_source_owners_before_any_completion(worker, source_partition, destination_partition):
    runtime, manager, registry = worker
    request = _request(manager, registry, source_partition, destination_partition)
    submission = runtime._submit_pd_chunk(request)
    assert not runtime._poll_pd_chunk(submission.job_id).complete
    active = tuple(rank for rank, agent in enumerate(runtime._pd_transfer_agents) if agent.tasks)
    assert active == tuple(range(source_partition * 4, source_partition * 4 + 4))
    for rank in active:
        agent = runtime._pd_transfer_agents[rank]
        task = agent.tasks[0]
        assert task.attempt.source.rank_id == rank
        assert task.attempt.destination.rank_id == destination_partition * 4 + rank % 4
        assert all(segment.source.owner == task.attempt.source for segment in task.segments)
        assert all(segment.destination.owner == task.attempt.destination for segment in task.segments)
    # The replay cannot enqueue duplicate native work while the original is live.
    assert runtime._submit_pd_chunk(request) == submission
    assert sum(len(agent.tasks) for agent in runtime._pd_transfer_agents) == 4
    for rank in active:
        runtime._pd_transfer_agents[rank].futures[0].set_result(
            SimpleNamespace(certainty=CompletionCertainty.COMPLETED, error=None)
        )
    result = runtime._poll_pd_chunk(submission.job_id)
    assert result.complete
    assert {(item.source_rank_id, item.destination_rank_id) for item in result.results} == {
        (pair.source_rank_id, pair.destination_rank_id) for pair in request.manifest.rank_mapping
    }


def test_peer_registration_is_batched_and_reused_per_source_owner(worker):
    runtime, _, _ = worker
    for rank, bridge in enumerate(runtime._pd_owner_bridges):
        assert len(bridge.installed) == 1
        entries = bridge.installed[0]
        assert len(entries) == 4 * 8
        assert {lease.owner.rank_id for lease, _ in entries} == set(range(rank % 4, 16, 4))
        assert len({lease.registration_key for lease, _ in entries}) == len(entries)


def test_unregistered_rank_mapping_never_starts_a_write(worker):
    runtime, manager, registry = worker
    request = _request(manager, registry, 1, 2)
    mapping = (RankMapping(5, 8), RankMapping(4, 9), RankMapping(6, 10), RankMapping(7, 11))
    request = msgspec.structs.replace(request, manifest=msgspec.structs.replace(request.manifest, rank_mapping=mapping))
    with pytest.raises(ValueError, match="not installed"):
        runtime._submit_pd_chunk(request)
    assert not any(agent.tasks for agent in runtime._pd_transfer_agents)


def test_invalid_last_destination_is_rejected_before_any_rank_is_submitted(worker):
    runtime, manager, registry = worker
    request = _request(manager, registry, 1, 2)
    units = list(request.manifest.expected_units)
    units[-1] = msgspec.structs.replace(units[-1], nbytes=units[-1].nbytes + 64)
    request = msgspec.structs.replace(request, manifest=msgspec.structs.replace(
        request.manifest, expected_units=tuple(units),
    ))
    with pytest.raises(ValueError, match="byte accounting"):
        runtime._submit_pd_chunk(request)
    assert not any(agent.tasks for agent in runtime._pd_transfer_agents)
