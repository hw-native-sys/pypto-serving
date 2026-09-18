# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pytest

from pypto_serving.model.deepseek_dspark.pd_adapter import DSV4_DSPARK_K7_CONTRACT
from pypto_serving.serving.memory.kv_cache import GroupReservationState
from pypto_serving.serving.pd.connector import DecodeConnector
from pypto_serving.serving.pd.planner import ChunkTransferPlanner
from pypto_serving.serving.pd.protocol import (
    AbortHandoff,
    ChunkManifest,
    CommitRequest,
    ContinuationMetadata,
    HandoffKey,
    PageCopy,
    ReserveAccepted,
    ReserveRejected,
    ReserveRequest,
    TransferResult,
    continuation_metadata_hash,
)
from pypto_serving.transfer.types import CompletionCertainty

from .helpers import make_cache_manager, make_capabilities, make_rank_registrations, make_registry


KEY = HandoffKey("request", "handoff", 1, 1, 1)


def _connector(capacity_slots: int = 2):
    manager = make_cache_manager(capacity_slots=capacity_slots)
    registry = make_registry(manager)
    connector = DecodeConnector(
        manager,
        make_capabilities(registry),
        registry,
        make_rank_registrations(registry),
        contract=DSV4_DSPARK_K7_CONTRACT,
    )
    return manager, registry, connector


def _reserve(connector: DecodeConnector, prompt: int = 33, maximum: int = 8):
    result = connector.reserve(
        ReserveRequest(KEY, prompt, maximum, connector.registry.layout_fingerprint)
    )
    assert isinstance(result, ReserveAccepted)
    return result


def _final_manifest(manager, registry, reservation):
    rank_ids = tuple(rank.rank_id for rank in reservation.ranks)
    tables = {rank_id: reservation.block_ids_by_group for rank_id in rank_ids}
    plan = ChunkTransferPlanner(
        registry, manager.group_specs, DSV4_DSPARK_K7_CONTRACT
    ).plan_chunk(
        KEY,
        chunk_id=0,
        start_token=0,
        end_token=33,
        final=True,
        rank_ids=rank_ids,
        source_blocks_by_rank=tables,
        destination_blocks_by_rank=tables,
    )
    continuation = ContinuationMetadata(
        prompt_token_ids=tuple(range(33)),
        max_new_tokens=8,
        temperature=0.0,
        top_p=1.0,
        top_k=None,
        seed=None,
        stop_strings=(),
        eos_token_id=None,
    )
    return ChunkManifest(
        key=KEY,
        chunk_id=0,
        start_token=0,
        end_token=33,
        final=True,
        manifest_hash=plan.manifest_hash,
        expected_units=plan.expected_units,
        copies_by_rank=plan.copies_by_rank,
        first_token=7,
        metadata_hash=continuation_metadata_hash(continuation),
        continuation=continuation,
    )


def test_reservation_uses_normal_pools_and_deterministic_abort_releases() -> None:
    manager, _, connector = _connector()
    free_before = {
        name: manager.group_num_free_blocks(name, 0)
        for name in manager.group_names
    }
    accepted = _reserve(connector)
    reservation = manager.group_cache_reservation(accepted.reservation_id)
    assert reservation.state is GroupReservationState.CONSTRUCTING
    assert reservation.write_authorized
    assert tuple(rank.rank_id for rank in accepted.ranks) == (0, 1, 2, 3)
    assert all(
        manager.group_num_free_blocks(name, accepted.partition) < free_before[name]
        for name in manager.group_names
    )
    first_abort = connector.abort(AbortHandoff(KEY, "cancel"), deterministic=True)
    assert manager.group_cache_reservation(accepted.reservation_id) is None
    assert connector.query(KEY).state == "ABORTED"
    assert connector.abort(
        AbortHandoff(KEY, "router-converged-after-p"), deterministic=True
    ) == first_abort
    assert isinstance(
        connector.reserve(
            ReserveRequest(KEY, 33, 8, connector.registry.layout_fingerprint)
        ),
        ReserveRejected,
    )


def test_capacity_failure_is_all_or_nothing() -> None:
    manager, _, connector = _connector(capacity_slots=1)
    free_before = {
        (name, partition): manager.group_num_free_blocks(name, partition)
        for name in manager.group_names
        for partition in range(manager.group_partition_count)
    }
    result = connector.reserve(
        ReserveRequest(KEY, 16384, 1, connector.registry.layout_fingerprint)
    )
    assert isinstance(result, ReserveRejected)
    free_after = {
        (name, partition): manager.group_num_free_blocks(name, partition)
        for name in manager.group_names
        for partition in range(manager.group_partition_count)
    }
    assert free_after == free_before


def test_d_recomputes_manifest_and_commits_only_complete_set() -> None:
    manager, registry, connector = _connector()
    accepted = _reserve(connector)
    manifest = _final_manifest(manager, registry, accepted)
    connector.register_chunk(manifest)
    results = [
        TransferResult(
            KEY,
            0,
            unit.rank_id,
            unit.component_id,
            f"a-{unit.rank_id}-{unit.component_id}",
            CompletionCertainty.COMPLETED.value,
        )
        for unit in manifest.expected_units
    ]
    for result in results[:-1]:
        connector.record_transfer(result)
    with pytest.raises(RuntimeError, match="incomplete"):
        connector.commit(
            CommitRequest(KEY, manifest.manifest_hash, 7, manifest.metadata_hash)
        )
    connector.record_transfer(results[-1])
    ready = connector.commit(
        CommitRequest(KEY, manifest.manifest_hash, 7, manifest.metadata_hash)
    )
    assert ready == connector.replay_ready_ack(KEY)
    assert connector.claim_decode_admission(KEY)
    assert not connector.claim_decode_admission(KEY)
    # A retransmitted commit after admission is a stable fact, not a state
    # regression or a second scheduler admission.
    assert connector.commit(
        CommitRequest(KEY, manifest.manifest_hash, 7, manifest.metadata_hash)
    ).admitted
    with pytest.raises(RuntimeError, match="committed handoff"):
        connector.abort(AbortHandoff(KEY, "late-cancel"), deterministic=True)

    # Scheduler completion releases allocator ownership first; the connector
    # then retains only a bounded terminal tombstone for idempotent queries.
    manager.release_group_cache(ready.reservation_id)
    connector.mark_decode_completed(KEY)
    assert manager.group_cache_reservation(ready.reservation_id) is None
    assert connector.query(KEY).state == "COMPLETED"


def test_invalid_destination_and_unknown_are_fail_closed() -> None:
    manager, registry, connector = _connector()
    accepted = _reserve(connector)
    manifest = _final_manifest(manager, registry, accepted)
    rank = next(iter(manifest.copies_by_rank))
    copies = list(manifest.copies_by_rank[rank])
    copies[0] = PageCopy(
        component_id=copies[0].component_id,
        layer=copies[0].layer,
        source_block=copies[0].source_block,
        destination_block=copies[0].destination_block + 1,
        valid_tokens=copies[0].valid_tokens,
    )
    bad_copies = dict(manifest.copies_by_rank)
    bad_copies[rank] = tuple(copies)
    tampered = ChunkManifest(
        key=manifest.key,
        chunk_id=manifest.chunk_id,
        start_token=manifest.start_token,
        end_token=manifest.end_token,
        final=manifest.final,
        manifest_hash=manifest.manifest_hash,
        expected_units=manifest.expected_units,
        copies_by_rank=bad_copies,
        first_token=manifest.first_token,
        metadata_hash=manifest.metadata_hash,
        continuation=manifest.continuation,
    )
    with pytest.raises(ValueError, match="reservation plan"):
        connector.register_chunk(tampered)

    connector.register_chunk(manifest)
    unit = manifest.expected_units[0]
    connector.record_transfer(
        TransferResult(
            KEY,
            0,
            unit.rank_id,
            unit.component_id,
            "unknown",
            CompletionCertainty.UNKNOWN.value,
        )
    )
    reservation = manager.group_cache_reservation(accepted.reservation_id)
    assert reservation.state is GroupReservationState.QUARANTINED
    connector.abort(AbortHandoff(KEY, "late-cancel"), deterministic=True)
    assert reservation.state is GroupReservationState.QUARANTINED
    with pytest.raises(RuntimeError, match="native writer"):
        manager.release_group_cache(accepted.reservation_id)
