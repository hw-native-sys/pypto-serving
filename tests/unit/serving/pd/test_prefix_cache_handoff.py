# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prefix-cache reservation, suffix transfer, and shared-page ownership tests."""

from dataclasses import replace

import pytest

from pypto_serving.model.deepseek_dspark.pd_adapter import (
    DSV4_DSPARK_K7_ADAPTER,
    DSV4_DSPARK_K7_CONTRACT,
)
from pypto_serving.serving.pd.integration import PrefillChunkReady
from pypto_serving.serving.memory.reservation import GroupReservationState
from pypto_serving.serving.pd.connector import DecodeConnector
from pypto_serving.serving.pd.protocol import (
    AbortHandoff,
    ChunkManifest,
    CommitRequest,
    ContinuationMetadata,
    HandoffKey,
    RankMapping,
    ReserveAccepted,
    ReserveRejected,
    ReserveRequest,
    TransferResult,
    continuation_metadata_hash,
)
from pypto_serving.transfer.types import CompletionCertainty

from .helpers import (
    make_cache_manager,
    make_capabilities,
    make_rank_registrations,
    make_registry,
)


def _connector(prefix_cache_mode: str = "d_only"):
    manager = make_cache_manager(capacity_slots=3, enable_prefix_cache=True)
    registry = make_registry(manager)
    capabilities = replace(
        make_capabilities(registry),
        prefix_cache_mode=prefix_cache_mode,
    )
    connector = DecodeConnector(
        manager,
        capabilities,
        registry,
        make_rank_registrations(registry),
        contract=DSV4_DSPARK_K7_CONTRACT,
        rank_mapping=DSV4_DSPARK_K7_ADAPTER.rank_mapping,
    )
    return manager, registry, connector


def _warm_prefix(manager, prompt, token_count: int, request_id: str) -> None:
    hashes = manager.compute_group_block_hashes(prompt)
    manager.ensure_group_blocks(request_id, token_count, partition=0)
    manager.cache_group_blocks(request_id, hashes, token_count, {})
    manager.release_all_group_requests(request_id)


def _spec(manager, prompt):
    return DSV4_DSPARK_K7_ADAPTER.build_prefix_match_spec(prompt, manager)


def _reserve(connector, key, prompt, spec):
    accepted = connector.reserve(
        ReserveRequest(
            key=key,
            prompt_token_count=len(prompt),
            max_new_tokens=8,
            layout_fingerprint=connector.registry.layout_fingerprint,
            prepared_digest="p" * 64,
            prefix_match_spec=spec,
        )
    )
    assert isinstance(accepted, ReserveAccepted)
    return accepted


def _manifest(
    manager,
    registry,
    key,
    accepted,
    prompt,
    *,
    source_prefix_hit_tokens: int = 0,
):
    rank_ids = tuple(rank.rank_id for rank in accepted.ranks)
    tables = {rank_id: accepted.block_ids_by_group for rank_id in rank_ids}
    plan = DSV4_DSPARK_K7_ADAPTER.make_planner(
        registry, manager.group_specs
    ).plan_chunk(
        key,
        chunk_id=0,
        start_token=0,
        end_token=len(prompt),
        final=True,
        rank_mapping=tuple(RankMapping(rank, rank) for rank in rank_ids),
        source_blocks_by_rank=tables,
        destination_blocks_by_rank=tables,
        destination_prefix_hit_tokens=accepted.prefix_hit_tokens,
        source_prefix_hit_tokens=source_prefix_hit_tokens,
    )
    continuation = ContinuationMetadata(
        prompt_token_ids=tuple(prompt),
        max_new_tokens=8,
        temperature=0.0,
        top_p=1.0,
        top_k=None,
        seed=None,
        stop_strings=(),
        eos_token_id=None,
    )
    return ChunkManifest(
        key=key,
        chunk_id=0,
        start_token=0,
        end_token=len(prompt),
        final=True,
        manifest_hash=plan.manifest_hash,
        rank_mapping=plan.rank_mapping,
        expected_units=plan.expected_units,
        copies_by_destination_rank=plan.copies_by_destination_rank,
        source_prefix_hit_tokens=source_prefix_hit_tokens,
        first_token=7,
        metadata_hash=continuation_metadata_hash(continuation),
        continuation=continuation,
        prepared_digest="p" * 64,
    )


def _complete(connector, manifest):
    connector.register_chunk(manifest)
    for unit in manifest.expected_units:
        connector.record_transfer(
            TransferResult(
                manifest.key,
                manifest.chunk_id,
                next(pair.source_rank_id for pair in manifest.rank_mapping
                     if pair.destination_rank_id == unit.destination_rank_id),
                unit.destination_rank_id,
                unit.component_id,
                f"attempt-{unit.destination_rank_id}-{unit.component_id}",
                CompletionCertainty.COMPLETED.value,
            )
        )
    return connector.commit(
        CommitRequest(
            manifest.key,
            manifest.manifest_hash,
            manifest.first_token,
            manifest.metadata_hash,
        )
    )


def test_commit_publishes_suffix_and_next_reservation_reuses_prefix() -> None:
    manager, registry, connector = _connector()
    prompt = list(range(256))
    spec = _spec(manager, prompt)

    cold_key = HandoffKey("cold", "cold-handoff", 1, 1, 1)
    cold = _reserve(connector, cold_key, prompt, spec)
    assert cold.prefix_hit_tokens == 0
    cold_manifest = _manifest(manager, registry, cold_key, cold, prompt)
    cold_ready = _complete(connector, cold_manifest)
    assert connector.claim_decode_admission(cold_key)
    manager.release_group_cache(cold_ready.reservation_id)
    connector.mark_decode_completed(cold_key)

    hit_key = HandoffKey("hit", "hit-handoff", 1, 1, 1)
    hit = _reserve(connector, hit_key, prompt, spec)
    assert hit.prefix_hit_tokens == 256
    reservation = manager.group_cache_reservation(hit.reservation_id)
    assert reservation is not None
    assert any(reservation.shared_block_ids_by_group.values())
    assert all(
        set(reservation.shared_block_ids_by_group[name]).isdisjoint(
            reservation.writable_block_ids_by_group[name]
        )
        for name in manager.group_names
    )

    full_hit_manifest = _manifest(manager, registry, hit_key, hit, prompt)
    non_final_components = {
        component.component_id
        for component in DSV4_DSPARK_K7_CONTRACT.components
        if not component.final_only
    }
    final_components = {
        component.component_id
        for component in DSV4_DSPARK_K7_CONTRACT.components
        if component.final_only
    }
    bytes_by_component = {
        unit.component_id: unit.nbytes
        for unit in full_hit_manifest.expected_units
        if unit.destination_rank_id == full_hit_manifest.expected_units[0].destination_rank_id
    }
    assert all(bytes_by_component[name] == 0 for name in non_final_components)
    assert all(bytes_by_component[name] > 0 for name in final_components)


def test_unknown_quarantines_writable_suffix_but_not_shared_prefix() -> None:
    manager, registry, connector = _connector()
    prompt = list(range(256))
    spec = _spec(manager, prompt)

    warm_key = HandoffKey("warm", "warm-handoff", 1, 1, 1)
    warm = _reserve(connector, warm_key, prompt, spec)
    ready = _complete(connector, _manifest(manager, registry, warm_key, warm, prompt))
    assert connector.claim_decode_admission(warm_key)
    manager.release_group_cache(ready.reservation_id)
    connector.mark_decode_completed(warm_key)

    key = HandoffKey("unknown", "unknown-handoff", 1, 1, 1)
    accepted = _reserve(connector, key, prompt, spec)
    manifest = _manifest(manager, registry, key, accepted, prompt)
    connector.register_chunk(manifest)
    writable_unit = next(unit for unit in manifest.expected_units if unit.nbytes > 0)
    connector.record_transfer(
        TransferResult(
            key,
            0,
            writable_unit.destination_rank_id,
            writable_unit.destination_rank_id,
            writable_unit.component_id,
            "unknown-attempt",
            CompletionCertainty.UNKNOWN.value,
        )
    )
    reservation = manager.group_cache_reservation(accepted.reservation_id)
    assert reservation is not None
    assert reservation.state is GroupReservationState.QUARANTINED
    assert (
        reservation.quarantined_block_ids_by_group
        == reservation.writable_block_ids_by_group
    )
    assert all(
        set(reservation.quarantined_block_ids_by_group[name]).isdisjoint(
            reservation.shared_block_ids_by_group[name]
        )
        for name in manager.group_names
    )
    connector.abort(AbortHandoff(key, "owner-recovery"), deterministic=False)


def test_prefix_reservation_failure_rolls_back_shared_refs_without_cache_loss() -> None:
    manager, registry, connector = _connector()
    prompt = list(range(256))
    spec = _spec(manager, prompt)

    warm_key = HandoffKey("rollback-warm", "rollback-warm-handoff", 1, 1, 1)
    warm = _reserve(connector, warm_key, prompt, spec)
    ready = _complete(connector, _manifest(manager, registry, warm_key, warm, prompt))
    assert connector.claim_decode_admission(warm_key)
    manager.release_group_cache(ready.reservation_id)
    connector.mark_decode_completed(warm_key)

    failed_key = HandoffKey("rollback-fail", "rollback-fail-handoff", 1, 1, 1)
    rejected = connector.reserve(
        ReserveRequest(
            key=failed_key,
            prompt_token_count=len(prompt),
            max_new_tokens=2_000_000,
            layout_fingerprint=registry.layout_fingerprint,
            prepared_digest="f" * 64,
            prefix_match_spec=spec,
        )
    )
    assert isinstance(rejected, ReserveRejected)
    assert manager.group_request_partition(failed_key.request_id) is None

    retry_key = HandoffKey("rollback-retry", "rollback-retry-handoff", 1, 1, 1)
    retry = _reserve(connector, retry_key, prompt, spec)
    assert retry.prefix_hit_tokens == 256
    aborted = connector.abort(
        AbortHandoff(retry_key, "deterministic-cancel"),
        deterministic=True,
    )
    assert aborted.state == "ABORTED"
    assert manager.group_cache_reservation(retry.reservation_id) is None


def test_p_hit_greater_than_d_hit_backfills_full_history_and_commits() -> None:
    d_manager, registry, connector = _connector("independent")
    prompt = list(range(640))
    _warm_prefix(d_manager, prompt, 128, "d-warm")
    spec = _spec(d_manager, prompt)
    key = HandoffKey("independent", "independent-handoff", 1, 1, 1)
    accepted = _reserve(connector, key, prompt, spec)
    assert accepted.prefix_hit_tokens == 128

    p_manager = make_cache_manager(capacity_slots=2, enable_prefix_cache=True)
    _warm_prefix(p_manager, prompt, 512, "p-warm")
    p_hashes = p_manager.compute_group_block_hashes(prompt)
    _, p_hit, p_partition = p_manager.acquire_group_prefix_blocks(
        "p-source",
        p_hashes,
        max_cache_hit_tokens=512,
    )
    assert p_hit == 512
    assert p_partition == 0
    p_tables = {
        name: tuple(ids)
        for name, ids in p_manager.ensure_group_blocks(
            "p-source",
            len(prompt),
            partition=p_partition,
        ).items()
    }
    rank_ids = tuple(rank.rank_id for rank in accepted.ranks)
    plan = DSV4_DSPARK_K7_ADAPTER.make_planner(
        registry,
        d_manager.group_specs,
    ).plan_chunk(
        key,
        chunk_id=0,
        start_token=0,
        end_token=len(prompt),
        final=True,
        rank_mapping=tuple(RankMapping(rank, rank) for rank in rank_ids),
        source_blocks_by_rank={rank_id: p_tables for rank_id in rank_ids},
        destination_blocks_by_rank={
            rank_id: accepted.block_ids_by_group for rank_id in rank_ids
        },
        destination_prefix_hit_tokens=accepted.prefix_hit_tokens,
        source_prefix_hit_tokens=p_hit,
    )
    continuation = ContinuationMetadata(
        prompt_token_ids=tuple(prompt),
        max_new_tokens=8,
        temperature=0.0,
        top_p=1.0,
        top_k=None,
        seed=None,
        stop_strings=(),
        eos_token_id=None,
    )
    manifest = DSV4_DSPARK_K7_ADAPTER.build_manifest(
        key=key,
        plan=plan,
        chunk=PrefillChunkReady(
            request_id=key.request_id,
            chunk_id=0,
            start_token=p_hit,
            end_token=len(prompt),
            final=True,
            first_token=7,
            block_ids_by_group=p_tables,
            cache_partition=p_partition,
        ),
        continuation=continuation,
        prepared_digest="p" * 64,
    )
    assert manifest.start_token == 0
    assert manifest.source_prefix_hit_tokens == 512
    ready = _complete(connector, manifest)
    committed = d_manager.group_cache_reservation(ready.reservation_id)
    assert committed is not None
    assert committed.publication_valid_from == {"ori": 384}
    assert connector.claim_decode_admission(key)
    d_manager.release_group_cache(ready.reservation_id)
    connector.mark_decode_completed(key)

    # The untouched rolling gap (D_hit..P_hit-window) must not be published.
    # A shorter request may still reuse D's original prefix, but cannot falsely
    # match pages that P never transferred.
    short_prompt = prompt[:256]
    short_key = HandoffKey("short", "short-handoff", 1, 1, 1)
    short = _reserve(connector, short_key, short_prompt, _spec(d_manager, short_prompt))
    assert short.prefix_hit_tokens == 128
    connector.abort(AbortHandoff(short_key, "test-cleanup"), deterministic=True)

    # The final rolling tail and every full-history page are valid, so the
    # complete prompt remains reusable even though the older rolling gap is not.
    next_key = HandoffKey("next", "next-handoff", 1, 1, 1)
    next_reservation = _reserve(connector, next_key, prompt, spec)
    assert next_reservation.prefix_hit_tokens == len(prompt)


def test_p_hit_is_rejected_outside_independent_mode() -> None:
    manager, registry, connector = _connector("d_only")
    prompt = list(range(256))
    key = HandoffKey("mode", "mode-handoff", 1, 1, 1)
    accepted = _reserve(connector, key, prompt, _spec(manager, prompt))
    manifest = _manifest(
        manager,
        registry,
        key,
        accepted,
        prompt,
        source_prefix_hit_tokens=128,
    )

    with pytest.raises(ValueError, match="independent cache mode"):
        connector.register_chunk(manifest)
