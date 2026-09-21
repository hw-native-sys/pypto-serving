# Copyright (c) PyPTO Contributors.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""Host-only Phase H Prefix Cache reservation and transfer tests."""

from dataclasses import replace

from pypto_serving.model.deepseek_dspark.pd_adapter import (
    DSV4_DSPARK_K7_ADAPTER,
    DSV4_DSPARK_K7_CONTRACT,
)
from pypto_serving.serving.memory.kv_cache import GroupReservationState
from pypto_serving.serving.pd.connector import DecodeConnector
from pypto_serving.serving.pd.protocol import (
    AbortHandoff,
    ChunkManifest,
    CommitRequest,
    ContinuationMetadata,
    HandoffKey,
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


def _connector():
    manager = make_cache_manager(capacity_slots=3, enable_prefix_cache=True)
    registry = make_registry(manager)
    capabilities = replace(
        make_capabilities(registry),
        prefix_cache_mode="d_only",
    )
    connector = DecodeConnector(
        manager,
        capabilities,
        registry,
        make_rank_registrations(registry),
        contract=DSV4_DSPARK_K7_CONTRACT,
    )
    return manager, registry, connector


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


def _manifest(manager, registry, key, accepted, prompt):
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
        rank_ids=rank_ids,
        source_blocks_by_rank=tables,
        destination_blocks_by_rank=tables,
        destination_prefix_hit_tokens=accepted.prefix_hit_tokens,
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
        expected_units=plan.expected_units,
        copies_by_rank=plan.copies_by_rank,
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
                unit.rank_id,
                unit.component_id,
                f"attempt-{unit.rank_id}-{unit.component_id}",
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
        if unit.rank_id == full_hit_manifest.expected_units[0].rank_id
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
            writable_unit.rank_id,
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
