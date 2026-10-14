# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pytest
import msgspec
import asyncio
from types import SimpleNamespace

from pypto_serving.serving.pd.completion import CompletionState, CompletionTracker
from pypto_serving.model.deepseek_dspark.pd_adapter import DSV4_DSPARK_K7_CONTRACT
from pypto_serving.serving.pd.protocol import (
    ChunkManifest,
    CommitRequest,
    ContinuationMetadata,
    HandoffKey,
    RankMapping,
    TransferResult,
    TransferResultBundle,
    TransferUnit,
    chunk_payload_hash,
    continuation_metadata_hash,
)
from pypto_serving.transfer.types import CompletionCertainty


KEY = HandoffKey("request", "handoff", 1, 2, 3)


@pytest.mark.parametrize("failure", ["none", "fsync", "ack"])
def test_service_completion_is_durable_before_ack_and_commit(failure, tmp_path, monkeypatch):
    from .test_journal import _config
    from pypto_serving.serving.pd.journal import DurablePDJournal
    from pypto_serving.serving.pd.protocol import OpenRoute, ControlAck, ReadyAck
    from pypto_serving.serving.pd.service import PDServingService

    async def exercise():
        tracker = CompletionTracker(KEY, "reservation")
        tracker.register_chunk(_final_manifest())
        bundle = TransferResultBundle(KEY, 0, tuple(
            _result(name, CompletionCertainty.COMPLETED)
            for name in DSV4_DSPARK_K7_CONTRACT.physical_regions
        ))
        journal = DurablePDJournal(str(tmp_path / "journal"), _config(tmp_path / "journal"))
        order = []

        def record(event, **kwargs):
            journal.append(event, **kwargs)
            order.append(event)

        def apply(message):
            assert order[-1] == "TRANSFER_RESULTS_RECORDED"
            for result in message.results:
                tracker.record_transfer(result)
            order.append("applied")

        async def send(message):
            if isinstance(message, ControlAck):
                assert order[-1] == "applied"
                if failure == "ack":
                    raise OSError("ACK lost")
                order.append("ack")
            if isinstance(message, ReadyAck):
                assert order[-1] == "HANDOFF_COMMITTED"
                order.append("ready")

        async def stream(key):
            assert key == KEY

        service = object.__new__(PDServingService)
        service.decode_connector = SimpleNamespace(
            validate_transfer_results=tracker.validate_results,
            query=lambda key: SimpleNamespace(reservation_id="reservation", state="TRANSFERRING"),
            record_transfer_results=apply, commit=tracker.commit,
        )
        service._validate_open_route = lambda message, session: None
        service._record, service._stream_decode = record, stream
        queue = asyncio.Queue()
        for message in (OpenRoute(KEY, "prepared", "reservation", "digest", "cap", "compat", "p", 1),
                        bundle, _commit(_final_manifest())):
            queue.put_nowait(message)
        if failure == "fsync":
            def failed_sync(fd):
                raise OSError("fsync failed")
            monkeypatch.setattr("os.fsync", failed_sync)
        try:
            if failure == "none":
                await service._serve_external_handoff(SimpleNamespace(send=send), queue, KEY)
                assert order == ["TRANSFER_RESULTS_RECORDED", "applied", "ack", "HANDOFF_COMMITTED", "ready"]
            else:
                with pytest.raises(OSError):
                    await service._serve_external_handoff(SimpleNamespace(send=send), queue, KEY)
                assert "ready" not in order and "HANDOFF_COMMITTED" not in order
                if failure == "fsync":
                    assert not any(unit.attempts for unit in tracker._chunks[0].units.values())
                else:
                    # A lost ACK does not undo durable facts; an identical
                    # replay is legal, with conflicting replay tested below.
                    tracker.validate_results(bundle)
        finally:
            monkeypatch.undo()
            journal.close()

    asyncio.run(exercise())


def test_bundle_validates_every_unit_before_applying_and_preserves_zero_byte_facts():
    tracker = CompletionTracker(KEY, "reservation")
    tracker.register_chunk(_final_manifest())
    results = tuple(_result(name, CompletionCertainty.COMPLETED)
                    for name in DSV4_DSPARK_K7_CONTRACT.physical_regions)
    with pytest.raises(ValueError, match="expected completion set"):
        tracker.validate_results(TransferResultBundle(KEY, 0, results[:-1]))
    with pytest.raises(ValueError, match="duplicate"):
        TransferResultBundle(KEY, 0, results + results[:1])
    invalid = results[:-1] + (msgspec.structs.replace(results[-1], source_rank_id=4),)
    with pytest.raises(ValueError, match="rank mapping"):
        tracker.validate_results(TransferResultBundle(KEY, 0, invalid))
    assert not any(unit.attempts for unit in tracker._chunks[0].units.values())
    bundle = TransferResultBundle(KEY, 0, results)
    tracker.validate_results(bundle)
    for result in results:
        tracker.record_transfer(result)
    tracker.validate_results(bundle)
    changed = results[:-1] + (msgspec.structs.replace(results[-1], error_code="changed"),)
    with pytest.raises(ValueError, match="conflicting"):
        tracker.validate_results(TransferResultBundle(KEY, 0, changed))
    assert tracker.commit(_commit(_final_manifest())).manifest_hash == _final_manifest().manifest_hash


def _final_manifest() -> ChunkManifest:
    continuation = ContinuationMetadata(
        prompt_token_ids=(1, 2, 3),
        max_new_tokens=8,
        temperature=0.0,
        top_p=1.0,
        top_k=None,
        seed=None,
        stop_strings=(),
        eos_token_id=4,
    )
    units = tuple(
        TransferUnit(0, component, 0)
        for component in DSV4_DSPARK_K7_CONTRACT.physical_regions
    )
    digest = chunk_payload_hash(
        KEY,
        chunk_id=0,
        start_token=0,
        end_token=3,
        final=True,
        expected_units=units,
        copies_by_destination_rank={0: ()},
        rank_mapping=(RankMapping(0, 0),),
    )
    return ChunkManifest(
        key=KEY,
        chunk_id=0,
        start_token=0,
        end_token=3,
        final=True,
        manifest_hash=digest,
        expected_units=units,
        copies_by_destination_rank={0: ()},
        rank_mapping=(RankMapping(0, 0),),
        first_token=9,
        metadata_hash=continuation_metadata_hash(continuation),
        continuation=continuation,
    )


def _result(component: str, certainty: CompletionCertainty, attempt: str = "a1"):
    return TransferResult(
        key=KEY,
        chunk_id=0,
        source_rank_id=0,
        destination_rank_id=0,
        component_id=component,
        attempt_id=attempt,
        certainty=certainty.value,
    )


def _chunk_manifest(
    chunk_id: int,
    start_token: int,
    end_token: int,
    *,
    source_prefix_hit_tokens: int,
) -> ChunkManifest:
    units = (TransferUnit(0, "ori", 0),)
    digest = chunk_payload_hash(
        KEY,
        chunk_id=chunk_id,
        start_token=start_token,
        end_token=end_token,
        final=False,
        expected_units=units,
        copies_by_destination_rank={0: ()},
        rank_mapping=(RankMapping(0, 0),),
        source_prefix_hit_tokens=source_prefix_hit_tokens,
    )
    return ChunkManifest(
        key=KEY,
        chunk_id=chunk_id,
        start_token=start_token,
        end_token=end_token,
        final=False,
        manifest_hash=digest,
        expected_units=units,
        copies_by_destination_rank={0: ()},
        rank_mapping=(RankMapping(0, 0),),
        source_prefix_hit_tokens=source_prefix_hit_tokens,
    )


def _commit(manifest: ChunkManifest) -> CommitRequest:
    return CommitRequest(KEY, manifest.manifest_hash, 9, manifest.metadata_hash)


def test_mapping_is_hashed_and_completion_cannot_claim_another_source():
    tracker = CompletionTracker(KEY, "reservation")
    manifest = _final_manifest()
    mapping = (RankMapping(4, 0),)
    changed = msgspec.structs.replace(manifest, rank_mapping=mapping)
    with pytest.raises(ValueError, match="physical write set"):
        tracker.register_chunk(changed)
    tracker.register_chunk(manifest)
    result = msgspec.structs.replace(_result("ori", CompletionCertainty.COMPLETED), source_rank_id=4)
    with pytest.raises(ValueError, match="rank mapping"):
        tracker.record_transfer(result)


def test_missing_component_cannot_commit_then_lost_ack_is_stable() -> None:
    tracker = CompletionTracker(KEY, "reservation")
    manifest = _final_manifest()
    tracker.register_chunk(manifest)
    for component in DSV4_DSPARK_K7_CONTRACT.physical_regions[:-1]:
        tracker.record_transfer(_result(component, CompletionCertainty.COMPLETED))
    with pytest.raises(RuntimeError, match="incomplete"):
        tracker.commit(_commit(manifest))

    tracker.record_transfer(
        _result(
            DSV4_DSPARK_K7_CONTRACT.physical_regions[-1],
            CompletionCertainty.COMPLETED,
        )
    )
    first = tracker.commit(_commit(manifest))
    assert tracker.commit(_commit(manifest)) == first
    assert tracker.replay_ready_ack() == first
    assert tracker.query().state == CompletionState.READY.value
    assert tracker.admit_decode()
    assert not tracker.admit_decode()
    with pytest.raises(RuntimeError, match="committed handoff"):
        tracker.abort(deterministic=True, error_code="late-cancel")


def test_duplicate_attempt_is_idempotent_but_conflicting_fact_is_rejected() -> None:
    tracker = CompletionTracker(KEY, "reservation")
    tracker.register_chunk(_final_manifest())
    result = _result("ori", CompletionCertainty.COMPLETED)
    tracker.record_transfer(result)
    tracker.record_transfer(result)
    with pytest.raises(ValueError, match="conflicting"):
        tracker.record_transfer(_result("ori", CompletionCertainty.UNKNOWN))


def test_unknown_quarantines_and_late_success_cannot_revive() -> None:
    tracker = CompletionTracker(KEY, "reservation")
    manifest = _final_manifest()
    tracker.register_chunk(manifest)
    tracker.record_transfer(_result("ori", CompletionCertainty.UNKNOWN))
    tracker.record_transfer(_result("ori", CompletionCertainty.COMPLETED, "a2"))
    assert tracker.query().state == CompletionState.QUARANTINED.value
    with pytest.raises(RuntimeError, match="uncertain"):
        tracker.commit(_commit(manifest))


def test_manifest_hash_and_generation_are_fail_closed() -> None:
    tracker = CompletionTracker(KEY, "reservation")
    manifest = _final_manifest()
    tampered = ChunkManifest(
        key=manifest.key,
        chunk_id=manifest.chunk_id,
        start_token=manifest.start_token,
        end_token=manifest.end_token + 1,
        final=manifest.final,
        manifest_hash=manifest.manifest_hash,
        rank_mapping=manifest.rank_mapping,
        expected_units=manifest.expected_units,
        copies_by_destination_rank=manifest.copies_by_destination_rank,
        first_token=manifest.first_token,
        metadata_hash=manifest.metadata_hash,
        continuation=manifest.continuation,
    )
    with pytest.raises(ValueError, match="physical write set"):
        tracker.register_chunk(tampered)

    tampered_p_hit = ChunkManifest(
        key=manifest.key,
        chunk_id=manifest.chunk_id,
        start_token=manifest.start_token,
        end_token=manifest.end_token,
        final=manifest.final,
        manifest_hash=manifest.manifest_hash,
        rank_mapping=manifest.rank_mapping,
        expected_units=manifest.expected_units,
        copies_by_destination_rank=manifest.copies_by_destination_rank,
        source_prefix_hit_tokens=128,
        first_token=manifest.first_token,
        metadata_hash=manifest.metadata_hash,
        continuation=manifest.continuation,
    )
    with pytest.raises(ValueError, match="physical write set"):
        tracker.register_chunk(tampered_p_hit)

    tracker.register_chunk(manifest)
    stale = TransferResult(
        key=HandoffKey("request", "handoff", 2, 2, 3),
        chunk_id=0,
        source_rank_id=0,
        destination_rank_id=0,
        component_id="ori",
        attempt_id="old",
        certainty=CompletionCertainty.COMPLETED.value,
    )
    with pytest.raises(ValueError, match="identity"):
        tracker.record_transfer(stale)


def test_p_prefix_hit_must_remain_stable_across_chunks() -> None:
    tracker = CompletionTracker(KEY, "reservation")
    tracker.register_chunk(
        _chunk_manifest(0, 0, 256, source_prefix_hit_tokens=128)
    )

    with pytest.raises(ValueError, match="remain stable"):
        tracker.register_chunk(
            _chunk_manifest(1, 256, 512, source_prefix_hit_tokens=256)
        )
