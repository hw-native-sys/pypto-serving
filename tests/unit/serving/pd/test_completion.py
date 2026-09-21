# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pytest

from pypto_serving.serving.pd.completion import CompletionState, CompletionTracker
from pypto_serving.model.deepseek_dspark.pd_adapter import DSV4_DSPARK_K7_CONTRACT
from pypto_serving.serving.pd.protocol import (
    ChunkManifest,
    CommitRequest,
    ContinuationMetadata,
    HandoffKey,
    TransferResult,
    TransferUnit,
    chunk_payload_hash,
    continuation_metadata_hash,
)
from pypto_serving.transfer.types import CompletionCertainty


KEY = HandoffKey("request", "handoff", 1, 2, 3)


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
        copies_by_rank={0: ()},
    )
    return ChunkManifest(
        key=KEY,
        chunk_id=0,
        start_token=0,
        end_token=3,
        final=True,
        manifest_hash=digest,
        expected_units=units,
        copies_by_rank={0: ()},
        first_token=9,
        metadata_hash=continuation_metadata_hash(continuation),
        continuation=continuation,
    )


def _result(component: str, certainty: CompletionCertainty, attempt: str = "a1"):
    return TransferResult(
        key=KEY,
        chunk_id=0,
        rank_id=0,
        component_id=component,
        attempt_id=attempt,
        certainty=certainty.value,
    )


def _commit(manifest: ChunkManifest) -> CommitRequest:
    return CommitRequest(KEY, manifest.manifest_hash, 9, manifest.metadata_hash)


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
        expected_units=manifest.expected_units,
        copies_by_rank=manifest.copies_by_rank,
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
        expected_units=manifest.expected_units,
        copies_by_rank=manifest.copies_by_rank,
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
        rank_id=0,
        component_id="ori",
        attempt_id="old",
        certainty=CompletionCertainty.COMPLETED.value,
    )
    with pytest.raises(ValueError, match="identity"):
        tracker.record_transfer(stale)
