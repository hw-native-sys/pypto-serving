# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Contract tests run without Torch, Mooncake, ACL, or an NPU."""

from dataclasses import asdict, replace
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from pypto_serving.transfer.errors import ErrorCode, TransferError, TransferFailure
from pypto_serving.transfer.events import StatusIndex
from .helpers import FakeTransferProvider
from pypto_serving.transfer.types import (
    CompletionCertainty as Certainty, OwnerRef, ProviderCapabilities,
    ProviderTransferTask, RegionLease, Segment, Stage, TransferAttemptRef,
)


def test_owner_identity_includes_worker_namespace():
    source = OwnerRef("run", 0, 1, 1, "prefill-worker")
    destination = OwnerRef("run", 0, 1, 1, "decode-worker")
    assert source != destination
    assert len({source, destination}) == 2


def test_transfer_identity_allows_independent_rank_placement():
    source = OwnerRef("run", 4, 1, 1, "P")
    destination = OwnerRef("run", 8, 1, 1, "D")
    attempt = TransferAttemptRef("r", "p", "h", "a", 1, 1, 0, source, destination, "m")
    assert attempt.source.rank_id == 4
    assert attempt.destination.rank_id == 8
    with pytest.raises(ValueError, match="same.run"):
        replace(attempt, destination=replace(destination, run_id="another-run"))


@pytest.fixture
def task():
    source, destination = OwnerRef("run", 0, 1, 1, "P"), OwnerRef("run", 0, 2, 2, "D")
    attempt = TransferAttemptRef("request", "plan", "handoff", "attempt", 1, 1, 0,
                                 source, destination, "manifest")
    return ProviderTransferTask(attempt, (Segment(
        "ori", RegionLease(source, "src", 1, 128), RegionLease(destination, "dst", 1, 128),
        0, 0, 64),))


def submitted(task):
    journal = StatusIndex(task.attempt.source)
    journal.create(task.attempt)
    for stage in (Stage.VALIDATED, Stage.QUEUED, Stage.WAITING_FENCE, Stage.READY, Stage.SUBMITTED):
        journal.advance(task.attempt, stage)
    return journal


def test_terminal_is_immutable_and_queries_are_idempotent(task):
    journal = submitted(task)
    terminal = journal.advance(task.attempt, Stage.COMPLETED)
    assert journal.query(task.attempt).event is terminal
    assert journal.query(task.attempt).event is terminal
    with pytest.raises(ValueError, match="immutable"):
        journal.advance(task.attempt, Stage.FAILED,
                        TransferError(ErrorCode.DEADLINE, Certainty.UNKNOWN))
    history = journal.history(task.attempt)
    assert all(right.parent_event_id == left.event_id for left, right in zip(history, history[1:]))
    assert terminal.certainty == Certainty.COMPLETED
    assert json.loads(json.dumps(terminal.to_dict()))["stage"] == "COMPLETED"


def test_unknown_cannot_be_overwritten_by_late_success(task):
    journal = submitted(task)
    error = TransferError(ErrorCode.DEADLINE, Certainty.UNKNOWN)
    journal.advance(task.attempt, Stage.FAILED, error)
    with pytest.raises(ValueError, match="immutable"):
        journal.advance(task.attempt, Stage.COMPLETED)
    assert journal.query(task.attempt).event.certainty == Certainty.UNKNOWN


def test_submitted_cannot_become_not_submitted(task):
    journal = submitted(task)
    with pytest.raises(ValueError, match="cannot become"):
        journal.advance(task.attempt, Stage.FAILED,
                        TransferError(ErrorCode.DEADLINE, Certainty.NOT_SUBMITTED))


def test_capacity_never_discards_unreleased_results(task):
    journal = StatusIndex(task.attempt.source, capacity=1)
    journal.create(task.attempt)
    other = replace(task.attempt, attempt_id="other", attempt_sequence=1)
    with pytest.raises(TransferFailure) as exc:
        journal.create(other)
    assert exc.value.error.certainty == Certainty.NOT_SUBMITTED
    with pytest.raises(ValueError):
        journal.release(task.attempt)
    journal.advance(task.attempt, Stage.FAILED,
                    TransferError(ErrorCode.INVALID_PLAN, Certainty.NOT_SUBMITTED))
    with pytest.raises(TransferFailure):
        journal.create(other)
    journal.release(task.attempt)
    journal.create(other)
    assert journal.query(task.attempt).kind == "unknown_not_observed"
    with pytest.raises(ValueError, match="retired"):
        journal.create(task.attempt)


def test_stale_and_unknown_are_not_claimed_unsubmitted(task):
    journal = StatusIndex(task.attempt.source)
    assert journal.query(task.attempt).kind == "unknown_not_observed"
    stale = replace(task.attempt, source=replace(task.attempt.source, generation=0))
    assert journal.query(stale).kind == "stale_owner"
    with pytest.raises(TransferFailure):
        journal.create(stale)


def test_duplicate_and_mismatched_identity_rejected(task):
    journal = StatusIndex(task.attempt.source)
    journal.create(task.attempt)
    with pytest.raises(ValueError, match="replayed"):
        journal.create(task.attempt)
    with pytest.raises(ValueError, match="identity"):
        journal.advance(replace(task.attempt, data_generation=2), Stage.VALIDATED)


@pytest.mark.parametrize("field,value", [
    ("source_offset", -1), ("destination_offset", True), ("length", 0),
    ("length", 129), ("length", 1 << 63), ("source_offset", 128),
])
def test_invalid_segments_rejected(task, field, value):
    with pytest.raises(ValueError):
        replace(task.segments[0], **{field: value})


def test_alignment_and_owner_validation(task):
    bad = ProviderTransferTask(task.attempt, (replace(task.segments[0], length=4),))
    with pytest.raises(ValueError, match="alignment"):
        bad.validate(ProviderCapabilities())
    with pytest.raises(ValueError, match="owner"):
        ProviderTransferTask(task.attempt, (replace(
            task.segments[0], source=replace(task.segments[0].source,
                                          owner=replace(task.attempt.source, generation=8))),))


def test_fake_block_and_lost_completion(task):
    provider = FakeTransferProvider("lost_completion")
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(provider.write, task)
        try:
            assert provider.started.wait(2)
            assert not future.done()
        finally:
            provider.unblock.set()
        with pytest.raises(TransferFailure) as exc:
            future.result(2)
    assert exc.value.error.certainty == Certainty.UNKNOWN


def test_fake_success_and_failure(task):
    FakeTransferProvider().write(task)
    with pytest.raises(TransferFailure):
        FakeTransferProvider("failure").write(task)


def test_wire_contract_contains_no_backend_objects(task):
    wire = json.dumps(asdict(task))
    assert "address" not in wire and "pointer" not in wire
    assert "mooncake" not in wire.lower()


@pytest.mark.parametrize("current", list(Stage))
@pytest.mark.parametrize("target", list(Stage))
def test_transition_matrix(task, current, target):
    order = [Stage.CREATED, Stage.VALIDATED, Stage.QUEUED, Stage.WAITING_FENCE,
             Stage.READY, Stage.SUBMITTED, Stage.COMPLETED]
    journal = StatusIndex(task.attempt.source)
    journal.create(task.attempt)
    if current == Stage.FAILED:
        journal.advance(task.attempt, Stage.FAILED, TransferError(ErrorCode.FENCE_FAILED, Certainty.NOT_SUBMITTED))
    else:
        for stage in order[1:order.index(current) + 1]:
            journal.advance(task.attempt, stage)
    allowed = not current.terminal and (target == Stage.FAILED or target == order[order.index(current) + 1])
    error = (TransferError(ErrorCode.BACKEND_FAILURE,
                          Certainty.UNKNOWN if current == Stage.SUBMITTED else Certainty.NOT_SUBMITTED)
             if target == Stage.FAILED else None)
    if allowed:
        assert journal.advance(task.attempt, target, error).stage == target
    else:
        before = journal.history(task.attempt)
        with pytest.raises(ValueError):
            journal.advance(task.attempt, target, error)
        assert journal.history(task.attempt) == before
