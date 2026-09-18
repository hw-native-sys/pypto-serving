# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU integration for provider lifecycle, bounded admission, and recovery."""

from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from pypto_serving.transfer.agent import AlreadyCompletedFence, TransferAgent
from pypto_serving.transfer.errors import ErrorCode, TransferError, TransferFailure
from pypto_serving.transfer.mooncake import MooncakeTransferProvider
from pypto_serving.transfer.provider import FakeTransferProvider
from pypto_serving.transfer.supervisor import OwnerSupervisor
from pypto_serving.transfer.types import (
    CompletionCertainty as Certainty, OwnerRef, ProviderTransferTask, RegionLease,
    Segment, Stage, TransferAttemptRef,
)


@pytest.fixture
def task():
    src, dst = OwnerRef("run", 0, 1, 1, "P"), OwnerRef("run", 0, 2, 2, "D")
    attempt = TransferAttemptRef("r", "p", "h", "a", 1, 1, 0, src, dst, "m")
    return ProviderTransferTask(attempt, (Segment(
        "ori", RegionLease(src, "src", 1, 256), RegionLease(dst, "dst", 1, 256), 64, 128, 64),))


class Engine:
    def __init__(self):
        self.calls = []
        self.write_code = 0

    def initialize(self, *args):
        self.calls.append(("init", args))
        return 0

    def get_rpc_port(self):
        return 1234

    def register_memory(self, address, extent):
        self.calls.append(("register", address, extent))
        return 0

    def unregister_memory(self, address):
        self.calls.append(("unregister", address))
        return 0

    def batch_transfer_sync_write(self, *args):
        self.calls.append(("write", args))
        return self.write_code


def configured_provider(task):
    engine = Engine()
    provider = MooncakeTransferProvider("host", engine_factory=lambda: engine)
    segment = task.segments[0]
    provider.register(segment.source, 4096)
    from dataclasses import asdict
    provider.install_destination(segment.destination, {
        "lease": asdict(segment.destination), "endpoint": "peer:1234", "address": 8192,
    })
    return provider, engine


def test_mooncake_segment_addresses_and_normal_release(task):
    provider, engine = configured_provider(task)
    provider.write(task)
    assert engine.calls[-1] == ("write", ("peer:1234", [4160], [8320], [64]))
    provider.release()
    assert engine.calls[-1] == ("unregister", 4096)


def test_native_failure_forbids_unregister_and_reuse(task):
    provider, engine = configured_provider(task)
    engine.write_code = -1
    with pytest.raises(TransferFailure) as exc:
        provider.write(task)
    assert exc.value.error.certainty == Certainty.UNKNOWN
    with pytest.raises(TransferFailure):
        provider.release()
    with pytest.raises(TransferFailure):
        provider.write(task)
    assert not any(call[0] == "unregister" for call in engine.calls)


def test_mooncake_thread_affinity(task):
    provider, _ = configured_provider(task)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(TransferFailure):
            pool.submit(provider.write, task).result()
    provider.release()


def test_agent_success_and_retained_status(task):
    agent = TransferAgent(task.attempt.source, FakeTransferProvider(), poison=lambda event: None)
    try:
        future = agent.submit(task, AlreadyCompletedFence())
        result = future.result(2)
        assert result.stage == Stage.COMPLETED
        assert agent.status.query(task.attempt).event is result
        assert not future.cancel()
    finally:
        agent.close()


def test_backpressure_counts_inflight_bytes(task):
    provider = FakeTransferProvider("block")
    agent = TransferAgent(task.attempt.source, provider, poison=lambda event: None, max_bytes=64)
    try:
        first = agent.submit(task, AlreadyCompletedFence())
        assert provider.started.wait(1)
        with pytest.raises(TransferFailure):
            agent.submit(replace(task, attempt=replace(task.attempt, attempt_id="b")), AlreadyCompletedFence())
        provider.unblock.set()
        assert first.result(2).stage == Stage.COMPLETED
    finally:
        provider.unblock.set()
        agent.close()


def test_native_deadline_poison_and_late_completion(task):
    provider = FakeTransferProvider("block")
    poisoned = []
    def poison(event):
        poisoned.append(event)
        provider.unblock.set()
    agent = TransferAgent(task.attempt.source, provider, poison=poison)
    try:
        event = agent.submit(task, AlreadyCompletedFence(), timeout=0.05).result(2)
        assert event.certainty == Certainty.UNKNOWN
        assert event.stage == Stage.FAILED
        assert len(poisoned) == 1
        assert agent.status.query(task.attempt).event == event
        with pytest.raises(TransferFailure):
            agent.submit(replace(task, attempt=replace(task.attempt, attempt_id="b")), AlreadyCompletedFence())
    finally:
        provider.unblock.set()
        agent.close()


def test_fence_deadline_never_submits(task):
    provider = FakeTransferProvider()
    agent = TransferAgent(task.attempt.source, provider, poison=lambda e: pytest.fail("unexpected poison"))
    try:
        event = agent.submit(task, threading.Event(), timeout=0.02).result(2)
        assert event.certainty == Certainty.NOT_SUBMITTED
        assert provider.calls == 0
        retry = replace(task, attempt=replace(task.attempt, attempt_id="retry", attempt_sequence=1))
        assert agent.submit(retry, AlreadyCompletedFence()).result(2).stage == Stage.COMPLETED
        assert provider.calls == 1
    finally:
        agent.close()


def test_definite_drained_failure_keeps_same_agent_usable(task):
    class DrainedProvider(FakeTransferProvider):
        def write(self, task, on_submitted=lambda: None):
            super().write(task, on_submitted)
            if self.calls == 1:
                # Test backend guarantees its write has stopped before reporting definite failure.
                raise TransferFailure(TransferError(ErrorCode.BACKEND_FAILURE, Certainty.FAILED_DEFINITE))
    provider = DrainedProvider()
    agent = TransferAgent(task.attempt.source, provider, poison=lambda e: pytest.fail("unexpected restart"))
    try:
        event = agent.submit(task, AlreadyCompletedFence()).result(2)
        assert event.certainty == Certainty.FAILED_DEFINITE and event.stage == Stage.FAILED
        retry = replace(task, attempt=replace(task.attempt, attempt_id="retry", attempt_sequence=1))
        assert agent.submit(retry, AlreadyCompletedFence()).result(2).stage == Stage.COMPLETED
        assert provider.calls == 2
    finally:
        agent.close()


def test_late_completion_cannot_publish_before_supervision_finishes(task):
    provider = FakeTransferProvider("block")
    poison_entered, allow_poison = threading.Event(), threading.Event()
    def poison(event):
        poison_entered.set()
        provider.unblock.set()
        assert allow_poison.wait(2)
    agent = TransferAgent(task.attempt.source, provider, poison=poison)
    try:
        future = agent.submit(task, AlreadyCompletedFence(), timeout=0.02)
        assert poison_entered.wait(1)
        # Join establishes that the late native return was handled, not just scheduled.
        agent.close()
        assert not future.done()
        allow_poison.set()
        assert future.result(1).certainty == Certainty.UNKNOWN
    finally:
        allow_poison.set()
        provider.unblock.set()
        agent.close()


def test_queued_deadline_expires_while_previous_native_call_is_blocked(task):
    provider = FakeTransferProvider("block")
    agent = TransferAgent(task.attempt.source, provider, poison=lambda event: None)
    try:
        first = agent.submit(task, AlreadyCompletedFence(), timeout=2)
        assert provider.started.wait(1)
        second_task = replace(task, attempt=replace(task.attempt, attempt_id="second", attempt_sequence=1))
        second = agent.submit(second_task, AlreadyCompletedFence(), timeout=0.02)
        assert second.result(1).certainty == Certainty.NOT_SUBMITTED
        assert provider.calls == 1
        assert not first.done()
        provider.unblock.set()
        assert first.result(1).stage == Stage.COMPLETED
    finally:
        provider.unblock.set()
        agent.close()


def test_supervisor_requires_both_owner_deaths(task):
    class Process:
        killed = False
        def terminate(self):
            pass
        def kill(self):
            self.killed = True
        def wait(self, timeout):
            if not self.killed:
                raise TimeoutError()
    process, events = Process(), []
    supervisor = OwnerSupervisor(task.attempt.source, (process,), emit=events.append)
    provider = FakeTransferProvider("failure")
    agent = TransferAgent(task.attempt.source, provider, poison=supervisor.poison)
    try:
        event = agent.submit(task, AlreadyCompletedFence()).result(2)
        supervisor.poison(event)
        assert [e.stage for e in events] == ["OwnerPoisoned", "PeerPoisonRequired", "OwnerDead"]
        fresh = replace(task.attempt.source, generation=3, endpoint_generation=3)
        with pytest.raises(RuntimeError):
            supervisor.permit_rebuild(fresh)
        with pytest.raises(ValueError):
            supervisor.confirm_peer_dead(replace(task.attempt.destination, generation=1))
        supervisor.confirm_peer_dead(task.attempt.destination)
        supervisor.permit_rebuild(fresh)
    finally:
        agent.close()
