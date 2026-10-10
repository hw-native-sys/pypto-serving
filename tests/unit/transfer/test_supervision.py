# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Process supervision tests; only the pidfd case needs Linux, never an NPU."""
from dataclasses import replace
import os
import socket
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from pypto_serving.transfer.events import StatusIndex
from pypto_serving.transfer.errors import ErrorCode, TransferError
from pypto_serving.transfer.supervisor import OwnerHealthMonitor, OwnerProcessHandle, OwnerSupervisor
from pypto_serving.transfer.types import CompletionCertainty, OwnerRef, Stage, TransferAttemptRef


def uncertain_event():
    source, destination = OwnerRef("run", 0, 1, 1, "P"), OwnerRef("run", 0, 1, 1, "D")
    ref = TransferAttemptRef("r", "p", "h", "a", 1, 1, 0, source, destination, "m")
    index = StatusIndex(source, 8)
    index.create(ref)
    for stage in (Stage.VALIDATED, Stage.QUEUED, Stage.WAITING_FENCE, Stage.READY, Stage.SUBMITTED):
        index.advance(ref, stage)
    return index.advance(ref, Stage.FAILED, TransferError(ErrorCode.OWNER_LOST, CompletionCertainty.UNKNOWN))


def test_idle_health_monitor_and_normal_stop():
    left, right = socket.socketpair()
    observed = threading.Event()
    monitor = OwnerHealthMonitor(SimpleNamespace(fd=left.fileno()), observed.set)
    try:
        right.close()
        assert observed.wait(1)
    finally:
        monitor.close()
        left.close()
    left, right = socket.socketpair()
    observed.clear()
    monitor = OwnerHealthMonitor(SimpleNamespace(fd=left.fileno()), observed.set)
    monitor.close()
    left.close()
    right.close()
    assert not observed.is_set()


@pytest.mark.skipif(not hasattr(os, "pidfd_open"), reason="Linux pidfd contract")
def test_pidfd_observes_exact_child_without_reaping_it():
    child = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE)
    handle = OwnerProcessHandle(child.pid)
    exited = threading.Event()
    monitor = OwnerHealthMonitor(handle, exited.set)
    try:
        handle.terminate()
        assert exited.wait(3)
        handle.wait(1)
        # The runtime, not the supervision handle, still owns reaping.
        assert os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOWAIT).si_pid == child.pid
    finally:
        monitor.close()
        handle.kill()
        child.wait(timeout=3)
        handle.close()
        child.stdin.close()


class Process:
    def __init__(self, fail=False):
        self.fail = fail
        self.terminated = False
    def terminate(self):
        self.terminated = True
        if self.fail:
            raise PermissionError()
    def wait(self, timeout):
        if self.fail:
            raise TimeoutError()
    def kill(self):
        if self.fail:
            raise PermissionError()


def test_full_group_recovery_requires_every_peer_death():
    event = uncertain_event()
    source, destination = event.attempt.source, event.attempt.destination
    local = (source, replace(source, rank_id=1))
    peers = (destination, replace(destination, rank_id=1))
    facts = []
    supervisor = OwnerSupervisor(source, (Process(), Process()), emit=facts.append,
                                 local_owners=local, peer_owners=peers)
    supervisor.poison(event)
    assert all(fact.recovery_set == local + peers for fact in facts)
    fresh = replace(source, generation=2, endpoint_generation=2)
    supervisor.confirm_peer_dead(peers[0])
    with pytest.raises(RuntimeError):
        supervisor.permit_rebuild(fresh)
    supervisor.confirm_peer_dead(peers[1])
    supervisor.permit_rebuild(fresh)
    with pytest.raises(ValueError):
        supervisor.permit_rebuild(replace(fresh, worker_id="different-worker"))


def test_failed_process_termination_does_not_skip_other_children():
    event = uncertain_event()
    failed, healthy = Process(True), Process()
    facts = []
    supervisor = OwnerSupervisor(event.attempt.source, (failed, healthy), emit=facts.append)
    with pytest.raises(RuntimeError, match="confirmed dead"):
        supervisor.poison(event)
    assert healthy.terminated
    assert not supervisor.dead
    assert facts[-1].stage == "OwnerTerminationFailed"
    assert not any(fact.stage == "OwnerDead" for fact in facts)
    with pytest.raises(RuntimeError, match="incomplete"):
        supervisor.poison(event)


def test_event_delivery_failure_still_terminates_every_process():
    event = uncertain_event()
    processes = (Process(), Process())
    def failed_emit(event):
        raise OSError()
    supervisor = OwnerSupervisor(event.attempt.source, processes, emit=failed_emit)
    with pytest.raises(RuntimeError, match="event delivery"):
        supervisor.poison(event)
    assert supervisor.dead and all(p.terminated for p in processes)
