# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Owner control channel validation without device initialization."""

from dataclasses import asdict
import socket
import struct
import threading

import pytest

from pypto_serving.transfer.errors import TransferFailure
from pypto_serving.transfer.owner import OwnerBridge, _receive, _send, _task
from pypto_serving.transfer.types import (
    CompletionCertainty, OwnerRef, ProviderTransferTask, RegionLease, Segment, TransferAttemptRef,
)


def test_task_wire_round_trip():
    source, destination = OwnerRef("run", 0, 1, 1, "P"), OwnerRef("run", 0, 2, 2, "D")
    ref = TransferAttemptRef("r", "p", "h", "a", 0, 0, 0, source, destination, "m")
    task = ProviderTransferTask(ref, (Segment(
        "ori", RegionLease(source, "s", 1, 64), RegionLease(destination, "d", 1, 64), 0, 0, 64),))
    assert _task(asdict(task)) == task
    malformed = asdict(task)
    malformed["segments"][0]["length"] = 65
    with pytest.raises(ValueError):
        _task(malformed)


def test_owner_preserves_numeric_backend_error():
    bridge = OwnerBridge(OwnerRef("run", 0, 1, 1, "P"), "host")
    def peer():
        request = _receive(bridge._child)
        _send(bridge._child, dict(ok=False, owner=request["owner"], sequence=request["sequence"],
                                 code="BACKEND_FAILURE", certainty="UNKNOWN", backend_code=-7))
    thread = threading.Thread(target=peer)
    thread.start()
    try:
        with pytest.raises(TransferFailure) as exc:
            bridge._request("write")
        assert exc.value.error.backend_code == -7
    finally:
        thread.join(2)
        bridge._parent.close()
        bridge._child.close()


@pytest.mark.parametrize("mode", ["valid", "stale", "eof"])
def test_owner_reply_sequence_and_eof(mode):
    bridge = OwnerBridge(OwnerRef("run", 0, 1, 1, "P"), "host")
    def peer():
        request = _receive(bridge._child)
        if mode != "eof":
            _send(bridge._child, dict(ok=True, owner=request["owner"],
                                     sequence=request["sequence"] if mode == "valid" else 0))
        bridge._child.close()
    thread = threading.Thread(target=peer)
    thread.start()
    try:
        if mode == "valid":
            assert bridge._request("probe")["ok"]
        else:
            with pytest.raises(TransferFailure) as exc:
                bridge._request("probe")
            assert exc.value.error.certainty == CompletionCertainty.UNKNOWN
            with pytest.raises(TransferFailure):
                bridge._request("probe")
        assert tuple(bridge.commands) == ({"sequence": 1, "operation": "probe"},)
    finally:
        bridge._parent.close()
        thread.join(2)
        assert not thread.is_alive()


def test_abandon_requires_owner_death_and_never_releases_pins():
    from types import SimpleNamespace

    bridge = OwnerBridge(OwnerRef("run", 0, 1, 1, "P"), "host")
    try:
        with pytest.raises(RuntimeError, match="death"):
            bridge.close_after_owner_death()
        def alive(timeout):
            raise TimeoutError()
        bridge.process_handle = SimpleNamespace(wait=alive, close=lambda: None)
        with pytest.raises(TimeoutError):
            bridge.close_after_owner_death()
        retained = object()
        bridge._retentions.append(retained)
        bridge.process_handle.wait = lambda timeout: None
        bridge.close_after_owner_death()
        bridge.close_after_owner_death()
        assert bridge._closed and bridge._poisoned
        assert bridge._retentions == [retained]
        assert not bridge.commands
    finally:
        bridge._parent.close()
        bridge._child.close()


def test_channel_rejects_oversize_frame_before_payload():
    left, right = socket.socketpair()
    try:
        left.sendall(struct.pack("!I", 1 << 30))
        with pytest.raises(ValueError, match="length"):
            _receive(right)
    finally:
        left.close()
        right.close()


def test_delayed_old_completion_cannot_complete_new_attempt():
    from pypto_serving.transfer.agent import AlreadyCompletedFence, TransferAgent
    from pypto_serving.transfer.types import Stage
    source, destination = OwnerRef("run", 0, 2, 2, "P"), OwnerRef("run", 0, 2, 2, "D")
    bridge = OwnerBridge(source, "host")
    src, dst = RegionLease(source, "s", 2, 64), RegionLease(destination, "d", 2, 64)
    bridge._sources["s"], bridge._destinations[dst.registration_key] = src, dst
    task = ProviderTransferTask(TransferAttemptRef("r", "p", "h", "a", 2, 2, 0, source, destination, "m"),
                               (Segment("ori", src, dst, 0, 0, 64),))
    requested, deliver = threading.Event(), threading.Event()
    def delayed_peer():
        request = _receive(bridge._child)
        requested.set()
        assert deliver.wait(2)
        old_owner = dict(request["owner"], generation=1, endpoint_generation=1)
        _send(bridge._child, dict(ok=True, owner=old_owner, sequence=request["sequence"]))
    peer = threading.Thread(target=delayed_peer)
    peer.start()
    poisoned = []
    agent = TransferAgent(source, bridge, poison=poisoned.append)
    try:
        future = agent.submit(task, AlreadyCompletedFence())
        assert requested.wait(1)
        assert agent.status.query(task.attempt).event.stage == Stage.SUBMITTED
        deliver.set()
        event = future.result(2)
        assert event.stage == Stage.FAILED and event.certainty == CompletionCertainty.UNKNOWN
        assert poisoned == [event]
        assert agent.status.query(task.attempt).event is event
        assert not any(item.stage == Stage.COMPLETED for item in agent.status.history(task.attempt))
    finally:
        deliver.set()
        peer.join(2)
        agent.close()
        bridge._parent.close()
        bridge._child.close()
