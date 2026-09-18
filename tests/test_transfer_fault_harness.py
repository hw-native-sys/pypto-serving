# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Fault-harness codecs and injection boundaries without native initialization."""
import importlib.util
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest

from pypto_serving.transfer.events import TransferEvent
from pypto_serving.transfer.errors import ErrorCode, TransferError
from pypto_serving.transfer.types import CompletionCertainty, OwnerRef, Stage, TransferAttemptRef


@pytest.fixture
def faults():
    path = Path(__file__).parent / "manual/pd/phase_c/faults.py"
    spec = importlib.util.spec_from_file_location("phase_c_fault_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fault_event_roundtrip(faults):
    source, destination = OwnerRef("r", 0, 1, 1, "P"), OwnerRef("r", 0, 1, 1, "D")
    ref = TransferAttemptRef("r", "p", "h", "a", 1, 1, 0, source, destination, "m")
    event = TransferEvent("e", "before", 7, 123, ref, Stage.FAILED, CompletionCertainty.UNKNOWN,
                          TransferError(ErrorCode.OWNER_LOST, CompletionCertainty.UNKNOWN))
    assert faults.decode_event(event.to_dict()) == event


def test_fault_wrapper_only_drops_successful_write_reply(faults, monkeypatch):
    import pypto_serving.transfer.owner as module
    monkeypatch.setattr(module, "_send", module._send)
    marker = object()
    assert faults.LostReplyFactory(lambda context: marker)(None) is marker
    left, right = socket.socketpair()
    try:
        module._send(left, {"operation": "probe", "ok": True})
        assert module._receive(right)["ok"]
        with pytest.raises(ConnectionError):
            module._send(left, {"operation": "write", "ok": True})
        assert right.recv(1) == b""
    finally:
        left.close()
        right.close()


def test_partial_peer_ack_cannot_allow_group_rebuild(faults):
    from dataclasses import asdict, replace
    peer = OwnerRef("r", 0, 1, 1, "D")
    expected = (peer, replace(peer, rank_id=1))
    supervisor = SimpleNamespace(confirm_peer_dead=lambda peer: pytest.fail("accepted incomplete group"))
    with pytest.raises(ValueError, match="recovery group"):
        faults.confirm_group(supervisor, {"owners": [asdict(peer)]}, expected)


def test_stale_envelope_is_internally_consistent(faults):
    from pypto_serving.transfer.types import ProviderTransferTask, RegionLease, Segment
    source, destination = OwnerRef("r", 0, 2, 2, "P"), OwnerRef("r", 0, 2, 2, "D")
    ref = TransferAttemptRef("r", "p", "h", "a", 2, 2, 0, source, destination, "m")
    task = ProviderTransferTask(ref, (Segment("ori", RegionLease(source, "s", 1, 64),
                                              RegionLease(destination, "d", 1, 64), 0, 0, 64),))
    stale = faults.previous_generation_task(task)
    assert stale.attempt.source.generation == stale.attempt.data_generation == 1
    assert stale.segments[0].source.owner == stale.attempt.source
    assert stale.segments[0].destination.owner == stale.attempt.destination
    assert task.attempt.source.generation == 2


@pytest.mark.parametrize("raises", [False, True])
def test_inflight_observer_brackets_real_backend_call(faults, monkeypatch, raises):
    import pypto_serving.transfer.owner as module
    observations, calls = [], []
    class Engine:
        def batch_transfer_sync_write(self, endpoint, source, destination, length):
            calls.append((len(source), len(destination), sum(length)))
            assert observations[-1]["kind"] == "native_enter"
            if raises:
                raise RuntimeError("backend exception")
            return -1
    class Provider:
        def __init__(self):
            self._engine = Engine()
    monkeypatch.setattr(module, "MooncakeTransferProvider", Provider)
    monkeypatch.setattr(module, "_send", lambda channel, message: observations.append(message))
    wrapped = faults.InflightFactory(lambda context: module.MooncakeTransferProvider(), None)(None)
    if raises:
        with pytest.raises(RuntimeError, match="backend exception"):
            wrapped._engine.batch_transfer_sync_write("peer", [64], [128], [64])
    else:
        assert wrapped._engine.batch_transfer_sync_write("peer", [64], [128], [64]) == -1
    assert calls == [(16384, 16384, 16384 * 64)]
    assert [v["kind"] for v in observations] == ["native_enter", "native_return"]


@pytest.mark.parametrize("damage", [None, "exit", "pid", "owners", "peers", "generation", "missing"])
def test_parent_recovery_requires_complete_death_evidence(damage):
    from dataclasses import asdict
    path = Path(__file__).parent / "manual/pd/phase_c/recovery_pair.py"
    spec = importlib.util.spec_from_file_location("phase_c_recovery_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    record = dict(kind="worker_retirement_certificate", run_id="r", role="sender",
                  generation=1, parent_pid=123,
                  owners=[asdict(OwnerRef("r", r, 1, 1, "sender")) for r in range(2)],
                  peers=[asdict(OwnerRef("r", r, 1, 1, "receiver")) for r in range(2)])
    kwargs = dict(run_id="r", role="sender", generation=1, ranks=2, pid=123, returncode=75)
    if damage == "exit":
        kwargs["returncode"] = 2
    elif damage == "pid":
        record["parent_pid"] = 124
    elif damage in ("owners", "peers"):
        record[damage].pop()
    elif damage == "generation":
        record["generation"] = 0
    elif damage == "missing":
        record = None
    if damage is None:
        module.validate_retirement(record, **kwargs)
    else:
        with pytest.raises(RuntimeError, match="certificate required"):
            module.validate_retirement(record, **kwargs)
