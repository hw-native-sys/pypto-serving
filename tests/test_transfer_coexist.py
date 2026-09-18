# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU checks for coexistence evidence and transfer range generation."""
import importlib.util
from pathlib import Path
import socket

import pytest

from pypto_serving.transfer.types import OwnerRef, ProviderCapabilities, RegionLease


@pytest.fixture
def probe():
    path = Path(__file__).parent / "manual/pd/phase_d/coexist_pair.py"
    spec = importlib.util.spec_from_file_location("coexist_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("left,right,expected", [((1, 5), (3, 8), 2), ((1, 3), (3, 9), 0),
                                                 ((1, 9), (3, 5), 2), ((8, 9), (1, 3), 0)])
def test_interval_intersection(probe, left, right, expected):
    assert probe.intersection_ns(left, right) == expected


def test_repeated_transfer_preserves_guards_and_identity(probe):
    p, d = OwnerRef("test", 0, 1, 1, "P"), OwnerRef("test", 0, 1, 1, "D")
    source, destination = RegionLease(p, "probe", 1, 4096), RegionLease(d, "probe", 1, 4096)
    task = probe.make_task("test", 0, 2, source, destination, 17)
    task.validate(ProviderCapabilities())
    assert len(task.segments) == 17
    assert len(set(task.segments)) == 1
    assert task.nbytes == 17 * (4096 - 128)
    assert task.segments[0].destination_offset == 64
    assert task.attempt.attempt_sequence == 3
    with pytest.raises(ValueError):
        probe.make_task("test", 0, 2, source, destination, 0)


@pytest.mark.parametrize("failed", [False, True])
def test_native_observer_repeats_only_after_first_call(probe, monkeypatch, failed):
    import pypto_serving.transfer.owner as owner
    calls = []

    class Engine:
        def batch_transfer_sync_write(self, endpoint, sources, destinations, lengths):
            calls.append((sources, destinations, lengths))
            if failed:
                raise RuntimeError("backend failure")
            return 0

    class Provider:
        def __init__(self):
            self._engine = Engine()

    monkeypatch.setattr(owner, "MooncakeTransferProvider", Provider)
    read, write = socket.socketpair()
    read.settimeout(1)
    try:
        provider = probe.ObserveFactory(lambda context: owner.MooncakeTransferProvider(), write, 7)(None)
        for count in (1, 7):
            if failed:
                with pytest.raises(RuntimeError, match="backend failure"):
                    provider._engine.batch_transfer_sync_write("peer", [64], [128], [256])
            else:
                assert provider._engine.batch_transfer_sync_write("peer", [64], [128], [256]) == 0
            enter, leave = owner._receive(read), owner._receive(read)
            assert enter["kind"] == "native_enter" and leave["kind"] == "native_return"
            assert enter["ns"] <= leave["ns"]
            assert enter["bytes"] == count * 256
            assert calls[-1] == ([64] * count, [128] * count, [256] * count)
    finally:
        read.close()
        write.close()


@pytest.fixture
def analyzer():
    path = Path(__file__).parent / "manual/pd/phase_d/analyze_coexist.py"
    spec = importlib.util.spec_from_file_location("coexist_analyzer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_overlap_union_preserves_timestamp_precision(analyzer):
    compute = [dict(ts="1788946891565583.094", dur="0.003")]
    transfer = [dict(ts="1788946891565583.095", dur="0.003")] * 2
    result = analyzer.overlap_proof(compute, transfer)
    assert result["pairs"] == 2
    assert result["union_us"] == "0.002"


@pytest.mark.parametrize("failure", [None, "same_stream", "no_overlap", "missing"])
def test_device_evidence_requires_independent_overlapping_streams(analyzer, failure):
    events = [
        dict(ph="X", name="aicore_kernel_0_mix_aic", ts="10", dur="3",
             args={"Physic Stream Id": 40}),
        dict(ph="X", name="batch_putAicpuKernel", ts="11", dur="3",
             args={"Physic Stream Id": 39}),
        dict(ph="X", name="RDMASend", ts="12", dur="0.3",
             args={"stream id": 50, "link type": "ROCE"}),
    ]
    if failure == "same_stream":
        events[1]["args"]["Physic Stream Id"] = 40
    elif failure == "no_overlap":
        events[0]["ts"] = "20"
    elif failure == "missing":
        events.pop()
    if failure:
        with pytest.raises(ValueError):
            analyzer.analyze_trace(events)
    else:
        assert analyzer.analyze_trace(events)["compute_streams"] == [40]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
