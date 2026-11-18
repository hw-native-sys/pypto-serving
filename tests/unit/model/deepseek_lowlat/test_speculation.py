# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from types import SimpleNamespace

import pytest
import torch

from pypto.runtime import DeviceTensor, StackedDeviceTensor
from pypto_serving.model.deepseek_lowlat.native_frames import NativeFrames, allocate_native_hosts
from pypto_serving.model.deepseek_lowlat.speculation import NativeState, validate_packets, validate_native_capacity


def packet(state, *, accepted=0):
    advance = accepted+1
    actual = min(advance, state.remaining)
    ids = tuple(range(100, 100+actual))
    bonus = 100+accepted
    control = [state.epoch, 1, state.anchor, accepted, advance, actual, state.anchor+advance,
               bonus, ids[-1], state.remaining-actual, 1, state.anchor+1, advance,
               int(state.remaining == actual), 0, 0]
    return control, list(ids)+[-1]*(6-actual)


def published(state, *, cycles=1, accepted=0):
    control = torch.zeros((8, 2, 16), dtype=torch.int64)
    emitted = torch.full((8, 2, 6), -1, dtype=torch.int64)
    current = state
    for i in range(cycles):
        c, e = packet(current, accepted=accepted)
        control[0, i] = torch.tensor(c)
        emitted[0, i] = torch.tensor(e)
        current = NativeState(c[0]+1, c[6], c[9], c[8])
    return control, emitted


@pytest.mark.parametrize("remaining", [1, 6, 7, 12, 23, 24, 2047])
@pytest.mark.parametrize("accepted", [0, 2, 5])
def test_native_actual_emission_and_physical_advance_are_separate(remaining, accepted):
    state = NativeState(9, 511, remaining, 42)
    control, emitted = published(state, accepted=accepted)
    result = validate_packets(control, emitted, state=state, cycles=1)
    assert len(result.token_ids) == min(remaining, accepted+1)
    assert result.state.anchor == state.anchor+accepted+1
    assert result.state.remaining == remaining-len(result.token_ids)
    assert result.accepted == (accepted,)
    # Unused cycle1 has no publication contract; do not inspect allocator residue.
    control[:, 1].fill_(-777)
    emitted[:, 1].fill_(-777)
    assert validate_packets(control, emitted, state=state, cycles=1) == result


@pytest.mark.parametrize("remaining", [7, 12, 23, 24])
def test_two_real_cycles_close_the_predecessor_state(remaining):
    state = NativeState(1, 127, remaining, 5)
    control, emitted = published(state, cycles=2, accepted=5)
    result = validate_packets(control, emitted, state=state, cycles=2)
    assert len(result.token_ids) == min(remaining, 12)
    assert result.advances == (6, 6)
    assert result.state.epoch == 3


@pytest.mark.parametrize("field", [0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15])
def test_corrupt_commit_never_returns_a_state(field):
    state = NativeState(1, 127, 64, 9)
    control, emitted = published(state, accepted=2)
    control[0, 0, field] += 1
    with pytest.raises(ValueError):
        validate_packets(control, emitted, state=state, cycles=1)


def test_nonowner_clear_and_unused_output_contracts():
    state = NativeState(1, 0, 64, 9)
    control, emitted = published(state)
    emitted[0, 0, 5] = 7
    with pytest.raises(ValueError, match="emitted"):
        validate_packets(control, emitted, state=state, cycles=1)
    control, emitted = published(state)
    control[7, 0, 0] = 1
    with pytest.raises(ValueError, match="nonowner"):
        validate_packets(control, emitted, state=state, cycles=1)
    control, emitted = published(state)
    with pytest.raises(ValueError, match="budget"):
        validate_packets(control, emitted, state=NativeState(1, 0, 6, 9), cycles=2)


def test_native_capacity_admission_reserves_final_lookahead():
    validate_native_capacity(prompt_tokens=16383, max_output_tokens=1, physical_capacity=16384)
    validate_native_capacity(prompt_tokens=16314, max_output_tokens=64, physical_capacity=16384)
    with pytest.raises(ValueError, match="lookahead"):
        validate_native_capacity(prompt_tokens=16315, max_output_tokens=64, physical_capacity=16384)


def adapter():
    hosts = allocate_native_hosts(2)
    rows = [{"name": f"input{i}", "shape": [8, 1], "dtype": "int32", "direction": "In"}
            for i in range(150)]
    rows += [{"name": "scratch", "shape": [8, 1], "dtype": "int32", "direction": "Out"},
             {"name": "s_control", "shape": [8, 2, 16], "dtype": "int64", "direction": "Out"},
             {"name": "s_emitted", "shape": [8, 2, 6], "dtype": "int64", "direction": "Out"}]
    def tensor(base):
        return StackedDeviceTensor([DeviceTensor(base+r*256, (1,), torch.int32,
                  buffer=SimpleNamespace(base=base+r*256, size=4, tensor=lambda *a: None)) for r in range(8)],
                  (8, 1), tuple(range(8)))
    common = {r["name"]: tensor(0x10000+i*0x1000) for i, r in enumerate(rows[:150])}
    outputs = [{"scratch": tensor(0x1000000+i*0x1000)} for i in range(2)]
    class Worker:
        def __init__(self):
            self._inherited_host_spans = [((v.data_ptr(), v.data_ptr()+v.numel()*v.element_size()), v)
                                          for h in hosts for v in h.values()]
            self._states = {p: {"param_infos": rows} for p in ("one", "two")}
            self.handle = SimpleNamespace(result=lambda: None)
        def _validate_prepared_dispatch_arg(self, info, value):
            pass
        def submit(self, *args, **kwargs):
            return self.handle
    worker = Worker()
    instance = NativeFrames(worker, {1: "one", 2: "two"}, {1: {"params": rows}, 2: {"params": rows}},
                            common, outputs, hosts, None)
    return instance, hosts, outputs


def test_frame_cannot_be_captured_or_reused_before_consumption():
    instance, hosts, _ = adapter()
    state = NativeState(1, 7, 6, 42)
    instance.submit(0, generation=1, cycles=1)
    with pytest.raises(RuntimeError, match="publication"):
        instance.capture(0, generation=1, state=state)
    with pytest.raises(RuntimeError, match="unavailable"):
        instance.submit(0, generation=1, cycles=1)
    control, emitted = published(state, accepted=5)
    hosts[0]["s_control"].copy_(control)
    hosts[0]["s_emitted"].copy_(emitted)
    instance.complete(0)
    result = instance.capture(0, generation=1, state=state)
    with pytest.raises(RuntimeError, match="owned"):
        instance.release(0)
    instance.consume(0)
    instance.release(0)
    hosts[0]["s_emitted"].fill_(-777)
    assert result.token_ids == tuple(range(100, 106))
    assert instance.idle()
    instance.submit(0, generation=2, cycles=1)
    instance.complete(0)
    with pytest.raises(RuntimeError, match="generation"):
        instance.capture(0, generation=1, state=state)


def test_failed_parent_keeps_frame_and_device_outputs_owned():
    instance, _, outputs = adapter()
    def fail():
        raise RuntimeError("device failure")
    instance.worker.handle.result = fail
    instance.submit(0, generation=1, cycles=1)
    with pytest.raises(RuntimeError, match="device failure"):
        instance.complete(0)
    assert instance.poisoned and instance.frames[0].leased
    assert instance.frames[0].handle is instance.worker.handle
    assert outputs[0]["scratch"] in instance.frames[0].arguments
    with pytest.raises(RuntimeError):
        instance.release(0)
    with pytest.raises(RuntimeError):
        instance.submit(1, generation=1, cycles=1)
