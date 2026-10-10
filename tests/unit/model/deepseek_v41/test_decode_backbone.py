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

from pypto_serving.model.deepseek_v41.composite import LayerState
from pypto_serving.model.deepseek_v41.decode_backbone import DecodeBackbone
from pypto_serving.model.deepseek_v41.swa_segment import SegmentTopology


class RecordingWorker:
    def __init__(self, *, fail=False):
        self.calls = []
        self.fail = fail

    def run(self, program, *args, config):
        self.calls.append((program, args, config))
        if self.fail:
            raise RuntimeError("device dispatch failed")


def test_decode_backbone_binds_resident_state_and_rejects_missing_cache():
    worker = RecordingWorker()
    program = SimpleNamespace(compiled=object())
    decoder = DecodeBackbone(
        worker, program, ("x_hc", "pre_mix", "window_cache", "attention_num_tokens"), "config",
    )
    state = LayerState(object(), object(), "tp_local_token")

    with pytest.raises(ValueError, match="window_cache"):
        decoder.run(state, {}, active_tokens=1)
    assert worker.calls == []

    assert decoder.run(state, {"window_cache": "resident-cache"}, active_tokens=7) is state
    _, args, config = worker.calls[0]
    assert args[:3] == (state.residual, state.pre_mix, "resident-cache")
    assert args[3].value == 7
    assert config == "config"


def test_decode_backbone_rejects_invalid_batch_and_poisoned_worker():
    worker = RecordingWorker(fail=True)
    decoder = DecodeBackbone(worker, SimpleNamespace(compiled="program"),
                             ("x_hc", "pre_mix", "attention_num_tokens"), "config")
    state = LayerState(object(), object(), "tp_local_token")

    with pytest.raises(ValueError, match="compact active batch"):
        decoder.run(state, {}, active_tokens=193)
    with pytest.raises(ValueError, match="TP-local"):
        decoder.run(LayerState(object(), object()), {}, active_tokens=1)
    with pytest.raises(ValueError, match="not argument buffers"):
        decoder.run(state, {"pre_mix": object()}, active_tokens=1)
    with pytest.raises(RuntimeError, match="device dispatch failed"):
        decoder.run(state, {}, active_tokens=1)
    with pytest.raises(RuntimeError, match="close the worker"):
        decoder.run(state, {}, active_tokens=1)
    assert len(worker.calls) == 1


@pytest.mark.parametrize("tp,expected", [(2, 96), (4, 48), (8, 32)])
def test_decode_topology_uses_lib_padded_moe_rows(tp, expected):
    topology = SegmentTopology.for_decode(tp=tp, dp=8 // tp)
    assert topology.local_capacity == expected
    assert topology.capacity >= 192
