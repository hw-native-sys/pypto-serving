# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU dispatch-contract tests; these do not execute NPU kernels."""
from types import SimpleNamespace
from unittest.mock import Mock
import sys

import pytest
import torch

from pypto_serving.model.deepseek_v41.composite import LayerState
from pypto_serving.model.deepseek_v41.swa_segment import (
    ATTENTION_ARGS, MOE_ARGS, SegmentTopology, SwaSegment, make_segment_worker,
)
from pypto_serving.model.deepseek_v41.prefill_segment import PREFILL_ARGUMENTS, PrefillSegment


@pytest.mark.parametrize("counts,expected", [
    ([64, 64], (16,) * 8), ([17, 0], (16, 1, 0, 0, 0, 0, 0, 0)),
    ([0, 63], (0, 0, 0, 0, 16, 16, 16, 15)),
])
def test_local_counts(counts, expected):
    group, local = SegmentTopology().counts(counts)
    assert local == expected
    assert group == tuple(n for n in counts for _ in range(4))
    assert sum(local) == sum(counts)


@pytest.mark.parametrize("counts", [[65, 0], [-1, 16], [True, 0], [16]])
def test_reject_counts(counts):
    with pytest.raises(ValueError):
        SegmentTopology().counts(counts)


def fixture(monkeypatch):
    # PyPTO is optional for CPU serving tests. The dispatcher shape/type guard
    # is independently exercised in the device smoke, not mocked as evidence.
    monkeypatch.setattr(SwaSegment, "_check_state", lambda self, state: None)
    monkeypatch.setattr(SwaSegment, "_check_weights", lambda self, weights: None)
    def buffer(address):
        return SimpleNamespace(worker_ids=(0, 1), shards=(
            SimpleNamespace(data_ptr=address), SimpleNamespace(data_ptr=address)))
    topology = SegmentTopology(tp=1, dp=2)
    worker = Mock()
    programs = (SimpleNamespace(compiled="attention"), SimpleNamespace(compiled="moe"))
    segment = SwaSegment(worker, programs, topology,
                         torch.zeros(2, 1, dtype=torch.int32).share_memory_(),
                         torch.zeros(2, dtype=torch.int32).share_memory_(), None)
    state = LayerState(buffer(1), buffer(2), "tp_local_token")
    a, m = dict.fromkeys(ATTENTION_ARGS), dict.fromkeys(MOE_ARGS)
    a.update(output=buffer(3), next_pre_mix=buffer(4))
    m.update(x_next=buffer(5), next_pre_mix=buffer(6))
    return segment, state, a, m


def test_device_handle_handoff_and_next_layer(monkeypatch):
    segment, state, a, m = fixture(monkeypatch)
    first = segment.run_layer(state, a, m, group_counts=[16, 3])
    calls = segment.worker.run.call_args_list
    assert calls[0].args[1] is state.residual
    assert calls[1].args[1] is a["output"]
    assert calls[1].args[2] is a["next_pre_mix"]
    assert first.residual is m["x_next"]
    assert first.pre_mix is m["next_pre_mix"]
    # Ping-pong outputs, retaining the returned handles as next-layer inputs.
    m.update(x_next=state.residual, next_pre_mix=state.pre_mix)
    segment.run_layer(first, a, m, group_counts=[16, 3])
    assert segment.worker.run.call_args_list[2].args[1] is first.residual
    assert segment.worker.run.call_args_list[2].args[-1].value == 2
    assert segment.moe_counts.tolist() == [16, 3]


def test_switching_attention_modes_keeps_per_program_epochs(monkeypatch):
    base, state, a, m = fixture(monkeypatch)
    programs = {mode: SimpleNamespace(compiled=object()) for mode in ("swa", "c2a_full", "c1a_reindex")}
    segment = PrefillSegment(base.worker, programs, base.programs[1], base.topology,
                             base.attention_counts, base.moe_counts, None)
    # The scratch/output allocations are owned and distinct for each half-layer.
    original = state
    sequence = ("swa", "swa", "c2a_full", "c1a_reindex", "c2a_full", "swa")
    for mode in sequence:
        args = dict.fromkeys(PREFILL_ARGUMENTS[mode])
        args.update(output=a["output"], x_hc_out=a["output"], next_pre_mix=a["next_pre_mix"])
        result = segment.run_layer(state, args, m, group_counts=[16, 3], mode=mode)
        last = segment.worker.run.call_args_list[-2:]
        assert last[0].args[0] is programs[mode].compiled
        assert last[0].args[1] is state.residual
        assert last[0].args[2] is state.pre_mix
        assert last[1].args[1] is a["output"]
        assert result.residual is m["x_next"]
        m.update(x_next=state.residual, next_pre_mix=state.pre_mix)
        state = result
    calls = segment.worker.run.call_args_list
    assert [c.args[-1].value for c in calls[::2]] == [1, 2, 1, 1, 2, 3]
    assert [c.args[-1].value for c in calls[1::2]] == [1, 2, 3, 4, 5, 6]
    assert original.layout == "tp_local_token"


def test_missing_compressed_metadata_fails_before_mutating_any_cache(monkeypatch):
    base, state, a, m = fixture(monkeypatch)
    segment = PrefillSegment(base.worker, {"c2a_full": base.programs[0]}, base.programs[1],
                             base.topology, base.attention_counts, base.moe_counts, None)
    args = dict.fromkeys(PREFILL_ARGUMENTS["c2a_full"])
    args.update(x_hc_out=a["output"], next_pre_mix=a["next_pre_mix"])
    del args["state_block_table"]
    with pytest.raises(KeyError, match="state_block_table"):
        segment.run_layer(state, args, m, group_counts=[16, 0], mode="c2a_full")
    with pytest.raises(ValueError, match="not compiled"):
        segment.run_layer(state, args, m, group_counts=[16, 0], mode="c1a_full")
    assert not segment.worker.run.called
    assert segment._epoch == 0 and segment._attention_epochs == {}


def test_failed_dispatch_poisoned(monkeypatch):
    segment, state, a, m = fixture(monkeypatch)
    segment.worker.run.side_effect = RuntimeError("device failure")
    with pytest.raises(RuntimeError, match="device failure"):
        segment.run_layer(state, a, m, group_counts=[16, 16])
    with pytest.raises(RuntimeError, match="close this worker"):
        segment.run_layer(state, a, m, group_counts=[16, 16])
    assert segment.worker.run.call_count == 1


def test_validate_both_stages_before_dispatch(monkeypatch):
    segment, state, a, m = fixture(monkeypatch)
    del m["routed_w3"]
    with pytest.raises(KeyError):
        segment.run_layer(state, a, m, group_counts=[16, 16])
    segment.worker.run.assert_not_called()


def test_alias_rejected_before_dispatch(monkeypatch):
    segment, state, a, m = fixture(monkeypatch)
    m["x_next"] = state.residual
    with pytest.raises(ValueError, match="alias"):
        segment.run_layer(state, a, m, group_counts=[16, 16])
    segment.worker.run.assert_not_called()


def test_retained_window_policy(monkeypatch):
    constructor = Mock()
    monkeypatch.setitem(sys.modules, "pypto.runtime", SimpleNamespace(DistributedWorker=constructor))
    programs = [SimpleNamespace(compiled="a"), SimpleNamespace(compiled="m")]
    make_segment_worker(programs, "config", ["source"])
    constructor.assert_called_once_with(
        ["a", "m"], config="config", persistent=True,
        reset_persistent_windows=False, inherited_host_tensors=["source"],
    )


def test_attention_epoch_cannot_overflow_int32(monkeypatch):
    segment, state, a, m = fixture(monkeypatch)
    segment._epoch = (2**31 - 1) // 2
    with pytest.raises(OverflowError):
        segment.run_layer(state, a, m, group_counts=[16, 16])
    segment.worker.run.assert_not_called()
