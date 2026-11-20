# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from pypto_serving.config.types import DecodeBatch
from pypto_serving.serving.model_extension import RequestExecutionContext
from pypto_serving.model.deepseek_lowlat.native_runner import NativeModelRunner
from pypto_serving.model.deepseek_lowlat.npu_runner import LowlatModelRunner
from pypto_serving.model.deepseek_lowlat.speculation import NativeState, validate_packets
from pypto_serving.model.deepseek_lowlat.state import LowlatRequestState


def runner():
    instance = NativeModelRunner.__new__(NativeModelRunner)
    LowlatModelRunner.__init__(instance, SimpleNamespace(single=SimpleNamespace(compiled=object())),
                              SimpleNamespace(compiled_context=16384))
    instance.generation = 2
    instance.state = LowlatRequestState("r", 2, 128, 128, 128, 9, "decode")
    instance.cache = SimpleNamespace(tables_for=Mock(return_value=({}, object())))
    instance.transfers = Mock(poisoned=False)
    native_state = NativeState(13, 127, 63, 9)
    control = torch.zeros((8, 2, 16), dtype=torch.int64)
    emitted = torch.full((8, 2, 6), -1, dtype=torch.int64)
    control[0, 0] = torch.tensor([13, 1, 127, 2, 3, 3, 130, 102, 102, 60, 1, 128, 3, 0, 0, 0])
    emitted[0, 0, :3] = torch.tensor([100, 101, 102])
    packet = validate_packets(control, emitted, state=native_state, cycles=1)
    handle = Mock()
    frames = SimpleNamespace(submit=Mock(return_value=handle), complete=Mock(),
                             capture=Mock(return_value=packet), consume=Mock(), release=Mock())
    instance.native = SimpleNamespace(frames=frames, state=native_state, commit=Mock())
    instance.request_contexts = {"r": RequestExecutionContext("r", 128, 64)}
    batch = DecodeBatch(["r"], torch.tensor([9]), None, torch.tensor([129]), block_ids_by_group=[{}])
    return instance, batch, handle


def test_one_cycle_returns_actual_ids_and_commits_after_validated_capture():
    instance, batch, handle = runner()
    result = instance.run_decode(None, batch)
    assert result.accepted_token_ids == [[100, 101, 102]]
    assert result.num_draft_tokens == [5]
    assert instance.state.kv_committed_tokens == 131
    assert instance.state.last_sampled_token == 102
    assert instance.state.pending_handles == []
    instance.cache.tables_for.assert_called_once_with({}, input_end=134)
    instance.transfers.stage_tables.assert_called_once()
    instance.transfers.stage.assert_not_called()
    instance.native.frames.complete.assert_called_once_with(0)
    instance.native.frames.consume.assert_called_once_with(0)
    instance.native.frames.release.assert_called_once_with(0)


@pytest.mark.parametrize("phase", ["submit", "complete", "capture"])
def test_native_failure_retains_request_state_and_accepted_handle(phase):
    instance, batch, handle = runner()
    getattr(instance.native.frames, phase).side_effect = RuntimeError(phase)
    with pytest.raises(RuntimeError, match=phase):
        instance.run_decode(None, batch)
    assert instance.state.kv_committed_tokens == 128 and instance.state.last_sampled_token == 9
    assert instance.state.pending_handles == ([] if phase == "submit" else [handle])
    instance.transfers.poison.assert_called_once()
    instance.native.commit.assert_not_called()
    instance.native.frames.release.assert_not_called()


def test_native_registration_rejects_missing_budget_interleave_and_physical_tail():
    instance, _, _ = runner()
    with pytest.raises(ValueError, match="interleave"):
        instance.register_contexts([RequestExecutionContext("other", 8, 64)])
    instance.state = None
    with pytest.raises(ValueError, match="lookahead"):
        instance.register_contexts([RequestExecutionContext("tail", 16315, 64)])
    assert "tail" not in instance.request_contexts
    with pytest.raises(ValueError, match="context"):
        instance._request(["unknown"], 8)


def test_truncated_final_cycle_keeps_raw_acceptance_in_trace_and_disables_visible_count_metric():
    from test_speculation import published

    instance, batch, _ = runner()
    native_state = NativeState(13, 127, 1, 9)
    control, emitted = published(native_state, accepted=5)
    packet = validate_packets(control, emitted, state=native_state, cycles=1)
    instance.native.state = native_state
    instance.native.frames.capture.return_value = packet
    result = instance.run_decode(None, batch)
    assert result.accepted_token_ids == [[100]] and packet.accepted == (5,)
    assert result.num_draft_tokens is None
    assert instance.state.kv_committed_tokens == 134
