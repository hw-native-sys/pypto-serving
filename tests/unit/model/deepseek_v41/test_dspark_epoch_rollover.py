# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check DSpark dispatch rollover against the measured A5 signal bound."""

import sys
from types import SimpleNamespace

import pytest

from pypto_serving.model.deepseek_v41.npu_runner import V41ResidentResources


@pytest.fixture
def dispatch(monkeypatch):
    monkeypatch.setitem(sys.modules, "models.deepseek_v4_1_flash.dspark_drafter",
                        SimpleNamespace(DSPARK_CHUNK_REQUESTS=13, DSPARK_DRAFT_LAYERS=3))
    compiled = object()

    class Worker:
        _reset_persistent_windows = False
        end_epoch = 0
        fail_on_reset = False

        def __init__(self):
            self.calls = []

        def run(self, program, *arguments, config):
            self.calls.append((program, arguments, self._reset_persistent_windows))
            if program is not compiled:
                return
            if self.fail_on_reset and self._reset_persistent_windows:
                raise RuntimeError("injected rollover failure")
            if self._reset_persistent_windows:
                self.end_epoch = 0
            attention_epoch, moe_epoch = arguments[-2:]
            assert attention_epoch == moe_epoch == self.end_epoch + 1
            self.end_epoch = attention_epoch + 2
            assert self.end_epoch * 2 <= 2048, "measured int32 AtomicAdd signal bound exceeded"

    resident = object.__new__(V41ResidentResources)
    resident.closed = resident.failed = False
    resident.worker = Worker()
    resident.run_config = None
    resident._dspark_drafter_epochs = {}
    resident.dspark_programs = {
        "dspark_drafter": SimpleNamespace(compiled=compiled),
        "prepare_target": SimpleNamespace(compiled=object()),
    }
    resident.dspark_param_names = {
        "dspark_drafter": ("head_hidden", "window_cache", "attention_epoch", "moe_epoch"),
        "prepare_target": ("window_cache",),
    }
    arguments = {
        "head_hidden": SimpleNamespace(shape=(4, 1, 5, 5120)),
        "window_cache": object(), "attention_epoch": 1, "moe_epoch": 1,
    }
    return resident, arguments


def test_repeated_dispatch_resets_only_communication_windows(dispatch):
    resident, arguments = dispatch
    for _ in range(700):
        resident.dispatch_dspark("dspark_drafter", arguments)
        assert resident.worker._reset_persistent_windows is False
    calls = resident.worker.calls
    assert any(reset for _, _, reset in calls)
    assert all(values[1] is arguments["window_cache"] for _, values, _ in calls)
    assert arguments["attention_epoch"] == arguments["moe_epoch"] == 1


def test_rollover_failure_restores_worker_policy(dispatch):
    resident, arguments = dispatch
    resident.worker.fail_on_reset = True
    with pytest.raises(RuntimeError, match="injected rollover failure"):
        for _ in range(342):
            resident.dispatch_dspark("dspark_drafter", arguments)
        pytest.fail("communication windows were never reset")
    assert resident.worker._reset_persistent_windows is False


def test_other_stage_preserves_drafter_epoch_and_window_policy(dispatch):
    resident, arguments = dispatch
    resident.dispatch_dspark("dspark_drafter", arguments)
    epochs = dict(resident._dspark_drafter_epochs)
    resident.dispatch_dspark("prepare_target", {"window_cache": arguments["window_cache"]})
    assert resident._dspark_drafter_epochs == epochs
    assert resident.worker.calls[-1][-1] is False
