# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""V4.1 final-output calls use the selected lib's window lifetime."""

from types import SimpleNamespace

import pypto_serving.model.deepseek_v41.npu_runner as runner
from pypto_serving.model.deepseek_v41.npu_runner import FinalOutput, LayerState


CURRENT_NAMES = (
    "x_hc", "pre_mix", "final_norm_weight", "lm_head_weight",
    "logit_row_indices", "logits", "sampled_ids",
)
LEGACY_NAMES = (
    "x_hc", "pre_mix", "norm_weight", "head_weight", "logit_row_indices",
    "normed", "logits", "sampled_ids", "done_epoch",
)
def _exercise(names):
    calls = []
    worker = SimpleNamespace(_persistent=True, _reset_persistent_windows=False)

    def run(*args, **kwargs):
        calls.append((dict(zip(names, args[1:])), worker._reset_persistent_windows))

    worker.run = run
    output = FinalOutput(worker, SimpleNamespace(compiled=object()), names, object())
    state = LayerState(object(), object(), "tp_local_token")
    arguments = {name: object() for name in names if name not in {"x_hc", "pre_mix", "done_epoch"}}
    for _ in range(2):
        assert output.run(state, arguments) == (arguments["logits"], arguments["sampled_ids"])
        assert worker._reset_persistent_windows is False
    return output, state, arguments, calls


def test_current_lib_final_output_reuses_persistent_windows():
    output, state, arguments, calls = _exercise(CURRENT_NAMES)
    assert output.fresh_windows
    assert [reset for _, reset in calls] == [False, False]
    for bound, _ in calls:
        assert bound["x_hc"] is state.residual
        assert bound["pre_mix"] is state.pre_mix
        assert bound["final_norm_weight"] is arguments["final_norm_weight"]
        assert bound["lm_head_weight"] is arguments["lm_head_weight"]


def test_diagnostic_final_output_keeps_incrementing_epoch():
    output, _, _, calls = _exercise(LEGACY_NAMES)
    assert not output.fresh_windows
    assert output.epoch == 2
    assert [bound["done_epoch"].value for bound, _ in calls] == [1, 2]


def test_current_lib_entry_selected(monkeypatch):
    entry = object()
    lib_head = SimpleNamespace(l3_final_output=entry, TP_SIZE=1, DP_SIZE=8, WORLD_SIZE=8)
    monkeypatch.setattr(runner, "load_segment_modules", lambda *args: None)
    monkeypatch.setattr(runner.importlib, "import_module", lambda name: lib_head)
    topology = SimpleNamespace(tp=1, dp=8, world=8)
    assert runner.make_final_output_program(object(), topology) is entry
