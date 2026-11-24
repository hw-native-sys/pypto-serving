# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DSpark bootstrap must retain compact Decoder taps across prefill chunks."""

from types import SimpleNamespace

import pytest
import torch

from pypto_serving.model.deepseek_v41.npu_runner import V41ResidentResources
from pypto_serving.model.deepseek_v41.task_args import ForwardStep, RequestSlice, plan_decoder_replay_batch


@pytest.mark.parametrize("bounded", [False, True])
def test_dspark_retains_terminal_window_from_actual_decoder_layout(bounded):
    runner = V41ResidentResources.__new__(V41ResidentResources)
    source = torch.full((2, 256, 6), float("nan"), dtype=torch.bfloat16)
    retained = torch.full((2, 128, 6), -1, dtype=torch.bfloat16)
    runner.prefill_host = SimpleNamespace(fields={"dspark_target_hidden": source})
    runner.dspark_host_resources = SimpleNamespace(
        layout=SimpleNamespace(dp_size=2),
        prefill_workspace={"prefill_target_hidden": retained},
    )
    # An empty Decoder stage, then 120 rows, then 8 rows. Unused output rows
    # deliberately contain NaNs so reading the old full-chunk offsets fails.
    for start, count in ((0, 256), (256, 256), (512, 8)):
        request = RequestSlice("r", 0, 0, start, (0,) * count, 520, {})
        step = ForwardStep("prefill", (request,), 1)
        plan = plan_decoder_replay_batch(step.requests, 2)
        runner.prefill_host.decoder_replay_plan = plan if bounded else None
        positions = plan.request_plans[0].positions if bounded else tuple(range(start, start + count))
        source.fill_(float("nan"))
        # BF16 exactly represents these values; encode the final 128 positions
        # as 0..127 to make the retained row order unambiguous.
        values = torch.tensor([p - 392 for p in positions], dtype=torch.bfloat16)
        source[0, :len(positions)] = values[:, None]
        runner._retain_dspark_prefill(step)
        if bounded and start == 0:
            assert runner._dspark_prefill_progress[0] == ("r", 256, 0)
    assert runner._dspark_prefill_progress[0] == ("r", 520, 128)
    assert torch.equal(retained[0], torch.arange(128, dtype=torch.bfloat16)[:, None].expand(128, 6))
