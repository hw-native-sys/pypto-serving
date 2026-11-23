# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Serving-side V4.1 DSpark request-state tests."""

import pytest

from pypto_serving.serving.sampling.capabilities import sampling_capabilities
from pypto_serving.model.deepseek_v41.dspark_runtime import V41DSparkRuntime


class _Dispatch:
    def __init__(self):
        self.stages = []

    def dispatch_dspark(self, stage, arguments):
        self.stages.append((stage, arguments))


def test_stochastic_k5_state_accepts_only_the_target_token():
    dispatch = _Dispatch()
    runtime = V41DSparkRuntime(dispatch, max_seq_len=128)
    state = runtime.record_bootstrap(
        "request", slot=0, generation=1, prompt_length=8, anchor=8,
        draft_ids=range(5), anchor_token=17, speculative=False,
    )

    assert state.speculative is False
    assert state.draft_ids == ()
    with pytest.raises(ValueError, match="exactly one target token"):
        state.validate_accept(2, 19)

    runtime.decode(
        ("request",), stages=(("decode", {}),),
        accepted_counts=(1,), last_sampled=(19,), draft_ids=((),),
    )
    assert state.anchor == 9
    assert state.draft_ids == ()


def test_v41_k5_admits_stochastic_requests_for_target_only_sampling():
    capabilities = sampling_capabilities("PyptoDeepSeekV41Executor", 5)

    assert capabilities.greedy_only is False
    assert capabilities.honor_min_tokens is False
