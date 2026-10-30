# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import pytest

from pypto_serving.model.deepseek_v41.dspark_runtime import (
    DECODE_STAGE_ORDER,
    DSparkDecodeDispatch,
    DSparkLoweredInputs,
    DSparkPrefillDispatch,
    DSPARK_SPECULATIVE_TOKENS,
    PREFILL_STAGE_ORDER,
    DSparkStageRunner,
    V41DSparkRuntime,
)


class _DispatchLog:
    def __init__(self):
        self.calls = []

    def dispatch_dspark(self, stage, arguments):
        self.calls.append((stage, arguments))


def _stages(names):
    return [(name, {"stage": name}) for name in names]


def test_dspark_prefill_and_decode_keep_lib_stage_order():
    dispatch = _DispatchLog()
    runtime = V41DSparkRuntime(dispatch, max_seq_len=128)
    runtime.bootstrap(
        "request-0", slot=2, generation=7, prompt_length=16, anchor=15,
        stages=_stages(PREFILL_STAGE_ORDER),
    )
    runtime.decode(
        ["request-0"],
        stages=_stages(DECODE_STAGE_ORDER),
        accepted_counts=[1],
        last_sampled=[42],
        draft_ids=[[0, 1, 0, 1, 0]],
    )

    assert [name for name, _ in dispatch.calls] == [
        *PREFILL_STAGE_ORDER,
        *DECODE_STAGE_ORDER,
    ]
    state = runtime.states.get("request-0")
    assert state.anchor == 16
    assert state.draft_ids == (0, 1, 0, 1, 0)


def test_dspark_rejects_a_reordered_stage_without_dispatch():
    dispatch = _DispatchLog()
    runtime = V41DSparkRuntime(dispatch, max_seq_len=128)
    with pytest.raises(ValueError, match="lib order"):
        runtime.bootstrap(
            "request-0", slot=0, generation=1, prompt_length=8, anchor=7,
            stages=_stages(("dspark_drafter", *PREFILL_STAGE_ORDER[1:])),
        )
    assert dispatch.calls == []


def test_dspark_requires_five_drafts_and_matching_acceptance_state():
    dispatch = _DispatchLog()
    runtime = V41DSparkRuntime(dispatch, max_seq_len=128)
    runtime.bootstrap(
        "request-0", slot=0, generation=1, prompt_length=8, anchor=7,
        stages=_stages(PREFILL_STAGE_ORDER),
    )
    with pytest.raises(ValueError, match="exactly five"):
        runtime.decode(
            ["request-0"], stages=_stages(DECODE_STAGE_ORDER),
            accepted_counts=[1], last_sampled=[42], draft_ids=[[1, 2]],
        )
    assert DSPARK_SPECULATIVE_TOKENS == 5


def test_dspark_validates_acceptance_before_dispatch():
    dispatch = _DispatchLog()
    runtime = V41DSparkRuntime(dispatch, max_seq_len=128)
    runtime.bootstrap(
        "request-0", slot=0, generation=1, prompt_length=8, anchor=7,
        stages=_stages(PREFILL_STAGE_ORDER),
    )
    before = len(dispatch.calls)
    with pytest.raises(ValueError, match="nonnegative token"):
        runtime.decode(
            ["request-0"], stages=_stages(DECODE_STAGE_ORDER),
            accepted_counts=[1], last_sampled=[-1], draft_ids=[[1, 2, 3, 4, 5]],
        )
    assert len(dispatch.calls) == before
    state = runtime.states.get("request-0")
    assert state.anchor == 7
    assert state.draft_ids == ()


def test_dspark_bootstrap_failure_releases_reserved_request():
    dispatch = _DispatchLog()
    runtime = V41DSparkRuntime(dispatch, max_seq_len=16)
    with pytest.raises(ValueError, match="position limit"):
        runtime.bootstrap(
            "request-0", slot=0, generation=1, prompt_length=17, anchor=16,
            stages=_stages(PREFILL_STAGE_ORDER),
        )
    assert "request-0" not in runtime.states.requests
    state = runtime.bootstrap(
        "request-0", slot=0, generation=2, prompt_length=8, anchor=7,
        stages=_stages(PREFILL_STAGE_ORDER),
    )
    assert state.valid


def test_dspark_batch_hooks_keep_optional_decode_state_optional():
    prefill = DSparkPrefillDispatch.from_batch(type("Batch", (), {
        "dspark_stages": {
            "request_id": "request-0", "slot": 2, "generation": 7,
            "prompt_length": 16, "anchor": 15,
            "stages": _stages(PREFILL_STAGE_ORDER),
        },
    })())
    assert prefill.request_id == "request-0"
    assert prefill.anchor == 15

    decode = DSparkDecodeDispatch.from_batch(type("Batch", (), {
        "dspark_stages": {
            "request_ids": ("request-0",), "stages": _stages(DECODE_STAGE_ORDER),
        },
    })())
    assert decode.request_ids == ("request-0",)
    assert decode.accepted_counts is None
    assert decode.draft_ids is None


def test_dspark_apply_decode_outputs_updates_state_without_redispatch():
    dispatch = _DispatchLog()
    runtime = V41DSparkRuntime(dispatch, max_seq_len=128)
    runtime.bootstrap(
        "request-0", slot=0, generation=1, prompt_length=8, anchor=7,
        stages=_stages(PREFILL_STAGE_ORDER),
    )
    before = len(dispatch.calls)
    runtime.apply_decode_outputs(
        ["request-0"], accepted_counts=[1], last_sampled=[42],
        draft_ids=[[0, 1, 0, 1, 0]],
    )
    assert len(dispatch.calls) == before
    state = runtime.states.get("request-0")
    assert state.anchor == 8
    assert state.draft_ids == (0, 1, 0, 1, 0)


def test_dspark_batch_hook_rejects_missing_required_fields():
    batch = type("Batch", (), {"dspark_stages": {"request_ids": ("request-0",)}})()
    with pytest.raises(ValueError, match="missing fields: stages"):
        DSparkDecodeDispatch.from_batch(batch)


def test_dspark_batch_hook_validates_complete_lowered_bundle():
    lowered = DSparkLoweredInputs.from_mapping({
        "state": object(), "workspace": {"accepted_ids": object()},
        "target_model": {}, "draft_model": {}, "markov_model": {},
        "cache_tables": {}, "rope_tables": {}, "attention_epoch": 3,
    })
    dispatch = DSparkDecodeDispatch.from_batch(type("Batch", (), {
        "dspark_stages": {
            "request_ids": ("request-0",), "stages": (), "lowered": lowered,
            "request_ranks": (2,),
        },
    })())
    assert dispatch.lowered is lowered
    assert dispatch.request_ranks == (2,)

    with pytest.raises(ValueError, match="missing: workspace"):
        DSparkLoweredInputs.from_mapping({"state": object()})
    with pytest.raises(TypeError, match="markov_model"):
        DSparkLoweredInputs.from_mapping({
            "state": object(), "workspace": {}, "markov_model": object(),
        })


def test_dspark_record_bootstrap_updates_state_without_redispatch():
    dispatch = _DispatchLog()
    runtime = V41DSparkRuntime(dispatch, max_seq_len=128)
    state = runtime.record_bootstrap(
        "request-0", slot=0, generation=1, prompt_length=8, anchor=8,
        draft_ids=[0, 1, 0, 1, 0],
    )
    assert state.valid
    assert state.anchor == 8
    assert state.draft_ids == (0, 1, 0, 1, 0)
    assert dispatch.calls == []


def test_dspark_stage_runner_rejects_unknown_phase_before_dispatch():
    dispatch = _DispatchLog()
    with pytest.raises(ValueError, match="phase must be"):
        DSparkStageRunner(dispatch).run([], phase="warmup")
    assert dispatch.calls == []
