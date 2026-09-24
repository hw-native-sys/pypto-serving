# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Recording adapters test orchestration only, not model numerical correctness."""
from types import SimpleNamespace
import pytest
import torch

from pypto_serving.model.deepseek_v41.composite import CompositeBindings, LayerState
from pypto_serving.model.deepseek_v41.execution_plan import LayerPlan, RankPlacement
from pypto_serving.model.deepseek_v41.npu_runner import V41ModelRunner
from pypto_serving.model.deepseek_v41.request_state import ForwardStep, RequestSlice


def make_runner(*, layout="tp_local_token", entry=None, output=None):
    events = []
    layers = tuple(LayerPlan(i, mode, None if i == 0 else 1, None if i == 0 else 1, None)
                   for i, mode in enumerate(("swa", "c2a_full", "c2a_reuse")))
    def initialize(embeddings, step, resources):
        events.append(("initialize", step.positions))
        return LayerState(object(), object(), layout)
    def call(layer, state, step, resources, weights):
        events.append(("layer", layer.layer_id, step.phase, layer.kv_source, weights))
        return LayerState(object(), object(), layout)
    def prepare(plans, layer, resources):
        assert [p.placement.rank for p in plans] == list(range(8))
        events.append(("weights", layer.layer_id))
        return layer.layer_id
    bindings = CompositeBindings(revision="recording-test-adapter", input_layout=layout, output_layout=layout,
        entries={(phase, layer.mode): entry or call for phase in ("prefill", "decode") for layer in layers},
        initialize=initialize, output=output or (lambda *a: None), allocate=lambda *a: (object(), 8),
        prepare_weights=prepare, reset_request=lambda *a: events.append(("reset", a[1])),
        wait=lambda *a: events.append(("wait",)), close=lambda *a: events.append(("close",)))
    config = SimpleNamespace(hidden_size=4, vocab_size=16, max_position_embeddings=128)
    plan = SimpleNamespace(placement=RankPlacement(0), layers=layers, weights=SimpleNamespace(config=config))
    plan.for_rank = lambda rank: SimpleNamespace(placement=RankPlacement(rank))
    runtime = SimpleNamespace(max_batch_size=2, max_seq_len=128)
    runner = V41ModelRunner(plan, bindings, device_ids=(9, 3, 8, 4, 1, 7, 2, 0), runtime=runtime)
    runner.preflight()
    return runner, events


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_all_layers_keep_state_and_logical_rank_mapping(phase):
    runner, events = make_runner()
    step = ForwardStep(phase, (RequestSlice("a", 1, 0, 7, (2,), 8, {}),), 4)
    state = runner._run_layers(step, torch.ones(1, 4, dtype=torch.bfloat16))
    assert state.layout == "tp_local_token"
    assert [e for e in events if e[0] == "layer"] == [
        ("layer", 0, phase, None, 0), ("layer", 1, phase, 1, 1), ("layer", 2, phase, 1, 2)]
    # A collective completion separates every layer from weight/window reuse.
    for i, event in enumerate(events):
        if event[0] == "layer":
            assert events[i + 1] == ("wait",)


def test_wrong_layout_fails_before_next_layer():
    runner, events = make_runner(entry=lambda *a: LayerState(object(), object(), "tp_replicated"))
    step = ForwardStep("prefill", (RequestSlice("a", 0, 0, 0, (2,), 1, {}),), 1)
    with pytest.raises(ValueError, match="layout"):
        runner._run_layers(step, torch.ones(1, 4, dtype=torch.bfloat16))
    assert [e for e in events if e[0] == "weights"] == [("weights", 0)]


def test_output_uses_shared_greedy_sampler_and_feeds_decode():
    from pypto_serving.config.types import SamplingParams
    from pypto_serving.model.common.executor.sampler import Sampler
    def head(state, step, resources):
        logits = torch.full((len(step.requests), 16), -10.0)
        for row, request in enumerate(step.requests):
            logits[row, (request.token_ids[-1] + 1) % 16] = 10.0
        return logits
    runner, events = make_runner(output=head)
    ledger = runner._request_ledger()
    step = ledger.begin_prefill([("a", 0, 0, (2, 3), 2, {})])
    result = runner._run_transaction(step, torch.ones(2, 4, dtype=torch.bfloat16))
    sampler, generated = Sampler(), []
    for position in range(2, 6):
        token = sampler.sample(result.logits[0], SamplingParams(temperature=0.0, top_p=1.0), "a")
        generated.append(token)
        step = ledger.begin_decode([("a", 0, position, token, {})])
        result = runner._run_transaction(step, torch.ones(1, 4, dtype=torch.bfloat16))
    assert generated == [4, 5, 6, 7]
    assert ledger.owners["a"].length == 6
    runner.release_finished_requests(["a"])
    assert ledger.owners == {} and ("reset", "a") in events


def test_bad_output_invalidates_state_before_sampling():
    runner, events = make_runner(output=lambda *a: torch.full((1, 16), float("nan")))
    ledger = runner._request_ledger()
    step = ledger.begin_prefill([("a", 0, 0, (2,), 1, {"window": (4,)})])
    with pytest.raises(ValueError, match="non-finite"):
        runner._run_transaction(step, torch.ones(1, 4, dtype=torch.bfloat16))
    assert ledger.owners == {} and ("reset", "a") in events


def test_result_does_not_alias_reused_adapter_output():
    scratch = torch.arange(16, dtype=torch.float32).view(1, 16)
    runner, _ = make_runner(output=lambda *a: scratch)
    step = ForwardStep("decode", (RequestSlice("a", 0, 0, 4, (3,), 4, {}),), 1)
    result = runner._finish_step(LayerState(object(), object(), "tp_local_token"), step)
    scratch.zero_()
    assert result.logits[0, -1].item() == 15
