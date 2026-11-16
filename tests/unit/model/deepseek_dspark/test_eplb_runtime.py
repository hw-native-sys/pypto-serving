# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime output ordering and maintenance publication failure boundaries."""

from __future__ import annotations

from types import SimpleNamespace
from threading import Event

import numpy as np
import pytest
import torch

from pypto_serving.config.types import EplbConfig
from pypto_serving.model.deepseek_dspark.eplb.controller import EplbController
from pypto_serving.model.deepseek_dspark.eplb import controller as controller_module
from pypto_serving.model.deepseek_dspark.eplb.policy import plan_rebalance
from pypto_serving.model.deepseek_dspark.eplb.runtime import EplbRuntime
from pypto_serving.model.deepseek_dspark.eplb.state import Placement
from pypto_serving.model.deepseek_dspark.eplb.transfer import ExpertBankLayout


def runtime_fixture():
    base = Placement.initial(layers=1, logical_experts=8, ranks=4, redundant_experts=4)
    config = EplbConfig(enabled=True, num_redundant_experts=4, window_size=2, step_interval=2,
                        max_migration_bytes=100_000)
    runtime = EplbRuntime(base, config)
    layout = ExpertBankLayout(4, 1, 3, (1,) * 6, (1, 4, 1, 4, 1, 4))
    runtime.controller = EplbController(base, config, expert_bytes=layout.expert_bytes)
    runtime.bank_layout = SimpleNamespace(layer_views=lambda *_: tuple(range(6)), controls=layout.controls,
                                          expert_bytes=layout.expert_bytes)
    return runtime, layout


def test_reclaim_order_follows_submission_and_copies_before_slot_reuse():
    runtime, _ = runtime_fixture()
    recorded = []
    original = runtime.controller.record

    def record(sample):
        recorded.append(sample)
        return original(sample)

    runtime.controller.record = record
    outputs = [torch.zeros((4, 1, 16), dtype=torch.int32) for _ in range(2)]
    outputs[0][:, :, :3] = 2
    outputs[1][:, :, :3] = 3
    controls = [torch.empty((4, 2), dtype=torch.int32) for _ in range(2)]
    first, second = [runtime.begin("decode", output, control)
                     for output, control in zip(outputs, controls)]
    with pytest.raises(RuntimeError, match="every submitted output"):
        runtime.assert_quiescent()
    runtime.complete(second)
    assert not recorded
    outputs[1].zero_()
    runtime.complete(first)
    outputs[0].fill_(-1)
    runtime.complete(first)
    assert [sample.dispatch_id for sample in recorded] == [1, 2]
    assert [int(sample.logical_counts().sum()) for sample in recorded] == [24, 36]
    runtime.assert_quiescent()
    assert controls[0][:, 0].tolist() == [1] * 4
    assert controls[1][:, 0].tolist() == [2] * 4


def test_failed_and_unrecorded_dispatches_do_not_stall_the_ordered_window():
    runtime, _ = runtime_fixture()
    output = torch.zeros((4, 1, 16), dtype=torch.int32)
    control = torch.empty((4, 2), dtype=torch.int32)
    first = runtime.begin("decode", output, control)
    second = runtime.begin("decode", output, control, record=False)
    third = runtime.begin("prefill", output, control)
    runtime.complete(third)
    runtime.complete(first, success=False)
    runtime.complete(second)
    runtime.assert_quiescent()
    assert runtime.take_ready_plan() is None


def test_corrupt_statistics_stop_future_dispatch_and_maintenance():
    runtime, _ = runtime_fixture()
    output = torch.zeros((4, 1, 16), dtype=torch.int32)
    control = torch.empty((4, 2), dtype=torch.int32)
    context = runtime.begin("decode", output, control)
    output[0, 0, 0] = -1
    with pytest.raises(ValueError, match="physical load sample"):
        runtime.complete(context)
    for operation in (runtime.take_ready_plan, runtime.assert_quiescent,
                      lambda: runtime.begin("decode", output, control)):
        with pytest.raises(RuntimeError, match="stopped"):
            operation()
    with pytest.raises(RuntimeError, match="fatal"):
        runtime.controller.current()


def ready_window(runtime):
    output = torch.zeros((4, 1, 16), dtype=torch.int32)
    mapping = runtime.placement.physical_to_logical[0].reshape(4, 3)
    for rank in range(4):
        for slot in range(3):
            output[rank, 0, slot] = 100 if mapping[rank, slot] == 0 else 80 if mapping[rank, slot] == 1 else 1
    control = torch.empty((4, 2), dtype=torch.int32)
    for _ in range(2):
        runtime.complete(runtime.begin("decode", output, control))
    assert runtime.controller.has_ready_history()
    return output, control


def test_cpu_planning_never_blocks_dispatch_or_reclaim_and_keeps_one_job(monkeypatch):
    runtime, _ = runtime_fixture()
    output, control = ready_window(runtime)
    entered, release = Event(), Event()
    calls = []
    original = controller_module.plan_rebalance

    def blocked(*args, **kwargs):
        calls.append(args[0])
        entered.set()
        assert release.wait(3)
        return original(*args, **kwargs)

    monkeypatch.setattr(controller_module, "plan_rebalance", blocked)
    try:
        assert runtime.take_ready_plan() is None
        assert entered.wait(3)
        for _ in range(2):
            runtime.complete(runtime.begin("decode", output, control))
            assert runtime.take_ready_plan() is None
        assert len(calls) == 1
        runtime.assert_quiescent()
        release.set()
        runtime._planning.result(timeout=3)
        plan = runtime.take_ready_plan()
        assert plan is not None and plan.base_generation == 0
        assert runtime.controller.has_ready_history()
    finally:
        release.set()
        runtime.close()


def test_async_planner_discards_a_candidate_after_an_intervening_commit():
    runtime, _ = runtime_fixture()
    ready_window(runtime)
    try:
        candidate = runtime.take_ready_plan()
        if candidate is None:
            candidate, _ = runtime._planning.result(timeout=3)
        assert candidate is not None
        runtime.controller.committed(candidate)
        runtime.placement = candidate.target
        assert runtime.take_ready_plan() is None
        assert runtime.placement.generation == 1
        runtime.check_healthy()
    finally:
        runtime.close()


def test_async_planner_failure_stops_future_submissions(monkeypatch):
    runtime, _ = runtime_fixture()
    output, control = ready_window(runtime)
    entered, release = Event(), Event()

    def fail(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        raise ValueError("invalid planning history")

    monkeypatch.setattr(controller_module, "plan_rebalance", fail)
    try:
        assert runtime.take_ready_plan() is None
        assert entered.wait(3)
        release.set()
        with pytest.raises(ValueError, match="invalid planning history"):
            runtime._planning.result(timeout=3)
        with pytest.raises(ValueError, match="invalid planning history"):
            runtime.take_ready_plan()
        with pytest.raises(RuntimeError, match="stopped"):
            runtime.begin("decode", output, control)
    finally:
        release.set()
        runtime.close()


@pytest.mark.parametrize("failure", [None, "transfer", "upload"])
def test_maintenance_publishes_only_after_all_rank_table_uploads(failure):
    runtime, layout = runtime_fixture()
    history = np.full((2, 1, 8), 10)
    history[:, :, 0] = 100
    history[:, :, 1] = 80
    plan = plan_rebalance(runtime.placement, history, runtime.config, expert_bytes=layout.expert_bytes)
    assert plan is not None
    runtime.tables = {
        name: SimpleNamespace(shards=[SimpleNamespace(data_ptr=100 + rank) for rank in range(4)])
        for name in ("eplb_mapping", "eplb_replicas")
    }
    events = []

    class Worker:
        def run(self, *args, **kwargs):
            events.append("install")
            if failure == "transfer":
                raise RuntimeError("uncertain partial installation")

        def copy_to(self, *args, **kwargs):
            events.append("upload")
            assert runtime.placement.generation == 0
            if failure == "upload":
                raise RuntimeError("rank table upload failed")

    if failure is None:
        runtime.apply(Worker(), {}, SimpleNamespace(compiled=object()), plan)
        assert runtime.placement is plan.target
        assert events == ["install"] + ["upload"] * 8
        assert runtime.controller.current().generation == 1
    else:
        with pytest.raises(RuntimeError):
            runtime.apply(Worker(), {}, SimpleNamespace(compiled=object()), plan)
        assert runtime.placement.generation == 0
        with pytest.raises(RuntimeError, match="stopped"):
            runtime.take_ready_plan()
        with pytest.raises(RuntimeError, match="fatal"):
            runtime.controller.current()
