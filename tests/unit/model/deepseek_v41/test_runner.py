# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU contract tests; these do not emulate lib kernels or claim device coverage."""
from types import SimpleNamespace
import pytest

from pypto_serving.model.deepseek_v41.composite import BuildOptions, CompositeBindings, MissingCompositeInterface
from pypto_serving.model.deepseek_v41.execution_plan import LayerPlan, RankPlacement
from pypto_serving.model.deepseek_v41.npu_runner import V41ModelRunner


def bindings(events, **overrides):
    def allocate(*args):
        events.append("allocate")
        return object(), 12
    values = dict(revision="test-double-only", entries={(p, "swa"): lambda *a: None
                  for p in ("prefill", "decode")}, initialize=lambda *a: None, output=lambda *a: None,
                  allocate=allocate, prepare_weights=lambda *a: None, reset_request=lambda *a: None,
                  wait=lambda *a: events.append("wait"), close=lambda *a: events.append("close"))
    return CompositeBindings(**(values | overrides))


def plan():
    return SimpleNamespace(placement=RankPlacement(0), layers=(LayerPlan(0, "swa", None, None, None),))


def test_missing_entry_before_allocation():
    events = []
    with pytest.raises(MissingCompositeInterface, match="decode/swa"):
        V41ModelRunner(plan(), bindings(events, entries={("prefill", "swa"): lambda *a: None}),
                       device_ids=range(8), runtime=None)
    assert events == []


def test_lifecycle_waits_before_free_and_is_idempotent():
    events = []
    runner = V41ModelRunner(plan(), bindings(events), device_ids=range(8), runtime=None)
    assert runner.preflight() == runner.preflight() == 12
    runner.close()
    runner.close()
    assert events == ["allocate", "wait", "wait", "close"]
    with pytest.raises(RuntimeError, match="closed"):
        runner.preflight()


def test_allocator_receives_build_options():
    options = BuildOptions(pypto_build_dir="worker-7-build", use_compile_cache=True)
    received = []
    runner = V41ModelRunner(
        plan(), bindings([], allocate=lambda *args: (received.append(args[-1]) or object(), 12)),
        device_ids=range(8), runtime=None, build_options=options,
    )
    runner.preflight()
    assert received == [options]
    runner.close()


def test_bad_allocator_result_is_closed():
    events = []
    runner = V41ModelRunner(plan(), bindings(events, allocate=lambda *a: (object(), 0)),
                           device_ids=range(8), runtime=None)
    with pytest.raises(ValueError, match="capacity"):
        runner.preflight()
    assert events == ["wait", "close"]


def test_wait_failure_never_frees_live_buffers():
    events = []
    def wait(*args):
        raise RuntimeError("device completion failed")
    runner = V41ModelRunner(plan(), bindings(events, wait=wait), device_ids=range(8), runtime=None)
    with pytest.raises(RuntimeError, match="completion"):
        runner.preflight()
    assert events == ["allocate"]
    assert runner.resources is not None
    with pytest.raises(RuntimeError, match="initialization failed"):
        runner.preflight()
