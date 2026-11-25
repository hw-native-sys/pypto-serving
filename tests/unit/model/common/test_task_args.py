# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Unit tests for the TaskArgs L3 argument container.
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Event
from types import SimpleNamespace

import pytest
import torch
from pypto.runtime import DeviceTensor, StackedDeviceTensor

from pypto_serving.config.types import DispatchInvocation
from pypto_serving.model.common.runner import l3_dispatch
from pypto_serving.model.common.runner.buffer_set import (
    ClearPolicy,
    Placement,
    Slot,
    StaticDeviceTensor,
)
from pypto_serving.model.common.runner.task_args import TaskArgs
from pypto_serving.tools.profile import dispatch_invocation_scope


def test_l3_submit_and_wait_keep_the_dispatch_identity(monkeypatch):
    submit_entered = Event()
    next_prepare_entered = Event()
    recorded = {}
    first = DispatchInvocation("replica:1", 1, 1, 1)
    second = DispatchInvocation("replica:2", 2, 0, 1)

    @contextmanager
    def span(name, **metadata):
        recorded[name] = metadata["args"]
        yield

    def submit(*args, **kwargs):
        submit_entered.set()
        assert next_prepare_entered.wait(timeout=5)
        return SimpleNamespace(result=lambda: None)

    def prepare_next():
        assert submit_entered.wait(timeout=5)
        with dispatch_invocation_scope(second):
            next_prepare_entered.set()
            return first_pending.result(timeout=5)

    def submit_first():
        with dispatch_invocation_scope(first):
            return runner._submit_l3(callable_spec)

    monkeypatch.setattr(l3_dispatch, "profile_span", span)
    runner = l3_dispatch.L3DispatchMixin()
    runner._init_l3_dispatch(stacked=False)
    monkeypatch.setattr(runner, "_shared_l3_worker", lambda: SimpleNamespace(submit=submit))
    callable_spec = SimpleNamespace(
        name="fused", aicpu_thread_num=1, block_dim=None, dispatch_args=(), compiled=object(),
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        first_pending = executor.submit(submit_first)
        pending = executor.submit(prepare_next).result(timeout=5)
    with dispatch_invocation_scope(second):
        pending.wait()
    assert recorded["fused.worker_submit"]["invocation_id"] == first.invocation_id
    assert recorded["fused.worker_wait"] == first.as_profile_args()


class _FakeAllocWorker:
    """Fake worker for device-buffer allocation: returns DeviceTensor handles."""

    def __init__(self, fail_on: int | None = None) -> None:
        self.allocations = 0
        self.freed: list[DeviceTensor] = []
        self.fail_on = fail_on

    def alloc_tensor(self, shape, dtype, *, worker_id=0, init=None) -> DeviceTensor:
        self.allocations += 1
        if self.fail_on is not None and self.allocations >= self.fail_on:
            raise RuntimeError("simulated alloc failure")
        return DeviceTensor(0x100000 * self.allocations * (worker_id + 1), tuple(shape), dtype)

    def free_tensor(self, tensor, *, worker_id=0) -> None:
        self.freed.append(tensor)


def _host_slot(name, shape, *, dtype=torch.float32, clear=ClearPolicy.NONE):
    return Slot(name, Placement.HOST_SHARED, dtype, lambda c, s=shape: s, clear=clear)


def _device_slot(name, shape, *, dtype=torch.float32):
    return Slot(name, Placement.DEVICE_RESIDENT, dtype, lambda c, s=shape: s)


def test_build_returns_args_in_registration_order_mixed_kinds():
    ta = TaskArgs(stacked=True)
    scalar = 7
    marker = StaticDeviceTensor(torch.zeros(2).share_memory_())
    shared = torch.ones(3).share_memory_()
    handle = StackedDeviceTensor([DeviceTensor(1, (2,), torch.float32)], (1, 2), (0,))

    ta.add_arg("scalar", scalar)
    ta.add_slot(_host_slot("buf", (2, 2)))
    ta.add_arg("marker", marker)
    ta.add_arg("shared", shared)
    ta.add_arg("handle", handle)
    ta.add_arg("kv_cache", lambda: handle)  # lazy source, resolved at build
    ta.allocate_host_shared(None)

    built = ta.build()
    assert built[:5] == (scalar, ta.tensors["buf"], marker, shared, handle)
    assert built[5] is handle
    assert ta.names == ("scalar", "buf", "marker", "shared", "handle", "kv_cache")

    with pytest.raises(ValueError, match="duplicate arg name"):
        ta.add_arg("buf", 1)


def test_allocate_splits_by_placement_and_is_idempotent():
    worker = _FakeAllocWorker()
    ta = TaskArgs()
    ta.add_slot(_host_slot("host_buf", (2, 3)))
    ta.add_slot(_device_slot("dev_buf", (4, 2, 8)))
    ta.allocate_host_shared(None)
    ta.allocate_device(worker, None)

    assert ta.tensors["host_buf"].is_shared()
    assert tuple(ta.tensors["host_buf"].shape) == (2, 3)
    assert isinstance(ta.tensors["dev_buf"], StackedDeviceTensor)
    assert worker.allocations == 4  # one shard per rank

    # Hot paths may call allocate_* unconditionally: second calls are no-ops.
    host, dev = ta.tensors["host_buf"], ta.tensors["dev_buf"]
    ta.allocate_host_shared(None)
    ta.allocate_device(worker, None)
    assert ta.tensors["host_buf"] is host and ta.tensors["dev_buf"] is dev

    # Host-shared allocation must precede device allocation (fork invariant).
    late = TaskArgs()
    late.add_slot(_device_slot("d", (4, 2)))
    late.allocate_device(worker, None)
    with pytest.raises(RuntimeError, match="before device-resident"):
        late.allocate_host_shared(None)


def test_allocate_device_failure_rolls_back_created_buffers():
    worker = _FakeAllocWorker(fail_on=6)  # fails partway through the 8-shard second slot
    ta = TaskArgs()
    ta.add_slot(_device_slot("first", (4, 2)))  # 4 shards, succeeds
    ta.add_slot(_device_slot("second", (4, 2)))  # fails on shard 2

    with pytest.raises(RuntimeError, match="simulated alloc failure"):
        ta.allocate_device(worker, None)

    assert "first" not in ta.tensors and "second" not in ta.tensors
    assert len(worker.freed) == 5  # the 4 first-slot shards + the 1 leaked second-slot shard

    # A retry starts clean and can succeed.
    worker.fail_on = None
    ta.allocate_device(worker, None)
    assert isinstance(ta.tensors["first"], StackedDeviceTensor)


def test_stage_and_clear_outputs_touch_host_slots_only():
    ta = TaskArgs()
    ta.add_slot(_host_slot("input_ids", (1,), dtype=torch.int64))
    ta.add_slot(_host_slot("hidden", (2, 2), clear=ClearPolicy.ZERO))
    ta.add_slot(_host_slot("ids", (2,), dtype=torch.int64, clear=ClearPolicy.FILL_NEG_ONE))
    ta.add_slot(_device_slot("dev_only", (2, 4)))
    ta.allocate_host_shared(None)
    ta.allocate_device(_FakeAllocWorker(), None)

    ta.stage({"input_ids": torch.tensor([5], dtype=torch.int64)})
    assert torch.equal(ta.tensors["input_ids"], torch.tensor([5], dtype=torch.int64))
    with pytest.raises(KeyError, match="no slot named 'missing'"):
        ta.stage({"missing": torch.zeros(1, dtype=torch.int64)})
    with pytest.raises(TypeError, match="not a host-shared slot"):
        ta.stage({"dev_only": torch.zeros(2)})

    ta.tensors["hidden"].fill_(7.0)
    ta.clear_outputs()
    assert torch.equal(ta.tensors["hidden"], torch.zeros(2, 2))
    assert torch.equal(ta.tensors["ids"], torch.full((2,), -1, dtype=torch.int64))


def test_build_cache_holds_stable_identity_across_in_place_staging():
    ta = TaskArgs()
    ta.add_slot(_host_slot("input_ids", (1,), dtype=torch.int64))
    ta.allocate_host_shared(None)

    first = ta.build(use_cache=True)
    assert ta.build(use_cache=True) is first

    # Per-step metadata is staged in-place: identity stable, contents change.
    ta.stage({"input_ids": torch.tensor([9], dtype=torch.int64)})
    rebuilt = ta.build(use_cache=True)
    assert rebuilt is first and rebuilt[0] is ta.tensors["input_ids"]
    assert ta.tensors["input_ids"][0].item() == 9

    ta.reset()
    assert ta.build(use_cache=True) is not first


def test_close_clears_tensors_and_cache():
    ta = TaskArgs()
    ta.add_slot(_host_slot("buf", (2,)))
    ta.allocate_host_shared(None)
    ta.build(use_cache=True)

    ta.close()

    assert ta.tensors == {}
    with pytest.raises(KeyError):
        ta.build(use_cache=True)
