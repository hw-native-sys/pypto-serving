# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
from simpler.task_interface import TensorArgType

from examples.model.qwen3_14b.runner import npu_runner
from examples.model.qwen3_14b.runner.npu_runner import Qwen314BModelRunner
from examples.model.qwen3_14b.runner.npu_runner import _CompiledKernels
from examples.model.qwen3_14b.runner.npu_runner import _L2Callable


@dataclass
class _ParamInfo:
    name: str
    shape: tuple[int, ...] | None
    direction: object | None = None


class _FakeOrchestrator:
    def __init__(self, worker: "_FakeL3Worker") -> None:
        self._worker = worker
        self.ops: list[tuple[Any, ...]] = []

    def malloc(self, worker_id: int, size: int) -> int:
        ptr = self._worker.next_ptr
        self._worker.next_ptr += int(size) + 64
        self.ops.append(("malloc", worker_id, size, ptr))
        return ptr

    def copy_to(self, worker_id: int, dst: int, src: int, size: int) -> None:
        self.ops.append(("copy_to", worker_id, dst, src, size))

    def copy_from(self, worker_id: int, dst: int, src: int, size: int) -> None:
        self.ops.append(("copy_from", worker_id, dst, src, size))

    def submit_next_level(self, callable_id: int, args: Any, config: Any, worker: int = -1) -> None:
        self.ops.append(("submit_next_level", callable_id, args, config.block_dim, worker))


class _FakeL3Worker:
    instances: list["_FakeL3Worker"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.level = int(kwargs["level"])
        self.initialized = False
        self.closed = False
        self.registered: list[object] = []
        self.runs: list[list[tuple[Any, ...]]] = []
        self.freed: list[int] = []
        self.next_ptr = 0x100000
        _FakeL3Worker.instances.append(self)

    def register(self, callable_obj: object) -> int:
        self.registered.append(callable_obj)
        return len(self.registered) - 1

    def init(self) -> None:
        self.initialized = True

    def run(self, callable_obj: object, args: Any = None, config: Any = None) -> object:
        orchestrator = _FakeOrchestrator(self)
        callable_obj(orchestrator, args, config)
        self.runs.append(orchestrator.ops)
        return SimpleNamespace(host_wall_us=1, device_wall_us=0)

    def submit_next_level(
        self,
        callable_id: int,
        args: Any,
        config: Any = None,
        *,
        worker_id: int = 0,
        orchestrator: _FakeOrchestrator,
    ) -> None:
        orchestrator.submit_next_level(callable_id, args, config, worker=worker_id)

    def malloc(self, nbytes: int, *, worker_id: int = 0, orchestrator: _FakeOrchestrator) -> int:
        return orchestrator.malloc(worker_id=worker_id, size=nbytes)

    def copy_to(
        self,
        dst: int,
        src: int,
        nbytes: int,
        *,
        worker_id: int = 0,
        orchestrator: _FakeOrchestrator,
    ) -> None:
        orchestrator.copy_to(worker_id=worker_id, dst=dst, src=src, size=nbytes)

    def copy_from(
        self,
        dst: int,
        src: int,
        nbytes: int,
        *,
        worker_id: int = 0,
        orchestrator: _FakeOrchestrator,
    ) -> None:
        orchestrator.copy_from(worker_id=worker_id, dst=dst, src=src, size=nbytes)

    def free(self, _ptr: int) -> None:
        self.freed.append(_ptr)

    def close(self) -> None:
        self.initialized = False
        self.closed = True

    def discard_l3_children(self) -> None:
        self.initialized = False
        self.closed = True


def _callable(runtime_name: str, params: tuple[_ParamInfo, ...]) -> _L2Callable:
    return _L2Callable(
        chip_callable=object(),
        runtime_name=runtime_name,
        block_dim=24,
        aicpu_thread_num=1,
        param_infos=params,
    )


def _compiled(callable_spec: _L2Callable) -> _CompiledKernels:
    return _CompiledKernels(
        prefill=callable_spec,
        decode=callable_spec,
        final_rms=None,
        lm_head=None,
        final_norm_weight=torch.ones(1, 4),
        rope_cos=torch.zeros(1, 4),
        rope_sin=torch.zeros(1, 4),
        padded_vocab=8,
        padded_lm_head_weight=torch.zeros(8, 4),
        layers=[],
        decode_weights={},
        decode_logits_buffer=torch.zeros(1, 8),
    )


def _runner(compiled: _CompiledKernels) -> Qwen314BModelRunner:
    return Qwen314BModelRunner(
        model_id="model",
        compiled=compiled,
        platform="a2a3",
        device_id=3,
        save_kernels_dir=None,
        l3_trace=False,
    )


def test_non_l3_program_runs_through_l3_worker(monkeypatch):
    _FakeL3Worker.instances.clear()
    monkeypatch.setattr(npu_runner, "LlmWorker", _FakeL3Worker)
    params = (
        _ParamInfo("weight", (2, 2)),
        _ParamInfo("out", (2, 2)),
    )
    callable_spec = _callable("prefill", params)
    runner = _runner(_compiled(callable_spec))
    weight = torch.ones(2, 2, dtype=torch.float32)
    out = torch.zeros(2, 2, dtype=torch.float32).share_memory_()

    runner._run_l2_program(callable_spec, runner._l2_child_tensor(callable_spec.runtime_name, weight), out)

    worker = _FakeL3Worker.instances[0]
    assert worker.kwargs["level"] == 3
    assert worker.kwargs["device_ids"] == [3]
    assert worker.kwargs["num_sub_workers"] == 0
    assert worker.closed
    assert callable_spec.chip_callable in worker.registered
    assert len(worker.runs) == 1
    assert [op[0] for op in worker.runs[0]] == ["malloc", "copy_to", "submit_next_level"]
    assert worker.runs[0][-1][2].tag(0) == TensorArgType.INPUT
    assert worker.runs[0][-1][2].tag(1) == TensorArgType.OUTPUT_EXISTING


def test_full_kv_host_tensors_are_inout_without_extra_copyback(monkeypatch):
    _FakeL3Worker.instances.clear()
    monkeypatch.setattr(npu_runner, "LlmWorker", _FakeL3Worker)
    params = (
        _ParamInfo("k_cache", (4, 2)),
        _ParamInfo("v_cache", (4, 2)),
        _ParamInfo("out", (1, 4)),
    )
    callable_spec = _callable("decode", params)
    runner = _runner(_compiled(callable_spec))
    k_cache = torch.zeros(4, 2, dtype=torch.bfloat16)
    v_cache = torch.zeros(4, 2, dtype=torch.bfloat16)
    out = torch.zeros(1, 4, dtype=torch.float32).share_memory_()

    runner._run_l2_program(
        callable_spec,
        k_cache,
        v_cache,
        out,
    )

    worker = _FakeL3Worker.instances[0]
    assert len(worker.runs) == 1
    submit_args = worker.runs[0][-1][2]
    assert submit_args.tag(0) == TensorArgType.INOUT
    assert submit_args.tag(1) == TensorArgType.INOUT
    assert submit_args.tag(2) == TensorArgType.OUTPUT_EXISTING


def test_switching_runtime_closes_previous_l3_worker(monkeypatch):
    _FakeL3Worker.instances.clear()
    monkeypatch.setattr(npu_runner, "LlmWorker", _FakeL3Worker)
    params = (
        _ParamInfo("weight", (2, 2)),
        _ParamInfo("out", (2, 2)),
    )
    prefill = _callable("prefill-rt", params)
    decode = _callable("decode-rt", params)
    compiled = _compiled(prefill)
    compiled.decode = decode
    runner = _runner(compiled)
    weight = torch.ones(2, 2, dtype=torch.float32)
    out = torch.zeros(2, 2, dtype=torch.float32).share_memory_()

    runner._run_l2_program(prefill, runner._l2_child_tensor("prefill-rt", weight), out)
    prefill_worker = _FakeL3Worker.instances[0]
    assert prefill_worker.closed

    runner._run_l2_program(decode, runner._l2_child_tensor("decode-rt", weight), out)

    assert prefill_worker.closed
    assert not prefill_worker.freed
    assert len(_FakeL3Worker.instances) == 2
    assert _FakeL3Worker.instances[1].kwargs["runtime"] == "decode-rt"
