# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from pathlib import Path

from pypto import passes

from pypto_serving.model.qwen.npu_executor import (
    Qwen314BPyptoExecutor as PyptoExecutor,
    _load_qwen3_14b_contract,
)


ROOT = Path(__file__).resolve().parents[1]
QWEN3_KERNEL_DIR = ROOT / "pypto-lib" / "models" / "qwen3" / "14b"


def test_qwen_compile_uses_current_distributed_config_interface(monkeypatch):
    import pypto.ir.distributed_compiled_program as distributed_program

    class StrictDistributedConfig:
        """Mirror runtimes that retain AICPU tuning but remove block_dim."""

        def __init__(self, *, device_ids, num_sub_workers, aicpu_thread_num):
            self.device_ids = device_ids
            self.num_sub_workers = num_sub_workers
            self.aicpu_thread_num = aicpu_thread_num

    captured = {}

    class FakeJitFunction:
        def compile(self, *args, config):
            captured["args"] = args
            captured["config"] = config
            return distributed_program.DistributedCompiledProgram.__new__(
                distributed_program.DistributedCompiledProgram
            )

    monkeypatch.setattr(distributed_program, "DistributedConfig", StrictDistributedConfig)
    executor = PyptoExecutor(device_ids=(3,))

    samples = (object(), object())
    callable_spec = executor._compile_jit_fwd_callable(
        "fake",
        FakeJitFunction(),
        *samples,
    )

    run_config = captured["config"]
    assert captured["args"] == samples
    assert run_config.memory_planner is None
    assert not hasattr(run_config, "block_dim")
    assert run_config.distributed_config.device_ids == [3]
    assert run_config.distributed_config.num_sub_workers == 0
    assert run_config.distributed_config.aicpu_thread_num == 4
    assert callable_spec.aicpu_thread_num == 4


def test_qwen_compile_threads_use_cache_to_compiler():
    """_compile_jit_fwd_callable forwards use_compile_cache to the compiler."""
    captured: dict[str, object] = {}

    class _FakeCompiler:
        def compile(
            self,
            name,
            jit_fn,
            *compile_args,
            use_cache=False,
            run_config_overrides=None,
        ):
            captured["name"] = name
            captured["compile_args"] = compile_args
            captured["use_cache"] = use_cache
            captured["run_config_overrides"] = run_config_overrides
            return "compiled"

    executor = PyptoExecutor(device_ids=(3,), use_compile_cache=True)
    executor._compiler = _FakeCompiler()

    sample = object()
    callable_spec = executor._compile_jit_fwd_callable(
        "fake",
        object(),
        sample,
        memory_planner=passes.MemoryPlanner.PTOAS,
    )

    assert callable_spec == "compiled"
    assert captured["compile_args"] == (sample,)
    assert captured["use_cache"] is True
    assert captured["run_config_overrides"] == {"memory_planner": passes.MemoryPlanner.PTOAS}
    assert captured["name"] == "fake"


def test_qwen_executor_loads_complete_contract_from_pypto_lib():
    """The serving adapter resolves all required Qwen stages through the registry."""
    contract = _load_qwen3_14b_contract()

    assert contract.model.family == "qwen3"
    assert contract.model.variant == "14b"
    assert contract.execution == {
        "prefill": ("prefill",),
        "decode": ("decode",),
        "topk_select": ("topk_select",),
    }
    assert {"prefill", "decode", "topk_select"} <= contract.kernels.keys()
    assert contract.kernels["topk_select"].runtime_args_builder is not None
