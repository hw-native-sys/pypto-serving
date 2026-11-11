# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from pathlib import Path
from types import SimpleNamespace

import pytest

from pypto_serving.model.qwen.npu_executor import Qwen314BPyptoExecutor as PyptoExecutor


ROOT = Path(__file__).resolve().parents[1]
QWEN3_KERNEL_DIR = ROOT / "pypto-lib" / "models" / "qwen3" / "14b"


def test_qwen_compile_uses_current_distributed_config_interface(monkeypatch):
    import pypto.ir.distributed_compiled_program as distributed_program

    class StrictDistributedConfig:
        """Mirror runtimes that retain AICPU tuning but remove block_dim."""

        def __init__(self, *, device_ids, num_sub_workers, aicpu_thread_num, runtime):
            self.runtime = runtime
            self.device_ids = device_ids
            self.num_sub_workers = num_sub_workers
            self.aicpu_thread_num = aicpu_thread_num

    captured = {}

    class FakeJitFunction:
        def compile(self, *args, config):
            captured["config"] = config
            return distributed_program.DistributedCompiledProgram.__new__(
                distributed_program.DistributedCompiledProgram
            )

    monkeypatch.setattr(distributed_program, "DistributedConfig", StrictDistributedConfig)
    executor = PyptoExecutor(device_ids=(3,))

    callable_spec = executor._compile_jit_fwd_callable(
        "fake",
        FakeJitFunction(),
    )

    run_config = captured["config"]
    assert not hasattr(run_config, "block_dim")
    assert run_config.distributed_config.device_ids == [3]
    assert run_config.distributed_config.num_sub_workers == 0
    assert run_config.distributed_config.aicpu_thread_num == 4
    assert callable_spec.aicpu_thread_num == 4


def test_qwen_compile_threads_delegate_to_compiler():
    """_compile_jit_fwd_callable compiles through the shared KernelCompiler."""
    captured: dict[str, object] = {}

    class _FakeCompiler:
        def compile(self, name, jit_fn):
            captured["name"] = name
            return "compiled"

    executor = PyptoExecutor(device_ids=(3,))
    executor._compiler = _FakeCompiler()

    callable_spec = executor._compile_jit_fwd_callable("fake", object())

    assert callable_spec == "compiled"
    assert captured["name"] == "fake"


@pytest.mark.parametrize("runtime", ["tensormap_and_ringbuffer", "host_build_graph"])
def test_loaded_model_selects_compiler_before_weight_work(monkeypatch, runtime):
    from pypto.ir.distributed_compiled_program import DistributedCompiledProgram
    from pypto.pypto_core import passes
    from pypto_serving.config.types import RuntimeConfig

    executor = PyptoExecutor(device_ids=(3,))
    def stop_before_weights():
        raise RuntimeError("before weight work")
    monkeypatch.setattr("pypto_serving.model.qwen.npu_executor._find_pypto_lib_qwen14b_dir", stop_before_weights)
    with pytest.raises(RuntimeError, match="before weight work"):
        executor._compile_model(SimpleNamespace(runtime=RuntimeConfig(chip_runtime=runtime)))
    captured = {}
    class JitLeaf:
        def compile(self, *, config):
            captured["config"] = config
            captured["runtime"] = passes.runtime_kind_to_name(passes.PassContext.current().get_runtime())
            return DistributedCompiledProgram.__new__(DistributedCompiledProgram)
    executor._compile_jit_fwd_callable("selection_barrier", JitLeaf())
    assert captured["config"].distributed_config.runtime == runtime
    assert captured["config"].distributed_config.device_ids == [3]
    assert captured["runtime"] == runtime
