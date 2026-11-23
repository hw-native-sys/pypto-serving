# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Real HOST JIT, passes and distributed codegen; only external PTOAS is a leaf."""

import ast
import dataclasses
import json
from pathlib import Path

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto import CacheConfig
from pypto.pypto_core import passes
from pypto.runtime import RunConfig
from pypto_serving.model.common.compiler.compiler import KernelCompiler
from pypto_serving.model.common.executor.utils import build_pypto_run_config

RUNTIMES = ("tensormap_and_ringbuffer", "host_build_graph")


def make_host():
    @pl.jit
    def chip(data: pld.DistributedTensor[[64], pl.FP32]):
        return data

    @pl.jit.host
    def host():
        data_buf = pld.alloc_window_buffer(64 * pl.FP32.get_byte())
        signal_buf = pld.alloc_window_buffer(pld.world_size() * pl.INT32.get_byte())
        data = pld.window(data_buf, [64], dtype=pl.FP32)
        signal = pld.window(signal_buf, [pld.world_size()], dtype=pl.INT32)
        for rank in pl.range(pld.world_size()):
            chip(data, device=rank)
        pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
        return 0

    return host


@pytest.fixture
def cpu_codegen(monkeypatch, tmp_path):
    # A local output cwd keeps the normal JIT allocation path without forcing
    # an explicit output-dir bypass, so the TMR/HBG/TMR test exercises caching.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PYPTO_PROG_BUILD_DIR", raising=False)
    monkeypatch.setattr("pypto.jit.decorator.find_ptoas_binary", lambda: Path("cpu-ptoas-leaf"))
    monkeypatch.setattr(
        "pypto.backend.pto_backend._compile_pto_module",
        lambda *_args, **_kwargs: '__global__ AICORE void cpu_external_tool_leaf() {}',
    )


def config(runtime, **changes):
    base = build_pypto_run_config(platform="a2a3", device_ids=(0, 1), runtime=runtime)
    return dataclasses.replace(base, cache_config=CacheConfig(enabled=False), **changes)


def assert_artifacts(compiled, runtime):
    root = Path(compiled.output_dir)
    metadata = json.loads((root / "distributed_meta.json").read_text())
    assert metadata["distributed_config"]["runtime"] == runtime
    paths = sorted((root / "next_levels").rglob("kernel_config.py"))
    assert len(paths) >= 2, "require both user chip and builtin collective configs"
    for path in paths:
        tree = ast.parse(path.read_text())
        values = [ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "RUNTIME_CONFIG" for t in node.targets)]
        assert len(values) == 1
        assert values[0]["runtime"] == runtime, f"{path}: actual codegen runtime differs from {runtime}"
    return root


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_real_jit_codegen_runtime(cpu_codegen, runtime):
    assert passes.PassContext.current() is None
    compiled = KernelCompiler(run_config=config(runtime)).compile("tiny", make_host()).compiled
    assert_artifacts(compiled, runtime)
    assert passes.PassContext.current() is None


def test_runtime_cache_cycle(cpu_codegen):
    host = make_host()
    results = [KernelCompiler(run_config=config(rt)).compile("tiny", host).compiled
               for rt in (RUNTIMES[0], RUNTIMES[1], RUNTIMES[0])]
    for result, rt in zip(results, (RUNTIMES[0], RUNTIMES[1], RUNTIMES[0])):
        assert_artifacts(result, rt)
    assert results[0] is results[2]
    assert results[0] is not results[1]


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_ambient_options_restore_and_conflicts(cpu_codegen, runtime):
    observed = []
    instrument = passes.CallbackInstrument(before_pass=lambda *_: observed.append(
        (passes.PassContext.current().get_memory_planner(), passes.PassContext.current().get_runtime())
    ))
    kind = passes.runtime_kind_from_name(runtime)
    with passes.PassContext([instrument], runtime=kind, memory_planner=passes.MemoryPlanner.PTOAS,
                            diagnostic_phase=passes.DiagnosticPhase.NONE, enable_buffer_ir=False) as outer:
        compiled = KernelCompiler(run_config=config(runtime)).compile("tiny", make_host()).compiled
        assert_artifacts(compiled, runtime)
        assert observed and all(v == (passes.MemoryPlanner.PTOAS, kind) for v in observed)
        current = passes.PassContext.current()
        assert current.get_runtime() == outer.get_runtime()
        assert current.get_memory_planner() == outer.get_memory_planner()
        assert current.get_diagnostic_phase() == outer.get_diagnostic_phase()
        assert current.get_enable_buffer_ir() == outer.get_enable_buffer_ir()
        assert [i.get_name() for i in current.get_instruments()] == [instrument.get_name()]
        assert not outer.get_enable_buffer_ir()
        assert outer.get_diagnostic_phase() == passes.DiagnosticPhase.NONE
        for option in ({"memory_planner": passes.MemoryPlanner.PYPTO},
                       {"diagnostic_phase": passes.DiagnosticPhase.NONE}):
            with pytest.raises(RuntimeError, match="while a PassContext is already active"):
                KernelCompiler(run_config=config(runtime, **option)).compile("tiny", make_host())
            assert passes.PassContext.current().get_runtime() == kind
            assert passes.PassContext.current().get_memory_planner() == passes.MemoryPlanner.PTOAS
        other = RUNTIMES[1] if runtime == RUNTIMES[0] else RUNTIMES[0]
        with pytest.raises(ValueError, match="conflicts with configured chip runtime"):
            KernelCompiler(run_config=config(other)).compile("tiny", make_host())
        assert passes.PassContext.current().get_runtime() == kind
        assert passes.PassContext.current().get_instruments()[0].get_name() == instrument.get_name()
    assert passes.PassContext.current() is None


@pytest.mark.parametrize("empty_mask", [False, True])
@pytest.mark.parametrize("phase", [passes.DiagnosticPhase.NONE, passes.get_default_diagnostic_phase()])
def test_explicit_diagnostics_keep_recompilation(cpu_codegen, phase, empty_mask):
    host = make_host()
    mask = passes.DiagnosticCheckSet() if empty_mask else None
    cfg = config(RUNTIMES[1], memory_planner=passes.MemoryPlanner.PTOAS, diagnostic_phase=phase, disabled_diagnostics=mask)
    observed = []
    class ObservedJit:
        def compile(self, **kwargs):
            ctx = passes.PassContext.current()
            observed.append((ctx.get_memory_planner(), ctx.get_diagnostic_phase()))
            if mask is not None:
                assert ctx.get_disabled_diagnostics() == mask
            return host.compile(**kwargs)
    results = [KernelCompiler(run_config=cfg).compile("tiny", ObservedJit()).compiled for _ in range(2)]
    assert observed == [(passes.MemoryPlanner.PTOAS, phase)] * 2
    for result in results:
        assert_artifacts(result, RUNTIMES[1])
    assert results[0] is not results[1], "explicit diagnostic request must still run passes"
    assert cfg.memory_planner == passes.MemoryPlanner.PTOAS and cfg.diagnostic_phase == phase
    assert passes.PassContext.current() is None


def test_context_restored_after_compile_error(cpu_codegen):
    class Failure:
        def compile(self, **kwargs):
            assert passes.PassContext.current().get_runtime() == passes.RuntimeKind.HOST_BUILD_GRAPH
            raise RuntimeError("compile sentinel")
    with pytest.raises(RuntimeError, match="compile sentinel"):
        KernelCompiler(run_config=config(RUNTIMES[1])).compile("tiny", Failure())
    assert passes.PassContext.current() is None


# The fixed older compiler has no software-pipeline option. CI's current
# public RunConfig schema adds true/false cases without inventing an old API.
SOFTWARE_PIPELINES = [None]
if "enable_software_pipeline" in {field.name for field in dataclasses.fields(RunConfig)}:
    SOFTWARE_PIPELINES += [False, True]


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
@pytest.mark.parametrize("software_pipeline", SOFTWARE_PIPELINES)
def test_explicit_planner_context_reaches_real_passes(cpu_codegen, monkeypatch, planner, software_pipeline):
    import importlib

    module = importlib.import_module("pypto.ir.compile")
    original = module._run_pass_pipeline
    seen = []
    def observe_pipeline(*args, **kwargs):
        ctx = passes.PassContext.current()
        seen.append(ctx.get_memory_planner())
        assert ctx.get_memory_planner() == planner
        if software_pipeline is not None:
            assert ctx.get_enable_software_pipeline() is software_pipeline
            assert ctx.get_enable_buffer_ir() is (not software_pipeline)
        # Observe then execute the original full pipeline, never fabricate IR.
        return original(*args, **kwargs)
    monkeypatch.setattr(module, "_run_pass_pipeline", observe_pipeline)
    changes = {"memory_planner": planner}
    if software_pipeline is not None:
        changes["enable_software_pipeline"] = software_pipeline
    cfg = config(RUNTIMES[1], **changes)
    assert passes.PassContext.current() is None
    if planner == passes.MemoryPlanner.PTOAS and software_pipeline is True:
        # Keep this matrix cell: the public context constructor rejects this
        # combination before any passes can run. Do not treat it as codegen success.
        with pytest.raises(ValueError, match="enable_software_pipeline requires memory_planner=PYPTO"):
            KernelCompiler(run_config=cfg).compile("tiny", make_host())
        assert seen == []
    else:
        result = KernelCompiler(run_config=cfg).compile("tiny", make_host()).compiled
        assert_artifacts(result, RUNTIMES[1])
        assert seen
    assert cfg.memory_planner == planner
    if software_pipeline is not None:
        assert cfg.enable_software_pipeline is software_pipeline
    assert passes.PassContext.current() is None
