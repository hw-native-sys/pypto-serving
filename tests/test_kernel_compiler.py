# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit tests for the shared :class:`KernelCompiler`.

These exercise the compile-core wiring (RunConfig/DistributedConfig building,
profile gating, type-check, L3Callable wrapping) without an NPU or the real
pypto compiler: a fake ``jit_fn`` records the ``RunConfig`` it is handed and
returns a ``spec``-typed stand-in for ``DistributedCompiledProgram``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pypto.ir.distributed_compiled_program import DistributedCompiledProgram
from pypto.runtime import RunConfig
from pypto_serving.model.common.compiler.compiler import KernelCompiler
from pypto_serving.model.common.compiler.l3_callable import L3Callable
from pypto_serving.model.common.executor.utils import build_pypto_run_config


def _base_run_config() -> RunConfig:
    """Build the base RunConfig an executor hands to KernelCompiler."""
    return build_pypto_run_config(platform="a2a3sim", device_ids=(0, 1))


class _FakeJitFn:
    """Record the args/config handed to ``compile`` and return a canned program."""

    def __init__(self, program: object) -> None:
        self._program = program
        self.last_config: RunConfig | None = None
        self.last_args: tuple[object, ...] = ()
        self.compile_calls = 0

    def compile(self, *args: object, config: RunConfig, **compile_kwargs: object) -> object:
        self.last_config = config
        self.last_args = args
        self.compile_calls += 1
        return self._program


def _make_compiler(**overrides: object) -> KernelCompiler:
    kwargs: dict[str, object] = dict(run_config=_base_run_config())
    kwargs.update(overrides)
    return KernelCompiler(**kwargs)  # type: ignore[arg-type]


def test_compile_threads_enable_scope_stats_into_run_config() -> None:
    """``enable_scope_stats`` and ``codegen_only`` reach the compile RunConfig."""
    jit_fn = _FakeJitFn(MagicMock(spec=DistributedCompiledProgram))
    compiler = _make_compiler(enable_scope_stats=True)

    result = compiler.compile("prefill", jit_fn, use_cache=True)

    assert jit_fn.last_config is not None
    assert jit_fn.last_config.enable_scope_stats is True
    assert jit_fn.last_config.codegen_only is True
    assert isinstance(result, L3Callable)
    assert result.compiled is jit_fn._program
    assert result.name == "prefill"


def test_compile_defaults_disable_scope_stats() -> None:
    """Without an explicit flag, scope stats stay off (qwen's behaviour)."""
    jit_fn = _FakeJitFn(MagicMock(spec=DistributedCompiledProgram))
    compiler = _make_compiler()

    compiler.compile("decode", jit_fn)

    assert jit_fn.last_config.enable_scope_stats is False


def test_compile_carries_aicpu_thread_num_from_run_config_to_callable() -> None:
    """``aicpu_thread_num`` from the base RunConfig reaches the L3Callable."""
    run_config = build_pypto_run_config(
        platform="a2a3sim", device_ids=(0, 1), aicpu_thread_num=8
    )
    jit_fn = _FakeJitFn(MagicMock(spec=DistributedCompiledProgram))
    compiler = KernelCompiler(run_config=run_config)

    result = compiler.compile("prefill", jit_fn, use_cache=True)

    distributed_config = jit_fn.last_config.distributed_config
    assert distributed_config.aicpu_thread_num == 8
    assert distributed_config.device_ids == [0, 1]
    assert distributed_config.num_sub_workers == 0
    assert result.aicpu_thread_num == 8


def test_compile_forwards_runtime_scalar_kwargs_to_jit_fn() -> None:
    """Extra compile kwargs (e.g. ``name=pl.RUNTIME``) reach ``jit_fn.compile``."""
    from pypto.language import RUNTIME

    jit_fn = _FakeJitFn(MagicMock(spec=DistributedCompiledProgram))
    compiler = _make_compiler()

    compiler.compile("mtp_prefill", jit_fn, num_tokens=RUNTIME)

    assert jit_fn.compile_calls == 1


def test_compile_forwards_contract_sample_args_to_jit_fn() -> None:
    """External Contract sample tensors reach the generic HOST wrapper."""
    jit_fn = _FakeJitFn(MagicMock(spec=DistributedCompiledProgram))
    compiler = _make_compiler()
    sample_args = (object(), object())

    compiler.compile("contract-stage", jit_fn, *sample_args)

    assert jit_fn.last_args == sample_args


def test_compile_applies_per_stage_run_config_overrides() -> None:
    """A stage can select its required memory planner without changing siblings."""
    from pypto import passes

    jit_fn = _FakeJitFn(MagicMock(spec=DistributedCompiledProgram))
    compiler = _make_compiler()

    compiler.compile(
        "topk-select",
        jit_fn,
        run_config_overrides={"memory_planner": passes.MemoryPlanner.PTOAS},
    )

    assert jit_fn.last_config.memory_planner is passes.MemoryPlanner.PTOAS


def test_compile_raises_on_non_distributed_compiled_program() -> None:
    """A result that is not a DistributedCompiledProgram raises TypeError."""
    jit_fn = _FakeJitFn(object())  # not a DistributedCompiledProgram
    compiler = _make_compiler()

    with pytest.raises(TypeError, match="DistributedCompiledProgram"):
        compiler.compile("prefill", jit_fn)


# --- on-disk cache (load) ----------------------------------------------------


def _seed_slot(cache_dir: Path, name: str) -> Path:
    """Create a cache slot that looks like it holds an assembled program."""
    slot = cache_dir / name
    slot.mkdir(parents=True)
    (slot / "distributed_meta.json").write_text("{}")
    return slot


def test_cache_miss_compiles_into_per_kernel_slot(tmp_path: Path) -> None:
    """An empty slot falls through to JIT, writing straight into cache_dir/<name>."""
    jit_fn = _FakeJitFn(MagicMock(spec=DistributedCompiledProgram))
    compiler = _make_compiler(cache_dir=str(tmp_path))

    result = compiler.compile("prefill", jit_fn, use_cache=True)

    assert jit_fn.compile_calls == 1
    assert result.compiled is jit_fn._program
    # pypto is pointed straight at the per-kernel slot (no separate copy step).
    assert jit_fn.last_config.save_kernels is True
    assert jit_fn.last_config.save_kernels_dir == str(tmp_path / "prefill")


def test_cache_hit_reloads_program_and_skips_jit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A populated slot is reloaded via from_dir and JIT is skipped."""
    _seed_slot(tmp_path, "prefill")
    cached = MagicMock(spec=DistributedCompiledProgram)
    monkeypatch.setattr(
        DistributedCompiledProgram,
        "from_dir",
        lambda slot, **kw: cached,
    )

    jit_fn = _FakeJitFn(MagicMock(spec=DistributedCompiledProgram))
    compiler = _make_compiler(cache_dir=str(tmp_path))

    result = compiler.compile("prefill", jit_fn, use_cache=True)

    assert jit_fn.compile_calls == 0
    assert result.compiled is cached
    assert result.name == "prefill"
    assert result.aicpu_thread_num == 4


def test_cache_disabled_recompiles_even_when_slot_populated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``use_cache=False`` ignores a populated slot and recompiles."""
    _seed_slot(tmp_path, "prefill")
    monkeypatch.setattr(
        DistributedCompiledProgram,
        "from_dir",
        lambda slot, **kw: MagicMock(spec=DistributedCompiledProgram),
    )
    jit_fn = _FakeJitFn(MagicMock(spec=DistributedCompiledProgram))
    compiler = _make_compiler(cache_dir=str(tmp_path))

    result = compiler.compile("prefill", jit_fn, use_cache=False)

    assert jit_fn.compile_calls == 1
    assert result.compiled is jit_fn._program
