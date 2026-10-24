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
cache-policy resolution, type-check, L3Callable wrapping) without an NPU or the
real pypto compiler: a fake ``jit_fn`` records the ``RunConfig`` it is handed
and returns a ``spec``-typed stand-in for ``DistributedCompiledProgram``.
"""

from __future__ import annotations

from unittest.mock import MagicMock
import warnings

import pytest
from pypto import CacheConfig

from pypto.ir.distributed_compiled_program import DistributedCompiledProgram
from pypto.backend import BackendType
from pypto.runtime import ExecutionMode, RunConfig
from pypto_serving.model.common.compiler.compiler import KernelCompiler
from pypto_serving.model.common.compiler.l3_callable import L3Callable
from pypto_serving.model.common.executor.utils import build_pypto_run_config


def _base_run_config() -> RunConfig:
    """Build the base RunConfig an executor hands to KernelCompiler."""
    return build_pypto_run_config(platform="a2a3sim", device_ids=(0, 1))


@pytest.mark.parametrize("platform,backend,mode", [
    ("a2a3", BackendType.Ascend910B, ExecutionMode.ONBOARD),
    ("a2a3sim", BackendType.Ascend910B, ExecutionMode.SIM),
    ("a5", BackendType.Ascend950, ExecutionMode.ONBOARD),
    ("a5sim", BackendType.Ascend950, ExecutionMode.SIM),
])
def test_run_config_uses_explicit_target_axes(platform, backend, mode):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        config = build_pypto_run_config(platform=platform, device_ids=(2, 3))
    assert config.backend_type == backend
    assert config.execution_mode == mode
    assert config.platform == platform
    assert config.device_id == 2
    assert config.distributed_config.device_ids == [2, 3]
    assert config.codegen_only is True


def test_run_config_rejects_unknown_platform():
    with pytest.raises(ValueError, match="Unsupported PyPTO platform"):
        build_pypto_run_config(platform="a2a3invalid", device_ids=(0,))


class _FakeJitFn:
    """Record the args/config handed to ``compile`` and return a canned program."""

    def __init__(self, program: object) -> None:
        self._program = program
        self.last_config: RunConfig | None = None
        self.compile_calls = 0
        self.compile_kwargs = {}

    def compile(self, *, config: RunConfig, **compile_kwargs: object) -> object:
        self.compile_kwargs = compile_kwargs
        self.last_config = config
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

    result = compiler.compile("prefill", jit_fn)

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
    run_config = build_pypto_run_config(platform="a2a3sim", device_ids=(0, 1), aicpu_thread_num=8)
    jit_fn = _FakeJitFn(MagicMock(spec=DistributedCompiledProgram))
    compiler = KernelCompiler(run_config=run_config)

    result = compiler.compile("prefill", jit_fn)

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

    assert jit_fn.compile_kwargs == {"num_tokens": RUNTIME}


def test_compile_raises_on_non_distributed_compiled_program() -> None:
    """A result that is not a DistributedCompiledProgram raises TypeError."""
    jit_fn = _FakeJitFn(object())  # not a DistributedCompiledProgram
    compiler = _make_compiler()

    with pytest.raises(TypeError, match="DistributedCompiledProgram"):
        compiler.compile("prefill", jit_fn)


def test_cache_environment_is_passed_through_untouched(monkeypatch):
    """Serving parses no PYPTO_CACHE_* variable; PyPTO resolves them itself.

    Even a hostile environment (``yes`` where PyPTO demands ``0``/``1``) must
    not fail the compile wiring, and no diagnostic output may be requested:
    the policy is ``None`` on the compile RunConfig either way.
    """
    monkeypatch.setenv("PYPTO_CACHE", "yes")
    monkeypatch.setenv("PYPTO_CACHE_DIR", "/nonexistent-artifact-root")
    monkeypatch.setenv("PYPTO_CACHE_READONLY", "yes")
    jit_fn = _FakeJitFn(MagicMock(spec=DistributedCompiledProgram))
    compiler = _make_compiler()

    compiler.compile("prefill", jit_fn)

    config = jit_fn.last_config
    assert jit_fn.compile_calls == 1
    assert config.cache_config is None
    assert not config.save_kernels
    assert config.save_kernels_dir is None


def test_old_named_slot_never_short_circuits_jit(tmp_path, monkeypatch):
    slot = tmp_path / "prefill"
    slot.mkdir()
    (slot / "distributed_meta.json").write_text("{}")

    def unexpected_load(*args, **kwargs):
        raise AssertionError("serving must never restore an unvalidated named slot")

    monkeypatch.setattr(DistributedCompiledProgram, "from_dir", unexpected_load)
    jit_fn = _FakeJitFn(MagicMock(spec=DistributedCompiledProgram))
    compiler = _make_compiler()
    compiler.compile("prefill", jit_fn)
    assert jit_fn.compile_calls == 1
    assert jit_fn.last_config.cache_config is None
    assert (slot / "distributed_meta.json").read_text() == "{}"


def test_run_config_policy_flows_through_untouched(tmp_path):
    """An executor-supplied RunConfig.cache_config reaches jit_fn.compile as-is.

    Serving never rebuilds the policy: root, read-only, extra source paths and
    the extra fingerprint (which takes part in the artifact key) survive
    verbatim; ``None`` stays ``None`` for PyPTO's own env resolution.
    """
    from dataclasses import replace

    policy = CacheConfig(
        enabled=True,
        root=tmp_path,
        readonly=True,
        extra_source_paths=(tmp_path / "kernels",),
        extra_fingerprint="model-v1",
    )
    config = replace(_base_run_config(), cache_config=policy)
    jit_fn = _FakeJitFn(MagicMock(spec=DistributedCompiledProgram))
    compiler = _make_compiler(run_config=config)
    compiler.compile("prefill", jit_fn)
    assert jit_fn.last_config.cache_config is policy
    assert config.cache_config is policy


def test_explicit_output_keeps_pypto_bypass_contract(tmp_path):
    jit_fn = _FakeJitFn(MagicMock(spec=DistributedCompiledProgram))
    compiler = _make_compiler(save_kernels=True, save_kernels_dir=str(tmp_path))
    compiler.compile("prefill", jit_fn)
    assert jit_fn.last_config.save_kernels is True
    assert jit_fn.last_config.save_kernels_dir == str(tmp_path)
    assert jit_fn.last_config.cache_config is None


def test_callable_uses_effective_distributed_config():
    from pypto.ir.distributed_compiled_program import DistributedConfig

    config = DistributedConfig(device_ids=[2], aicpu_thread_num=8)
    jit_fn = _FakeJitFn(MagicMock(spec=DistributedCompiledProgram))
    result = _make_compiler(distributed_config=config).compile("prefill", jit_fn)
    assert result.aicpu_thread_num == 8
    assert jit_fn.last_config.distributed_config is config


def test_explicit_output_base_keeps_programs_separate(tmp_path):
    config = build_pypto_run_config(
        platform="a2a3sim", device_ids=[0], pypto_build_dir=str(tmp_path),
    )
    compiler = _make_compiler(run_config=config)
    jit_fn = _FakeJitFn(MagicMock(spec=DistributedCompiledProgram))
    for name in ("prefill", "decode"):
        compiler.compile(name, jit_fn)
        assert jit_fn.last_config.save_kernels_dir == str(tmp_path / name)
    assert config.save_kernels_dir == str(tmp_path)


@pytest.mark.parametrize("failure", [None, "allocation", "preflight"])
def test_registration_closes_unpublished_runner_on_capacity_failure(failure):
    from types import SimpleNamespace
    from pypto_serving.model.common.executor.pypto_executor import PyptoExecutor

    closed = []
    def init(*_args):
        if failure == "allocation":
            raise MemoryError("allocation failed")
        return 7
    def preflight(_record):
        if failure == "preflight":
            raise RuntimeError("preflight failed")
    runner = SimpleNamespace(init_kv_cache=init, preflight=preflight, close=lambda: closed.append(True))
    compiled = object()
    executor = SimpleNamespace(_compile_model=lambda _runtime: compiled,
        _create_runner=lambda *_args: runner, _runners={}, _compiled={})
    record = SimpleNamespace(runtime_model=object(), config=object(), runtime=object())
    if failure:
        with pytest.raises((MemoryError, RuntimeError), match=failure):
            PyptoExecutor.register_model(executor, "model", record)
        assert closed == [True] and not executor._runners and not executor._compiled
    else:
        assert PyptoExecutor.register_model(executor, "model", record) == 7
        assert not closed and executor._runners == {"model": runner}
