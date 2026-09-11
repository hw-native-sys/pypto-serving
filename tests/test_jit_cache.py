# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Real shared-compiler/L3 cache acceptance, run in a dedicated NPU invocation.

The parent never initializes a device. Each phase starts a fresh interpreter,
so READY reuse cannot accidentally pass through the in-process object cache.
"""

import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import pypto.language as pl
import torch
from pypto import cache_stats

from pypto_serving.model.common.compiler.compiler import KernelCompiler
from pypto_serving.model.common.executor.utils import build_pypto_run_config
from pypto_serving.model.common.runner.l3_dispatch import L3DispatchMixin


EXTRA = int(os.environ.get("SERVING_CACHE_TEST_EXTRA", "0"))


@pl.jit
def cache_chip(
    x: pl.Tensor[[16, 16], pl.FP32], out: pl.Out[pl.Tensor[[16, 16], pl.FP32]], factor: pl.Scalar[pl.INT32]
):
    with pl.at(level=pl.Level.CORE_GROUP):
        pl.store(pl.add(pl.add(pl.load(x, [0, 0], [16, 16]), factor), EXTRA), [0, 0], out)
    return out


@pl.jit.host
def cache_host(
    x: pl.Tensor[[16, 16], pl.FP32], out: pl.Out[pl.Tensor[[16, 16], pl.FP32]], factor: pl.Scalar[pl.INT32]
):
    return cache_chip(x, out, factor)


class _Runner(L3DispatchMixin):
    def __init__(self, spec):
        self._init_l3_dispatch(stacked=False)
        self.spec = spec

    def _shared_l3_worker(self):
        from pypto.runtime import DistributedWorker

        if self._l3_worker is None:
            self._l3_worker = DistributedWorker([self.spec.compiled])
        return self._l3_worker

    def close(self):
        if self._l3_worker is not None:
            self._l3_worker.close()


def _run_phase(phase, factor, report):
    from contextlib import ExitStack

    from pypto.runtime.kernel_compiler import KernelCompiler as DeviceCompiler

    consume = phase == "hit"
    counters = dict(generation=0, ptoas=0, incore=0, orchestration=0)

    def count_stage(name, original):
        def wrapped(*args, **kwargs):
            counters[name] += 1
            assert not consume, f"READY reuse invoked {name}"
            return original(*args, **kwargs)

        return wrapped

    import pypto.backend.pto_backend as backend

    start = perf_counter()
    with ExitStack() as stack:
        for owner, attr, label in (
            (cache_host, "_compile", "generation"),
            (backend, "_run_ptoas", "ptoas"),
            (DeviceCompiler, "compile_incore", "incore"),
            (DeviceCompiler, "compile_orchestration", "orchestration"),
        ):
            stack.enter_context(patch.object(owner, attr, count_stage(label, getattr(owner, attr))))
        compiler = KernelCompiler(
            run_config=build_pypto_run_config(
                platform=os.environ.get("PYPTO_QWEN3_PLATFORM", "a2a3"),
                device_ids=[int(os.environ["DEVICE_ID"])],
            )
        )
        spec = compiler.compile("same-serving-name", cache_host, factor=factor)
        assert cache_stats().bypasses == 0, cache_stats().last_bypass_reason
        assert cache_stats().forced_rebuilds == 0
        assert (spec.compiled.program is None) == consume
        runner = _Runner(spec)
        x = torch.full((16, 16), 2.0).share_memory_()
        out = torch.zeros_like(x).share_memory_()
        try:
            runner._run_l3(spec, x, out, factor)
            torch.testing.assert_close(out, torch.full_like(out, 2.0 + factor + EXTRA))
        finally:
            runner.close()
    stats = cache_stats()
    assert stats.bypasses == 0, stats.last_bypass_reason
    assert stats.forced_rebuilds == 0
    assert stats.ready_hits == int(consume)
    assert stats.generation_builds == int(not consume)
    assert stats.binary_builds == int(not consume)
    if consume:
        assert not any(counters.values()), counters
    else:
        assert all(counters.values()), counters
    Path(report).write_text(
        json.dumps(
            {
                "phase": phase,
                "factor": factor,
                "extra": EXTRA,
                "elapsed_seconds": perf_counter() - start,
                "stats": asdict(stats),
                "compiler_calls": counters,
            },
            indent=2,
        )
    )


def test_shared_compiler_cache_across_processes(tmp_path):
    """Ordinary L3 execution publishes; a new process validates and reuses READY.

    A second scalar specialization uses the same serving name and must compile
    separately. A changed helper global also invalidates the artifact. Earlier
    specializations remain reusable, including read-only hits.
    """
    assert os.environ.get("DEVICE_ID") is not None, "Run through task-submit with DEVICE_ID"
    root = tmp_path / "cache"
    phases = (
        ("miss", 3, 0),
        ("hit", 3, 0),
        ("miss", 4, 0),
        ("hit", 4, 0),
        ("miss", 3, 1),
        ("hit", 3, 1),
        ("hit", 3, 0),
    )
    for index, (phase, factor, extra) in enumerate(phases):
        environment = dict(
            os.environ,
            PYPTO_CACHE="1",
            PYPTO_CACHE_DIR=str(root),
            PYPTO_CACHE_READONLY=str(int(phase == "hit")),
            SERVING_CACHE_TEST_EXTRA=str(extra),
        )
        environment.pop("PYPTO_PROG_BUILD_DIR", None)
        report = tmp_path / f"phase-{index}.json"
        log = tmp_path / f"phase-{index}.log"
        with log.open("w") as output:
            result = subprocess.run(
                [sys.executable, "-m", "tests.test_jit_cache", phase, str(factor), str(report)],
                env=environment,
                stdout=output,
                stderr=subprocess.STDOUT,
                timeout=300,
            )
        assert result.returncode == 0, log.read_text()
        print(report.read_text(), flush=True)
    assert not list(root.rglob("__pycache__"))


if __name__ == "__main__":
    _run_phase(sys.argv[1], int(sys.argv[2]), sys.argv[3])
