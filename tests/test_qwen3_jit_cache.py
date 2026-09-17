# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Cold/READY Qwen generation through real spawned serving workers.

Run separately from other hardware tests so the parent has not initialized an
NPU. The child module installs compiler guards again when multiprocessing
imports it as __mp_main__; ordinary pytest collection installs no guards.
"""

import json
import os
import subprocess
import sys
from pathlib import Path


def _install_worker_checks():
    from dataclasses import asdict
    from time import perf_counter

    import pypto
    import pypto.backend.pto_backend as backend
    from pypto.language import JITFunction
    from pypto.runtime.kernel_compiler import KernelCompiler as DeviceCompiler

    from pypto_serving.model.common.compiler.compiler import KernelCompiler
    from pypto_serving.model.qwen.npu_executor import Qwen314BPyptoExecutor

    consume = os.environ["SERVING_CACHE_TEST_PHASE"] == "hit"
    counters = dict(generation=0, ptoas=0, incore=0, orchestration=0)
    names = []

    def count_stage(name, original):
        def wrapped(*args, **kwargs):
            counters[name] += 1
            assert not consume, f"Qwen READY reuse invoked {name}"
            return original(*args, **kwargs)

        return wrapped

    for owner, attr, name in (
        (JITFunction, "_compile", "generation"),
        (backend, "_run_ptoas", "ptoas"),
        (DeviceCompiler, "compile_incore", "incore"),
        (DeviceCompiler, "compile_orchestration", "orchestration"),
    ):
        setattr(owner, attr, count_stage(name, getattr(owner, attr)))

    original_compile = KernelCompiler.compile

    def compile_checked(self, name, *args, **kwargs):
        result = original_compile(self, name, *args, **kwargs)
        assert (result.compiled.program is None) == consume
        names.append(name)
        stats = pypto.cache_stats()
        assert stats.bypasses == 0, stats.last_bypass_reason
        assert stats.forced_rebuilds == 0
        return result

    KernelCompiler.compile = compile_checked
    original_register = Qwen314BPyptoExecutor.register_model

    def register_checked(self, *args, **kwargs):
        start = perf_counter()
        result = original_register(self, *args, **kwargs)
        stats = pypto.cache_stats()
        assert len(names) >= 3, names
        assert stats.ready_hits == (len(names) if consume else 0), stats
        assert stats.generation_builds == (0 if consume else len(names)), stats
        assert stats.binary_builds == 0 if consume else stats.binary_builds > 0
        assert stats.storage_errors == 0 and stats.invalid_entries == 0, stats
        if consume:
            assert not any(counters.values()), counters
        else:
            assert all(counters.values()), counters
        Path(os.environ["SERVING_CACHE_TEST_REPORT"]).write_text(
            json.dumps(
                {
                    "worker_pid": os.getpid(),
                    "kernels": names,
                    "compiler_calls": counters,
                    "stats": asdict(stats),
                    "registration_seconds": perf_counter() - start,
                },
                indent=2,
            )
        )
        return result

    Qwen314BPyptoExecutor.register_model = register_checked


def test_qwen_generation_publishes_and_reuses_ready(tmp_path):
    """Both independent launches must produce the established greedy token IDs."""
    model = os.environ.get("PYPTO_QWEN3_MODEL_DIR")
    assert model and Path(model).is_dir(), "PYPTO_QWEN3_MODEL_DIR must point to Qwen3-14B weights"
    assert os.environ.get("DEVICE_ID") is not None, "Run through task-submit with DEVICE_ID"
    root = tmp_path / "cache"
    reports = []
    for phase in ("miss", "hit"):
        report = tmp_path / f"{phase}.json"
        log = tmp_path / f"{phase}.log"
        environment = dict(
            os.environ,
            PYPTO_CACHE="1",
            PYPTO_CACHE_DIR=str(root),
            PYPTO_CACHE_READONLY=str(int(phase == "hit")),
            SERVING_CACHE_TEST_PHASE=phase,
            SERVING_CACHE_TEST_REPORT=str(report),
        )
        environment.pop("PYPTO_PROG_BUILD_DIR", None)
        with log.open("w") as output:
            result = subprocess.run(
                [sys.executable, "-m", "tests.test_qwen3_jit_cache"],
                env=environment,
                stdout=output,
                stderr=subprocess.STDOUT,
                timeout=900,
            )
        assert result.returncode == 0, log.read_text()
        assert report.is_file(), f"No worker cache evidence: {log.read_text()}"
        reports.append(json.loads(report.read_text()))
        print(report.read_text(), flush=True)
    assert reports[0]["worker_pid"] != reports[1]["worker_pid"]
    assert reports[0]["kernels"] == reports[1]["kernels"]
    assert not list(root.rglob("__pycache__"))


if __name__ in ("__main__", "__mp_main__"):
    _install_worker_checks()
    if __name__ == "__main__":
        from tests.test_qwen3_accuracy import test_qwen3_output_matches_expected_tokens

        test_qwen3_output_matches_expected_tokens()
