# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from types import SimpleNamespace

from pypto.runtime import RunConfig

from pypto_serving.model.common.compiler.l3_callable import L3Callable
from pypto_serving.model.common.runner.l3_dispatch import L3DispatchMixin


class _Handle:
    def result(self):
        return None


class _Worker:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.configs = []

    def run(self, _compiled, *args, config=None):
        del args
        self.configs.append(config)
        dfx = self.output_dir / "dfx_outputs" / "rank0" / "d0"
        dfx.mkdir(parents=True, exist_ok=True)
        (dfx / "chip_swimlane_records.json").write_text(str(len(self.configs)))

    def submit(self, compiled, *args, config=None):
        self.run(compiled, *args, config=config)
        return _Handle()


class _Runner(L3DispatchMixin):
    def __init__(self, worker):
        self._init_l3_dispatch(stacked=False)
        self._worker = worker

    def _shared_l3_worker(self):
        return self._worker


def test_chip_profile_preserves_dispatch_config_and_archives_each_run(tmp_path):
    worker = _Worker(tmp_path / "build")
    runner = _Runner(worker)
    spec = L3Callable(
        compiled=SimpleNamespace(
            output_dir=tmp_path / "immutable-cache",
            _artifact_runtime=SimpleNamespace(run_directory=worker.output_dir),
        ),
        name="decode",
    )
    baseline = RunConfig(ring_heap=1024)

    runner.set_chip_swimlane_level(2, tmp_path / "captures")
    runner._run_l3(spec, config=baseline)
    pending = runner._submit_l3(spec, config=baseline)
    pending.wait()
    runner.set_chip_swimlane_level(0)
    runner._run_l3(spec, config=baseline)

    assert [config.enable_chip_swimlane for config in worker.configs] == [2, 2, 0]
    assert all(config.ring_heap == 1024 for config in worker.configs)
    captures = sorted((tmp_path / "captures").glob("worker-*/*/rank0/d0/chip_swimlane_records.json"))
    assert [capture.read_text() for capture in captures] == ["1", "2"]
