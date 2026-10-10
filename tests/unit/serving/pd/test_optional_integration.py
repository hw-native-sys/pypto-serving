# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Cold-import checks for the optional Serving PD boundary."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap


def test_ordinary_serving_does_not_load_pd_or_transfer(tmp_path):
    source = textwrap.dedent("""
        import importlib.abc
        import importlib.util
        import sys
        from types import SimpleNamespace

        class RejectPD(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if (fullname.startswith(("pypto_serving.serving.pd", "pypto_serving.transfer",
                                         "pypto_serving.router", "mooncake"))
                        or fullname.endswith(".pd_adapter")):
                    raise AssertionError("ordinary mode loaded " + fullname)
        sys.meta_path.insert(0, RejectPD())

        from pypto_serving.cli.main import build_parser
        from pypto_serving.config.types import GenerateConfig
        from pypto_serving.serving.engine.async_engine import AsyncLLMEngine, EngineConfig
        from pypto_serving.serving.server.server import ServingServer
        from pypto_serving.serving.server.serving_worker import WorkerProcess
        from pypto_serving.serving.server.ipc import DecodeRequest, NewRequestData

        assert build_parser().parse_args(["--model", "unused"]).pd_role == "disabled"
        config = EngineConfig()
        assert not hasattr(config, "pd_config")
        engine = AsyncLLMEngine(config, SimpleNamespace(eos_token_id=2, bos_token_id=1))
        core = engine.single_core()
        assert core._result_handler == core._process_step_output
        assert core.kv_cache_manager._reservations is None
        assert core.kv_cache_manager.group_cache_reservations == ()
        assert core.kv_cache_manager._reservations is None
        assert not any(name.startswith("_pd") for name in vars(core))
        server = ServingServer(engine, "model", GenerateConfig())
        paths = {route.path for route in server.app.routes}
        assert "/v1/chat/completions" in paths
        assert not any(path.startswith("/internal/pd") for path in paths)

        worker = WorkerProcess(config, None, None)
        assert worker._services is None
        assert worker._services_factory is None
        assert worker._batch_builder == worker._make_decode_batch
        worker.executor = SimpleNamespace(supports_device_decode_embedding=True)
        worker._req_cache["r"] = NewRequestData("r", [1], 0.0, 1.0, None, None)
        model = SimpleNamespace(runtime=SimpleNamespace(device="cpu"))
        for _ in range(3):
            batch = worker._batch_builder(
                [DecodeRequest("r", 2, 2, [])], model, resolve_tokens=True,
                allow_device_greedy_sampling=True, allow_device_topk_sampling=False,
            )
            assert batch.initial_request_ids == ()
        assert not any(name.startswith("_pd") for name in vars(worker))

        # Compiler dependencies are optional on a CPU-only developer machine.
        if importlib.util.find_spec("pypto") is not None:
            import pypto_serving.model.deepseek_dspark.npu_executor
        print("ordinary path isolated")
    """)
    root = Path(__file__).parents[4]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(root), os.environ.get("PYTHONPATH", "")))}
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ordinary path isolated" in result.stdout
