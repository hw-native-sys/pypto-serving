# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""/health reports serving readiness, not just route liveness.

A router in front of several replicas has to stop routing to a replica whose
worker died or whose engine loop stopped scheduling; both look identical to a
healthy idle server unless /health actually consults the engine.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from pypto_serving.config.types import GenerateConfig
from pypto_serving.serving.engine.async_engine import ReplicaEngineCore
from pypto_serving.serving.server.server import ServingServer


class _Engine:
    """Minimal engine stub: /health only touches is_ready()."""

    def __init__(self, ready: bool) -> None:
        self._ready = ready

    def is_ready(self) -> bool:
        return self._ready


def _server(ready: bool) -> ServingServer:
    return ServingServer(
        async_engine=_Engine(ready),
        model_id="test-model",
        generate_config=GenerateConfig(),
    )


def _core(*, running: bool, loop_done: bool, worker_alive: bool) -> ReplicaEngineCore:
    """Build a core without touching a worker process or a device."""
    core = ReplicaEngineCore.__new__(ReplicaEngineCore)
    core._running = running
    core._loop_task = SimpleNamespace(done=lambda: loop_done)
    core._worker_process = SimpleNamespace(is_alive=lambda: worker_alive)
    return core


def test_health_is_ok_when_the_engine_is_ready():
    response = asyncio.run(_server(ready=True)._health())
    assert response.status_code == 200
    assert json.loads(response.body)["status"] == "ok"


def test_health_is_503_when_the_engine_is_not_ready():
    response = asyncio.run(_server(ready=False)._health())
    assert response.status_code == 503
    assert json.loads(response.body)["status"] == "not_ready"


def test_core_is_ready_only_when_every_part_is_live():
    assert _core(running=True, loop_done=False, worker_alive=True).is_ready() is True


def test_core_is_not_ready_before_start():
    core = ReplicaEngineCore.__new__(ReplicaEngineCore)
    core._running = False
    core._loop_task = None
    core._worker_process = None
    assert core.is_ready() is False


def test_core_is_not_ready_when_the_worker_died():
    """The engine would otherwise block on the output queue for 1200s by default."""
    assert _core(running=True, loop_done=False, worker_alive=False).is_ready() is False


def test_core_is_not_ready_when_the_engine_loop_died():
    """A loop task that raised leaves _running True while nothing is scheduled."""
    assert _core(running=True, loop_done=True, worker_alive=True).is_ready() is False


def test_core_is_not_ready_after_stop_clears_running():
    assert _core(running=False, loop_done=False, worker_alive=True).is_ready() is False


def test_engine_is_ready_only_when_all_cores_are():
    from pypto_serving.serving.engine.async_engine import AsyncLLMEngine

    engine = AsyncLLMEngine.__new__(AsyncLLMEngine)
    engine._cores = [
        SimpleNamespace(is_ready=lambda: True),
        SimpleNamespace(is_ready=lambda: True),
    ]
    assert engine.is_ready() is True

    engine._cores[1] = SimpleNamespace(is_ready=lambda: False)
    assert engine.is_ready() is False


def test_engine_with_no_cores_is_not_ready():
    from pypto_serving.serving.engine.async_engine import AsyncLLMEngine

    engine = AsyncLLMEngine.__new__(AsyncLLMEngine)
    engine._cores = []
    assert engine.is_ready() is False
