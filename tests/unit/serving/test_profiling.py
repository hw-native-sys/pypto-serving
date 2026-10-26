# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import pypto_serving.cli.main as cli
from pypto_serving.config.types import GenerateConfig
from pypto_serving.serving.engine.async_engine import EngineConfig, ReplicaEngineCore
from pypto_serving.serving.server import server as server_module
from pypto_serving.serving.server import serving_worker
from pypto_serving.serving.server.ipc import (
    ProfileCommand,
    ProfileResult,
    ShutdownCommand,
    decode_command,
    decode_profile_result,
    encode_command,
    encode_profile_result,
)
from pypto_serving.tools.profile.env import create_profile_config, load_profile_config
from pypto_serving.tools.profile.merge import merge_fragments
from pypto_serving.tools.profile.recorder import ProfileRecorder


class _Queue:
    def __init__(self, values=None) -> None:
        self.values = list(values or [])

    def put(self, value) -> None:
        self.values.append(value)

    def get(self, timeout=None):
        del timeout
        return self.values.pop(0)


def test_profile_cli_config_reaches_engine_workers(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    output = tmp_path / "profile"
    args = cli.build_parser().parse_args(
        [
            "--model",
            str(model_dir),
            "--profile",
            "--profile-output",
            str(output),
            "--profile-level",
            "verbose",
            "--profile-chip-level",
            "2",
        ]
    )

    config = cli.build_serving_engine_config(args).profile_config

    assert config.enabled is True
    assert config.output == output.resolve()
    assert config.includes("e2e")
    assert config.includes("kernel")
    assert config.chip_swimlane_level == 2
    assert config.chip_artifacts_dir == output / "chip"


def test_chip_profile_requires_profile_flag(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    args = cli.build_parser().parse_args(
        ["--model", str(model_dir), "--profile-chip-level", "2"]
    )
    with pytest.raises(ValueError, match="require --profile"):
        cli.build_serving_engine_config(args)


def test_profile_recorder_can_start_and_stop_without_process_exit(tmp_path):
    config = load_profile_config(
        {
            "SA_PROFILE_OUTPUT": str(tmp_path),
            "SA_PROFILE_LEVEL": "e2e,kernel",
        }
    )
    recorder = ProfileRecorder(config, process_name="test", initially_active=False)

    with recorder.span("before-start"):
        pass
    assert not config.fragments_dir.joinpath(f"trace.{recorder.pid}.jsonl").exists()

    assert recorder.start() is True
    with recorder.span("captured"):
        pass
    assert recorder.stop() is True
    assert recorder.active is False

    event_count = merge_fragments(config.fragments_dir, config.trace_file)
    assert event_count >= 2
    assert '"name":"captured"' in config.fragments_dir.joinpath(f"trace.{recorder.pid}.jsonl").read_text()


def test_worker_profile_commands_ack_after_state_change(monkeypatch):
    class _Profiler:
        active = False

        def start(self):
            self.active = True

        def stop(self):
            self.active = False

    profiler = _Profiler()
    monkeypatch.setattr(serving_worker, "get_profiler", lambda **_kwargs: profiler)

    worker = serving_worker.WorkerProcess.__new__(serving_worker.WorkerProcess)
    worker.config = EngineConfig(profile_config=create_profile_config(enabled=True))
    worker.input_queue = _Queue(
        [
            encode_command(ProfileCommand(active=True)),
            encode_command(ProfileCommand(active=False)),
            encode_command(ShutdownCommand()),
        ]
    )
    worker.profile_output_queue = _Queue()
    worker.busy_loop()

    results = [decode_profile_result(raw) for raw in worker.profile_output_queue.values]
    assert results == [
        ProfileResult(active=True),
        ProfileResult(active=False),
    ]


def test_worker_chip_profile_follows_start_and_stop(monkeypatch, tmp_path):
    class _Profiler:
        active = False

        def start(self):
            self.active = True

        def stop(self):
            self.active = False

    class _Executor:
        def __init__(self):
            self.calls = []

        def set_chip_swimlane_level(self, level, output_dir=None):
            self.calls.append((level, output_dir))

    profiler = _Profiler()
    monkeypatch.setattr(serving_worker, "get_profiler", lambda **_kwargs: profiler)
    config = create_profile_config(enabled=True, output=tmp_path, chip_swimlane_level=2)
    worker = serving_worker.WorkerProcess.__new__(serving_worker.WorkerProcess)
    worker.config = SimpleNamespace(profile_config=config)
    worker.executor = _Executor()
    worker.profile_output_queue = _Queue()

    worker._handle_profile_command(ProfileCommand(active=True))
    worker._handle_profile_command(ProfileCommand(active=True))
    worker._handle_profile_command(ProfileCommand(active=False))

    assert worker.executor.calls == [(2, config.chip_artifacts_dir / "captures"), (0, None)]
    assert [decode_profile_result(raw) for raw in worker.profile_output_queue.values] == [
        ProfileResult(active=True), ProfileResult(active=True), ProfileResult(active=False),
    ]


def test_replica_core_waits_for_profile_ack(monkeypatch):
    async def run_inline(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", run_inline)
    sent = _Queue()
    acknowledgements = _Queue(
        [
            encode_profile_result(ProfileResult(active=True)),
            encode_profile_result(ProfileResult(active=False)),
        ]
    )
    core = ReplicaEngineCore.__new__(ReplicaEngineCore)
    core._profile_lock = asyncio.Lock()
    core._input_queue = sent
    core._profile_output_queue = acknowledgements
    core._step_timeout = 1.0

    async def control_profile():
        await core.start_profile()
        await core.stop_profile()

    asyncio.run(control_profile())

    commands = [decode_command(raw) for raw in sent.values]
    assert commands == [
        ProfileCommand(active=True),
        ProfileCommand(active=False),
    ]


def test_http_profile_endpoints_control_workers_and_merge(monkeypatch):
    calls = []

    class _Engine:
        async def start_profile(self):
            calls.append("workers-start")

        async def stop_profile(self):
            calls.append("workers-stop")

    monkeypatch.setattr(
        server_module,
        "get_profiler",
        lambda **_kwargs: SimpleNamespace(enabled=True),
    )
    monkeypatch.setattr(
        server_module,
        "start_sa_profile",
        lambda: calls.append("main-start") or True,
    )
    monkeypatch.setattr(
        server_module,
        "stop_sa_profile",
        lambda: calls.append("main-stop") or True,
    )
    monkeypatch.setattr(
        server_module,
        "merge_profile",
        lambda: calls.append("merge") or 12,
    )

    server = server_module.ServingServer(_Engine(), model_id="test-model", generate_config=GenerateConfig())
    paths = {route.path for route in server.app.routes}
    assert "/start_profile" in paths
    assert "/stop_profile" in paths

    async def control_profile():
        start_response = await server._start_profile()
        stop_response = await server._stop_profile()
        return start_response, stop_response

    start_response, stop_response = asyncio.run(control_profile())
    assert start_response.status_code == 200
    assert stop_response.status_code == 200
    assert calls == [
        "main-start",
        "workers-start",
        "workers-stop",
        "main-stop",
        "merge",
    ]
