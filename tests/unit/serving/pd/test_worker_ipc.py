# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from types import SimpleNamespace

from pypto_serving.serving.server.ipc import (
    PDWorkerCommand,
    ProfileCommand,
    decode_command,
    decode_pd_result,
    decode_profile_result,
    encode_command,
)
from pypto_serving.serving.server.serving_worker import WorkerProcess


def test_pd_worker_command_round_trip_and_ordered_result() -> None:
    command = PDWorkerCommand(17, "inspect_registry", b"request")
    assert decode_command(encode_command(command)) == command

    outputs = []
    worker = WorkerProcess.__new__(WorkerProcess)
    worker.executor = SimpleNamespace(
        handle_pd_command=lambda operation, payload: operation.encode() + b":" + payload
    )
    worker.profile_output_queue = SimpleNamespace(put=outputs.append)
    worker._handle_pd_worker_command(command)
    result = decode_pd_result(outputs[0])
    assert result.command_id == 17
    assert result.payload == b"inspect_registry:request"
    assert result.error is None


def test_pd_worker_error_is_returned_without_pickle_or_address_objects() -> None:
    outputs = []
    worker = WorkerProcess.__new__(WorkerProcess)

    def fail(_operation, _payload):
        raise ValueError("bad payload")

    worker.executor = SimpleNamespace(handle_pd_command=fail)
    worker.profile_output_queue = SimpleNamespace(put=outputs.append)
    worker._handle_pd_worker_command(PDWorkerCommand(3, "transfer_chunk", b"bad"))
    result = decode_pd_result(outputs[0])
    assert result.command_id == 3
    assert result.payload == b""
    assert result.error == "bad payload"


def test_profile_command_controls_transfer_owner_before_ack(monkeypatch) -> None:
    events = []

    class _Profiler:
        active = False

        def start(self):
            events.append("worker-start")
            self.active = True

        def stop(self):
            events.append("worker-stop")
            self.active = False

    profiler = _Profiler()
    monkeypatch.setattr(
        "pypto_serving.serving.server.serving_worker.get_profiler",
        lambda initially_active=False: profiler,
    )
    outputs = []
    worker = WorkerProcess.__new__(WorkerProcess)
    worker.executor = SimpleNamespace(
        set_transfer_profile_active=lambda active: events.append(
            "owner-start" if active else "owner-stop"
        )
    )
    worker.profile_output_queue = SimpleNamespace(put=outputs.append)

    worker._handle_profile_command(ProfileCommand(active=True))
    started = decode_profile_result(outputs.pop())
    assert started.active
    assert started.error is None
    assert events == ["worker-start", "owner-start"]

    worker._handle_profile_command(ProfileCommand(active=False))
    stopped = decode_profile_result(outputs.pop())
    assert not stopped.active
    assert stopped.error is None
    assert events[-2:] == ["owner-stop", "worker-stop"]
