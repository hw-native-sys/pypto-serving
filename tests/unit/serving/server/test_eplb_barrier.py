# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""EPLB drains FIFO output ownership and keeps failed maintenance fatal."""

from queue import Queue
from threading import Event, Semaphore, Thread
from types import SimpleNamespace

import pytest

from pypto_serving.serving.server import serving_worker as module
from pypto_serving.serving.server.ipc import (
    ShutdownCommand, StepCommand, StepResult, decode_result, encode_command, encode_result,
)


def step(index):
    return StepCommand(new_requests=[], prefill_requests=[], decode_requests=[],
                       finished_request_ids=[], step_id=index)


@pytest.mark.parametrize("reclaim_error", [None, "native reclaim failure"])
def test_maintenance_waits_for_reclaim_slot_release_and_result_publication(reclaim_error):
    worker = module.WorkerProcess.__new__(module.WorkerProcess)
    worker.model_record = SimpleNamespace(runtime_model=object())
    worker.output_queue = Queue()
    reclaim_started, allow_reclaim, marker_queued, applied = Event(), Event(), Event(), Event()
    slots = [Semaphore(0)]
    failures = []

    class OutputQueue(Queue):
        def put(self, item, *args, **kwargs):
            super().put(item, *args, **kwargs)
            if isinstance(item, module._EplbDrainBarrier):
                marker_queued.set()

    queue = OutputQueue()

    def reclaim(work):
        reclaim_started.set()
        assert allow_reclaim.wait(3)
        return StepResult(new_tokens={}, step_id=work.cmd.step_id, error=reclaim_error)

    def apply(model, plan):
        assert slots[0].acquire(blocking=False)
        assert worker.output_queue.qsize() == 1
        applied.set()

    worker._reclaim_pending_decode = reclaim
    worker._finish_pending_decode_reclaims = lambda _: None
    worker.executor = SimpleNamespace(eplb_enabled=True, take_ready_eplb_plan=lambda _: object(),
                                      validate_eplb_plan=lambda *_: None, apply_eplb_plan=apply)
    queue.put(module._PendingDecodeOutput(cmd=step(1), scheduled=(), pending=object(), buffer_slot=0,
                                         batch=object()))
    output = Thread(target=worker._output_reclaim_loop, args=(queue, slots), daemon=True)
    output.start()
    assert reclaim_started.wait(3)

    def maintain():
        try:
            worker._maintain_eplb(queue)
        except RuntimeError as error:
            failures.append(str(error))

    execution = Thread(target=maintain, daemon=True)
    execution.start()
    try:
        assert marker_queued.wait(3)
        assert not applied.is_set()
        allow_reclaim.set()
        execution.join(3)
        assert not execution.is_alive()
        if reclaim_error is None:
            assert applied.is_set() and not failures
        else:
            assert not applied.is_set() and reclaim_error in failures[0]
    finally:
        allow_reclaim.set()
        queue.put(ShutdownCommand())
        execution.join(3)
        output.join(3)
    assert not output.is_alive()


def test_failed_maintenance_rejects_every_later_compute_command():
    worker = module.WorkerProcess.__new__(module.WorkerProcess)
    worker.model_record = SimpleNamespace(runtime_model=object())
    worker.output_queue = Queue()
    plans = [object()]
    computed = []

    def apply(*_):
        raise RuntimeError("partial weight installation")

    worker.executor = SimpleNamespace(eplb_enabled=True, take_ready_eplb_plan=lambda _: plans.pop(),
                                      validate_eplb_plan=lambda *_: None, apply_eplb_plan=apply)
    worker._execute_step = lambda *args, **kwargs: computed.append(args)
    work, output_work = Queue(), Queue()
    for command in (step(1), step(2), ShutdownCommand()):
        work.put((command, None, {}))
    output = Thread(target=worker._output_reclaim_loop, args=(output_work, []), daemon=True)
    output.start()
    execution = Thread(target=worker._device_execution_loop, args=(work, output_work), daemon=True)
    execution.start()
    execution.join(3)
    output.join(3)
    assert not execution.is_alive() and not output.is_alive()
    assert not computed
    assert worker.output_queue.qsize() == 2
    assert "partial weight installation" in worker._executor_fatal_error


@pytest.mark.parametrize("maintenance_error", [None, "partial installation"])
def test_serial_worker_commits_between_reclaimed_steps_and_keeps_failure_fatal(maintenance_error):
    worker = module.WorkerProcess.__new__(module.WorkerProcess)
    worker.model_record = SimpleNamespace(runtime_model=object())
    worker.input_queue, worker.output_queue = Queue(), Queue()
    events = []
    plans = iter((None, object(), None))

    def apply(*_):
        assert events == ["compute-1"]
        events.append("install")
        if maintenance_error is not None:
            raise RuntimeError(maintenance_error)

    worker.executor = SimpleNamespace(eplb_enabled=True, take_ready_eplb_plan=lambda _: next(plans),
                                      validate_eplb_plan=lambda *_: None, apply_eplb_plan=apply)

    def execute(command):
        events.append(f"compute-{command.step_id}")
        worker.output_queue.put(encode_result(StepResult(new_tokens={}, step_id=command.step_id)))

    worker._handle_step_command = execute
    for command in (step(1), step(2), step(3), ShutdownCommand()):
        worker.input_queue.put(encode_command(command))
    worker._serial_busy_loop()
    results = [decode_result(worker.output_queue.get_nowait()) for _ in range(3)]
    assert [result.step_id for result in results] == [1, 2, 3]
    if maintenance_error is None:
        assert events == ["compute-1", "install", "compute-2", "compute-3"]
        assert all(result.error is None for result in results)
    else:
        assert events == ["compute-1", "install"]
        assert results[0].error is None
        assert all(maintenance_error in result.error for result in results[1:])
