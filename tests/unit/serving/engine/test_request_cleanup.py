# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import asyncio
from collections import deque
from types import SimpleNamespace

from pypto_serving.serving.engine.async_engine import (
    ReplicaEngineCore,
    TokenOutput,
)
from pypto_serving.serving.server.ipc import (
    StepResult,
    decode_command,
    encode_result,
)


def test_worker_step_error_queues_finished_ids_for_executor_release():
    aborted: list[str] = []
    discarded: list[SimpleNamespace] = []
    core = ReplicaEngineCore.__new__(ReplicaEngineCore)
    core._request_contexts = {}
    core.scheduler = SimpleNamespace(
        abort_request=aborted.append,
        discard_scheduled_request=discarded.append,
    )
    core._pending_free_ids = []
    core._batch_queue = deque()
    core._discard_result_step_ids = set()
    contexts = {
        "req-a": SimpleNamespace(queue=asyncio.Queue()),
        "req-b": SimpleNamespace(queue=asyncio.Queue()),
    }
    core._request_contexts = dict(contexts)
    scheduler_output = SimpleNamespace(
        scheduled_requests=[
            SimpleNamespace(request=SimpleNamespace(request_id="req-a")),
            SimpleNamespace(request=SimpleNamespace(request_id="req-b")),
        ]
    )

    # Error path: the failed step's result was already consumed (result_pending
    # False); no in-flight batches, so nothing to drain.
    core._handle_step_error(7, scheduler_output, result_pending=False)

    assert aborted == ["req-a", "req-b"]
    assert discarded == scheduler_output.scheduled_requests
    assert core._pending_free_ids == ["req-a", "req-b"]
    assert not core._request_contexts
    for request_id in ("req-a", "req-b"):
        token = contexts[request_id].queue.get_nowait()
        assert isinstance(token, TokenOutput)
        assert token.finished is True
        assert token.finish_reason == "error"


def test_abort_request_schedules_worker_cleanup():
    """An aborted request must ride the next StepCommand's finished_request_ids,
    otherwise its worker-side _req_cache entry and device slots leak."""
    aborted: list[str] = []
    core = ReplicaEngineCore.__new__(ReplicaEngineCore)
    core._request_contexts = {}
    core.scheduler = SimpleNamespace(abort_request=aborted.append)
    core._pending_free_ids = []
    core._request_contexts = {"req-x": SimpleNamespace(
        queue=asyncio.Queue(), worker_release_done=asyncio.Event())}

    asyncio.run(core.abort_request("req-x"))

    # The terminal consumer still needs its context to receive the release ACK.
    assert aborted == ["req-x"]
    context = core._request_contexts["req-x"]
    core._acknowledge_worker_frees(("req-x",))
    assert context.worker_release_done.is_set()
    # The id is queued for worker release exactly once.
    assert core._pending_free_ids == ["req-x"]

    # Idempotent: a second abort (or an abort racing the finish path) must not
    # enqueue a duplicate free id.
    asyncio.run(core.abort_request("req-x"))
    assert core._pending_free_ids == ["req-x"]


def test_abort_request_emits_abort_token_before_scheduling_free():
    """The client-facing queue receives a FINISHED_ABORTED token on abort."""
    core = ReplicaEngineCore.__new__(ReplicaEngineCore)
    core._request_contexts = {}
    core.scheduler = SimpleNamespace(abort_request=lambda _req_id: None)
    core._pending_free_ids = []
    queue: asyncio.Queue = asyncio.Queue()
    core._request_contexts = {"req-y": SimpleNamespace(queue=queue)}

    asyncio.run(core.abort_request("req-y"))

    token = queue.get_nowait()
    assert isinstance(token, TokenOutput)
    assert token.finished is True
    assert token.finish_reason == "FINISHED_ABORTED"
    assert core._pending_free_ids == ["req-y"]


def test_flush_pending_frees_sends_cleanup_only_step_command(monkeypatch):
    """Aborting the last active request must not pin it on the worker: when no
    work is schedulable, _flush_pending_frees emits a cleanup-only StepCommand
    carrying the pending ids and drains the worker reply."""

    async def run_inline(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", run_inline)
    core = ReplicaEngineCore.__new__(ReplicaEngineCore)
    core._request_contexts = {}
    core.config = SimpleNamespace(executor_cls="PyptoQwen14BExecutor")
    core._worker_known_req_ids = {"aborted"}
    core._pending_free_ids = ["aborted"]
    core._batch_queue = deque()
    core._discard_result_step_ids = set()
    core._step_counter = 0
    core._step_timeout = 300.0

    sent: list[bytes] = []
    core._input_queue = SimpleNamespace(put=sent.append)
    # Worker replies with an empty StepResult for the cleanup-only step.
    core._output_queue = SimpleNamespace(get=lambda timeout=None: encode_result(StepResult(new_tokens={})))

    asyncio.run(core._flush_pending_frees())

    # Exactly one cleanup command was sent, carrying the pending id and no work.
    assert len(sent) == 1
    cmd = decode_command(sent[0])
    assert cmd.finished_request_ids == ["aborted"]
    assert cmd.new_requests == []
    assert cmd.prefill_requests == []
    assert cmd.decode_requests == []
    # Pending list drained; known-set no longer tracks the released id.
    assert core._pending_free_ids == []
    assert "aborted" not in core._worker_known_req_ids
