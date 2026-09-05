# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import signal
import threading
from queue import Queue
from types import SimpleNamespace

import pytest

from pypto_serving.serving.engine.async_engine import (
    ReplicaEngineCore,
)
from pypto_serving.serving.memory.kv_cache import KvCacheManager
from pypto_serving.serving.sched.scheduler import (
    Request,
    RequestStatus,
    ScheduledRequest,
    SchedulerOutput,
)
from pypto_serving.serving.server import serving_worker
from pypto_serving.serving.server.ipc import (
    DecodeRequest,
    NewRequestData,
    PrefillRequest,
    StepCommand,
    StepResult,
    ShutdownCommand,
    decode_command,
    decode_result,
    encode_command,
)
from pypto_serving.config.types import DecodeResult
from pypto_serving.serving.server.serving_worker import WorkerProcess

from ..device_sampling_fakes import _FixedSampler, _ImmediateEosExecutor, _model


def test_step_command_preserves_grouped_cache_metadata_on_preempted_restart():
    core = ReplicaEngineCore.__new__(ReplicaEngineCore)
    core._worker_known_req_ids = {"req"}
    request = Request(
        request_id="req",
        prompt_token_ids=[1, 2],
        max_new_tokens=1,
        status=RequestStatus.RUNNING,
    )
    scheduled = ScheduledRequest(
        request=request,
        num_new_tokens=2,
        is_prefill=True,
        block_ids_by_group={"ori": [3, 4], "state": [5]},
        cache_partition=2,
    )
    output = SchedulerOutput(scheduled_requests=[scheduled])

    command = core._build_step_command(output, finished_ids=["req"])
    decoded = decode_command(encode_command(command))

    assert [item.request_id for item in decoded.new_requests] == ["req"]
    assert decoded.finished_request_ids == ["req"]
    assert decoded.prefill_requests[0].block_ids_by_group == {
        "ori": [3, 4],
        "state": [5],
    }
    assert decoded.prefill_requests[0].cache_partition == 2


def test_partitioned_prefill_chunks_keep_cache_partitions_unique():
    requests = [
        PrefillRequest(
            request_id=request_id,
            chunk_tokens=[1],
            num_computed_tokens=0,
            block_ids=[],
            cache_partition=partition,
        )
        for request_id, partition in (("a", 0), ("b", 0), ("c", 1))
    ]

    chunks = WorkerProcess._partitioned_prefill_chunks(requests, max_batch=2)

    assert [[request.request_id for request in chunk] for chunk in chunks] == [
        ["a", "c"],
        ["b"],
    ]


def test_partitioned_prefill_chunks_honor_wider_partition_capacity():
    requests = [
        PrefillRequest(
            request_id=f"p0-{index}",
            chunk_tokens=[1],
            num_computed_tokens=0,
            block_ids=[],
            cache_partition=0,
        )
        for index in range(5)
    ]
    requests.append(
        PrefillRequest(
            request_id="p1-0",
            chunk_tokens=[1],
            num_computed_tokens=0,
            block_ids=[],
            cache_partition=1,
        )
    )

    chunks = WorkerProcess._partitioned_prefill_chunks(
        requests,
        max_batch=5,
        max_per_partition=4,
    )

    assert [[request.request_id for request in chunk] for chunk in chunks] == [
        ["p0-0", "p0-1", "p0-2", "p0-3", "p1-0"],
        ["p0-4"],
    ]


def test_worker_releases_preempted_state_before_same_command_reregistration():
    released: list[str] = []
    results: list[bytes] = []
    worker = WorkerProcess.__new__(WorkerProcess)
    worker.executor = SimpleNamespace(release_finished_requests=released.extend)
    worker._req_cache = {
        "req": NewRequestData("req", [0], 0.0, 1.0, None),
    }
    worker._last_tokens = {}
    worker.output_queue = SimpleNamespace(put=results.append)
    worker._execute_step = lambda _cmd: StepResult(new_tokens={})
    replacement = NewRequestData("req", [1, 2], 0.0, 1.0, None)
    command = StepCommand(
        new_requests=[replacement],
        prefill_requests=[],
        decode_requests=[],
        finished_request_ids=["req"],
    )

    worker._handle_step_command(command)

    assert released == ["req"]
    assert worker._req_cache["req"] == replacement
    assert len(results) == 1


def test_worker_release_does_not_remove_a_later_same_id_registration():
    released: list[str] = []
    worker = WorkerProcess.__new__(WorkerProcess)
    worker.executor = SimpleNamespace(release_finished_requests=released.extend)
    old = NewRequestData("req", [0], 0.0, 1.0, None)
    replacement = NewRequestData("req", [1], 0.0, 1.0, None)
    worker._req_cache = {"req": replacement}
    worker._last_tokens = {"req": [7]}

    worker._release_finished_request_state(
        ["req"],
        expected_cache_entries={"req": old},
    )

    assert released == ["req"]
    assert worker._req_cache["req"] is replacement
    assert worker._last_tokens["req"] == [7]


def test_serving_worker_packs_variable_length_prefill_chunks():
    model = _model(max_batch_size=2, eos_token_id=0)
    manager = KvCacheManager()
    executor = _ImmediateEosExecutor(manager)
    worker = WorkerProcess.__new__(WorkerProcess)
    worker.executor = executor
    worker.sampler = _FixedSampler(token_id=0)
    worker.model_record = SimpleNamespace(config=model.config)
    worker._req_cache = {
        "long": NewRequestData(
            request_id="long",
            prompt_token_ids=[1, 2, 3, 4],
            temperature=0.0,
            top_p=1.0,
            top_k=None,
        ),
        "short": NewRequestData(
            request_id="short",
            prompt_token_ids=[5],
            temperature=0.0,
            top_p=1.0,
            top_k=None,
        ),
    }
    scheduled = [
        PrefillRequest(
            request_id="long",
            chunk_tokens=[2, 3, 4],
            num_computed_tokens=1,
            block_ids=[0],
        ),
        PrefillRequest(
            request_id="short",
            chunk_tokens=[5],
            num_computed_tokens=0,
            block_ids=[1],
        ),
    ]
    new_tokens: dict[str, list[int]] = {}

    worker._batch_prefill(scheduled, model, new_tokens)

    assert new_tokens == {"long": [0], "short": [0]}
    assert len(executor.prefill_batches) == 1
    prefill_batch = executor.prefill_batches[0]
    assert prefill_batch.token_ids.ndim == 1
    assert prefill_batch.token_ids.tolist() == [2, 3, 4, 5]
    assert prefill_batch.seq_lens == [4, 1]
    assert prefill_batch.chunk_lens == [3, 1]
    assert prefill_batch.chunk_offsets == [0, 3]
    assert prefill_batch.chunk_starts == [1, 0]
    assert prefill_batch.token_ids.numel() == sum(prefill_batch.chunk_lens)
    assert prefill_batch.input_embeddings is not None
    assert prefill_batch.input_embeddings.shape == (4, model.config.hidden_size)
    assert executor.embedding_lookup_shapes == [(4,)]
    assert executor.finalized_prefills == [(["long", "short"], [0, 0])]


def test_worker_close_releases_executor_once():
    executor = SimpleNamespace(close_calls=0)

    def close():
        executor.close_calls += 1

    executor.close = close
    worker = serving_worker.WorkerProcess.__new__(serving_worker.WorkerProcess)
    worker.executor = executor

    worker.close()
    worker.close()

    assert executor.close_calls == 1
    assert worker.executor is None


def test_worker_prepares_next_decode_while_prior_device_step_runs():
    """MRV2 cadence: prepare and reclaim overlap the FIFO device lane."""
    model = _model(max_batch_size=1, eos_token_id=0)
    first_running = threading.Event()
    allow_first_finish = threading.Event()
    second_prepared = threading.Event()
    third_prepared = threading.Event()
    second_dispatched = threading.Event()
    first_reclaim_running = threading.Event()
    allow_first_reclaim = threading.Event()
    calls: list[tuple[str, int]] = []
    released: list[str] = []

    class Executor:
        supports_async_decode_prepare = True
        supports_async_decode_reclaim = True
        supports_device_decode_embedding = True
        supports_device_sampling = True
        device_topk_sampling_k = 0
        device_token = 10

        @staticmethod
        def prepared_decode_requires_token(_prepared):
            return False

        @staticmethod
        def prepare_decode(_model, batch, *, buffer_slot):
            calls.append(("prepare", buffer_slot))
            prepare_count = len([call for call in calls if call[0] == "prepare"])
            if prepare_count == 2:
                second_prepared.set()
            elif prepare_count == 3:
                third_prepared.set()
            # Step N+1 may carry an optimistically resolved old host token.  The
            # simulated persistent device state below is authoritative.
            return SimpleNamespace(slot=buffer_slot)

        @classmethod
        def dispatch_prepared_decode(cls, _model, _batch, prepared):
            calls.append(("execute", prepared.slot))
            if len([call for call in calls if call[0] == "execute"]) == 1:
                first_running.set()
                assert allow_first_finish.wait(timeout=5)
            else:
                second_dispatched.set()
            cls.device_token += 1
            return SimpleNamespace(slot=prepared.slot, token=cls.device_token)

        @staticmethod
        def reclaim_prepared_decode(pending):
            calls.append(("reclaim", pending.slot))
            if pending.slot == 1:
                first_reclaim_running.set()
                assert allow_first_reclaim.wait(timeout=5)
            return DecodeResult(
                hidden_states=None,
                logits=None,
                accepted_token_ids=[[pending.token]],
            )

        @staticmethod
        def release_finished_requests(request_ids):
            released.extend(request_ids)

    input_queue: Queue = Queue()
    output_queue: Queue = Queue()
    worker = WorkerProcess.__new__(WorkerProcess)
    worker.config = SimpleNamespace(resolve_async_scheduling=lambda: True)
    worker.input_queue = input_queue
    worker.output_queue = output_queue
    worker.profile_output_queue = None
    worker.executor = Executor()
    worker.sampler = _FixedSampler(token_id=0)
    worker.model_record = SimpleNamespace(runtime_model=model)
    worker._req_cache = {
        "req": NewRequestData("req", [1], 0.0, 1.0, None),
        "old": NewRequestData("old", [2], 0.0, 1.0, None),
    }
    worker._last_tokens = {"req": [10]}

    first = StepCommand(
        new_requests=[],
        prefill_requests=[],
        decode_requests=[DecodeRequest("req", 10, 2, [])],
        finished_request_ids=[],
        step_id=1,
    )
    second = StepCommand(
        new_requests=[],
        prefill_requests=[],
        decode_requests=[DecodeRequest("req", -1, 3, [])],
        finished_request_ids=["old"],
        step_id=2,
    )
    third = StepCommand(
        new_requests=[],
        prefill_requests=[],
        decode_requests=[DecodeRequest("req", -1, 4, [])],
        finished_request_ids=[],
        step_id=3,
    )
    thread = threading.Thread(target=worker.busy_loop)
    thread.start()
    input_queue.put(encode_command(first))
    assert first_running.wait(timeout=5)
    input_queue.put(encode_command(second))
    assert second_prepared.wait(timeout=5)
    # N+1 was prepared while N was still blocked on the simulated device.
    assert not allow_first_finish.is_set()
    # Lifecycle deltas remain on the FIFO execution lane; async preparation
    # must not release a preempted/finished slot while prior device work runs.
    assert released == []
    allow_first_finish.set()

    # N+1 is fully bound by prepare, so its device dispatch is not held behind
    # N's host-side output processing.
    assert first_reclaim_running.wait(timeout=5)
    assert second_dispatched.wait(timeout=5)
    input_queue.put(encode_command(third))
    # Step N+2 maps back to N's slot and must not prepare until reclaim has
    # finished reading that slot's captured outputs.
    assert not third_prepared.wait(timeout=0.1)
    assert output_queue.empty()
    allow_first_reclaim.set()
    assert third_prepared.wait(timeout=5)

    first_result = decode_result(output_queue.get(timeout=5))
    second_result = decode_result(output_queue.get(timeout=5))
    third_result = decode_result(output_queue.get(timeout=5))
    input_queue.put(encode_command(ShutdownCommand()))
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert first_result.new_tokens == {"req": [11]}
    # The optimistic host placeholder may still be stale here; persistent
    # device state supplies the authoritative token to N+1.
    assert second_result.new_tokens == {"req": [12]}
    assert third_result.new_tokens == {"req": [13]}
    assert released == ["old"]
    assert calls.index(("prepare", 0)) < calls.index(("execute", 0))
    assert calls.index(("execute", 0)) < calls.index(("reclaim", 0))


def test_worker_does_not_prepare_prefill_asynchronously():
    worker = WorkerProcess.__new__(WorkerProcess)
    command = StepCommand(
        new_requests=[],
        prefill_requests=[PrefillRequest("req", [1], 0, [])],
        decode_requests=[],
        finished_request_ids=[],
        step_id=1,
    )

    assert worker._prepare_step_command(command) is None


def test_worker_does_not_prepare_host_embedding_decode_asynchronously():
    worker = WorkerProcess.__new__(WorkerProcess)
    worker.executor = SimpleNamespace(supports_device_decode_embedding=False)
    command = StepCommand(
        new_requests=[],
        prefill_requests=[],
        decode_requests=[DecodeRequest("req", -1, 2, [])],
        finished_request_ids=[],
        step_id=1,
    )

    assert worker._prepare_step_command(command) is None


def test_worker_routes_decode_failures_through_device_fifo():
    worker = WorkerProcess.__new__(WorkerProcess)
    worker._apply_command_lifecycle = lambda _cmd, _entries: None
    worker._can_split_decode_output = lambda _cmd, _prepared: False
    worker._execute_step = lambda cmd, prepared_decode=None: StepResult(step_id=cmd.step_id)
    work_queue: Queue = Queue()
    output_work_queue: Queue = Queue()
    command = StepCommand(
        new_requests=[],
        prefill_requests=[],
        decode_requests=[],
        finished_request_ids=[],
        step_id=7,
    )
    work_queue.put((command, None, {}))
    work_queue.put(
        (
            serving_worker._DecodeCommandFailure(
                StepResult(new_tokens={}, error="malformed")
            ),
            None,
            {},
        )
    )
    work_queue.put((ShutdownCommand(), None, {}))

    worker._device_execution_loop(work_queue, output_work_queue)

    first = output_work_queue.get_nowait()
    second = output_work_queue.get_nowait()
    assert isinstance(first, serving_worker._CompletedStepOutput)
    assert first.result.step_id == 7
    assert isinstance(second, serving_worker._CompletedStepOutput)
    assert second.result.error == "malformed"
    assert isinstance(output_work_queue.get_nowait(), ShutdownCommand)


def test_worker_reclaims_device_accepted_tokens_after_request_cache_release():
    """A stale fused result is self-contained and must not reread request config."""
    worker = WorkerProcess.__new__(WorkerProcess)
    worker.executor = SimpleNamespace()
    worker._req_cache = {}
    new_tokens = {}

    worker._consume_decode_result(
        [DecodeRequest("released", -1, 3, [])],
        DecodeResult(hidden_states=None, logits=None, accepted_token_ids=[[7, 8]]),
        new_tokens,
    )

    assert new_tokens == {"released": [7, 8]}


@pytest.mark.parametrize("busy_loop_fails", [False, True])
def test_worker_entry_always_closes_worker(monkeypatch, busy_loop_fails):
    calls = SimpleNamespace(close=0, ready=0)

    class FakeWorker:
        def __init__(self, config, input_queue, output_queue, profile_output_queue=None):
            pass

        def init_device_and_model(self):
            return 7

        def busy_loop(self):
            if busy_loop_fails:
                raise RuntimeError("worker failed")

        def close(self):
            calls.close += 1

    monkeypatch.setattr(serving_worker, "WorkerProcess", FakeWorker)
    monkeypatch.setattr(signal, "signal", lambda *_args: None)
    ready_event = SimpleNamespace(set=lambda: setattr(calls, "ready", calls.ready + 1))
    num_pages_value = SimpleNamespace(value=0)

    serving_worker._worker_entry(
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        ready_event,
        num_pages_value,
    )

    assert num_pages_value.value == 7
    assert calls.ready >= 1
    assert calls.close == 1
