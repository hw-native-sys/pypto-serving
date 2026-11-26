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

import pytest

from pypto_serving.config.types import GenerateConfig, RuntimeConfig
from pypto_serving.observability import InMemoryStatLogger
from pypto_serving.serving.constraints import ConstraintSpec
from pypto_serving.serving.engine.async_engine import (
    EngineConfig,
    ReplicaEngineCore,
)
from pypto_serving.serving.memory.kv_cache import KvCacheManager
from pypto_serving.serving.sched.scheduler import (
    Request,
    RequestStatus,
    ScheduledRequest,
    Scheduler,
    SchedulerConfig,
    SchedulerOutput,
)
from pypto_serving.serving.server.ipc import (
    PLACEHOLDER_TOKEN,
    StepResult,
    ReleaseResult,
    decode_command,
    encode_result,
)


class _Tokenizer:
    eos_token_id: int | None = None

    def encode(self, text: str) -> list[int]:
        return [max(1, len(text))]

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(token_id) for token_id in token_ids)


def _running_decode_request(req_id="r", prompt=(1, 2), first_output=99):
    """A RUNNING request that finished prefill and has one decoded token, i.e.
    ready to schedule its next decode step (num_new_tokens_needed == 1)."""
    return Request(
        request_id=req_id,
        prompt_token_ids=list(prompt),
        max_new_tokens=8,
        num_computed_tokens=len(prompt),
        output_token_ids=[first_output],
        status=RequestStatus.RUNNING,
    )


def _tool_constraint():
    return ConstraintSpec(
        provider_id="xgrammar", format_id="deepseek_v4",
        tools=({"type": "function", "function": {"name": "shell"}},),
        tool_choice="required", reasoning=False,
    )


def _async_pipeline_core(*, num_speculative_tokens: int = 0):
    """A ReplicaEngineCore wired for async pipelining with a fake in-process
    worker: input_queue records dispatched commands, output_queue synthesises a
    deterministic StepResult (each scheduled decode samples last_token+1)."""

    manager = KvCacheManager(num_blocks=32, block_size=2, enable_prefix_cache=False)
    scheduler = Scheduler(
        SchedulerConfig(
            enable_prefix_cache=False,
            async_scheduling=True,
            num_speculative_tokens=num_speculative_tokens,
        ),
        manager,
    )
    core = ReplicaEngineCore.__new__(ReplicaEngineCore)
    core.scheduler = scheduler
    core.kv_cache_manager = manager
    core.tokenizer = _Tokenizer()
    core.config = EngineConfig(
        executor_cls="PyptoQwen14BExecutor",
        runtime_config=RuntimeConfig(num_speculative_tokens=num_speculative_tokens),
    )
    core._async_scheduling = True
    core._pending_free_ids = []
    core._worker_known_req_ids = set()
    core._request_contexts = {}
    core._batch_queue = deque()
    core._discard_result_step_ids = set()
    core._step_timeout = 300.0
    core._max_in_flight = 2
    core._step_counter = 0

    dispatched: list = []
    results: "deque" = deque()

    def _put_cmd(raw: bytes):
        cmd = decode_command(raw)
        dispatched.append(cmd)
        # Synthesise the worker result: each decode/prefill-completing req yields
        # one token = resolved_last_token + 1 (placeholder resolves from prior).
        new_tokens: dict[str, list[int]] = {}
        for dr in cmd.decode_requests:
            # Mirror the worker's placeholder resolution against our fake cache.
            last = _fake_worker_last.get(dr.request_id)
            base = dr.last_token if dr.last_token != PLACEHOLDER_TOKEN else last
            tok = base + 1
            new_tokens[dr.request_id] = [tok]
            _fake_worker_last[dr.request_id] = tok
        for pr in cmd.prefill_requests:
            # Prefill that completes the prompt samples one token.
            tok = 1000
            new_tokens[pr.request_id] = [tok]
            _fake_worker_last[pr.request_id] = tok
        if cmd.finished_request_ids:
            results.append(encode_result(ReleaseResult(cmd.finished_request_ids, cmd.step_id)))
        results.append(encode_result(StepResult(new_tokens=new_tokens, step_id=cmd.step_id)))

    def _get_result(timeout=None):
        return results.popleft()

    _fake_worker_last: dict[str, int] = {}
    core._input_queue = SimpleNamespace(put=_put_cmd)
    core._output_queue = SimpleNamespace(get=_get_result)
    return core, dispatched


@pytest.mark.parametrize("drafts,tokens,error", [
    (1, [51, 52], None), (7, [51, 52, 53], None),
    (1, [51], None), (0, [51], None), (7, [51, 52], "failed"),
])
def test_pipeline_records_actual_drafts_before_length_truncation(monkeypatch, drafts, tokens, error):
    async def run_inline(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", run_inline)
    core, commands = _async_pipeline_core(num_speculative_tokens=max(1, drafts))
    metrics = InMemoryStatLogger("test-model", [0])
    core.set_stat_logger(metrics, 0)
    request = _running_decode_request(first_output=50)
    request.max_new_tokens = 2
    core.scheduler.running.append(request)
    core.scheduler.requests[request.request_id] = request
    assert core._try_dispatch_step()
    result = StepResult(
        new_tokens={"r": tokens}, num_draft_tokens={"r": drafts},
        step_id=commands[0].step_id, error=error,
    )
    core._output_queue = SimpleNamespace(get=lambda timeout=None: encode_result(result))
    assert asyncio.run(core._await_and_apply_oldest()) is (error is None)
    counters = metrics.snapshot()["replicas"][0]["counters"]
    assert counters["draft_tokens"] == (drafts if error is None else 0)
    assert counters["accepted_tokens"] == (len(tokens) - 1 if error is None and drafts else 0)
    assert counters["speculative_drafts"] == int(error is None and drafts > 0)
    assert counters["speculative_fallbacks"] == int(error is None and drafts == 0)
    positions = metrics.snapshot()["replicas"][0]["accepted_tokens_per_pos"]
    if error is None and drafts:
        assert positions == [1] * (len(tokens) - 1) + [0] * (drafts - len(tokens) + 1)
    assert metrics.snapshot()["replicas"][0]["gauges"]["running"] == 0
    if error is None:
        assert request.output_token_ids == [50, 51]


def test_speculation_metrics_ignore_results_for_already_aborted_requests():
    core, _ = _async_pipeline_core(num_speculative_tokens=7)
    metrics = InMemoryStatLogger("model", [0], num_speculative_tokens=7)
    core.set_stat_logger(metrics, 0)
    request = _running_decode_request()
    core.scheduler.running.append(request)
    core.scheduler.requests[request.request_id] = request
    scheduled = core.scheduler.schedule()
    core.scheduler.abort_request(request.request_id)
    core._process_step_output(scheduled, {request.request_id: [1, 2, 3]}, {request.request_id: 7})
    counters = metrics.snapshot()["replicas"][0]["counters"]
    assert counters["draft_tokens"] == 0
    assert counters["accepted_tokens"] == 0


def test_constrained_abort_drops_inflight_output_and_frees_worker_before_reuse():
    core, dispatched = _async_pipeline_core(num_speculative_tokens=7)
    request = _running_decode_request(first_output=50)
    request.constraint_spec = _tool_constraint()
    core.scheduler.running.append(request)
    core.scheduler.requests[request.request_id] = request
    ctx = SimpleNamespace(queue=asyncio.Queue(), request=request, stream=True,
                          worker_release_done=asyncio.Event())
    core._request_contexts[request.request_id] = ctx

    assert core._try_dispatch_step()
    asyncio.run(core.abort_request(request.request_id))
    assert request.status is RequestStatus.FINISHED_ABORTED
    assert asyncio.run(core._await_and_apply_oldest())
    assert request.output_token_ids == [50]
    asyncio.run(core._flush_pending_frees())
    assert dispatched[-1].finished_request_ids == [request.request_id]
    assert request.request_id not in core._worker_known_req_ids
    output = asyncio.run(anext(core._iterate_request_output(ctx)))
    assert output.finish_reason == "FINISHED_ABORTED"
    assert not core._pending_free_ids and not core._request_contexts

    next_request = Request("next", [3], max_new_tokens=1)
    core.scheduler.add_request(next_request)
    assert core._try_dispatch_step()
    assert [item.request_id for item in dispatched[-1].new_requests] == ["next"]


def test_constrained_step_error_cleans_up_before_next_request():
    core, dispatched = _async_pipeline_core(num_speculative_tokens=7)
    request = _running_decode_request(first_output=50)
    request.constraint_spec = _tool_constraint()
    core.scheduler.running.append(request)
    core.scheduler.requests[request.request_id] = request
    ctx = SimpleNamespace(queue=asyncio.Queue(), request=request, stream=True,
                          worker_release_done=asyncio.Event())
    core._request_contexts[request.request_id] = ctx
    assert core._try_dispatch_step()

    results = deque([encode_result(StepResult(
        new_tokens={}, error="invalid grammar mask", step_id=dispatched[0].step_id,
    ))])
    core._output_queue = SimpleNamespace(get=lambda timeout=None: results.popleft())
    assert asyncio.run(core._await_and_apply_oldest()) is False
    assert ctx.queue.get_nowait().finish_reason == "error"
    assert request.status is RequestStatus.FINISHED_ABORTED

    def put(raw):
        command = decode_command(raw)
        dispatched.append(command)
        results.append(encode_result(ReleaseResult(command.finished_request_ids, command.step_id)))
        results.append(encode_result(StepResult(new_tokens={}, step_id=command.step_id)))

    core._input_queue = SimpleNamespace(put=put)
    asyncio.run(core._flush_pending_frees())
    assert ctx.worker_release_done.is_set()
    assert dispatched[-1].finished_request_ids == [request.request_id]
    assert not core._pending_free_ids and not core._worker_known_req_ids

    next_request = Request("next", [3], max_new_tokens=1)
    core.scheduler.add_request(next_request)
    assert core._try_dispatch_step()
    assert [item.request_id for item in dispatched[-1].new_requests] == ["next"]


def test_rejected_request_does_not_leave_an_engine_context():
    core, _ = _async_pipeline_core()
    config = SimpleNamespace(
        max_new_tokens=1, stop=None, ignore_eos=True, temperature=0,
        top_p=1, top_k=None, seed=None, stream=True,
    )

    async def consume():
        async for _ in core.add_request(
            "too-long", "ignored", config, prompt_token_ids=list(range(4096)),
            constraint_spec=_tool_constraint(),
        ):
            pass

    with pytest.raises(ValueError, match="leaves no room for generation"):
        asyncio.run(consume())
    assert not core._request_contexts and not core.scheduler.requests
    assert not core._pending_free_ids


@pytest.mark.parametrize("executor_cls,speculative_tokens,expected_min_tokens", [
    ("PyptoQwen14BExecutor", 0, 2),
    ("PyptoDeepSeekV4Executor", 1, 0),
    ("PyptoDeepSeekV4DSparkExecutor", 7, 2),
])
def test_min_tokens_reaches_worker_registration(executor_cls, speculative_tokens, expected_min_tokens):
    core, dispatched = _async_pipeline_core(num_speculative_tokens=speculative_tokens)
    core.config.executor_cls = executor_cls
    core.tokenizer.eos_token_id = 9

    class Queued(Exception):
        pass

    async def enqueue():
        async for _ in core.add_request(
            "eos-policy", "prompt", GenerateConfig(max_new_tokens=4, min_tokens=2,
                                                     ignore_eos=True),
            prompt_token_ids=[1], on_queued=lambda: (_ for _ in ()).throw(Queued()),
        ):
            pass

    with pytest.raises(Queued):
        asyncio.run(enqueue())
    request = core.scheduler.requests["eos-policy"]
    assert request.eos_token_id == 9
    assert request.ignore_eos is True
    assert request.min_tokens == expected_min_tokens
    assert core._try_dispatch_step()
    assert len(dispatched[0].new_requests) == 1
    registration = dispatched[0].new_requests[0]
    assert registration.min_tokens == expected_min_tokens
    assert registration.eos_token_id == 9
    assert registration.initial_output_tokens == 0


def test_restarted_request_registers_committed_output_length(monkeypatch):
    core, _ = _async_pipeline_core()
    request = Request(
        request_id="restarted", prompt_token_ids=[1, 2], max_new_tokens=6,
        min_tokens=3, eos_token_id=9, output_token_ids=[7, 8],
        status=RequestStatus.PREEMPTED,
    )
    core._worker_known_req_ids.add(request.request_id)
    context = SimpleNamespace(worker_release_done=asyncio.Event(), worker_release_ack=asyncio.Event())
    context.worker_release_done.set()
    context.worker_release_ack.set()
    core._request_contexts[request.request_id] = context
    scheduled = SchedulerOutput(scheduled_requests=[
        ScheduledRequest(request=request, num_new_tokens=1, is_prefill=True),
    ])

    command = core._build_step_command(scheduled, [request.request_id])

    assert command.finished_request_ids == [request.request_id]
    assert len(command.new_requests) == 1
    assert command.new_requests[0].initial_output_tokens == 2
    assert command.new_requests[0].min_tokens == 3
    assert not context.worker_release_done.is_set() and not context.worker_release_ack.is_set()
    # A release followed by re-registration in one FIFO command only retires
    # the old incarnation; its reply must not acknowledge the new live state.
    monkeypatch.setattr(core.scheduler, "schedule", lambda **kwargs: scheduled)
    core._pending_free_ids = [request.request_id]
    assert core._try_dispatch_step()
    assert not getattr(core, "_worker_free_ids_by_step", {})
    assert asyncio.run(core._await_and_apply_oldest())
    assert not context.worker_release_done.is_set() and not context.worker_release_ack.is_set()


def test_step_command_carries_stable_cross_layer_invocation_identity():
    core, _ = _async_pipeline_core()
    request = _running_decode_request()
    core.scheduler.running.append(request)
    core.scheduler.requests[request.request_id] = request
    first = core._build_step_command(
        SchedulerOutput(scheduled_requests=[ScheduledRequest(
            request=request, num_new_tokens=1, is_prefill=False,
        )]),
        [],
        step_id=1,
    )
    second = core._build_step_command(
        SchedulerOutput(scheduled_requests=[ScheduledRequest(
            request=request, num_new_tokens=1, is_prefill=False,
        )]),
        [],
        step_id=3,
    )
    assert first.invocation_id.endswith(":1")
    assert second.invocation_id.endswith(":3")
    assert first.execution_slot == second.execution_slot == 1
    assert first.pipeline_generation == 1
    assert second.pipeline_generation == 2


def test_async_pipeline_dispatches_two_steps_before_applying_first(monkeypatch):
    """Depth-2: the loop dispatches step N+1 while step N is still in flight,
    so two commands reach the worker before the first result is applied."""

    async def run_inline(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", run_inline)
    core, dispatched = _async_pipeline_core()
    req = _running_decode_request(prompt=(1, 2), first_output=50)
    core.scheduler.running.append(req)
    core.scheduler.requests[req.request_id] = req

    # Dispatch twice without applying — mirrors the loop filling the queue.
    assert core._try_dispatch_step() is True
    assert core._try_dispatch_step() is True
    assert len(core._batch_queue) == 2  # two steps in flight
    assert len(dispatched) == 2

    # Second dispatched decode carries a PLACEHOLDER (its input token is the
    # not-yet-applied result of the first in-flight step).
    second = dispatched[1]
    assert second.decode_requests[0].last_token == PLACEHOLDER_TOKEN

    # Now drain both in order; output must be the greedy chain 51, 52.
    asyncio.run(core._await_and_apply_oldest())
    asyncio.run(core._await_and_apply_oldest())
    assert req.output_token_ids == [50, 51, 52]
    assert req.num_output_placeholders == 0


def test_async_max_tokens_prevents_tail_decode_dispatch(monkeypatch):
    async def run_inline(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", run_inline)
    core, dispatched = _async_pipeline_core()
    request = _running_decode_request(prompt=(1, 2), first_output=50)
    request.max_new_tokens = 3
    core.scheduler.running.append(request)
    core.scheduler.requests[request.request_id] = request

    # Tokens 51 and 52 are the only two decodes needed to reach max_new_tokens.
    assert core._try_dispatch_step() is True
    assert core._try_dispatch_step() is True
    assert len(dispatched) == 2

    assert asyncio.run(core._await_and_apply_oldest()) is True
    # Token 52 is still in flight and already consumes the final output slot.
    assert core._try_dispatch_step() is False
    assert len(dispatched) == 2

    assert asyncio.run(core._await_and_apply_oldest()) is True
    assert request.output_token_ids == [50, 51, 52]
    assert request.status is RequestStatus.FINISHED_LENGTH
    assert request.num_output_placeholders == 0


def test_async_pipeline_waits_for_terminal_prefill_before_first_decode(monkeypatch):
    async def run_inline(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", run_inline)
    core, dispatched = _async_pipeline_core(num_speculative_tokens=1)
    request = Request(
        request_id="r",
        prompt_token_ids=[1, 2],
        max_new_tokens=4,
    )
    core.scheduler.add_request(request)

    assert core._try_dispatch_step() is True
    assert dispatched[0].prefill_requests
    assert request.terminal_prefill_in_flight

    # Filling pipeline slot N+1 must stop at the terminal-prefill boundary.
    assert core._try_dispatch_step() is False
    assert len(dispatched) == 1

    assert asyncio.run(core._await_and_apply_oldest()) is True
    assert not request.terminal_prefill_in_flight
    assert core._try_dispatch_step() is True
    assert dispatched[1].decode_requests


def test_delayed_prefill_rejection_does_not_stop_valid_dispatch():
    core, dispatched = _async_pipeline_core()
    manager = KvCacheManager(num_blocks=8, block_size=2, enable_prefix_cache=True)
    core.scheduler = Scheduler(
        SchedulerConfig(
            max_num_scheduled_tokens=4,
            max_prefill_tokens_per_request=4,
            max_seq_len=8,
            enable_prefix_cache=True,
            enable_chunk_prefill=False,
            async_scheduling=True,
        ),
        manager,
    )
    cached_block = manager.allocate_blocks(1)[0]
    manager.cache_block(cached_block, manager.compute_block_hashes([1, 2])[0])
    manager.release(cached_block)

    rejected = Request("rejected", [1, 2, 3, 4, 5, 6, 7], max_new_tokens=1)
    valid = Request("valid", [9, 8, 7], max_new_tokens=1)
    core.scheduler.add_request(rejected)
    core.scheduler.add_request(valid)
    rejection_queue = asyncio.Queue()
    core._request_contexts[rejected.request_id] = SimpleNamespace(queue=rejection_queue)

    assert core._try_dispatch_step() is True

    error = rejection_queue.get_nowait()
    assert isinstance(error, ValueError)
    assert "uncached prompt length 5" in str(error)
    assert [request.request_id for request in core.scheduler.running] == [valid.request_id]
    assert dispatched[0].prefill_requests[0].request_id == valid.request_id


def test_async_decode_seq_len_excludes_inflight_placeholders():
    """seq_len must be the context length for THIS step, not req.num_tokens.

    Under async scheduling req.num_tokens also counts in-flight placeholders, so
    using it inflates seq_len past the KV actually written — the kernel then
    computes shifted positions (observed on device as duplicated/misplaced tokens
    with chunked prefill at pipeline depth 2).
    """
    core, dispatched = _async_pipeline_core()
    # prompt=2 tokens, one decoded token already -> next decode covers position 3.
    req = _running_decode_request(prompt=(1, 2), first_output=50)
    core.scheduler.running.append(req)
    core.scheduler.requests[req.request_id] = req

    assert core._try_dispatch_step() is True
    first = dispatched[0].decode_requests[0]
    # 2 prompt positions computed + the 1 token this step decodes = 3. This is
    # exactly what the synchronous path sent via req.num_tokens.
    assert first.seq_len == 3

    # Dispatch a second step while the first is still in flight: its seq_len must
    # advance by exactly one position, NOT jump by the extra placeholder.
    assert core._try_dispatch_step() is True
    second = dispatched[1].decode_requests[0]
    assert second.seq_len == 4
    # Guard the specific regression: req.num_tokens is now inflated by the two
    # in-flight placeholders, so seq_len must differ from it.
    assert req.num_output_placeholders == 2
    assert second.seq_len < req.num_tokens


def test_eos_from_step_n_keeps_n_plus_1_but_prevents_n_plus_2(monkeypatch):
    """EOS is discovered from N output: N+1 is stale, N+2 is not scheduled."""

    async def run_inline(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", run_inline)
    core, dispatched = _async_pipeline_core()
    req = _running_decode_request(prompt=(1, 2), first_output=50)
    req.eos_token_id = 51
    core.scheduler.running.append(req)
    core.scheduler.requests[req.request_id] = req

    assert core._try_dispatch_step() is True  # N
    assert core._try_dispatch_step() is True  # N+1, prepared optimistically
    assert len(dispatched) == 2

    assert asyncio.run(core._await_and_apply_oldest()) is True
    assert req.status is RequestStatus.FINISHED_EOS
    # Processing N output affects the next schedule decision, not the already
    # dispatched N+1 snapshot.
    assert core._try_dispatch_step() is False
    assert len(dispatched) == 2

    # N+1 still drains in FIFO order, but its token is discarded for finished A.
    assert asyncio.run(core._await_and_apply_oldest()) is True
    assert req.output_token_ids == [50, 51]


def test_async_pipeline_drains_stale_result_after_error(monkeypatch):
    """If the oldest in-flight step errors while a later step is queued, the
    later step's result (already in transit from the FIFO worker) must be drained
    and NOT misapplied to a subsequent batch."""
    from pypto_serving.serving.server.ipc import decode_result

    async def run_inline(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", run_inline)
    core, _dispatched = _async_pipeline_core()
    req = _running_decode_request(prompt=(1, 2), first_output=50)
    core.scheduler.running.append(req)
    core.scheduler.requests[req.request_id] = req
    ctx = SimpleNamespace(queue=asyncio.Queue(), request=req, stream=True,
                          worker_release_done=asyncio.Event())
    core._request_contexts = {"r": ctx}

    # Two steps dispatched: both commands reached the FIFO worker, so both
    # results are already in transit.
    assert core._try_dispatch_step() is True
    assert core._try_dispatch_step() is True
    assert len(core._batch_queue) == 2
    first_step_id, second_step_id = core._batch_queue[0][0], core._batch_queue[1][0]

    # Drive the output queue by hand: step N errors, step N+1's (now stale)
    # result sits behind it, then a fresh live step (id=99) lands.
    outq = deque(
        [
            encode_result(StepResult(new_tokens={}, error="boom", step_id=first_step_id)),
            encode_result(StepResult(new_tokens={"r": [51]}, step_id=second_step_id)),
            encode_result(StepResult(new_tokens={"r": [77]}, step_id=99)),
        ]
    )
    core._output_queue = SimpleNamespace(get=lambda timeout=None: outq.popleft())

    # Apply the oldest: it errors -> both in-flight batches discarded, the
    # second step's id recorded for draining, request aborted with an error.
    assert asyncio.run(core._await_and_apply_oldest()) is False
    assert not core._batch_queue
    assert second_step_id in core._discard_result_step_ids
    tok = ctx.queue.get_nowait()
    assert core._request_contexts == {"r": ctx}
    assert not ctx.worker_release_done.is_set()
    assert tok.finished is True and tok.finish_reason == "error"

    # Next live fetch must SKIP the stale second-step result and return step 99,
    # leaving the discard set empty (no stale result misapplied).
    got = decode_result(asyncio.run(core._get_live_result()))
    assert got.step_id == 99
    assert core._discard_result_step_ids == set()


@pytest.fixture
def observed_pipeline(tmp_path):
    from pypto_serving.observability.dplb_dfx import DplbDfxConfig, DplbDfxRecorder

    core, commands = _async_pipeline_core(num_speculative_tokens=2)
    recorder = DplbDfxRecorder(DplbDfxConfig(level="load"), tmp_path)
    core._dplb_dfx = recorder
    core.kv_cache_manager.dplb_dfx = recorder
    core._dfx_dispatch_ordinal = 0
    core._dfx_dispatch_ordinals = {}
    assert recorder.start("first")
    request = _running_decode_request(first_output=50)
    core.scheduler.running.append(request)
    core.scheduler.requests[request.request_id] = request
    yield core, commands, recorder, request
    assert recorder.close()


def _dfx_rows(recorder):
    import json

    assert recorder.close()
    return [json.loads(line) for line in recorder.sink.path.read_text().splitlines()]


def test_dfx_pipeline_pairs_two_inflight_commands(observed_pipeline):
    core, commands, recorder, request = observed_pipeline
    assert core._try_dispatch_step()
    assert core._try_dispatch_step()
    assert len(core._batch_queue) == 2
    asyncio.run(core._await_and_apply_oldest())
    asyncio.run(core._await_and_apply_oldest())
    rows = _dfx_rows(recorder)
    plans = [row for row in rows if row["event"] == "step_plan"]
    outcomes = [row for row in rows if row["event"] == "step_outcome"]
    assert [row["step_id"] for row in plans] == [cmd.step_id for cmd in commands]
    assert [row["dispatch_ordinal"] for row in plans] == [row["dispatch_ordinal"] for row in outcomes]
    assert plans[0]["requests"][0]["num_computed_tokens"] == 2
    assert plans[1]["requests"][0]["num_computed_tokens"] == 5
    assert request.output_token_ids == [50, 51, 52]
    assert not core._dfx_dispatch_ordinals


def test_dfx_cancelled_request_keeps_original_identity(observed_pipeline):
    core, _, recorder, request = observed_pipeline
    assert core._try_dispatch_step()
    core.scheduler.abort_request(request.request_id)
    assert asyncio.run(core._await_and_apply_oldest())
    rows = _dfx_rows(recorder)
    plan = next(row for row in rows if row["event"] == "step_plan")
    outcome = next(row for row in rows if row["event"] == "step_outcome")
    terminal = next(row for row in rows if row["event"] == "request_terminal")
    assert outcome["requests"][0]["request_uid"] == plan["requests"][0]["request_uid"]
    assert terminal["request_uid"] == plan["requests"][0]["request_uid"]
    assert not recorder._requests
    assert request.output_token_ids == [50]


def test_dfx_late_result_does_not_enter_next_capture(observed_pipeline):
    core, _, recorder, _ = observed_pipeline
    assert core._try_dispatch_step()
    assert recorder.close()
    assert recorder.start("second")
    assert asyncio.run(core._await_and_apply_oldest())
    rows = _dfx_rows(recorder)
    assert not any(row["event"] == "step_outcome" for row in rows)
    assert not recorder._requests
    assert not core._dfx_dispatch_ordinals


def test_dfx_midflight_start_does_not_invent_a_plan(observed_pipeline):
    core, _, recorder, _ = observed_pipeline
    assert recorder.close()
    assert core._try_dispatch_step()
    assert recorder.start("second")
    assert asyncio.run(core._await_and_apply_oldest())
    assert not any(row["event"] == "step_outcome" for row in _dfx_rows(recorder))


@pytest.mark.parametrize("timeout", [False, True])
def test_dfx_failed_pipeline_has_error_and_discard_outcomes(observed_pipeline, timeout):
    import queue

    core, commands, recorder, _ = observed_pipeline
    assert core._try_dispatch_step()
    assert core._try_dispatch_step()

    def result(timeout=None):
        if timeout_failure:
            raise queue.Empty
        return encode_result(StepResult(new_tokens={}, step_id=commands[0].step_id, error="injected"))

    timeout_failure = timeout
    core._output_queue = SimpleNamespace(get=result)
    assert not asyncio.run(core._await_and_apply_oldest())
    outcomes = [row for row in _dfx_rows(recorder) if row["event"] == "step_outcome"]
    assert [row["status"] for row in outcomes] == ["error", "discarded"]
    assert not core._dfx_dispatch_ordinals


def test_dfx_fixed_input_replay_matches_disabled_commands_and_tokens(tmp_path):
    from pypto_serving.observability.dplb_dfx import DplbDfxConfig, DplbDfxRecorder
    from pypto_serving.serving.server.ipc import encode_command

    runs = []
    for observed in (False, True):
        core, commands = _async_pipeline_core(num_speculative_tokens=2)
        recorder = None
        if observed:
            recorder = DplbDfxRecorder(DplbDfxConfig(level="load"), tmp_path)
            core._dplb_dfx = recorder
            core.kv_cache_manager.dplb_dfx = recorder
            core._dfx_dispatch_ordinal = 0
            core._dfx_dispatch_ordinals = {}
            assert recorder.start("replay")
        request = _running_decode_request(first_output=50)
        core.scheduler.running.append(request)
        core.scheduler.requests[request.request_id] = request
        for _ in range(2):
            assert core._try_dispatch_step()
        for _ in range(2):
            assert asyncio.run(core._await_and_apply_oldest())
        runs.append(([encode_command(cmd) for cmd in commands], list(request.output_token_ids)))
        if recorder is not None:
            assert recorder.close()
            assert not core._dfx_step_contexts
    assert runs[0] == runs[1]


def test_dfx_preemption_reuses_uid_and_preserves_original_epoch(observed_pipeline):
    core, _, recorder, request = observed_pipeline
    assert core._try_dispatch_step()
    # The real allocator/scheduler preemption path resets the request while its
    # worker result remains queued, before any re-admission is allowed.
    victim = core.scheduler._preempt_lowest_priority(Request("other", [1], 1), set(), {}, SchedulerOutput())
    assert victim["request"] is request
    recorder.request_fields(request.request_id, acquire=True)
    assert asyncio.run(core._await_and_apply_oldest())
    rows = _dfx_rows(recorder)
    plan = next(row for row in rows if row["event"] == "step_plan")
    outcome = next(row for row in rows if row["event"] == "step_outcome")
    assert outcome["requests"][0]["request_uid"] == plan["requests"][0]["request_uid"]
    assert outcome["requests"][0]["admission_epoch"] == plan["requests"][0]["admission_epoch"]
    assert outcome["requests"][0]["result_disposition"] == "discarded"


def test_dfx_stop_explains_step_already_waiting_for_worker(observed_pipeline):
    core, commands, recorder, _ = observed_pipeline
    core._dfx_profile_lock = asyncio.Lock()

    async def run():
        entered, release = asyncio.Event(), asyncio.Event()

        async def held_result():
            entered.set()
            await release.wait()
            return encode_result(StepResult(new_tokens={"r": [51]}, step_id=commands[0].step_id))

        async def acknowledge(_active):
            return None

        core._get_live_result = held_result
        core._set_worker_profile_active = acknowledge
        assert core._try_dispatch_step()
        waiting = asyncio.create_task(core._await_and_apply_oldest())
        await entered.wait()
        assert not core._batch_queue
        await core.stop_profile()
        release.set()
        assert await waiting

    asyncio.run(run())
    end = next(row for row in _dfx_rows(recorder) if row["event"] == "capture_end")
    assert end["in_flight_step_ids"] == [commands[0].step_id]


def test_dfx_reused_request_id_does_not_relabel_old_result(observed_pipeline):
    core, _, recorder, request = observed_pipeline
    assert core._try_dispatch_step()
    core.scheduler.abort_request(request.request_id)
    replacement = Request(request.request_id, [2, 3], max_new_tokens=4)
    core.scheduler.add_request(replacement)
    assert asyncio.run(core._await_and_apply_oldest())
    rows = _dfx_rows(recorder)
    planned = next(row for row in rows if row["event"] == "step_plan")["requests"][0]
    actual = next(row for row in rows if row["event"] == "step_outcome")["requests"][0]
    arrived = next(row for row in rows if row["event"] == "request_arrived")
    assert actual["request_uid"] == planned["request_uid"]
    assert actual["request_uid"] != arrived["request_uid"]
    assert replacement.output_token_ids == []
