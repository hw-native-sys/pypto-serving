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
)
from pypto_serving.serving.memory.kv_cache import KvCacheManager
from pypto_serving.serving.sched.scheduler import (
    Request,
    RequestStatus,
    Scheduler,
    SchedulerConfig,
)
from pypto_serving.serving.server.ipc import (
    ExternalKVTransferResult,
    PLACEHOLDER_TOKEN,
    StepResult,
    decode_command,
    encode_result,
)


class _Tokenizer:
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
    core.config = SimpleNamespace(executor_cls="PyptoQwen14BExecutor")
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
        results.append(encode_result(StepResult(new_tokens=new_tokens, step_id=cmd.step_id)))

    def _get_result(timeout=None):
        return results.popleft()

    _fake_worker_last: dict[str, int] = {}
    core._input_queue = SimpleNamespace(put=_put_cmd)
    core._output_queue = SimpleNamespace(get=_get_result)
    return core, dispatched


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
    external_completions = []
    core.scheduler.finish_external_cache_save = (
        lambda job_id, *, succeeded: external_completions.append((job_id, succeeded))
    )
    req = _running_decode_request(prompt=(1, 2), first_output=50)
    core.scheduler.running.append(req)
    core.scheduler.requests[req.request_id] = req
    core._request_contexts = {"r": SimpleNamespace(queue=asyncio.Queue(), request=req, stream=True)}

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
            encode_result(
                StepResult(
                    new_tokens={"r": [51]},
                    step_id=second_step_id,
                    external_cache_completions=[
                        ExternalKVTransferResult(
                            job_id="save-1",
                            request_id="r",
                            operation="save",
                            succeeded=True,
                        )
                    ],
                )
            ),
            encode_result(StepResult(new_tokens={"r": [77]}, step_id=99)),
        ]
    )
    core._output_queue = SimpleNamespace(get=lambda timeout=None: outq.popleft())

    # Apply the oldest: it errors -> both in-flight batches discarded, the
    # second step's id recorded for draining, request aborted with an error.
    assert asyncio.run(core._await_and_apply_oldest()) is False
    assert not core._batch_queue
    assert second_step_id in core._discard_result_step_ids
    tok = core._request_contexts["r"].queue.get_nowait()
    assert tok.finished is True and tok.finish_reason == "error"

    # Next live fetch must SKIP the stale second-step result and return step 99,
    # leaving the discard set empty (no stale result misapplied).
    got = decode_result(asyncio.run(core._get_live_result()))
    assert got.step_id == 99
    assert core._discard_result_step_ids == set()
    assert external_completions == [("save-1", True)]
