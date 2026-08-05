# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

from pypto_serving.serving.memory.kv_cache import KVCacheCapacityError, KvCacheManager
from pypto_serving.serving.memory.prefix_cache import PrefixCacheStats, RadixKey
from pypto_serving.serving.memory.request_kv_pool import RequestKVPool

if TYPE_CHECKING:
    from pypto_serving.serving.memory.prefix_cache import TreeNode

logger = logging.getLogger(__name__)


class RequestStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    PREEMPTED = auto()
    FINISHED_EOS = auto()
    FINISHED_LENGTH = auto()
    FINISHED_STOP = auto()
    FINISHED_ABORTED = auto()

    @property
    def is_finished(self) -> bool:
        return self in (
            RequestStatus.FINISHED_EOS,
            RequestStatus.FINISHED_LENGTH,
            RequestStatus.FINISHED_STOP,
            RequestStatus.FINISHED_ABORTED,
        )


@dataclass
class SchedulerConfig:
    max_num_running_reqs: int = 32
    max_num_scheduled_tokens: int = 4096
    long_prefill_token_threshold: int = 2048
    max_seq_len: int = 4096
    # Feature flags
    enable_prefix_cache: bool = True
    enable_chunk_prefill: bool = True
    prefix_cache_backend: str = "hash"
    enable_radix_in_batch_dedup: bool = True
    num_speculative_tokens: int = 0
    # Async (pipelined) scheduling: schedule step N+1 before step N's sampled
    # token returns, advancing request state optimistically via placeholders.
    async_scheduling: bool = False

    def __post_init__(self) -> None:
        if self.num_speculative_tokens < 0:
            raise ValueError("num_speculative_tokens must be non-negative")


@dataclass
class Request:
    request_id: str
    prompt_token_ids: list[int]
    max_new_tokens: int
    model_id: str = ""
    arrival_time: float = field(default_factory=time.time)
    status: RequestStatus = RequestStatus.WAITING
    num_computed_tokens: int = 0
    output_token_ids: list[int] = field(default_factory=list)
    stop_strings: tuple[str, ...] = ()
    eos_token_id: int | None = None
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int | None = None
    cached_block_ids: list[int] = field(default_factory=list)
    allocated_block_ids: list[int] = field(default_factory=list)
    allocated_group_block_ids: dict[str, list[int]] = field(default_factory=dict)
    cache_partition: int | None = None
    block_hashes: list[int] = field(default_factory=list)
    num_blocks_cached: int = 0  # Track how many blocks have been published to prefix cache
    # Async scheduling: tokens scheduled optimistically but not yet sampled.
    # Stands in for output tokens still in flight so the next schedule() advances
    # correctly; decremented as real tokens are applied in update_from_output.
    num_output_placeholders: int = 0
    prefix_indices: list[int] = field(default_factory=list)
    last_node: "TreeNode | None" = None
    cache_protected_len: int = 0
    kv_committed_len: int = 0

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_tokens(self) -> int:
        # Placeholders count as (not-yet-materialised) output tokens so that
        # num_new_tokens_needed and is_prefill stay consistent when the next step
        # is scheduled before the in-flight token has been appended.
        return self.num_prompt_tokens + len(self.output_token_ids) + self.num_output_placeholders

    @property
    def num_new_tokens_needed(self) -> int:
        return self.num_tokens - self.num_computed_tokens

    @property
    def is_prefill(self) -> bool:
        return self.num_computed_tokens < self.num_prompt_tokens

    @property
    def all_token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids


@dataclass
class ScheduledRequest:
    request: Request
    num_new_tokens: int
    is_prefill: bool
    num_computed_tokens: int = 0
    block_ids: list[int] = field(default_factory=list)
    block_ids_by_group: dict[str, list[int]] = field(default_factory=dict)
    cache_partition: int | None = None
    resumed_from_preemption: bool = False


@dataclass
class SchedulerOutput:
    scheduled_requests: list[ScheduledRequest] = field(default_factory=list)
    preempted_requests: list[Request] = field(default_factory=list)
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0

    @property
    def is_empty(self) -> bool:
        return len(self.scheduled_requests) == 0


@dataclass
class RequestOutput:
    request_id: str
    new_token_id: int | None = None
    finished: bool = False
    finish_reason: str = ""


class Scheduler:
    """Continuous batching scheduler with chunked prefill and preemption."""

    def __init__(self, config: SchedulerConfig, kv_cache_manager: KvCacheManager) -> None:
        if config.prefix_cache_backend not in ("hash", "radix"):
            raise ValueError("prefix_cache_backend must be 'hash' or 'radix'")
        self.config = config
        self.kv_cache_manager = kv_cache_manager
        if self.kv_cache_manager.has_groups and self.config.enable_prefix_cache:
            raise ValueError("Prefix caching is not supported with grouped KV caches")
        self.request_kv_pool = RequestKVPool(kv_cache_manager.block_size)
        self.prefix_cache_stats = PrefixCacheStats()
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.requests: dict[str, Request] = {}

    def add_request(self, request: Request) -> None:
        prompt_len = len(request.prompt_token_ids)
        max_seq_len = self.config.max_seq_len
        if prompt_len > max_seq_len:
            # vLLM-style: reject rather than silently truncate. A prompt that
            # cannot fit max_seq_len can never be served, so failing loudly is
            # safer than silently dropping the tail of the prompt.
            raise ValueError(
                f"Request {request.request_id} prompt length {prompt_len} "
                f"exceeds max_seq_len {max_seq_len}; request rejected."
            )
        # Cap generation so prompt + generated tokens never exceed max_seq_len
        # (vLLM-style: effective max_tokens = max_seq_len - prompt_len). This
        # keeps every request within the KV-cache capacity budgeted per request
        # and avoids overflow-driven preemption.
        remaining = max_seq_len - prompt_len
        if remaining <= 0:
            raise ValueError(
                f"Request {request.request_id} prompt length {prompt_len} "
                f"leaves no room for generation within max_seq_len {max_seq_len}; "
                f"request rejected."
            )
        if request.max_new_tokens > remaining:
            logger.warning(
                "Request %s: capping max_new_tokens %d -> %d to fit max_seq_len %d "
                "(prompt_len=%d).",
                request.request_id, request.max_new_tokens, remaining,
                max_seq_len, prompt_len,
            )
            request.max_new_tokens = remaining
        if self.config.enable_prefix_cache and not self._use_radix_cache:
            request.block_hashes = self.kv_cache_manager.compute_block_hashes(request.prompt_token_ids)
        request.status = RequestStatus.WAITING
        self.waiting.append(request)
        self.requests[request.request_id] = request

    def abort_request(self, request_id: str) -> None:
        request = self.requests.get(request_id)
        if request is None:
            return
        request.status = RequestStatus.FINISHED_ABORTED
        self._free_request_blocks(request)
        self.running = [r for r in self.running if r.request_id != request_id]
        self.waiting = deque(r for r in self.waiting if r.request_id != request_id)
        del self.requests[request_id]

    def finish_request(self, request_id: str, status: RequestStatus) -> None:
        """Mark a running request as finished and free its resources."""
        request = self.requests.get(request_id)
        if request is None:
            return
        request.status = status
        self._free_request_blocks(request)
        self.running = [r for r in self.running if r.request_id != request_id]

    def has_work(self) -> bool:
        return len(self.running) > 0 or len(self.waiting) > 0

    def schedule(self) -> SchedulerOutput:
        output = SchedulerOutput()
        token_budget = self.config.max_num_scheduled_tokens
        grouped_phase = self._grouped_cache_phase()

        # Phase 1: schedule RUNNING requests (decode or resumed prefill)
        scheduled_req_ids: set[str] = set()
        num_scheduled_tokens: dict[str, int] = {}
        running_to_keep: list[Request] = []
        for request in self.running:
            # A request later in this snapshot may have been preempted while
            # scheduling an earlier request. Do not schedule it again from the
            # stale iteration snapshot.
            if request.status is RequestStatus.PREEMPTED:
                continue
            if grouped_phase is not None and request.is_prefill != (grouped_phase == "prefill"):
                running_to_keep.append(request)
                continue
            num_new = request.num_new_tokens_needed
            if num_new <= 0:
                running_to_keep.append(request)
                continue

            if self.config.enable_chunk_prefill and self.config.long_prefill_token_threshold > 0:
                num_new = min(num_new, self.config.long_prefill_token_threshold)
            num_new = min(num_new, token_budget)

            if num_new <= 0:
                running_to_keep.append(request)
                continue

            is_prefill = request.is_prefill
            speculative_tokens = (
                self.config.num_speculative_tokens
                if not is_prefill and request.temperature <= 0.0
                else 0
            )
            scheduled_tokens = num_new + speculative_tokens
            if scheduled_tokens > token_budget:
                running_to_keep.append(request)
                continue

            if not self._try_allocate_request_blocks(request, scheduled_tokens):
                preempted = self._preempt_lowest_priority(
                    request, scheduled_req_ids, num_scheduled_tokens, output
                )
                if preempted is None:
                    running_to_keep.append(request)
                    continue
                token_budget += preempted.get("returned_tokens", 0)
                output.preempted_requests.append(preempted["request"])
                if not self._try_allocate_request_blocks(request, scheduled_tokens):
                    running_to_keep.append(request)
                    continue

            all_block_ids = self._request_block_ids(request)
            output.scheduled_requests.append(
                ScheduledRequest(
                    request=request,
                    num_new_tokens=num_new,
                    is_prefill=is_prefill,
                    num_computed_tokens=request.num_computed_tokens,
                    block_ids=list(all_block_ids),
                    block_ids_by_group={
                        name: list(block_ids)
                        for name, block_ids in request.allocated_group_block_ids.items()
                    },
                    cache_partition=request.cache_partition,
                )
            )
            scheduled_req_ids.add(request.request_id)
            num_scheduled_tokens[request.request_id] = scheduled_tokens
            if is_prefill:
                output.num_prefill_tokens += num_new
                if self._use_radix_cache:
                    self.prefix_cache_stats.scheduled_prefill_tokens += num_new
            else:
                output.num_decode_tokens += scheduled_tokens
            token_budget -= scheduled_tokens
            running_to_keep.append(request)

        # Victims that appeared earlier in the iteration may already be in
        # running_to_keep. They now live in the waiting queue and must not be
        # retained in both queues.
        self.running = [
            request
            for request in running_to_keep
            if request.status is not RequestStatus.PREEMPTED
        ]

        # Fixed-shape DeepSeek decode rows use otherwise-free cache blocks as
        # scratch space. Do not admit a new prefill wave while a decode wave is
        # active, or those padding rows could overwrite the new request's cache.
        if grouped_phase == "decode":
            return output

        # Phase 2: schedule WAITING requests (new prefill)
        remaining_waiting: deque[Request] = deque()
        while self.waiting and token_budget > 0:
            if len(self.running) >= self.config.max_num_running_reqs:
                break

            request = self.waiting.popleft()

            # Prefix cache lookup
            if self._use_radix_cache:
                self._match_radix_request(request)
                cached_blocks = []
                if self._should_defer_for_radix_producer(request):
                    self.prefix_cache_stats.in_batch_deferred += 1
                    self._release_radix_request(request)
                    remaining_waiting.append(request)
                    continue
            elif self.config.enable_prefix_cache:
                cached_blocks = self.kv_cache_manager.get_computed_blocks(request.prompt_token_ids)
                if cached_blocks:
                    request.cached_block_ids = [b.block_id for b in cached_blocks]
                    request.num_computed_tokens = len(cached_blocks) * self.kv_cache_manager.block_size
                    request.num_blocks_cached = len(cached_blocks)  # Mark cached blocks as already published
            else:
                cached_blocks = []

            num_new = request.num_new_tokens_needed
            if self.config.enable_chunk_prefill and self.config.long_prefill_token_threshold > 0:
                num_new = min(num_new, self.config.long_prefill_token_threshold)
            num_new = min(num_new, token_budget)

            if num_new <= 0:
                # Full prefix-cache hit: leave 1 token for prefill so the
                # output uses the SAME kernel as the cold run (prefill, not
                # decode), producing identical first generated token.
                if request.num_computed_tokens >= request.num_prompt_tokens:
                    request.num_computed_tokens = max(0, request.num_prompt_tokens - 1)
                    num_new = 1
                else:
                    remaining_waiting.append(request)
                    continue

            if not self._try_allocate_request_blocks(request, num_new):
                if self._use_radix_cache:
                    self._release_radix_request(request)
                else:
                    self.kv_cache_manager.release_cached_blocks(cached_blocks)
                request.cached_block_ids = []
                request.num_computed_tokens = 0
                remaining_waiting.append(request)
                break

            request.status = RequestStatus.RUNNING
            self.running.append(request)
            all_block_ids = self._request_block_ids(request)
            output.scheduled_requests.append(
                ScheduledRequest(
                    request=request,
                    num_new_tokens=num_new,
                    is_prefill=True,
                    num_computed_tokens=request.num_computed_tokens,
                    block_ids=list(all_block_ids),
                    block_ids_by_group={
                        name: list(block_ids)
                        for name, block_ids in request.allocated_group_block_ids.items()
                    },
                    cache_partition=request.cache_partition,
                )
            )
            output.num_prefill_tokens += num_new
            if self._use_radix_cache:
                self.prefix_cache_stats.scheduled_prefill_tokens += num_new
            token_budget -= num_new

        remaining_waiting.extend(self.waiting)
        self.waiting = remaining_waiting

        return output

    def _grouped_cache_phase(self) -> str | None:
        """Choose one execution phase when grouped caches share decode scratch space."""
        if not self.kv_cache_manager.has_groups:
            return None
        if any(request.is_prefill and request.num_new_tokens_needed > 0 for request in self.running):
            return "prefill"
        if any(not request.is_prefill and request.num_new_tokens_needed > 0 for request in self.running):
            return "decode"
        return None

    def advance_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        """Optimistically advance state for a just-scheduled step (async mode).

        Called right after ``schedule()`` and before the worker result returns,
        so the next ``schedule()`` sees consistent state and does not re-schedule
        the same slot. For each scheduled request:

        - ``num_computed_tokens += num_new_tokens`` (the tokens this step covers),
          mirroring what ``update_from_output`` does synchronously.
        - decode requests reserve one ``num_output_placeholders`` for the token
          this step will sample but that is not yet known. Prefill chunks that do
          not complete the prompt sample nothing, so they add no placeholder;
          the chunk that completes the prompt reserves one (its first generated
          token), matching the sync path where that token is appended.

        The reconciliation in ``update_from_output`` removes the placeholder and
        applies the real token when the result arrives.
        """
        if not self.config.async_scheduling:
            return
        for scheduled in scheduler_output.scheduled_requests:
            request = scheduled.request
            completes_prompt = (
                request.num_computed_tokens + scheduled.num_new_tokens
                >= request.num_prompt_tokens
            )
            request.num_computed_tokens += scheduled.num_new_tokens
            # Do NOT publish prefix-cache blocks here: this runs at dispatch,
            # before the worker confirms the KV was computed. A failed/timed-out
            # step would leave block hashes published for uncomputed KV, which a
            # later same-prompt request could hit via get_computed_blocks().
            # Publication is deferred to _reconcile_async_output (confirmed result).
            # A step samples a token iff it is a decode step or the prefill chunk
            # that completes the prompt.
            if not scheduled.is_prefill or completes_prompt:
                # Reserve the MAXIMUM tokens this step can emit. Speculative /MTP
                # decode returns a variable count (1 .. 1+num_speculative_tokens)
                # that is only known once the worker replies, so we optimistically
                # reserve the upper bound — matching the block allocation that
                # schedule() already made for num_new + speculative_tokens — and
                # subtract the shortfall in _reconcile_async_output.
                reserved = 1 + self._speculative_tokens_for(request, scheduled)
                request.num_output_placeholders += reserved
                request.num_computed_tokens += reserved - 1

    def _speculative_tokens_for(
        self, request: "Request", scheduled: "ScheduledRequest"
    ) -> int:
        """Extra tokens beyond the first that a sampling step may emit.

        Mirrors the accounting ``schedule()`` uses when allocating blocks: only
        greedy decode steps get speculative capacity.
        """
        if scheduled.is_prefill or request.temperature > 0.0:
            return 0
        return self.config.num_speculative_tokens

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        new_token_ids: dict[str, int | list[int]],
    ) -> list[RequestOutput]:
        """Update request states after model execution. Returns outputs for finished/streaming."""
        outputs: list[RequestOutput] = []

        for scheduled in scheduler_output.scheduled_requests:
            request = scheduled.request
            # Async pipelining: a request that finished (EOS/length/stop), was
            # aborted, or was preempted at step N may still have step N+1 in
            # flight. Discard that stale result — the request has left `running`
            # (blocks freed) or had its computed-token/placeholder state reset,
            # so applying tokens/advancing state would corrupt bookkeeping.
            #
            # NOTE: the PREEMPTED check is correct while max_in_flight == 2 —
            # after preempting a request with an in-flight step, the batch queue
            # is full, so that step is drained (and discarded here) before the
            # next schedule() can re-admit the request to RUNNING. If pipeline
            # depth grows past 2, switch to a per-request scheduling epoch to
            # also catch the preempt -> re-RUNNING case.
            if request.status.is_finished or request.status is RequestStatus.PREEMPTED:
                continue
            token_value = new_token_ids.get(request.request_id)
            token_ids = (
                []
                if token_value is None
                else [int(token_value)]
                if isinstance(token_value, int)
                else [int(token_id) for token_id in token_value]
            )
            if self.config.async_scheduling:
                # num_computed_tokens and block caching were already advanced in
                # advance_after_schedule(); here we only apply the real sampled
                # token(s) and release the matching placeholder(s).
                self._reconcile_async_output(request, scheduled, token_ids, outputs)
            elif scheduled.is_prefill:
                request.num_computed_tokens += scheduled.num_new_tokens
                self._cache_request_prefix(request)
                if request.num_computed_tokens < request.num_prompt_tokens:
                    continue
                for token_id in token_ids:
                    request.output_token_ids.append(token_id)
                    outputs.append(RequestOutput(request_id=request.request_id, new_token_id=token_id))
                    if self._check_finish(request) is not None:
                        break
            else:
                retained_tokens = 0
                for token_id in token_ids:
                    request.output_token_ids.append(token_id)
                    retained_tokens += 1
                    outputs.append(RequestOutput(request_id=request.request_id, new_token_id=token_id))
                    if self._check_finish(request) is not None:
                        break
                request.num_computed_tokens += retained_tokens
                self._cache_request_prefix(request)

        finished_ids: list[str] = []
        for request in self.running:
            if request.status.is_finished:
                continue
            finish_reason = self._check_finish(request)
            if finish_reason is not None:
                request.status = finish_reason
                finished_ids.append(request.request_id)
                for out in reversed(outputs):
                    if out.request_id == request.request_id:
                        out.finished = True
                        out.finish_reason = finish_reason.name
                        break
                else:
                    outputs.append(RequestOutput(
                        request_id=request.request_id,
                        finished=True,
                        finish_reason=finish_reason.name,
                    ))

        for req_id in finished_ids:
            request = self.requests.get(req_id)
            if request is not None:
                self._free_request_blocks(request)
            self.running = [r for r in self.running if r.request_id != req_id]

        return outputs

    def _reconcile_async_output(
        self,
        request: "Request",
        scheduled: "ScheduledRequest",
        token_ids: list[int],
        outputs: list["RequestOutput"],
    ) -> None:
        """Apply the real sampled token(s) for an optimistically-scheduled step.

        ``advance_after_schedule`` already advanced ``num_computed_tokens`` and
        reserved ``num_output_placeholders`` for the token(s) this step would
        sample. Here — now that the worker has CONFIRMED the step — we:

        - append the retained token(s) and release this step's placeholders, and
        - publish exactly the KV range confirmed by this result (deferred from
          dispatch so failed/timed-out work never becomes prefix-visible).

        A prefill chunk that did not complete the prompt sampled nothing (no
        placeholder was reserved and ``token_ids`` is empty), so it emits nothing
        but still publishes its confirmed blocks.

        A single-token (Qwen) step reserves exactly one slot, so the release
        matches one-for-one. Speculative / MTP decode reserves the upper bound
        (``1 + num_speculative_tokens``) because the accepted count is unknown at
        dispatch; if fewer tokens come back — through rejection or an early EOS —
        the shortfall is subtracted from ``num_computed_tokens`` here so the
        request's accounting ends up identical to the synchronous path.
        """
        # This step reserved placeholders iff it sampled: a decode step, or a
        # prefill chunk that completed the prompt. num_computed_tokens was already
        # advanced, so "completed the prompt" == num_computed >= prompt.
        sampled_this_step = (
            not scheduled.is_prefill
            or request.num_computed_tokens >= request.num_prompt_tokens
        )
        reserved = (
            1 + self._speculative_tokens_for(request, scheduled)
            if sampled_this_step
            else 0
        )

        retained_tokens = 0
        for token_id in token_ids:
            request.output_token_ids.append(token_id)
            retained_tokens += 1
            outputs.append(RequestOutput(request_id=request.request_id, new_token_id=token_id))
            if self._check_finish(request) is not None:
                # Tokens after a finish are dropped (mirrors the sync path), so
                # they must not count as retained.
                break

        if reserved:
            # Release every placeholder this step reserved.
            request.num_output_placeholders = max(
                0, request.num_output_placeholders - reserved
            )
            # Reclaim only the SPECULATIVE positions that produced no retained
            # token. advance_after_schedule added `reserved - 1` extra positions
            # on top of scheduled.num_new_tokens; the latter is this step's real
            # KV work (a prefill chunk, or the decode's own token) and must never
            # be reverted — doing so would re-schedule the same prefill chunk and
            # decode it twice.
            speculative_positions = reserved - 1
            unused_speculative = max(0, speculative_positions - max(0, retained_tokens - 1))
            if unused_speculative > 0:
                request.num_computed_tokens = max(
                    0, request.num_computed_tokens - unused_speculative
                )

        # Publish only the range confirmed by this worker result. In depth-2
        # mode, request.num_computed_tokens may already include the next
        # in-flight step, which must not become visible to prefix matching yet.
        confirmed_kv_len = scheduled.num_computed_tokens + scheduled.num_new_tokens
        if not scheduled.is_prefill:
            confirmed_kv_len += max(0, retained_tokens - 1)
        self._cache_request_prefix(request, computed_kv_len=confirmed_kv_len)

    def _check_finish(self, request: Request) -> RequestStatus | None:
        if not request.output_token_ids:
            return None
        last_token = request.output_token_ids[-1]
        if request.eos_token_id is not None and last_token == request.eos_token_id:
            return RequestStatus.FINISHED_EOS
        if len(request.output_token_ids) >= request.max_new_tokens:
            return RequestStatus.FINISHED_LENGTH
        return None

    def _blocks_needed(self, request: Request, num_new_tokens: int) -> int:
        current_total_tokens = request.num_computed_tokens + num_new_tokens
        if self._use_radix_cache:
            return self.request_kv_pool.blocks_needed(request.request_id, current_total_tokens)
        current_blocks = len(request.cached_block_ids) + len(request.allocated_block_ids)
        block_size = self.kv_cache_manager.block_size
        needed_blocks = (current_total_tokens + block_size - 1) // block_size
        return max(0, needed_blocks - current_blocks)

    def _try_allocate_blocks(self, request: Request, num_blocks: int) -> bool:
        if num_blocks <= 0:
            return True
        radix_size_before = (
            self.kv_cache_manager.radix_cache.total_size() if self._use_radix_cache else 0
        )
        block_ids = self.kv_cache_manager.allocate_block_ids(num_blocks)
        if self._use_radix_cache:
            radix_size_after = self.kv_cache_manager.radix_cache.total_size()
            self.prefix_cache_stats.evicted_tokens += max(0, radix_size_before - radix_size_after)
        if block_ids is None:
            return False
        request.allocated_block_ids.extend(block_ids)
        if self._use_radix_cache:
            self.request_kv_pool.extend_pages(request.request_id, block_ids)
        return True

    def _try_allocate_request_blocks(self, request: Request, num_new_tokens: int) -> bool:
        """Grow either grouped or generic cache blocks for one scheduling step."""
        if self.kv_cache_manager.has_groups:
            total_tokens = request.num_computed_tokens + num_new_tokens
            try:
                request.allocated_group_block_ids = self.kv_cache_manager.ensure_group_blocks(
                    request.request_id,
                    total_tokens,
                    partition=request.cache_partition,
                )
                request.cache_partition = self.kv_cache_manager.group_request_partition(
                    request.request_id
                )
            except KVCacheCapacityError:
                return False
            return True
        return self._try_allocate_blocks(request, self._blocks_needed(request, num_new_tokens))

    def _preempt_lowest_priority(
        self,
        exclude: Request,
        scheduled_req_ids: set[str],
        num_scheduled_tokens: dict[str, int],
        output: SchedulerOutput,
    ) -> dict | None:
        """Preempt the lowest-priority running request to free blocks.

        If the victim was already scheduled in this iteration, it is removed
        from the scheduled output and its token budget is returned.
        """
        if not self.running:
            return None
        candidates = [r for r in self.running if r.request_id != exclude.request_id]
        if self.kv_cache_manager.has_groups and exclude.cache_partition is not None:
            same_partition = [
                request
                for request in candidates
                if request.cache_partition == exclude.cache_partition
            ]
            if same_partition:
                candidates = same_partition
        if not candidates:
            return None
        victim = max(candidates, key=lambda r: r.arrival_time)

        returned_tokens = 0
        if victim.request_id in scheduled_req_ids:
            scheduled_req_ids.discard(victim.request_id)
            returned_tokens = num_scheduled_tokens.pop(victim.request_id, 0)
            output.scheduled_requests = [
                s for s in output.scheduled_requests if s.request.request_id != victim.request_id
            ]
            if victim.is_prefill:
                output.num_prefill_tokens -= returned_tokens
                if self._use_radix_cache:
                    self.prefix_cache_stats.scheduled_prefill_tokens -= returned_tokens
            else:
                output.num_decode_tokens -= returned_tokens

        self._free_request_blocks(victim)
        victim.status = RequestStatus.PREEMPTED
        victim.num_computed_tokens = 0
        victim.cached_block_ids = []
        victim.allocated_block_ids = []
        victim.allocated_group_block_ids = {}
        victim.cache_partition = None
        victim.num_blocks_cached = 0
        # Async: drop any optimistic placeholder so the re-queued request restarts
        # from a clean prefill state (its in-flight step's result, if any, is
        # discarded engine-side since the request left `running`).
        victim.num_output_placeholders = 0
        self.running = [r for r in self.running if r.request_id != victim.request_id]
        self.waiting.appendleft(victim)
        return {"request": victim, "returned_tokens": returned_tokens}

    def _free_request_blocks(self, request: Request) -> None:
        if self._use_radix_cache:
            self._release_radix_request(request)
            return
        self.kv_cache_manager.release_blocks_by_ids(
            request.cached_block_ids,
            request.allocated_block_ids,
        )
        request.cached_block_ids = []
        request.allocated_block_ids = []
        if request.allocated_group_block_ids:
            self.kv_cache_manager.release_all_group_requests(request.request_id)
            request.allocated_group_block_ids = {}
        request.cache_partition = None

    @property
    def _use_radix_cache(self) -> bool:
        return self.config.enable_prefix_cache and self.config.prefix_cache_backend == "radix"

    def _request_block_ids(self, request: Request) -> list[int]:
        if self._use_radix_cache:
            return self.request_kv_pool.page_ids(request.request_id)
        return request.cached_block_ids + request.allocated_block_ids

    def _radix_extra_key(self, request: Request) -> tuple[str, ...]:
        return (request.model_id,)

    def _match_radix_request(self, request: Request) -> None:
        result = self.kv_cache_manager.match_radix_prefix(
            request.all_token_ids,
            extra_key=self._radix_extra_key(request),
        )
        matched_slots = list(result.device_indices)
        matched_pages = self.request_kv_pool.page_ids_from_slots(matched_slots)
        if matched_pages:
            self.kv_cache_manager.retain_pages_for_request(matched_pages)
        self.kv_cache_manager.radix_cache.inc_lock_ref(result.last_device_node)
        self.request_kv_pool.set_pages(request.request_id, matched_pages)
        request.prefix_indices = matched_slots
        request.cached_block_ids = matched_pages
        request.allocated_block_ids = []
        request.last_node = result.last_device_node
        request.cache_protected_len = len(matched_slots)
        request.kv_committed_len = len(matched_slots)
        request.num_computed_tokens = len(matched_slots)
        request.num_blocks_cached = len(matched_pages)
        self.prefix_cache_stats.lookups += 1
        self.prefix_cache_stats.matched_tokens += len(matched_slots)

    def _cache_unfinished_radix(
        self,
        request: Request,
        *,
        computed_kv_len: int | None = None,
    ) -> None:
        page_size = self.kv_cache_manager.block_size
        if computed_kv_len is None:
            computed_kv_len = request.num_computed_tokens
        computed_kv_len = min(computed_kv_len, len(request.all_token_ids))
        committed_len = computed_kv_len // page_size * page_size
        if committed_len <= request.kv_committed_len:
            request.prefix_indices = self.request_kv_pool.slot_indices(
                request.request_id,
                computed_kv_len,
            )
            return

        slots = self.request_kv_pool.slot_indices(request.request_id, committed_len)
        insert_result = self.kv_cache_manager.insert_radix_prefix(
            request.all_token_ids,
            slots,
            extra_key=self._radix_extra_key(request),
        )
        key = RadixKey.from_tokens(
            request.all_token_ids,
            extra_key=self._radix_extra_key(request),
            limit=committed_len,
        )
        match_result = self.kv_cache_manager.radix_cache.match_prefix(key)
        canonical_slots = list(match_result.device_indices)
        if len(canonical_slots) != committed_len:
            raise RuntimeError("Radix cache failed to return the prefix that was just inserted")
        canonical_pages = self.request_kv_pool.page_ids_from_slots(canonical_slots)

        old_pages = self.request_kv_pool.page_ids(request.request_id)
        private_tail = old_pages[committed_len // page_size :]
        new_pages = canonical_pages + private_tail
        old_page_set = set(old_pages)
        new_page_set = set(new_pages)
        added_pages = [page_id for page_id in new_pages if page_id not in old_page_set]
        removed_pages = [page_id for page_id in old_pages if page_id not in new_page_set]
        if added_pages:
            self.kv_cache_manager.retain_pages_for_request(added_pages)
        if removed_pages:
            self.kv_cache_manager.release_pages_from_request(removed_pages)
        self.request_kv_pool.set_pages(request.request_id, new_pages)

        new_node = match_result.last_device_node
        self.kv_cache_manager.radix_cache.inc_lock_ref(new_node)
        if request.last_node is not None:
            self.kv_cache_manager.radix_cache.dec_lock_ref(request.last_node)

        mapped_slots = self.request_kv_pool.slot_indices(request.request_id, computed_kv_len)
        request.prefix_indices = canonical_slots + mapped_slots[committed_len:]
        request.cached_block_ids = canonical_pages
        request.allocated_block_ids = private_tail
        request.last_node = new_node
        request.cache_protected_len = committed_len
        request.kv_committed_len = committed_len
        request.num_blocks_cached = len(canonical_pages)
        self.prefix_cache_stats.inserts += 1
        self.prefix_cache_stats.inserted_tokens += insert_result.inserted_len
        self.prefix_cache_stats.duplicate_pages_freed += len(removed_pages)

    def _release_radix_request(self, request: Request) -> None:
        pages = self.request_kv_pool.free(request.request_id)
        if pages:
            self.kv_cache_manager.release_pages_from_request(pages)
        if request.last_node is not None:
            self.kv_cache_manager.radix_cache.dec_lock_ref(request.last_node)
        request.cached_block_ids = []
        request.allocated_block_ids = []
        request.prefix_indices = []
        request.last_node = None
        request.cache_protected_len = 0
        request.kv_committed_len = 0
        request.num_blocks_cached = 0

    def _should_defer_for_radix_producer(self, request: Request) -> bool:
        if not self.config.enable_radix_in_batch_dedup or request.num_computed_tokens > 0:
            return False
        page_size = self.kv_cache_manager.block_size
        if len(request.all_token_ids) < page_size:
            return False
        first_page = tuple(request.all_token_ids[:page_size])
        namespace = self._radix_extra_key(request)
        for producer in self.running:
            if producer.request_id == request.request_id:
                continue
            # num_computed_tokens can advance optimistically before the worker
            # result arrives. Only a committed page means the producer's first
            # page is actually available through the Radix tree.
            if producer.kv_committed_len >= page_size:
                continue
            if self._radix_extra_key(producer) != namespace:
                continue
            if tuple(producer.all_token_ids[:page_size]) == first_page:
                return True
        return False

    def _cache_completed_blocks(
        self,
        request: Request,
        *,
        computed_kv_len: int | None = None,
    ) -> None:
        """Register completed blocks in the prefix cache."""
        if not self.config.enable_prefix_cache:
            return
        if computed_kv_len is None:
            computed_kv_len = request.num_computed_tokens
        total_blocks_computed = min(
            computed_kv_len // self.kv_cache_manager.block_size,
            len(request.block_hashes)
        )
        already_cached = request.num_blocks_cached
        if total_blocks_computed <= already_cached:
            return  # Nothing new to cache
        all_block_ids = request.cached_block_ids + request.allocated_block_ids
        self.kv_cache_manager.cache_block_ids(
            all_block_ids,
            request.block_hashes,
            already_cached,
            total_blocks_computed,
        )
        request.num_blocks_cached = total_blocks_computed

    def _cache_request_prefix(
        self,
        request: Request,
        *,
        computed_kv_len: int | None = None,
    ) -> None:
        if self._use_radix_cache:
            self._cache_unfinished_radix(request, computed_kv_len=computed_kv_len)
        else:
            self._cache_completed_blocks(request, computed_kv_len=computed_kv_len)
