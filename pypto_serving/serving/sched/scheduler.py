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

from pypto_serving.serving.memory.kv_cache import KVCacheCapacityError, KvCacheManager

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
    max_prefill_tokens_per_request: int | None = None
    max_seq_len: int = 4096
    # Feature flags
    enable_prefix_cache: bool = True
    enable_chunk_prefill: bool = True
    num_speculative_tokens: int = 0
    supports_chunked_prefill_with_speculation: bool = True
    # Async (pipelined) scheduling: schedule step N+1 before step N's sampled
    # token returns, advancing request state optimistically via placeholders.
    async_scheduling: bool = False

    def __post_init__(self) -> None:
        if self.num_speculative_tokens < 0:
            raise ValueError("num_speculative_tokens must be non-negative")
        if self.num_speculative_tokens + 1 > self.max_num_scheduled_tokens:
            raise ValueError(
                "max_num_scheduled_tokens must fit one decode token plus "
                "num_speculative_tokens"
            )
        if (
            self.max_prefill_tokens_per_request is not None
            and self.max_prefill_tokens_per_request <= 0
        ):
            raise ValueError("max_prefill_tokens_per_request must be positive when specified")


@dataclass
class Request:
    request_id: str
    prompt_token_ids: list[int]
    max_new_tokens: int
    arrival_time: float = field(default_factory=time.time)
    status: RequestStatus = RequestStatus.WAITING
    num_computed_tokens: int = 0
    output_token_ids: list[int] = field(default_factory=list)
    stop_strings: tuple[str, ...] = ()
    eos_token_id: int | None = None
    temperature: float = 0.8
    top_p: float = 1.0
    top_k: int | None = None
    seed: int | None = None
    cached_block_ids: list[int] = field(default_factory=list)
    allocated_block_ids: list[int] = field(default_factory=list)
    allocated_group_block_ids: dict[str, list[int]] = field(default_factory=dict)
    cache_partition: int | None = None
    block_hashes: list[int] = field(default_factory=list)
    group_block_hashes: dict[str, list[int]] = field(default_factory=dict)
    num_blocks_cached: int = 0  # Track how many blocks have been published to prefix cache
    num_group_blocks_cached: dict[str, int] = field(default_factory=dict)
    # Async scheduling: tokens scheduled optimistically but not yet sampled.
    # Stands in for output tokens still in flight so the next schedule() advances
    # correctly; decremented as real tokens are applied in update_from_output.
    num_output_placeholders: int = 0
    # Async scheduling must not turn a terminal prefill into a decode wave until
    # the worker has initialized request-local decode state and confirmed the
    # prefill result.  This is a per-request barrier, so unrelated ready requests
    # can continue to use the depth-2 pipeline.
    terminal_prefill_in_flight: bool = False

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
    group_blocks_retained: bool = False


@dataclass
class SchedulerOutput:
    scheduled_requests: list[ScheduledRequest] = field(default_factory=list)
    preempted_requests: list[Request] = field(default_factory=list)
    rejected_requests: dict[str, str] = field(default_factory=dict)
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
        self.config = config
        self.kv_cache_manager = kv_cache_manager
        if (
            self.kv_cache_manager.has_groups
            and self.config.enable_prefix_cache
            and self.config.num_speculative_tokens > 0
            and not self.kv_cache_manager.has_eagle_groups
        ):
            raise ValueError(
                "Grouped speculative/MTP prefix caching requires an EAGLE cache group"
            )
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
        if not self.config.enable_prefix_cache:
            rejection = self._single_prefill_rejection(request)
            if rejection is not None:
                raise ValueError(rejection)
        if self.config.enable_prefix_cache:
            if self.kv_cache_manager.has_groups:
                request.group_block_hashes = self.kv_cache_manager.compute_group_block_hashes(
                    request.prompt_token_ids
                )
            else:
                request.block_hashes = self.kv_cache_manager.compute_block_hashes(
                    request.prompt_token_ids
                )
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
            if request.terminal_prefill_in_flight:
                running_to_keep.append(request)
                continue
            if grouped_phase is not None and request.is_prefill != (grouped_phase == "prefill"):
                running_to_keep.append(request)
                continue
            num_new = request.num_new_tokens_needed
            if num_new <= 0:
                running_to_keep.append(request)
                continue

            num_new = self._limit_scheduled_tokens(request, token_budget)

            if num_new <= 0:
                running_to_keep.append(request)
                continue

            is_prefill = request.is_prefill
            speculative_tokens = self._speculative_tokens_for_step(
                request,
                is_prefill=is_prefill,
                num_computed_tokens=request.num_computed_tokens,
                num_new_tokens=num_new,
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

            all_block_ids = request.cached_block_ids + request.allocated_block_ids
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
            if self.config.enable_prefix_cache:
                if self.kv_cache_manager.has_groups:
                    (
                        request.allocated_group_block_ids,
                        request.num_computed_tokens,
                        request.cache_partition,
                    ) = self.kv_cache_manager.acquire_group_prefix_blocks(
                        request.request_id,
                        request.group_block_hashes,
                        max_cache_hit_tokens=max(0, request.num_prompt_tokens - 1),
                    )
                    request.num_group_blocks_cached = (
                        self.kv_cache_manager.published_group_block_counts(
                            request.num_computed_tokens
                        )
                        if request.num_computed_tokens
                        else {}
                    )
                    if request.num_computed_tokens:
                        logger.info(
                            "prefix_cache_hit request_id=%s prefix_cache_hit_tokens=%d",
                            request.request_id,
                            request.num_computed_tokens,
                        )
                    cached_blocks = []
                else:
                    cached_blocks = self.kv_cache_manager.get_computed_blocks(
                        request.prompt_token_ids
                    )
                    if cached_blocks:
                        request.cached_block_ids = [b.block_id for b in cached_blocks]
                        request.num_computed_tokens = (
                            len(cached_blocks) * self.kv_cache_manager.block_size
                        )
                        # Mark cached blocks as already published.
                        request.num_blocks_cached = len(cached_blocks)
            else:
                cached_blocks = []

            grouped_prefix_hit = (
                self.kv_cache_manager.has_groups and request.num_computed_tokens > 0
            )

            rejection = self._single_prefill_rejection(request)
            if rejection is not None:
                self._release_waiting_prefix_blocks(request)
                request.status = RequestStatus.FINISHED_ABORTED
                self.requests.pop(request.request_id, None)
                output.rejected_requests[request.request_id] = rejection
                continue

            num_new = self._limit_scheduled_tokens(request, token_budget)

            if num_new <= 0:
                # Full prefix-cache hit: leave 1 token for prefill so the
                # output uses the SAME kernel as the cold run (prefill, not
                # decode), producing identical first generated token.
                if request.num_computed_tokens >= request.num_prompt_tokens:
                    request.num_computed_tokens = max(0, request.num_prompt_tokens - 1)
                    num_new = 1
                else:
                    self._release_waiting_prefix_blocks(request)
                    remaining_waiting.append(request)
                    continue

            allocated = self._try_allocate_request_blocks(request, num_new)
            if not allocated and grouped_prefix_hit:
                # A grouped hit is rank-local. If its partition cannot hold the
                # uncached suffix, release the hit and retry cold so an idle
                # partition can admit the request instead of repeatedly
                # selecting the same capacity-constrained cached partition.
                hit_partition = request.cache_partition
                self._release_waiting_prefix_blocks(request)
                cold_num_new = self._limit_scheduled_tokens(request, token_budget)
                if cold_num_new > 0:
                    allocated = self._try_allocate_request_blocks(
                        request,
                        cold_num_new,
                    )
                    if allocated:
                        logger.info(
                            "prefix_cache_fallback request_id=%s "
                            "cached_partition=%s cold_partition=%s",
                            request.request_id,
                            hit_partition,
                            request.cache_partition,
                        )
                        num_new = cold_num_new

            if not allocated:
                self._release_waiting_prefix_blocks(request)
                remaining_waiting.append(request)
                break

            request.status = RequestStatus.RUNNING
            self.running.append(request)
            all_block_ids = request.cached_block_ids + request.allocated_block_ids
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
            token_budget -= num_new

        remaining_waiting.extend(self.waiting)
        self.waiting = remaining_waiting

        return output

    def _release_waiting_prefix_blocks(self, request: Request) -> None:
        """Roll back prefix-cache references when a waiting request is deferred."""
        self._free_request_blocks(request)
        request.num_computed_tokens = 0
        request.num_blocks_cached = 0
        request.num_group_blocks_cached = {}

    def _prefill_chunk_limit(self) -> int | None:
        """Return the configured per-request prefill limit before step budget."""
        limits: list[int] = []
        if self.config.max_prefill_tokens_per_request is not None:
            limits.append(self.config.max_prefill_tokens_per_request)
        if self.config.enable_chunk_prefill and self.config.long_prefill_token_threshold > 0:
            limits.append(self.config.long_prefill_token_threshold)
        return min(limits) if limits else None

    def _single_prefill_dispatch_limit(self) -> int:
        """Return the largest prompt that can be guaranteed to run in one step."""
        limit = self._prefill_chunk_limit()
        if limit is None:
            return self.config.max_num_scheduled_tokens
        return min(limit, self.config.max_num_scheduled_tokens)

    def _requires_single_prefill_dispatch(self) -> bool:
        """Return whether unfinished prefill must fit in one scheduler step."""
        return (
            not self.config.enable_chunk_prefill
            or (
                self.config.num_speculative_tokens > 0
                and not self.config.supports_chunked_prefill_with_speculation
            )
        )

    def _single_prefill_rejection(self, request: Request) -> str | None:
        """Return an admission error when the uncached prompt cannot run in one step."""
        if not self._requires_single_prefill_dispatch():
            return None
        uncached_tokens = max(1, request.num_prompt_tokens - request.num_computed_tokens)
        single_dispatch_limit = self._single_prefill_dispatch_limit()
        if uncached_tokens <= single_dispatch_limit:
            return None
        if not self.config.enable_chunk_prefill:
            return (
                f"Request {request.request_id} uncached prompt length {uncached_tokens} exceeds "
                f"the effective single-dispatch prefill limit {single_dispatch_limit} while "
                "chunked prefill is disabled."
            )
        return (
            f"Request {request.request_id} uncached prompt length {uncached_tokens} requires "
            "chunked prefill, which is not supported with speculative decoding for this model; "
            f"the single-dispatch limit is {single_dispatch_limit}. Disable speculative "
            "decoding or shorten the uncached prompt suffix."
        )

    def _limit_scheduled_tokens(
        self,
        request: Request,
        token_budget: int,
    ) -> int:
        """Apply per-request prefill limits and the remaining step budget."""
        needed = request.num_new_tokens_needed
        limit = token_budget
        if request.is_prefill:
            chunk_limit = self._prefill_chunk_limit()
            if chunk_limit is not None:
                limit = min(limit, chunk_limit)
            if self._requires_single_prefill_dispatch() and needed > limit:
                return 0
        return min(needed, limit)

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
            if scheduled.block_ids_by_group:
                if scheduled.cache_partition is None:
                    raise RuntimeError("Grouped async step has no cache partition")
                self.kv_cache_manager.retain_group_block_snapshot(
                    scheduled.block_ids_by_group,
                    scheduled.cache_partition,
                )
                scheduled.group_blocks_retained = True
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
            if (
                scheduled.is_prefill
                and completes_prompt
                and self.config.num_speculative_tokens > 0
            ):
                request.terminal_prefill_in_flight = True

    def _speculative_tokens_for(
        self, request: "Request", scheduled: "ScheduledRequest"
    ) -> int:
        """Extra tokens beyond the first that a sampling step may emit.

        Mirrors the accounting ``schedule()`` uses when allocating blocks: only
        greedy decode steps get speculative capacity.
        """
        return self._speculative_tokens_for_step(
            request,
            is_prefill=scheduled.is_prefill,
            num_computed_tokens=scheduled.num_computed_tokens,
            num_new_tokens=scheduled.num_new_tokens,
        )

    def _speculative_tokens_for_step(
        self,
        request: "Request",
        *,
        is_prefill: bool,
        num_computed_tokens: int,
        num_new_tokens: int,
    ) -> int:
        """Return draft positions that fit this request's remaining context."""
        if is_prefill or request.temperature > 0.0:
            return 0
        step_context_len = num_computed_tokens + num_new_tokens
        remaining_context = max(0, self.config.max_seq_len - step_context_len)
        return min(self.config.num_speculative_tokens, remaining_context)

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
                self._release_scheduled_group_blocks(scheduled)
                continue
            if (
                scheduled.is_prefill
                and scheduled.num_computed_tokens + scheduled.num_new_tokens
                >= request.num_prompt_tokens
            ):
                request.terminal_prefill_in_flight = False
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
                self._cache_completed_blocks(request)
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
                self._cache_completed_blocks(request)

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

        - release the placeholder(s) reserved for this step,
        - publish the now-computed prefix-cache blocks (deferred from dispatch so
          a failed/timed-out step never leaves hashes for uncomputed KV), and
        - append the real token(s) that came back, emitting RequestOutputs.

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

        # Publish confirmed blocks through the exact physical table used by this
        # step. A newer in-flight chunk may already have advanced the request's
        # rolling table, so consulting current manager state here is incorrect.
        confirmed_tokens = scheduled.num_computed_tokens + scheduled.num_new_tokens
        try:
            if self.kv_cache_manager.has_groups:
                if scheduled.cache_partition is None:
                    raise RuntimeError("Grouped async step has no cache partition")
                request.num_group_blocks_cached = (
                    self.kv_cache_manager.cache_group_blocks_from_snapshot(
                        request.group_block_hashes,
                        confirmed_tokens,
                        request.num_group_blocks_cached,
                        scheduled.block_ids_by_group,
                        scheduled.cache_partition,
                    )
                )
            else:
                self._cache_completed_blocks(request, confirmed_tokens)
        finally:
            self._release_scheduled_group_blocks(scheduled)

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

    def _release_scheduled_group_blocks(self, scheduled: ScheduledRequest) -> None:
        if not scheduled.group_blocks_retained:
            return
        if scheduled.cache_partition is None:
            raise RuntimeError("Retained grouped async step has no cache partition")
        self.kv_cache_manager.release_group_block_snapshot(
            scheduled.block_ids_by_group,
            scheduled.cache_partition,
        )
        scheduled.group_blocks_retained = False

    def discard_scheduled_request(self, scheduled: ScheduledRequest) -> None:
        """Drop one failed/stale async step without publishing its KV hashes."""
        self._release_scheduled_group_blocks(scheduled)

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
        current_blocks = len(request.cached_block_ids) + len(request.allocated_block_ids)
        block_size = self.kv_cache_manager.block_size
        needed_blocks = (current_total_tokens + block_size - 1) // block_size
        return max(0, needed_blocks - current_blocks)

    def _try_allocate_blocks(self, request: Request, num_blocks: int) -> bool:
        if num_blocks <= 0:
            return True
        if self.kv_cache_manager.num_free_blocks < num_blocks:
            return False
        block_ids = self.kv_cache_manager.allocate_block_ids(num_blocks)
        if block_ids is None:
            return False
        request.allocated_block_ids.extend(block_ids)
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
        victim.num_group_blocks_cached = {}
        # Async: drop any optimistic placeholder so the re-queued request restarts
        # from a clean prefill state (its in-flight step's result, if any, is
        # discarded engine-side since the request left `running`).
        victim.num_output_placeholders = 0
        victim.terminal_prefill_in_flight = False
        self.running = [r for r in self.running if r.request_id != victim.request_id]
        self.waiting.appendleft(victim)
        return {"request": victim, "returned_tokens": returned_tokens}

    def _free_request_blocks(self, request: Request) -> None:
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
        request.num_group_blocks_cached = {}

    def _cache_completed_blocks(
        self,
        request: Request,
        num_computed_tokens: int | None = None,
    ) -> None:
        """Register completed blocks in the prefix cache."""
        if not self.config.enable_prefix_cache:
            return
        confirmed_tokens = (
            request.num_computed_tokens
            if num_computed_tokens is None
            else num_computed_tokens
        )
        if self.kv_cache_manager.has_groups:
            request.num_group_blocks_cached = self.kv_cache_manager.cache_group_blocks(
                request.request_id,
                request.group_block_hashes,
                confirmed_tokens,
                request.num_group_blocks_cached,
            )
            return
        total_blocks_computed = min(
            confirmed_tokens // self.kv_cache_manager.block_size,
            len(request.block_hashes),
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
