# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto

from .kv_cache import KvCacheManager


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
    block_hashes: list[int] = field(default_factory=list)

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_tokens(self) -> int:
        return self.num_prompt_tokens + len(self.output_token_ids)

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
        self.config = config
        if self.config.prefix_cache_backend not in {"hash", "radix"}:
            raise ValueError(
                "prefix_cache_backend must be either 'hash' or 'radix', "
                f"got {self.config.prefix_cache_backend!r}"
            )
        self.kv_cache_manager = kv_cache_manager
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.requests: dict[str, Request] = {}
        self._radix_nodes: dict[str, object] = {}
        self._radix_debug = os.environ.get("PYPTO_RADIX_DEBUG", "").lower() not in {"", "0", "false", "no"}

    def add_request(self, request: Request) -> None:
        if len(request.prompt_token_ids) > self.config.max_seq_len:
            request.prompt_token_ids = request.prompt_token_ids[: self.config.max_seq_len]
        if self.config.enable_prefix_cache and self.config.prefix_cache_backend == "hash":
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

        # Phase 1: schedule RUNNING requests (decode or resumed prefill)
        scheduled_req_ids: set[str] = set()
        num_scheduled_tokens: dict[str, int] = {}
        running_to_keep: list[Request] = []
        for request in self.running:
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

            num_blocks_needed = self._blocks_needed(request, num_new)
            if not self._try_allocate_blocks(request, num_blocks_needed):
                preempted = self._preempt_lowest_priority(
                    request, scheduled_req_ids, num_scheduled_tokens, output
                )
                if preempted is None:
                    running_to_keep.append(request)
                    continue
                token_budget += preempted.get("returned_tokens", 0)
                output.preempted_requests.append(preempted["request"])
                if not self._try_allocate_blocks(request, num_blocks_needed):
                    running_to_keep.append(request)
                    continue

            is_prefill = request.is_prefill
            all_block_ids = request.cached_block_ids + request.allocated_block_ids
            output.scheduled_requests.append(
                ScheduledRequest(
                    request=request,
                    num_new_tokens=num_new,
                    is_prefill=is_prefill,
                    num_computed_tokens=request.num_computed_tokens,
                    block_ids=list(all_block_ids),
                )
            )
            scheduled_req_ids.add(request.request_id)
            num_scheduled_tokens[request.request_id] = num_new
            if is_prefill:
                output.num_prefill_tokens += num_new
            else:
                output.num_decode_tokens += num_new
            token_budget -= num_new
            running_to_keep.append(request)

        self.running = running_to_keep

        # Phase 2: schedule WAITING requests (new prefill)
        remaining_waiting: deque[Request] = deque()
        while self.waiting and token_budget > 0:
            if len(self.running) >= self.config.max_num_running_reqs:
                break

            request = self.waiting.popleft()

            # Prefix cache lookup
            if self.config.enable_prefix_cache and self.config.prefix_cache_backend == "radix":
                self._match_radix_prefix(request)
                cached_blocks = []
            elif self.config.enable_prefix_cache:
                cached_blocks = self.kv_cache_manager.get_computed_blocks(request.prompt_token_ids)
                if cached_blocks:
                    request.cached_block_ids = [b.block_id for b in cached_blocks]
                    request.num_computed_tokens = len(cached_blocks) * self.kv_cache_manager.block_size
            else:
                cached_blocks = []

            num_new = request.num_new_tokens_needed
            if self.config.enable_chunk_prefill and self.config.long_prefill_token_threshold > 0:
                num_new = min(num_new, self.config.long_prefill_token_threshold)
            num_new = min(num_new, token_budget)

            if num_new <= 0:
                if self.config.prefix_cache_backend == "radix":
                    self.kv_cache_manager.release_pages_from_request(request.cached_block_ids)
                    self._release_radix_prefix(request)
                remaining_waiting.append(request)
                continue

            num_blocks_needed = self._blocks_needed(request, num_new)
            if not self._try_allocate_blocks(request, num_blocks_needed):
                self.kv_cache_manager.release_cached_blocks(cached_blocks)
                if self.config.prefix_cache_backend == "radix":
                    self.kv_cache_manager.release_pages_from_request(request.cached_block_ids)
                self._release_radix_prefix(request)
                request.cached_block_ids = []
                request.num_computed_tokens = 0
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
                )
            )
            self._debug_radix(
                "schedule",
                request,
                scheduled_prefill_tokens=num_new,
                computed_tokens=request.num_computed_tokens,
                cached_pages=len(request.cached_block_ids),
                allocated_pages=len(request.allocated_block_ids),
            )
            output.num_prefill_tokens += num_new
            token_budget -= num_new

        remaining_waiting.extend(self.waiting)
        self.waiting = remaining_waiting

        return output

    def update_from_output(
        self, scheduler_output: SchedulerOutput, new_token_ids: dict[str, int]
    ) -> list[RequestOutput]:
        """Update request states after model execution. Returns outputs for finished/streaming."""
        outputs: list[RequestOutput] = []

        for scheduled in scheduler_output.scheduled_requests:
            request = scheduled.request
            request.num_computed_tokens += scheduled.num_new_tokens

            self._cache_completed_blocks(request)

            if request.is_prefill:
                if request.num_computed_tokens < request.num_prompt_tokens:
                    continue
                token_id = new_token_ids.get(request.request_id)
                if token_id is not None:
                    request.output_token_ids.append(token_id)
                    outputs.append(RequestOutput(request_id=request.request_id, new_token_id=token_id))
            else:
                token_id = new_token_ids.get(request.request_id)
                if token_id is not None:
                    request.output_token_ids.append(token_id)
                    outputs.append(RequestOutput(request_id=request.request_id, new_token_id=token_id))

        finished_ids: list[str] = []
        for request in self.running:
            if request.status.is_finished:
                continue
            finish_reason = self._check_finish(request)
            if finish_reason is not None:
                request.status = finish_reason
                finished_ids.append(request.request_id)
                for out in outputs:
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
        block_ids = self.kv_cache_manager.allocate_block_ids(num_blocks)
        if block_ids is None:
            return False
        request.allocated_block_ids.extend(block_ids)
        return True

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
        self.running = [r for r in self.running if r.request_id != victim.request_id]
        self.waiting.appendleft(victim)
        return {"request": victim, "returned_tokens": returned_tokens}

    def _free_request_blocks(self, request: Request) -> None:
        self.kv_cache_manager.release_blocks_by_ids(
            request.cached_block_ids,
            request.allocated_block_ids,
        )
        self._release_radix_prefix(request)
        request.cached_block_ids = []
        request.allocated_block_ids = []

    def _cache_completed_blocks(self, request: Request) -> None:
        """Register completed blocks in the prefix cache."""
        if not self.config.enable_prefix_cache:
            return
        if self.config.prefix_cache_backend == "radix":
            all_block_ids = request.cached_block_ids + request.allocated_block_ids
            completed_len = min(request.num_computed_tokens, len(request.all_token_ids))
            completed_pages = min(
                completed_len // self.kv_cache_manager.block_size,
                len(all_block_ids),
            )
            completed_len = completed_pages * self.kv_cache_manager.block_size
            self.kv_cache_manager.insert_radix_prefix(
                request.model_id,
                request.all_token_ids[:completed_len],
                all_block_ids[:completed_pages],
            )
            self._debug_radix(
                "insert",
                request,
                total_tokens=completed_len,
                pages=completed_pages,
            )
            return
        total_blocks_computed = request.num_computed_tokens // self.kv_cache_manager.block_size
        already_cached = len(request.cached_block_ids)
        all_block_ids = request.cached_block_ids + request.allocated_block_ids
        self.kv_cache_manager.cache_block_ids(
            all_block_ids,
            request.block_hashes,
            already_cached,
            total_blocks_computed,
        )

    def _match_radix_prefix(self, request: Request) -> None:
        """Attach a radix prefix hit to a waiting request."""
        max_match_len = max(0, request.num_prompt_tokens - 1)
        match_tokens = request.prompt_token_ids[:max_match_len]
        match = self.kv_cache_manager.match_radix_prefix(request.model_id, match_tokens)
        self._debug_radix(
            "match",
            request,
            prompt_tokens=request.num_prompt_tokens,
            max_match_len=max_match_len,
            matched_tokens=match.prefix_len,
            matched_pages=len(match.page_ids),
        )
        if match.prefix_len <= 0:
            return
        request.cached_block_ids = list(match.page_ids)
        request.num_computed_tokens = match.prefix_len
        self.kv_cache_manager.retain_pages_for_request(request.cached_block_ids)
        self._radix_nodes[request.request_id] = match.last_node

    def _debug_radix(self, event: str, request: Request, **fields: object) -> None:
        if not self._radix_debug or self.config.prefix_cache_backend != "radix":
            return
        details = " ".join(f"{key}={value}" for key, value in fields.items())
        print(
            f"[radix] {event} request={request.request_id} model={request.model_id} {details}",
            flush=True,
        )

    def _release_radix_prefix(self, request: Request) -> None:
        node = self._radix_nodes.pop(request.request_id, None)
        if node is not None:
            self.kv_cache_manager.radix_cache.dec_lock_ref(node)
