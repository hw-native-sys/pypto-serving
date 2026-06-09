# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto

from .kv_cache import KvCacheManager
from .kv_offload import KVBlockLocation, TransferJob


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
    max_cpu_offload_blocks: int = 0


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
    transfer_jobs: list[TransferJob] = field(default_factory=list)
    jobs_to_flush: set[int] = field(default_factory=set)
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0

    @property
    def is_empty(self) -> bool:
        return (
            len(self.scheduled_requests) == 0
            and len(self.transfer_jobs) == 0
            and len(self.jobs_to_flush) == 0
        )


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
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.requests: dict[str, Request] = {}
        self.num_cpu_store_jobs: int = 0
        self.num_cpu_load_jobs: int = 0
        self.num_cpu_store_blocks: int = 0
        self.num_cpu_load_blocks: int = 0

    def add_request(self, request: Request) -> None:
        if len(request.prompt_token_ids) > self.config.max_seq_len:
            request.prompt_token_ids = request.prompt_token_ids[: self.config.max_seq_len]
        if self.config.enable_prefix_cache:
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
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]
            num_new = request.num_new_tokens_needed
            if num_new <= 0:
                req_index += 1
                continue

            if self.config.enable_chunk_prefill and self.config.long_prefill_token_threshold > 0:
                num_new = min(num_new, self.config.long_prefill_token_threshold)
            num_new = min(num_new, token_budget)

            if num_new <= 0:
                req_index += 1
                continue

            num_blocks_needed = self._blocks_needed(request, num_new)
            if not self._try_allocate_blocks(request, num_blocks_needed):
                preempted_request = self._pop_lowest_priority_running_request(request)
                if preempted_request is None:
                    req_index += 1
                    continue
                preempted = self._preempt_lowest_priority(
                    preempted_request,
                    scheduled_req_ids,
                    num_scheduled_tokens,
                    output,
                )
                if preempted is None:
                    self.running.insert(req_index, preempted_request)
                    req_index += 1
                    continue
                token_budget += preempted.get("returned_tokens", 0)
                output.preempted_requests.append(preempted["request"])
                break

            is_prefill = request.is_prefill
            all_block_ids = request.cached_block_ids + request.allocated_block_ids
            physical_block_ids = self.kv_cache_manager.resident_block_ids(all_block_ids)
            output.scheduled_requests.append(
                ScheduledRequest(
                    request=request,
                    num_new_tokens=num_new,
                    is_prefill=is_prefill,
                    num_computed_tokens=request.num_computed_tokens,
                    block_ids=physical_block_ids,
                )
            )
            scheduled_req_ids.add(request.request_id)
            num_scheduled_tokens[request.request_id] = num_new
            if is_prefill:
                output.num_prefill_tokens += num_new
            else:
                output.num_decode_tokens += num_new
            token_budget -= num_new
            req_index += 1

        remaining_waiting: deque[Request] = deque()
        # Phase 2: schedule WAITING requests (new prefill or resumed preemption).
        if not output.preempted_requests:
            while self.waiting and token_budget > 0:
                if len(self.running) >= self.config.max_num_running_reqs:
                    break

                request = self.waiting.popleft()

                # Prefix cache lookup
                existing_block_ids = request.cached_block_ids + request.allocated_block_ids
                if self.config.enable_prefix_cache and not existing_block_ids:
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
                    remaining_waiting.append(request)
                    continue

                all_block_ids = request.cached_block_ids + request.allocated_block_ids
                missing_block_ids = self.kv_cache_manager.non_resident_block_ids(all_block_ids)
                num_blocks_needed = self._blocks_needed(request, num_new)
                if self.kv_cache_manager.num_free_blocks < len(missing_block_ids) + num_blocks_needed:
                    self.kv_cache_manager.release_cached_blocks(cached_blocks)
                    request.cached_block_ids = []
                    request.num_computed_tokens = 0
                    remaining_waiting.append(request)
                    break
                if not self._ensure_blocks_resident(request, all_block_ids, output):
                    remaining_waiting.append(request)
                    continue

                if not self._try_allocate_blocks(request, num_blocks_needed):
                    self.kv_cache_manager.release_cached_blocks(cached_blocks)
                    request.cached_block_ids = []
                    request.num_computed_tokens = 0
                    remaining_waiting.append(request)
                    break

                request.status = RequestStatus.RUNNING
                self.running.append(request)
                all_block_ids = request.cached_block_ids + request.allocated_block_ids
                physical_block_ids = self.kv_cache_manager.resident_block_ids(all_block_ids)
                output.scheduled_requests.append(
                    ScheduledRequest(
                        request=request,
                        num_new_tokens=num_new,
                        is_prefill=True,
                        num_computed_tokens=request.num_computed_tokens,
                        block_ids=physical_block_ids,
                    )
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
        if self.kv_cache_manager.num_free_blocks < num_blocks:
            return False
        block_ids = self.kv_cache_manager.allocate_block_ids(num_blocks)
        if block_ids is None:
            return False
        request.allocated_block_ids.extend(block_ids)
        return True

    def _ensure_blocks_resident(
        self,
        request: Request,
        block_ids: list[int],
        output: SchedulerOutput,
    ) -> bool:
        if any(self.kv_cache_manager.blocks[block_id].pending_job_id is not None for block_id in block_ids):
            return False
        missing_block_ids = self.kv_cache_manager.non_resident_block_ids(block_ids)
        if not missing_block_ids:
            return True
        if self.config.max_cpu_offload_blocks <= 0:
            return False
        try:
            job = self.kv_cache_manager.build_cpu_load_job(
                missing_block_ids,
                request_id=request.request_id,
            )
        except RuntimeError:
            return False
        output.transfer_jobs.append(job)
        output.jobs_to_flush.add(job.job_id)
        self.num_cpu_load_jobs += 1
        self.num_cpu_load_blocks += len(missing_block_ids)
        return False

    def _pop_lowest_priority_running_request(self, exclude: Request) -> Request | None:
        """Remove and return the lowest-priority running request."""
        if not self.running:
            return None
        candidates = [r for r in self.running if r.request_id != exclude.request_id]
        if candidates:
            victim = max(candidates, key=lambda r: r.arrival_time)
            self.running.remove(victim)
            return victim
        if exclude in self.running:
            self.running.remove(exclude)
            return exclude
        return None

    def _preempt_lowest_priority(
        self,
        victim: Request,
        scheduled_req_ids: set[str],
        num_scheduled_tokens: dict[str, int],
        output: SchedulerOutput,
    ) -> dict | None:
        """Preempt a request that has already been removed from the running queue.

        If the victim was already scheduled in this iteration, it is removed
        from the scheduled output and its token budget is returned.
        """
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

        if self.config.max_cpu_offload_blocks > 0:
            resident_block_ids = [
                block_id for block_id in victim.cached_block_ids + victim.allocated_block_ids
                if self.kv_cache_manager.blocks[block_id].location == KVBlockLocation.NPU
                and self.kv_cache_manager.blocks[block_id].physical_page_id is not None
                and self.kv_cache_manager.blocks[block_id].pending_job_id is None
                and self.kv_cache_manager.blocks[block_id].ref_cnt == 1
            ]
            if resident_block_ids:
                try:
                    job = self.kv_cache_manager.build_cpu_store_job(
                        resident_block_ids,
                        request_id=victim.request_id,
                    )
                except RuntimeError:
                    job = None
                if job is not None:
                    output.transfer_jobs.append(job)
                    output.jobs_to_flush.add(job.job_id)
                    self.num_cpu_store_jobs += 1
                    self.num_cpu_store_blocks += len(resident_block_ids)
                else:
                    self._reset_preempted_request_to_recompute(victim)
            else:
                self._reset_preempted_request_to_recompute(victim)
        else:
            self._reset_preempted_request_to_recompute(victim)
        victim.status = RequestStatus.PREEMPTED
        self.waiting.appendleft(victim)
        return {"request": victim, "returned_tokens": returned_tokens}

    def _reset_preempted_request_to_recompute(self, request: Request) -> None:
        self._free_request_blocks(request)
        request.num_computed_tokens = 0
        request.cached_block_ids = []
        request.allocated_block_ids = []

    def _free_request_blocks(self, request: Request) -> None:
        self.kv_cache_manager.release_blocks_by_ids(
            request.cached_block_ids,
            request.allocated_block_ids,
        )
        request.cached_block_ids = []
        request.allocated_block_ids = []

    def _cache_completed_blocks(self, request: Request) -> None:
        """Register completed blocks in the prefix cache."""
        if not self.config.enable_prefix_cache:
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
