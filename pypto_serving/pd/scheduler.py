# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Continue one prefill per partition; use only its terminal slack for new work."""

from dataclasses import replace
from math import ceil

from pypto_serving.serving.memory.kv_cache import KVCacheCapacityError
from pypto_serving.serving.sched.scheduler import RequestStatus, ScheduledRequest, Scheduler


class PrefillScheduler(Scheduler):
    def __init__(self, *, config, kv_cache_manager, waiting_window=5):
        super().__init__(replace(config, stop_after_prefill=True), kv_cache_manager)
        if type(waiting_window) is not int or waiting_window < 1:
            raise ValueError("Prefill waiting window must be a positive integer")
        self.waiting_window = waiting_window
        self._waiting_head = None
        self._lookahead = 0
        self._next_partition = 0
        self._deferred = {}
        self._prepared = {}

    def _schedule(self, output, *, allow_reject_stalled):
        cache = self.kv_cache_manager
        tokens = self.config.max_prefill_tokens_per_request or self.config.max_num_scheduled_tokens
        rows = min((group.max_requests_per_partition for group in cache.group_specs
                    if group.max_requests_per_partition is not None), default=self.config.max_num_running_reqs)
        # This is a single selection pass, not a second request/owner lifecycle.
        partitions = cache.group_partition_count if cache.has_groups else 1
        self._remaining = [tokens] * partitions
        self._rows = [rows] * partitions
        self._scheduled_end = [0] * partitions
        self._continuing = {}
        self._chunk = self._single_prefill_dispatch_limit()
        for request in self.running:
            if (request.status == RequestStatus.RUNNING and request.is_prefill
                    and not request.prefill_complete and not request.terminal_prefill_in_flight):
                partition = request.cache_partition if cache.has_groups else 0
                self._continuing.setdefault(partition, request.request_id)
                if self._continuing[partition] == request.request_id and request.scheduling_held:
                    self._remaining[partition] = 0
        # A pending send can free pages even with no compute batch in flight.
        return super()._schedule(output, allow_reject_stalled=False)

    def _can_admit_waiting(self):
        return any(self._can_admit_waiting_partition(p) for p in range(len(self._remaining)))

    def _can_admit_waiting_partition(self, partition):
        return self._remaining[partition] > 0 and self._rows[partition] > 0 and partition not in self._continuing

    def _can_scan_waiting(self, deferred):
        head = (deferred or self.waiting)[0].request_id
        if head != self._waiting_head:
            self._waiting_head = head
            self._lookahead = self.waiting_window - 1
        return len(deferred) <= self._lookahead

    def _on_waiting_scheduled(self, *, bypassed):
        if bypassed:
            self._lookahead -= 1

    def _group_budget(self, partition, token_budget):
        if self._rows[partition] <= 0 or partition in self._continuing:
            return 0
        return min(token_budget, self._remaining[partition])

    def _wait_rounds(self, partition):
        request = self.requests.get(self._continuing.get(partition))
        if request is None:
            return 1
        if request.scheduling_held:
            return float("inf")
        end = max(request.num_computed_tokens, self._scheduled_end[partition])
        return max(1, ceil((request.num_prefill_tokens - end) / self._chunk))

    def _admission_state(self, token_budget, head, partition):
        cache = self.kv_cache_manager
        partitions = range(cache.group_partition_count) if partition is None else (partition,)
        ddr = cache.tiering
        return (head, ddr.tier.revision if ddr is not None and partition is None else 0,
                tuple((self._group_budget(p, token_budget), self._wait_rounds(p),
                       cache.group_admission_state(p)) for p in partitions))

    def _first_chunk(self, request, budget, hit):
        return super()._limit_scheduled_tokens(request, budget, computed=self._prefill_start(request, hit))

    def _candidate(self, request, budget, match, *, use_ddr=False):
        tables, hit, partition = match
        cache = self.kv_cache_manager
        fresh, size, needs_load = (cache.tiering.restore_demand(tables, partition)
                                   if use_ddr else (None, 0, False))
        chunk = self._first_chunk(request, budget, hit)
        end = self._prefill_start(request, hit) + chunk
        if not chunk or not cache.can_place_group_prefix(partition, tables, hit, end, fresh):
            return None
        return match, chunk, needs_load, size

    def _choose_first_chunk(self, request, token_budget, *, head):
        cache, boundary = self.kv_cache_manager, request.group_checkpoint_boundary or 0
        hashes, tiering = request.group_block_hashes, self.kv_cache_manager.tiering
        local = cache.select_group_prefix_blocks(hashes, max_cache_hit_tokens=boundary) if boundary else ({}, 0, None)
        _, hit, origin = local
        origin_cost = float("inf")
        if hit >= self._chunk:
            plan = self._candidate(request, self._group_budget(origin, token_budget), local)
            if plan is not None:
                return plan, None
            if self._candidate(request, self._chunk, local) is not None:
                wait = self._wait_rounds(origin)
                if wait <= 2:
                    return None, origin
                origin_cost = wait + ceil((request.num_prefill_tokens - self._prefill_start(request, hit))
                                          / self._chunk)

        best, best_score = None, None
        for partition in range(cache.group_partition_count):
            budget = self._group_budget(partition, token_budget)
            if budget <= 0 or (hit >= self._chunk and partition == origin):
                continue
            if tiering is not None and boundary and tiering.tier.pages:
                match = tiering.match_prefix(hashes, boundary, partition=partition, score=False)
            elif partition == origin:
                match = local
            elif hit:
                match = cache.select_group_prefix_blocks(hashes, max_cache_hit_tokens=boundary, partition=partition)
            else:
                match = {}, 0, None
            if not match[1]:
                match = {}, 0, partition
            candidate = self._candidate(request, budget, match, use_ddr=tiering is not None)
            if candidate is None:
                continue
            _, chunk, needs_load, size = candidate
            if needs_load and not head:
                continue
            remaining = request.num_prefill_tokens - self._prefill_start(request, match[1]) - chunk
            # Estimate copy at 6.6 GB/s and compute at 700 ms per Prefill chunk.
            copy_rounds = (max(1, size / 4.62e9) + tiering.restore_queue_rounds(partition)) if needs_load else 0
            cost = 1 + ceil(remaining / self._chunk) + copy_rounds
            if hit >= self._chunk and cost >= origin_cost:
                continue
            score = ((cost, -match[1], size) if hit >= self._chunk else (-match[1], size, cost))
            score += ((partition - self._next_partition) % cache.group_partition_count,)
            if best_score is None or score < best_score:
                best, best_score = candidate, score
        return best, None

    def _prepare_waiting_group(self, request, token_budget, *, head):
        if self._requires_single_prefill_dispatch():
            return super()._prepare_waiting_group(request, token_budget, head=head)
        cache, request_id = self.kv_cache_manager, request.request_id
        tiering = cache.tiering
        if tiering is not None and tiering.tier.error:
            raise RuntimeError("DDR work is quarantined") from tiering.tier.error
        prepared = self._prepared.get(request_id)
        if prepared is not None:
            budget = self._group_budget(request.cache_partition, token_budget)
            if budget < prepared.num_new_tokens:
                return False
            return tiering.prepare(request_id, request.group_block_hashes, request.num_prompt_tokens,
                                   maximum=request.group_checkpoint_boundary)[0]

        previous = self._deferred.get(request_id)
        if previous is not None and previous[1] == self._admission_state(token_budget, head, previous[0]):
            return False
        plan, wait_partition = self._choose_first_chunk(request, token_budget, head=head)
        if plan is None:
            self._deferred[request_id] = wait_partition, self._admission_state(token_budget, head, wait_partition)
            return False
        self._deferred.pop(request_id, None)
        match, chunk, needs_load, _ = plan
        tables, hit, partition = match
        scheduled = ScheduledRequest(request, chunk, True)

        def reserve_step(blocks):
            if hit:
                self._record_waiting_group_prefix(request, cache.claim_group_prefix_blocks(
                    request_id, blocks, hit, partition,
                ))
            elif self.config.enable_prefix_cache:
                self._record_waiting_group_prefix(request, ({}, 0, partition))
            else:
                request.cache_partition = partition
            if not self._try_allocate_request_blocks(request, chunk, scheduled=scheduled):
                raise KVCacheCapacityError("Prefill first step no longer fits")
            self._prepared[request_id] = scheduled

        try:
            if needs_load:
                tiering.start_restore(request_id, request.group_block_hashes, request.num_prompt_tokens,
                                      match, reserve_step=reserve_step)
                return False
            if tiering is not None:
                tiering.touch_prefix(tables)
            reserve_step(tables)
            return True
        except KVCacheCapacityError:
            self._release_waiting_prefix_blocks(request)
            return False
        except BaseException:
            self._release_waiting_prefix_blocks(request)
            raise

    def _allocate_waiting(self, request, num_new, scheduled):
        prepared = self._prepared.pop(request.request_id, None)
        if prepared is None:
            return super()._allocate_waiting(request, num_new, scheduled)
        scheduled.read_block_ids_by_group = prepared.read_block_ids_by_group
        scheduled.group_read_snapshot = prepared.group_read_snapshot
        self._next_partition = (request.cache_partition + 1) % self.kv_cache_manager.group_partition_count
        return True

    def _free_request_blocks(self, request):
        self._deferred.pop(request.request_id, None)
        prepared = self._prepared.pop(request.request_id, None)
        if prepared is not None:
            self._release_scheduled_group_blocks(prepared)
        super()._free_request_blocks(request)

    def _limit_scheduled_tokens(self, request, token_budget):
        partition = request.cache_partition if self.kv_cache_manager.has_groups else 0
        if partition is None:
            available = max((tokens for partition, tokens in enumerate(self._remaining)
                             if self._can_admit_waiting_partition(partition)), default=0)
        elif (self._rows[partition] <= 0
              or self._continuing.get(partition, request.request_id) != request.request_id):
            return 0
        else:
            available = self._remaining[partition]
        prepared = self._prepared.get(request.request_id)
        if prepared is not None:
            available = min(available, prepared.num_new_tokens)
        return super()._limit_scheduled_tokens(request, min(token_budget, available))

    def _try_allocate_request_blocks(self, request, num_new_tokens, *, scheduled=None):
        cache = self.kv_cache_manager
        if cache.has_groups and request.cache_partition is None:
            candidates = [partition for partition, tokens in enumerate(self._remaining)
                          if tokens >= num_new_tokens and self._can_admit_waiting_partition(partition)]
            partition = cache.select_group_partition(request.request_id,
                request.num_computed_tokens + num_new_tokens, partitions=candidates)
            if partition is None:
                return False
            request.cache_partition = partition
        return super()._try_allocate_request_blocks(request, num_new_tokens, scheduled=scheduled)

    def _on_scheduled(self, scheduled):
        partition = scheduled.cache_partition if self.kv_cache_manager.has_groups else 0
        self._scheduled_end[partition] = scheduled.num_computed_tokens + scheduled.num_new_tokens
        self._rows[partition] -= 1
        if self._scheduled_end[partition] < scheduled.request.num_prefill_tokens:
            self._continuing[partition] = scheduled.request.request_id
            self._remaining[partition] = 0
        else:
            self._continuing.pop(partition, None)
            self._remaining[partition] -= scheduled.num_new_tokens

    def _preempt_lowest_priority(self, *args):
        return None  # Backpressure must not evict a continuation to open another context.
