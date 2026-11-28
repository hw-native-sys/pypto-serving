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

from pypto_serving.serving.sched.scheduler import RequestStatus, Scheduler


class PrefillScheduler(Scheduler):
    def __init__(self, *, config, kv_cache_manager):
        super().__init__(replace(config, stop_after_prefill=True), kv_cache_manager)

    def _schedule(self, output, *, allow_reject_stalled):
        cache = self.kv_cache_manager
        tokens = self.config.max_prefill_tokens_per_request or self.config.max_num_scheduled_tokens
        rows = min((group.max_requests_per_partition for group in cache.group_specs
                    if group.max_requests_per_partition is not None), default=self.config.max_num_running_reqs)
        # This is a single selection pass, not a second request/owner lifecycle.
        partitions = cache.group_partition_count if cache.has_groups else 1
        self._remaining = [tokens] * partitions
        self._rows = [rows] * partitions
        self._continuing = {}
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

    def _partition_token_budget(self, request, token_budget, partition):
        if not self.kv_cache_manager.has_groups:
            partition = 0
        if partition is None:
            available = max((tokens for partition, tokens in enumerate(self._remaining)
                             if self._can_admit_waiting_partition(partition)), default=0)
        elif (self._rows[partition] <= 0
              or self._continuing.get(partition, request.request_id) != request.request_id):
            return 0
        else:
            available = self._remaining[partition]
        return min(token_budget, available)

    def _try_allocate_request_blocks(self, request, num_new_tokens, *, scheduled=None):
        partitions = None
        if self.kv_cache_manager.has_groups and request.cache_partition is None:
            partitions = [partition for partition, tokens in enumerate(self._remaining)
                          if tokens >= num_new_tokens and self._can_admit_waiting_partition(partition)]
        return super()._try_allocate_request_blocks(
            request, num_new_tokens, scheduled=scheduled, partitions=partitions,
        )

    def _on_scheduled(self, scheduled):
        partition = scheduled.cache_partition if self.kv_cache_manager.has_groups else 0
        self._rows[partition] -= 1
        if scheduled.num_computed_tokens + scheduled.num_new_tokens < scheduled.request.num_prefill_tokens:
            self._continuing[partition] = scheduled.request.request_id
            self._remaining[partition] = 0
        else:
            self._continuing.pop(partition, None)
            self._remaining[partition] -= scheduled.num_new_tokens

    def _preempt_lowest_priority(self, *args):
        return None  # Backpressure must not evict a continuation to open another context.
