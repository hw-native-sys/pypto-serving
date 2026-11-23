# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Stable ring ownership and K5 parent policy for the lowlat scheduler adapter."""

from dataclasses import dataclass

from .decoding import DecodeReservation, choose_native_parent
from pypto_serving.serving.sched.execution import SchedulingPolicy, RequestPolicyState, StepPolicyState
from pypto_serving.serving.memory.protocol import GroupCacheCopyPlan
from .prefix_protocol import PrefixResumeDescriptor


@dataclass
class LowlatScheduleState(RequestPolicyState):
    prefix_resume: PrefixResumeDescriptor | None = None
    checkpoint_complete: bool = False
    physical_end: int = 0
    raw_output_count: int = 1


@dataclass
class LowlatStepPlan(StepPolicyState):
    checkpoint: GroupCacheCopyPlan | None = None


@dataclass(frozen=True)
class LowlatSchedulingPolicy(SchedulingPolicy):
    prefix_namespace: str = ""

    @property
    def uses_model_execution_plan(self):
        return True

    @property
    def explicit_cache_publication(self):
        return bool(self.prefix_namespace)

    @property
    def serial_prefill(self):
        return self.explicit_cache_publication

    def validate_config(self, config):
        if (
            config.num_speculative_tokens != 5
            or not config.async_scheduling
            or config.max_num_running_reqs != 1
            or config.max_num_scheduled_tokens < 12
            or (config.enable_prefix_cache and not self.prefix_namespace)
        ):
            raise ValueError(
                "native pipeline requires async B1, K5, confirmed prefix policy and12token budget"
            )

    def admit(self, scheduler, request):
        request.execution_state = LowlatScheduleState()
        if request.request_id in scheduler.requests or scheduler.pending_reservations(request.request_id):
            raise ValueError("native request ID still owns a generation")
        if self.allocation_extent(scheduler.config, request, 0) > scheduler.config.max_seq_len:
            raise ValueError("native request has no six-position physical lookahead")
        request.execution_generation = scheduler.next_execution_generation()
        request.execution_state.physical_end = request.num_prompt_tokens

    def reserve_execution_plan(self, scheduler, request):
        pending = scheduler.pending_reservations(request.request_id)
        remaining = request.max_new_tokens - len(request.output_token_ids)
        choice = choose_native_parent(remaining, pending)
        if choice is None:
            return None
        cycles, lower_bound = choice
        reservation_id = scheduler.next_reservation_id()
        return DecodeReservation(
            reservation_id,
            request.request_id,
            request.execution_generation,
            cycles,
            min(cycles * 6, lower_bound),
            request.num_prompt_tokens + request.max_new_tokens + 6,
            lower_bound,
            next(i for i in (0, 1) if i not in {r.frame_id for r in pending}),
        )

    def adopt_prefix(self, scheduler, request):
        if not (request.num_computed_tokens and self.prefix_namespace):
            return
        manager = scheduler.kv_cache_manager
        tables, pages = manager.detach_prefix_ring(request.request_id)
        request.execution_state.prefix_resume = PrefixResumeDescriptor(
            self.prefix_namespace,
            request.request_id,
            request.execution_generation,
            request.num_computed_tokens,
            request.num_prompt_tokens - request.num_computed_tokens,
            1,
            pages,
            manager.group_checkpoint_ranges(request.num_computed_tokens),
            tables,
        )

    def prepare_steps(self, scheduler, output):
        if not self.prefix_namespace:
            return
        manager = scheduler.kv_cache_manager
        for scheduled in output.scheduled_requests:
            request = scheduled.request
            end = scheduled.num_computed_tokens + scheduled.num_new_tokens
            if (
                scheduled.is_prefill
                and end == request.group_checkpoint_boundary
                and end > 0
                and not request.execution_state.checkpoint_complete
            ):
                plan = manager.reserve_prefix_checkpoint(
                    request.request_id,
                    generation=request.execution_generation,
                    boundary=end,
                    snapshot=scheduled.block_ids_by_group,
                )
                scheduled.execution_plan = LowlatStepPlan(plan)
            auxiliary = []
            if (
                isinstance(scheduled.execution_plan, LowlatStepPlan)
                and scheduled.execution_plan.checkpoint is not None
            ):
                auxiliary.append(scheduled.execution_plan.checkpoint.destination_tables)
            if (
                scheduled.is_prefill
                and request.execution_state.prefix_resume is not None
                and scheduled.num_computed_tokens == request.execution_state.prefix_resume.hit_tokens
            ):
                auxiliary.append(
                    {
                        name: tuple(p for p in ids if p >= 0)
                        for name, ids in request.execution_state.prefix_resume.tables.items()
                    }
                )
            # Generic snapshot pins keep copy endpoints live through cancellation.
            leases = []
            try:
                for snapshot in auxiliary:
                    leases.append(manager.retain_group_block_snapshot(snapshot, scheduled.cache_partition))
            except BaseException:
                for lease in leases:
                    manager.release_group_block_snapshot(lease)
                raise
            scheduled.cache_aux_snapshots = tuple(leases)

    def limit_prefill(self, request, limit):
        boundary = request.group_checkpoint_boundary or 0
        return (
            min(limit, boundary - request.num_computed_tokens)
            if (self.prefix_namespace and request.num_computed_tokens < boundary)
            else limit
        )

    def allocation_extent(self, config, request, extent):
        return (
            request.num_prompt_tokens
            + request.max_new_tokens
            + (6 if request.max_new_tokens > 1 else 0)
        )

    def reset_waiting(self, request):
        request.execution_state.prefix_resume = None

    def confirmed_prefill(self, scheduler, request):
        if self.prefix_namespace:
            request.terminal_prefill_in_flight = False
            scheduler.kv_cache_manager.release_prefix_resume_pins(request.request_id)
