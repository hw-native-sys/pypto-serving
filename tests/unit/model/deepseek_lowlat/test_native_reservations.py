# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from dataclasses import replace

import pytest

from pypto_serving.model.deepseek_lowlat.scheduling import LowlatSchedulingPolicy
from pypto_serving.config.decoding import DecodeReservation
from pypto_serving.model.deepseek_lowlat.decoding import choose_native_parent
from pypto_serving.model.deepseek_lowlat.contract import build_lowlat_cache_group_specs
from pypto_serving.serving.memory.kv_cache import KvCacheManager
from pypto_serving.serving.sched.scheduler import Request, RequestStatus, Scheduler, SchedulerConfig
from pypto_serving.serving.server.ipc import (DecodeRequest, DecodeReservationData, StepCommand,
                                             decode_command, encode_command)


@pytest.mark.parametrize(("remaining", "cycles", "capacity"),
                         [(1, 1, 1), (6, 1, 6), (7, 2, 7), (12, 2, 12), (23, 2, 12), (24, 2, 12)])
def test_parent_tail_policy_and_explicit_output_capacity(remaining, cycles, capacity):
    assert choose_native_parent(remaining, []) == (cycles, remaining)
    r = DecodeReservation(1, "r", 7, cycles, capacity, 256, remaining, 0)
    assert choose_native_parent(remaining, [r]) == ((2, 12) if remaining == 24 else None)
    assert choose_native_parent(remaining, [r, replace(r, reservation_id=2, frame_id=1)]) is None


def scheduler(remaining=24, *, eos=None):
    manager = KvCacheManager(enable_prefix_cache=False)
    manager.init_groups(build_lowlat_cache_group_specs(max_seq_len=512), max_batch_size=1)
    config = SchedulerConfig(max_num_running_reqs=1, max_num_scheduled_tokens=128,
        max_seq_len=512, enable_prefix_cache=False, num_speculative_tokens=5,
        async_scheduling=True, execution_policy=LowlatSchedulingPolicy())
    instance = Scheduler(config, manager)
    request = Request("r", [1, 2, 3, 4], remaining+1, temperature=0, eos_token_id=eos)
    instance.add_request(request)
    prefill = instance.schedule()
    instance.advance_after_schedule(prefill)
    instance.update_from_output(prefill, {"r": [9]})
    return instance, request


def accept(instance):
    step = instance.schedule()
    if step.scheduled_requests:
        instance.advance_after_schedule(step)
    return step


def test_two_parents_have_separate_reservations_and_reconcile_actual_variable_counts():
    instance, request = scheduler()
    a, b = accept(instance), accept(instance)
    first, second = a.scheduled_requests[0], b.scheduled_requests[0]
    assert first.decode_reservation.cycles == second.decode_reservation.cycles == 2
    assert first.decode_reservation.frame_id != second.decode_reservation.frame_id
    assert first.decode_reservation.generation == request.execution_generation > 0
    assert request.num_output_placeholders == 24
    assert instance.schedule().is_empty
    assert first.block_ids_by_group == second.block_ids_by_group
    # Even the first prefill claimed the effective request budget plus6real slots.
    assert len(first.block_ids_by_group["hca_state"]) == (4+25+6+7)//8
    instance.update_from_output(a, {"r": [10, 11]})
    assert request.num_output_placeholders == 12
    instance.update_from_output(b, {"r": [12, 13]})
    assert request.num_output_placeholders == 0
    assert request.output_token_ids == [9, 10, 11, 12, 13]
    assert request.num_computed_tokens == request.num_prompt_tokens+len(request.output_token_ids)-1
    assert not instance._pending_reservations
    assert instance.config.num_speculative_tokens == 5


def test_23token_budget_does_not_accept_a_second_parent_before_retirement():
    instance, _ = scheduler(23)
    a = accept(instance)
    assert instance.schedule().is_empty
    instance.update_from_output(a, {"r": [10, 11]})
    assert accept(instance).scheduled_requests[0].decode_reservation.remaining_lower_bound == 21


@pytest.mark.parametrize("terminal", ["eos", "abort"])
def test_eos_or_abort_discards_and_retires_the_already_accepted_successor(terminal):
    instance, request = scheduler(eos=42)
    a, b = accept(instance), accept(instance)
    if terminal == "abort":
        instance.abort_request("r")
    instance.update_from_output(a, {"r": [42, 10]})
    assert request.status == (RequestStatus.FINISHED_EOS if terminal == "eos" else RequestStatus.FINISHED_ABORTED)
    assert len(instance._pending_reservations["r"]) == 1
    before = list(request.output_token_ids)
    instance.update_from_output(b, {"r": [20, 21]})
    assert request.output_token_ids == before
    assert not instance._pending_reservations
    assert not a.scheduled_requests[0].group_blocks_retained
    assert not b.scheduled_requests[0].group_blocks_retained


def test_wire_round_trip_preserves_generation_cycle_budget_range_and_frame():
    reservation = DecodeReservation(13, "r", 5, 2, 12, 256, 24, 1)
    dr = DecodeRequest("r", -1, 128, [], decode_reservation=DecodeReservationData(**vars(reservation)))
    command = StepCommand([], [], [dr], [], 88)
    restored = decode_command(encode_command(command)).decode_requests[0].decode_reservation
    assert {name: getattr(restored, name) for name in restored.__struct_fields__} == vars(reservation)


@pytest.mark.parametrize("field", ["reservation_id", "generation", "physical_input_start", "physical_input_end",
                                   "raw_output_tokens", "remaining_output_tokens", "next_epoch"])
def test_physical_retirement_rejects_a_corrupt_cursor_budget_or_generation(field):
    from pypto_serving.model.deepseek_lowlat.decoding import NativeDecodeRetirement
    from pypto_serving.model.deepseek_lowlat.decoding import validate_native_retirement

    reservation = DecodeReservation(1, "r", 7, 2, 12, 256, 24, 0)
    receipt = NativeDecodeRetirement(1, "r", 7, 128, 131, (1, 2), (0, 1), 3, 21, 4)
    validate_native_retirement(receipt, reservation, previous_input_end=128, previous_raw_outputs=1,
                               max_output_tokens=25, actual_token_count=3)
    value = getattr(receipt, field)
    bad = replace(receipt, **{field: 0 if field == "next_epoch" else value+1})
    with pytest.raises(ValueError, match="physical retirement"):
        validate_native_retirement(bad, reservation, previous_input_end=128, previous_raw_outputs=1,
                                   max_output_tokens=25, actual_token_count=3)


def test_publication_uses_actual_committed_range_after_draft_rejection():
    instance, request = scheduler()
    a = accept(instance)
    scheduled = a.scheduled_requests[0]
    scheduled.confirmed_input_end = 6
    observed = []
    original = instance.kv_cache_manager.cache_group_blocks_from_snapshot
    def track(hashes, confirmed_tokens, *args, **kwargs):
        observed.append(confirmed_tokens)
        return original(hashes, confirmed_tokens, *args, **kwargs)
    instance.kv_cache_manager.cache_group_blocks_from_snapshot = track
    instance.update_from_output(a, {"r": [10, 11]})
    assert observed == [6]
    assert request.num_computed_tokens == 6


@pytest.mark.parametrize("budget", [1, 6, 7, 12, 23, 24, 31, 63, 127, 255])
def test_continuous_variable_acceptance_reconciles_two_inflight_parents_through_the_tail(budget):
    from pypto_serving.model.deepseek_lowlat.decoding import NativeDecodeRetirement
    from pypto_serving.model.deepseek_lowlat.decoding import validate_native_retirement

    instance, request = scheduler(budget)
    queue = []
    raw_remaining, physical_end, raw_count, epoch = budget, 4, 1, 1
    iteration = 0
    while raw_remaining or queue:
        while len(queue) < 2:
            step = accept(instance)
            if step.is_empty:
                break
            queue.append(step)
        assert queue, (budget, raw_remaining)
        step = queue.pop(0)
        scheduled = step.scheduled_requests[0]
        reservation = scheduled.decode_reservation
        advances, accepted, emitted = [], [], 0
        for cycle in range(reservation.cycles):
            # Independent cycle acceptance, including0and5; never infer from parent width.
            count = (iteration+cycle*5) % 6
            advances.append(count+1)
            accepted.append(count)
            count_emitted = min(count+1, raw_remaining)
            emitted += count_emitted
            raw_remaining -= count_emitted
            assert count_emitted > 0
        receipt = NativeDecodeRetirement(reservation.reservation_id, "r", request.execution_generation,
            physical_end, physical_end+sum(advances), tuple(advances), tuple(accepted), emitted,
            raw_remaining, epoch+reservation.cycles)
        validate_native_retirement(receipt, reservation, previous_input_end=physical_end,
            previous_raw_outputs=raw_count, max_output_tokens=budget+1, actual_token_count=emitted)
        physical_end, epoch = receipt.physical_input_end, receipt.next_epoch
        raw_count += emitted
        scheduled.confirmed_input_end = physical_end
        instance.update_from_output(step, {"r": list(range(100+iteration, 100+iteration+emitted))})
        assert request.num_output_placeholders == sum(s.scheduled_requests[0].decode_reservation.max_output_tokens
                                                      for s in queue)
        if not request.status.is_finished:
            assert request.num_computed_tokens == 4+len(request.output_token_ids)+request.num_output_placeholders-1
        iteration += 1
        assert iteration <= budget
    assert len(request.output_token_ids) == budget+1 and raw_count == budget+1
    assert request.status == RequestStatus.FINISHED_LENGTH and not instance._pending_reservations


def test_successor_snapshot_keeps_actual_pages_unavailable_until_retirement():
    instance, request = scheduler(eos=42)
    a, b = accept(instance), accept(instance)
    names = list(a.scheduled_requests[0].block_ids_by_group)
    totals = {s.name: s.num_blocks for s in instance.kv_cache_manager.group_specs}
    instance.update_from_output(a, {"r": [42, 10]})
    assert all(instance.kv_cache_manager.group_num_free_blocks(name, 0) < totals[name] for name in names)
    instance.update_from_output(b, {"r": [20, 21]})
    assert all(instance.kv_cache_manager.group_num_free_blocks(name, 0) == totals[name] for name in names)


def test_same_request_id_cannot_start_a_new_generation_while_old_reservation_is_pending():
    instance, request = scheduler()
    a, b = accept(instance), accept(instance)
    old_generation = request.execution_generation
    instance.abort_request("r")
    with pytest.raises(ValueError, match="still owns a generation"):
        instance.add_request(Request("r", [1, 2, 3, 4], 8))
    instance.update_from_output(a, {"r": [10, 11]})
    instance.update_from_output(b, {"r": [12, 13]})
    next_request = Request("r", [1, 2, 3, 4], 8)
    instance.add_request(next_request)
    assert next_request.execution_generation > old_generation
