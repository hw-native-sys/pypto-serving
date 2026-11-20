# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock
import threading

import pytest
import torch

from pypto_serving.model.deepseek_lowlat.decoding import DecodeReservation
from pypto_serving.config.types import DecodeBatch
from pypto_serving.serving.model_extension import RequestExecutionContext
from pypto_serving.serving.model_extension import ModelStepExtension
from pypto_serving.model.deepseek_lowlat.native_pipeline_runner import NativePipelineRunner
from pypto_serving.model.deepseek_lowlat.speculation import NativeState, validate_packets
from pypto_serving.model.deepseek_lowlat.state import LowlatRequestState
from test_speculation import adapter, published


def runner(remaining=24):
    instance = NativePipelineRunner.__new__(NativePipelineRunner)
    instance._closed = False
    instance.state = LowlatRequestState("r", 2, 128, 128, 128, 9, "decode")
    instance.request_contexts = {"r": RequestExecutionContext("r", 128, remaining+1, execution_generation=7)}
    instance.cache = SimpleNamespace(specs={"ori": object()}, _page_ids={"ori": (4, 2)}, release=Mock())
    instance.transfers = Mock(poisoned=False)
    instance._native_lock = threading.RLock()
    instance._accepted, instance._seen_reservations = [], set()
    frames, hosts, _ = adapter()
    frames.worker.submit = Mock(side_effect=lambda *a, **k: Mock())
    instance.native = SimpleNamespace(frames=frames, state=NativeState(1, 127, remaining, 9), release=Mock())
    instance.native.commit = lambda packet: setattr(instance.native, "state", packet.state)
    reservation = DecodeReservation(1, "r", 7, 2 if remaining >= 7 else 1,
        min(12 if remaining >= 7 else 6, remaining), 128+remaining+1+6, remaining, 0)
    batch = DecodeBatch(["r"], torch.tensor([0]), None, torch.tensor([999]),
        block_ids_by_group=[{"ori": [4, 2]}],
        model_extensions=(ModelStepExtension("deepseek_v4_flash_lowlat", 1, reservation),))
    return instance, batch, hosts


def publish(hosts, index, state, cycles=2):
    control, emitted = published(state, cycles=cycles, accepted=5)
    hosts[index]["s_control"].copy_(control)
    hosts[index]["s_emitted"].copy_(emitted)
    return validate_packets(control, emitted, state=state, cycles=cycles)


def accepted_pair(instance, batch):
    first = instance.dispatch_prepared_decode(None, batch, instance.prepare_decode(None, batch))
    second_reservation = replace(batch.model_extensions[0].payload, reservation_id=2, remaining_lower_bound=12, frame_id=1)
    second_batch = replace(batch, model_extensions=(replace(batch.model_extensions[0], payload=second_reservation),))
    second = instance.dispatch_prepared_decode(None, second_batch, instance.prepare_decode(None, second_batch))
    return first, second


def test_two_accepted_parents_capture_independent_frames_and_return_real_retirement_receipts():
    instance, batch, hosts = runner()
    first, second = accepted_pair(instance, batch)
    before = instance.native.state
    packet = publish(hosts, 0, before)
    publish(hosts, 1, packet.state)
    output_a = instance.reclaim_prepared_decode(first)
    assert output_a.model_extension.payload.retired_reservation_ids == (1,) and len(output_a.accepted_token_ids[0]) == 12
    assert output_a.num_draft_tokens is None
    assert instance._accepted == [second]
    hosts[0]["s_emitted"].fill_(-777)
    output_b = instance.reclaim_prepared_decode(second)
    assert output_b.model_extension.payload.retired_reservation_ids == (2,) and len(output_b.accepted_token_ids[0]) == 12
    assert output_a.accepted_token_ids[0][0] == 100
    assert instance.native.state.remaining == 0 and instance.state.kv_committed_tokens == 152
    assert instance.state.pending_handles == [] and instance.native.frames.idle()
    instance.transfers.stage_tables.assert_not_called()


def test_second_parent_can_be_accepted_while_first_result_waits_without_holding_admission_lock():
    instance, batch, hosts = runner()
    first = instance.dispatch_prepared_decode(None, batch, instance.prepare_decode(None, batch))
    entered, release = threading.Event(), threading.Event()
    def wait():
        entered.set()
        assert release.wait(5)
    first.handle.result.side_effect = wait
    packet = publish(hosts, 0, instance.native.state)
    errors = []
    def reclaim():
        try:
            instance.reclaim_prepared_decode(first)
        except Exception as exc:
            errors.append(exc)
    thread = threading.Thread(target=reclaim)
    thread.start()
    assert entered.wait(2)
    successor = replace(batch.model_extensions[0].payload, reservation_id=2, remaining_lower_bound=12, frame_id=1)
    second_batch = replace(batch, model_extensions=(replace(batch.model_extensions[0], payload=successor),))
    second = instance.dispatch_prepared_decode(None, second_batch, instance.prepare_decode(None, second_batch))
    publish(hosts, 1, packet.state)
    release.set()
    thread.join(5)
    assert not thread.is_alive() and not errors
    instance.reclaim_prepared_decode(second)


def test_end_request_captures_and_discards_both_accepted_parents_before_releasing_cache():
    instance, batch, hosts = runner()
    accepted_pair(instance, batch)
    packet = publish(hosts, 0, instance.native.state)
    publish(hosts, 1, packet.state)
    instance.native.release.side_effect = lambda: instance.native.frames.idle() or pytest.fail("live frame")
    instance.release_finished_requests(["r"])
    assert instance.state is None and instance._accepted == []
    instance.cache.release.assert_called_once()
    assert instance.native.frames.idle()


@pytest.mark.parametrize("change", ["generation", "pages", "duplicate", "third", "same_frame"])
def test_stale_or_unowned_reservation_never_submits(change):
    instance, batch, _ = runner()
    plan = instance.prepare_decode(None, batch)
    if change == "generation":
        plan = replace(plan, cache_generation=99)
    elif change == "pages":
        plan = replace(plan, pages=(("ori", (0, 1)),))
    else:
        first = instance.dispatch_prepared_decode(None, batch, plan)
        if change == "third":
            successor = replace(plan.reservation, reservation_id=2, frame_id=1, remaining_lower_bound=12)
            instance.dispatch_prepared_decode(None, batch, replace(plan, reservation=successor))
        if change == "same_frame":
            plan = replace(plan, reservation=replace(plan.reservation, reservation_id=2))
        assert first in instance.state.pending_handles
    with pytest.raises(RuntimeError):
        instance.dispatch_prepared_decode(None, batch, plan)


def test_corrupt_successor_keeps_its_owner_and_does_not_rewind_confirmed_predecessor():
    instance, batch, hosts = runner()
    first, second = accepted_pair(instance, batch)
    packet = publish(hosts, 0, instance.native.state)
    publish(hosts, 1, packet.state)
    instance.reclaim_prepared_decode(first)
    before = instance.native.state
    hosts[1]["s_control"][0, 0, 0] += 1
    with pytest.raises(ValueError):
        instance.reclaim_prepared_decode(second)
    assert instance.native.state == before and instance.native.frames.poisoned
    assert instance._accepted == [second] and second in instance.state.pending_handles
    instance.transfers.poison.assert_called_once()


def test_predecessor_can_retire_while_successor_submission_is_blocked():
    instance, batch, hosts = runner()
    first = instance.dispatch_prepared_decode(None, batch, instance.prepare_decode(None, batch))
    packet = publish(hosts, 0, instance.native.state)
    publish(hosts, 1, packet.state)
    entered, release = threading.Event(), threading.Event()
    def blocked_submit(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return Mock()
    instance.native.frames.worker.submit.side_effect = blocked_submit
    successor = replace(batch.model_extensions[0].payload, reservation_id=2, frame_id=1, remaining_lower_bound=12)
    second_batch = replace(batch, model_extensions=(replace(batch.model_extensions[0], payload=successor),))
    errors, accepted = [], []
    def dispatch():
        try:
            accepted.append(instance.dispatch_prepared_decode(None, second_batch,
                instance.prepare_decode(None, second_batch)))
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=dispatch)
    thread.start()
    assert entered.wait(2)
    retired = threading.Event()
    def reclaim():
        try:
            instance.reclaim_prepared_decode(first)
        except BaseException as error:
            errors.append(error)
        finally:
            retired.set()
    output = threading.Thread(target=reclaim)
    output.start()
    try:
        assert retired.wait(2), "predecessor capture waited for successor submit"
        assert not errors and not accepted
        queued = instance._accepted[0]
        assert not queued.submission.done()
        assert instance.state.pending_handles == [queued]
        instance.cache.release.assert_not_called()
    finally:
        release.set()
        thread.join(5)
        output.join(5)
    assert not errors and not thread.is_alive() and not output.is_alive()
    instance.reclaim_prepared_decode(accepted[0])
    assert instance.native.frames.idle() and not instance.state.pending_handles


def test_failed_reserved_submission_keeps_cache_owned_until_worker_teardown():
    instance, batch, _ = runner()
    instance.native.frames.worker.submit.side_effect = RuntimeError("submit refused")
    with pytest.raises(RuntimeError, match="submit refused"):
        instance.dispatch_prepared_decode(None, batch, instance.prepare_decode(None, batch))
    pending = instance._accepted[0]
    assert pending in instance.state.pending_handles and pending.submission.done()
    assert instance.native.frames.poisoned
    with pytest.raises(RuntimeError, match="submit refused"):
        instance.release_finished_requests(["r"])
    instance.cache.release.assert_not_called()
