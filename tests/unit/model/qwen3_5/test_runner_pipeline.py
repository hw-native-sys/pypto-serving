# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pending ownership and request-state isolation without an NPU."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from pypto_serving.config.types import DecodeBatch, PrefillBatch
from pypto_serving.model.qwen3_5.npu_runner import Qwen35ModelRunner
from pypto_serving.model.qwen3_5.task_args import allocate_host_slots


class _Dispatch:
    def __init__(self, slot, *, fail: bool = False) -> None:
        self.slot = slot
        self.fail = fail
        self.waited = False

    def wait(self) -> None:
        self.waited = True
        if self.fail:
            raise RuntimeError("device failure")
        self.slot.logits[0, 7] = 42


def _runner(monkeypatch):
    compiled = SimpleNamespace(prefill="prefill", decode="decode")
    runner = Qwen35ModelRunner(compiled=compiled, device_id=0)
    runner._prefill_slots = allocate_host_slots(1)
    runner._decode_slots = allocate_host_slots(1)
    runner._static_args = ()
    freed = []
    runner._l3_worker = SimpleNamespace(free_tensor=freed.append)

    def new_state(request_id):
        state = (object(),)
        runner._request_states[request_id] = state
        return state

    monkeypatch.setattr(runner, "_new_request_state", new_state)
    return runner, freed


def test_prefill_and_decode_stay_pending_until_reclaim(monkeypatch) -> None:
    runner, freed = _runner(monkeypatch)
    dispatches = []

    def submit(_program, *args):
        slot = runner._prefill_slots[0] if not dispatches else runner._decode_slots[0]
        dispatch = _Dispatch(slot)
        dispatches.append((dispatch, args))
        return dispatch

    monkeypatch.setattr(runner, "_submit_l3", submit)
    assert runner.supports_async_prefill_reclaim and runner.supports_async_decode_reclaim
    prefill = PrefillBatch(
        request_ids=["a"], token_ids=torch.tensor([2, 3]), input_embeddings=None,
        seq_lens=[2], chunk_lens=[2], chunk_offsets=[0], chunk_starts=[0],
    )
    prepared = runner.prepare_prefill(None, prefill, buffer_slot=0)
    pending = runner.dispatch_prepared_prefill(None, prefill, prepared)
    assert prepared.slot.busy and not dispatches[0][0].waited
    with pytest.raises(RuntimeError, match="pending reclaim"):
        runner.prepare_prefill(None, prefill, buffer_slot=0)
    assert runner.reclaim_prepared_prefill(pending).logits[0, 7] == 42
    assert dispatches[0][0].waited and not prepared.slot.busy

    decode = DecodeBatch(
        request_ids=["a"], token_ids=torch.tensor([[5]]), hidden_states=None,
        seq_lens=torch.tensor([3]),
    )
    prepared_decode = runner.prepare_decode(None, decode, buffer_slot=0)
    pending_decode = runner.dispatch_prepared_decode(None, decode, prepared_decode)
    assert prepared_decode.slot.busy and not dispatches[1][0].waited
    assert prepared_decode.slot.token_ids[2] == 5
    assert runner.reclaim_prepared_decode(pending_decode).logits[0, 7] == 42
    assert dispatches[1][0].waited and not prepared_decode.slot.busy
    runner.release_finished_requests(["a"])
    assert len(freed) == 1 and not runner._request_states


def test_reclaim_failure_releases_state_and_slot(monkeypatch) -> None:
    runner, freed = _runner(monkeypatch)
    monkeypatch.setattr(runner, "_submit_l3", lambda _program, *args: _Dispatch(runner._prefill_slots[0], fail=True))
    batch = PrefillBatch(
        request_ids=["failed"], token_ids=torch.tensor([2]), input_embeddings=None,
        seq_lens=[1], chunk_lens=[1], chunk_offsets=[0], chunk_starts=[0],
    )
    prepared = runner.prepare_prefill(None, batch, buffer_slot=0)
    pending = runner.dispatch_prepared_prefill(None, batch, prepared)
    with pytest.raises(RuntimeError, match="device failure"):
        runner.reclaim_prepared_prefill(pending)
    assert not prepared.slot.busy and not runner._request_states and len(freed) == 1


def test_prepare_only_cancellation_releases_slots(monkeypatch) -> None:
    runner, freed = _runner(monkeypatch)
    prefill = PrefillBatch(
        request_ids=["cancelled"], token_ids=torch.tensor([2]), input_embeddings=None,
        seq_lens=[1], chunk_lens=[1], chunk_offsets=[0], chunk_starts=[0],
    )
    prepared = runner.prepare_prefill(None, prefill, buffer_slot=0)
    assert prepared.slot.busy and runner._reservations == [prepared]
    runner.release_finished_requests(["cancelled"])
    assert not prepared.slot.busy and not runner._reservations and not freed
    replacement = runner.prepare_prefill(None, prefill, buffer_slot=0)
    assert replacement.slot.busy
    runner.release_finished_requests(["cancelled"])

    runner._request_states["cancelled"] = (object(),)
    decode = DecodeBatch(
        request_ids=["cancelled"], token_ids=torch.tensor([[5]]), hidden_states=None,
        seq_lens=torch.tensor([2]),
    )
    prepared_decode = runner.prepare_decode(None, decode, buffer_slot=0)
    runner.release_finished_requests(["cancelled"])
    assert not prepared_decode.slot.busy and not runner._reservations and len(freed) == 1
    assert runner.prepare_decode(None, decode, buffer_slot=0).slot.busy
