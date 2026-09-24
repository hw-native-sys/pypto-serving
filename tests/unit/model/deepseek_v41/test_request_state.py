# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Lifecycle evidence uses no model math or device emulation."""
import pytest
from pypto_serving.model.deepseek_v41.request_state import RequestLedger


def item(key="a", partition=0, start=0, tokens=(4, 5), length=4, pages=(3,)):
    return key, partition, start, tokens, length, {"window": pages}


def test_slots_survive_pause_and_batch_reorder():
    ledger = RequestLedger(max_requests=2, max_seq_len=128)
    first = ledger.begin_prefill([item("a"), item("b")])
    slots = {r.request_id: r.state_slot for r in first.requests}
    ledger.commit(first)
    second = ledger.begin_prefill([item("b", start=2)])
    assert second.requests[0].state_slot == slots["b"]
    assert ledger.owners["a"].slot == slots["a"]
    assert second.positions == (2, 3)
    ledger.commit(second)
    with pytest.raises(ValueError, match="stale"):
        ledger.commit(first)


def test_bad_batch_does_not_reserve_partial_ownership():
    ledger = RequestLedger(max_requests=1, max_seq_len=128)
    with pytest.raises(ValueError, match="capacity"):
        ledger.begin_prefill([item("a"), item("b")])
    assert ledger.owners == {} and ledger.free[0] == [0]


def test_failed_chunk_invalidates_prefix_and_clears_new_pages():
    ledger = RequestLedger(max_requests=2, max_seq_len=128)
    step = ledger.begin_prefill([item()])
    ledger.commit(step)
    failed = ledger.begin_prefill([item(start=2, pages=(3, 9))])
    calls = []
    ledger.abort(failed, lambda key, owner: calls.append((key, owner.slot, owner.pages["window"])))
    assert calls == [("a", 0, (3, 9))]
    assert ledger.owners == {} and ledger.pending is None
    with pytest.raises(ValueError, match="position zero"):
        ledger.begin_prefill([item(start=2)])
    retry = ledger.begin_prefill([item()])
    assert retry.requests[0].state_slot == 0


def test_reset_failure_keeps_slot_and_blocks_future_requests():
    ledger = RequestLedger(max_requests=1, max_seq_len=128)
    step = ledger.begin_prefill([item()])
    def reset(*args):
        raise RuntimeError("device lost")
    with pytest.raises(RuntimeError, match="device lost"):
        ledger.abort(step, reset)
    assert "a" in ledger.owners and not ledger.free[0]
    with pytest.raises(RuntimeError, match="recovery"):
        ledger.begin_prefill([item("b")])


def test_cannot_release_until_forward_completes():
    ledger = RequestLedger(max_requests=1, max_seq_len=128)
    step = ledger.begin_prefill([item()])
    with pytest.raises(RuntimeError, match="in-flight"):
        ledger.release(["a"], lambda *args: None)
    ledger.commit(step)
    ledger.release(["a", "a", "unknown"], lambda *args: None)
    assert ledger.free[0] == [0]


def test_decode_continues_prefill_and_cannot_skip_or_repeat_positions():
    ledger = RequestLedger(max_requests=2, max_seq_len=128)
    first = ledger.begin_prefill([item(length=2)])
    slot = first.requests[0].state_slot
    ledger.commit(first)
    for position in range(2, 6):
        step = ledger.begin_decode([("a", 0, position, 7, {"window": (3,)})])
        assert step.positions == (position,) and step.requests[0].state_slot == slot
        ledger.commit(step)
    for position in (4, 7):
        with pytest.raises(ValueError, match="committed position"):
            ledger.begin_decode([("a", 0, position, 7, {"window": (3,)})])


def test_decode_rejects_partial_prefill_and_unknown_requests():
    ledger = RequestLedger(max_requests=2, max_seq_len=128)
    ledger.commit(ledger.begin_prefill([item(length=4)]))
    for key in ("a", "unknown"):
        with pytest.raises(ValueError, match="completed prefill"):
            ledger.begin_decode([(key, 0, 2, 7, {"window": (3,)})])
