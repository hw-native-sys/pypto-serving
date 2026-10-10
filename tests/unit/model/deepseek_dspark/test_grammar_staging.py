# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host-side mask repacking and K7 rank/row staging contracts."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pypto_serving.model.deepseek_dspark.npu_runner import (
    DSPARK_DECODE_BATCH,
    DSPARK_DRAFTER_QUERY_WIDTH,
    DSPARK_GRAMMAR_SEGMENTS,
    DSPARK_GRAMMAR_SEGMENT_TOKENS,
    DSPARK_GRAMMAR_SEGMENT_WORDS,
    DSPARK_MAX_LOGIT_ROWS,
    DSPARK_RANKS,
    DSPARK_VOCAB_SIZE,
    DSparkModelRunner,
)


def test_repacked_bits_preserve_model_vocabulary() -> None:
    words = np.full((DSPARK_VOCAB_SIZE // 32,), -1, dtype=np.int32)
    words[0] = -2
    words[-1] = 0x7FFF
    destination = torch.empty(
        (DSPARK_GRAMMAR_SEGMENTS, DSPARK_GRAMMAR_SEGMENT_WORDS),
        dtype=torch.int16,
    )
    DSparkModelRunner._copy_grammar_mask_row(destination, words)
    physical = np.unpackbits(destination.numpy().view(np.uint8), bitorder="little").reshape(
        DSPARK_GRAMMAR_SEGMENTS, DSPARK_GRAMMAR_SEGMENT_WORDS * 16
    )
    original = np.unpackbits(words.view(np.uint8), bitorder="little")
    np.testing.assert_array_equal(
        physical[:, :DSPARK_GRAMMAR_SEGMENT_TOKENS].reshape(-1), original
    )
    assert int(destination[0, -1]) == 0
    physical[0, -16:] = 1  # Reserved row-mode header, outside vocabulary bits.
    assert np.all(physical[:, DSPARK_GRAMMAR_SEGMENT_TOKENS:] == 1)


@pytest.mark.parametrize("strided", [False, True])
def test_direct_byte_packing_matches_bitwise_reference_and_clears_old_padding(strided):
    rng = np.random.default_rng(9)
    words = rng.integers(-(2**31), 2**31, DSPARK_VOCAB_SIZE // 32, dtype=np.int32)
    if strided:
        storage = np.zeros((len(words), 2), dtype=np.int32)
        storage[:, 0] = words
        words = storage[:, 0]
        assert not words.flags.c_contiguous
    expected_bits = np.ones((DSPARK_GRAMMAR_SEGMENTS, DSPARK_GRAMMAR_SEGMENT_WORDS * 16), np.uint8)
    expected_bits[:, :DSPARK_GRAMMAR_SEGMENT_TOKENS] = np.unpackbits(
        np.ascontiguousarray(words).view(np.uint8), bitorder="little"
    ).reshape(DSPARK_GRAMMAR_SEGMENTS, DSPARK_GRAMMAR_SEGMENT_TOKENS)
    expected = np.packbits(expected_bits, axis=1, bitorder="little").view(np.int16)
    expected[0, -1] = 0
    # A row view with a nonzero storage offset, initially containing stale bytes.
    storage = torch.zeros((2, DSPARK_GRAMMAR_SEGMENTS, DSPARK_GRAMMAR_SEGMENT_WORDS), dtype=torch.int16)
    DSparkModelRunner._copy_grammar_mask_row(storage[1], words)
    np.testing.assert_array_equal(storage[1].numpy(), expected)
    assert torch.count_nonzero(storage[0]) == 0


def test_bad_mask_is_rejected_before_destination_is_modified():
    destination = torch.full((DSPARK_GRAMMAR_SEGMENTS, DSPARK_GRAMMAR_SEGMENT_WORDS), 123, dtype=torch.int16)
    for words in (np.ones(1, dtype=np.int32), np.zeros(DSPARK_VOCAB_SIZE // 32, dtype=np.int32)):
        with pytest.raises(ValueError):
            DSparkModelRunner._copy_grammar_mask_row(destination, words)
        assert torch.all(destination == 123)


@pytest.mark.parametrize("length", range(1, 9))
def test_bulk_mask_pack_matches_each_reference_row(length):
    rng = np.random.default_rng(71)
    masks = rng.integers(-(2**31), 2**31, (length, DSPARK_VOCAB_SIZE // 32), dtype=np.int32)
    destination = torch.empty((length, DSPARK_GRAMMAR_SEGMENTS, DSPARK_GRAMMAR_SEGMENT_WORDS), dtype=torch.int16)
    DSparkModelRunner._copy_grammar_mask_rows(destination, masks)
    physical = np.unpackbits(destination.numpy().view(np.uint8), axis=-1, bitorder="little")
    np.testing.assert_array_equal(
        physical[:, :, :DSPARK_GRAMMAR_SEGMENT_TOKENS].reshape(length, -1),
        np.unpackbits(masks.view(np.uint8), axis=-1, bitorder="little"),
    )
    assert torch.all(destination[:, 0, -1] == 0)
    masks[-1].fill(0)
    destination.fill_(123)
    with pytest.raises(ValueError, match="no allowed token"):
        DSparkModelRunner._copy_grammar_mask_rows(destination, masks)
    assert torch.all(destination == 123)


def test_k7_mask_rows_and_draft_caps_are_isolated() -> None:
    class State:
        def plan_draft_rows(self, token_ids):
            assert token_ids == [11, 12, 13]
            masks = np.full((3, DSPARK_VOCAB_SIZE // 32), -1, dtype=np.int32)
            masks[:, 0] = -2
            return 2, masks

    runner = DSparkModelRunner.__new__(DSparkModelRunner)
    masks = torch.full(
        (DSPARK_RANKS, DSPARK_MAX_LOGIT_ROWS, DSPARK_GRAMMAR_SEGMENTS, DSPARK_GRAMMAR_SEGMENT_WORDS),
        -1, dtype=torch.int16,
    )
    counts = torch.full((DSPARK_RANKS, DSPARK_DECODE_BATCH), DSPARK_DRAFTER_QUERY_WIDTH, dtype=torch.int32)
    runner._decode_task_args = [SimpleNamespace(tensors={"grammar_mask": masks, "valid_draft_counts": counts})]
    runner._decode_grammar_rows = [set(), set()]
    runner._compiled = SimpleNamespace(layout=SimpleNamespace(tp_size=4))
    runner._drafter_state = lambda request_id: SimpleNamespace(pending_draft_tokens=[11, 12, 13])
    inputs = SimpleNamespace(
        buffer_slot=0, request_ids=("r",), groups=(1,), group_ordinals=(5,), sampled_slots=((6, 40),)
    )

    runner._stage_decode_grammar(SimpleNamespace(constraint_states={"r": State()}), inputs)
    assert counts[4:8, 5].tolist() == [2, 2, 2, 2]
    assert int(masks[6, 40, 0, 0]) == -2
    assert int(masks[6, 42, 0, 0]) == -2
    assert int(masks[6, 43, 0, 0]) == -1
    assert int(masks[5, 40, 0, 0]) == -1

    runner._stage_decode_grammar(SimpleNamespace(constraint_states={}), inputs)
    # Ordinary batches select immutable defaults and do not rewrite the dirty slot.
    assert int(masks[6, 40, 0, 0]) == -2
    assert counts[4:8, 5].tolist() == [2, 2, 2, 2]
    next_inputs = SimpleNamespace(
        buffer_slot=0, request_ids=("r",), groups=(1,), group_ordinals=(5,), sampled_slots=((6, 48),)
    )
    runner._stage_decode_grammar(SimpleNamespace(constraint_states={"r": State()}), next_inputs)
    assert int(masks[6, 40, 0, 0]) == -1
    assert int(masks[6, 40, 0, -1]) == -1
    assert int(masks[6, 48, 0, -1]) == 0


def test_ordinary_dispatch_reuses_immutable_device_defaults() -> None:
    from pypto_serving.model.common.runner.buffer_set import StaticDeviceTensor, resolve_l3_arg

    runner = DSparkModelRunner.__new__(DSparkModelRunner)
    runner._default_grammar_args = {}
    runner._compiled = SimpleNamespace(layout=SimpleNamespace(ranks=1))
    runner._ensure_shared_host_allocation_before_worker = lambda _name: None
    runner._prepare_default_grammar_args()
    defaults = runner._default_grammar_args
    runner._prepare_default_grammar_args()
    assert runner._default_grammar_args is defaults

    names = ("logits", "grammar_mask", "valid_draft_counts", "sampled_ids")
    args = tuple(object() for _ in names)
    ordinary = runner._grammar_dispatch_args(args, names, constrained=False)
    assert ordinary[0] is args[0] and ordinary[3] is args[3]
    assert isinstance(ordinary[1], StaticDeviceTensor)
    assert ordinary[1].tensor.is_shared()
    assert torch.all(ordinary[1].tensor == -1)
    assert torch.all(ordinary[2].tensor == DSPARK_DRAFTER_QUERY_WIDTH)
    assert runner._grammar_dispatch_args(args, names, constrained=True) is args
    assert runner._grammar_dispatch_args(args, names, constrained=False)[1] is ordinary[1]

    uploads = []
    worker = SimpleNamespace(alloc_stacked_tensor=lambda tensor: uploads.append(tensor) or object())
    cache = {}
    first = resolve_l3_arg(worker, ordinary[1], cache)
    second = resolve_l3_arg(worker, ordinary[1], cache)
    assert first is second and len(uploads) == 1


def test_prefill_uses_the_same_default_mask_without_decode_count_argument() -> None:
    from pypto_serving.model.common.runner.buffer_set import StaticDeviceTensor

    runner = DSparkModelRunner.__new__(DSparkModelRunner)
    marker = StaticDeviceTensor(torch.full((1,), -1, dtype=torch.int16))
    runner._default_grammar_args = {"grammar_mask": marker, "valid_draft_counts": object()}
    output = object()
    assert runner._grammar_dispatch_args(
        (object(), output), ("grammar_mask", "sampled_ids"), constrained=False
    ) == (marker, output)


def test_decode_double_buffers_shrink_and_failure_clear_old_rows():
    class State:
        valid = 7
        fail = False

        def plan_draft_rows(self, drafts):
            rows = np.full((self.valid + 1, DSPARK_VOCAB_SIZE // 32), -2, dtype=np.int32)
            if self.fail:
                rows[-1].fill(0)
            return self.valid, rows

    state = State()
    runner = DSparkModelRunner.__new__(DSparkModelRunner)
    runner._compiled = SimpleNamespace(layout=SimpleNamespace(tp_size=1))
    runner._decode_task_args = [SimpleNamespace(tensors={
        "grammar_mask": torch.full((1, 16, DSPARK_GRAMMAR_SEGMENTS, DSPARK_GRAMMAR_SEGMENT_WORDS), -1, dtype=torch.int16),
        "valid_draft_counts": torch.full((1, 2), 7, dtype=torch.int32),
    }) for _ in range(2)]
    runner._decode_grammar_rows = [set(), set()]
    runner._drafter_state = lambda _: SimpleNamespace(pending_draft_tokens=list(range(7)))
    inputs = SimpleNamespace(buffer_slot=0, request_ids=("r",), groups=(0,), group_ordinals=(0,), sampled_slots=((0, 0),))
    batch = SimpleNamespace(constraint_states={"r": state})
    runner._stage_decode_grammar(batch, inputs)
    state.valid = 1
    inputs.buffer_slot = 1
    runner._stage_decode_grammar(batch, inputs)
    assert runner._decode_grammar_rows == [{(0, i) for i in range(8)}, {(0, 0), (0, 1)}]
    inputs.buffer_slot = 0
    runner._stage_decode_grammar(batch, inputs)
    assert torch.all(runner._decode_task_args[0].tensors["grammar_mask"][0, 2:] == -1)
    state.fail = True
    inputs.sampled_slots = ((0, 8),)
    with pytest.raises(ValueError, match="no allowed token"):
        runner._stage_decode_grammar(batch, inputs)
    assert torch.all(runner._decode_task_args[0].tensors["grammar_mask"] == -1)
    assert runner._decode_grammar_rows[0] == set()
    assert runner._decode_grammar_rows[1] == {(0, 0), (0, 1)}


def test_failed_decode_mask_staging_cannot_poison_reused_slot() -> None:
    class State:
        def plan_draft_rows(self, token_ids):
            rows = np.full((2, DSPARK_VOCAB_SIZE // 32), -1, dtype=np.int32)
            rows[0, 0] = -2
            rows[1].fill(0)
            return 1, rows

    runner = DSparkModelRunner.__new__(DSparkModelRunner)
    masks = torch.full(
        (DSPARK_RANKS, DSPARK_MAX_LOGIT_ROWS, DSPARK_GRAMMAR_SEGMENTS, DSPARK_GRAMMAR_SEGMENT_WORDS),
        -1, dtype=torch.int16,
    )
    counts = torch.full((DSPARK_RANKS, DSPARK_DECODE_BATCH), DSPARK_DRAFTER_QUERY_WIDTH, dtype=torch.int32)
    runner._decode_task_args = [SimpleNamespace(tensors={"grammar_mask": masks, "valid_draft_counts": counts})]
    runner._decode_grammar_rows = [set(), set()]
    runner._compiled = SimpleNamespace(layout=SimpleNamespace(tp_size=4))
    runner._drafter_state = lambda request_id: SimpleNamespace(pending_draft_tokens=[11])
    inputs = SimpleNamespace(
        buffer_slot=0, request_ids=("r",), groups=(1,), group_ordinals=(5,), sampled_slots=((6, 40),)
    )

    with pytest.raises(ValueError, match="no allowed token"):
        runner._stage_decode_grammar(SimpleNamespace(constraint_states={"r": State()}), inputs)
    assert runner._decode_grammar_rows[0] == set()
    assert int(masks[6, 40, 0, 0]) == -1
    assert counts[4:8, 5].tolist() == [DSPARK_DRAFTER_QUERY_WIDTH] * 4

    runner._stage_decode_grammar(SimpleNamespace(constraint_states={}), inputs)
    assert int(masks[6, 40, 0, 0]) == -1


def test_failed_prefill_mask_staging_cannot_poison_next_request() -> None:
    class State:
        def __init__(self, allowed):
            self.allowed = allowed

        def plan_draft_rows(self, token_ids):
            row = np.full((1, DSPARK_VOCAB_SIZE // 32), -1 if self.allowed else 0, dtype=np.int32)
            if self.allowed:
                row[0, 0] = -2
            return 0, row

    runner = DSparkModelRunner.__new__(DSparkModelRunner)
    masks = torch.full(
        (DSPARK_RANKS, DSPARK_MAX_LOGIT_ROWS, DSPARK_GRAMMAR_SEGMENTS, DSPARK_GRAMMAR_SEGMENT_WORDS),
        -1, dtype=torch.int16,
    )
    runner._prefill_task_arg_slots = [SimpleNamespace(tensors={"grammar_mask": masks})]
    runner._prefill_grammar_rows = [set(), set()]
    inputs = SimpleNamespace(request_ids=("good", "bad"), sampled_slots=((6, 40), (6, 41)))

    with pytest.raises(ValueError, match="no allowed token"):
        runner._stage_prefill_grammar(
            SimpleNamespace(constraint_states={"good": State(True), "bad": State(False)}), inputs
        )
    assert runner._prefill_grammar_rows[0] == set()
    assert int(masks[6, 40, 0, 0]) == -1

    runner._stage_prefill_grammar(SimpleNamespace(constraint_states={}), inputs)
    assert int(masks[6, 40, 0, 0]) == -1
