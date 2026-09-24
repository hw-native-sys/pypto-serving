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
    assert np.all(physical[:, DSPARK_GRAMMAR_SEGMENT_TOKENS:] == 1)


def test_k7_mask_rows_and_draft_caps_are_isolated() -> None:
    class State:
        def validate_draft_prefix(self, token_ids):
            assert token_ids == [11, 12, 13]
            return 2

        def masks_for_rows(self, token_ids):
            assert token_ids == [11, 12]
            masks = np.full((3, DSPARK_VOCAB_SIZE // 32), -1, dtype=np.int32)
            masks[:, 0] = -2
            return masks

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
    assert int(masks[6, 40, 0, 0]) == -1
    assert counts[4:8, 5].tolist() == [DSPARK_DRAFTER_QUERY_WIDTH] * 4


def test_failed_decode_mask_staging_cannot_poison_reused_slot() -> None:
    class State:
        def validate_draft_prefix(self, token_ids):
            return 1

        def masks_for_rows(self, token_ids):
            rows = np.full((2, DSPARK_VOCAB_SIZE // 32), -1, dtype=np.int32)
            rows[0, 0] = -2
            rows[1].fill(0)
            return rows

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

        def masks_for_rows(self, token_ids):
            row = np.full((1, DSPARK_VOCAB_SIZE // 32), -1 if self.allowed else 0, dtype=np.int32)
            if self.allowed:
                row[0, 0] = -2
            return row

    runner = DSparkModelRunner.__new__(DSparkModelRunner)
    masks = torch.full(
        (DSPARK_RANKS, DSPARK_MAX_LOGIT_ROWS, DSPARK_GRAMMAR_SEGMENTS, DSPARK_GRAMMAR_SEGMENT_WORDS),
        -1, dtype=torch.int16,
    )
    runner._prefill_task_args = SimpleNamespace(tensors={"grammar_mask": masks})
    runner._prefill_grammar_rows = set()
    inputs = SimpleNamespace(request_ids=("good", "bad"), sampled_slots=((6, 40), (6, 41)))

    with pytest.raises(ValueError, match="no allowed token"):
        runner._stage_prefill_grammar(
            SimpleNamespace(constraint_states={"good": State(True), "bad": State(False)}), inputs
        )
    assert runner._prefill_grammar_rows == set()
    assert int(masks[6, 40, 0, 0]) == -1

    runner._stage_prefill_grammar(SimpleNamespace(constraint_states={}), inputs)
    assert int(masks[6, 40, 0, 0]) == -1
