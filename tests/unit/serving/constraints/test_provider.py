# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Speculative mask planning is single-pass, borrowed, and rollback-safe."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pypto_serving.serving.constraints.provider import XGrammarProvider, XGrammarState


def test_provider_uses_model_vocab_width_without_weakening_tokenizer_validation(monkeypatch):
    seen = []
    fake_xgrammar = SimpleNamespace(
        TokenizerInfo=SimpleNamespace(from_huggingface=lambda _, vocab_size: seen.append(vocab_size)),
        GrammarCompiler=lambda *args, **kwargs: object(),
    )
    monkeypatch.setitem(sys.modules, "xgrammar", fake_xgrammar)
    tokenizer = SimpleNamespace(tokenizer=object(), get_vocab=lambda: {"a": 0, "b": 1})
    provider = XGrammarProvider(tokenizer, 64)
    assert provider.vocab_size == 64
    assert seen == [64]
    with pytest.raises(ValueError, match="smaller than the tokenizer"):
        XGrammarProvider(tokenizer, 1)
    tokenizer.get_vocab = lambda: {"a": 0, "b": 2}
    with pytest.raises(ValueError, match="contiguous"):
        XGrammarProvider(tokenizer, 64)


class Matcher:
    def __init__(self, *, terminated_at=None, fail_at=None):
        self.history = []
        self.rollbacks = []
        self.accept_calls = []
        self.terminated_at = terminated_at
        self.fail_at = fail_at

    def is_terminated(self):
        return len(self.history) == self.terminated_at

    def accept_token(self, token):
        self.accept_calls.append(token)
        if token != 10 + len(self.history):
            return False
        self.history.append(token)
        return True

    def rollback(self, count):
        self.rollbacks.append(count)
        del self.history[-count:]

    def fill_next_token_bitmask(self, masks, row):
        if len(self.history) == self.fail_at:
            raise ValueError("injected fill failure")
        masks[row].zero_()
        masks[row, 0] = 1 << len(self.history)


def state_for(matcher):
    allocations = []

    def allocate(rows, vocab):
        allocations.append((rows, vocab))
        return torch.full((rows, (vocab + 31) // 32), -1, dtype=torch.int32)

    return XGrammarState(matcher, SimpleNamespace(allocate_token_bitmask=allocate), 64), allocations


@pytest.mark.parametrize("length", range(8))
def test_valid_prefix_is_traversed_once_and_keeps_bonus_row(length):
    matcher = Matcher()
    state, allocations = state_for(matcher)
    valid, masks = state.plan_draft_rows(list(range(10, 10 + length)))
    assert valid == length
    assert masks.shape == (length + 1, 2)
    assert masks[:, 0].tolist() == [1 << i for i in range(length + 1)]
    assert matcher.history == []
    assert matcher.accept_calls == list(range(10, 10 + length))
    assert matcher.rollbacks == ([length] if length else [])
    assert allocations == [(8, 64)]


@pytest.mark.parametrize("valid_prefix", range(7))
def test_first_invalid_draft_keeps_its_sampling_row(valid_prefix):
    matcher = Matcher()
    state, _ = state_for(matcher)
    drafts = list(range(10, 10 + valid_prefix)) + [999] * (7 - valid_prefix)
    valid, masks = state.plan_draft_rows(drafts)
    assert valid == valid_prefix
    assert masks.shape == (valid + 1, 2)
    assert int(masks[-1, 0]) == 1 << valid
    assert matcher.history == []
    assert len(matcher.accept_calls) == valid + 1


def test_plan_reuses_storage_and_documents_borrowed_lifetime():
    matcher = Matcher()
    state, allocations = state_for(matcher)
    _, first = state.plan_draft_rows([10, 11, 12])
    retained = first.copy()
    state.accept([10])
    _, second = state.plan_draft_rows([])
    assert np.shares_memory(first, second)
    assert first[0, 0] == 2 and retained[0, 0] == 1
    assert matcher.history == [10]
    assert allocations == [(8, 64)]
    state.close()
    assert state._matcher is state._bitmask is state._mask_array is None


@pytest.mark.parametrize("fail_at", range(8))
def test_fill_failure_rolls_back_exactly_the_advanced_prefix(fail_at):
    matcher = Matcher(fail_at=fail_at)
    state, _ = state_for(matcher)
    with pytest.raises(ValueError, match="injected fill"):
        state.plan_draft_rows(list(range(10, 17)))
    assert matcher.history == []
    assert matcher.rollbacks == ([fail_at] if fail_at else [])
    matcher.fail_at = None
    valid, rows = state.plan_draft_rows([10])
    assert valid == 1 and rows[:, 0].tolist() == [1, 2]


@pytest.mark.parametrize("terminated_at", [0, 3, 7])
def test_termination_preserves_all_allowed_tail_semantics(terminated_at):
    matcher = Matcher(terminated_at=terminated_at)
    state, _ = state_for(matcher)
    valid, rows = state.plan_draft_rows(list(range(10, 17)))
    assert valid == 7
    assert np.all(rows[terminated_at:] == -1)
    assert matcher.history == []
    assert len(matcher.accept_calls) == terminated_at


def test_states_have_independent_storage_and_reject_excessive_drafts():
    first, _ = state_for(Matcher())
    second, _ = state_for(Matcher())
    assert not np.shares_memory(first.plan_draft_rows([])[1], second.plan_draft_rows([])[1])
    with pytest.raises(ValueError, match="rollback capacity"):
        first.plan_draft_rows(list(range(10, 18)))
    with pytest.raises(ValueError, match="violates its grammar"):
        first.accept([999])
