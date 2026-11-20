# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from itertools import combinations
from random import Random

import pytest

from pypto_serving.serving.agentinfer_core.scheduling import (
    ProgramRef, ProgramState, ProgramStatus, ProgramTokenObservation, ProgramView,
)
from pypto_serving.serving.agentinfer_core.scheduling.progress_ttl import (
    ProgressTTLConfig, build_progress_ttl_strategy,
)
from pypto_serving.serving.agentinfer_core.scheduling.progress_ttl import strategy as strategy_module


def _program(name: str, tokens: int) -> ProgramView:
    return ProgramView(
        ref=ProgramRef(name, 0), state=ProgramState.ACTIVE, status=ProgramStatus.ACTING,
        tokens=ProgramTokenObservation(estimated_context_tokens=tokens), backend_id="backend",
    )


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("extra_relief", [0, 5])
def test_small_capacity_fits_preserve_exact_selection(seed, extra_relief):
    strategy = build_progress_ttl_strategy(ProgressTTLConfig(decode_buffer_tokens=0)).strategy
    rng = Random(seed)
    programs = [_program(f"program-{i}", rng.randrange(30)) for i in range(8)]
    deficit = rng.randrange(1, 150)
    covers = [
        (sum(item.tokens.estimated_context_tokens + extra_relief for item in selected),
         len(selected), tuple(item.ref for item in selected), selected)
        for size in range(1, len(programs) + 1)
        for selected in combinations(programs, size)
        if sum(item.tokens.estimated_context_tokens + extra_relief for item in selected) >= deficit
    ]
    expected = min(covers, key=lambda item: item[:3])[3] if covers else None

    assert strategy._select_capacity_fit(
        list(reversed(programs)), deficit, per_program_relief_tokens=extra_relief,
    ) == expected


@pytest.mark.parametrize("extra_relief", [0, 17])
def test_large_capacity_fit_fallback_covers_deficit_deterministically(monkeypatch, extra_relief):
    monkeypatch.setattr(strategy_module, "_CAPACITY_FIT_SEARCH_STEPS", 0)
    strategy = build_progress_ttl_strategy(ProgressTTLConfig(decode_buffer_tokens=0)).strategy
    rng = Random(0)
    programs = [_program(f"program-{i:03}", rng.randrange(1000, 5000)) for i in range(128)]
    programs.append(_program("zero-relief", 0))
    deficit = sum(item.tokens.estimated_context_tokens + extra_relief for item in programs) // 2

    selected = strategy._select_capacity_fit(programs, deficit, per_program_relief_tokens=extra_relief)

    assert selected
    assert len({item.ref for item in selected}) == len(selected)
    assert set(selected) <= set(programs)
    total = sum(item.tokens.estimated_context_tokens + extra_relief for item in selected)
    assert total >= deficit
    assert all(total - item.tokens.estimated_context_tokens - extra_relief < deficit for item in selected)
    assert selected == strategy._select_capacity_fit(
        list(reversed(programs)), deficit, per_program_relief_tokens=extra_relief,
    )


def test_capacity_fit_keeps_better_exact_cover_when_budget_runs_out(monkeypatch):
    monkeypatch.setattr(strategy_module, "_CAPACITY_FIT_SEARCH_STEPS", 1)
    strategy = build_progress_ttl_strategy(ProgressTTLConfig(decode_buffer_tokens=0)).strategy
    programs = [_program("a", 10), _program("b", 6), _program("c", 4)]

    assert strategy._select_capacity_fit(programs, 10) == (programs[0],)


@pytest.mark.parametrize("budget", [0, 4096])
@pytest.mark.parametrize("deficit, expected", [(0, ()), (-1, ()), (10, None)])
def test_capacity_fit_empty_and_insufficient_relief(monkeypatch, budget, deficit, expected):
    monkeypatch.setattr(strategy_module, "_CAPACITY_FIT_SEARCH_STEPS", budget)
    strategy = build_progress_ttl_strategy(ProgressTTLConfig(decode_buffer_tokens=0)).strategy

    assert strategy._select_capacity_fit([], deficit) == expected
    assert strategy._select_capacity_fit([_program("zero", 0), _program("small", 3)], deficit) == expected


def test_capacity_fit_fallback_preserves_pause_priority_layers(monkeypatch):
    monkeypatch.setattr(strategy_module, "_CAPACITY_FIT_SEARCH_STEPS", 0)
    strategy = build_progress_ttl_strategy(ProgressTTLConfig(decode_buffer_tokens=0)).strategy
    older = _program("older", 3)
    younger = [_program("large", 10), _program("exact", 7), _program("small", 3)]

    assert strategy._select_layered_capacity_fit(([older], younger), 10) == (older, younger[1])
