# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Temporal greedy replica allocation, rank placement and whole-layer budgeting."""

from __future__ import annotations

from collections import deque

import numpy as np

from pypto_serving.config.types import EplbConfig
from .plan import RebalancePlan, layer_move_plan, validate_plan
from .state import Placement


def objective(history: np.ndarray, mapping: np.ndarray, ranks: int) -> float:
    """Sum per-dispatch maximum rank load, assuming equal division among replicas."""
    counts = np.bincount(mapping, minlength=history.shape[1])
    per_copy = history / counts[None, :]
    rank_load = per_copy[:, mapping].reshape(len(history), ranks, -1).sum(axis=2)
    return float(rank_load.max(axis=1).sum())


def _replicas(history: np.ndarray, physical: int, ranks: int) -> np.ndarray:
    experts = history.shape[1]
    counts = np.ones(experts, dtype=np.int64)
    for _ in range(physical - experts):
        per_copy = history / counts[None, :]
        if experts == 1:
            other = np.zeros_like(per_copy)
        else:
            top = np.partition(per_copy, -2, axis=1)[:, -2:]
            other = np.where(per_copy == top[:, -1:], top[:, -2:-1], top[:, -1:])
        costs = np.maximum(other, history / (counts[None, :] + 1)).sum(axis=0)
        costs[counts == ranks] = np.inf
        counts[int(np.argmin(costs))] += 1
    return counts


def _repair(assigned: list[set[int]], expert: int, slots: int) -> None:
    """Find an augmenting path in the expert/rank b-matching, preserving uniqueness."""
    queue = deque([expert])
    seen_experts = {expert}
    # Each visited rank was reached from an expert it does not yet contain.
    rank_parent: dict[int, int] = {}
    expert_parent: dict[int, int] = {}
    while queue:
        current = queue.popleft()
        for rank, contents in enumerate(assigned):
            if rank in rank_parent or current in contents:
                continue
            rank_parent[rank] = current
            if len(contents) < slots:
                while True:
                    incoming = rank_parent[rank]
                    assigned[rank].add(incoming)
                    if incoming == expert:
                        return
                    previous = expert_parent[incoming]
                    assigned[previous].remove(incoming)
                    rank = previous
            for displaced in sorted(contents):
                if displaced not in seen_experts:
                    seen_experts.add(displaced)
                    expert_parent[displaced] = rank
                    queue.append(displaced)
    raise ValueError("No feasible expert placement within the fixed capacity")


def balance_layer(history: np.ndarray, old: np.ndarray, *, ranks: int) -> np.ndarray:
    """Allocate replicas, place copies and retain unchanged rank-local slots."""
    slots = len(old) // ranks
    counts = _replicas(history, len(old), ranks)
    per_copy = history / counts[None, :]
    copies = sorted((expert for expert, count in enumerate(counts) for _ in range(count)),
                    key=lambda expert: (-float(per_copy[:, expert].sum()), expert))
    assigned: list[set[int]] = [set() for _ in range(ranks)]
    occupied = np.zeros(ranks, dtype=np.int64)
    present = np.zeros((ranks, history.shape[1]), dtype=np.bool_)
    local = np.zeros_like(present)
    local[np.arange(ranks)[:, None], old.reshape(ranks, slots)] = True
    rank_load = np.zeros((len(history), ranks), dtype=np.float64)
    for expert in copies:
        eligible = (occupied < slots) & ~present[:, expert]
        if eligible.any():
            # Nonnegative additions leave every other rank unchanged. Evaluate
            # every destination together without copying the whole load matrix.
            maximum = rank_load.max(axis=1)
            trial = rank_load.T + per_copy[:, expert]
            costs = np.maximum(trial, maximum).sum(axis=1)
            costs[~eligible] = np.inf
            best = costs.min()
            tied = eligible & (costs == best)
            retained = tied & local[:, expert]
            rank = int(np.flatnonzero(retained if retained.any() else tied)[0])
            assigned[rank].add(expert)
            present[rank, expert] = True
            occupied[rank] += 1
            rank_load[:, rank] += per_copy[:, expert]
        else:
            _repair(assigned, expert, slots)
            present.fill(False)
            for rank in range(ranks):
                contents = sorted(assigned[rank])
                present[rank, contents] = True
                occupied[rank] = len(contents)
                rank_load[:, rank] = per_copy[:, contents].sum(axis=1)
    result = np.full(len(old), -1, dtype=np.int64)
    for rank, contents in enumerate(assigned):
        retained = set()
        for slot in range(slots):
            index = rank * slots + slot
            if int(old[index]) in contents:
                result[index] = old[index]
                retained.add(int(old[index]))
        remaining = iter(sorted(contents - retained))
        for slot in range(slots):
            index = rank * slots + slot
            if result[index] == -1:
                result[index] = next(remaining)
    return result


def plan_rebalance(base: Placement, history: np.ndarray, config: EplbConfig,
                   *, expert_bytes: int) -> RebalancePlan | None:
    """Budget complete layer candidates; never truncate a cycle into individual moves."""
    history = np.asarray(history, dtype=np.float64)
    if (history.ndim != 3 or history.shape[1:] != base.replica_count.shape or not len(history)
            or not np.isfinite(history).all() or np.any(history < 0)):
        raise ValueError("Invalid logical load history")
    if not history.any():
        return None
    scoring = history if config.policy == "temporal_greedy" else history.sum(axis=0, keepdims=True)
    candidates = []
    before_costs = [objective(scoring[:, layer], base.physical_to_logical[layer], base.ranks)
                    for layer in range(base.layers)]
    old_cost = sum(before_costs)
    # No placement can make the maximum rank load smaller than the mean.
    # Bound each layer's possible gain before running the placement search.
    total_loads = scoring.sum(axis=2).sum(axis=0)
    slack = np.finfo(np.float64).eps * total_loads * (base.logical_experts + base.ranks + 4)
    upper_gains = np.maximum(np.asarray(before_costs) - total_loads / base.ranks, 0) + slack
    order = sorted(range(base.layers), key=lambda layer: (-upper_gains[layer], layer))
    for layer in order:
        remaining_bytes = config.max_migration_bytes
        selected = []
        for candidate in sorted(candidates, key=lambda item: (-item[0], item[1])):
            payload = int(np.count_nonzero(base.physical_to_logical[candidate[1]] != candidate[2])) * expert_bytes
            if payload <= remaining_bytes:
                selected.append(candidate)
                remaining_bytes -= payload
                if len(selected) == config.max_layers_per_rebalance:
                    break
        if len(selected) == config.max_layers_per_rebalance and upper_gains[layer] < selected[-1][0]:
            break
        old = base.physical_to_logical[layer]
        loads = scoring[:, layer]
        if not loads.any():
            continue
        new = balance_layer(loads, old, ranks=base.ranks)
        before = before_costs[layer]
        gain = before - objective(loads, new, base.ranks)
        if gain > 0 and gain / before >= config.min_relative_gain:
            candidates.append((gain, layer, new))
    mapping = base.physical_to_logical.copy()
    moves = []
    accepted_gain = 0.0
    used_bytes = 0
    for gain, layer, new in sorted(candidates, key=lambda candidate: (-candidate[0], candidate[1])):
        if len(moves) >= config.max_layers_per_rebalance:
            break
        trial_map = mapping.copy()
        trial_map[layer] = new
        trial = Placement.from_map(trial_map, logical_experts=base.logical_experts,
                                   ranks=base.ranks, generation=base.generation + 1)
        move = layer_move_plan(base, trial, layer, expert_bytes=expert_bytes)
        if used_bytes + move.payload_bytes > config.max_migration_bytes:
            continue
        mapping = trial_map
        moves.append(move)
        used_bytes += move.payload_bytes
        accepted_gain += gain
    if not moves:
        return None
    target = Placement.from_map(mapping, logical_experts=base.logical_experts,
                                ranks=base.ranks, generation=base.generation + 1)
    plan = RebalancePlan(base.generation, target, tuple(moves), accepted_gain / old_cost)
    validate_plan(base, plan, expert_bytes=expert_bytes, max_layers=config.max_layers_per_rebalance,
                  max_migration_bytes=config.max_migration_bytes)
    return plan
