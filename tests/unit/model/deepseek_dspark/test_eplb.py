# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""EPLB layout, planning, weight placement and window ownership invariants."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pypto_serving.config.types import EplbConfig
from pypto_serving.model.deepseek_dspark.eplb.controller import EplbController
from pypto_serving.model.deepseek_dspark.eplb.plan import (
    EXPERT_TENSORS, RebalancePlan, layer_move_plan, validate_plan,
)
from pypto_serving.model.deepseek_dspark.eplb.policy import balance_layer, objective, plan_rebalance
from pypto_serving.model.deepseek_dspark.eplb.state import LoadSample, Placement
from pypto_serving.model.deepseek_dspark.eplb.transfer import ExpertBankLayout
from pypto_serving.model.deepseek_dspark.weight_spec import dspark_expert_parallel


def enabled(**kwargs):
    options = dict(enabled=True, window_size=2, step_interval=2, max_migration_bytes=100_000)
    return EplbConfig(**{**options, **kwargs})


@pytest.mark.parametrize("kwargs", [
    {"window_size": True}, {"step_interval": 0}, {"window_size": 3},
    {"num_redundant_experts": -1}, {"max_migration_bytes": 0},
    {"max_migration_bytes": 1.5}, {"min_relative_gain": float("nan")},
    {"min_relative_gain": float("inf")}, {"min_relative_gain": True},
    {"min_relative_gain": 1}, {"load_collection_phase": "both"},
    {"balance_drafter": True}, {"transfer_backend": "host_staged"}, {"policy": "unknown"},
])
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        enabled(**kwargs)


def test_role_and_topology_contract():
    config = enabled(num_redundant_experts=16)
    assert config.phase_for_role("mixed") == "decode"
    assert config.phase_for_role("prefill") == "prefill"
    assert config.validate_topology(logical_experts=256, ranks=16, tp_size=4, execution_role="mixed") == 17
    with pytest.raises(ValueError, match="P/D"):
        config.validate_topology(logical_experts=256, ranks=16, tp_size=4, execution_role="decode")
    with pytest.raises(ValueError, match="TP4"):
        config.validate_topology(logical_experts=256, ranks=16, tp_size=1, execution_role="mixed")
    with pytest.raises(ValueError, match="conflicts"):
        enabled(load_collection_phase="decode").phase_for_role("prefill")
    with pytest.raises(ValueError, match="capacity"):
        enabled(num_redundant_experts=1).validate_topology(
            logical_experts=256, ranks=16, tp_size=4, execution_role="mixed")


@pytest.mark.parametrize("ranks,extra", [(8, 0), (8, 8), (16, 0), (16, 16), (16, 256)])
def test_production_layout_covers_every_expert_and_is_immutable(ranks, extra):
    layout = Placement.initial(layers=43, logical_experts=256, ranks=ranks, redundant_experts=extra)
    assert layout.local_slots == (256 + extra) // ranks
    assert np.all(layout.replica_count >= 1)
    assert np.all(layout.replica_count.sum(axis=1) == 256 + extra)
    with pytest.raises(ValueError):
        layout.physical_to_logical.setflags(write=True)
    with pytest.raises(ValueError):
        layout.logical_to_physical[0, 0, 0] = 100


@pytest.mark.parametrize("mapping", [
    [[0, 0, 1, 2]], [[0, 1, 2, 2]], [[0, 1, 2, 4]], [[0.0, 1.0, 2.0, 3.0]],
])
def test_invalid_or_incomplete_layout_rejected(mapping):
    with pytest.raises(ValueError):
        Placement.from_map(mapping, logical_experts=4, ranks=2)


@pytest.mark.parametrize("corruption", ["count", "wrong_expert", "reversed", "tail"])
def test_inverse_tables_reject_corruption_across_layers(corruption):
    base = Placement.initial(layers=3, logical_experts=8, ranks=4, redundant_experts=4)
    inverse = base.logical_to_physical.copy()
    counts = base.replica_count.copy()
    if corruption == "count":
        counts[2, 0] += 1
    elif corruption == "wrong_expert":
        inverse[2, 0, 0] = inverse[2, 1, 0]
    elif corruption == "reversed":
        inverse[2, 0, :2] = inverse[2, 0, :2][::-1]
    else:
        inverse[2, 0, -1] = 0
    with pytest.raises(ValueError):
        replace(base, logical_to_physical=inverse, replica_count=counts)


def test_inverse_locations_keep_physical_order_after_shuffled_multi_layer_placement():
    initial = Placement.initial(layers=3, logical_experts=8, ranks=4, redundant_experts=4)
    mapping = initial.physical_to_logical.reshape(3, 4, 3).copy()
    mapping[1] = mapping[1, ::-1, ::-1]
    mapping[2] = np.roll(mapping[2], 1, axis=0)
    placement = Placement.from_map(mapping.reshape(3, -1), logical_experts=8, ranks=4)
    for layer in range(3):
        for expert in range(8):
            expected = [index for index, logical in enumerate(mapping[layer].reshape(-1)) if logical == expert]
            count = len(expected)
            assert placement.logical_to_physical[layer, expert, :count].tolist() == expected
            assert np.all(placement.logical_to_physical[layer, expert, count:] == -1)


def test_load_sample_uses_rank_major_order_and_owns_its_counts():
    layout = Placement.initial(layers=2, logical_experts=4, ranks=2, redundant_experts=2)
    counts = np.arange(12).reshape(2, 2, 3)
    sample = LoadSample(1, "decode", layout, counts)
    counts[:] = 0
    np.testing.assert_array_equal(sample.logical_counts(), [[7, 9, 2, 6], [13, 15, 5, 9]])
    assert sample.logical_counts().sum() == 66
    with pytest.raises(ValueError):
        LoadSample(2, "decode", layout, -np.ones((2, 2, 3), dtype=int))
    with pytest.raises(ValueError, match="overflow"):
        LoadSample(2, "decode", layout, np.full((2, 2, 3), np.iinfo(np.int64).max))


def test_cycle_and_replica_plan_preserves_all_six_weight_tensors():
    base = Placement.initial(layers=1, logical_experts=6, ranks=3, redundant_experts=3)
    # Change replica allocation and include a three-rank cycle.
    target = Placement.from_map([[3, 0, 4, 0, 1, 5, 0, 2, 1]], logical_experts=6, ranks=3, generation=1)
    move = layer_move_plan(base, target, 0, expert_bytes=96)
    plan = RebalancePlan(0, target, (move,), 0.1)
    validate_plan(base, plan, expert_bytes=96, max_layers=1, max_migration_bytes=10000)
    old_ids = base.physical_to_logical[0].reshape(3, 3)
    new_ids = target.physical_to_logical[0].reshape(3, 3)
    for index, name in enumerate(EXPERT_TENSORS):
        dtype = np.float32 if "scale" in name else np.int8
        bank = (old_ids[:, :, None] * 7 + np.arange(4) + index).astype(dtype)
        reference = (new_ids[:, :, None] * 7 + np.arange(4) + index).astype(dtype)
        # All source reads precede any resident writes; cycles must survive.
        received = bank[move.sources[:, :, 0], move.sources[:, :, 1]].copy()
        bank[move.changed] = received[move.changed]
        np.testing.assert_array_equal(bank, reference)
    bad_sources = move.sources.copy()
    bad_sources[0, 0] = (0, 0)
    bad = replace(plan, layers=(replace(move, sources=bad_sources),))
    with pytest.raises(ValueError, match="source does not contain"):
        validate_plan(base, bad, expert_bytes=96, max_layers=1, max_migration_bytes=10000)
    with pytest.raises(ValueError, match="budget"):
        validate_plan(base, plan, expert_bytes=96, max_layers=1, max_migration_bytes=1)


@pytest.mark.parametrize("ranks,experts,extra", [(2, 4, 0), (3, 6, 3), (4, 8, 4), (4, 8, 16)])
def test_policy_keeps_capacity_coverage_and_rank_uniqueness(ranks, experts, extra):
    rng = np.random.default_rng(431)
    base = Placement.initial(layers=1, logical_experts=experts, ranks=ranks, redundant_experts=extra)
    for _ in range(20):
        loads = rng.integers(0, 100, (7, experts)).astype(float)
        new = balance_layer(loads, base.physical_to_logical[0], ranks=ranks)
        result = Placement.from_map(new[None, :], logical_experts=experts, ranks=ranks)
        assert result.local_slots == base.local_slots
        # Equal-cost deterministic inputs must produce the same placement.
        np.testing.assert_array_equal(new, balance_layer(loads, base.physical_to_logical[0], ranks=ranks))


def test_hot_expert_replication_improves_load_and_budget_accepts_whole_layers():
    base = Placement.initial(layers=2, logical_experts=8, ranks=4, redundant_experts=4)
    loads = np.full((3, 2, 8), 10)
    loads[:, :, 0] = 100
    loads[:, :, 1] = 80
    config = enabled(num_redundant_experts=4, max_layers_per_rebalance=2)
    plan = plan_rebalance(base, loads, config, expert_bytes=100)
    assert plan is not None and len(plan.layers) == 2
    assert plan.target.replica_count[0, 0] > base.replica_count[0, 0]
    for layer in range(2):
        assert objective(loads[:, layer], plan.target.physical_to_logical[layer], 4) < objective(
            loads[:, layer], base.physical_to_logical[layer], 4)
    one_layer_bytes = plan.layers[0].payload_bytes
    limited = plan_rebalance(base, loads, replace(config, max_migration_bytes=one_layer_bytes), expert_bytes=100)
    assert limited is not None and len(limited.layers) == 1
    assert limited.layers[0].layer_id == 0
    assert plan_rebalance(base, loads, replace(config, max_migration_bytes=one_layer_bytes - 1),
                          expert_bytes=100) is None
    assert plan_rebalance(base, np.zeros_like(loads), config, expert_bytes=100) is None


def test_temporal_policy_separates_experts_with_correlated_bursts():
    base = Placement.from_map([[0, 2, 1, 3]], logical_experts=4, ranks=2)
    history = np.array([[[100, 0, 100, 0]], [[0, 100, 0, 100]]])
    temporal = plan_rebalance(base, history, enabled(), expert_bytes=100)
    cumulative = plan_rebalance(base, history, enabled(policy="cumulative_greedy"), expert_bytes=100)
    assert temporal is not None
    assert temporal.predicted_gain == 0.5
    # Identical cumulative demand conceals the bursts and gives no accepted change.
    assert cumulative is None


def test_layer_gain_bound_does_not_hide_a_smaller_plan_that_fits_the_budget():
    base = Placement.initial(layers=2, logical_experts=8, ranks=4)
    # The highest-gain layer needs five moves; the second needs only a swap.
    history = np.array([[[100, 100, 100, 100, 1, 1, 1, 1], [4, 7, 7, 2, 3, 5, 7, 2]]])
    plan = plan_rebalance(base, history, enabled(max_migration_bytes=300), expert_bytes=100)
    assert plan is not None
    assert [layer.layer_id for layer in plan.layers] == [1]
    assert plan.payload_bytes <= 300
    assert np.array_equal(plan.target.physical_to_logical[0], base.physical_to_logical[0])
    assert objective(history[:, 1], plan.target.physical_to_logical[1], 4) == 10


def test_controller_phase_once_only_and_generation_publication():
    base = Placement.initial(layers=1, logical_experts=8, ranks=4, redundant_experts=4)
    config = enabled(num_redundant_experts=4)
    controller = EplbController(base, config, expert_bytes=100)
    logical_load = np.array([100, 80, 10, 10, 10, 10, 10, 10])
    counts = logical_load[base.physical_to_logical[0]].reshape(4, 1, 3)
    assert not controller.record(LoadSample(0, "prefill", base, counts))
    assert controller.take_ready_plan() is None
    sample = LoadSample(1, "decode", base, counts)
    assert controller.record(sample)
    assert not controller.record(sample)
    assert controller.take_ready_plan() is None
    assert controller.record(LoadSample(2, "decode", base, counts))
    plan = controller.take_ready_plan()
    assert plan is not None
    assert controller.take_ready_plan() is None
    controller.validate_base(plan)
    controller.committed(plan)
    assert controller.current() is plan.target
    assert not controller.record(sample)
    with pytest.raises(ValueError, match="Stale"):
        controller.validate_base(plan)
    assert controller.take_ready_plan() is None
    controller.failed(RuntimeError("partial device write"))
    with pytest.raises(RuntimeError, match="fatal"):
        controller.current()


def test_weight_policy_packs_extra_slots_on_both_allocation_paths():
    mapping = (0, 1, 2, 3, 0, 1)
    policy = dspark_expert_parallel(2, 4, physical_to_logical=mapping)
    lookup = lambda expert: torch.full((2, 3), expert, dtype=torch.int8)
    direct = policy.apply("routed_w1", lookup, dtype=torch.int8, destination=None)
    destination = torch.empty_like(direct)
    assert policy.apply("routed_w1", lookup, dtype=torch.int8, destination=destination) is destination
    assert tuple(destination.shape) == (2, 3, 2, 3)
    assert torch.equal(direct, destination)
    for physical, logical in enumerate(mapping):
        assert torch.equal(destination[physical // 3, physical % 3], lookup(logical))
    with pytest.raises(ValueError, match="placement"):
        dspark_expert_parallel(2, 4, physical_to_logical=(0, 0, 1, 2, 3, 1))


def test_resident_weight_budget_includes_scales_and_rejects_bad_bank_geometry():
    placement = Placement.initial(layers=43, logical_experts=256, ranks=16, redundant_experts=16)
    tails = ((2048, 4096), (2048,), (2048, 4096), (2048,), (4096, 2048), (4096,))
    banks = {
        name: SimpleNamespace(full_shape=(16, 43 * 17, *tail), worker_ids=tuple(range(16)),
                              dtype=torch.float32 if name.endswith("_scale") else torch.int8)
        for name, tail in zip(EXPERT_TENSORS, tails)
    }
    layout = ExpertBankLayout.inspect(banks, placement)
    assert layout.expert_bytes == 25_198_592
    assert layout.receive_bytes_per_rank == 428_376_064
    sources, changed = layout.controls(None)
    assert sources.shape == (16, 16, 17, 2) and sources.dtype == torch.int32
    assert changed.shape == (16, 16, 17) and not changed.any()
    banks["routed_w2_scale"].full_shape = (16, 43 * 17, 2048)
    with pytest.raises(ValueError, match="geometry"):
        ExpertBankLayout.inspect(banks, placement)
