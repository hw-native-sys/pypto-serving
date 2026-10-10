# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Full-backbone cache layout checks before a worker opens device resources."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pypto_serving.config.types import KVCacheGroupSpec, KVCacheSpec
from pypto_serving.model.deepseek_v41.cache_contract import V41CacheGroups, validate_cache_groups
from pypto_serving.model.deepseek_v41.composite import CompositeBindings
from pypto_serving.model.deepseek_v41.execution_plan import RankPlacement, plan_layers
from pypto_serving.model.deepseek_v41.npu_runner import V41ModelRunner


@pytest.fixture
def contract():
    path = Path(__file__).resolve().parents[4] / "tests/fixtures/deepseek_v41/config.json"
    layers = plan_layers(json.loads(path.read_text(encoding="utf-8")))
    groups = (
        KVCacheGroupSpec("window", tuple(range(40)), KVCacheSpec(128, 16),
                         65, num_blocks=65, num_partitions=2),
        KVCacheGroupSpec("c2a", (2, 8, 14), KVCacheSpec(256, 16, 2),
                         33, num_blocks=33, num_partitions=2),
        KVCacheGroupSpec("c1a", (20,), KVCacheSpec(128, 16),
                         65, num_blocks=65, num_partitions=2),
    )
    return layers, groups


def test_full_backbone_resolves_three_independent_cache_families(contract):
    layers, groups = contract
    assert validate_cache_groups(layers, groups, 8320) == V41CacheGroups(
        window="window", c2a="c2a", c1a="c1a")


def test_cache_family_names_follow_producers_not_labels(contract):
    layers, groups = contract
    renamed = tuple(replace(group, name=f"pool_{index}") for index, group in enumerate(groups))
    assert validate_cache_groups(layers, renamed, 8320) == V41CacheGroups(
        window="pool_0", c2a="pool_1", c1a="pool_2")


@pytest.mark.parametrize("change,match", [
    (lambda groups: groups[:2], "separate"),
    (lambda groups: (groups[0], replace(groups[1], layer_indices=(2, 8, 14, 20)), groups[2]),
     "producer"),
    (lambda groups: (groups[0], groups[1], replace(groups[2], spec=KVCacheSpec(256, 16, 2))),
     "c1a.*layout"),
    (lambda groups: (replace(groups[0], max_blocks_per_seq=64), *groups[1:]),
     "window.*max_seq_len"),
    (lambda groups: (replace(groups[0], num_blocks=64), *groups[1:]),
     "window.*max_seq_len"),
    (lambda groups: (replace(groups[0], num_partitions=1), *groups[1:]),
     "window.*layout"),
    (lambda groups: (groups[0], replace(groups[1], max_blocks_per_seq=32), groups[2]),
     "c2a.*max_seq_len"),
])
def test_mismatched_cache_family_rejected(contract, change, match):
    layers, groups = contract
    with pytest.raises(ValueError, match=match):
        validate_cache_groups(layers, change(groups), 8320)


def test_runner_rejects_bad_layout_before_device_allocation(contract):
    layers, groups = contract
    calls = []
    bindings = CompositeBindings(
        revision="recording-only", entries={("prefill", layer.mode): lambda *a: None
                                            for layer in layers},
        initialize=lambda *a: None, output=lambda *a: None,
        allocate=lambda *a: calls.append("allocate"), prepare_weights=lambda *a: None,
        reset_request=lambda *a: None, wait=lambda *a: None, close=lambda *a: None,
        cache_groups=groups[:2], decode_backbone=lambda *a: None,
    )
    plan = SimpleNamespace(placement=RankPlacement(0), layers=layers)
    with pytest.raises(ValueError, match="separate"):
        V41ModelRunner(plan, bindings, device_ids=range(8),
                       runtime=SimpleNamespace(max_seq_len=8320))
    assert calls == []
