# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Reference diagnostics must reject local or accumulated numerical failures."""
from types import SimpleNamespace

import pytest
import torch

from tools.validate_v41_swa_segment import compare_saved
from tools.diagnose_v41_swa_precision import quantization_metrics, trace_attention


def test_attention_trace_uses_selected_rank_weights_and_cache(monkeypatch):
    import sys
    from types import ModuleType

    calls = []
    linear, rope = object(), object()
    reference = SimpleNamespace(official_linear=linear, official_rope=rope,
                                official_reference=lambda inputs: calls.append(inputs))
    package = ModuleType("models.deepseek_v4_1_flash")
    package.decode_attn_swa = reference
    monkeypatch.setitem(sys.modules, "models", ModuleType("models"))
    monkeypatch.setitem(sys.modules, "models.deepseek_v4_1_flash", package)
    tensors = {name: torch.arange(4).reshape(4, 1) for name in ("weight", "cache")}
    actual, expected = torch.ones(1), torch.zeros(1)
    assert trace_attention(SimpleNamespace(HC_INPUT_NAMES=("weight", "cache", "x_hc")),
                           tensors, actual, expected, rank=2) == [{}, {}]
    assert [int(c["weight"]) for c in calls] == [2, 2]
    assert [int(c["cache"]) for c in calls] == [2, 2]
    assert calls[0]["x"] is actual and calls[1]["x"] is expected
    assert reference.official_linear is linear and reference.official_rope is rope


@pytest.mark.parametrize("failure", [None, "residual", "pre_mix", "stage"])
def test_saved_comparison_keeps_all_acceptance_gates(failure):
    residual = torch.ones(1, 2, 4, 3)
    mix = torch.ones(1, 2, 4)
    data = {"actual_residual": residual.clone(), "expected_residual": residual,
            "actual_pre_mix": mix.clone(), "expected_pre_mix": mix}
    if failure in ("residual", "pre_mix"):
        data["actual_" + failure].add_(1)
    if failure == "stage":
        data["stages"] = [{"results": {"output": (False, "stage mismatch")}}]

    def comparator(actual, expected, *, actual_outputs, expected_outputs, inputs, rtol, atol):
        assert inputs["num_tokens"].tolist() == [2]
        assert actual_outputs["x_next"] is actual
        assert expected_outputs["x_next"] is expected
        return torch.equal(actual, expected), "residual mismatch"

    moe = SimpleNamespace(_local_mhc_compare=lambda counts: comparator)
    topology = SimpleNamespace(local_capacity=2, world=1)
    if failure is None:
        compare_saved(data, moe, topology, "v41-local")
    else:
        with pytest.raises(AssertionError):
            compare_saved(data, moe, topology, "v41-local")


@pytest.mark.parametrize("failure", [None, "rank", "pre_mix", "stage"])
def test_dsv4_profile_keeps_per_rank_and_auxiliary_gates(monkeypatch, failure):
    import sys

    calls = []
    def factory(**settings):
        assert settings == {"diff_thd": 0.01, "pct_thd": 0.05}
        def comparator(actual, expected, **kwargs):
            calls.append(actual.shape)
            return torch.equal(actual, expected), "rank mismatch"
        return comparator
    monkeypatch.setitem(sys.modules, "golden.validation", SimpleNamespace(ratio_reldiff=factory))
    actual = torch.ones(2, 2, 4, 3)
    data = {"actual_residual": actual, "expected_residual": actual.clone(),
            "actual_pre_mix": torch.ones(2, 2, 4), "expected_pre_mix": torch.ones(2, 2, 4)}
    if failure == "rank":
        actual[1].add_(1)
    elif failure == "pre_mix":
        data["actual_pre_mix"].add_(1)
    elif failure == "stage":
        data["stages"] = [{"results": {"output": (False, "stage mismatch")}}]
    topology = SimpleNamespace(local_capacity=2, world=2)
    if failure is None:
        compare_saved(data, None, topology)
    else:
        with pytest.raises(AssertionError):
            compare_saved(data, None, topology)
    assert calls == [(2, 4, 3), (2, 4, 3)]


def test_quantization_probe_compares_values_instead_of_payload_codes():
    payload = torch.ones(2, 64)
    codes = torch.full((2, 2), 127, dtype=torch.uint8)
    # Different payload/exponent pairs can represent exactly the same values.
    same = quantization_metrics(payload, codes, payload / 2, codes + 1)
    assert same["dequantized"]["rel_l2"] == 0
    assert same["payload_changed"] == 128 and same["scale_changed"] == 4
    # Identical payloads are not equal physical values when exponents differ.
    different = quantization_metrics(payload, codes + 1, payload, codes)
    assert different["payload_changed"] == 0
    assert different["dequantized"]["rel_l2"] == 1


@pytest.mark.parametrize("error,passes", [(0.0005, True), (0.002, False)])
def test_provisional_premix_budget_requires_every_element(error, passes):
    residual = torch.ones(1, 2, 4, 3)
    expected = torch.zeros(1, 2, 4)
    actual = expected.clone()
    actual[0, 0, 0] = error
    data = {"actual_residual": residual, "expected_residual": residual.clone(),
            "actual_pre_mix": actual, "expected_pre_mix": expected}
    moe = SimpleNamespace(_local_mhc_compare=lambda counts: lambda *args, **kwargs: (True, ""))
    topology = SimpleNamespace(local_capacity=2, world=1)
    if passes:
        compare_saved(data, moe, topology, "v41-local")
    else:
        with pytest.raises(AssertionError):
            compare_saved(data, moe, topology, "v41-local")
