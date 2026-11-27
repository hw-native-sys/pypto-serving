# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""EPLB CLI contracts are validated before loading weights or creating workers."""

import json

import pytest

from pypto_serving.cli.main import _resolve_eplb_config, build_parser


def arguments(*options):
    return build_parser().parse_args([
        "--model", "unused", "--speculative-config", '{"method":"dspark","num_speculative_tokens":0}',
        "--tp", "4", "--dp", "4", "--ep", "16",
        *options,
    ])


def test_disabled_eplb_keeps_executor_configuration_unchanged():
    assert _resolve_eplb_config(arguments(), "deepseek_v4", {}) is None
    with pytest.raises(ValueError, match="requires --enable-eplb"):
        _resolve_eplb_config(arguments("--eplb-config", "{}"), "deepseek_v4", {})


def test_enable_eplb_freezes_the_tp4_dp4_physical_capacity():
    args = arguments("--enable-eplb", "--eplb-config", json.dumps({
        "num_redundant_experts": 16, "max_migration_bytes": 1 << 30,
    }))
    config = _resolve_eplb_config(args, "deepseek_v4", {})
    assert config.enabled
    assert config.num_redundant_experts == 16
    assert config.phase_for_role("mixed") == "decode"


@pytest.mark.parametrize("value", ["[]", "null", "invalid", '{"typo": 1}', '{"enabled": true}',
                                  '{"max_migration_bytes": false}'])
def test_eplb_rejects_invalid_json_and_unknown_fields(value):
    with pytest.raises(ValueError):
        _resolve_eplb_config(arguments("--enable-eplb", "--eplb-config", value), "deepseek_v4", {})


@pytest.mark.parametrize("family,flags", [("qwen", []), ("deepseek_v41", []),
                                         ("deepseek_v4", ["--tp", "1"])])
def test_eplb_rejects_unsupported_model_role_or_topology(family, flags):
    args = arguments("--enable-eplb", "--eplb-config", '{"max_migration_bytes": 1024}', *flags)
    with pytest.raises(ValueError):
        _resolve_eplb_config(args, family, {})


@pytest.mark.parametrize("role,phase", [("prefill", "prefill"), ("decode", "decode")])
def test_pd_eplb_resolves_role_and_rejects_a_conflicting_phase(role, phase):
    args = arguments("--enable-eplb", "--pd-role", role, "--eplb-config",
                     '{"max_migration_bytes":1024}')
    config = _resolve_eplb_config(args, "deepseek_v4", {})
    assert config.load_collection_phase is None
    assert config.phase_for_role(role) == phase
    args.eplb_config = json.dumps({"max_migration_bytes": 1024,
                                  "load_collection_phase": "decode" if role == "prefill" else "prefill"})
    with pytest.raises(ValueError, match="conflicts"):
        _resolve_eplb_config(args, "deepseek_v4", {})


def test_pd_decode_tp1_preserves_independent_head_and_expert_capacity():
    args = arguments("--enable-eplb", "--pd-role", "decode", "--tp", "1", "--dp", "16",
                     "--lm-head-tp", "4", "--eplb-config",
                     '{"num_redundant_experts":16,"max_migration_bytes":1024}')
    config = _resolve_eplb_config(args, "deepseek_v4", {})
    assert config.validate_topology(logical_experts=256, ranks=16, tp_size=1, execution_role="decode") == 17
    args.pd_role = "prefill"
    with pytest.raises(ValueError, match="TP4"):
        _resolve_eplb_config(args, "deepseek_v4", {})


def test_eplb_accepts_workspace_reserve_and_rate_ceiling():
    args = arguments("--enable-eplb", "--eplb-config", json.dumps({
        "max_migration_bytes": 1 << 30,
        "maintenance_workspace_bytes": 4 << 30,
        "max_maintenances_per_minute": 6,
    }))
    config = _resolve_eplb_config(args, "deepseek_v4", {})
    assert config.maintenance_workspace_bytes == 4 << 30
    assert config.max_maintenances_per_minute == 6
