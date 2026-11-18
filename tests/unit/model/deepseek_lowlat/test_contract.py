# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Real scheduler, allocator, IPC and page-copy ownership regressions."""

import json

import pytest

from pypto_serving.cli.main import build_parser, build_serving_engine_config
from pypto_serving.config.types import GenerateConfig
from pypto_serving.model.deepseek_lowlat.contract import LOWLAT_EXECUTOR, validate_sampling


@pytest.fixture
def cli_args(tmp_path):
    raw = {
        "model_type": "deepseek_v4", "hidden_size": 4096, "num_hidden_layers": 43,
        "num_attention_heads": 64, "vocab_size": 129280, "n_routed_experts": 256,
        "moe_intermediate_size": 2048,
        "quantization_config": {"quant_method": "compressed-tensors"},
        "compress_ratios": [0, 0] + [r for _ in range(20) for r in (4, 128)] + [4],
    }
    (tmp_path / "config.json").write_text(json.dumps(raw))
    return build_parser().parse_args([
        "--model", str(tmp_path), "--devices", "7,6,5,4,3,2,1,0",
        "--dp", "1", "--tp", "8", "--ep", "1", "--lm-head-tp", "8",
        "--max-num-seqs", "1", "--block-size", "128", "--max-model-len", "512",
        "--max-num-batched-tokens", "128", "--long-prefill-token-threshold", "128",
        "--no-enable-prefix-caching",
        "--speculative-config", '{"method":"dspark","num_speculative_tokens":5}',
    ])


def test_dspark_k5_routes_to_stable_lowlat_contract(cli_args):
    config = build_serving_engine_config(cli_args)
    assert config.executor_cls == LOWLAT_EXECUTOR
    assert config.worker_device_ids() == (7, 6, 5, 4, 3, 2, 1, 0)
    assert config.parallel_config.placement_mode == "replica"
    assert config.resolve_async_scheduling() is True
    assert config.executor_kwargs["num_speculative_tokens"] == 5
    assert config.executor_kwargs["cache_initialization"] == "on-demand"
    assert "native_pipeline" not in config.executor_kwargs
    assert "native_launch_depth" not in config.executor_kwargs
    assert config.runtime_config.requires_homogeneous_prefill_decode
    assert any(group.sliding_window is not None for group in config.runtime_config.kv_cache_groups)


def test_dspark_k0_keeps_regular_dspark_executor(cli_args):
    cli_args.speculative_config = {"method": "dspark", "num_speculative_tokens": 0}
    cli_args.devices = ",".join(str(index) for index in range(16))
    cli_args.data_parallel_size = 4
    cli_args.tensor_parallel_size = 4
    cli_args.expert_parallel_size = 16
    cli_args.lm_head_tp = 4
    cli_args.block_size = 32
    config = build_serving_engine_config(cli_args)
    assert config.executor_cls == "PyptoDeepSeekV4DSparkExecutor"
    assert config.runtime_config.num_speculative_tokens == 0


@pytest.mark.parametrize(
    ("name", "value"),
    [("expert_parallel_size", 8), ("tensor_parallel_size", 4),
     ("data_parallel_size", 2), ("lm_head_tp", 4), ("max_num_seqs", 2),
     ("block_size", 32),
     ("long_prefill_token_threshold", 8193), ("pd_role", "decode")],
)
def test_invalid_topologies_fail_before_device_work(cli_args, name, value):
    setattr(cli_args, name, value)
    with pytest.raises(ValueError):
        build_serving_engine_config(cli_args)


@pytest.mark.parametrize("value", [True, False, 0.0, 2, 6, -1])
def test_rejects_invalid_dspark_speculation(cli_args, value):
    cli_args.speculative_config["num_speculative_tokens"] = value
    with pytest.raises(ValueError, match="num_speculative_tokens"):
        build_serving_engine_config(cli_args)


def test_lowlat_prefix_is_fixed_to_rolling_and_on_demand(cli_args):
    cli_args.enable_prefix_caching = True
    config = build_serving_engine_config(cli_args)
    assert config.enable_prefix_cache is True
    assert config.executor_kwargs["prefix_cache_namespace"]
    assert config.runtime_config.speculative_prefix_cache_replay_tokens == 128
    assert config.runtime_config.kv_cache_groups[0].num_blocks > config.runtime_config.kv_cache_groups[0].max_blocks_per_seq


def test_lowlat_public_parser_has_no_private_options():
    options = {action.option_strings[0] for action in build_parser()._actions if action.option_strings}
    assert "--speculative-config" in options
    assert "--lowlat-native-pipeline" not in options
    assert "--lowlat-native-launch-depth" not in options
    assert "--lowlat-cache-layout" not in options
    assert "--lowlat-cache-initialization" not in options
    assert "--num-speculative-tokens" not in options


@pytest.mark.parametrize(
    "options", [dict(temperature=0.1), dict(min_tokens=1), dict(top_p=0.9), dict(top_k=3)]
)
def test_unsupported_sampling_is_rejected(options):
    with pytest.raises(ValueError, match="greedy"):
        validate_sampling(GenerateConfig(**options))


def test_min_tokens_is_rejected_before_downgrade(cli_args):
    config = build_serving_engine_config(cli_args)
    with pytest.raises(ValueError, match="min_tokens"):
        config.resolve_min_tokens(1)
    with pytest.raises(ValueError, match="structured"):
        config.validate_generate_config(GenerateConfig(), constraint_spec=object())


def test_million_context_cli_uses_rolling_capacity(cli_args):
    cli_args.max_model_len = 1048576
    config = build_serving_engine_config(cli_args)
    specs = {spec.name: spec for spec in config.resolve_runtime_config().kv_cache_groups}
    assert specs["cmp_c4"].num_blocks == 8192
    assert specs["hca_state"].num_blocks == 18
