# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations


import pytest

import pypto_serving.cli.main as cli


def _parse_cli_args(argv: list[str]):
    return cli.build_parser().parse_args(argv)


def test_cli_keeps_generic_model_chunk_sizes_unrestricted():
    args = _parse_cli_args(
        ["--model", "model", "--long-prefill-token-threshold", "128"]
    )

    cli._validate_prefill_chunk_size("qwen", args.long_prefill_token_threshold)
    assert args.long_prefill_token_threshold == 128


def test_cli_keeps_dspark_chunk_sizes_under_its_kernel_contract():
    args = _parse_cli_args(
        ["--model", "model", "--long-prefill-token-threshold", "128"]
    )

    cli._validate_prefill_chunk_size(
        "deepseek_v4",
        args.long_prefill_token_threshold,
        variant="dspark",
    )
    assert args.long_prefill_token_threshold == 128


def test_build_serving_engine_config_rejects_unsupported_deepseek_chunk_size(
    tmp_path,
):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type":"deepseek_v4"}')
    args = _parse_cli_args(
        ["--model", str(model_dir), "--long-prefill-token-threshold", "3072"]
    )

    with pytest.raises(
        ValueError,
        match="--long-prefill-token-threshold must be one of",
    ):
        cli.build_serving_engine_config(args)


def test_build_serving_engine_config_uses_parallel_config_for_devices(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    args = _parse_cli_args(
        [
            "--model",
            str(model_dir),
            "--devices",
            "0,1,2,3",
            "--dp",
            "2",
            "--tp",
            "2",
        ]
    )

    config = cli.build_serving_engine_config(args)

    assert config.device_id == 0
    assert config.device_ids == ()
    assert config.worker_device_ids() == (0, 1)
    assert config.parallel_config.data_parallel_size == 2
    assert config.parallel_config.tensor_parallel_size == 2
    assert config.parallel_config.replica_device_groups == ((0, 1), (2, 3))


def test_build_serving_engine_config_rejects_invalid_parallel_topology(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    args = _parse_cli_args(
        [
            "--model",
            str(model_dir),
            "--devices",
            "0,1,2",
            "--dp",
            "2",
            "--tp",
            "2",
        ]
    )

    with pytest.raises(ValueError, match="number of devices"):
        cli.build_serving_engine_config(args)


def test_generate_config_rejects_wrong_value_types():
    for invalid in (
        {"stream": "false"},
        {"ignore_eos": "true"},
        {"stop": "END"},
        {"stop": [1, 2]},
        {"max_new_tokens": 2.5},
        {"max_new_tokens": True},
        {"max_new_tokens": 0},
        {"temperature": -0.1},
        {"temperature": "0"},
        {"top_p": 0},
        {"top_p": 1.5},
        {"top_k": 0},
        {"top_k": "5"},
    ):
        with pytest.raises(ValueError, match="--generate-config"):
            cli._build_generate_config(invalid)


def test_generate_config_accepts_valid_values():
    config = cli._build_generate_config(
        {
            "max_new_tokens": 8,
            "temperature": 0.2,
            "top_p": 1,
            "top_k": None,
            "stop": ["END"],
            "stream": True,
            "ignore_eos": False,
        }
    )

    assert config.max_new_tokens == 8
    assert config.temperature == 0.2
    assert config.top_p == 1
    assert config.top_k is None
    assert config.stop == ("END",)
    assert config.stream is True
    assert config.ignore_eos is False


@pytest.mark.parametrize("options", [None, {"max_new_tokens": 8}])
def test_generate_config_defaults_to_greedy(options):
    config = cli._build_generate_config(options)

    assert config.temperature == 0.0
