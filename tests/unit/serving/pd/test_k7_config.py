# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import json

import pytest

import pypto_serving.cli.main as cli
from pypto_serving.serving.pd.config import PD_DSPARK_SPECULATIVE_TOKENS


def _pd_args(tmp_path, *, role: str, speculative_tokens: int):
    model_dir = tmp_path / f"dspark-{role}-{speculative_tokens}"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DeepseekV4ForCausalLM"],
                "model_type": "deepseek_v4",
                "num_hidden_layers": 43,
                "compress_ratios": [4] * 43,
            }
        )
    )
    local = "p" if role == "prefill" else "d"
    peer = "d" if role == "prefill" else "p"
    return cli.build_parser().parse_args(
        [
            "--model",
            str(model_dir),
            "--devices",
            ",".join(str(index) for index in range(16)),
            "--dp",
            "4",
            "--tp",
            "4",
            "--ep",
            "16",
            "--block-size",
            "32",
            "--max-model-len",
            "256",
            "--max-num-seqs",
            "1",
            "--speculative-config",
            json.dumps(
                {
                    "method": "dspark",
                    "num_speculative_tokens": speculative_tokens,
                }
            ),
            "--pd-role",
            role,
            "--pd-node-id",
            local,
            "--pd-peer-node-id",
            peer,
            "--pd-run-id",
            "run-k7",
            "--pd-control-host",
            "127.0.0.1",
            "--pd-peer-host",
            "127.0.0.1",
            "--pd-transfer-hostname",
            "127.0.0.1",
            "--pd-model-revision",
            "dsv4-flash-dspark",
            "--pd-connect-timeout-seconds",
            "600",
            "--pd-request-timeout-seconds",
            "300",
            "--pd-journal-path",
            str(tmp_path / f"{role}-journal.jsonl"),
        ]
    )


@pytest.mark.parametrize(
    ("role", "expected_local_tokens"),
    [("prefill", 0), ("decode", PD_DSPARK_SPECULATIVE_TOKENS)],
)
def test_pd_k7_uses_target_only_prefill_and_k7_decode(
    tmp_path,
    role,
    expected_local_tokens,
) -> None:
    config = cli.build_serving_engine_config(
        _pd_args(
            tmp_path,
            role=role,
            speculative_tokens=PD_DSPARK_SPECULATIVE_TOKENS,
        )
    )

    assert config.pd_config.decode_speculative_tokens == PD_DSPARK_SPECULATIVE_TOKENS
    assert config.executor_kwargs["num_speculative_tokens"] == expected_local_tokens
    assert config.runtime_config.num_speculative_tokens == expected_local_tokens
    assert config.async_scheduling is False
    assert config.pd_config.connect_timeout_seconds == 600
    assert config.pd_config.request_timeout_seconds == 300
    assert config.pd_config.max_pending_handoffs == 8
    assert config.pd_config.max_transfer_attempts == 2
    config.validate_pd()


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_pd_rejects_k0_at_startup(tmp_path, role) -> None:
    with pytest.raises(ValueError, match="requires DeepSeek V4 DSpark K7"):
        cli.build_serving_engine_config(
            _pd_args(tmp_path, role=role, speculative_tokens=0)
        )


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_external_router_config_does_not_require_a_fixed_peer(tmp_path, role) -> None:
    args = _pd_args(
        tmp_path,
        role=role,
        speculative_tokens=PD_DSPARK_SPECULATIVE_TOKENS,
    )
    args.pd_deployment_mode = "external-router"
    args.pd_control_advertise_host = "192.0.2.10"
    args.pd_peer_node_id = ""
    args.pd_peer_host = ""
    config = cli.build_serving_engine_config(args)
    assert config.pd_config.external_router
    assert config.pd_config.peer_node_id == ""
    assert config.pd_config.control_advertise_host == "192.0.2.10"
