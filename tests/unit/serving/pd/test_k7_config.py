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
from pypto_serving.model.deepseek_dspark.pd_adapter import DSV4_DSPARK_K7_CONTRACT


K7 = DSV4_DSPARK_K7_CONTRACT.decode_speculative_tokens


def _pd_args(
    tmp_path,
    *,
    role: str,
    speculative_tokens: int,
    prefix_cache_mode: str = "disabled",
):
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
    pd_config = tmp_path / "pd.json"
    pd_config.write_text(
        json.dumps(
            {
                "runtime": {
                    "run_id": "run-k7",
                    "prefix_cache_mode": prefix_cache_mode,
                    "prefill": [
                        {"host": "127.0.0.1", "port": 8101, "node_id": "p"}
                    ],
                    "decode": [
                        {"host": "127.0.0.1", "port": 8102, "node_id": "d"}
                    ],
                },
                "observability": {"root": str(tmp_path / "logs")},
            }
        ),
        encoding="utf-8",
    )
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
            "--pd-config",
            str(pd_config),
        ]
    )


@pytest.mark.parametrize(
    ("role", "expected_local_tokens"),
    [("prefill", 0), ("decode", K7)],
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
            speculative_tokens=K7,
        )
    )

    assert config.pd_config.model_contract.decode_speculative_tokens == K7
    assert config.pd_config.model_contract.adapter_id == "deepseek-v4-dspark-k7"
    assert config.executor_kwargs["num_speculative_tokens"] == expected_local_tokens
    assert config.runtime_config.num_speculative_tokens == expected_local_tokens
    assert config.async_scheduling is False
    assert config.pd_config.connect_timeout_seconds == 30
    assert config.pd_config.request_timeout_seconds == 600
    assert config.pd_config.max_pending_handoffs == 8
    assert config.pd_config.max_transfer_attempts == 2
    config.validate_pd()


@pytest.mark.parametrize(
    ("mode", "role", "expected"),
    (
        ("d_only", "prefill", False),
        ("d_only", "decode", True),
        ("independent", "prefill", True),
        ("independent", "decode", True),
    ),
)
def test_pd_prefix_cache_profile_drives_local_engine_flag(
    tmp_path, mode, role, expected
) -> None:
    config = cli.build_serving_engine_config(
        _pd_args(
            tmp_path,
            role=role,
            speculative_tokens=K7,
            prefix_cache_mode=mode,
        )
    )
    assert config.enable_prefix_cache is expected
    if expected:
        assert config.runtime_config.speculative_prefix_cache_replay_tokens == max(
            group.sliding_window or 0
            for group in config.runtime_config.kv_cache_groups
        )
    config.validate_pd()


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_pd_rejects_k0_at_startup(tmp_path, role) -> None:
    with pytest.raises(ValueError, match="match exactly one PD model adapter"):
        cli.build_serving_engine_config(
            _pd_args(tmp_path, role=role, speculative_tokens=0)
        )


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_external_router_config_is_derived_from_shared_document(tmp_path, role) -> None:
    args = _pd_args(
        tmp_path,
        role=role,
        speculative_tokens=K7,
    )
    config = cli.build_serving_engine_config(args)
    assert config.pd_config.run_id == "run-k7"
    assert config.pd_config.control_advertise_host == "127.0.0.1"
    assert config.pd_config.journal_path.endswith("state/journal.jsonl")
