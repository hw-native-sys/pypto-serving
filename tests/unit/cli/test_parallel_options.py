# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import contextlib

import pytest

from pypto_serving.cli import generate as generate_module
from pypto_serving.cli import main as cli_main
from pypto_serving.cli import serve as serve_module
from pypto_serving.cli.parser import parse_launch_options
from pypto_serving.config.resolve import resolve_generate_config, resolve_launch_options
from pypto_serving.model.integrations import get_integration, resolve_speculation


def _qwen_dir(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type":"qwen3"}')
    return model_dir


def test_cli_keeps_generic_model_chunk_sizes_unrestricted(tmp_path):
    model_dir = _qwen_dir(tmp_path)
    args = parse_launch_options(
        ["--model", str(model_dir), "--long-prefill-token-threshold", "128"]
    )

    assert resolve_launch_options(args).engine.long_prefill_token_threshold == 128


def test_cli_keeps_dspark_chunk_sizes_under_its_kernel_contract():
    args = parse_launch_options(
        ["--model", "model", "--long-prefill-token-threshold", "128"]
    )

    assert args.scheduler.long_prefill_token_threshold == 128


def test_resolve_launch_options_rejects_unsupported_deepseek_chunk_size(
    tmp_path,
):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type":"deepseek_v4"}')
    args = parse_launch_options(
        ["--model", str(model_dir), "--long-prefill-token-threshold", "3072"]
    )

    with pytest.raises(
        ValueError,
        match="--long-prefill-token-threshold must be one of",
    ):
        resolve_launch_options(args)


def test_resolve_launch_options_uses_parallel_config_for_devices(tmp_path):
    model_dir = _qwen_dir(tmp_path)
    args = parse_launch_options(
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

    config = resolve_launch_options(args).engine

    assert args.parallel.data_parallel_size == 2
    assert args.parallel.tensor_parallel_size == 2
    assert args.device.devices == "0,1,2,3"
    assert config.device_id == 0
    assert config.device_ids == ()
    assert config.worker_device_ids() == (0, 1)
    assert config.parallel_config.data_parallel_size == 2
    assert config.parallel_config.tensor_parallel_size == 2
    assert config.parallel_config.replica_device_groups == ((0, 1), (2, 3))


def test_resolve_launch_options_rejects_invalid_parallel_topology(tmp_path):
    model_dir = _qwen_dir(tmp_path)
    args = parse_launch_options(
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
        resolve_launch_options(args)


@pytest.mark.parametrize("config", [
    '{"model_type":"unknown"}',
    '{"model_type":"qwen3","architectures":["DeepseekV4ForCausalLM"]}',
])
def test_unknown_or_ambiguous_model_fails_before_worker_start(tmp_path, config):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(config)

    with pytest.raises(ValueError, match="unsupported or ambiguous model architecture"):
        resolve_launch_options(parse_launch_options(["--model", str(model_dir)]))


def test_missing_model_config_is_reported_at_resolution(tmp_path):
    with pytest.raises(ValueError, match="missing model config"):
        resolve_launch_options(parse_launch_options(["--model", str(tmp_path)]))


def test_unknown_worker_integration_is_rejected():
    with pytest.raises(ValueError, match="unsupported model integration"):
        get_integration("missing")


def test_empty_prompt_still_selects_offline_mode(tmp_path):
    options = parse_launch_options(["--model", str(_qwen_dir(tmp_path)), "--prompt", ""])
    assert resolve_launch_options(options).prompts == ("",)


@pytest.mark.parametrize("ring_arg,expected", [
    ("42", 42),
    ("1,2,3,4", [1, 2, 3, 4]),
])
def test_ring_options_reach_runtime_config(tmp_path, ring_arg, expected):
    options = parse_launch_options([
        "--model", str(_qwen_dir(tmp_path)),
        "--ring-dep-pool", ring_arg,
        "--ring-task-window", ring_arg,
        "--ring-heap", ring_arg,
    ])
    runtime = resolve_launch_options(options).engine.runtime_config

    assert options.runtime.ring_dep_pool == runtime.ring_dep_pool == expected
    assert options.runtime.ring_task_window == runtime.ring_task_window == expected
    assert options.runtime.ring_heap == runtime.ring_heap == expected


def test_invalid_ring_input_is_rejected_by_parser(tmp_path):
    with pytest.raises(SystemExit):
        parse_launch_options([
            "--model", str(_qwen_dir(tmp_path)), "--ring-heap", "1,,2",
        ])


def test_generate_config_is_shared_by_serving_and_offline_modes(tmp_path):
    model_dir = str(_qwen_dir(tmp_path))
    for mode_args, prompts in (([], None), (["--prompt", "hello"], ("hello",))):
        options = parse_launch_options([
            "--model", model_dir, "--generate-config", '{"max_new_tokens":7}', *mode_args,
        ])
        launch = resolve_launch_options(options)

        assert options.generate_config == {"max_new_tokens": 7}
        assert launch.generation.max_new_tokens == 7
        assert launch.prompts == prompts
        assert launch.serve_host == "0.0.0.0"
        assert launch.serve_port == 8000


def test_main_dispatches_serving_and_offline_generation(tmp_path, monkeypatch):
    model_dir = _qwen_dir(tmp_path)
    calls = []
    monkeypatch.setattr(cli_main, "_startup_log_context", lambda **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(serve_module, "serve", lambda config, generation, **kwargs: calls.append(
        ("serve", config.integration_id, generation.max_new_tokens, kwargs)
    ))
    monkeypatch.setattr(generate_module, "generate", lambda config, prompts, generation: calls.append(
        ("generate", config.integration_id, generation.max_new_tokens, prompts)
    ))

    assert cli_main.main([
        "--model", str(model_dir), "--port", "9000", "--generate-config", '{"max_new_tokens":7}',
    ]) == 0
    assert cli_main.main([
        "--model", str(model_dir), "--prompt", "", "--generate-config", '{"max_new_tokens":7}',
    ]) == 0
    assert calls == [
        ("serve", "qwen3_14b", 7, {"host": "0.0.0.0", "port": 9000}),
        ("generate", "qwen3_14b", 7, ("",)),
    ]


def test_main_reports_model_resolution_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli_main, "_startup_log_context", lambda **kwargs: contextlib.nullcontext())

    assert cli_main.main(["--model", str(tmp_path)]) == 2
    assert "missing model config" in capsys.readouterr().err


def test_generate_config_rejects_invalid_json_input():
    for invalid in (
        [],
        "temperature=0",
        {1: 0},
        {"unknown": 1},
        {"stop": "END"},
        {"stop": ("END",)},
        {"stop": None},
    ):
        with pytest.raises(ValueError, match="--generate-config"):
            resolve_generate_config(invalid)


@pytest.mark.parametrize("invalid,field", [
    ({"temperature": -0.1}, "temperature"),
    ({"temperature": float("inf")}, "temperature"),
    ({"min_tokens": 3, "max_new_tokens": 2}, "min_tokens"),
    ({"stop": [1]}, "stop"),
])
def test_generate_config_wraps_shared_validation_error(invalid, field):
    with pytest.raises(ValueError, match=f"--generate-config: .*{field}") as caught:
        resolve_generate_config(invalid)
    assert isinstance(caught.value.__cause__, ValueError)


def test_generate_config_accepts_valid_values():
    config = resolve_generate_config(
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


def test_generate_config_does_not_mutate_json_input():
    values = {"stop": ["END"]}
    config = resolve_generate_config(values)
    values["stop"].append("MORE")
    assert config.stop == ("END",)


@pytest.mark.parametrize("options", [None, {"max_new_tokens": 8}])
def test_generate_config_defaults_to_greedy(options):
    config = resolve_generate_config(options)

    assert config.temperature == 0.0


def test_dspark_topology_accepts_the_8_card_and_16_card_worlds():
    # The DSpark serving topology is TP4 with expert parallelism spanning the
    # whole world: --ep is the rank count (16 in production, 8 on a half
    # node) and --dp is the derived scheduler-partition count.
    for dp, ep in ((2, 8), (4, 16)):
        devices = ",".join(str(device) for device in range(ep))
        args = parse_launch_options(
            [
                "--model", "model",
                "--dp", str(dp),
                "--ep", str(ep),
                "--tp", "4",
                "--devices", devices,
                "--block-size", "32",
                "--max-num-seqs", "8",
                "--max-model-len", "1024",
                "--speculative-config", '{"method":"dspark"}',
            ]
        )

        from pypto_serving.model.integrations import get_integration

        assert args.parallel.data_parallel_size == dp
        assert args.parallel.tensor_parallel_size == 4
        assert args.parallel.expert_parallel_size == ep
        plan = get_integration("deepseek_v4_dspark").resolve(
            args, {"model_type": "deepseek_v4"},
            tuple(range(ep)), resolve_speculation(args.features),
        )
        assert plan.parallel.data_parallel_size == 1
        assert plan.parallel.tensor_parallel_size == 1
        assert plan.parallel.expert_parallel_size == ep
        assert plan.parallel.replica_device_groups == (tuple(range(ep)),)


@pytest.mark.parametrize("requested", [None, "", "diagnostic-output"])
def test_worker_only_exports_explicit_build_directory(monkeypatch, requested):
    import os
    from pathlib import Path
    from types import SimpleNamespace

    from pypto_serving.serving.server.serving_worker import WorkerProcess

    if requested is None:
        monkeypatch.delenv("PYPTO_PROG_BUILD_DIR", raising=False)
    else:
        monkeypatch.setenv("PYPTO_PROG_BUILD_DIR", requested)
    worker = WorkerProcess.__new__(WorkerProcess)
    worker.config = SimpleNamespace(dp_rank=2)
    directory = worker._configure_pypto_build_dir((3, 4))
    assert directory == (Path(requested) / "serving_dp2_d3_4" if requested else None)
    assert os.environ.get("PYPTO_PROG_BUILD_DIR") == (str(directory) if requested else requested)
