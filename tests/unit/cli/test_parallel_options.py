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


@pytest.mark.parametrize(("role", "specified", "expected"), [
    (None, None, 16), ("prefill", None, 16), ("decode", None, 72),
    (None, 32, 32), ("prefill", 8, 8), ("decode", 16, 16),
])
def test_request_capacity_defaults_reach_scheduler_and_cache(tmp_path, role, specified, expected):
    model = tmp_path / "model"
    model.mkdir()
    argv = ["--model", str(model)]
    if role is not None:
        argv += ["--pd-role", role]
    if specified is not None:
        argv += ["--max-num-seqs", str(specified)]
    config = cli.build_serving_engine_config(_parse_cli_args(argv))
    assert config.max_num_running_reqs == expected
    assert config.runtime_config.max_batch_size == expected


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


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_dspark_topology_accepts_the_8_card_and_16_card_worlds(tp):
    # Attention TP determines cache partitions; expert parallelism spans the world.
    for ep in (8, 16):
        dp = ep // tp
        devices = ",".join(str(device) for device in range(ep))
        args = _parse_cli_args(
            [
                "--model", "model",
                "--dp", str(dp),
                "--ep", str(ep),
                "--tp", str(tp),
                "--devices", devices,
                "--block-size", "32",
                "--max-num-seqs", "8",
                "--max-model-len", "1024",
            ]
        )

        cli._validate_dspark_topology(args)


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


@pytest.mark.parametrize("role,budget,threshold,pages", [
    ("prefill", 8192, 8192, (145, 517, 517)), ("decode", 2048, 128, (35, 77, 77)),
    ("decode", 16384, 8192, (35, 77, 77)), (None, 8192, 8192, (1041, 4101, 4101)),
])
def test_role_specific_state_rings_do_not_change_the_chunk_contract(role, budget, threshold, pages):
    argv = ["--model", "model", "--max-model-len", "1048576", "--max-num-batched-tokens", str(budget),
            "--long-prefill-token-threshold", str(threshold),
            "--speculative-config", '{"method":"dspark","num_speculative_tokens":7}']
    if role:
        argv += ["--pd-role", role]
    runtime = cli._build_runtime_config(_parse_cli_args(argv), model_family="deepseek_v4",
        config_data={"num_hidden_layers": 43, "compress_ratios": [0, 0] + [4] * 21 + [128] * 20})
    assert runtime.max_prefill_tokens_per_request == (min(budget, 8192) if role == "decode" else threshold)
    assert tuple(g.max_blocks_per_seq for g in runtime.kv_cache_groups if g.name.endswith("state")) == pages


@pytest.mark.parametrize("ep,tp", [(8, 1), (8, 4), (16, 1), (16, 4)])
def test_dspark_pd_prefix_hits_preserve_partition_request_capacity(ep, tp):
    from dataclasses import replace
    from pypto_serving.serving.memory.kv_cache import KVCacheCapacityError, KvCacheManager

    partitions = ep // tp
    limit = (16 + partitions - 1) // partitions
    args = _parse_cli_args([
        "--model", "model", "--pd-role", "decode", "--ep", str(ep), "--tp", str(tp),
        "--dp", str(partitions), "--max-num-seqs", "16", "--max-model-len", "2048",
        "--max-num-batched-tokens", "1024",
        "--speculative-config", '{"method":"dspark","num_speculative_tokens":7}',
    ])
    runtime = cli._build_runtime_config(args, model_family="deepseek_v4",
        config_data={"num_hidden_layers": 3, "compress_ratios": [0, 4, 128]})
    assert {g.max_requests_per_partition for g in runtime.kv_cache_groups} == {limit}
    # The physical rings fit the configured local concurrency, while history
    # pages alone could admit more owners to one hot prefix partition.
    counts = {"ori": 42 * limit, "hca_state": 53 * limit,
              "csa_state": 86 * limit, "csa_inner_state": 86 * limit,
              "cmp_c128": 64, "cmp_c4": 64, "idx": 64}
    groups = tuple(replace(g, num_blocks=counts[g.name]) for g in runtime.kv_cache_groups)
    cache = KvCacheManager(enable_prefix_cache=True)
    cache.initialize(replace(runtime, kv_cache_groups=groups), num_blocks=counts["ori"],
                     group_num_blocks=counts)
    tokens = list(range(132))
    history = tuple(g.name for g in groups if g.sliding_window is None)
    hashes = cache.compute_group_block_hashes(tokens, group_names=history)

    def reserve(name, partition=None):
        return cache.reserve_group_cache(name, name, 260, partition=partition,
            prompt_token_count=132, prefix_block_hashes=hashes, shareable_group_names=history)

    reserve("warm", 0)
    cache.authorize_group_cache_write("warm")
    cache.commit_group_cache("warm")
    cache.release_group_cache("warm", confirmed_stopped=True)
    leases = [reserve(str(i)) for i in range(16)]
    assert sum(r.partition == 0 for r in leases) == limit
    assert sum(r.prefix_hit_tokens == 128 for r in leases) == limit
    assert {p: sum(r.partition == p for r in leases) for p in range(partitions)} == {
        p: limit for p in range(partitions)}
    with pytest.raises(KVCacheCapacityError):
        reserve("overflow")
    for i in range(16):
        cache.authorize_group_cache_write(str(i))
        cache.commit_group_cache(str(i))
        cache.adopt_group_cache(str(i))
    # Every admitted request can grow through K7 lookahead without exhausting
    # the hot partition while other partitions sit empty.
    for end in range(140, 261, 8):
        for i in range(16):
            cache.ensure_group_blocks(str(i), end)
    for i in range(16):
        cache.release_group_cache(str(i), confirmed_stopped=True)
    assert not cache._group_request_partitions


@pytest.mark.parametrize("interval", [0, 128, 2048, 8192, 16384])
@pytest.mark.parametrize("prefix", [False, True])
def test_retention_interval_reaches_runtime_and_scheduler(interval, prefix):
    from pypto_serving.serving.engine.async_engine import EngineConfig
    from pypto_serving.serving.sched.scheduler import SchedulerConfig

    argv = ["--model", "model", "--prefix-cache-retention-interval", str(interval),
            "--max-num-batched-tokens", "4096",
            "--speculative-config", '{"method":"dspark","num_speculative_tokens":7}']
    runtime = cli._build_runtime_config(_parse_cli_args(argv), model_family="deepseek_v4")
    config = EngineConfig(runtime_config=runtime, enable_prefix_cache=prefix)
    runtime = config.resolve_runtime_config()
    assert runtime.prefix_cache_retention_interval == (interval if prefix else 0)
    scheduler = SchedulerConfig(prefix_cache_retention_interval=runtime.prefix_cache_retention_interval)
    scheduler.validate_cache_groups(runtime.kv_cache_groups)
    assert _parse_cli_args(["--model", "model"]).prefix_cache_retention_interval == 8192


@pytest.mark.parametrize("interval", [-1, True, 1.5, 129])
def test_retention_validation_and_default(interval):
    from pypto_serving.config.types import RuntimeConfig
    from pypto_serving.model.deepseek_dspark.npu_runner import build_dspark_cache_group_specs
    from pypto_serving.serving.sched.scheduler import SchedulerConfig

    assert _parse_cli_args(["--model", "model"]).prefix_cache_retention_interval == 8192
    if interval != 129:
        with pytest.raises(ValueError, match="retention_interval"):
            RuntimeConfig(prefix_cache_retention_interval=interval)
        with pytest.raises(ValueError, match="retention_interval"):
            SchedulerConfig(prefix_cache_retention_interval=interval)
    else:
        config = SchedulerConfig(prefix_cache_retention_interval=interval)
        with pytest.raises(ValueError, match="retention_interval"):
            config.validate_cache_groups(build_dspark_cache_group_specs(43))
