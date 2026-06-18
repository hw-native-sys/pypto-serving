# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import python.cli.main as cli


def _parse_args(argv: list[str]):
    return cli.build_parser().parse_args(argv)


def test_build_serving_engine_config_uses_cli_args(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    monkeypatch.setenv("PYPTO_ROOT", "/tmp/pypto")
    monkeypatch.setenv("PYPTO_SAVE_KERNELS_DIR", "/tmp/kernels")
    args = _parse_args([
        "--model", str(model_dir),
        "--served-model-name", "api-model",
        "--backend", "npu",
        "--platform", "a5",
        "--device", "2",
        "--max-model-len", "1024",
        "--block-size", "64",
        "--dtype", "bfloat16",
        "--kv-cache-dtype", "auto",
        "--max-new-tokens", "8",
        "--max-num-seqs", "4",
        "--max-num-batched-tokens", "256",
        "--long-prefill-token-threshold", "64",
        "--no-enable-prefix-caching",
        "--prefix-cache-backend", "radix",
        "--no-enable-chunked-prefill",
    ])

    config = cli.build_serving_engine_config(args)

    assert config.model_id == "api-model"
    assert config.model_dir == str(model_dir.resolve())
    assert config.platform == "a5"
    assert config.device_id == 2
    assert config.executor_cls == "PyptoQwen14BExecutor"
    assert config.executor_kwargs == {
        "pypto_root": "/tmp/pypto",
        "save_kernels_dir": "/tmp/kernels",
    }
    assert config.runtime_config.page_size == 64
    assert config.runtime_config.max_batch_size == 4
    assert config.runtime_config.max_seq_len == 1024
    assert config.runtime_config.kv_dtype == "bfloat16"
    assert config.runtime_config.weight_dtype == "bfloat16"
    assert config.runtime_config.max_new_tokens == 8
    assert config.max_num_running_reqs == 4
    assert config.max_num_scheduled_tokens == 256
    assert config.long_prefill_token_threshold == 64
    assert config.enable_prefix_cache is False
    assert config.prefix_cache_backend == "radix"
    assert config.enable_chunk_prefill is False


def test_prefix_cache_backend_defaults_to_hash_even_with_env(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    monkeypatch.setenv("PYPTO_PREFIX_CACHE_BACKEND", "radix")

    args = _parse_args(["--model", str(model_dir), "--backend", "npu"])
    config = cli.build_serving_engine_config(args)

    assert args.prefix_cache_backend == "hash"
    assert config.prefix_cache_backend == "hash"


def test_parser_rejects_invalid_backend(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    with pytest.raises(SystemExit):
        _parse_args(["--model", str(model_dir), "--backend", "cpu"])


def test_parser_rejects_removed_prompt_mode(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    with pytest.raises(SystemExit):
        _parse_args(["--model", str(model_dir), "--backend", "npu", "--prompt", "hello"])


def test_main_starts_serving(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    calls = []
    monkeypatch.setattr(
        cli,
        "run_serve",
        lambda config, *, host, port: calls.append((config, host, port)),
    )

    assert cli.main(["--model", str(model_dir), "--backend", "npu", "--host", "127.0.0.1", "--port", "8899"]) == 0

    config, host, port = calls[-1]
    assert config.model_id == model_dir.name
    assert host == "127.0.0.1"
    assert port == 8899


def test_main_suppresses_startup_logs_by_default(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    startup_context_flags = []
    monkeypatch.setattr(cli, "run_serve", lambda config, *, host, port: None)
    monkeypatch.setattr(
        cli,
        "_startup_log_context",
        lambda *, enabled: _RecordingContext(startup_context_flags, enabled),
    )

    cli.main(["--model", str(model_dir), "--backend", "npu"])

    assert startup_context_flags == [True]


def test_main_can_show_startup_logs(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    startup_context_flags = []
    monkeypatch.setattr(cli, "run_serve", lambda config, *, host, port: None)
    monkeypatch.setattr(
        cli,
        "_startup_log_context",
        lambda *, enabled: _RecordingContext(startup_context_flags, enabled),
    )

    cli.main(["--model", str(model_dir), "--backend", "npu", "--show-startup-logs"])

    assert startup_context_flags == [False]


class _RecordingContext:
    def __init__(self, flags: list[bool], enabled: bool) -> None:
        self._flags = flags
        self._enabled = enabled

    def __enter__(self):
        self._flags.append(self._enabled)

    def __exit__(self, exc_type, exc, traceback):
        return False
