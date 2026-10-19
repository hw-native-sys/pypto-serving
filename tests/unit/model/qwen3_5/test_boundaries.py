# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Reject unsupported topology and HTTP input before model execution."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from pypto_serving.cli.main import build_parser, build_serving_engine_config
from pypto_serving.config.types import GenerateConfig
from pypto_serving.serving.server.server import (
    ChatCompletionRequest, CompletionRequest, ServingServer,
)


def _model_root() -> Path:
    if "MODEL_ROOT" not in os.environ:
        pytest.skip("MODEL_ROOT is required for real Qwen3.8 checkpoint checks")
    root = Path(os.environ["MODEL_ROOT"])
    if not (root / "config.json").is_file():
        pytest.skip("pinned Qwen3.8 checkpoint is unavailable")
    return root


def _args(*extra: str):
    return build_parser().parse_args([
        "--model", str(_model_root()), "--devices", "0", "--max-model-len", "128",
        "--max-num-seqs", "1", "--max-num-batched-tokens", "128",
        "--no-enable-prefix-caching", "--no-enable-chunked-prefill", *extra,
    ])


def test_cli_routes_and_rejects_unsupported_shapes() -> None:
    config = build_serving_engine_config(_args())
    assert config.executor_cls == "PyptoQwen35Executor"
    assert config.runtime_config.max_prefill_tokens_per_request == 128
    for options, expected in ((["--max-model-len", "129"], "128 positions"),
                              (["--dp", "2", "--devices", "0,1"], "one NPU")):
        with pytest.raises(ValueError, match=expected):
            build_serving_engine_config(_args(*options))


def test_media_and_bad_numeric_params_fail_before_engine_call() -> None:
    server = ServingServer.__new__(ServingServer)
    server.engine = SimpleNamespace(config=SimpleNamespace(executor_cls="PyptoQwen35Executor"))
    server.generate_config = GenerateConfig()
    for label in ("image", "video"):
        content = [
            {"type": "text", "text": "Hi"},
            {"type": f"{label}_url", f"{label}_url": {"url": "data:ignored"}},
        ]
        request = ChatCompletionRequest(messages=[{"role": "user", "content": content}])
        with pytest.raises(ValueError, match=rf"{label} input is unsupported"):
            asyncio.run(server._chat_completions(request))
    with pytest.raises(ValueError, match="max_tokens"):
        server._resolve_generate_config(CompletionRequest(prompt="x", max_tokens=0))
