# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Qwen3 output accuracy guard for CI.

Runs offline generation through the same engine as serving
(``AsyncLLMEngine`` -> ``Scheduler`` -> ``WorkerProcess``), so this guard also
pins offline/online parity: greedy output must be identical whether prompts
arrive from the CLI generate mode or the HTTP server.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest


MODEL_DIR_ENV = os.environ.get("PYPTO_QWEN3_MODEL_DIR")
MODEL_DIR = Path(MODEL_DIR_ENV) if MODEL_DIR_ENV else None
MODEL_ID = "qwen3-14b-accuracy"
PLATFORM = os.environ.get("PYPTO_QWEN3_PLATFORM", "a2a3")
DEVICE_ID_ENV = os.environ.get("DEVICE_ID")
DEVICE_ID = int(DEVICE_ID_ENV) if DEVICE_ID_ENV is not None else None
PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 8

EXPECTED_TOKEN_IDS = [12095, 13, 3555, 374, 279, 6722, 315, 279]


def test_qwen3_output_matches_expected_tokens():
    if MODEL_DIR is None or not MODEL_DIR.is_dir():
        pytest.fail(f"PYPTO_QWEN3_MODEL_DIR not set or not a directory: {MODEL_DIR}")
    if DEVICE_ID is None:
        pytest.fail("DEVICE_ID is required")

    from pypto_serving.config.types import GenerateConfig, RuntimeConfig
    from pypto_serving.model.tokenizer import load_tokenizer
    from pypto_serving.serving.engine.async_engine import AsyncLLMEngine, EngineConfig

    async def run():
        engine = AsyncLLMEngine(
            config=EngineConfig(
                model_id=MODEL_ID,
                model_dir=str(MODEL_DIR),
                integration_id="qwen3_14b",
                platform=PLATFORM,
                device_id=DEVICE_ID,
                device_ids=(DEVICE_ID,),
                runtime_config=RuntimeConfig(
                    page_size=128,
                    max_batch_size=16,
                    max_seq_len=512,
                    max_new_tokens=MAX_NEW_TOKENS,
                    device="cpu",
                    kv_dtype="bfloat16",
                    weight_dtype="float32",
                    # Simpler ring sizing (per-dispatch RunConfig). The CI-era
                    # PTO2_RING_* value 16384 exhausts the dep pool mid-run
                    # ("raise runtime_env.ring_dep_pool"); 262144 is the
                    # Qwen serving-proven sizing. Heap stays 1 GiB so the
                    # float32-weight KV budget still fits.
                    ring_dep_pool=262144,
                    ring_task_window=262144,
                    ring_heap=1073741824,
                ),
                max_num_running_reqs=16,
                enable_prefix_cache=False,
            ),
            tokenizer=load_tokenizer(str(MODEL_DIR)),
        )
        await engine.start()
        try:
            return await engine.generate_result(
                PROMPT,
                GenerateConfig(
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=0.0,
                    top_p=1.0,
                    top_k=None,
                ),
            )
        finally:
            await engine.stop()

    result = asyncio.run(run())
    assert result.token_ids == EXPECTED_TOKEN_IDS, (
        f"Qwen3 output changed for prompt {PROMPT!r}:\n"
        f"expected token_ids: {EXPECTED_TOKEN_IDS}\n"
        f"actual token_ids:   {result.token_ids}\n"
        f"actual text:        {result.text!r}\n"
        f"finish_reason:      {result.finish_reason}"
    )
