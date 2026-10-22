# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Offline generation through the shared serving engine."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence

from pypto_serving.config.types import GenerateConfig
from pypto_serving.serving.engine.async_engine import AsyncLLMEngine, EngineConfig
from pypto_serving.tools.profile import (
    configure_profiler, merge_profile, profile_span, start_profile, stop_profile,
)


def generate(config: EngineConfig, prompts: Sequence[str], generation: GenerateConfig) -> None:
    """Run offline prompts through an engine; streaming requires one prompt."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for name in ("simpler_setup", "pypto", "simpler"):
        logging.getLogger(name).setLevel(logging.WARNING)
    if generation.stream and len(prompts) > 1:
        raise ValueError("--generate-config stream=true requires a single --prompt")

    from pypto_serving.model.tokenizer import load_tokenizer

    configure_profiler(
        config.profile_config, process_name="pypto-serving-generate", initially_active=False,
    )
    engine = AsyncLLMEngine(
        config=config, tokenizer=load_tokenizer(config.model_dir, integration_id=config.integration_id),
    )
    print(f"Generating {len(prompts)} prompt(s): model={config.model_id}")
    asyncio.run(_run_generation(engine, config, prompts, generation))


async def _run_generation(
    engine: AsyncLLMEngine, config: EngineConfig,
    prompts: Sequence[str], generation: GenerateConfig,
) -> None:
    """Start the engine, run prompts, and stop profiling and engine on exit."""
    await engine.start()
    profile_enabled = config.profile_config.enabled
    try:
        if profile_enabled:
            start_profile()
            await engine.start_profile()
        started = time.perf_counter()
        with profile_span(
            "cli.generate", cat="request",
            args={
                "model_id": config.model_id, "num_prompts": len(prompts),
                "max_new_tokens": generation.max_new_tokens, "stream": generation.stream,
            },
        ):
            num_tokens = await _generate_prompts(engine, prompts, generation)
        elapsed = time.perf_counter() - started
        print(f"[generate] {num_tokens} tokens in {elapsed:.2f}s ({num_tokens / elapsed:.1f} tok/s)")
    finally:
        try:
            if profile_enabled:
                await engine.stop_profile()
                stop_profile()
                merge_profile()
        finally:
            await engine.stop()


async def _generate_prompts(
    engine: AsyncLLMEngine, prompts: Sequence[str], generation: GenerateConfig,
) -> int:
    """Generate each prompt and return the total number of completion tokens."""
    if generation.stream:
        return await _generate_stream(engine, prompts[0], generation)
    results = await engine.generate_batch(list(prompts), generation)
    for index, result in enumerate(results):
        print(f"-- prompt {index + 1}/{len(prompts)} --")
        print(f"text: {result.text}")
        token_ids = result.token_ids
        print(f"token_ids: {token_ids[:64]}{'...' if len(token_ids) > 64 else ''}")
        print(f"finish_reason: {result.finish_reason}")
    return sum(len(result.token_ids) for result in results)


async def _generate_stream(engine: AsyncLLMEngine, prompt: str, generation: GenerateConfig) -> int:
    """Stream one prompt and return its final completion token count."""
    request_id = engine.generate_request_id()
    previous_text = ""
    token_count = 0
    async for output in engine.add_request(request_id, prompt, generation):
        delta = output.text[len(previous_text):] if output.text else ""
        previous_text = output.text or previous_text
        if output.token_id is not None:
            token_count += 1
        if delta:
            print(delta, end="", flush=True)
        if output.finished:
            print()
            print(f"finish_reason: {engine.normalize_finish_reason(output.finish_reason)}")
            token_count = output.completion_tokens or token_count
    return token_count
