# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Verify prefix cache and chunk prefill work on NPU."""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from python.core.async_engine import DPEngineCore, EngineConfig
from python.core.tokenizer import TransformersTokenizerAdapter
from python.core.types import GenerateConfig, RuntimeConfig


def build_prompt_with_min_tokens(tokenizer, min_tokens: int, max_tokens: int) -> str:
    words = []
    while True:
        words.append(f"prefixword{len(words)}")
        prompt = " ".join(words)
        num_tokens = len(tokenizer.encode(prompt))
        if num_tokens >= min_tokens:
            if num_tokens > max_tokens:
                raise RuntimeError(
                    f"Unable to build prompt within token budget: {num_tokens} > {max_tokens}"
                )
            return prompt


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="/data/linyifan/models/Qwen3-14B")
    parser.add_argument("--platform", type=str, default="a2a3")
    parser.add_argument("--device", "-d", type=int, default=0)
    args = parser.parse_args()

    model_dir = args.model_dir
    print(f"Model: {model_dir}, Device: {args.device}")

    tokenizer = TransformersTokenizerAdapter.from_pretrained(model_dir)

    page_size = 256
    runtime_config = RuntimeConfig(
        page_size=page_size,
        max_batch_size=16,
        max_seq_len=512,
        device="cpu",
        kv_dtype="bfloat16",
        weight_dtype="float32",
        max_new_tokens=5,
    )

    engine_config = EngineConfig(
        model_id="qwen3-14b",
        model_dir=model_dir,
        platform=args.platform,
        device_id=args.device,
        executor_cls="PyptoQwen14BExecutor",
        runtime_config=runtime_config,
        max_num_running_reqs=4,
        max_num_scheduled_tokens=4096,
        long_prefill_token_threshold=128,
    )

    engine = DPEngineCore(
        config=engine_config,
        tokenizer=tokenizer,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )

    original_schedule = engine.scheduler.schedule
    first_scheduled_computed_tokens: dict[str, int] = {}
    prefill_step_counts: dict[str, int] = {}

    def counted_schedule():
        scheduler_output = original_schedule()
        for scheduled in scheduler_output.scheduled_requests:
            if scheduled.is_prefill:
                prefill_step_counts[scheduled.request.request_id] = (
                    prefill_step_counts.get(scheduled.request.request_id, 0) + 1
                )
                first_scheduled_computed_tokens.setdefault(
                    scheduled.request.request_id,
                    scheduled.num_computed_tokens,
                )
        return scheduler_output

    engine.scheduler.schedule = counted_schedule

    await engine.start()
    print("Engine started\n")

    gen_config = GenerateConfig(max_new_tokens=5, temperature=0.0)

    # --- Test 1: Prefix Cache ---
    print("=== Test 1: Prefix Cache ===")
    prompt = build_prompt_with_min_tokens(
        tokenizer,
        min_tokens=page_size + 32,
        max_tokens=360,
    )
    prompt_tokens = len(tokenizer.encode(prompt))
    print(f"Shared prefix token count: {prompt_tokens} (block_size={page_size})")

    # First request: full prefill
    print("Request 1: long shared prefix")
    t0 = time.time()
    async for output in engine.add_request("req-1", prompt, gen_config):
        if output.finished:
            break
    t1 = time.time()
    print(f"  Time: {t1 - t0:.2f}s, Output: {output.text[:50]}")

    # Check scheduler state: how many blocks are cached
    cached_count = len(engine.kv_cache_manager.hash_to_block)
    print(f"  Cached blocks in pool: {cached_count}")
    if cached_count <= 0:
        raise AssertionError("Prefix cache did not retain any full prompt blocks")

    # Second request with same prefix + extra
    prompt2 = prompt + " continuation tokens for the second request"
    prompt2_tokens = len(tokenizer.encode(prompt2))
    print(f"Request 2: shared prefix + continuation ({prompt2_tokens} tokens)")
    t2 = time.time()
    async for output in engine.add_request("req-2", prompt2, gen_config):
        if output.finished:
            break
    t3 = time.time()
    print(f"  Time: {t3 - t2:.2f}s, Output: {output.text[:50]}")

    cached_tokens = first_scheduled_computed_tokens.get("req-2", 0)
    print(f"  Request 2 cached prompt tokens: {cached_tokens}")
    if cached_tokens < page_size:
        raise AssertionError("Second request did not start from a cached full block")
    print("  Prefix cache hit validated")
    print()

    # --- Test 2: Chunk Prefill ---
    print("=== Test 2: Chunk Prefill ===")
    # Use a prompt that exceeds long_prefill_token_threshold (128) but fits max_seq_len (512)
    long_prompt = "Explain in detail the history of " + " ".join(
        [f"word{i}" for i in range(60)]
    )
    token_count = len(tokenizer.encode(long_prompt))
    print(f"Long prompt token count: {token_count} (threshold=128, max_seq=512)")

    t4 = time.time()
    async for output in engine.add_request("req-3", long_prompt, gen_config):
        if output.finished:
            break
    t5 = time.time()
    print(f"  Time: {t5 - t4:.2f}s, Output: {output.text[:50]}")
    chunk_steps = prefill_step_counts.get("req-3", 0)
    print(f"  Prefill steps: {chunk_steps}")
    if chunk_steps < 2:
        raise AssertionError("Chunk prefill did not split the long prompt")
    print("  Chunk prefill completed")
    print()

    await engine.stop()
    print("=== All NPU verification tests passed! ===")


if __name__ == "__main__":
    asyncio.run(main())
