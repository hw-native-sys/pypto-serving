# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Qwen3 serving-path accuracy guards for CI.

The offline ``LLMEngine`` accuracy guard (``test_qwen3_accuracy.py``) never
routes through the scheduler, so it cannot exercise prefix caching or chunked
prefill. Those features only run on the serving path
(``AsyncLLMEngine`` -> ``Scheduler`` -> ``WorkerProcess``). This module drives
that real path and guards three optimizations, each of which must keep greedy
output identical to the baseline while the scheduler is observed to actually
apply the optimization:

* multi-batch   - several concurrent requests co-batched by the scheduler
* chunked prefill - one prompt split across multiple prefill steps
* prefix cache  - a repeated long prompt reusing cached prefix blocks

To keep CI cheap the model is loaded exactly once (a single worker process);
each test only tweaks main-process scheduler state, which the worker follows.
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from python.core.async_engine import AsyncLLMEngine


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


MODEL_DIR_ENV = os.environ.get("PYPTO_QWEN3_MODEL_DIR")
MODEL_DIR = Path(MODEL_DIR_ENV) if MODEL_DIR_ENV else None
MODEL_ID = "qwen3-14b-serving"
PLATFORM = os.environ.get("PYPTO_QWEN3_PLATFORM", "a2a3")
DEVICE_ID_ENV = os.environ.get("DEVICE_ID")
DEVICE_ID = int(DEVICE_ID_ENV) if DEVICE_ID_ENV is not None else None

# Kernel requires page_size == 128 (examples/.../npu_executor.py), so a full
# prefix-cache block is 128 tokens. Short prompts therefore never populate the
# prefix cache, keeping the multi-batch / chunked-prefill tests isolated from it.
PAGE_SIZE = 128
MAX_BATCH_SIZE = 16
MAX_SEQ_LEN = 512
# Compile-time generation ceiling baked into the runner; per-request
# GenerateConfig stays at or below this.
RUNTIME_MAX_NEW_TOKENS = 16

# Same greedy prompt / expected tokens as the offline accuracy guard. The
# serving path shares the executor kernels, so greedy output must match.
PROMPT = "The capital of France is"
SHORT_NEW_TOKENS = 8
EXPECTED_TOKEN_IDS = [12095, 13, 3555, 374, 279, 6722, 315, 279]

# A prompt long enough to fill at least one 128-token prefix-cache block.
LONG_PROMPT = "Paris is the capital of France. " * 25
LONG_NEW_TOKENS = 8


@dataclass
class _Harness:
    """One shared serving engine plus scheduler observability for the tests."""

    engine: "AsyncLLMEngine"
    loop: asyncio.AbstractEventLoop
    default_threshold: int
    schedule_events: list[list[tuple[str, bool, int]]] = field(default_factory=list)
    cache_hits: list[int] = field(default_factory=list)

    def reset(self) -> None:
        """Clear recorded scheduler activity and restore the default threshold."""
        self.schedule_events.clear()
        self.cache_hits.clear()
        self.engine.scheduler.config.long_prefill_token_threshold = self.default_threshold

    def run(self, coro):
        return self.loop.run_until_complete(coro)


async def _collect(engine, prompt: str, max_new_tokens: int) -> list[int]:
    """Drive one request to completion and return its generated token ids."""
    from python.core.types import GenerateConfig

    request_id = engine.generate_request_id()
    config = GenerateConfig(
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=None,
    )
    tokens: list[int] = []
    async for output in engine.add_request(request_id, prompt, config):
        if output.token_id is not None:
            tokens.append(output.token_id)
    return tokens


@pytest.fixture(scope="module")
def harness():
    if MODEL_DIR is None or not MODEL_DIR.is_dir():
        pytest.fail(f"PYPTO_QWEN3_MODEL_DIR not set or not a directory: {MODEL_DIR}")
    if DEVICE_ID is None:
        pytest.fail("DEVICE_ID is required")

    from python.core.async_engine import AsyncLLMEngine, EngineConfig
    from python.core.tokenizer import TransformersTokenizerAdapter
    from python.core.types import RuntimeConfig

    tokenizer = TransformersTokenizerAdapter.from_pretrained(str(MODEL_DIR))
    # Default to no chunking; the chunked-prefill test lowers this per-request.
    default_threshold = MAX_SEQ_LEN
    config = EngineConfig(
        model_id=MODEL_ID,
        model_dir=str(MODEL_DIR),
        platform=PLATFORM,
        device_id=DEVICE_ID,
        executor_cls="PyptoQwen14BExecutor",
        runtime_config=RuntimeConfig(
            page_size=PAGE_SIZE,
            max_batch_size=MAX_BATCH_SIZE,
            max_seq_len=MAX_SEQ_LEN,
            max_new_tokens=RUNTIME_MAX_NEW_TOKENS,
            device="cpu",
            kv_dtype="bfloat16",
            weight_dtype="float32",
        ),
        max_num_running_reqs=MAX_BATCH_SIZE,
        long_prefill_token_threshold=default_threshold,
        enable_prefix_cache=True,
        enable_chunk_prefill=True,
    )

    engine = AsyncLLMEngine(
        config=config,
        tokenizer=tokenizer,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    h = _Harness(engine=engine, loop=loop, default_threshold=default_threshold)

    # Spy on the (main-process) scheduler so tests can prove each optimization
    # actually fired, instead of only checking output equality.
    orig_schedule = engine.scheduler.schedule

    def schedule_spy():
        output = orig_schedule()
        if output.scheduled_requests:
            h.schedule_events.append(
                [
                    (sr.request.request_id, sr.is_prefill, sr.num_new_tokens)
                    for sr in output.scheduled_requests
                ]
            )
        return output

    engine.scheduler.schedule = schedule_spy

    orig_get_computed = engine.kv_cache_manager.get_computed_blocks

    def get_computed_spy(token_ids):
        blocks = orig_get_computed(token_ids)
        h.cache_hits.append(len(blocks))
        return blocks

    engine.kv_cache_manager.get_computed_blocks = get_computed_spy

    try:
        loop.run_until_complete(engine.start())
    except Exception:
        # Best-effort cleanup without masking the original startup failure.
        try:
            loop.run_until_complete(engine.stop())
        except Exception:
            pass
        finally:
            loop.close()
            asyncio.set_event_loop(None)
        raise

    try:
        yield h
    finally:
        try:
            loop.run_until_complete(engine.stop())
        finally:
            loop.close()
            asyncio.set_event_loop(None)


def _max_batch_width(events: list[list[tuple[str, bool, int]]]) -> int:
    return max((len(event) for event in events), default=0)


def test_serving_single_request_matches_expected_tokens(harness):
    """Baseline: the serving path reproduces the offline greedy output."""
    harness.reset()

    tokens = harness.run(_collect(harness.engine, PROMPT, SHORT_NEW_TOKENS))

    assert tokens == EXPECTED_TOKEN_IDS, (
        f"Serving path greedy output diverged from the offline baseline:\n"
        f"expected: {EXPECTED_TOKEN_IDS}\n"
        f"actual:   {tokens}"
    )


def test_multi_batch_matches_single_request(harness):
    """Several concurrent requests are co-batched and each matches the baseline."""
    harness.reset()
    num_requests = 4

    async def _run_all():
        return await asyncio.gather(
            *(_collect(harness.engine, PROMPT, SHORT_NEW_TOKENS) for _ in range(num_requests))
        )

    results = harness.run(_run_all())

    for idx, tokens in enumerate(results):
        assert tokens == EXPECTED_TOKEN_IDS, (
            f"Batched request {idx} diverged from the baseline:\n"
            f"expected: {EXPECTED_TOKEN_IDS}\n"
            f"actual:   {tokens}"
        )

    # Prove the scheduler actually co-batched at least two requests in one step.
    max_width = _max_batch_width(harness.schedule_events)
    assert max_width >= 2, (
        f"Expected the scheduler to co-batch >=2 requests in one step, "
        f"but the widest step had {max_width} request(s)."
    )


def test_chunked_prefill_matches_single_request(harness):
    """A prompt split across multiple prefill steps still matches the baseline."""
    harness.reset()
    # Force chunking: at most 2 prompt tokens are prefilled per scheduler step.
    harness.engine.scheduler.config.long_prefill_token_threshold = 2

    tokens = harness.run(_collect(harness.engine, PROMPT, SHORT_NEW_TOKENS))

    assert tokens == EXPECTED_TOKEN_IDS, (
        f"Chunked-prefill output diverged from the baseline:\n"
        f"expected: {EXPECTED_TOKEN_IDS}\n"
        f"actual:   {tokens}"
    )

    # Prove the prompt was actually chunked: the request must appear in >=2
    # separate prefill scheduling steps.
    prefill_steps = Counter(
        request_id
        for event in harness.schedule_events
        for request_id, is_prefill, _ in event
        if is_prefill
    )
    max_prefill_steps = max(prefill_steps.values(), default=0)
    assert max_prefill_steps >= 2, (
        f"Expected the prompt to be chunked into >=2 prefill steps, "
        f"but the most any request saw was {max_prefill_steps}."
    )


def test_prefix_cache_reuses_prefix_and_preserves_output(harness):
    """A repeated long prompt hits the prefix cache without changing output."""
    harness.reset()

    prompt_len = len(harness.engine.tokenizer.encode(LONG_PROMPT))
    assert prompt_len > PAGE_SIZE, (
        f"Prefix-cache test needs a prompt longer than one {PAGE_SIZE}-token "
        f"block, got {prompt_len} tokens."
    )
    assert prompt_len + LONG_NEW_TOKENS <= MAX_SEQ_LEN

    # Cold run publishes the prefix blocks; hot run must reuse them.
    cold = harness.run(_collect(harness.engine, LONG_PROMPT, LONG_NEW_TOKENS))
    hits_after_cold = max(harness.cache_hits, default=0)
    hot = harness.run(_collect(harness.engine, LONG_PROMPT, LONG_NEW_TOKENS))

    assert hot == cold, (
        f"Prefix-cache hit changed greedy output:\n"
        f"cold: {cold}\n"
        f"hot:  {hot}"
    )
    # Cold run starts with an empty cache; the hot run must reuse >=1 block.
    assert hits_after_cold == 0, (
        f"Prefix cache was unexpectedly warm before the cold run "
        f"(hit {hits_after_cold} blocks)."
    )
    max_hit = max(harness.cache_hits, default=0)
    assert max_hit >= 1, (
        "Expected the repeated prompt to reuse >=1 cached prefix block, "
        "but no prefix-cache hit was observed."
    )
