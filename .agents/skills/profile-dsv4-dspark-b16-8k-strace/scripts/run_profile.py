# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the fixed DeepSeek V4 DSpark B16/8K/256 workload under serving profiling."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from pypto_serving import GenerateConfig
from pypto_serving.cli.main import build_parser, build_serving_engine_config
from pypto_serving.model.tokenizer import load_tokenizer
from pypto_serving.serving.engine.async_engine import AsyncLLMEngine
from pypto_serving.tools.profile import configure_profiler, merge_profile, start_profile, stop_profile


REQUEST_COUNT = 16
PROMPT_TOKENS = 8192
OUTPUT_TOKENS = 256
SPECULATIVE_TOKENS = 7
DEVICE_COUNT = 16
CHAT_PREFIX = "<｜begin▁of▁sentence｜><｜User｜>"
CHAT_SUFFIX = "<｜Assistant｜></think>"
# With the validated DeepSeek V4 tokenizer, the fixed wrappers contribute four
# tokens and each leading-space ASCII "a" contributes exactly one token.
ALIGNED_PROMPT = CHAT_PREFIX + " a" * (PROMPT_TOKENS - 4) + CHAT_SUFFIX


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--served-model-name", default="dsv4-flash-dspark-w8a8")
    parser.add_argument("--use-compile-cache", action="store_true")
    return parser.parse_args()


async def wait_for_engine_idle(engine: AsyncLLMEngine, timeout: float = 300.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        idle = all(
            not core._batch_queue
            and not core._pending_free_ids
            and not core.scheduler.has_work()
            for core in engine._cores
        )
        if idle and not engine._request_to_replica and engine.pending_token_load() == 0:
            return
        if loop.time() >= deadline:
            raise TimeoutError("engine did not become idle after the warmup batch")
        await asyncio.sleep(0.01)


def serialize_results(results) -> list[dict]:
    return [
        {
            "index": index,
            "text": result.text,
            "token_ids": list(result.token_ids),
            "finish_reason": result.finish_reason,
            "prompt_tokens": PROMPT_TOKENS,
            "completion_tokens": len(result.token_ids),
        }
        for index, result in enumerate(results)
    ]


def validate_results(results: list[dict], label: str) -> None:
    if len(results) != REQUEST_COUNT:
        raise RuntimeError(f"{label} returned {len(results)} requests")
    lengths = [item["completion_tokens"] for item in results]
    if lengths != [OUTPUT_TOKENS] * REQUEST_COUNT:
        raise RuntimeError(f"{label} output lengths are not aligned: {lengths}")
    errors = [item["index"] for item in results if item["finish_reason"] == "error"]
    if errors:
        raise RuntimeError(f"{label} requests failed: {errors}")


async def run(args: argparse.Namespace) -> None:
    artifact_dir = args.artifact_dir.resolve()
    model_dir = args.model_dir.resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {model_dir}")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if (
        len(devices) != DEVICE_COUNT
        or len(set(devices)) != DEVICE_COUNT
        or any(not item.isdigit() for item in devices)
    ):
        raise ValueError(
            f"--devices must contain exactly {DEVICE_COUNT} unique device IDs: "
            f"{args.devices!r}"
        )

    tokenizer = load_tokenizer(model_dir)
    prompt_ids = tokenizer.encode(ALIGNED_PROMPT)
    if len(prompt_ids) != PROMPT_TOKENS:
        raise RuntimeError(
            f"aligned prompt token mismatch: expected {PROMPT_TOKENS}, got {len(prompt_ids)}"
        )
    prompts = [ALIGNED_PROMPT] * REQUEST_COUNT

    cli_args = [
        "--model", str(model_dir),
        "--served-model-name", args.served_model_name,
        "--backend", "npu",
        "--platform", "a2a3",
        "--devices", ",".join(devices),
        "--dp", "4",
        "--ep", "16",
        "--tp", "4",
        "--block-size", "32",
        "--max-model-len", "16384",
        "--max-num-seqs", "16",
        "--max-num-batched-tokens", "8192",
        "--long-prefill-token-threshold", "128",
        "--speculative-config", '{"method":"dspark","num_speculative_tokens":7}',
        "--generate-config",
        '{"max_new_tokens":256,"temperature":0,"top_p":1,"top_k":null,'
        '"stream":false,"ignore_eos":true}',
        "--no-enable-prefix-caching",
        "--enable-chunked-prefill",
        "--ring-heap", "2147483648,2147483648,4294967296,8589934592",
        "--profile",
        "--profile-output", str(artifact_dir / "serving-trace"),
        "--profile-level", "verbose",
    ]
    if args.use_compile_cache:
        cli_args.append("--use-compile-cache")

    parsed = build_parser().parse_args(cli_args)
    engine_config = build_serving_engine_config(parsed)
    if not engine_config.profile_config.enabled:
        raise RuntimeError("verbose serving profiler was not enabled")
    configure_profiler(
        engine_config.profile_config,
        process_name="pypto-dspark-b16-8k-profile-controller",
        initially_active=False,
    )

    generate_config = GenerateConfig(
        max_new_tokens=OUTPUT_TOKENS,
        temperature=0.0,
        top_p=1.0,
        top_k=None,
        stream=False,
        ignore_eos=True,
    )
    engine = AsyncLLMEngine(config=engine_config, tokenizer=tokenizer)
    started = False
    try:
        await engine.start()
        started = True

        warmup = serialize_results(await engine.generate_batch(prompts, generate_config))
        validate_results(warmup, "warmup")
        (artifact_dir / "warmup-responses.json").write_text(
            json.dumps(warmup, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        await wait_for_engine_idle(engine)
        print("Unprofiled DSpark B16/8K/256 warmup completed", flush=True)

        controller_started = start_profile()
        if not controller_started:
            raise RuntimeError("failed to start the offline controller profiler")
        workers_started = False
        try:
            await engine.start_profile()
            workers_started = True
            batch_started = time.perf_counter()
            results = await engine.generate_batch(prompts, generate_config)
            elapsed = time.perf_counter() - batch_started
        finally:
            stop_error = None
            if workers_started:
                try:
                    await engine.stop_profile()
                except BaseException as exc:  # preserve the original failure if one exists
                    stop_error = exc
            stop_profile()
            merged_events = merge_profile()
            print(f"PROFILE_MERGED_EVENTS={merged_events}", flush=True)
            if stop_error is not None:
                raise stop_error

        serialized = serialize_results(results)
        validate_results(serialized, "profiled")
        (artifact_dir / "responses.json").write_text(
            json.dumps(serialized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        lengths = [item["completion_tokens"] for item in serialized]
        total_tokens = sum(lengths)
        summary = {
            "batch_elapsed_seconds": elapsed,
            "request_count": REQUEST_COUNT,
            "prompt_tokens_per_request": PROMPT_TOKENS,
            "tokens_per_request": lengths,
            "total_output_tokens": total_tokens,
            "throughput_tokens_per_second": total_tokens / elapsed,
            "effective_tpot_ms": elapsed * 1000.0 / OUTPUT_TOKENS,
            "profiler_enabled": True,
            "profile_level": "verbose",
            "speculative_method": "dspark",
            "num_speculative_tokens": SPECULATIVE_TOKENS,
        }
        (artifact_dir / "performance_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    finally:
        if started:
            await engine.stop()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
