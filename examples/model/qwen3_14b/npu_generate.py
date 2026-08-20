# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import argparse
import dataclasses
import statistics
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

from pypto_serving import GenerateConfig, LLMEngine, RuntimeConfig
from pypto_serving.config.parallel import ParallelConfig, parse_device_ids
from pypto_serving.config.types import LoadedModel
from pypto_serving.model.qwen.npu_executor import Qwen314BPyptoExecutor as PyptoExecutor
from pypto_serving.serving.memory.kv_cache import KvCacheManager
from pypto_serving.tools.profile import get_profiler, merge_profile, profile_span


# -----------------------------------------------------------------------------
# Profiling helpers (3-level: phase / executor API / per-layer kernel)
# -----------------------------------------------------------------------------


class _TimingCollector:
    """Collects timings at three levels and groups per-layer kernel calls per
    decode step so the report can show per-step breakdowns.
    """

    def __init__(self) -> None:
        # name -> total seconds (additive across calls)
        self.phases: dict[str, float] = {}
        # kernel_name -> flat list of seconds (one entry per kernel invocation)
        self.kernel_times: dict[str, list[float]] = defaultdict(list)
        self.kernel_host_wall_times: dict[str, list[float]] = defaultdict(list)
        self.kernel_device_wall_times: dict[str, list[float]] = defaultdict(list)
        # kernel_name -> list[list[float]]: outer = decode step, inner = layer
        self.kernel_per_decode_step: dict[str, list[list[float]]] = defaultdict(list)
        # Bumped by BeginDecodeStep before each run_decode invocation.
        self._decode_step_idx: int = -1

    @contextmanager
    def TimePhase(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.phases[name] = self.phases.get(name, 0.0) + (time.perf_counter() - t0)

    def WrapKernel(self, fn, name: str, *, group_by_decode_step: bool = False):
        """Return a wrapper that records every call's duration under `name`."""

        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                dt = time.perf_counter() - t0
                self.kernel_times[name].append(dt)
                if group_by_decode_step and self._decode_step_idx >= 0:
                    bucket = self.kernel_per_decode_step[name]
                    while len(bucket) <= self._decode_step_idx:
                        bucket.append([])
                    bucket[self._decode_step_idx].append(dt)

        return wrapper

    def BeginDecodeStep(self) -> None:
        self._decode_step_idx += 1

    def RecordRunTiming(self, name: str, timing: object) -> None:
        host_wall_us = getattr(timing, "host_wall_us", None)
        device_wall_us = getattr(timing, "device_wall_us", None)
        if host_wall_us is not None:
            self.kernel_host_wall_times[name].append(float(host_wall_us) / 1_000_000.0)
        if device_wall_us is not None:
            self.kernel_device_wall_times[name].append(float(device_wall_us) / 1_000_000.0)

    @property
    def num_decode_steps(self) -> int:
        return self._decode_step_idx + 1


def _install_num_layers_override(engine: LLMEngine, n: int) -> None:
    """Monkey-patch the loader to truncate the model to N layers post-load.

    Validation-only knob. Must be installed BEFORE engine.init_model so the
    truncation lands before the executor compiles kernels for the wrong layer
    count (decode_stacked program is parameterised on num_layers).
    """
    if n <= 0:
        raise ValueError(f"--num-layers-override must be positive, got {n}")
    orig_load = engine._model_loader.load

    def load_truncated(*args, **kwargs):
        loaded = orig_load(*args, **kwargs)
        if n >= loaded.config.num_hidden_layers:
            return loaded
        new_config = dataclasses.replace(loaded.config, num_hidden_layers=n)
        # The layer count is the whole override now: staging reads `config.num_hidden_layers`
        # and pulls exactly that many layers from the checkpoint, so there is no eager
        # `runtime_model.layers` list left to truncate.
        loaded.runtime_model.config = new_config
        print(
            f"[override] num_hidden_layers: "
            f"{loaded.config.num_hidden_layers} -> {n}",
            flush=True,
        )
        return LoadedModel(
            model_id=loaded.model_id,
            model_dir=loaded.model_dir,
            config=new_config,
            tokenizer=loaded.tokenizer,
            layer_specs=loaded.layer_specs[:n],
            runtime_model=loaded.runtime_model,
        )

    engine._model_loader.load = load_truncated


def InstallProfiling(engine: LLMEngine, model_id: str, collector: _TimingCollector) -> None:
    """Wrap executor.run_prefill / run_decode and compiled kernels so
    timings flow into `collector`. Must run AFTER engine.init_model().
    """
    executor = engine._executor  # type: ignore[attr-defined]
    compiled = executor._compiled[model_id]  # type: ignore[attr-defined]

    # Kernels are dispatched by Qwen314BModelRunner.
    if hasattr(compiled.prefill, "chip_callable") or hasattr(compiled.prefill, "compiled"):
        runner = executor._runners[model_id]  # type: ignore[attr-defined]
        orig_run_program = runner._run_distributed_program  # type: ignore[attr-defined]
        kernel_names = {
            id(compiled.prefill): ("kernel.prefill_fwd", False),
            id(compiled.decode): ("kernel.decode_layer", True),
        }

        def timed_run_program(callable_spec, *args, **kwargs):
            kernel_info = kernel_names.get(id(callable_spec))
            if kernel_info is None:
                return orig_run_program(callable_spec, *args, **kwargs)
            name, group_by_decode_step = kernel_info
            t0 = time.perf_counter()
            try:
                timing = orig_run_program(callable_spec, *args, **kwargs)
            finally:
                dt = time.perf_counter() - t0
                collector.kernel_times[name].append(dt)
                if group_by_decode_step and collector._decode_step_idx >= 0:
                    bucket = collector.kernel_per_decode_step[name]
                    while len(bucket) <= collector._decode_step_idx:
                        bucket.append([])
                    bucket[collector._decode_step_idx].append(dt)
            collector.RecordRunTiming(name, timing)
            return timing

        runner._run_distributed_program = timed_run_program  # type: ignore[attr-defined]
    else:
        # Per-layer kernel wrappers. compiled.prefill / compiled.decode are invoked
        # once per transformer layer inside run_prefill / run_decode respectively.
        compiled.prefill = collector.WrapKernel(compiled.prefill, "kernel.prefill_fwd")
        compiled.decode = collector.WrapKernel(
            compiled.decode, "kernel.decode_layer", group_by_decode_step=True
        )

    # Top-level executor API wrappers.
    orig_prefill = executor.run_prefill
    orig_decode = executor.run_decode

    def timed_prefill(*args, **kwargs):
        with collector.TimePhase("api.run_prefill"):
            return orig_prefill(*args, **kwargs)

    def timed_decode(*args, **kwargs):
        collector.BeginDecodeStep()
        with collector.TimePhase("api.run_decode"):
            return orig_decode(*args, **kwargs)

    executor.run_prefill = timed_prefill
    executor.run_decode = timed_decode


def SummarizeTimes(times: list[float]) -> str:
    if not times:
        return "n=0"
    total_ms = sum(times) * 1000
    avg_ms = statistics.mean(times) * 1000
    min_ms = min(times) * 1000
    max_ms = max(times) * 1000
    return (
        f"n={len(times):4d}  total={total_ms:9.1f}ms  "
        f"avg={avg_ms:7.2f}ms  min={min_ms:7.2f}ms  max={max_ms:7.2f}ms"
    )


def PrintTimingReport(collector: _TimingCollector, num_tokens: int, verbose: bool) -> None:
    print("\n" + "=" * 70)
    print("=== Timing Report ===")
    print("=" * 70)

    # Level 1: phases
    init = collector.phases.get("init_model", 0.0)
    gen = collector.phases.get("generate_total", 0.0)
    print(f"[phase] init_model           : {init:8.2f}s   (weight load + kernel compile)")
    print(f"[phase] generate (e2e)       : {gen:8.2f}s   ({num_tokens} tokens generated)")
    if num_tokens > 0 and gen > 0:
        print(f"[phase] throughput (e2e)     : {num_tokens / gen:8.2f} tok/s")

    # Level 2: executor API
    api_prefill = collector.phases.get("api.run_prefill", 0.0)
    api_decode = collector.phases.get("api.run_decode", 0.0)
    n_decode = collector.num_decode_steps
    print(f"[api]   run_prefill          : {api_prefill:8.2f}s   (1 call, TTFT-ish)")
    if n_decode > 0:
        print(
            f"[api]   run_decode total     : {api_decode:8.2f}s   "
            f"({n_decode} steps, avg {api_decode / n_decode * 1000:7.1f} ms/step, "
            f"{n_decode / api_decode:6.2f} step/s)"
        )
    else:
        print(f"[api]   run_decode total     : {api_decode:8.2f}s   (0 steps)")

    # Level 3: per-kernel aggregate
    print()
    print("[kernel] aggregate over all invocations:")
    for kname in (
        "kernel.prefill_fwd",
        "kernel.decode_layer",
    ):
        print(f"  {kname:24s}: {SummarizeTimes(collector.kernel_times.get(kname, []))}")
        host_times = collector.kernel_host_wall_times.get(kname, [])
        device_times = collector.kernel_device_wall_times.get(kname, [])
        if host_times or device_times:
            print(f"  {'  host_wall':24s}: {SummarizeTimes(host_times)}")
            print(f"  {'  device_wall':24s}: {SummarizeTimes(device_times)}")

    if not verbose:
        print(
            "\n(use --profile-verbose for per-layer prefill times and per-step decode "
            "layer breakdowns)"
        )
        return

    # Verbose: per-layer prefill times (single prefill call -> N layer kernels)
    prefill_layers = collector.kernel_times.get("kernel.prefill_fwd", [])
    if prefill_layers:
        print("\n--- per-layer prefill kernel times (single run_prefill) ---")
        for layer_idx, t in enumerate(prefill_layers):
            print(f"  layer {layer_idx:02d}: {t * 1000:8.2f} ms")

    # Verbose: per-step summary for decode (each step has N layer kernel calls)
    per_step = collector.kernel_per_decode_step.get("kernel.decode_layer", [])
    if per_step:
        print("\n--- per-decode-step layer kernel breakdown ---")
        print(f"{'step':>5}  {'layers':>6}  {'total(ms)':>10}  {'avg/layer(ms)':>14}  "
              f"{'min(ms)':>9}  {'max(ms)':>9}")
        for step_idx, layer_times in enumerate(per_step):
            if not layer_times:
                continue
            total_ms = sum(layer_times) * 1000
            avg_ms = statistics.mean(layer_times) * 1000
            min_ms = min(layer_times) * 1000
            max_ms = max(layer_times) * 1000
            print(
                f"{step_idx:>5d}  {len(layer_times):>6d}  {total_ms:>10.2f}  "
                f"{avg_ms:>14.2f}  {min_ms:>9.2f}  {max_ms:>9.2f}"
            )


def InstallThroughputMeter(engine: LLMEngine) -> dict:
    """Wrap executor.run_prefill / run_decode with always-on wall-clock meters.

    Unlike InstallProfiling this touches only the two public dispatch methods
    (no profiling internals), so it is safe to install unconditionally and feeds
    the token/s summary printed for every run.
    """
    executor = engine._executor  # type: ignore[attr-defined]
    meters = {"prefill_s": 0.0, "decode_s": 0.0, "decode_steps": 0}
    orig_prefill = executor.run_prefill
    orig_decode = executor.run_decode

    def metered_prefill(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return orig_prefill(*args, **kwargs)
        finally:
            meters["prefill_s"] += time.perf_counter() - t0

    def metered_decode(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return orig_decode(*args, **kwargs)
        finally:
            meters["decode_s"] += time.perf_counter() - t0
            meters["decode_steps"] += 1

    executor.run_prefill = metered_prefill
    executor.run_decode = metered_decode
    return meters


def PrintThroughput(num_tokens: int, gen_elapsed: float, meters: dict) -> None:
    """Print a concise token/s summary (always shown; no --profile needed)."""
    print("\n=== throughput ===")
    if num_tokens > 0 and gen_elapsed > 0:
        print(
            f"[perf] generated {num_tokens} tokens in {gen_elapsed:.3f}s "
            f"-> {num_tokens / gen_elapsed:.2f} tok/s (overall, incl. prefill)"
        )
    else:
        print(f"[perf] generated {num_tokens} tokens in {gen_elapsed:.3f}s")
    steps = int(meters.get("decode_steps", 0))
    decode_s = float(meters.get("decode_s", 0.0))
    prefill_s = float(meters.get("prefill_s", 0.0))
    if steps > 0 and decode_s > 0:
        print(
            f"[perf] prefill/TTFT {prefill_s:.3f}s | decode {decode_s:.3f}s over "
            f"{steps} steps -> {steps / decode_s:.2f} tok/s "
            f"({decode_s / steps * 1000:.1f} ms/token)"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local Qwen3-14B generation with the bundled PyPTO kernels.")
    parser.add_argument("--model-dir", required=True, help="Local model directory, e.g. a Hugging Face snapshot.")
    parser.add_argument("--prompt", required=True, help="Prompt text.")
    parser.add_argument("--model-id", default="qwen3-14b-local")
    parser.add_argument("--platform", default="a2a3", choices=["a2a3sim", "a2a3", "a5sim", "a5"])
    parser.add_argument("--device-id", type=int, default=0, help="Default NPU device id when --devices is unset.")
    parser.add_argument(
        "--devices",
        default=None,
        help="Comma-separated NPU device ids for one tensor-parallel L3 worker group. Overrides --device-id.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        "--tp",
        type=int,
        default=1,
        help="Tensor-parallel group size.",
    )
    parser.add_argument(
        "--data-parallel-size",
        "--dp",
        type=int,
        default=1,
        help="Offline generation does not launch DP replicas; values > 1 fail fast.",
    )
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-num-seqs", type=int, default=16, help="Max batch size / concurrent requests.")
    parser.add_argument("--num-prompts", type=int, default=1,
                        help="Number of prompts to generate (replicates --prompt). When > 1 calls "
                             "generate_batch for real batched inference. Default 1 = single request.")
    parser.add_argument("--block-size", type=int, default=128, help="KV cache page size.")
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=4096,
        help="Total tokens per scheduling step (used by warmup and chunked prefill).",
    )
    parser.add_argument(
        "--npu-memory-utilization",
        type=float,
        default=0.90,
        help="Fraction of total NPU HBM the server is allowed to use (weights + KV).",
    )
    parser.add_argument("--dtype", default="bfloat16", help="Weight data type.")
    parser.add_argument("--kv-cache-dtype", default="bfloat16", help="KV cache data type.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--stream", action="store_true", default=False)
    parser.add_argument("--save-kernels-dir", default=None)
    parser.add_argument(
        "--num-layers-override",
        type=int,
        default=None,
        help="Validation knob: truncate the loaded model to N transformer "
             "layers before compile/dispatch. Used to reduce HBM footprint "
             "while validating Qwen3-14B kernels on memory-constrained devices.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print phase / executor-API / per-kernel timing summary at the end.",
    )
    parser.add_argument(
        "--profile-verbose",
        action="store_true",
        help="Implies --profile. Also dump per-layer prefill times and "
             "per-decode-step layer breakdowns.",
    )
    parser.add_argument(
        "--no-enable-prefix-caching",
        action="store_true",
        dest="no_enable_prefix_caching",
        help="Disable prefix caching in the KV cache manager.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for _n in ("simpler_setup", "pypto", "simpler"):
        logging.getLogger(_n).setLevel(logging.WARNING)
    get_profiler(process_name="npu_generate")
    model_dir = Path(args.model_dir).resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    profile_enabled = args.profile or args.profile_verbose
    collector = _TimingCollector() if profile_enabled else None
    parallel_config = ParallelConfig(
        data_parallel_size=args.data_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
        devices=parse_device_ids(args.devices, default_device=args.device_id),
    )
    if parallel_config.data_parallel_size != 1:
        raise ValueError(
            "offline npu_generate.py uses the single-process LLMEngine and does not launch "
            "serving replicas; data_parallel_size must be 1"
        )
    device_ids = parallel_config.replica_device_groups[0]

    kv_cache_manager = KvCacheManager(
        enable_prefix_cache=not args.no_enable_prefix_caching,
    )
    executor = PyptoExecutor(
        kv_cache_manager,
        platform=args.platform,
        device_ids=device_ids,
        save_kernels_dir=args.save_kernels_dir,
    )
    engine = LLMEngine(
        kv_cache_manager=kv_cache_manager,
        executor=executor,
    )

    if args.num_layers_override is not None:
        _install_num_layers_override(engine, args.num_layers_override)

    try:
        init_t0 = time.perf_counter()
        engine.init_model(
            model_id=args.model_id,
            model_dir=str(model_dir),
            model_format="huggingface",
            runtime_config=RuntimeConfig(
                page_size=args.block_size,
                max_batch_size=args.max_num_seqs,
                max_seq_len=args.max_seq_len,
                max_new_tokens=args.max_new_tokens,
                device="cpu",
                kv_dtype=args.kv_cache_dtype,
                weight_dtype=args.dtype,
                npu_memory_utilization=args.npu_memory_utilization,
                max_num_batched_tokens=args.max_num_batched_tokens,
            ),
        )
        if collector is not None:
            collector.phases["init_model"] = time.perf_counter() - init_t0
            # Profiling must be installed AFTER init_model: register_model populates
            # executor._compiled[model_id] with the four kernel callables we wrap.
            InstallProfiling(engine, args.model_id, collector)

        config = GenerateConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            stream=args.stream,
        )

        # Always-on token/s meter (installed after init_model; independent of
        # --profile so every run prints a throughput summary).
        meters = InstallThroughputMeter(engine)

        num_tokens = 0
        gen_t0 = time.perf_counter()
        with profile_span(
            "npu_generate.generate",
            cat="request",
            args={
                "model_id": args.model_id,
                "max_new_tokens": args.max_new_tokens,
                "stream": args.stream,
            },
        ):
            if args.stream:
                text_parts: list[str] = []
                result = engine.generate(args.model_id, args.prompt, config)
                for chunk in result:
                    text_parts.append(chunk)
                    print(chunk, end="", flush=True)
                print()
                num_tokens = len(text_parts)
            else:
                if args.num_prompts > 1:
                    prompts = [args.prompt] * args.num_prompts
                    results = engine.generate_batch(args.model_id, prompts, config)
                    num_tokens = sum(len(r.token_ids) for r in results)
                    for i, r in enumerate(results):
                        print(f"-- prompt {i+1}/{args.num_prompts} --")
                        print(f"text: {r.text}")
                        print(f"token_ids: {r.token_ids[:64]}{'...' if len(r.token_ids)>64 else ''}")
                        print(f"finish_reason: {r.finish_reason}")
                else:
                    result = engine.generate_result(args.model_id, args.prompt, config)
                    num_tokens = len(result.token_ids)
                    print(f"text: {result.text}")
                    print(f"token_ids: {result.token_ids}")
                    print(f"finish_reason: {result.finish_reason}")

        gen_elapsed = time.perf_counter() - gen_t0
        PrintThroughput(num_tokens, gen_elapsed, meters)

        if collector is not None:
            collector.phases["generate_total"] = gen_elapsed
            PrintTimingReport(collector, num_tokens=num_tokens, verbose=args.profile_verbose)
    finally:
        try:
            executor.close()
        finally:
            merge_profile()


if __name__ == "__main__":
    main()
