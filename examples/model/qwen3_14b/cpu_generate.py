# Copyright (c) PyPTO Contributors.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU-only Qwen3-14B generation (unified entry point).

Runs full-precision (FP) by default through the reference executor
(`CpuModelExecutor`). Pass ``--tq`` to enable TurboQuant KV cache
compression (`CpuTqModelExecutor`), or ``--compare`` to run both and
print side-by-side output.

Usage:
    # Single prompt (FP, default)
    python cpu_generate.py \\
        --model-dir /path/to/Qwen3-14B \\
        --prompt "你好，请介绍一下你自己" \\
        --max-new-tokens 128

    # Interactive chat
    python cpu_generate.py \\
        --model-dir /path/to/Qwen3-14B \\
        --interactive

    # TurboQuant KV cache compression
    python cpu_generate.py \\
        --model-dir /path/to/Qwen3-14B \\
        --prompt "What is machine learning?" --tq

    # Compare TQ vs FP
    python cpu_generate.py \\
        --model-dir /path/to/Qwen3-14B \\
        --prompt "What is machine learning?" \\
        --compare
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _bootstrap_package_root() -> None:
    this_file = Path(__file__).resolve()
    for candidate in (this_file, *this_file.parents):
        if (candidate / "python" / "core").is_dir() and (candidate / "examples" / "model" / "qwen3_14b" / "runner").is_dir():
            repo_root = str(candidate)
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            return
    raise RuntimeError(f"Unable to locate the pypto-serving repo root from {this_file}")


_bootstrap_package_root()

from python.core import GenerateConfig, LLMEngine, RuntimeConfig
from python.core.kv_cache import KvCacheManager
from python.core.types import KvQuantConfig
from examples.model.qwen3_14b.runner.cpu_executor import CpuTqModelExecutor, CpuModelExecutor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CPU-only Qwen3-14B generation with TurboQuant KV cache compression.",
    )
    parser.add_argument("--model-dir", required=True, help="Local model directory (HuggingFace safetensors).")
    parser.add_argument("--prompt", default=None, help="Prompt text. Omit for interactive mode.")
    parser.add_argument("--interactive", action="store_true", help="Enter interactive chat mode.")
    parser.add_argument("--model-id", default="qwen3-14b-tq-cpu")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--stream", action="store_true", default=False)
    parser.add_argument("--tq", action="store_true",
                        help="Enable TurboQuant KV cache compression.")
    parser.add_argument("--compare", action="store_true",
                        help="Run both TQ and FP, print side-by-side comparison.")
    parser.add_argument("--dump-dir", default=None,
                        help="Directory to save per-layer intermediate tensors for debugging.")
    parser.add_argument("--num-layers-override", type=int, default=None,
                        help="Only run first N transformer layers.")
    return parser


def run_single(
    engine: LLMEngine,
    model_id: str,
    prompt: str,
    config: GenerateConfig,
) -> None:
    """Run a single prompt and print the result."""
    t0 = time.perf_counter()
    result = engine.generate_result(model_id, prompt, config)
    dt = time.perf_counter() - t0
    n_tok = len(result.token_ids)
    print(f"\n{'=' * 60}")
    print(f"Response ({n_tok} tokens, {dt:.2f}s, {n_tok / dt:.1f} tok/s):")
    print(f"{'=' * 60}")
    print(result.text)
    print(f"{'=' * 60}")
    print(f"token_ids: {result.token_ids}")
    print(f"finish_reason: {result.finish_reason}")


def run_interactive(
    engine: LLMEngine,
    model_id: str,
    config: GenerateConfig,
) -> None:
    """Interactive chat loop."""
    print("\n" + "=" * 60)
    print("Qwen3-14B CPU Chat with TurboQuant (type 'quit' to exit)")
    print("=" * 60)

    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not prompt or prompt.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        t0 = time.perf_counter()
        result = engine.generate_result(model_id, prompt, config)
        dt = time.perf_counter() - t0
        n_tok = len(result.token_ids)

        print(f"\nAssistant ({n_tok} tokens, {dt:.2f}s):")
        print(result.text)


def run_compare(
    engine_tq: LLMEngine,
    engine_fp: LLMEngine,
    model_id: str,
    prompt: str,
    config: GenerateConfig,
) -> None:
    """Run same prompt on both TQ and FP, print comparison."""
    print(f"\nPrompt: {prompt}")
    print("-" * 60)

    t0 = time.perf_counter()
    result_tq = engine_tq.generate_result(model_id, prompt, config)
    dt_tq = time.perf_counter() - t0

    t0 = time.perf_counter()
    result_fp = engine_fp.generate_result(model_id, prompt, config)
    dt_fp = time.perf_counter() - t0

    print(f"\n[TQ] ({len(result_tq.token_ids)} tokens, {dt_tq:.2f}s):")
    print(result_tq.text)
    print(f"\n[FP] ({len(result_fp.token_ids)} tokens, {dt_fp:.2f}s):")
    print(result_fp.text)

    # Token overlap
    common = len(set(result_tq.token_ids) & set(result_fp.token_ids))
    total = max(len(result_tq.token_ids), len(result_fp.token_ids))
    print(f"\nToken overlap: {common}/{total} ({common / total * 100:.1f}%)")


def create_engine(
    model_dir: str,
    model_id: str,
    max_seq_len: int,
    max_new_tokens: int,
    use_tq: bool = True,
    dump_dir: str | None = None,
    num_layers_override: int | None = None,
) -> LLMEngine:
    """Create an LLMEngine with the appropriate executor."""
    kv_cache_manager = KvCacheManager()

    if use_tq:
        executor = CpuTqModelExecutor(kv_cache_manager, dump_dir=dump_dir, num_layers_override=num_layers_override)
    else:
        executor = CpuModelExecutor(kv_cache_manager, dump_dir=dump_dir)

    kv_quant_config = None
    if use_tq:
        kv_quant_config = KvQuantConfig(
            enabled=True,
            key_bits=4,
            value_bits=4,
        )

    engine = LLMEngine(
        kv_cache_manager=kv_cache_manager,
        executor=executor,
    )

    engine.init_model(
        model_id=model_id,
        model_dir=str(Path(model_dir).resolve()),
        model_format="huggingface",
        runtime_config=RuntimeConfig(
            page_size=128,
            max_batch_size=1,
            max_seq_len=max_seq_len,
            max_new_tokens=max_new_tokens,
            device="cpu",
            kv_dtype="bfloat16",
            weight_dtype="float32",
            kv_quant_config=kv_quant_config,
        ),
    )

    return engine


def main() -> None:
    args = build_parser().parse_args()
    model_dir = Path(args.model_dir).resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    config = GenerateConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        stream=args.stream,
    )

    if args.compare:
        # Run both TQ and FP
        dump_base = args.dump_dir
        tq_dump = f"{dump_base}/tq" if dump_base else None
        fp_dump = f"{dump_base}/fp" if dump_base else None

        print("Loading model for TQ ...")
        engine_tq = create_engine(args.model_dir, args.model_id,
                                 args.max_seq_len, args.max_new_tokens, use_tq=True,
                                 dump_dir=tq_dump,
                                 num_layers_override=args.num_layers_override)
        print("Loading model for FP ...")
        engine_fp = create_engine(args.model_dir, "qwen3-14b-fp-cpu",
                                  args.max_seq_len, args.max_new_tokens, use_tq=False,
                                  dump_dir=fp_dump,
                                  num_layers_override=args.num_layers_override)

        prompt = args.prompt or "Hello, how are you?"
        run_compare(engine_tq, engine_fp, args.model_id, prompt, config)
        return

    # Single engine mode
    use_tq = args.tq
    label = "TQ" if use_tq else "FP"
    print(f"Loading model ({label}) ...")
    engine = create_engine(args.model_dir, args.model_id,
                           args.max_seq_len, args.max_new_tokens, use_tq=use_tq,
                           dump_dir=args.dump_dir,
                           num_layers_override=args.num_layers_override)
    print(f"Model loaded ({label}).")

    if args.interactive or args.prompt is None:
        run_interactive(engine, args.model_id, config)
    else:
        run_single(engine, args.model_id, args.prompt, config)


if __name__ == "__main__":
    main()
