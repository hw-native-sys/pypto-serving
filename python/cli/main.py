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
import contextlib
import os
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.async_engine import EngineConfig

from python.profile import get_profiler, merge_profile

RuntimeConfig = None


_VALID_BACKENDS = {"npu"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pypto-serving",
        description="Start PyPTO Serving with an OpenAI-compatible API.",
    )

    # Model
    parser.add_argument("--model", required=True, help="Path to the model directory.")
    parser.add_argument("--served-model-name", default=None, help="Model name used in the API. Defaults to the model directory name.")

    # Backend and device
    parser.add_argument("--backend", default="npu", choices=sorted(_VALID_BACKENDS), help="Inference backend (default: npu).")
    parser.add_argument("--platform", default="a2a3", help="NPU platform (default: a2a3).")
    parser.add_argument("--device", type=int, default=0, help="NPU device ID (default: 0).")
    # Dtype
    parser.add_argument("--dtype", default="float32", help="Weight data type (default: float32).")
    parser.add_argument("--kv-cache-dtype", default="bfloat16", help="KV cache data type. 'auto' follows --dtype (default: bfloat16).")

    # Runtime
    parser.add_argument("--max-model-len", type=int, default=512, help="Maximum sequence length (default: 512).")
    parser.add_argument("--block-size", type=int, default=128, help="KV cache block size (default: 128).")

    # Generation
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Maximum new tokens to generate (default: 32).")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default: 0.0).")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling probability (default: 1.0).")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling cutoff (default: disabled).")

    # Serving
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the serving server (default: 0.0.0.0).")
    parser.add_argument("--port", type=int, default=8000, help="Port for the serving server (default: 8000).")
    parser.add_argument("--max-num-seqs", type=int, default=16, help="Max concurrent requests in serving mode (default: 32).")
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096, help="Max tokens scheduled per iteration (default: 4096).")
    parser.add_argument(
        "--long-prefill-token-threshold",
        type=int,
        default=2048,
        help="Chunked prefill threshold in serving mode (default: 2048).",
    )
    parser.add_argument(
        "--enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable prefix caching (default: True). Use --no-enable-prefix-caching to disable.",
    )
    parser.add_argument(
        "--prefix-cache-backend",
        default="hash",
        choices=("hash", "radix"),
        help="Prefix cache backend: hash or radix (default: hash).",
    )
    parser.add_argument(
        "--enable-chunked-prefill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable chunked prefill (default: True). Use --no-enable-chunked-prefill to disable.",
    )

    # Misc
    parser.add_argument(
        "--show-startup-logs",
        action="store_true",
        help="Show model loading and kernel compilation logs. Startup logs are suppressed by default.",
    )
    return parser


def build_serving_engine_config(args: argparse.Namespace) -> EngineConfig:
    _ensure_core_imports()
    _validate_backend(args.backend)

    try:
        from ..core.async_engine import EngineConfig
    except ImportError:
        from python.core.async_engine import EngineConfig

    model_dir = str(Path(args.model).resolve())
    executor_kwargs = _build_executor_kwargs()

    return EngineConfig(
        model_id=args.served_model_name or Path(args.model).name,
        model_dir=model_dir,
        platform=args.platform,
        device_id=args.device,
        executor_cls="PyptoQwen14BExecutor",
        executor_kwargs=executor_kwargs,
        runtime_config=_build_runtime_config(args),
        max_num_running_reqs=args.max_num_seqs,
        max_num_scheduled_tokens=args.max_num_batched_tokens,
        long_prefill_token_threshold=args.long_prefill_token_threshold,
        enable_prefix_cache=args.enable_prefix_caching,
        enable_chunk_prefill=args.enable_chunked_prefill,
        prefix_cache_backend=args.prefix_cache_backend,
    )


def _build_runtime_config(args: argparse.Namespace):
    kv_dtype = args.kv_cache_dtype
    if kv_dtype == "auto":
        kv_dtype = args.dtype

    return RuntimeConfig(
        page_size=args.block_size,
        max_batch_size=args.max_num_seqs,
        max_seq_len=args.max_model_len,
        device="cpu",
        kv_dtype=kv_dtype,
        weight_dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
    )


def _build_executor_kwargs() -> dict[str, object]:
    executor_kwargs: dict[str, object] = {}
    pypto_root = os.environ.get("PYPTO_ROOT")
    save_kernels_dir = os.environ.get("PYPTO_SAVE_KERNELS_DIR")
    if pypto_root:
        executor_kwargs["pypto_root"] = pypto_root
    if save_kernels_dir:
        executor_kwargs["save_kernels_dir"] = save_kernels_dir
    return executor_kwargs


def run_serve(
    config: EngineConfig,
    *,
    host: str = "0.0.0.0",
    port: int = 8000,
) -> None:
    try:
        import uvicorn
    except ImportError as e:
        raise ImportError("Serving mode requires uvicorn. Install with: pip install uvicorn") from e

    from ..core.async_engine import AsyncLLMEngine
    from ..core.server import create_serving_app
    from ..core.tokenizer import TransformersTokenizerAdapter

    model_id = config.model_id
    get_profiler(process_name="pypto-serving-api")
    tokenizer = TransformersTokenizerAdapter.from_pretrained(config.model_dir)

    async_engine = AsyncLLMEngine(
        config=config,
        tokenizer=tokenizer,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )

    app = create_serving_app(async_engine, model_id)

    @app.on_event("startup")
    async def startup():
        await async_engine.start()

    @app.on_event("shutdown")
    async def shutdown():
        await async_engine.stop()
        merge_profile()

    print(f"Starting PyPTO serving on {host}:{port}")
    print(f"  Model: {model_id} (loaded in worker process)")
    print(f"  Platform: {config.platform}, Device: {config.device_id}")
    print(f"  Max running requests: {config.max_num_running_reqs}")
    print(f"  Max scheduled tokens/iter: {config.max_num_scheduled_tokens}")
    print(f"  Chunked prefill threshold: {config.long_prefill_token_threshold}")
    print(f"  Prefix cache: {'enabled' if config.enable_prefix_cache else 'disabled'}")
    print(f"  Prefix cache backend: {config.prefix_cache_backend}")
    print(f"  Chunk prefill: {'enabled' if config.enable_chunk_prefill else 'disabled'}")
    print("  Endpoints: /v1/completions, /v1/chat/completions, /v1/models, /health")

    uvicorn.run(app, host=host, port=port, log_level="info")


def _validate_backend(backend: str) -> None:
    if backend != "npu":
        raise ValueError(f"Only NPU backend is supported, got: {backend}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    get_profiler(process_name="pypto-serving")

    with _startup_log_context(enabled=not args.show_startup_logs):
        config = build_serving_engine_config(args)

    run_serve(
        config,
        host=args.host,
        port=args.port,
    )
    return 0


def _ensure_core_imports() -> None:
    global RuntimeConfig

    if RuntimeConfig is None:
        try:
            from ..core import RuntimeConfig as imported_runtime_config
        except ImportError:
            from python.core import RuntimeConfig as imported_runtime_config

        RuntimeConfig = imported_runtime_config


@contextlib.contextmanager
def _startup_log_context(*, enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return

    old_log_level = os.environ.get("PTO_LOG_LEVEL")
    os.environ.setdefault("PTO_LOG_LEVEL", "error")
    sys.stdout.flush()
    sys.stderr.flush()
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)
        if old_log_level is None:
            os.environ.pop("PTO_LOG_LEVEL", None)
        else:
            os.environ["PTO_LOG_LEVEL"] = old_log_level


if __name__ == "__main__":
    raise SystemExit(main())
