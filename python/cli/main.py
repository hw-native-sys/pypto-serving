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
import json
import os
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.async_engine import EngineConfig

from python.profile import get_profiler, merge_profile

RuntimeConfig = None
ParallelConfig = None
parse_device_ids = None


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
    parser.add_argument(
        "--devices",
        default=None,
        help="Comma-separated NPU device ids for DP x TP placement, for example 0,1,2,3.",
    )
    parser.add_argument(
        "--data-parallel-size",
        "--dp",
        type=int,
        default=1,
        help="Data-parallel replica count.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        "--tp",
        type=int,
        default=1,
        help="Tensor-parallel group size.",
    )
    parser.add_argument(
        "--data-parallel-routing",
        default="least_pending_tokens",
        choices=["least_pending_tokens"],
        help="Data-parallel request routing policy.",
    )
    # Dtype
    parser.add_argument("--dtype", default="bfloat16", help="Weight data type (default: bfloat16).")
    parser.add_argument("--kv-cache-dtype", default="bfloat16", help="KV cache data type. 'auto' follows --dtype (default: bfloat16).")

    # Runtime
    parser.add_argument("--max-model-len", type=int, default=1024, help="Maximum sequence length (prompt + generated; default: 1024).")
    parser.add_argument("--block-size", type=int, default=128, help="KV cache block size (default: 128).")
    parser.add_argument(
        "--npu-memory-utilization",
        type=float,
        default=0.90,
        help="Fraction of total NPU HBM the server is allowed to use "
        "(weights + activations + KV cache). Default: 0.90.",
    )

    # Generation
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
    devices = parse_device_ids(args.devices, default_device=args.device)
    model_family = _detect_model_family(Path(model_dir))
    if model_family == "deepseek_v4":
        executor_kwargs["compile_kernels"] = True
    parallel_config = ParallelConfig(
        data_parallel_size=args.data_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
        devices=devices,
        data_parallel_routing=args.data_parallel_routing,
    )
    _validate_model_topology(model_family, args, parallel_config)
    first_group = parallel_config.replica_device_groups[0]
    worker_device_ids = first_group if parallel_config.data_parallel_size == 1 else ()
    enable_prefix_cache = args.enable_prefix_caching
    if model_family == "deepseek_v4":
        enable_prefix_cache = False

    return EngineConfig(
        model_id=args.served_model_name or Path(args.model).name,
        model_dir=model_dir,
        platform=args.platform,
        device_id=first_group[0],
        device_ids=worker_device_ids,
        parallel_config=parallel_config,
        executor_cls=_executor_cls_for_model_family(model_family),
        executor_kwargs=executor_kwargs,
        runtime_config=_build_runtime_config(args),
        max_num_running_reqs=args.max_num_seqs,
        max_num_scheduled_tokens=args.max_num_batched_tokens,
        long_prefill_token_threshold=args.long_prefill_token_threshold,
        enable_prefix_cache=enable_prefix_cache,
        enable_chunk_prefill=args.enable_chunked_prefill,
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
        npu_memory_utilization=args.npu_memory_utilization,
        max_num_batched_tokens=args.max_num_batched_tokens,
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


def _detect_model_family(model_dir: Path) -> str:
    """Return the serving model family inferred from config.json."""
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return "qwen"
    try:
        config_data = json.loads(config_path.read_text())
    except json.JSONDecodeError:
        return "qwen"
    model_type = str(config_data.get("model_type") or "").lower()
    architectures = {str(item).lower() for item in (config_data.get("architectures") or [])}
    if model_type == "deepseek_v4" or "deepseekv4forcausallm" in architectures:
        return "deepseek_v4"
    return "qwen"


def _executor_cls_for_model_family(model_family: str) -> str:
    """Map model family metadata to the worker executor class id."""
    if model_family == "deepseek_v4":
        return "PyptoDeepSeekV4Executor"
    return "PyptoQwen14BExecutor"


def _validate_model_topology(
    model_family: str,
    args: argparse.Namespace,
    parallel_config,
) -> None:
    """Validate model-specific serving topology constraints."""
    if model_family != "deepseek_v4":
        return
    config_data = json.loads((Path(args.model).resolve() / "config.json").read_text())
    quantization = config_data.get("quantization_config") or {}
    if quantization.get("quant_method") != "compressed-tensors":
        raise ValueError(
            "DeepSeekV4 serving requires the quantized W8A8 compressed-tensors checkpoint "
            "such as /data/models/dsv4-flash-w8a8; the original checkpoint is too large for 8 NPUs."
        )
    if parallel_config.data_parallel_size != 1 or parallel_config.tensor_parallel_size != 8:
        raise ValueError("DeepSeekV4 serving requires --dp 1 --tp 8")
    if len(parallel_config.devices) != 8:
        raise ValueError("DeepSeekV4 serving requires exactly 8 NPU device ids")
    if args.block_size != 128:
        raise ValueError("DeepSeekV4 kernels require --block-size 128")
    if args.max_num_seqs > 64:
        raise ValueError("DeepSeekV4 decode kernels support at most --max-num-seqs 64")
    if args.max_model_len > 260:
        raise ValueError(
            "DeepSeekV4 pypto-lib decode CSA state tables currently support at most "
            "--max-model-len 260. Increase the decode CSA state table depth in pypto-lib "
            "before serving longer contexts."
        )


def run_serve(
    config: EngineConfig,
    *,
    host: str = "0.0.0.0",
    port: int = 8000,
) -> None:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for _n in ("simpler_setup", "pypto", "simpler"):
        logging.getLogger(_n).setLevel(logging.WARNING)
    try:
        import uvicorn
    except ImportError as e:
        raise ImportError("Serving mode requires uvicorn. Install with: pip install uvicorn") from e

    from python.core.async_engine import AsyncLLMEngine
    from python.core.server import create_serving_app
    from python.core.tokenizer import TransformersTokenizerAdapter

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
    print(f"  Platform: {config.platform}, Device groups: {_format_device_groups(config)}")
    print(f"  Max running requests: {config.max_num_running_reqs}")
    print(f"  Max scheduled tokens/iter: {config.max_num_scheduled_tokens}")
    print(f"  Chunked prefill threshold: {config.long_prefill_token_threshold}")
    print(f"  Prefix cache: {'enabled' if config.enable_prefix_cache else 'disabled'}")
    print(f"  Chunk prefill: {'enabled' if config.enable_chunk_prefill else 'disabled'}")
    print("  Endpoints: /v1/completions, /v1/chat/completions, /v1/models, /health")

    uvicorn.run(app, host=host, port=port, log_level="info")


def _format_device_groups(config: EngineConfig) -> str:
    parallel_config = config.parallel_config
    if parallel_config is None:
        return str(list(config.worker_device_ids()))
    return str([list(group) for group in parallel_config.replica_device_groups])


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
    global ParallelConfig, RuntimeConfig, parse_device_ids

    if RuntimeConfig is None:
        try:
            from ..core import RuntimeConfig as imported_runtime_config
        except ImportError:
            from python.core import RuntimeConfig as imported_runtime_config

        RuntimeConfig = imported_runtime_config
    if ParallelConfig is None or parse_device_ids is None:
        try:
            from ..core.parallel import ParallelConfig as imported_parallel_config
            from ..core.parallel import parse_device_ids as imported_parse_device_ids
        except (ImportError, ValueError):
            from python.core.parallel import ParallelConfig as imported_parallel_config
            from python.core.parallel import parse_device_ids as imported_parse_device_ids

        ParallelConfig = imported_parallel_config
        parse_device_ids = imported_parse_device_ids


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
