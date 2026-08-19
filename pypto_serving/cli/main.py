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
    from pypto_serving.serving.engine.async_engine import EngineConfig

from pypto_serving.observability.access_log import create_uvicorn_log_config
from pypto_serving.tools.profile import (
    ProfileConfig,
    configure_profiler,
    create_profile_config,
    get_profiler,
    merge_profile,
)

RuntimeConfig = None
ParallelConfig = None
parse_device_ids = None


_VALID_BACKENDS = {"npu"}
_LEGACY_SERVING_PROFILE_ENV = ("SA_PROFILE_OUTPUT", "SA_PROFILE_LEVEL")


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
    parser.add_argument(
        "--use-compile-cache",
        action="store_true",
        default=False,
        help=(
            "Reuse compiled kernels across launches. Each kernel is written to "
            "<pypto_build_dir>/<name> and reloaded on the next launch, skipping the JIT "
            "and the device-binary assembly. Off by default. NOTE: there is no "
            "fingerprinting, so reuse the same build dir only for the same config and "
            "kernel sources; clear it on a config/kernel change to avoid stale binaries."
        ),
    )
    parser.add_argument("--device", type=int, default=0, help="NPU device ID (default: 0).")
    parser.add_argument(
        "--devices",
        default=None,
        help="Comma-separated NPU device ids for the requested parallel placement.",
    )
    parser.add_argument(
        "--data-parallel-size",
        "--dp",
        type=int,
        default=1,
        help="Data-parallel size. DeepSeekV4 uses model-local attention DP.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        "--tp",
        type=int,
        default=1,
        help="Tensor-parallel group size.",
    )
    parser.add_argument(
        "--expert-parallel-size",
        "--ep",
        type=int,
        default=1,
        help="Expert-parallel group size.",
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
    parser.add_argument(
        "--enable-mtp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Deprecated alias for DeepSeek V4 MTP with one draft token.",
    )
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        default=None,
        help=(
            "Maximum DeepSeek V4 MTP draft tokens per iteration. "
            "Any positive value enables MTP; 0 disables it (default: 0)."
        ),
    )
    parser.add_argument(
        "--speculative-config",
        type=_parse_speculative_config,
        default=None,
        metavar="JSON",
        help=(
            "Speculative decoding configuration as JSON. DeepSeek V4 supports "
            "method='mtp' and a positive num_speculative_tokens value."
        ),
    )

    # Serving
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the serving server (default: 0.0.0.0).")
    parser.add_argument("--port", type=int, default=8000, help="Port for the serving server (default: 8000).")
    parser.add_argument("--max-num-seqs", type=int, default=16, help="Max concurrent requests in serving mode (default: 16).")
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

    # Profiling
    parser.add_argument(
        "--profile",
        action="store_true",
        help=(
            "Enable on-demand SA profiling through POST /start_profile and "
            "POST /stop_profile."
        ),
    )
    parser.add_argument(
        "--profile-output",
        default=None,
        metavar="PATH",
        help="Profile output directory or .json path (default: ./profile_out).",
    )
    parser.add_argument(
        "--profile-level",
        default=None,
        metavar="LEVELS",
        help="Comma-separated profile levels: e2e, kernel, or verbose (default: e2e,kernel).",
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

    from pypto_serving.serving.engine.async_engine import EngineConfig

    model_dir = str(Path(args.model).resolve())
    executor_kwargs: dict[str, object] = {}
    devices = parse_device_ids(args.devices, default_device=args.device)
    model_config_data = _read_model_config(Path(model_dir))
    model_family = _detect_model_family(Path(model_dir), config_data=model_config_data)
    num_speculative_tokens = _resolve_num_speculative_tokens(args)
    if model_family == "deepseek_v4":
        executor_kwargs["compile_kernels"] = True
        executor_kwargs["num_speculative_tokens"] = num_speculative_tokens
    elif num_speculative_tokens:
        raise ValueError(
            "--speculative-config/--num-speculative-tokens/--enable-mtp is only "
            "supported for DeepSeek V4"
        )
    executor_kwargs["use_compile_cache"] = args.use_compile_cache
    parallel_config = ParallelConfig(
        data_parallel_size=args.data_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
        expert_parallel_size=args.expert_parallel_size,
        enable_expert_parallel=args.expert_parallel_size > 1,
        devices=devices,
        data_parallel_routing=args.data_parallel_routing,
        placement_mode="overlapped" if model_family == "deepseek_v4" else "replica",
    )
    _validate_model_topology(
        model_family,
        args,
        parallel_config,
        config_data=model_config_data,
    )
    first_group = parallel_config.replica_device_groups[0]
    worker_device_ids = first_group if parallel_config.num_replicas == 1 else ()
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
        runtime_config=_build_runtime_config(
            args,
            model_family=model_family,
            config_data=model_config_data,
        ),
        profile_config=_build_profile_config(args),
        max_num_running_reqs=args.max_num_seqs,
        max_num_scheduled_tokens=args.max_num_batched_tokens,
        long_prefill_token_threshold=args.long_prefill_token_threshold,
        enable_prefix_cache=enable_prefix_cache,
        enable_chunk_prefill=args.enable_chunked_prefill,
    )


def _build_runtime_config(
    args: argparse.Namespace,
    *,
    model_family: str = "qwen",
    config_data: dict[str, object] | None = None,
):
    num_speculative_tokens = _resolve_num_speculative_tokens(args)
    kv_dtype = args.kv_cache_dtype
    if kv_dtype == "auto":
        kv_dtype = args.dtype

    kv_cache_groups = ()
    if model_family == "deepseek_v4":
        from pypto_serving.model.deepseek.npu_runner import (
            build_deepseek_v4_cache_group_specs,
            deepseek_v4_decode_layout,
        )

        config_data = config_data or {}
        compress_ratios = config_data.get("compress_ratios")
        if not isinstance(compress_ratios, list):
            compress_ratios = None
        num_hidden_layers = int(config_data.get("num_hidden_layers", 43))
        layout = deepseek_v4_decode_layout(num_speculative_tokens)
        kv_cache_groups = build_deepseek_v4_cache_group_specs(
            num_hidden_layers,
            compress_ratios,
            decode_batch=layout.decode_batch,
        )

    return RuntimeConfig(
        page_size=args.block_size,
        max_batch_size=args.max_num_seqs,
        max_seq_len=args.max_model_len,
        device="cpu",
        kv_dtype=kv_dtype,
        weight_dtype=args.dtype,
        npu_memory_utilization=args.npu_memory_utilization,
        max_num_batched_tokens=args.max_num_batched_tokens,
        num_speculative_tokens=num_speculative_tokens,
        kv_cache_groups=kv_cache_groups,
    )


def _resolve_num_speculative_tokens(args: argparse.Namespace) -> int:
    """Resolve the vLLM-style config and deprecated standalone aliases."""
    speculative_config = getattr(args, "speculative_config", None)
    configured = getattr(args, "num_speculative_tokens", None)
    legacy_value = getattr(args, "enable_mtp", None)
    if speculative_config is not None:
        if configured is not None or legacy_value is not None:
            raise ValueError(
                "--speculative-config cannot be combined with --num-speculative-tokens "
                "or --enable-mtp/--no-enable-mtp"
            )
        if speculative_config.get("method") != "mtp":
            raise ValueError("DeepSeek V4 --speculative-config requires method='mtp'")
        if "num_speculative_tokens" not in speculative_config:
            raise ValueError(
                "DeepSeek V4 --speculative-config requires num_speculative_tokens"
            )
        configured = speculative_config["num_speculative_tokens"]
        if isinstance(configured, bool) or not isinstance(configured, int):
            raise ValueError("num_speculative_tokens must be an integer")
        if configured <= 0:
            raise ValueError("num_speculative_tokens must be positive")
        return configured

    legacy_enabled = bool(legacy_value)
    if configured is None:
        return 1 if legacy_enabled else 0
    configured = int(configured)
    if configured < 0:
        raise ValueError("--num-speculative-tokens must be non-negative")
    if legacy_enabled and configured == 0:
        raise ValueError("--enable-mtp conflicts with --num-speculative-tokens 0")
    return configured


def _parse_speculative_config(value: str) -> dict[str, object]:
    """Parse one vLLM-style speculative decoding JSON object."""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            f"--speculative-config must be valid JSON: {exc.msg}"
        ) from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("--speculative-config must be a JSON object")
    return parsed


def _build_profile_config(args: argparse.Namespace) -> ProfileConfig:
    if not args.profile and (args.profile_output is not None or args.profile_level is not None):
        raise ValueError("--profile-output and --profile-level require --profile")
    output = Path(args.profile_output or "./profile_out").expanduser().resolve()
    return create_profile_config(
        enabled=args.profile,
        output=output,
        levels=args.profile_level or "e2e,kernel",
    )


def _warn_deprecated_serving_profile_env(args: argparse.Namespace) -> None:
    """Warn when legacy profile environment variables cannot enable HTTP profiling."""
    if args.profile:
        return
    legacy_vars = [name for name in _LEGACY_SERVING_PROFILE_ENV if name in os.environ]
    if not legacy_vars:
        return
    print(
        "WARNING: "
        f"{', '.join(legacy_vars)} are deprecated for HTTP serving and are ignored "
        "without --profile. Use --profile with --profile-output/--profile-level instead.",
        file=sys.stderr,
        flush=True,
    )


def _read_model_config(model_dir: Path) -> dict[str, object]:
    """Read config.json once for model detection, validation, and runtime setup."""
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return {}
    try:
        data = json.loads(config_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _detect_model_family(
    model_dir: Path,
    *,
    config_data: dict[str, object] | None = None,
) -> str:
    """Return the serving model family inferred from config.json."""
    config_data = _read_model_config(model_dir) if config_data is None else config_data
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
    *,
    config_data: dict[str, object] | None = None,
) -> None:
    """Validate model-specific serving topology constraints."""
    if model_family != "deepseek_v4":
        return
    if config_data is None:
        config_data = _read_model_config(Path(args.model).resolve())
    quantization = config_data.get("quantization_config") or {}
    if quantization.get("quant_method") != "compressed-tensors":
        raise ValueError(
            "DeepSeekV4 serving requires the quantized W8A8 compressed-tensors checkpoint "
            "such as /data/models/dsv4-flash-w8a8; the original checkpoint is too large for 8 NPUs."
        )
    if (
        parallel_config.placement_mode != "overlapped"
        or parallel_config.data_parallel_size != 8
        or parallel_config.expert_parallel_size != 8
        or parallel_config.tensor_parallel_size != 1
    ):
        raise ValueError("DeepSeekV4 serving requires --dp 8 --ep 8 with --tp 1 (the default)")
    if len(parallel_config.devices) != 8:
        raise ValueError("DeepSeekV4 serving requires exactly 8 NPU device ids")
    if args.block_size != 128:
        raise ValueError("DeepSeekV4 kernels require --block-size 128")
    from pypto_serving.model.deepseek.npu_runner import deepseek_v4_decode_layout

    layout = deepseek_v4_decode_layout(_resolve_num_speculative_tokens(args))
    max_global_batch = layout.ranks * layout.decode_batch
    if args.max_num_seqs > max_global_batch:
        raise ValueError(
            "DeepSeekV4 decode kernels support at most "
            f"--max-num-seqs {max_global_batch} ({layout.decode_batch} per rank)"
        )
    max_model_len = layout.prefill_csa_state_max_blocks * layout.c4_state_block_size
    if args.max_model_len > max_model_len:
        raise ValueError(
            "DeepSeekV4 pypto-lib decode CSA state tables currently support at most "
            f"--max-model-len {max_model_len}. Increase the decode CSA state table depth "
            "in pypto-lib before serving longer contexts."
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

    from pypto_serving.model.tokenizer import load_tokenizer
    from pypto_serving.serving.engine.async_engine import AsyncLLMEngine
    from pypto_serving.serving.server.server import create_serving_app

    model_id = config.model_id
    configure_profiler(
        config.profile_config,
        process_name="pypto-serving-api",
        initially_active=False,
    )
    tokenizer = load_tokenizer(config.model_dir)
    async_engine = AsyncLLMEngine(
        config=config,
        tokenizer=tokenizer
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
    print(f"  Parallelism: {_format_parallelism(config)}")
    print(f"  Max running requests: {config.max_num_running_reqs}")
    print(f"  Max scheduled tokens/iter: {config.max_num_scheduled_tokens}")
    print(f"  Chunked prefill threshold: {config.long_prefill_token_threshold}")
    print(f"  Prefix cache: {'enabled' if config.enable_prefix_cache else 'disabled'}")
    print(f"  Chunk prefill: {'enabled' if config.enable_chunk_prefill else 'disabled'}")
    endpoints = "/v1/completions, /v1/chat/completions, /v1/models, /health, /metrics"
    if get_profiler().enabled:
        endpoints += ", /start_profile, /stop_profile"
    print(f"  Endpoints: {endpoints}")

    log_config = create_uvicorn_log_config(["/metrics", "/metrics/json"])
    uvicorn.run(app, host=host, port=port, log_level="info", log_config=log_config)


def _format_device_groups(config: EngineConfig) -> str:
    parallel_config = config.parallel_config
    if parallel_config is None:
        return str(list(config.worker_device_ids()))
    return str([list(group) for group in parallel_config.replica_device_groups])


def _format_parallelism(config: EngineConfig) -> str:
    parallel_config = config.parallel_config
    if parallel_config is None:
        return f"replicas=1, worker_group_size={len(config.worker_device_ids())}"
    return (
        f"mode={parallel_config.placement_mode}, replicas={parallel_config.num_replicas}, "
        f"dp={parallel_config.data_parallel_size}, tp={parallel_config.tensor_parallel_size}, "
        f"ep={parallel_config.expert_parallel_size}"
    )


def _validate_backend(backend: str) -> None:
    if backend != "npu":
        raise ValueError(f"Only NPU backend is supported, got: {backend}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _warn_deprecated_serving_profile_env(args)

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
        from pypto_serving.config.types import RuntimeConfig as imported_runtime_config

        RuntimeConfig = imported_runtime_config
    if ParallelConfig is None or parse_device_ids is None:
        from pypto_serving.config.parallel import ParallelConfig as imported_parallel_config
        from pypto_serving.config.parallel import parse_device_ids as imported_parse_device_ids

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
