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
import dataclasses
import json
import os
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from pypto_serving.config.parallel import ParallelConfig, parse_device_ids
from pypto_serving.config.types import GenerateConfig, RuntimeConfig
from pypto_serving.tools.profile import (
    ProfileConfig,
    configure_profiler,
    create_profile_config,
    get_profiler,
    merge_profile,
    profile_span,
    start_profile,
    stop_profile,
)

if TYPE_CHECKING:
    from pypto_serving.serving.engine.async_engine import AsyncLLMEngine, EngineConfig


_VALID_BACKENDS = {"npu"}
_LEGACY_SERVING_PROFILE_ENV = ("SA_PROFILE_OUTPUT", "SA_PROFILE_LEVEL")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pypto-serving",
        description=(
            "Start PyPTO Serving with an OpenAI-compatible API, or run offline "
            "generation with --prompt."
        ),
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
    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        metavar="TEXT",
        help=(
            "Offline generate mode: run the given prompt(s) to completion and "
            "exit instead of starting the server. Repeat the flag for multiple "
            "prompts (scheduled with continuous batching)."
        ),
    )
    parser.add_argument(
        "--generate-config",
        type=_parse_generate_config,
        default=None,
        metavar="JSON",
        help=(
            "GenerateConfig fields as a JSON object: max_new_tokens, "
            "temperature, top_p, top_k, stop, stream, ignore_eos. Generate "
            "mode: the run's sampling config. Serve mode: server-wide defaults "
            "for request fields that are omitted (default: GenerateConfig "
            "defaults, e.g. max_new_tokens 256, temperature 0.8, top_p 0.95)."
        ),
    )
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
    parser.add_argument(
        "--moe-stats-output",
        default=None,
        metavar="JSONL",
        help=(
            "Dump per-layer DeepSeek V4 physical-expert token counts to JSONL. "
            "Omit this option to disable device-to-host statistics transfer."
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
        if args.moe_stats_output:
            executor_kwargs["moe_stats_output"] = args.moe_stats_output
    elif num_speculative_tokens:
        raise ValueError(
            "--speculative-config/--num-speculative-tokens/--enable-mtp is only "
            "supported for DeepSeek V4"
        )
    elif args.moe_stats_output:
        raise ValueError("--moe-stats-output is only supported for DeepSeek V4")
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
    # DeepSeek prefix caching currently covers autoregressive decoding and the
    # one-token MTP path.  Keep the newer arbitrary-depth MTP implementation
    # available, but do not advertise prefix-cache compatibility for it yet.
    enable_prefix_cache = args.enable_prefix_caching
    if model_family == "deepseek_v4" and num_speculative_tokens > 1:
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
    max_prefill_tokens_per_request = None
    supports_chunked_prefill_with_speculation = True
    if model_family == "deepseek_v4":
        from pypto_serving.model.deepseek.npu_runner import (
            DEEPSEEK_V4_MAIN_PREFILL_MAX_TOKENS,
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
            enable_mtp=num_speculative_tokens == 1,
            max_seq_len=args.max_model_len,
        )
        max_prefill_tokens_per_request = DEEPSEEK_V4_MAIN_PREFILL_MAX_TOKENS

    return RuntimeConfig(
        page_size=args.block_size,
        max_batch_size=args.max_num_seqs,
        max_seq_len=args.max_model_len,
        device="cpu",
        kv_dtype=kv_dtype,
        weight_dtype=args.dtype,
        npu_memory_utilization=args.npu_memory_utilization,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_prefill_tokens_per_request=max_prefill_tokens_per_request,
        supports_chunked_prefill_with_speculation=supports_chunked_prefill_with_speculation,
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


def _parse_generate_config(value: str) -> dict[str, object]:
    """Parse the --generate-config JSON object (GenerateConfig fields)."""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            f"--generate-config must be valid JSON: {exc.msg}"
        ) from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("--generate-config must be a JSON object")
    return parsed


def _build_generate_config(data: dict[str, object] | None) -> GenerateConfig:
    """Materialise the parsed --generate-config into a GenerateConfig."""
    if data is None:
        return GenerateConfig()
    valid_fields = {field.name for field in dataclasses.fields(GenerateConfig)}
    unknown = set(data) - valid_fields
    if unknown:
        raise ValueError(
            f"--generate-config has unknown fields: {sorted(unknown)}; "
            f"valid fields: {sorted(valid_fields)}"
        )
    options = dict(data)
    _validate_generate_config_options(options)
    if isinstance(options.get("stop"), list):
        options["stop"] = tuple(options["stop"])
    return GenerateConfig(**options)


def _validate_generate_config_options(options: dict[str, object]) -> None:
    """Check value types and ranges before the engine starts.

    ``argparse`` only verifies that ``--generate-config`` is a JSON object;
    without this gate ``{"stream": "false"}`` would be truthy and enable
    streaming, ``{"stop": "END"}`` would later split into three one-character
    stop strings, and ``{"max_new_tokens": 2.5}`` would only fail deep inside
    the scheduler.
    """

    def is_number(value: object) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def is_positive_int(value: object) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value > 0

    checks: dict[str, tuple[str, object]] = {
        "max_new_tokens": ("a positive int", is_positive_int),
        "temperature": ("a non-negative number", lambda v: is_number(v) and v >= 0),
        "top_p": ("a number in (0, 1]", lambda v: is_number(v) and 0 < v <= 1),
        "top_k": ("a positive int or null", lambda v: v is None or is_positive_int(v)),
        "stop": ("a list of strings", lambda v: isinstance(v, list) and all(isinstance(item, str) for item in v)),
        "stream": ("a boolean", lambda v: isinstance(v, bool)),
        "ignore_eos": ("a boolean", lambda v: isinstance(v, bool)),
    }
    for name, (expected, check) in checks.items():
        if name in options and not check(options[name]):
            raise ValueError(
                f"--generate-config field {name!r} must be {expected}; "
                f"got {options[name]!r}"
            )


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
    generate_config: GenerateConfig,
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

    app = create_serving_app(async_engine, model_id, generate_config)

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
    runtime = config.runtime_config
    model_prefill_limit = (
        runtime.max_prefill_tokens_per_request
        if runtime is not None
        else None
    )
    print(
        "  Model prefill token limit: "
        f"{model_prefill_limit if model_prefill_limit is not None else 'unlimited'}"
    )
    prefill_limits = [config.max_num_scheduled_tokens]
    if config.enable_chunk_prefill and config.long_prefill_token_threshold > 0:
        prefill_limits.append(config.long_prefill_token_threshold)
    if model_prefill_limit is not None:
        prefill_limits.append(model_prefill_limit)
    print(f"  Effective prefill chunk size: {min(prefill_limits)}")
    if runtime is not None and runtime.num_speculative_tokens > 0:
        speculation_support = (
            "supported"
            if runtime.supports_chunked_prefill_with_speculation
            else "unsupported"
        )
        print(f"  Chunked prefill with speculative decoding: {speculation_support}")
    print(f"  Prefix cache: {'enabled' if config.enable_prefix_cache else 'disabled'}")
    print(f"  Chunk prefill: {'enabled' if config.enable_chunk_prefill else 'disabled'}")
    endpoints = "/v1/completions, /v1/chat/completions, /v1/models, /health"
    if get_profiler().enabled:
        endpoints += ", /start_profile, /stop_profile"
    print(f"  Endpoints: {endpoints}")

    uvicorn.run(app, host=host, port=port, log_level="info")


def run_generate(
    config: EngineConfig,
    prompts: list[str],
    generate_config: GenerateConfig,
) -> None:
    """Run offline generation through the serving engine, then exit.

    Offline and online generation share one engine: the prompts go through the
    same AsyncLLMEngine, scheduler, and worker as HTTP requests.
    """
    import asyncio
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for _n in ("simpler_setup", "pypto", "simpler"):
        logging.getLogger(_n).setLevel(logging.WARNING)

    from pypto_serving.model.tokenizer import load_tokenizer
    from pypto_serving.serving.engine.async_engine import AsyncLLMEngine

    if generate_config.stream and len(prompts) > 1:
        raise ValueError("--generate-config stream=true requires a single --prompt")

    configure_profiler(
        config.profile_config,
        process_name="pypto-serving-generate",
        initially_active=False,
    )
    tokenizer = load_tokenizer(config.model_dir)
    async_engine = AsyncLLMEngine(config=config, tokenizer=tokenizer)

    print(f"Generating {len(prompts)} prompt(s): model={config.model_id}")
    asyncio.run(_run_generation(async_engine, config, prompts, generate_config))


async def _run_generation(
    engine: AsyncLLMEngine,
    config: EngineConfig,
    prompts: list[str],
    generate_config: GenerateConfig,
) -> None:
    import time

    await engine.start()
    profile_enabled = config.profile_config.enabled
    try:
        if profile_enabled:
            # Mirror the HTTP profile endpoints: start the main-process
            # recorder as well as every worker's. The recorder was configured
            # initially_active=False, so without this the cli.generate,
            # scheduler, and engine spans would be missing from the trace and
            # only the worker fragments would merge.
            start_profile()
            await engine.start_profile()
        started = time.perf_counter()
        with profile_span(
            "cli.generate",
            cat="request",
            args={
                "model_id": config.model_id,
                "num_prompts": len(prompts),
                "max_new_tokens": generate_config.max_new_tokens,
                "stream": generate_config.stream,
            },
        ):
            num_tokens = await _generate_prompts(engine, prompts, generate_config)
        elapsed = time.perf_counter() - started
        print(
            f"[generate] {num_tokens} tokens in {elapsed:.2f}s "
            f"({num_tokens / elapsed:.1f} tok/s)"
        )
    finally:
        # Profiling cleanup must not strand the worker process and device:
        # engine.stop() runs even when stop_profile/merge_profile raise.
        try:
            if profile_enabled:
                await engine.stop_profile()
                stop_profile()
                merge_profile()
        finally:
            await engine.stop()


async def _generate_prompts(
    engine: AsyncLLMEngine,
    prompts: list[str],
    generate_config: GenerateConfig,
) -> int:
    """Generate all prompts; returns the total number of completion tokens."""
    if generate_config.stream:
        return await _generate_stream(engine, prompts[0], generate_config)

    results = await engine.generate_batch(prompts, generate_config)
    for index, result in enumerate(results):
        print(f"-- prompt {index + 1}/{len(prompts)} --")
        print(f"text: {result.text}")
        token_ids = result.token_ids
        print(f"token_ids: {token_ids[:64]}{'...' if len(token_ids) > 64 else ''}")
        print(f"finish_reason: {result.finish_reason}")
    return sum(len(result.token_ids) for result in results)


async def _generate_stream(
    engine: AsyncLLMEngine,
    prompt: str,
    generate_config: GenerateConfig,
) -> int:
    request_id = engine.generate_request_id()
    previous_text = ""
    token_count = 0
    async for output in engine.add_request(request_id, prompt, generate_config):
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
        generate_config = _build_generate_config(args.generate_config)

    if args.prompt:
        run_generate(config, prompts=args.prompt, generate_config=generate_config)
    else:
        run_serve(
            config,
            generate_config,
            host=args.host,
            port=args.port,
        )
    return 0


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
