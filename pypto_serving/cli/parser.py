# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CLI syntax and conversion into unresolved launch options."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from pypto_serving.config.launch_options import (
    DeviceOptions,
    FeatureOptions,
    LaunchOptions,
    ModelOptions,
    ParallelOptions,
    ProfileOptions,
    PromptOptions,
    RuntimeOptions,
    SchedulerOptions,
    ServeOptions,
)


def create_parser() -> argparse.ArgumentParser:
    """Build the CLI parser without resolving model-dependent defaults."""
    parser = argparse.ArgumentParser(
        prog="pypto-serving",
        description="Start an OpenAI-compatible server, or run offline generation with --prompt.",
    )
    _add_model_args(parser.add_argument_group("model"))
    _add_device_args(parser.add_argument_group("backend and devices"))
    _add_parallel_args(parser.add_argument_group("parallelism"))
    _add_scheduler_args(parser.add_argument_group("scheduler"))
    _add_runtime_args(parser.add_argument_group("runtime"))
    _add_generation_args(parser.add_argument_group("generation"))
    _add_serving_args(parser.add_argument_group("serving"))
    _add_profile_args(parser.add_argument_group("profiling"))
    parser.add_argument("--show-startup-logs", action="store_true",
                        help="Show model loading and kernel compilation logs.")
    return parser


def _add_model_args(group: argparse._ArgumentGroup) -> None:
    """Register model and checkpoint options."""
    group.add_argument("--model", required=True, help="Path to the model directory.")
    group.add_argument("--served-model-name", default=None,
                       help="Model name used in the API. Defaults to the model directory name.")
    group.add_argument("--dtype", default="bfloat16", help="Weight data type (default: bfloat16).")
    group.add_argument("--kv-cache-dtype", default="bfloat16",
                       help="KV cache data type. 'auto' follows --dtype (default: bfloat16).")


def _add_device_args(group: argparse._ArgumentGroup) -> None:
    """Register backend and device selection options."""
    group.add_argument("--backend", default="npu", choices=["npu"],
                       help="Inference backend (default: npu).")
    group.add_argument("--platform", default="a2a3", help="NPU platform (default: a2a3).")
    group.add_argument("--device", type=int, default=0, help="NPU device ID (default: 0).")
    group.add_argument("--devices", default=None,
                       help="Comma-separated NPU device ids for the requested parallel placement.")


def _add_parallel_args(group: argparse._ArgumentGroup) -> None:
    """Register requested parallel sizes and routing."""
    group.add_argument("--data-parallel-size", "--dp", type=int, default=1,
                       help="Data-parallel size. DeepSeekV4 uses model-local attention DP.")
    group.add_argument("--tensor-parallel-size", "--tp", type=int, default=1,
                       help="Tensor-parallel group size.")
    group.add_argument("--expert-parallel-size", "--ep", type=int, default=1,
                       help="Expert-parallel group size.")
    group.add_argument("--data-parallel-routing", default="least_pending_tokens",
                       choices=["least_pending_tokens"], help="Data-parallel request routing policy.")


def _add_scheduler_args(group: argparse._ArgumentGroup) -> None:
    """Register scheduling and cache policy options."""
    group.add_argument("--max-model-len", type=int, default=1024,
                       help="Maximum sequence length (prompt + generated; default: 1024).")
    group.add_argument("--block-size", type=int, default=128,
                       help="KV cache block size (default: 128).")
    group.add_argument("--npu-memory-utilization", type=float, default=0.90,
                       help="Fraction of total NPU HBM available to the server (default: 0.90).")
    group.add_argument("--max-num-seqs", type=int, default=16,
                       help="Max concurrent requests in serving mode (default: 16).")
    group.add_argument("--max-num-batched-tokens", type=int, default=4096,
                       help="Max tokens scheduled per iteration (default: 4096).")
    group.add_argument("--long-prefill-token-threshold", type=int, default=2048,
                       help="Maximum tokens per request in a chunked-prefill step (default: 2048).")
    group.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction,
                       default=True, help="Enable prefix caching.")
    group.add_argument("--enable-chunked-prefill", action=argparse.BooleanOptionalAction,
                       default=True, help="Enable chunked prefill.")


def _add_runtime_args(group: argparse._ArgumentGroup) -> None:
    """Register lower-level ring capacity options."""
    for flag, description in (
        ("--ring-dep-pool", "Simpler ring dependency-edge pool capacity"),
        ("--ring-task-window", "Simpler ring task-slot window capacity"),
        ("--ring-heap", "Simpler per-ring output-heap size in bytes"),
    ):
        group.add_argument(flag, type=_parse_ring_value, default=None, metavar="N | N,N,N,N",
                           help=f"{description}. One int broadcasts; four ints size rings 0..3.")


def _add_generation_args(group: argparse._ArgumentGroup) -> None:
    """Register prompts and shared generation parameters."""
    group.add_argument("--prompt", action="append", default=None, metavar="TEXT",
                       help="Run offline generation instead of serving. Repeat for multiple prompts.")
    group.add_argument("--generate-config", type=lambda value: _parse_json_object(value, "--generate-config"),
                       default=None, metavar="JSON", help="GenerateConfig fields as a JSON object.")
    group.add_argument("--num-speculative-tokens", type=int, default=None,
                       help="Deprecated DeepSeek V4 MTP draft-token count.")
    group.add_argument("--speculative-config",
                       type=lambda value: _parse_json_object(value, "--speculative-config"),
                       default=None, metavar="JSON", help="Speculative decoding configuration as JSON.")


def _add_serving_args(group: argparse._ArgumentGroup) -> None:
    """Register HTTP binding options."""
    group.add_argument("--host", default="0.0.0.0", help="HTTP bind host (default: 0.0.0.0).")
    group.add_argument("--port", type=int, default=8000, help="HTTP bind port (default: 8000).")


def _add_profile_args(group: argparse._ArgumentGroup) -> None:
    """Register on-demand profiling options."""
    group.add_argument("--profile", action="store_true", help="Enable on-demand SA profiling.")
    group.add_argument("--profile-output", default=None, metavar="PATH",
                       help="Profile output directory or .json path (default: ./profile_out).")
    group.add_argument("--profile-level", default=None, metavar="LEVELS",
                       help="Comma-separated profile levels (default: e2e,kernel).")


def _parse_ring_value(value: str) -> int | list[int]:
    """Parse a single ring size or comma-separated ring sizes."""
    parts = [part.strip() for part in value.split(",")]
    if not all(parts):
        raise argparse.ArgumentTypeError(f"ring value must be an int or comma-separated ints, got {value!r}")
    try:
        sizes = [int(part) for part in parts]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"ring value must be an int or comma-separated ints, got {value!r}"
        ) from None
    if any(size < 0 for size in sizes):
        raise argparse.ArgumentTypeError(f"ring sizes must be non-negative, got {value!r}")
    return sizes[0] if len(sizes) == 1 else sizes


def _parse_json_object(value: str, flag: str) -> dict[str, object]:
    """Parse a JSON object, reporting malformed input as an argparse error."""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"{flag} must be valid JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(f"{flag} must be a JSON object")
    return parsed


def parse_launch_options(argv: Sequence[str] | None = None) -> LaunchOptions:
    """Parse CLI arguments into unresolved launch inputs."""
    args = create_parser().parse_args(argv)
    return LaunchOptions(
        model=ModelOptions(args.model, args.served_model_name, args.dtype, args.kv_cache_dtype),
        device=DeviceOptions(
            backend=args.backend, platform=args.platform,
            default_device=args.device, devices=args.devices,
        ),
        parallel=ParallelOptions(
            data_parallel_size=args.data_parallel_size,
            tensor_parallel_size=args.tensor_parallel_size,
            expert_parallel_size=args.expert_parallel_size,
            data_parallel_routing=args.data_parallel_routing,
        ),
        scheduler=SchedulerOptions(args.max_model_len, args.block_size, args.npu_memory_utilization,
                                   args.max_num_seqs, args.max_num_batched_tokens,
                                   args.long_prefill_token_threshold, args.enable_prefix_caching,
                                   args.enable_chunked_prefill),
        runtime=RuntimeOptions(args.ring_dep_pool, args.ring_task_window, args.ring_heap),
        features=FeatureOptions(args.speculative_config, args.num_speculative_tokens),
        profile=ProfileOptions(args.profile, args.profile_output, args.profile_level),
        serve=ServeOptions(args.host, args.port),
        generate=PromptOptions(tuple(args.prompt)) if args.prompt else None,
        generate_config=args.generate_config,
        show_startup_logs=args.show_startup_logs,
    )
