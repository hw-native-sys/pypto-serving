# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Discovery and import geometry for the GLM kernel package."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys


KERNEL_BATCH = 16
PREFILL_CHUNK_ALIGNMENT = 16
MAX_CAPACITY = 8192
# Keep the supported budget range unchanged when reducing query padding.
MAX_NUM_BATCHED_TOKENS = 8177
MAX_CONTEXT = 8192


def build_glm53_cache_group_specs(max_seq_len, max_num_batched_tokens, *, tp_size=16, cache_tokens=None):
    """Describe jointly allocated attention pages and rolling KDA checkpoints."""
    from pypto_serving.config.types import KVCacheGroupSpec, KVCacheSpec

    validate_topology(tp_size, tp_size)
    if not 1 <= max_num_batched_tokens <= MAX_CAPACITY:
        raise ValueError("GLM cache groups require 1..8192 query rows")
    capacity = ((max_num_batched_tokens + 63) // 64) * 64
    blocks = (max_seq_len + 127) // 128
    attention_pages = min(16 * blocks, 1048576 // 128) if cache_tokens is None else cache_tokens // 128
    state_blocks = min(blocks, (capacity + 127) // 128 + 2)
    state_pages = KERNEL_BATCH * (state_blocks + 2)
    sparse_layers = tuple(range(3, 44, 4)) + (45,)
    kda_layers = tuple(layer for layer in range(45) if layer not in sparse_layers)
    attention_bytes = len(sparse_layers) * (128 * (512 * 2 + 256 * 4) + 32 * (128 + 8 * 4))
    state_bytes = len(kda_layers) * (3 * (3 * 64 * 128 // tp_size) * 2 + (64 // tp_size) * 128**2 * 4)
    return (
        KVCacheGroupSpec("attention", sparse_layers, KVCacheSpec(128, attention_bytes), blocks,
                         num_blocks=attention_pages, max_requests_per_partition=KERNEL_BATCH),
        KVCacheGroupSpec("kda_state", kda_layers, KVCacheSpec(128, state_bytes, 128), state_blocks,
                         num_blocks=state_pages, sliding_window=128,
                         max_requests_per_partition=KERNEL_BATCH),
    )


def validate_topology(tp_size: int, ep_size: int, dp_size: int = 1, device_ids=None) -> None:
    """Accept one overlapped worker group containing eight or sixteen ranks."""
    if tp_size not in (8, 16) or ep_size != tp_size or dp_size != 1:
        raise ValueError("GLM-5.3-Flash requires TP8/EP8/DP1 or TP16/EP16/DP1")
    if device_ids is not None and (len(device_ids) != tp_size or len(set(device_ids)) != tp_size):
        raise ValueError(f"GLM TP{tp_size}/EP{ep_size} requires {tp_size} distinct NPU device ids")


def packed_capacity(max_num_batched_tokens: int) -> int:
    """Round the scheduler's new-token budget to 64 rows without extra slot padding."""
    if max_num_batched_tokens <= 0 or max_num_batched_tokens > MAX_NUM_BATCHED_TOKENS:
        raise ValueError(f"GLM max-num-batched-tokens must be 1..{MAX_NUM_BATCHED_TOKENS}")
    return ((int(max_num_batched_tokens) + 63) // 64) * 64


def validate_serving_limits(max_requests: int, max_seq_len: int, max_num_batched_tokens: int) -> None:
    """Validate request slots, context storage and the new-token dispatch budget."""
    if not 1 <= max_requests <= KERNEL_BATCH:
        raise ValueError(f"GLM max-num-seqs must be 1..{KERNEL_BATCH}")
    if not 1 <= max_seq_len <= MAX_CONTEXT:
        raise ValueError(f"GLM max-model-len must be 1..{MAX_CONTEXT}")
    packed_capacity(max_num_batched_tokens)


def import_kernel(
    name: str,
    *,
    capacity: int | None = None,
    max_seq_len: int | None = None,
    tp_size: int = 16,
    ep_size: int = 16,
    cache_tokens: int | None = None,
    context_limit: int | None = MAX_CONTEXT,
):
    """Import GLM with process-frozen topology, query capacity and context extent."""
    validate_topology(tp_size, ep_size)
    if max_seq_len is not None:
        if max_seq_len < 1 or context_limit is not None and max_seq_len > context_limit:
            limit = f"1..{context_limit}" if context_limit is not None else "positive"
            raise ValueError(f"GLM max-model-len must be {limit}")
    override = os.environ.get("PYPTO_LIB_ROOT")
    candidates = (
        [Path(override)]
        if override
        else [parent / "pypto-lib" for parent in Path(__file__).resolve().parents]
    )
    required = ("utils.py", "attention.py")
    root = next(
        (
            path.resolve()
            for path in candidates
            if all((path / "models/glm5_3_flash" / name).is_file() for name in required)
        ),
        None,
    )
    if root is None:
        raise FileNotFoundError(
            "Set PYPTO_LIB_ROOT to a pypto-lib checkout with the consolidated GLM-5.3-Flash kernels"
        )
    module_name = f"models.glm5_3_flash.{name}"
    for loaded_name in (
        "models.glm5_3_flash.config",
        "models.glm5_3_flash.layer",
        "models.glm5_3_flash.prefill_fwd",
        "models.glm5_3_flash.decode_fwd",
        module_name,
    ):
        previous = sys.modules.get(loaded_name)
        if previous is not None and not Path(previous.__file__).resolve().is_relative_to(root):
            raise RuntimeError("GLM kernels have already been imported from a different PYPTO_LIB_ROOT")
    config = sys.modules.get("models.glm5_3_flash.config")
    if config is not None and (config.TP_SIZE, config.EP_SIZE) != (tp_size, ep_size):
        raise ValueError("GLM topology is import-frozen; load each serving model in its own worker process")
    for owner_name in ("models.glm5_3_flash.layer", "models.glm5_3_flash.prefill_fwd",
                       "models.glm5_3_flash.decode_fwd"):
        owner = sys.modules.get(owner_name)
        if owner is None:
            continue
        if capacity is not None and owner.CAPACITY != capacity:
            raise ValueError(
                "GLM capacity is import-frozen; load each serving model in its own worker process"
            )
        if max_seq_len is not None and owner.MAX_SEQ_LEN != max_seq_len:
            raise ValueError(
                "GLM context extent is import-frozen; load each serving model in its own worker process"
            )
        if cache_tokens is not None and owner.CACHE_TOKENS != cache_tokens:
            raise ValueError(
                "GLM cache capacity is import-frozen; load each serving model in its own worker process"
            )
    old_argv, old_path = sys.argv, list(sys.path)
    sys.path.insert(0, str(root))
    sys.argv = ["pypto-serving-glm53", "--tp", str(tp_size), "--ep", str(ep_size)]
    if capacity is not None:
        # MoE owns the interleaved token rows for one TP rank, padded to 16.
        moe_tokens = ((capacity + tp_size * 16 - 1) // (tp_size * 16)) * 16
        sys.argv += ["--capacity", str(capacity), "--tokens", str(moe_tokens)]
    if max_seq_len is not None:
        sys.argv += ["--max-seq-len", str(max_seq_len)]
    if cache_tokens is not None:
        sys.argv += ["--cache-tokens", str(cache_tokens)]
    try:
        module = importlib.import_module(module_name)
    finally:
        sys.argv = old_argv
        sys.path[:] = old_path
    if not Path(module.__file__).resolve().is_relative_to(root):
        raise RuntimeError("GLM kernels have already been imported from a different PYPTO_LIB_ROOT")
    if name in ("layer", "prefill_fwd", "decode_fwd") and capacity is not None and module.CAPACITY != capacity:
        raise ValueError("GLM capacity is import-frozen; load each serving model in its own worker process")
    if name in ("layer", "prefill_fwd", "decode_fwd") and max_seq_len is not None and module.MAX_SEQ_LEN != max_seq_len:
        raise ValueError(
            "GLM context extent is import-frozen; load each serving model in its own worker process"
        )
    if name in ("layer", "prefill_fwd", "decode_fwd") and cache_tokens is not None and module.CACHE_TOKENS != cache_tokens:
        raise ValueError(
            "GLM cache capacity is import-frozen; load each serving model in its own worker process"
        )
    return module
