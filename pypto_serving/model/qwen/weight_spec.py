# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3 as data: Hugging Face checkpoint names, the stacked decode layout, and the globals.

Two things differ from DeepSeekV4 and both are visible here rather than in the shared code:

* **No rank axis.** A Qwen weight is stacked over layers on axis 0, where a DeepSeekV4 weight
  leads with the rank axis its upload shards on and stacks on axis 1.
* **Projections are stored transposed.** The kernels want ``[in, out]`` while the checkpoint
  ships ``[out, in]``, so every projection carries ``transpose=True`` and the stacked slab is
  ``[num_layers * in, out]``.
"""

import torch

from pypto_serving.model.common.weights.pipeline import StagingPolicy
from pypto_serving.model.common.weights.shard import NoShard
from pypto_serving.model.common.weights.spec import (
    DefaultedWeightRule,
    GlobalWeightRule,
    LayerRule,
    LayerWeightRule,
)
from pypto_serving.model.common.weights.stacker import StackGroup
from pypto_serving.model.common.weights.store import LazySafetensorsStore

# The fused LM head hard-codes its padded vocabulary; this must stay a multiple of the kernel's
# VOCAB_CHUNK (64).
QWEN_VOCAB_PAD_MULTIPLE = 512

QWEN_MISSING_NAME_ERROR = "Missing Qwen3 weight tensor in index: {name}"
QWEN_MISSING_NAMES_ERROR = "Qwen3 checkpoint is missing required tensors: {names}"
QWEN_MISSING_SHARD_ERROR = "Missing safetensors shard for Qwen3 weight load: {path}"


class QwenWeightStore(LazySafetensorsStore):
    """Lazy name-addressed access to a Hugging Face Qwen3 checkpoint."""

    missing_name_error = QWEN_MISSING_NAME_ERROR
    missing_names_error = QWEN_MISSING_NAMES_ERROR
    missing_shard_error = QWEN_MISSING_SHARD_ERROR


# Kernel name <- checkpoint suffix, in the order `_stage_stacked_decode_weights` builds them.
# Order is contract for the same reason it is on the DeepSeekV4 side: it is the order the slabs
# are laid out in.
#
# Projections stack their transpose in bfloat16; norm gammas stack as `[1, dim]` rows in float32
# — hence `flatten_to_row`, which is what makes a 1-D gamma stackable at all.
QWEN_LAYER_RULES: tuple[LayerRule, ...] = (
    LayerWeightRule(
        "decode_input_rms_weight", "input_layernorm.weight", torch.float32, flatten_to_row=True
    ),
    LayerWeightRule("decode_wq", "self_attn.q_proj.weight", torch.bfloat16, transpose=True),
    LayerWeightRule("decode_wk", "self_attn.k_proj.weight", torch.bfloat16, transpose=True),
    LayerWeightRule("decode_wv", "self_attn.v_proj.weight", torch.bfloat16, transpose=True),
    # Absent in some Qwen3 checkpoints, and that is a variant rather than a fault: the kernel
    # still multiplies by a gamma, so the neutral value is ones. Zeros would annihilate the
    # activations instead of leaving them unscaled.
    DefaultedWeightRule(
        "decode_q_norm_weight",
        "self_attn.q_norm.weight",
        torch.float32,
        default_shape=("head_dim",),
        default_fill="ones",
        flatten_to_row=True,
    ),
    DefaultedWeightRule(
        "decode_k_norm_weight",
        "self_attn.k_norm.weight",
        torch.float32,
        default_shape=("head_dim",),
        default_fill="ones",
        flatten_to_row=True,
    ),
    LayerWeightRule("decode_wo", "self_attn.o_proj.weight", torch.bfloat16, transpose=True),
    LayerWeightRule(
        "decode_post_rms_weight", "post_attention_layernorm.weight", torch.float32, flatten_to_row=True
    ),
    LayerWeightRule("decode_w_gate", "mlp.gate_proj.weight", torch.bfloat16, transpose=True),
    LayerWeightRule("decode_w_up", "mlp.up_proj.weight", torch.bfloat16, transpose=True),
    LayerWeightRule("decode_w_down", "mlp.down_proj.weight", torch.bfloat16, transpose=True),
)

QWEN_NORM_WEIGHT_NAMES = frozenset(
    {
        "decode_input_rms_weight",
        "decode_q_norm_weight",
        "decode_k_norm_weight",
        "decode_post_rms_weight",
    }
)

# The whole-model weights. `lm_head.weight` falls back to the embedding for a tied checkpoint,
# and the two pad differently: see GlobalWeightRule for why an LM head cannot pad with zeros.
QWEN_GLOBAL_RULES: tuple[GlobalWeightRule, ...] = (
    GlobalWeightRule(
        "embed_weight",
        "model.embed_tokens.weight",
        torch.bfloat16,
        pad_to_multiple=QWEN_VOCAB_PAD_MULTIPLE,
        pad_fill="zeros",
    ),
    GlobalWeightRule(
        "lm_head_weight",
        "lm_head.weight",
        torch.bfloat16,
        fallback_source="model.embed_tokens.weight",
        pad_to_multiple=QWEN_VOCAB_PAD_MULTIPLE,
        pad_fill="first_row",
    ),
    GlobalWeightRule("final_norm_weight", "model.norm.weight", torch.float32, flatten_to_row=True),
)


def qwen_layer_prefix(layer_id: int) -> str:
    """Return the checkpoint prefix for one decoder layer."""
    return f"model.layers.{int(layer_id)}"


def qwen_layer_weight_names(layer_id: int) -> tuple[str, ...]:
    """Return every checkpoint name one layer may carry, optional ones included."""
    prefix = qwen_layer_prefix(layer_id)
    return tuple(f"{prefix}.{rule.source}" for rule in QWEN_LAYER_RULES)


def qwen_stack_groups(num_layers: int) -> tuple[StackGroup, ...]:
    """One group: every Qwen layer contributes to every stacked weight."""
    return (StackGroup(id="decode", members=None, layer_ids=tuple(range(int(num_layers)))),)


def qwen_policy() -> NoShard:
    """Qwen kernels take one copy of each weight, so nothing is distributed across ranks."""
    return NoShard()


def qwen_staging_policy(num_layers: int, workers: int) -> StagingPolicy:
    """Qwen overlaps layers: they are small, so read latency dominates the copy.

    Each worker is pinned to one torch thread, without which N staging threads each fan out
    into torch's own pool and oversubscribe a many-core host — the copies then run slower than
    serially.
    """
    return StagingPolicy(workers=max(1, min(int(workers), int(num_layers))), pin_torch_threads=True)
