# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The accepted Qwen3.8 text checkpoint's exact 23-bank PyPTO ABI."""

from __future__ import annotations

from contextlib import ExitStack
from collections.abc import Callable, Sequence
from pathlib import Path

import torch

PREFIX = "model.language_model."
HIDDEN = 5120
MLP_WIDTH = 17408
VOCAB = 248320
NUM_LAYERS = 64
MAX_SEQUENCE = 128
DELTA_LAYERS = 48
ATTN_LAYERS = 16
QKV = 10240
VALUE_WIDTH = 6144
KV_WIDTH = 1024
Q_WIDTH = 6144
TEXT_WEIGHT_BYTES = 53_791_996_928

WEIGHT_ORDER = (
    "embedding", "input_norms", "post_norms", "mlp_gates", "mlp_ups", "mlp_downs",
    "delta_qkvs", "delta_convs", "delta_as", "delta_bs", "delta_zs", "delta_a_logs",
    "delta_dt_biases", "delta_norms", "delta_outs", "attn_qs", "attn_ks", "attn_vs",
    "attn_q_norms", "attn_k_norms", "attn_outs", "final_norm", "lm_head",
)

COMMON_BANKS = {
    "input_norms": ("input_layernorm.weight", (1, HIDDEN)),
    "post_norms": ("post_attention_layernorm.weight", (1, HIDDEN)),
    "mlp_gates": ("mlp.gate_proj.weight", (MLP_WIDTH, HIDDEN)),
    "mlp_ups": ("mlp.up_proj.weight", (MLP_WIDTH, HIDDEN)),
    "mlp_downs": ("mlp.down_proj.weight", (HIDDEN, MLP_WIDTH)),
}
DELTA_BANKS = {
    "delta_qkvs": ("linear_attn.in_proj_qkv.weight", (QKV, HIDDEN)),
    "delta_convs": ("linear_attn.conv1d.weight", (4, QKV)),
    "delta_as": ("linear_attn.in_proj_a.weight", (128, HIDDEN)),
    "delta_bs": ("linear_attn.in_proj_b.weight", (128, HIDDEN)),
    "delta_zs": ("linear_attn.in_proj_z.weight", (VALUE_WIDTH, HIDDEN)),
    "delta_a_logs": ("linear_attn.A_log", (1, 128)),
    "delta_dt_biases": ("linear_attn.dt_bias", (1, 128)),
    "delta_norms": ("linear_attn.norm.weight", (1, 128)),
    "delta_outs": ("linear_attn.out_proj.weight", (HIDDEN, VALUE_WIDTH)),
}
ATTN_BANKS = {
    "attn_qs": ("self_attn.q_proj.weight", (2 * Q_WIDTH, HIDDEN)),
    "attn_ks": ("self_attn.k_proj.weight", (KV_WIDTH, HIDDEN)),
    "attn_vs": ("self_attn.v_proj.weight", (KV_WIDTH, HIDDEN)),
    "attn_q_norms": ("self_attn.q_norm.weight", (1, 256)),
    "attn_k_norms": ("self_attn.k_norm.weight", (1, 256)),
    "attn_outs": ("self_attn.o_proj.weight", (HIDDEN, Q_WIDTH)),
}


def _layer_groups() -> tuple[tuple[range | list[int], dict], ...]:
    layers = range(NUM_LAYERS)
    return (
        (layers, COMMON_BANKS),
        ([layer for layer in layers if layer % 4 != 3], DELTA_BANKS),
        ([layer for layer in layers if layer % 4 == 3], ATTN_BANKS),
    )


def validate_qwen35_weight_index(
    weight_map: dict[str, str], config_data: dict,
) -> dict[str, list[str]]:
    """Enumerate every consumed or intentionally ignored checkpoint name."""
    text = config_data.get("text_config")
    if (config_data.get("model_type") != "qwen3_5" or not isinstance(text, dict)
            or text.get("model_type") != "qwen3_5_text"):
        raise ValueError("checkpoint is not Qwen3.8 qwen3_5 text")
    expected_schedule = ["full_attention" if layer % 4 == 3 else "linear_attention"
                         for layer in range(NUM_LAYERS)]
    required_config = {
        "num_hidden_layers": NUM_LAYERS, "hidden_size": HIDDEN,
        "intermediate_size": MLP_WIDTH, "vocab_size": VOCAB,
        "num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256,
        "layer_types": expected_schedule, "dtype": "bfloat16",
    }
    for name, value in required_config.items():
        if text.get(name) != value:
            raise ValueError(f"text_config.{name} differs from the Qwen3.8 kernel ABI")
    expected = {"lm_head.weight", PREFIX + "embed_tokens.weight", PREFIX + "norm.weight"}
    for layers, banks in _layer_groups():
        expected.update(f"{PREFIX}layers.{layer}.{suffix}" for layer in layers
                        for suffix, _ in banks.values())
    actual = set(weight_map)
    missing = sorted(expected - actual)
    if missing:
        raise KeyError(f"missing text weight {missing[0]}")
    vision = sorted(name for name in actual if name.startswith("model.visual."))
    mtp = sorted(name for name in actual if name.startswith("mtp."))
    extra = sorted(actual - expected - set(vision) - set(mtp))
    if extra:
        raise ValueError(f"unconsumed checkpoint weight {extra[0]}")
    return {"consumed_text": sorted(expected), "ignored_vision": vision, "ignored_mtp": mtp}


def _adapt_tensor(value: torch.Tensor, suffix: str, shape: tuple[int, ...]) -> torch.Tensor:
    if value.dtype != torch.bfloat16:
        raise ValueError(f"unexpected dtype for {suffix}: {value.dtype}")
    if suffix == "linear_attn.conv1d.weight":
        value = value.squeeze(1).T
    elif value.ndim == 1 and len(shape) == 2 and shape[0] == 1:
        value = value.reshape(1, -1)
    if value.ndim != len(shape) or any(actual > wanted for actual, wanted in zip(value.shape, shape)):
        raise ValueError(f"shape mismatch for {suffix}: {tuple(value.shape)} vs {shape}")
    padded = {"linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight",
              "linear_attn.A_log", "linear_attn.dt_bias"}
    if suffix not in padded and tuple(value.shape) != shape:
        raise ValueError(f"unexpected padding for {suffix}: {tuple(value.shape)} vs {shape}")
    return value


def _stage_bank(
    layers: Sequence[int], suffix: str, shape: tuple[int, ...],
    source: Callable[[str], torch.Tensor],
) -> torch.Tensor:
    padded = suffix in {"linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight",
                        "linear_attn.A_log", "linear_attn.dt_bias"}
    bank = (torch.zeros if padded else torch.empty)((len(layers), *shape), dtype=torch.bfloat16)
    for row, layer in enumerate(layers):
        value = _adapt_tensor(source(f"{PREFIX}layers.{layer}.{suffix}"), suffix, shape)
        bank[(row, *(slice(0, width) for width in value.shape))].copy_(value)
    return bank


def load_qwen35_weight_banks(
    root: Path, weight_map: dict[str, str], config_data: dict,
) -> dict[str, torch.Tensor]:
    """Stage the real BF16 checkpoint in the accepted token-major bank order."""
    from safetensors import safe_open

    coverage = validate_qwen35_weight_index(weight_map, config_data)
    consumed = coverage["consumed_text"]
    with ExitStack() as stack:
        shards = {filename: stack.enter_context(safe_open(root / filename, framework="pt", device="cpu"))
                  for filename in sorted({weight_map[name] for name in consumed})}

        def source(name: str) -> torch.Tensor:
            value = shards[weight_map[name]].get_tensor(name)
            if value.dtype != torch.bfloat16:
                raise ValueError(f"unexpected checkpoint dtype for {name}: {value.dtype}")
            return value

        banks = {
            "embedding": source(PREFIX + "embed_tokens.weight"),
            "final_norm": source(PREFIX + "norm.weight").reshape(1, HIDDEN),
            "lm_head": source("lm_head.weight"),
        }
        if banks["embedding"].shape != (VOCAB, HIDDEN) or banks["lm_head"].shape != (VOCAB, HIDDEN):
            raise ValueError("embedding or untied LM-head shape changed")
        for layers, specs in _layer_groups():
            for bank_name, (suffix, shape) in specs.items():
                banks[bank_name] = _stage_bank(layers, suffix, shape, source)
    if set(banks) != set(WEIGHT_ORDER):
        raise RuntimeError("Qwen3.8 bank coverage is incomplete")
    staged_bytes = sum(value.numel() * value.element_size() for value in banks.values())
    if staged_bytes < TEXT_WEIGHT_BYTES:
        raise ValueError("staged text weights are smaller than the real checkpoint")
    return banks
