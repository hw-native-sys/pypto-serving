# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Metadata-only V4.1 text configuration; no weights or device runtime are loaded."""

from dataclasses import dataclass
import json
import math
from pathlib import Path

from ..model_family import is_deepseek_v41_config


@dataclass(frozen=True)
class V41TextConfig:
    """Backbone dimensions, not a claim that a checkpoint can execute yet.

    Vision, Engram and draft metadata may exist in the checkpoint. They are
    outside this stage; compression modes here cover only the target backbone.
    """

    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    head_dim: int
    max_position_embeddings: int
    hc_mult: int
    rms_norm_eps: float
    compress_ratios: tuple[int, ...]
    bos_token_id: int | None
    eos_token_id: int | None
    pad_token_id: int | None

    @classmethod
    def from_dict(cls, raw: dict) -> "V41TextConfig":
        if not isinstance(raw, dict) or not is_deepseek_v41_config(raw):
            raise ValueError("expected a DeepSeek V4.1 model config")
        text = raw.get("text_config")
        if not isinstance(text, dict) or text.get("model_type") != "deepseek_v41_text":
            raise ValueError("text_config.model_type must be deepseek_v41_text")
        names = ("vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
                 "head_dim", "max_position_embeddings", "hc_mult")
        values = {}
        for name in names:
            value = text.get(name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"text_config.{name} must be a positive integer")
            values[name] = value
        eps = text.get("rms_norm_eps")
        if type(eps) not in (float, int) or not math.isfinite(eps) or eps <= 0:
            raise ValueError("text_config.rms_norm_eps must be finite and positive")
        ratios = text.get("compress_ratios")
        layers = values["num_hidden_layers"]
        if not isinstance(ratios, list) or len(ratios) < layers:
            raise ValueError("compress_ratios must cover every backbone layer")
        if any(type(value) is not int or value not in (0, 1, 2) for value in ratios[:layers]):
            raise ValueError("backbone compress_ratios must use 0 (SWA), 1 (C1A), or 2 (C2A)")
        for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
            value = raw.get(name, text.get(name))
            if value is not None and (type(value) is not int or not 0 <= value < values["vocab_size"]):
                raise ValueError(f"{name} must be a vocabulary index or null")
            values[name] = value
        return cls(**values, rms_norm_eps=float(eps), compress_ratios=tuple(ratios[:layers]))


def load_text_config(model_dir: str | Path) -> V41TextConfig:
    """Read only config.json; unknown optional modules are not initialized."""
    raw = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    return V41TextConfig.from_dict(raw)
