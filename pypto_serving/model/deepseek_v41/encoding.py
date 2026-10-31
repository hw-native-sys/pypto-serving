# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek V4.1 encoding."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pypto_serving.model.tokenizer import TransformersTokenizerAdapter

from ..model_family import is_deepseek_v41_config


# config
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


# encoding
BOS_TOKEN = "<\uff5cbegin\u2581of\u2581sentence\uff5c>"
EOS_TOKEN = "<\uff5cend\u2581of\u2581sentence\uff5c>"
SYSTEM_TOKEN = "<\uff5cSystem\uff5c>"
USER_TOKEN = "<\uff5cUser\uff5c>"
ASSISTANT_TOKEN = "<\uff5cAssistant\uff5c>"
LATEST_REMINDER_TOKEN = "<\uff5clatest_reminder\uff5c>"
IMAGE_TOKEN = "<\uff5cdeepseek_image\uff5c>"
REFERENCE_REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"
REFERENCE_SHA256 = "502bdaec8a3fd88ebc24c4721a7038fbe42f2063c664638127056107920035c1"


def _reasoning_budget(effort: str | int | None) -> int:
    if effort is None:
        return 75
    if type(effort) is int and 1 <= effort <= 100:
        return effort
    if isinstance(effort, str) and effort in {"low", "high", "max"}:
        return {"low": 50, "high": 75, "max": 100}[effort]
    raise ValueError("V4.1 reasoning_effort must be low/high/max or an integer in [1, 100]")


def encode_messages(
    messages: Sequence[Mapping[str, object]],
    *,
    thinking_mode: str = "chat",
    reasoning_effort: str | int | None = None,
    drop_thinking: bool = True,
) -> str:
    """Render the supported pure-text subset byte-for-byte like the pinned reference."""
    if thinking_mode not in ("chat", "thinking") or type(drop_thinking) is not bool:
        raise ValueError("thinking_mode must be chat/thinking and drop_thinking must be bool")
    budget = _reasoning_budget(reasoning_effort)
    if not messages:
        raise ValueError("V4.1 chat requires at least one message")
    records = tuple(messages)
    for message in records:
        if not isinstance(message, Mapping):
            raise ValueError("V4.1 chat messages must be objects")
        unknown = set(message) - {"role", "content", "reasoning_content", "wo_eos"}
        if unknown:
            raise ValueError(f"V4.1 text encoding does not support message fields: {sorted(unknown)}")
        if message.get("role") not in ("system", "user", "assistant", "latest_reminder"):
            raise ValueError(f"V4.1 text encoding does not support role {message.get('role')!r}")
        if not isinstance(message.get("content"), str):
            raise ValueError(
                "V4.1 text message content must be a string; image/content blocks are unsupported"
            )
        reasoning = message.get("reasoning_content")
        if reasoning is not None and (not isinstance(reasoning, str) or message["role"] != "assistant"):
            raise ValueError("reasoning_content is supported only as an assistant string")
        if type(message.get("wo_eos", False)) is not bool:
            raise ValueError("wo_eos must be bool")
        forbidden = (IMAGE_TOKEN, "<image>", "</image>")
        if any(marker in message["content"] for marker in forbidden) or (
            isinstance(reasoning, str) and IMAGE_TOKEN in reasoning
        ):
            raise ValueError("raw image special tokens require an implemented vision input path")
    # The official tool preprocessor also merges adjacent text-only user messages.
    merged = []
    for message in records:
        if merged and message["role"] == "user" and merged[-1]["role"] == "user":
            merged[-1]["content"] += "\n\n" + message["content"]
        else:
            merged.append(dict(message))
    records = tuple(merged)
    last_user = max(
        (
            i
            for i, message in enumerate(records)
            if message["role"] == "user" or (message["role"] == "system" and i > 0)
        ),
        default=-1,
    )
    parts = [BOS_TOKEN]
    for index, message in enumerate(records):
        role, content = message["role"], message["content"]
        if index == 0:
            if thinking_mode == "thinking" or role == "system":
                parts.append(SYSTEM_TOKEN)
            if thinking_mode == "thinking":
                parts.append(
                    f"Reasoning Effort: {budget} "
                    "(range 1-100, the higher the value, the more thorough the reasoning)\n\n"
                )
        if role == "system":
            if index > 0:
                parts.append(SYSTEM_TOKEN)
            parts.append(content)
        elif role == "user":
            parts.extend((USER_TOKEN, content))
        elif role == "latest_reminder":
            parts.extend((LATEST_REMINDER_TOKEN, content))
        else:
            if thinking_mode == "thinking" and (not drop_thinking or index > last_user):
                parts.extend((message.get("reasoning_content") or "", "</think>"))
            parts.append(content)
            if not message.get("wo_eos", False):
                parts.append(EOS_TOKEN)
        next_role = records[index + 1]["role"] if index + 1 < len(records) else None
        if next_role not in (None, "assistant", "latest_reminder"):
            continue
        if role == "user" or (role == "system" and index > 0):
            thinking = thinking_mode == "thinking" and (not drop_thinking or index >= last_user)
            parts.extend((ASSISTANT_TOKEN, "<think>" if thinking else "</think>"))
    return "".join(parts)


# tokenizer
class DeepSeekV41TokenizerAdapter(TransformersTokenizerAdapter):
    """Pure-text generation adapter; the checkpoint has no Jinja chat_template."""

    def apply_chat_template(self, messages: Sequence[Mapping[str, object]], **kwargs):
        allowed = {
            "thinking",
            "enable_thinking",
            "thinking_mode",
            "reasoning_effort",
            "drop_thinking",
            "tokenize",
            "add_generation_prompt",
        }
        unknown = set(kwargs) - allowed
        if unknown:
            raise ValueError(f"V4.1 text generation does not support template options: {sorted(unknown)}")
        for name in ("thinking", "enable_thinking", "tokenize", "add_generation_prompt", "drop_thinking"):
            if name in kwargs and type(kwargs[name]) is not bool:
                raise ValueError(f"V4.1 {name} must be bool")
        if not kwargs.get("add_generation_prompt", True):
            raise ValueError("V4.1 currently supports add_generation_prompt=True only")
        flags = [kwargs[name] for name in ("thinking", "enable_thinking") if name in kwargs]
        if len(set(flags)) > 1:
            raise ValueError("thinking and enable_thinking must agree")
        mode = kwargs.get("thinking_mode", "thinking" if flags and flags[0] else "chat")
        if mode not in ("chat", "thinking"):
            raise ValueError("thinking_mode must be chat/thinking")
        if "thinking_mode" in kwargs and flags and (mode == "thinking") != flags[0]:
            raise ValueError("thinking_mode conflicts with the supplied thinking flag")
        effort = kwargs.get("reasoning_effort")
        if effort == "none":
            mode, effort = "chat", None
        prompt = encode_messages(
            messages,
            thinking_mode=mode,
            reasoning_effort=effort,
            drop_thinking=kwargs.get("drop_thinking", True),
        )
        return self.encode(prompt) if kwargs.get("tokenize", False) else prompt
