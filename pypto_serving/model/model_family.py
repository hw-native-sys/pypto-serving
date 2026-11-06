# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal


ModelFamily = Literal["deepseek_v4", "qwen3_8_27b", "qwen"]


def read_model_config(model_dir: str | Path) -> dict[str, object]:
    """Read model config metadata used for family detection and validation."""
    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        return {}
    try:
        data = json.loads(config_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def is_deepseek_v4_config(config_data: dict[str, object]) -> bool:
    """Return whether config metadata names DeepSeek V4."""
    model_type = str(config_data.get("model_type") or "").lower()
    raw_architectures = config_data.get("architectures") or ()
    architectures = (
        {str(item).lower() for item in raw_architectures}
        if isinstance(raw_architectures, (list, tuple))
        else set()
    )
    return model_type == "deepseek_v4" or "deepseekv4forcausallm" in architectures


def qwen38_text_config(config_data: dict[str, object]) -> dict[str, object]:
    """Return the text model fields from a flat or multimodal Qwen config."""
    text = config_data.get("text_config")
    return text if isinstance(text, dict) else config_data


def is_qwen38_config(config_data: dict[str, object]) -> bool:
    """Recognise the hybrid Qwen3.8 family before the Qwen3-14B fallback."""
    text = qwen38_text_config(config_data)
    model_type = str(config_data.get("model_type") or text.get("model_type") or "").lower()
    architectures = config_data.get("architectures") or ()
    names = {model_type}
    if isinstance(architectures, (list, tuple)):
        names.update(str(name).lower() for name in architectures)
    hybrid = "linear_num_value_heads" in text or "full_attention_interval" in text
    named = any("qwen3_8" in name or "qwen3.8" in name for name in names)
    # Some multimodal configs expose only the text dimensions. Keep those on
    # this family so validation can reject a malformed checkpoint explicitly.
    identified_dimensions = (
        text.get("hidden_size") == 5120
        and text.get("num_hidden_layers") == 64
        and text.get("num_attention_heads") == 24
    )
    return named or (hybrid and identified_dimensions)


def detect_model_family(config_data: dict[str, object]) -> ModelFamily:
    """Return the serving model family inferred from config metadata."""
    if is_deepseek_v4_config(config_data):
        return "deepseek_v4"
    if is_qwen38_config(config_data):
        return "qwen3_8_27b"
    return "qwen"
