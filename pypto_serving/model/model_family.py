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


ModelFamily = Literal["deepseek_v41", "deepseek_v4", "qwen"]


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


def is_deepseek_v41_config(config_data: dict[str, object]) -> bool:
    """Identify V4.1 before the V4 and generic Hugging Face paths."""
    architectures = config_data.get("architectures", ())
    return str(config_data.get("model_type", "")).lower() == "deepseek_v41" or (
        isinstance(architectures, (list, tuple))
        and any(str(name).lower() == "deepseekv41forcausallm" for name in architectures)
    )


def detect_model_family(config_data: dict[str, object]) -> ModelFamily:
    """Return the serving model family inferred from config metadata."""
    if is_deepseek_v41_config(config_data):
        return "deepseek_v41"
    return "deepseek_v4" if is_deepseek_v4_config(config_data) else "qwen"
