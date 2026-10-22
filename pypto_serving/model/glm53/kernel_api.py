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
MAX_CAPACITY = 8192


def packed_capacity(max_seq_len: int) -> int:
    """Reserve one token for each inactive kernel slot, rounded to 64 rows."""
    rows = int(max_seq_len) + KERNEL_BATCH - 1
    capacity = ((rows + 63) // 64) * 64
    if max_seq_len <= 0 or capacity > MAX_CAPACITY:
        raise ValueError(f"GLM full-prefix serving supports max_model_len 1..{MAX_CAPACITY - 15}")
    return capacity


def import_kernel(name: str, *, capacity: int | None = None):
    """Import GLM kernels with fixed TP16/EP16 extents."""
    override = os.environ.get("PYPTO_LIB_ROOT")
    candidates = [Path(override)] if override else [parent / "pypto-lib" for parent in Path(__file__).resolve().parents]
    root = next((path.resolve() for path in candidates if (path / "models/glm5_3_flash/weights.py").is_file()), None)
    if root is None:
        raise FileNotFoundError("Set PYPTO_LIB_ROOT to a pypto-lib checkout with GLM-5.3-Flash kernels")
    module_name = f"models.glm5_3_flash.{name}"
    previous = sys.modules.get(module_name)
    if previous is not None and not Path(previous.__file__).resolve().is_relative_to(root):
        raise RuntimeError("GLM kernels have already been imported from a different PYPTO_LIB_ROOT")
    old_argv, old_path = sys.argv, list(sys.path)
    sys.path.insert(0, str(root))
    sys.argv = ["pypto-serving-glm53", "--tp", "16", "--ep", "16"]
    if capacity is not None:
        sys.argv += ["--capacity", str(capacity), "--tokens", str(((capacity + 255) // 256) * 16)]
    try:
        module = importlib.import_module(module_name)
    finally:
        sys.argv = old_argv
        sys.path[:] = old_path
    if name == "prefill_fwd" and capacity is not None and module.CAPACITY != capacity:
        raise ValueError("GLM capacity is import-frozen; load each serving model in its own worker process")
    return module
