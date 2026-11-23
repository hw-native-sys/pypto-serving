# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unified L3-callable handle for compiled PyPTO kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class L3Callable:
    """A HOST-dispatched compiled program plus its launch metadata.

    This is a field superset of the model-specific handles it replaces
    (qwen's ``_L3Callable`` and DeepSeek's ``DeepSeekV4L3Callable``); each
    model package aliases this class under its historical name, so callers and
    tests keep working unchanged. The optional fields are model-specific:

    * ``dispatch_args`` -- qwen only (static launch args threaded into the
      distributed-program dispatch).
    * ``block_dim`` -- DeepSeek only.
    """

    compiled: object
    name: str
    aicpu_thread_num: int = 4
    block_dim: int | None = None
    dispatch_args: tuple[Any, ...] = ()
    # Explicit semantic metadata for DPLB timing joins.  These fields are
    # carried alongside the compiled handle and never participate in the
    # compiler cache key.  A fused program may cover several layers, so the
    # layer list is retained instead of inventing one representative layer.
    semantic_scope: str | None = None
    attention_kinds: tuple[str, ...] = ()
    layer_ids: tuple[int, ...] = ()
