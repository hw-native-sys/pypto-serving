# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Normalized errors; arbitrary backend messages are not public event fields."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .types import CompletionCertainty


class ErrorCode(str, Enum):
    INVALID_PLAN = "INVALID_PLAN"
    STALE_GENERATION = "STALE_GENERATION"
    BACKPRESSURE = "BACKPRESSURE"
    FENCE_FAILED = "FENCE_FAILED"
    DEADLINE = "DEADLINE"
    OWNER_LOST = "OWNER_LOST"
    BACKEND_FAILURE = "BACKEND_FAILURE"
    INTEGRITY = "INTEGRITY"
    POISONED = "POISONED"


@dataclass(frozen=True)
class TransferError:
    code: ErrorCode
    certainty: CompletionCertainty
    retryable: bool = False
    backend_code: int | None = None


class TransferFailure(RuntimeError):
    def __init__(self, error: TransferError):
        self.error = error
        super().__init__(error.code.value)
