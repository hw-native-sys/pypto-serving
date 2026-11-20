# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Typed seams between common serving and model-specific execution.

The common engine owns the lifetime of these values.  Their payload is owned by
the selected model adapter and is never inspected by the HTTP server or the
generic scheduler.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RequestExecutionContext:
    """Admission snapshot shared with a model adapter before its first step."""

    request_id: str
    prompt_tokens: int
    max_output_tokens: int
    initial_output_tokens: int = 0
    execution_generation: int = 0

    def __post_init__(self) -> None:
        values = (
            self.prompt_tokens,
            self.max_output_tokens,
            self.initial_output_tokens,
            self.execution_generation,
        )
        if (
            not isinstance(self.request_id, str)
            or not self.request_id
            or any(type(value) is not int for value in values)
            or self.prompt_tokens <= 0
            or self.max_output_tokens <= 0
            or self.execution_generation < 0
            or not 0 <= self.initial_output_tokens <= self.max_output_tokens
        ):
            raise ValueError("invalid request execution context")


@dataclass(frozen=True)
class ModelStepExtension:
    """Opaque model-owned metadata carried by one engine/worker step."""

    kind: str
    version: int
    payload: Any

    def __post_init__(self) -> None:
        if not self.kind or type(self.version) is not int or self.version <= 0:
            raise ValueError("invalid model step extension")


@dataclass(frozen=True)
class ModelExecutionCapabilities:
    """Executor capabilities resolved once after model registration."""

    device_sampling: bool = False
    device_stochastic_sampling: bool = False
    device_embedding: bool = False
    device_decode_embedding: bool = False
    async_decode_prepare: bool = False
    async_decode_reclaim: bool = False
    async_prefill_reclaim: bool = False
    prefill_tail_requires_reclaim: bool = False

