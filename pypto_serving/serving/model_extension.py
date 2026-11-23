# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Request context shared with a model adapter.

Per-step model data stays opaque bytes in common serving types and IPC.
The selected adapter owns its encoding and validation.
"""

from dataclasses import dataclass


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
