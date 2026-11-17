# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Serializable generation-constraint contract for an individual request."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ConstraintSpec:
    """Provider input, not a device mask or an output-parser configuration."""

    provider_id: str
    format_id: str
    tools: tuple[dict[str, Any], ...]
    tool_choice: str | dict[str, Any]
    reasoning: bool
    parallel_tool_calls: bool = True

    def __post_init__(self) -> None:
        if not self.provider_id or not self.format_id or not self.tools:
            raise ValueError("active constraints require provider, format, and tools")

    def to_wire(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "format_id": self.format_id,
            "tools": list(self.tools),
            "tool_choice": self.tool_choice,
            "reasoning": self.reasoning,
            "parallel_tool_calls": self.parallel_tool_calls,
        }

    @classmethod
    def from_wire(cls, value: dict[str, Any]) -> "ConstraintSpec":
        return cls(
            provider_id=value["provider_id"],
            format_id=value["format_id"],
            tools=tuple(value["tools"]),
            tool_choice=value["tool_choice"],
            reasoning=value["reasoning"],
            parallel_tool_calls=value.get("parallel_tool_calls", True),
        )
