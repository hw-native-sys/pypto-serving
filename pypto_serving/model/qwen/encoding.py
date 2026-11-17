# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3 tokenizer and chat-history adaptation."""

from __future__ import annotations

from dataclasses import dataclass

from pypto_serving.model.tokenizer import TransformersTokenizerAdapter


@dataclass
class Qwen3TokenizerAdapter(TransformersTokenizerAdapter):
    """Qwen3 chat output uses thinking and JSON tool-call terminals."""

    def apply_chat_template(self, messages: list[dict[str, object]], **kwargs) -> str:
        """Pass saved assistant reasoning under the checkpoint template's field name."""
        template_messages = []
        for message in messages:
            if message.get("role") == "assistant" and isinstance(message.get("reasoning"), str):
                converted = dict(message)
                converted["reasoning_content"] = converted.pop("reasoning")
                template_messages.append(converted)
            else:
                template_messages.append(message)
        return super().apply_chat_template(template_messages, **kwargs)

    @property
    def output_parser_id(self) -> str | None:
        return "qwen3"

    @property
    def output_parser_requires_tool_context(self) -> bool:
        """Qwen3 uses its parser only when the request has tool context."""
        return True
