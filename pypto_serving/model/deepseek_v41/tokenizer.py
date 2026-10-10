# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""V4.1 chat prompts over the existing local fast-tokenizer loading and decoding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pypto_serving.model.tokenizer import TransformersTokenizerAdapter

from .encoding import encode_messages


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
