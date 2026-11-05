# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Fit text-action history to the local Qwen chat template and token budget."""

import json
from pathlib import Path

from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

from .model import CompleteTextModel


class ContextBudgetExceeded(ValueError):
    """The pinned task and latest response cannot fit the configured context."""


class LocalTextCounter:
    """Count the local model's text-only chat template with its thinking setting."""

    def __init__(self, directory: Path, *, enable_thinking: bool = False):
        config = json.loads((directory / "tokenizer_config.json").read_text())
        self.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.enable_thinking = enable_thinking
        self.template = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True).from_string(
            config["chat_template"]
        )

    def count(self, messages: list[dict]) -> int:
        # Match Serving's conversion: only role/content reach the chat template.
        text = self.template.render(
            messages=[{"role": m["role"], "content": m.get("content", "")} for m in messages],
            add_generation_prompt=True, enable_thinking=self.enable_thinking,
        )
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def shorten(self, content: str, keep: int) -> str:
        ids = self.tokenizer.encode(content, add_special_tokens=False).ids
        head = keep // 2
        tail = keep - head
        return (
            self.tokenizer.decode(ids[:head], skip_special_tokens=False)
            + "\n[Observation shortened to fit context. Full output is in the trajectory. "
              "Inspect a smaller file range if needed.]\n"
            + (self.tokenizer.decode(ids[-tail:], skip_special_tokens=False) if tail else "")
        )


class BudgetedTextModel(CompleteTextModel):
    def __init__(self, *, tokenizer_path: Path, context_limit: int, **kwargs):
        super().__init__(**kwargs)
        thinking = self.config.model_kwargs.get("extra_body", {}).get("chat_template_kwargs", {}).get(
            "enable_thinking", False
        )
        self.counter = LocalTextCounter(tokenizer_path, enable_thinking=thinking)
        self.context_limit = context_limit
        self.tokenizer_path = str(tokenizer_path)
        self.context_events: list[dict] = []

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        budget = self.context_limit - self.config.model_kwargs["max_tokens"]
        pinned = set(range(min(2, len(messages))))
        groups = []
        for index, message in enumerate(messages[2:], 2):
            extra = message.get("extra", {})
            if extra.get("interrupt_type") in ("UserNewTask", "UserInterruption", "UserRejection"):
                pinned.add(index)
            elif message.get("role") == "assistant" or "response" in extra or not groups:
                groups.append([index])
            else:
                groups[-1].append(index)
        selected = set(range(len(messages)))
        shortened = {}

        def prepare():
            candidates = [dict(messages[i], content=shortened.get(i, messages[i].get("content", "")))
                          for i in sorted(selected)]
            return super(BudgetedTextModel, self)._prepare_messages_for_api(candidates)

        prepared = prepare()
        prompt_tokens = self.counter.count(prepared)
        for group in groups[:-1]:
            if prompt_tokens <= budget:
                break
            selected.difference_update(group)
            prepared = prepare()
            prompt_tokens = self.counter.count(prepared)
        # Never shorten the task or user's follow-up instructions. If a single
        # observation is too large, preserve its beginning/end and mark the loss.
        for index in sorted(selected):
            if prompt_tokens <= budget:
                break
            if index in pinned or "raw_output" not in messages[index].get("extra", {}):
                continue
            content = messages[index].get("content", "")
            low, high = 0, len(self.counter.tokenizer.encode(content, add_special_tokens=False).ids)
            best = None
            while low <= high:
                keep = (low + high) // 2
                shortened[index] = self.counter.shorten(content, keep)
                candidate = prepare()
                if self.counter.count(candidate) <= budget:
                    best = shortened[index]
                    low = keep + 1
                else:
                    high = keep - 1
            shortened[index] = best if best is not None else self.counter.shorten(content, 0)
            prepared = prepare()
            prompt_tokens = self.counter.count(prepared)
        if prompt_tokens > budget:
            raise ContextBudgetExceeded(
                f"Pinned task and latest action need {prompt_tokens} input tokens; budget is {budget}. "
                "Start a fresh session or reduce the task/output allowance."
            )
        event = {
            "prompt_tokens": prompt_tokens,
            "output_reserve": self.config.model_kwargs["max_tokens"],
            "dropped_message_indices": sorted(set(range(len(messages))) - selected),
            "shortened_message_indices": sorted(shortened),
            "messages_sent": prepared,
        }
        self.context_events.append(event)
        if event["dropped_message_indices"] or shortened:
            print(f"Context: {prompt_tokens} input tokens; older observations omitted from this request.", flush=True)
        return prepared

    def serialize(self):
        data = super().serialize()
        data.setdefault("info", {})["context_management"] = {
            "tokenizer_path": self.tokenizer_path,
            "context_limit": self.context_limit,
            "requests": self.context_events,
        }
        return data
