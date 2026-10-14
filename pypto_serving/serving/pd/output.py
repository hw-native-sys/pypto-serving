# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Linear-size Decode output wire with one authoritative terminal summary."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .protocol import DecodeOutputWire, HandoffKey

if TYPE_CHECKING:
    from pypto_serving.serving.engine.async_engine import TokenOutput


class DecodeOutputEncoder:
    def __init__(self, key: HandoffKey, *, parsed: bool, streaming: bool) -> None:
        self.key = key
        self.parsed = parsed
        self.streaming = streaming
        self.sequence = 0
        self.previous_text = ""

    def encode(self, output: TokenOutput) -> DecodeOutputWire:
        if not self.streaming:
            text_delta = ""
        elif self.parsed:
            # The Serving parser owns content/reasoning/tool-call deltas.
            text_delta = output.text_delta
        else:
            if not output.text.startswith(self.previous_text):
                raise RuntimeError("Decode content output changed an already published prefix")
            # Compute from authoritative text, including the terminal Unicode
            # flush; upstream text_delta is formed before that final flush.
            text_delta = output.text[len(self.previous_text):]
            self.previous_text = output.text
        self.sequence += 1
        return DecodeOutputWire(
            key=self.key,
            token_id=output.token_id,
            text=output.text if output.finished else "",
            finished=output.finished,
            finish_reason=output.finish_reason,
            prompt_tokens=output.prompt_tokens,
            completion_tokens=output.completion_tokens,
            token_ids=output.token_ids if output.finished else (),
            output_sequence=self.sequence,
            reasoning=output.reasoning if output.finished else "",
            text_delta=text_delta,
            reasoning_delta=output.reasoning_delta if self.streaming else "",
            tool_call_deltas=output.tool_call_deltas if self.streaming else (),
            tool_calls=output.tool_calls if output.finished else (),
        )
