# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Public output formatting shared by normal and disaggregated generation."""

import asyncio
import json

import pytest

from pypto_serving.serving.engine.async_engine import TokenOutput
from pypto_serving.serving.reasoning import ParsedToolCall, ToolCallDelta
from pypto_serving.serving.server.responses import output_response, stream_outputs


async def _outputs(values):
    for value in values:
        yield value


def _events(frames):
    return [json.loads(frame[6:]) for frame in frames if frame.strip() != "data: [DONE]"]


def test_raw_terminal_unicode_flush_is_emitted_once():
    async def exercise():
        frames = [frame async for frame in stream_outputs(_outputs([
            TokenOutput(text="", completion_tokens=1),
            TokenOutput(text="city", completion_tokens=2),
            TokenOutput(text="city\ufffd", text_delta="", finished=True,
                        completion_tokens=3, finish_reason="FINISHED_LENGTH"),
        ]), "request", "model", kind="completion")]
        events = _events(frames)
        assert "".join(choice["text"] for event in events for choice in event["choices"]) == "city\ufffd"
        assert events[-1]["usage"]["completion_tokens"] == 3
        assert sum(choice["finish_reason"] is not None for event in events for choice in event["choices"]) == 1
        assert frames.count("data: [DONE]\n\n") == 1
    asyncio.run(exercise())


def test_raw_prefix_rewrite_is_reported_without_publishing_wrong_content():
    async def exercise():
        frames = [frame async for frame in stream_outputs(_outputs([
            TokenOutput(text="known"), TokenOutput(text="other", finished=True),
        ]), "request", "model", kind="completion")]
        events = _events(frames)
        assert events[0]["choices"][0]["text"] == "known"
        assert events[1]["error"]["code"] == 503
        assert len(events) == 2
    asyncio.run(exercise())


@pytest.mark.parametrize("stream", (False, True))
def test_serving_parser_owns_content_reasoning_and_tool_output(stream):
    async def exercise():
        call = ParsedToolCall("call-1", "lookup", '{"city":"Paris"}')
        delta = ToolCallDelta(0, "call-1", "lookup", call.arguments)
        values = [TokenOutput(
            text="Result", reasoning="Need data", finished=True, token_ids=(1, 2),
            text_delta="Result", reasoning_delta="Need data", tool_call_deltas=(delta,), tool_calls=(call,),
            finish_reason="FINISHED_EOS", prompt_tokens=4, completion_tokens=2,
        )]
        response = await output_response(_outputs(values), "request", "model",
                                         kind="chat", stream=stream, parsed=True)
        if stream:
            events = _events([frame async for frame in response.body_iterator])
            choice = events[0]["choices"][0]
            assert choice["delta"]["content"] == "Result"
            assert choice["delta"]["reasoning"] == "Need data"
            assert choice["delta"]["tool_calls"][0]["function"]["arguments"] == call.arguments
        else:
            choice = json.loads(response.body)["choices"][0]
            assert choice["message"]["content"] == "Result"
            assert choice["message"]["reasoning"] == "Need data"
            assert choice["message"]["tool_calls"][0]["function"]["arguments"] == call.arguments
            assert len(response.headers["x-pypto-token-ids-sha256"]) == 64
        assert choice["finish_reason"] == "tool_calls"
    asyncio.run(exercise())


def test_synthetic_128k_completion_sse_is_linear_without_token_history_on_wire():
    async def size(count):
        async def outputs():
            content = "城" * count
            for end in range(8, count + 1, 8):
                yield TokenOutput(text=content[:end], completion_tokens=end,
                                  finished=end == count, finish_reason="FINISHED_LENGTH" if end == count else "",
                                  token_ids=(1,) * count if end == count else ())
        total = 0
        async for frame in stream_outputs(outputs(), "request", "model", kind="completion"):
            assert "token_ids" not in frame
            total += len(frame.encode())
        return total
    small, large = asyncio.run(size(65536)), asyncio.run(size(131072))
    assert large < 2.1 * small
    assert large < 10_000_000
