# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import pytest

from pypto_serving.serving.engine.async_engine import TokenOutput
from pypto_serving.serving.pd.http_api import DecodeStreamFrame, encode_json, decode_json
from pypto_serving.serving.pd.output import DecodeOutputEncoder
from pypto_serving.serving.pd.protocol import HandoffKey
from pypto_serving.serving.reasoning import ParsedToolCall, ToolCallDelta


KEY = HandoffKey("request", "handoff", 1, 1, 1)


def test_raw_deltas_include_terminal_unicode_flush_and_keep_one_final_summary():
    encoder = DecodeOutputEncoder(KEY, parsed=False, streaming=True)
    outputs = [
        TokenOutput(text="", completion_tokens=1),
        TokenOutput(text="city", text_delta="city", completion_tokens=2),
        TokenOutput(text="city\ufffd", text_delta="", finished=True, token_ids=(1, 2, 3),
                    completion_tokens=3, finish_reason="FINISHED_LENGTH"),
    ]
    wires = [encoder.encode(output) for output in outputs]
    assert [wire.output_sequence for wire in wires] == [1, 2, 3]
    assert "".join(wire.text_delta for wire in wires) == "city\ufffd"
    assert [wire.text for wire in wires] == ["", "", "city\ufffd"]
    assert [wire.token_ids for wire in wires] == [(), (), (1, 2, 3)]
    frame = DecodeStreamFrame("finished", wires[-1], "COMPLETED")
    assert decode_json(encode_json(frame), DecodeStreamFrame) == frame


def test_changed_raw_prefix_is_rejected_before_sequence_advances():
    encoder = DecodeOutputEncoder(KEY, parsed=False, streaming=True)
    encoder.encode(TokenOutput(text="known"))
    with pytest.raises(RuntimeError, match="already published prefix"):
        encoder.encode(TokenOutput(text="other", finished=True))
    assert encoder.sequence == 1


@pytest.mark.parametrize("streaming", (False, True))
def test_parsed_content_reasoning_and_tool_deltas_are_owned_by_serving(streaming):
    encoder = DecodeOutputEncoder(KEY, parsed=True, streaming=streaming)
    call = ParsedToolCall("call-1", "lookup", '{"city":"Paris"}')
    delta = ToolCallDelta(0, "call-1", "lookup", call.arguments)
    wires = []
    if streaming:
        wires.append(encoder.encode(TokenOutput(text_delta="Result", reasoning_delta="Need data",
                                               tool_call_deltas=(delta,))))
    wires.append(encoder.encode(TokenOutput(
        text="Result", reasoning="Need data", finished=True, token_ids=(1, 2), tool_calls=(call,),
        text_delta="" if streaming else "Result", reasoning_delta="" if streaming else "Need data",
        tool_call_deltas=() if streaming else (delta,),
    )))
    assert "".join(wire.text_delta for wire in wires) == ("Result" if streaming else "")
    assert "".join(wire.reasoning_delta for wire in wires) == ("Need data" if streaming else "")
    assert tuple(delta for wire in wires for delta in wire.tool_call_deltas) == ((delta,) if streaming else ())
    assert wires[-1].tool_calls == (call,) and wires[-1].reasoning == "Need data"
    assert all(not wire.text and not wire.reasoning and not wire.token_ids for wire in wires[:-1])


def test_nonstream_raw_output_only_carries_complete_summary():
    wire = DecodeOutputEncoder(KEY, parsed=False, streaming=False).encode(
        TokenOutput(text="full result", text_delta="result", finished=True, token_ids=(1, 2)),
    )
    assert wire.text == "full result" and wire.token_ids == (1, 2)
    assert not wire.text_delta


def test_synthetic_128k_completion_wire_grows_linearly():
    def size(count):
        encoder = DecodeOutputEncoder(KEY, parsed=False, streaming=True)
        content = "\u57ce" * count
        total = 0
        for end in range(8, count + 1, 8):
            finished = end == count
            wire = encoder.encode(TokenOutput(
                text=content[:end], token_id=1234, finished=finished, completion_tokens=end,
                token_ids=(1234,) * count if finished else (),
            ))
            assert finished or (not wire.text and not wire.token_ids)
            total += len(encode_json(DecodeStreamFrame("finished" if finished else "output", wire)))
        return total

    small, large = size(65536), size(131072)
    assert large < 2.1 * small
    assert large < 10_000_000
