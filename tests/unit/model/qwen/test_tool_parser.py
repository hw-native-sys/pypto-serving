# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3 token-delimited function call parsing."""

import json

import pytest

from pypto_serving.serving.reasoning import OutputParserSpec, create_output_parser
from pypto_serving.model.qwen.tool_parser import TOOL_END, TOOL_START
from tests.unit.serving.reasoning.test_deepseek_v4_tools import ToolTokenizer


class QwenToolTokenizer(ToolTokenizer):
    specials = ("<think>", "</think>", TOOL_START, TOOL_END, "<eos>")
    vocab = {text: index + 1 for index, text in enumerate(specials)}
    all_special_ids = tuple(vocab.values())
    output_parser_requires_tool_context = True


def _parser(*, thinking=False, choice="auto", include_reasoning=True):
    return create_output_parser(OutputParserSpec(
        parser_id="qwen3", initial_state="reasoning" if thinking else "content",
        include_reasoning=include_reasoning, tool_choice=choice,
        tool_names=("lookup",) if choice != "none" else (),
    ), QwenToolTokenizer())


@pytest.mark.parametrize("chunk_size", [1, 2, 7, 100000])
def test_stream_and_complete_match_for_multiple_json_calls(chunk_size):
    arguments = {"city": "杭州", "nested": {"count": 2}, "literal": TOOL_END}
    text = (
        "plan</think>Before "
        + TOOL_START + json.dumps({"name": "lookup", "arguments": arguments}, ensure_ascii=False) + TOOL_END
        + "\n" + TOOL_START + '{"name":"lookup","arguments":{"city":"Berlin"}}' + TOOL_END
        + " after<eos>"
    )
    tokenizer = QwenToolTokenizer()
    ids = tokenizer.encode(text)
    parser = _parser(thinking=True)
    outputs = [
        parser.feed(tokenizer.decode(ids[i:i + chunk_size], skip_special_tokens=False), ids[i:i + chunk_size])
        for i in range(0, len(ids), chunk_size)
    ]
    outputs.append(parser.finish())
    complete = _parser(thinking=True).parse_complete(text, ids)
    assert "".join(out.reasoning for out in outputs) == complete.reasoning == "plan"
    assert "".join(out.content for out in outputs) == complete.content == "Before  after"
    calls = [delta for out in outputs for delta in out.tool_call_deltas]
    assert len(calls) == len(complete.tool_calls) == 2
    assert [(call.name, json.loads(call.arguments)) for call in complete.tool_calls] == [
        ("lookup", arguments), ("lookup", {"city": "Berlin"}),
    ]
    assert [(delta.name, json.loads(delta.arguments)) for delta in calls] == [
        (call.name, json.loads(call.arguments)) for call in complete.tool_calls
    ]
    assert all(call.complete for call in outputs[-1].tool_calls)


@pytest.mark.parametrize("chunk_size", [1, 7, 100000])
def test_tool_only_calls_keep_template_separator_out_of_content(chunk_size):
    text = (
        "plan</think>\n\n" + TOOL_START + '{"name":"lookup","arguments":{}}' + TOOL_END
        + "\n" + TOOL_START + '{"name":"lookup","arguments":{}}' + TOOL_END + "<eos>"
    )
    tokenizer = QwenToolTokenizer()
    ids = tokenizer.encode(text)
    parser = _parser(thinking=True)
    outputs = [
        parser.feed(tokenizer.decode(ids[i:i + chunk_size], skip_special_tokens=False), ids[i:i + chunk_size])
        for i in range(0, len(ids), chunk_size)
    ]
    outputs.append(parser.finish())
    assert "".join(output.reasoning for output in outputs) == "plan"
    assert "".join(output.content for output in outputs) == ""
    assert len(outputs[-1].tool_calls) == 2


@pytest.mark.parametrize("chunk_size", [1, 7, 100000])
def test_trailing_whitespace_after_tool_is_not_content(chunk_size):
    text = TOOL_START + '{"name":"lookup","arguments":{}}' + TOOL_END + "\n<eos>"
    tokenizer = QwenToolTokenizer()
    ids = tokenizer.encode(text)
    parser = _parser()
    outputs = [
        parser.feed(tokenizer.decode(ids[i:i + chunk_size], skip_special_tokens=False), ids[i:i + chunk_size])
        for i in range(0, len(ids), chunk_size)
    ]
    outputs.append(parser.finish())

    assert "".join(output.content for output in outputs) == ""
    assert len(outputs[-1].tool_calls) == 1
    assert _parser().parse_complete(text, ids).content == ""


def test_none_hides_tool_calls_without_hiding_answer():
    text = "before" + TOOL_START + '{"name":"lookup","arguments":{}}' + TOOL_END + "after"
    parsed = _parser(choice="none").parse_complete(text, QwenToolTokenizer().encode(text))
    assert parsed.content == "beforeafter"
    assert parsed.tool_calls == ()


@pytest.mark.parametrize("text", [
    "\n  answer",
    TOOL_START + '{"name":"lookup","arguments":{}}' + TOOL_END + "\n  answer",
])
@pytest.mark.parametrize("chunk_size", [1, 7, 100000])
def test_answer_whitespace_is_preserved(text, chunk_size):
    tokenizer = QwenToolTokenizer()
    ids = tokenizer.encode(text)
    parser = _parser()
    outputs = [
        parser.feed(tokenizer.decode(ids[i:i + chunk_size], skip_special_tokens=False), ids[i:i + chunk_size])
        for i in range(0, len(ids), chunk_size)
    ]
    outputs.append(parser.finish())
    assert "".join(output.content for output in outputs) == "\n  answer"
    assert _parser().parse_complete(text, ids).content == "\n  answer"


@pytest.mark.parametrize("body", [
    '{"name":"lookup","arguments":[]}',
    '{"name":"lookup","arguments":{broken}}',
    '{"name":1,"arguments":{}}',
    '{"name":"weather.get","arguments":{}}',
    '{"name":"' + "x" * 65 + '","arguments":{}}',
])
def test_invalid_complete_call_fails_locally(body):
    text = TOOL_START + body + TOOL_END
    with pytest.raises(ValueError):
        _parser().parse_complete(text, QwenToolTokenizer().encode(text))


def test_truncation_does_not_publish_incomplete_call():
    text = TOOL_START + '{"name":"lookup","arguments":{"city":"Lon'
    parser = _parser()
    tokenizer = QwenToolTokenizer()
    assert not parser.feed(text, tokenizer.encode(text)).tool_call_deltas
    assert not parser.finish(truncated=True).tool_calls
    with pytest.raises(ValueError, match="incomplete Qwen3 tool call"):
        _parser().parse_complete(text, tokenizer.encode(text))
