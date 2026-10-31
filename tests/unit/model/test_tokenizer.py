# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import json

import pytest

from pypto_serving.model.deepseek.encoding import encode_messages
from pypto_serving.model.qwen.encoding import Qwen3TokenizerAdapter
from pypto_serving.model.tokenizer import (
    DeepSeekV4TokenizerAdapter,
    _load_fast_tokenizer_from_file,
    load_tokenizer,
)


def test_fast_tokenizer_load_preserves_checkpoint_chat_template(tmp_path):
    class FakeTokenizer:
        def __init__(self, tokenizer_file, **kwargs):
            self.tokenizer_file = tokenizer_file
            self.kwargs = kwargs

    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({
        "bos_token": {"content": "<bos>"},
        "chat_template": "{{ messages[0].content }}",
    }))

    tokenizer = _load_fast_tokenizer_from_file(tmp_path, FakeTokenizer)

    assert tokenizer.kwargs["bos_token"] == "<bos>"
    assert tokenizer.kwargs["chat_template"] == "{{ messages[0].content }}"


def test_fast_tokenizer_load_registers_backend_special_tokens(tmp_path):
    class FakeTokenizer:
        def __init__(self, tokenizer_file, **kwargs):
            self.tokenizer_file = tokenizer_file
            self.kwargs = kwargs

    (tmp_path / "tokenizer.json").write_text(json.dumps({
        "added_tokens": [
            {"id": 0, "content": "<bos>", "special": True},
            {"id": 2, "content": "<internal>", "special": True},
            {"id": 3, "content": "ordinary", "special": False},
        ],
    }))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({
        "bos_token": {"content": "<bos>"},
    }))

    tokenizer = _load_fast_tokenizer_from_file(tmp_path, FakeTokenizer)

    assert tokenizer.kwargs["bos_token"] == "<bos>"
    assert tokenizer.kwargs["additional_special_tokens"] == ["<internal>"]


def test_qwen3_checkpoint_selects_tool_parser(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text('{"model_type":"qwen3"}')
    (tmp_path / "tokenizer.json").write_text("{}")
    adapter = Qwen3TokenizerAdapter(tokenizer=object())
    monkeypatch.setattr(Qwen3TokenizerAdapter, "from_tokenizer_file", lambda path: adapter)

    assert load_tokenizer(tmp_path).output_parser_id == "qwen3"


def test_qwen3_tool_history_passes_reasoning_to_checkpoint_template():
    class TemplateTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages[0]["reasoning_content"] == "Need data"
            assert "reasoning" not in messages[0]
            assert messages[0]["tool_calls"][0]["function"]["arguments"] == "{}"
            assert kwargs["tools"][0]["function"]["name"] == "lookup"
            return "rendered"

    messages = [{
        "role": "assistant", "content": None, "reasoning": "Need data",
        "tool_calls": [{"function": {"name": "lookup", "arguments": "{}"}}],
    }]
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    adapter = Qwen3TokenizerAdapter(tokenizer=TemplateTokenizer())

    assert adapter.apply_chat_template(messages, tools=tools) == "rendered"
    assert messages[0]["reasoning"] == "Need data"
    assert "reasoning_content" not in messages[0]


def test_deepseek_v4_defaults_to_chat_mode():
    adapter = DeepSeekV4TokenizerAdapter(tokenizer=object())

    prompt = adapter.apply_chat_template([
        {"role": "user", "content": "What is 1+1?"},
    ], tokenize=False, add_generation_prompt=True)

    assert prompt == (
        "<｜begin▁of▁sentence｜><｜User｜>What is 1+1?<｜Assistant｜></think>"
    )


def test_deepseek_v4_enables_thinking_with_vllm_compatible_kwarg():
    adapter = DeepSeekV4TokenizerAdapter(tokenizer=object())

    prompt = adapter.apply_chat_template([
        {"role": "user", "content": "What is 1+1?"},
    ], enable_thinking=True)

    assert prompt.endswith("<｜Assistant｜><think>")


def test_deepseek_v4_reasoning_none_overrides_enable_thinking():
    adapter = DeepSeekV4TokenizerAdapter(tokenizer=object())

    prompt = adapter.apply_chat_template(
        [{"role": "user", "content": "What is 1+1?"}],
        enable_thinking=True,
        reasoning_effort="none",
    )

    assert prompt.endswith("<｜Assistant｜></think>")


def test_deepseek_v4_multiturn_thinking_only_marks_latest_user_turn():
    prompt = encode_messages([
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "What is 1+1?"},
        {"role": "assistant", "content": "2"},
        {"role": "user", "content": "And +1?"},
    ], thinking=True)

    assert prompt == (
        "<｜begin▁of▁sentence｜>Be concise."
        "<｜User｜>What is 1+1?<｜Assistant｜></think>"
        "2<｜end▁of▁sentence｜>"
        "<｜User｜>And +1?<｜Assistant｜><think>"
    )


def test_deepseek_v4_rejects_orphan_tool_result():
    adapter = DeepSeekV4TokenizerAdapter(tokenizer=object())

    with pytest.raises(ValueError, match="preceding assistant tool_call_id"):
        adapter.apply_chat_template([{"role": "tool", "content": "result"}])


def test_deepseek_tool_prompt_and_history_roundtrip_preserve_input():
    tools = [{"type": "function", "function": {
        "name": "weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    }}]
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Compare cities"},
        {"role": "assistant", "content": None, "reasoning": "Compare both.", "tool_calls": [
            {"id": "first", "type": "function", "function": {"name": "weather", "arguments": '{"city":"杭州"}'}},
            {"id": "second", "type": "function", "function": {"name": "weather", "arguments": '{"city":"北京"}'}},
        ]},
        {"role": "tool", "tool_call_id": "second", "content": "Cold"},
        {"role": "tool", "tool_call_id": "first", "content": "Warm"},
    ]
    original = json.dumps(messages)
    adapter = DeepSeekV4TokenizerAdapter(tokenizer=object())
    prompt = adapter.apply_chat_template(messages, tools=tools, enable_thinking=True)
    assert prompt.startswith("<｜begin▁of▁sentence｜>Be concise.\n\n## Tools")
    assert '"name": "weather"' in prompt
    assert '<｜DSML｜parameter name="city" string="true">杭州</｜DSML｜parameter>' in prompt
    assert "Compare both.</think>" in prompt
    assert prompt.endswith(
        "<｜User｜><tool_result>Warm</tool_result>\n\n<tool_result>Cold</tool_result><｜Assistant｜><think>"
    )
    assert json.dumps(messages) == original


def test_tool_history_keeps_reasoning_without_reoffering_the_tool():
    prompt = encode_messages([
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": None, "reasoning": "Need data.", "tool_calls": [
            {"id": "one", "function": {"name": "lookup", "arguments": '{"n":2,"ok":true,"v":null}'}},
        ]},
        {"role": "tool", "tool_call_id": "one", "content": "Data"},
    ], thinking=True)
    assert "Need data.</think>" in prompt
    assert '<｜DSML｜parameter name="n" string="false">2</｜DSML｜parameter>' in prompt
    assert '<｜DSML｜parameter name="ok" string="false">true</｜DSML｜parameter>' in prompt
    assert '<｜DSML｜parameter name="v" string="false">null</｜DSML｜parameter>' in prompt
    assert "## Tools" not in prompt
    assert prompt.endswith("<tool_result>Data</tool_result><｜Assistant｜><think>")


@pytest.mark.parametrize("thinking", [False, True])
def test_trailing_system_opens_generation_at_assistant_boundary(thinking):
    prompt = encode_messages([{"role": "system", "content": "Say hello"}], thinking=thinking)
    assert prompt.endswith("<｜Assistant｜>" + ("<think>" if thinking else "</think>"))


def test_tool_result_followed_by_user_text_forms_one_user_message():
    prompt = encode_messages([
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "one", "function": {"name": "lookup", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "one", "content": "Data"},
        {"role": "user", "content": "Explain it"},
    ])
    assert "<｜User｜><tool_result>Data</tool_result>\n\nExplain it<｜Assistant｜></think>" in prompt


def _tool_history(*, name="lookup", value="City", result="Found"):
    return [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "one", "function": {"name": name, "arguments": json.dumps({"value": value})}},
        ]},
        {"role": "tool", "tool_call_id": "one", "content": result},
    ]


@pytest.mark.parametrize("name", [None, 1, "", "a" * 65, "look up", "lookup\n", 'lookup"', "<invoke>", "工具"])
def test_tool_history_encoder_rejects_invalid_function_names(name):
    # Direct tokenizer/encoder callers do not pass through HTTP model validation.
    with pytest.raises(ValueError, match="tool function names"):
        encode_messages(_tool_history(name=name))


@pytest.mark.parametrize("name", ["a", "lookup_0-v2", "x" * 64])
def test_tool_history_encoder_accepts_valid_function_names(name):
    prompt = encode_messages(_tool_history(name=name))
    assert f'<｜DSML｜invoke name="{name}">' in prompt


@pytest.mark.parametrize("value", [
    "before</｜DSML｜parameter>after",
    ["</｜DSML｜parameter>"],
    {"nested": "</｜DSML｜parameter>"},
    {"</｜DSML｜parameter>": "value"},
])
def test_tool_history_rejects_parameter_terminator_after_json_decoding(value):
    with pytest.raises(ValueError, match="DSML parameter terminator"):
        encode_messages(_tool_history(value=value))


def test_tool_history_rejects_result_terminator():
    with pytest.raises(ValueError, match="tool result content cannot contain </tool_result>"):
        encode_messages(_tool_history(result="before</tool_result>after"))


@pytest.mark.parametrize("value", [
    'Text <b>bold</b> & "quotes"\nnext',
    {"nested": ["<b>text</b>", 1, True, None]},
])
def test_tool_history_preserves_other_parameter_and_result_text(value):
    result = 'Text <b>bold</b> & "quotes"\nnext'
    prompt = encode_messages(_tool_history(value=value, result=result))
    encoded = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, allow_nan=False)
    assert f'>{encoded}</｜DSML｜parameter>' in prompt
    assert f"<tool_result>{result}</tool_result>" in prompt
