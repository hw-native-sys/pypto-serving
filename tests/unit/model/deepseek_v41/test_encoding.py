# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pinned official V4.1 text-prompt goldens and serving adapter input contracts."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[4]
MODEL_ROOT = ROOT / "pypto_serving/model"
FIXTURE = json.loads((ROOT / "tests/fixtures/deepseek_v41/chat_encoding_golden.json").read_text())


def load(name, path, monkeypatch=None):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    if monkeypatch is not None:
        monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


encoding = load("v41_text_encoding_test", MODEL_ROOT / "deepseek_v41/encoding.py")


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda case: case["name"])
def test_pinned_official_text_prompt(case):
    before = copy.deepcopy(case["messages"])
    actual = encoding.encode_messages(case["messages"], **case["options"])
    assert actual == case["expected"]
    assert actual.encode("utf-8") == case["expected"].encode("utf-8")
    assert case["messages"] == before


def test_golden_provenance_is_the_pinned_encoder():
    assert FIXTURE["revision"] == encoding.REFERENCE_REVISION
    assert FIXTURE["source_sha256"] == encoding.REFERENCE_SHA256
    assert f"/blob/{encoding.REFERENCE_REVISION}/encoding/encoding.py" in FIXTURE["source"]


@pytest.mark.parametrize(
    "messages,match",
    [
        ([], "at least one"),
        (["user"], "objects"),
        ([{"role": "developer", "content": "rules"}], "role"),
        ([{"role": "tool", "content": "result"}], "role"),
        ([{"role": "user"}], "string"),
        ([{"role": "user", "content": [{"type": "text", "text": "hi"}]}], "string"),
        ([{"role": "user", "content": "hi", "tools": []}], "fields"),
        ([{"role": "assistant", "content": "hi", "tool_calls": []}], "fields"),
        ([{"role": "system", "content": "hi", "response_format": {}}], "fields"),
        ([{"role": "user", "content": "hi", "content_blocks": []}], "fields"),
        ([{"role": "user", "content": "hi", "task": "action"}], "fields"),
        ([{"role": "user", "content": "hi", "reasoning_content": "why"}], "assistant"),
        ([{"role": "assistant", "content": "hi", "reasoning_content": []}], "assistant"),
        ([{"role": "assistant", "content": "hi", "wo_eos": 1}], "bool"),
        ([{"role": "user", "content": encoding.IMAGE_TOKEN}], "image"),
        ([{"role": "user", "content": "describe <image>cat.jpg</image>"}], "image"),
        ([{"role": "user", "content": "malformed </image>"}], "image"),
        ([{"role": "assistant", "content": "hi", "reasoning_content": encoding.IMAGE_TOKEN}], "image"),
    ],
)
def test_unsupported_or_invalid_message_is_explicit(messages, match):
    with pytest.raises(ValueError, match=match):
        encoding.encode_messages(messages)


@pytest.mark.parametrize("effort", ["medium", "xhigh", "none", "50", 0, 101, True, 75.0, []])
def test_invalid_reference_reasoning_effort(effort):
    with pytest.raises(ValueError, match="reasoning_effort"):
        encoding.encode_messages([{"role": "user", "content": "hi"}], reasoning_effort=effort)


@pytest.fixture
def adapter(monkeypatch):
    # Load the actual shared adapter without executing the torch-dependent package initializer.
    for name, path in (
        ("pypto_serving", ROOT / "pypto_serving"),
        ("pypto_serving.model", MODEL_ROOT),
        ("pypto_serving.model.deepseek_v41", MODEL_ROOT / "deepseek_v41"),
    ):
        package = ModuleType(name)
        package.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, name, package)
    for suffix, path in (
        ("model_family", "model_family.py"),
        ("tokenizer", "tokenizer.py"),
        ("deepseek_v41.encoding", "deepseek_v41/encoding.py"),
        ("deepseek_v41.tokenizer", "deepseek_v41/tokenizer.py"),
    ):
        module = load(f"pypto_serving.model.{suffix}", MODEL_ROOT / path, monkeypatch)

    class RecordingTokenizer:
        def __init__(self):
            self.calls = []

        def encode(self, text, *, add_special_tokens):
            self.calls.append((text, add_special_tokens))
            return [100, 200, 300]

    return module.DeepSeekV41TokenizerAdapter(RecordingTokenizer())


@pytest.mark.parametrize(
    "options,budget",
    [
        ({"thinking": True}, 75),
        ({"enable_thinking": True, "reasoning_effort": "low"}, 50),
        ({"enable_thinking": True, "reasoning_effort": "high"}, 75),
        ({"thinking_mode": "thinking", "reasoning_effort": "max"}, 100),
        ({"thinking_mode": "thinking", "reasoning_effort": 1}, 1),
        ({"thinking_mode": "thinking", "reasoning_effort": 100}, 100),
    ],
)
def test_server_thinking_options(adapter, options, budget):
    actual = adapter.apply_chat_template(
        [{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True, **options
    )
    assert actual == (
        encoding.BOS_TOKEN + encoding.SYSTEM_TOKEN + f"Reasoning Effort: {budget} "
        "(range 1-100, the higher the value, the more thorough the reasoning)\n\n"
        + encoding.USER_TOKEN
        + "hi"
        + encoding.ASSISTANT_TOKEN
        + "<think>"
    )


@pytest.mark.parametrize(
    "options", [{}, {"enable_thinking": False}, {"thinking": True, "reasoning_effort": "none"}]
)
def test_chat_and_none_budget(adapter, options):
    case = FIXTURE["cases"][0]
    assert adapter.apply_chat_template(case["messages"], **options) == case["expected"]


def test_tokenize_preserves_single_official_bos(adapter):
    case = FIXTURE["cases"][0]
    assert adapter.apply_chat_template(case["messages"], tokenize=True) == [100, 200, 300]
    assert adapter.tokenizer.calls == [(case["expected"], False)]


@pytest.mark.parametrize(
    "options,match",
    [
        ({"enable_thinking": 1}, "bool"),
        ({"tokenize": 1}, "bool"),
        ({"drop_thinking": 0}, "bool"),
        ({"add_generation_prompt": False}, "add_generation_prompt=True"),
        ({"thinking": True, "enable_thinking": False}, "agree"),
        ({"thinking_mode": "chat", "thinking": True}, "conflicts"),
        ({"thinking_mode": "invalid", "reasoning_effort": "none"}, "thinking_mode"),
        ({"reasoning_effort": "medium"}, "reasoning_effort"),
        ({"reasoning_effort": "xhigh"}, "reasoning_effort"),
        ({"tools": []}, "options"),
        ({"return_tensors": "pt"}, "options"),
    ],
)
def test_adapter_rejects_unsupported_options(adapter, options, match):
    with pytest.raises(ValueError, match=match):
        adapter.apply_chat_template([{"role": "user", "content": "hi"}], **options)


@pytest.mark.parametrize(
    "family,expected",
    [
        ("deepseek_v41", "DeepSeekV41TokenizerAdapter"),
        ("deepseek_v4", "DeepSeekV4TokenizerAdapter"),
        ("qwen3", "TransformersTokenizerAdapter"),
    ],
)
def test_shared_tokenizer_loader_dispatches_independent_adapter(
    adapter, tmp_path, monkeypatch, family, expected
):
    shared = sys.modules["pypto_serving.model.tokenizer"]
    monkeypatch.setattr(
        shared.TransformersTokenizerAdapter,
        "from_tokenizer_file",
        classmethod(lambda cls, model_dir: cls(object())),
    )
    (tmp_path / "config.json").write_text(json.dumps({"model_type": family}))
    (tmp_path / "tokenizer.json").write_text("{}")
    result = shared.load_tokenizer(tmp_path)
    assert type(result).__name__ == expected
