# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Request-level constraint selection without an HTTP client or device worker."""

from types import SimpleNamespace

import pytest

from pypto_serving.config.types import GenerateConfig
from pypto_serving.serving.reasoning import OutputParserSpec
from pypto_serving.serving.server import server as server_module
from pypto_serving.serving.server.api_types import ChatCompletionRequest, validate_chat_request


def _request(*, choice=None, strict=False):
    tool = {"type": "function", "function": {"name": "lookup", "strict": strict}}
    return ChatCompletionRequest(
        messages=[{"role": "user", "content": "Find a city"}],
        tools=[tool], tool_choice=choice,
    )


@pytest.mark.parametrize("choice,strict,constrained", [
    (None, False, False),
    ("auto", True, True),
    ("required", False, True),
    ({"type": "function", "function": {"name": "lookup"}}, False, True),
    ("none", True, False),
])
def test_constraint_is_per_request_and_ordinary_auto_stays_unconstrained(
    monkeypatch, choice, strict, constrained,
):
    monkeypatch.setattr(server_module.importlib.util, "find_spec", lambda name: object())
    engine = SimpleNamespace(config=SimpleNamespace(executor_cls="PyptoDeepSeekV4DSparkExecutor"))
    serving = server_module.ServingServer(engine, "model", GenerateConfig())
    request = _request(choice=choice, strict=strict)
    parser = OutputParserSpec("deepseek_v4", "content", tool_choice="auto", tool_names=("lookup",))
    spec = serving._constraint_spec(request, parser)
    assert (spec is not None) is constrained
    if spec is not None:
        assert spec.tool_choice == validate_chat_request(request)
        assert spec.provider_id == "xgrammar"


@pytest.mark.parametrize("choice", [
    {"type": "function", "function": {"name": "missing"}},
    {"type": "function", "function": {"name": "lookup", "extra": True}},
    "unknown",
])
def test_invalid_tool_choice_is_rejected_before_generation(choice):
    with pytest.raises(ValueError):
        validate_chat_request(_request(choice=choice))


def test_required_choice_needs_tools():
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "Find a city"}], tool_choice="required",
    )
    with pytest.raises(ValueError, match="non-empty tools"):
        validate_chat_request(request)


def test_default_tool_request_does_not_load_xgrammar(monkeypatch):
    def unexpected_import(_name):
        raise AssertionError("ordinary auto must not inspect XGrammar")

    monkeypatch.setattr(server_module.importlib.util, "find_spec", unexpected_import)
    serving = server_module.ServingServer(SimpleNamespace(), "model", GenerateConfig())
    parser = OutputParserSpec("deepseek_v4", "content", tool_choice="auto", tool_names=("lookup",))
    assert serving._constraint_spec(_request(), parser) is None
