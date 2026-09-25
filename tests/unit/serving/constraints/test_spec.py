# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Keep tool constraints request-local and independent of the output parser."""

import asyncio
from types import SimpleNamespace

import pytest

from pypto_serving.config.types import GenerateConfig
from pypto_serving.serving.constraints import ConstraintSpec
from pypto_serving.serving.server.ipc import NewRequestData, decode_command, encode_command, StepCommand
from pypto_serving.serving.server.server import ChatCompletionRequest, ServingServer


TOOLS = [{
    "type": "function",
    "function": {
        "name": "shell",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
            "additionalProperties": False,
        },
    },
}]


def _server(monkeypatch):
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object() if name == "xgrammar" else None)
    engine = SimpleNamespace(
        tokenizer=SimpleNamespace(output_parser_id="deepseek_v4"),
        config=SimpleNamespace(executor_cls="PyptoDeepSeekV4DSparkExecutor"),
    )
    return ServingServer(engine, "model", GenerateConfig())


def _request(**kwargs):
    payload = {"messages": [{"role": "user", "content": "Run ls"}], "tools": TOOLS}
    payload.update(kwargs)
    return ChatCompletionRequest(**payload)


@pytest.mark.parametrize("tool_choice,expected", [
    ("required", "required"),
    ({"type": "function", "function": {"name": "shell"}}, "required"),
])
def test_required_and_named_constraints_are_distinct_from_output_parser(monkeypatch, tool_choice, expected):
    server = _server(monkeypatch)
    request = _request(tool_choice=tool_choice)
    parser_spec = server._output_parser_spec(request)
    constraint_spec = server._constraint_spec(request, parser_spec)
    assert parser_spec.tool_choice == expected
    assert constraint_spec.tool_choice == tool_choice
    assert constraint_spec.format_id == "deepseek_v4"
    assert ConstraintSpec.from_wire(constraint_spec.to_wire()) == constraint_spec


def test_auto_only_activates_when_a_tool_is_strict(monkeypatch):
    server = _server(monkeypatch)
    request = _request()
    assert server._constraint_spec(request, server._output_parser_spec(request)) is None
    strict_request = _request(tools=[{
        "type": "function", "function": {**TOOLS[0]["function"], "strict": True},
    }])
    spec = server._constraint_spec(strict_request, server._output_parser_spec(strict_request))
    assert spec.tool_choice == "auto"


def test_constraint_spec_survives_worker_ipc():
    spec = ConstraintSpec(
        provider_id="xgrammar", format_id="deepseek_v4", tools=tuple(TOOLS),
        tool_choice="required", reasoning=False,
    )
    command = StepCommand(
        new_requests=[NewRequestData(
            request_id="r", prompt_token_ids=[1], temperature=0.0, top_p=1.0,
            top_k=None, constraint_spec=spec.to_wire(),
        )],
        prefill_requests=[], decode_requests=[], finished_request_ids=[],
    )
    restored = decode_command(encode_command(command))
    assert ConstraintSpec.from_wire(restored.new_requests[0].constraint_spec) == spec


def test_preflight_rejects_invalid_schema_before_worker_registration(monkeypatch):
    server = _server(monkeypatch)
    spec = server._constraint_spec(
        _request(tool_choice="required"),
        server._output_parser_spec(_request(tool_choice="required")),
    )
    calls = []

    class Provider:
        def __init__(self, tokenizer):
            calls.append(tokenizer)

        def compile(self, candidate):
            assert candidate is spec
            raise RuntimeError("unsupported schema")

    monkeypatch.setattr("pypto_serving.serving.server.server.XGrammarProvider", Provider)
    with pytest.raises(ValueError, match="invalid tool constraint: unsupported schema"):
        asyncio.run(server._preflight_constraint(spec))
    assert calls == [server.engine.tokenizer]
