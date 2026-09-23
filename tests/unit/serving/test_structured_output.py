# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Exercise real grammar masks and speculative rollback without model weights."""

import copy
from types import SimpleNamespace

import pytest
import torch

from pypto_serving.serving.structured_output import TokenConstraint, tool_grammar

xgr = pytest.importorskip("xgrammar")

TOOLS = [{"type": "function", "function": {
    "name": "shell", "strict": False,
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                   "required": ["command"], "additionalProperties": False},
}}]
START = '\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name="shell">\n'
END = '</｜DSML｜invoke>\n</｜DSML｜tool_calls>'
PARAM = '<｜DSML｜parameter name="command" string="true">pwd</｜DSML｜parameter>\n'


@pytest.fixture(scope="module")
def compiler():
    info = xgr.TokenizerInfo([bytes([i]) for i in range(256)] + [b"<eos>"], stop_token_ids=[256])
    return xgr.GrammarCompiler(info, max_threads=1)


def constraint(compiler, choice="required", thinking=False):
    grammar = tool_grammar(TOOLS, choice, thinking=thinking, parallel=False, enforce=True)
    return TokenConstraint(xgr.GrammarMatcher(compiler.compile_structural_tag(grammar)), 257, 256)


def test_auto_opt_in_and_server_override_do_not_mutate_client_schema():
    original = copy.deepcopy(TOOLS)
    assert tool_grammar(TOOLS, "auto", thinking=False, parallel=True) is None
    assert tool_grammar(TOOLS, "none", thinking=False, parallel=True, enforce=True) is None
    assert tool_grammar(TOOLS, "auto", thinking=False, parallel=True, enforce=True)
    assert TOOLS == original
    strict = copy.deepcopy(TOOLS)
    strict[0]["function"]["strict"] = True
    assert tool_grammar(strict, "auto", thinking=False, parallel=True)


@pytest.mark.parametrize("body,valid", [
    (PARAM, True),
    (PARAM.replace('name="command"', 'name="cmd"'), False),
    ("", False),
    (PARAM.replace('string="true"', 'string="false"').replace('>pwd<', '>123<'), False),
])
def test_required_command_schema(compiler, body, valid):
    assert constraint(compiler).matcher.accept_string(START + body + END) == valid


def test_named_tool_and_auto_normal_answer(compiler):
    named = {"type": "function", "function": {"name": "shell"}}
    assert constraint(compiler, named).matcher.accept_string(START + PARAM + END)
    assert not constraint(compiler, named).matcher.accept_string(START.replace('"shell"', '"execute"'))
    normal = constraint(compiler, "auto")
    normal.accept(list(b"I will analyze the repository."))
    normal.accept([256])
    assert normal.matcher.is_terminated()


def test_thinking_prefix_then_tool_call(compiler):
    state = constraint(compiler, thinking=True)
    assert state.matcher.accept_string("Need to inspect files.</think>" + START + PARAM + END)


def test_mask_blocks_wrong_parameter_before_sampling(compiler):
    state = constraint(compiler)
    assert state.matcher.accept_string(START + '<｜DSML｜parameter name="c')
    logits = torch.zeros(257)
    logits[ord("m")] = 1000  # The model wants cmd; schema requires command.
    logits[ord("o")] = 1
    masked = state.mask_logits(logits)
    assert torch.isneginf(masked[ord("m")])
    assert int(masked.argmax()) == ord("o")


def test_draft_preview_rolls_back_and_commits_only_accepted_prefix(compiler):
    state = constraint(compiler)
    assert state.matcher.accept_string(START + '<｜DSML｜parameter name="c')
    before = state.bitmask().clone()
    masks = state.preview([ord("o"), ord("x"), ord("z")], 4)
    assert torch.equal(before, state.bitmask())
    assert (int(masks[0, ord("o") // 32]) >> (ord("o") % 32)) & 1
    assert not (int(masks[1, ord("x") // 32]) >> (ord("x") % 32)) & 1
    assert not (int(before[ord("m") // 32]) >> (ord("m") % 32)) & 1
    state.accept([ord("o"), ord("m")])
    assert not torch.equal(before, state.bitmask())
    with pytest.raises(RuntimeError, match="outside the tool schema"):
        state.accept([ord("x")])


def test_eos_only_after_complete_required_call(compiler):
    state = constraint(compiler)
    assert not (int(state.bitmask()[8]) & 1)
    state.accept(list((START + PARAM + END).encode()))
    state.accept([256])
    mask = state.bitmask()
    assert mask.tolist() == [0] * 8 + [1]
    assert torch.equal(state.preview([256, 256], 3), mask.repeat(3, 1))
    state.accept([256])


def test_worker_samples_only_allowed_tokens_and_commits_matcher(compiler):
    from pypto_serving.config.types import SamplingParams
    from pypto_serving.model.common.executor.sampler import Sampler
    from pypto_serving.serving.server.serving_worker import WorkerProcess

    state = constraint(compiler)
    assert state.matcher.accept_string(START + '<｜DSML｜parameter name="c')
    worker = WorkerProcess.__new__(WorkerProcess)
    worker._tool_constraints = {"r": state}
    worker.sampler = Sampler()
    worker._last_tokens = {}
    logits = torch.zeros(257)
    logits[ord("m")] = 1000
    logits[ord("o")] = 1
    token = worker._sample_result_row(
        SimpleNamespace(), logits, SamplingParams(temperature=0, top_p=1, top_k=None), "r", 0,
        allow_device_sampled=False, allow_device_topk_sampling=False,
    )
    assert token == ord("o")
    worker._record_last_tokens("r", [token])
    assert worker._last_tokens["r"] == [token]


@pytest.mark.parametrize("value,valid", [("2", True), ('"2"', False), ("0", False)])
def test_nested_json_schema_and_local_refs(compiler, value, valid):
    tools = copy.deepcopy(TOOLS)
    schema = tools[0]["function"]["parameters"]
    schema["properties"]["command"] = {"$ref": "#/$defs/options"}
    schema["$defs"] = {"options": {"type": "object", "properties": {
        "count": {"type": "integer", "minimum": 1}}, "required": ["count"], "additionalProperties": False}}
    grammar = tool_grammar(tools, "required", thinking=False, parallel=False, enforce=True)
    state = xgr.GrammarMatcher(compiler.compile_structural_tag(grammar))
    parameter = '<｜DSML｜parameter name="command" string="false">{"count":' + value + '}</｜DSML｜parameter>\n'
    assert state.accept_string(START + parameter + END) == valid


def test_unsupported_root_schema_is_explicit_error():
    tools = copy.deepcopy(TOOLS)
    tools[0]["function"]["parameters"]["oneOf"] = []
    with pytest.raises(ValueError, match="root schema keywords"):
        tool_grammar(tools, "auto", thinking=False, parallel=True, enforce=True)


def test_worker_constraint_lifecycle_and_ipc(compiler, monkeypatch):
    from pypto_serving.serving.server.ipc import (
        NewRequestData, PrefillRequest, StepCommand, encode_command, decode_command,
    )
    from pypto_serving.serving.server.serving_worker import WorkerProcess
    from pypto_serving.serving import structured_output

    state = constraint(compiler)
    created = []
    attached = []
    factory = SimpleNamespace(create=lambda grammar: created.append(grammar) or state)
    monkeypatch.setattr(structured_output, "ConstraintCompiler", lambda *args: factory)
    worker = WorkerProcess.__new__(WorkerProcess)
    request = NewRequestData("r", [1], 0, 1, None, tool_grammar="grammar")
    cmd = StepCommand(new_requests=[request], prefill_requests=[PrefillRequest("r", [1], 0, [])],
                      decode_requests=[], finished_request_ids=[])
    cmd = decode_command(encode_command(cmd))
    assert cmd.new_requests[0].tool_grammar == "grammar"
    worker._req_cache = {"r": cmd.new_requests[0]}
    worker._last_tokens = {}
    worker.sampler = SimpleNamespace()
    worker.model_record = SimpleNamespace(config=SimpleNamespace(vocab_size=257, model_id="model"),
                                          tokenizer=object())
    worker.executor = SimpleNamespace(set_request_constraint=lambda *args: attached.append(args))
    worker._initialize_tool_constraints(cmd)
    worker._initialize_tool_constraints(cmd)
    assert created == ["grammar"]
    assert attached == [("model", "r", state)]
    worker._release_finished_request_state(["r"])
    assert not worker._tool_constraints
    assert attached[-1] == ("model", "r", None)
