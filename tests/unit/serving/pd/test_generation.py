# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Minimal generation contract and original-parameter preservation."""

import json
from types import SimpleNamespace

import msgspec
import pytest

from pypto_serving.serving.pd.config import PD_SCHEMA_VERSION, PDRole
from pypto_serving.serving.pd.generation import (
    MAX_NODE_BODY_BYTES, generation_body, parse_generation, public_document, request_fingerprint,
)
from pypto_serving.serving.pd.protocol import (
    BeginHandoff, ContinuationMetadata, GenerationContext, HandoffKey,
    decode_message, encode_message, prepared_request_hash,
)
from pypto_serving.serving.server.server import ServingServer
from pypto_serving.config.types import GenerateConfig


def _context(raw, kind="completion"):
    return GenerationContext(HandoffKey("request", "attempt", 1, 1, 1),
                             request_fingerprint(kind, public_document(raw)),
                             "p", 1, "d", 1, "localhost", 12345, PD_SCHEMA_VERSION)


@pytest.mark.parametrize("prompt", ("hello", [1, 2, 3]))
@pytest.mark.parametrize("explicit", (False, True))
def test_defaults_and_explicit_empty_values_are_not_materialized_or_overridden(prompt, explicit):
    value = {"prompt": prompt}
    if explicit:
        value.update(stop=[], top_k=None, seed=None, max_tokens=128)
    raw = msgspec.json.encode(value)
    public, context = parse_generation(generation_body(raw, "completion", _context(raw)), "completion")
    server = ServingServer(SimpleNamespace(), "model", GenerateConfig(
        max_new_tokens=256, stop=("default-stop",), top_k=3, seed=9,
    ))
    resolved = server._resolve_generate_config(public)
    assert resolved.stop == (() if explicit else ("default-stop",))
    assert resolved.top_k == (None if explicit else 3)
    assert resolved.seed == (None if explicit else 9)
    assert resolved.max_new_tokens == (128 if explicit else 256)
    restored = json.loads(generation_body(raw, "completion", context))
    restored.pop("pd_transfer")
    assert restored == value
    assert "temperature" not in public.model_fields_set


@pytest.mark.parametrize("failure", ("missing", "version", "unknown", "unknown_key", "public", "kind", "empty", "oversize"))
def test_invalid_or_mismatched_context_is_rejected(failure):
    raw = b'{"prompt":"hello"}'
    body = generation_body(raw, "completion", _context(raw))
    value = json.loads(body)
    kind = "completion"
    if failure == "missing":
        value.pop("pd_transfer")
    elif failure == "version":
        value["pd_transfer"]["schema_version"] -= 1
    elif failure == "unknown":
        value["pd_transfer"]["action"] = "prepare"
    elif failure == "unknown_key":
        value["pd_transfer"]["key"]["prepared_id"] = "not-supported"
    elif failure == "public":
        value["prompt"] = "another request"
    elif failure == "kind":
        kind = "chat"
    elif failure == "empty":
        body = b""
    elif failure == "oversize":
        body = b"x" * (MAX_NODE_BODY_BYTES + 1)
    if failure not in ("empty", "oversize"):
        body = msgspec.json.encode(value)
    with pytest.raises((ValueError, msgspec.ValidationError)):
        parse_generation(body, kind)


@pytest.mark.parametrize("role", (PDRole.PREFILL, PDRole.DECODE))
@pytest.mark.parametrize("field", ("generation", "route_epoch", "control_incarnation", "node_id"))
def test_context_is_bound_to_local_role_and_incarnation(role, field):
    context = _context(b'{"prompt":"hello"}')
    config = SimpleNamespace(role=role, node_id="p" if role is PDRole.PREFILL else "d",
                             generation=1, route_epoch=1, control_incarnation=1)
    context.validate_node(config)
    setattr(config, field, "foreign" if field == "node_id" else 2)
    with pytest.raises(ValueError, match="incarnation"):
        context.validate_node(config)


def test_normalized_control_roundtrip_and_digest_integrity():
    raw = b'{"prompt":[1,2,3]}'
    context = _context(raw)
    continuation = ContinuationMetadata((1, 2, 3), 128, 0., 1., None, None, (), None)
    begin = BeginHandoff(context.key, context, continuation, None, prepared_request_hash(continuation, None))
    assert decode_message(encode_message(begin)) == begin
    with pytest.raises(ValueError, match="digest"):
        BeginHandoff(context.key, context, continuation, None, "b" * 64)
    with pytest.raises(ValueError, match="identity"):
        BeginHandoff(HandoffKey("other", "attempt", 1, 1, 1), context, continuation, None, begin.prepared_digest)


def test_normalized_control_rejects_changed_grammar_before_reservation():
    raw = b'{"messages":[{"role":"user","content":"hello"}]}'
    context = _context(raw, kind="chat")
    continuation = ContinuationMetadata(
        (1, 2, 3), 128, 0., 1., None, None, (), None,
        constraint_spec={"provider_id": "xgrammar", "tool_choice": "required"},
    )
    digest = prepared_request_hash(continuation, None)
    altered = msgspec.structs.replace(
        continuation, constraint_spec={"provider_id": "xgrammar", "tool_choice": "auto"},
    )
    with pytest.raises(ValueError, match="digest"):
        BeginHandoff(context.key, context, altered, None, digest)


def test_public_client_cannot_supply_router_assignment():
    raw = b'{"prompt":"hello"}'
    body = generation_body(raw, "completion", _context(raw))
    with pytest.raises(ValueError, match="reserved"):
        generation_body(body, "completion", _context(raw))
