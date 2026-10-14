# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""One request extension; public parameters retain their ordinary defaults."""

from __future__ import annotations

import hashlib
import json
import msgspec

from pypto_serving.serving.server.api_types import CompletionRequest, ChatCompletionRequest
from .protocol import GenerationContext

MAX_NODE_BODY_BYTES = 4 << 20


def public_document(wire):
    if not wire or len(wire) > MAX_NODE_BODY_BYTES:
        raise ValueError("invalid bounded generation body")
    value = msgspec.json.decode(wire)
    if not isinstance(value, dict):
        raise ValueError("generation body must be an object")
    return value


def request_fingerprint(kind, public):
    # Do not materialize defaults: omission and explicit null/[] are different.
    wire = json.dumps((kind, public), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(wire).hexdigest()


def parse_generation(wire, kind):
    value = public_document(wire)
    try:
        context = msgspec.convert(value.pop("pd_transfer"), type=GenerationContext)
    except KeyError as exc:
        raise ValueError("PD node requires Router generation context") from exc
    if context.request_fingerprint != request_fingerprint(kind, value):
        raise ValueError("generation context differs from public request")
    request_type = ChatCompletionRequest if kind == "chat" else CompletionRequest
    return request_type.model_validate(value), context


def generation_body(wire, kind, context):
    value = public_document(wire)
    if "pd_transfer" in value:
        raise ValueError("pd_transfer is reserved for the Router")
    if context.request_fingerprint != request_fingerprint(kind, value):
        raise ValueError("Router request fingerprint mismatch")
    value["pd_transfer"] = msgspec.to_builtins(context)
    result = msgspec.json.encode(value)
    if len(result) > MAX_NODE_BODY_BYTES:
        raise ValueError("generation body exceeds node limit")
    return result
