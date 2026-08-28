# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Lightweight IPC protocol for the engine↔worker boundary.

Replaces the old pickle-serialised full-graph path with
msgspec structs that ship only per-step deltas.  Prompt tokens and sampling
parameters are registered once per request (``NewRequestData``) and cached
inside the worker, so steady-state decode steps carry only:
  - per-request: last token, prev token, seq_len, block_ids
  - ~1 KB total instead of ~160 KB with the old full-graph pickle

Wire format: msgpack (msgspec), single encoded bytes object placed on the
existing ``multiprocessing.Queue``.  Switching the queue to a raw ``Pipe``
(Tier 2) is a drop-in swap at the ``encode_command`` / ``decode_command``
call sites.
"""

from __future__ import annotations

from typing import Union

import msgspec

# Sentinel for a decode input token the worker must resolve from its own cache.
# Under async (pipelined) scheduling the engine builds step N+1 before step N's
# sampled token is known, so it sends this placeholder and the worker substitutes
# the token it last sampled for that request.
PLACEHOLDER_TOKEN: int = -1


# ---------------------------------------------------------------------------
# Request-scoped structs (engine → worker)
# ---------------------------------------------------------------------------

class NewRequestData(msgspec.Struct):
    """Full request data sent exactly once when a request is first scheduled.

    The worker caches this and references it by ``request_id`` for the lifetime
    of the request.  ``prompt_token_ids`` are never re-sent after this message.
    """

    request_id: str
    prompt_token_ids: list[int]
    temperature: float
    top_p: float
    top_k: int | None
    seed: int | None = None


class PrefillRequest(msgspec.Struct):
    """Per-request payload for a prefill (or chunked-prefill) step."""

    request_id: str
    # Token chunk to compute this step: prompt_token_ids[num_computed : num_computed+num_new]
    chunk_tokens: list[int]
    # Absolute position of the first token in this chunk (for RoPE).
    num_computed_tokens: int
    # KV-cache block table for this request (may grow step-over-step).
    block_ids: list[int]
    # Model-specific grouped cache tables and their stable rank partition.
    block_ids_by_group: dict[str, list[int]] = msgspec.field(default_factory=dict)
    cache_partition: int | None = None


class DecodeRequest(msgspec.Struct):
    """Per-request payload for a decode step — delta only, no prompt tokens.

    Under async scheduling ``last_token`` may be ``PLACEHOLDER_TOKEN`` (-1): the
    engine scheduled this step before the prior step's token was sampled, so the
    worker substitutes the token it last committed for this request.
    """

    request_id: str
    # output_token_ids[-1] (the token to decode from), or PLACEHOLDER_TOKEN.
    last_token: int
    # Total tokens computed so far: num_prompt_tokens + len(output_token_ids).
    # Recomputed worker-side on the placeholder path (speculative decoders commit
    # a variable number of tokens per step, which the engine cannot know).
    seq_len: int
    # Full KV block table for this request.
    block_ids: list[int]
    # Model-specific grouped cache tables and their stable rank partition.
    block_ids_by_group: dict[str, list[int]] = msgspec.field(default_factory=dict)
    cache_partition: int | None = None


# ---------------------------------------------------------------------------
# Step-level commands (engine → worker)
# ---------------------------------------------------------------------------

class StepCommand(msgspec.Struct, tag="step"):
    """One scheduling step: admits new requests and runs prefill + decode.

    ``new_requests`` is non-empty only when freshly admitted requests enter the
    running state.  During steady-state decode it is always empty, keeping the
    payload at ~1 KB regardless of prompt length or batch size.
    """

    new_requests: list[NewRequestData]
    prefill_requests: list[PrefillRequest]
    decode_requests: list[DecodeRequest]
    # Request IDs that finished last step; worker releases device resources.
    finished_request_ids: list[str]
    # Monotonic step counter, echoed back on StepResult. Cheap ordering guard for
    # the pipelined loop (the worker is FIFO, so results must return in order).
    step_id: int = 0


class ShutdownCommand(msgspec.Struct, tag="shutdown"):
    """Signals the worker to exit its busy-loop cleanly."""


class ProfileCommand(msgspec.Struct, tag="profile"):
    """Starts or stops the process-local SA profiler."""

    active: bool


# Union used for the decoder — tag field ("type") discriminates.
Command = Union[StepCommand, ShutdownCommand, ProfileCommand]


# ---------------------------------------------------------------------------
# Step result (worker → engine)
# ---------------------------------------------------------------------------

class StepResult(msgspec.Struct):
    """Sampled tokens returned after executing one step.

    Values are always ``list[int]``:
    - Standard decode: ``[token_id]`` (single element).
    - MTP / speculative: ``[t0, t1, ...]`` (multiple accepted tokens).

    The engine merges this back into the ``dict[str, int | list[int]]`` that
    ``scheduler.update_from_output`` expects.
    """

    new_tokens: dict[str, list[int]]
    error: str | None = None
    # Echoes the originating StepCommand.step_id (pipeline ordering guard).
    step_id: int = 0


class ProfileResult(msgspec.Struct):
    """Acknowledges a profile command after the worker has applied it."""

    active: bool
    error: str | None = None


# ---------------------------------------------------------------------------
# Codec — thin wrappers so call sites are transport-agnostic
# ---------------------------------------------------------------------------

_cmd_encoder: msgspec.msgpack.Encoder = msgspec.msgpack.Encoder()
_cmd_decoder: msgspec.msgpack.Decoder = msgspec.msgpack.Decoder(Command)
_result_encoder: msgspec.msgpack.Encoder = msgspec.msgpack.Encoder()
_result_decoder: msgspec.msgpack.Decoder = msgspec.msgpack.Decoder(StepResult)
_profile_result_encoder: msgspec.msgpack.Encoder = msgspec.msgpack.Encoder()
_profile_result_decoder: msgspec.msgpack.Decoder = msgspec.msgpack.Decoder(ProfileResult)


def encode_command(cmd: Command) -> bytes:
    """Encode a command to bytes for queue transport."""
    return _cmd_encoder.encode(cmd)


def decode_command(data: bytes) -> Command:
    """Decode bytes from the input queue into a typed command."""
    return _cmd_decoder.decode(data)


def encode_result(result: StepResult) -> bytes:
    """Encode a step result to bytes for queue transport."""
    return _result_encoder.encode(result)


def decode_result(data: bytes) -> StepResult:
    """Decode bytes from the output queue into a typed step result."""
    return _result_decoder.decode(data)


def encode_profile_result(result: ProfileResult) -> bytes:
    """Encode a profile-control acknowledgement."""
    return _profile_result_encoder.encode(result)


def decode_profile_result(data: bytes) -> ProfileResult:
    """Decode a profile-control acknowledgement."""
    return _profile_result_decoder.decode(data)
