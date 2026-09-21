# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Address-free HTTP contracts between the external Router and P/D nodes."""

from __future__ import annotations

import hashlib
from typing import TypeVar

import msgspec

from .protocol import (
    CapabilityWire,
    ContinuationMetadata,
    DecodeOutputWire,
    HandoffKey,
    PrefixMatchSpec,
)


MAX_INTERNAL_BODY_BYTES = 4 << 20


class NodeDescriptor(msgspec.Struct, frozen=True):
    node_id: str
    role: str
    run_id: str
    control_host: str
    control_port: int
    owner_generation: int
    endpoint_generation: int
    control_incarnation: int
    capabilities: CapabilityWire
    health: str


class CapacitySnapshot(msgspec.Struct, frozen=True):
    node_id: str
    role: str
    active_handoffs: int
    prepared_requests: int
    reservations: int
    quarantined_reservations: int
    snapshot_sequence: int
    active_limit: int = 1
    queued_handoffs: int = 0
    inflight_transfer_bytes: int = 0
    inflight_transfer_byte_limit: int = 0


class PrepareRequestHTTP(msgspec.Struct, frozen=True):
    request_id: str
    request_kind: str
    request_json: bytes


class PreparedRequest(msgspec.Struct, frozen=True):
    request_id: str
    prepared_request_id: str
    prepared_digest: str
    continuation: ContinuationMetadata
    expires_at_ns: int
    prefix_match_spec: PrefixMatchSpec | None = None


class ReservePlacementHTTP(msgspec.Struct, frozen=True):
    key: HandoffKey
    prepared_request_id: str
    prepared_digest: str
    prompt_token_count: int
    max_new_tokens: int
    layout_fingerprint: str
    prefill_node_id: str
    prefill_endpoint_generation: int
    prefix_match_spec: PrefixMatchSpec | None = None


class PlacementReservation(msgspec.Struct, frozen=True):
    key: HandoffKey
    prepared_request_id: str
    reservation_id: str
    partition: int
    block_ids_by_group: dict[str, tuple[int, ...]]
    prepared_digest: str
    reservation_capability: str
    decode_node_id: str
    decode_control_host: str
    decode_control_port: int
    decode_endpoint_generation: int
    prefix_hit_tokens: int = 0


class PlacementRejection(msgspec.Struct, frozen=True):
    key: HandoffKey
    decode_node_id: str
    decode_endpoint_generation: int
    reason: str
    retryable: bool


class ReservePlacementResult(msgspec.Struct, frozen=True):
    reservation: PlacementReservation | None = None
    rejection: PlacementRejection | None = None

    def __post_init__(self) -> None:
        if (self.reservation is None) == (self.rejection is None):
            raise ValueError("reserve result must contain exactly one outcome")


class AuthorizeRouteHTTP(msgspec.Struct, frozen=True):
    key: HandoffKey
    prepared_request_id: str
    prepared_digest: str
    reservation_id: str
    reservation_capability: str
    compatibility_digest: str
    prefill_node_id: str
    prefill_endpoint_generation: int
    decode_node_id: str
    decode_endpoint_generation: int


class ExecutePrefillHTTP(msgspec.Struct, frozen=True):
    key: HandoffKey
    prepared_request_id: str
    prepared_digest: str
    reservation_id: str
    partition: int
    block_ids_by_group: dict[str, tuple[int, ...]]
    reservation_capability: str
    compatibility_digest: str
    prefill_node_id: str
    prefill_endpoint_generation: int
    decode_node_id: str
    decode_control_host: str
    decode_control_port: int
    decode_endpoint_generation: int
    prefix_hit_tokens: int = 0


class PrefillHandoffResult(msgspec.Struct, frozen=True):
    key: HandoffKey
    reservation_id: str
    manifest_hash: str
    state: str


class HandoffHTTP(msgspec.Struct, frozen=True):
    key: HandoffKey


class AbortHandoffHTTP(msgspec.Struct, frozen=True):
    key: HandoffKey
    reason: str
    deterministic: bool = True


class DecodeStreamFrame(msgspec.Struct, frozen=True):
    event: str
    output: DecodeOutputWire | None = None
    state: str = ""
    error_code: str = ""


T = TypeVar("T")


def encode_json(value: object) -> bytes:
    wire = msgspec.json.encode(value)
    if not wire or len(wire) > MAX_INTERNAL_BODY_BYTES:
        raise ValueError("PD internal HTTP body exceeds the bounded size")
    return wire


def decode_json(wire: bytes, type_: type[T]) -> T:
    if not wire or len(wire) > MAX_INTERNAL_BODY_BYTES:
        raise ValueError("invalid PD internal HTTP body size")
    return msgspec.json.decode(wire, type=type_)


def capability_compatibility_digest(value: CapabilityWire) -> str:
    """Digest only fields that must match across P and D owner registries."""
    contract = (
        value.schema_version,
        value.adapter_id,
        value.contract_version,
        value.contract_digest,
        value.continuation_schema,
        value.model_revision,
        value.layout_fingerprint,
        value.topology,
        value.provider,
        value.logical_groups,
        value.physical_regions,
        value.prefix_cache_mode,
    )
    return hashlib.sha256(msgspec.msgpack.encode(contract)).hexdigest()
