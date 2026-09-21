# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded, address-free Host protocol for Router-assigned PD handoffs."""

from __future__ import annotations

import hashlib
import os
import socket
import struct
from typing import Union

import msgspec

from pypto_serving.serving.reasoning import OutputParserSpec

from .config import PDCapabilities, PDRole


# A final handoff can carry the prompt continuation plus an expanded
# page/layer write set for final-only model state. Keep a hard bound sized for
# the supported model envelope; at four active
# handoffs this still caps decoded control payload memory at 64 MiB per peer.
MAX_CONTROL_MESSAGE_BYTES = 16 << 20
MAX_ID_BYTES = 256
PREFIX_MATCH_SCHEMA_VERSION = 1
MAX_PREFIX_MATCH_SPEC_BYTES = 2 << 20


def _identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not value or len(value.encode()) > MAX_ID_BYTES:
        raise ValueError(f"{name} must be a nonempty identifier of at most {MAX_ID_BYTES} bytes")


class HandoffKey(msgspec.Struct, frozen=True):
    request_id: str
    handoff_id: str
    data_generation: int
    route_epoch: int
    control_incarnation: int

    def __post_init__(self) -> None:
        _identifier(self.request_id, "request_id")
        _identifier(self.handoff_id, "handoff_id")
        for name in ("data_generation", "route_epoch", "control_incarnation"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")


class CapabilityWire(msgspec.Struct, frozen=True):
    schema_version: int
    adapter_id: str
    contract_version: int
    contract_digest: str
    continuation_schema: str
    model_revision: str
    registry_fingerprint: str
    layout_fingerprint: str
    topology: tuple[int, ...]
    provider: str
    logical_groups: tuple[str, ...]
    physical_regions: tuple[str, ...]
    prefix_cache_mode: str = "disabled"

    @classmethod
    def from_capabilities(cls, value: PDCapabilities) -> "CapabilityWire":
        return cls(
            value.schema_version,
            value.adapter_id,
            value.contract_version,
            value.contract_digest,
            value.continuation_schema,
            value.model_revision,
            value.registry_fingerprint,
            value.layout_fingerprint,
            value.topology,
            value.provider,
            value.logical_groups,
            value.physical_regions,
            value.prefix_cache_mode,
        )

    def to_capabilities(self) -> PDCapabilities:
        return PDCapabilities(
            schema_version=self.schema_version,
            adapter_id=self.adapter_id,
            contract_version=self.contract_version,
            contract_digest=self.contract_digest,
            continuation_schema=self.continuation_schema,
            model_revision=self.model_revision,
            registry_fingerprint=self.registry_fingerprint,
            layout_fingerprint=self.layout_fingerprint,
            topology=self.topology,
            provider=self.provider,
            logical_groups=self.logical_groups,
            physical_regions=self.physical_regions,
            prefix_cache_mode=self.prefix_cache_mode,
        )


class Hello(msgspec.Struct, tag="hello", frozen=True):
    node_id: str
    role: str
    run_id: str
    nonce: bytes
    capabilities: CapabilityWire
    endpoint_generation: int
    control_incarnation: int


class RegionRegistration(msgspec.Struct, frozen=True):
    component_id: str
    lease: int
    extent: int
    # Canonical owner envelope encoded by the D worker. Host code transports it
    # as opaque bytes and must never parse addresses from it.
    provider_envelope: bytes


class RankRegistration(msgspec.Struct, frozen=True):
    rank_id: int
    owner_generation: int
    endpoint_generation: int
    worker_id: str
    regions: tuple[RegionRegistration, ...]


class RegistryAdvertisement(msgspec.Struct, tag="registry", frozen=True):
    """Address-bearing provider envelopes transported as opaque worker output."""

    model_revision: str
    topology: tuple[int, ...]
    registry_fingerprint: str
    layout_fingerprint: str
    ranks: tuple[RankRegistration, ...]


class PrefixGroupMatch(msgspec.Struct, frozen=True):
    group_name: str
    block_hashes: tuple[bytes, ...]


class PrefixMatchSpec(msgspec.Struct, frozen=True):
    """Bounded, versioned and address-free cache lookup material."""

    schema_version: int
    token_count: int
    alignment: int
    contract_digest: str
    groups: tuple[PrefixGroupMatch, ...]
    identity_digest: str


class ReserveRequest(msgspec.Struct, tag="reserve", frozen=True):
    key: HandoffKey
    prompt_token_count: int
    max_new_tokens: int
    layout_fingerprint: str
    requested_partition: int | None = None
    prepared_digest: str = ""
    prefix_match_spec: PrefixMatchSpec | None = None


class ReserveAccepted(msgspec.Struct, tag="reserve_ok", frozen=True):
    key: HandoffKey
    reservation_id: str
    partition: int
    block_ids_by_group: dict[str, tuple[int, ...]]
    ranks: tuple[RankRegistration, ...]
    prefix_hit_tokens: int = 0


class ReserveRejected(msgspec.Struct, tag="reserve_rejected", frozen=True):
    key: HandoffKey
    reason: str
    retryable: bool


class PageCopy(msgspec.Struct, frozen=True):
    component_id: str
    layer: int
    source_block: int
    destination_block: int
    valid_tokens: int


class TransferUnit(msgspec.Struct, frozen=True):
    rank_id: int
    component_id: str
    nbytes: int


class ContinuationMetadata(msgspec.Struct, frozen=True):
    prompt_token_ids: tuple[int, ...]
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int | None
    seed: int | None
    stop_strings: tuple[str, ...]
    eos_token_id: int | None
    stream: bool = True
    output_parser_spec: OutputParserSpec | None = None


class ChunkManifest(msgspec.Struct, tag="chunk_manifest", frozen=True):
    key: HandoffKey
    chunk_id: int
    start_token: int
    end_token: int
    final: bool
    manifest_hash: str
    expected_units: tuple[TransferUnit, ...]
    copies_by_rank: dict[int, tuple[PageCopy, ...]]
    source_prefix_hit_tokens: int = 0
    first_token: int | None = None
    metadata_hash: str = ""
    continuation: ContinuationMetadata | None = None
    prepared_digest: str = ""


class TransferResult(msgspec.Struct, tag="transfer_result", frozen=True):
    key: HandoffKey
    chunk_id: int
    rank_id: int
    component_id: str
    attempt_id: str
    certainty: str
    error_code: str = ""


class CommitRequest(msgspec.Struct, tag="commit", frozen=True):
    key: HandoffKey
    manifest_hash: str
    first_token: int
    metadata_hash: str


class ReadyAck(msgspec.Struct, tag="ready_ack", frozen=True):
    key: HandoffKey
    reservation_id: str
    manifest_hash: str
    admitted: bool = False


class ControlAck(msgspec.Struct, tag="control_ack", frozen=True):
    key: HandoffKey
    operation: str
    chunk_id: int = -1


class ControlError(msgspec.Struct, tag="control_error", frozen=True):
    key: HandoffKey
    operation: str
    error_code: str
    deterministic: bool


class DecodeOutputWire(msgspec.Struct, tag="decode_output", frozen=True):
    key: HandoffKey
    token_id: int | None
    text: str
    finished: bool
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    token_ids: tuple[int, ...] = ()
    output_sequence: int = 0
    reasoning: str = ""
    text_delta: str = ""
    reasoning_delta: str = ""


class OpenRoute(msgspec.Struct, tag="open_route", frozen=True):
    key: HandoffKey
    prepared_request_id: str
    reservation_id: str
    prepared_digest: str
    reservation_capability: str
    compatibility_digest: str
    prefill_node_id: str
    prefill_endpoint_generation: int


class RouteOpened(msgspec.Struct, tag="route_opened", frozen=True):
    key: HandoffKey
    reservation_id: str


class QueryHandoff(msgspec.Struct, tag="query", frozen=True):
    key: HandoffKey


class HandoffStatus(msgspec.Struct, tag="status", frozen=True):
    key: HandoffKey
    state: str
    reservation_id: str = ""
    manifest_hash: str = ""
    admitted: bool = False
    error_code: str = ""


class AbortHandoff(msgspec.Struct, tag="abort", frozen=True):
    key: HandoffKey
    reason: str
    deterministic: bool = True


class ReleaseHandoff(msgspec.Struct, tag="release", frozen=True):
    key: HandoffKey
    reason: str


class PoisonHandoff(msgspec.Struct, tag="poison", frozen=True):
    key: HandoffKey
    rank_id: int
    attempt_id: str
    error_code: str


ControlMessage = Union[
    Hello,
    RegistryAdvertisement,
    ReserveRequest,
    ReserveAccepted,
    ReserveRejected,
    ChunkManifest,
    TransferResult,
    CommitRequest,
    ReadyAck,
    ControlAck,
    ControlError,
    DecodeOutputWire,
    OpenRoute,
    RouteOpened,
    QueryHandoff,
    HandoffStatus,
    AbortHandoff,
    ReleaseHandoff,
    PoisonHandoff,
]


def message_handoff_key(message: ControlMessage) -> HandoffKey | None:
    """Return the correlation identity carried by a request-scoped frame."""
    key = getattr(message, "key", None)
    return key if isinstance(key, HandoffKey) else None


class _Frame(msgspec.Struct):
    sender: str
    sequence: int
    payload: bytes


_message_encoder = msgspec.msgpack.Encoder()
_message_decoder = msgspec.msgpack.Decoder(ControlMessage)
_frame_encoder = msgspec.msgpack.Encoder()
_frame_decoder = msgspec.msgpack.Decoder(_Frame)


def encode_message(message: ControlMessage) -> bytes:
    wire = _message_encoder.encode(message)
    if not wire or len(wire) > MAX_CONTROL_MESSAGE_BYTES:
        raise ValueError(
            "PD control message exceeds the bounded wire size "
            f"({len(wire)} > {MAX_CONTROL_MESSAGE_BYTES})"
        )
    return wire


def decode_message(wire: bytes) -> ControlMessage:
    if not wire or len(wire) > MAX_CONTROL_MESSAGE_BYTES:
        raise ValueError("invalid PD control message size")
    return _message_decoder.decode(wire)


def continuation_metadata_hash(metadata: ContinuationMetadata) -> str:
    """Return a canonical digest bound into the final handoff commit."""
    if not isinstance(metadata, ContinuationMetadata):
        raise TypeError("metadata must be ContinuationMetadata")
    return hashlib.sha256(_message_encoder.encode(metadata)).hexdigest()


def make_prefix_match_spec(
    *,
    token_count: int,
    alignment: int,
    contract_digest: str,
    group_block_hashes: dict[str, list[bytes]],
) -> PrefixMatchSpec:
    if type(token_count) is not int or token_count <= 0:
        raise ValueError("prefix match token_count must be positive")
    if type(alignment) is not int or alignment <= 0:
        raise ValueError("prefix match alignment must be positive")
    if len(contract_digest) != 64:
        raise ValueError("prefix match contract digest must be SHA-256")
    groups = tuple(
        PrefixGroupMatch(name, tuple(group_block_hashes[name]))
        for name in sorted(group_block_hashes)
    )
    for group in groups:
        _identifier(group.group_name, "prefix group")
        if any(len(block_hash) != 32 for block_hash in group.block_hashes):
            raise ValueError("prefix block hashes must be SHA-256 digests")
    identity_payload = (
        PREFIX_MATCH_SCHEMA_VERSION,
        token_count,
        alignment,
        contract_digest,
        groups,
    )
    identity_digest = hashlib.sha256(
        msgspec.msgpack.encode(identity_payload)
    ).hexdigest()
    spec = PrefixMatchSpec(
        PREFIX_MATCH_SCHEMA_VERSION,
        token_count,
        alignment,
        contract_digest,
        groups,
        identity_digest,
    )
    if len(msgspec.msgpack.encode(spec)) > MAX_PREFIX_MATCH_SPEC_BYTES:
        raise ValueError("prefix match specification exceeds its bounded wire size")
    return spec


def validate_prefix_match_spec(spec: PrefixMatchSpec) -> None:
    rebuilt = make_prefix_match_spec(
        token_count=spec.token_count,
        alignment=spec.alignment,
        contract_digest=spec.contract_digest,
        group_block_hashes={
            group.group_name: list(group.block_hashes) for group in spec.groups
        },
    )
    if rebuilt != spec:
        raise ValueError("prefix match specification is non-canonical or tampered")


def prepared_request_hash(
    continuation: ContinuationMetadata,
    prefix_match_spec: PrefixMatchSpec | None,
) -> str:
    """Bind execution metadata to the exact cache match material."""
    payload = (
        continuation_metadata_hash(continuation),
        "" if prefix_match_spec is None else prefix_match_spec.identity_digest,
    )
    return hashlib.sha256(msgspec.msgpack.encode(payload)).hexdigest()


def chunk_payload_hash(
    key: HandoffKey,
    *,
    chunk_id: int,
    start_token: int,
    end_token: int,
    final: bool,
    expected_units: tuple[TransferUnit, ...],
    copies_by_rank: dict[int, tuple[PageCopy, ...]],
    source_prefix_hit_tokens: int = 0,
) -> str:
    """Digest the canonical physical write set, independent of wire ordering."""
    units = sorted(
        expected_units,
        key=lambda unit: (unit.rank_id, unit.component_id, unit.nbytes),
    )
    copies = [
        (
            rank_id,
            sorted(
                rank_copies,
                key=lambda copy: (
                    copy.component_id,
                    copy.layer,
                    copy.destination_block,
                    copy.source_block,
                    copy.valid_tokens,
                ),
            ),
        )
        for rank_id, rank_copies in sorted(copies_by_rank.items())
    ]
    payload = (
        key,
        chunk_id,
        start_token,
        end_token,
        final,
        source_prefix_hit_tokens,
        tuple(units),
        tuple((rank_id, tuple(rank_copies)) for rank_id, rank_copies in copies),
    )
    return hashlib.sha256(_message_encoder.encode(payload)).hexdigest()


class FramedChannel:
    """Bounded frame transport with peer identity and replay sequencing."""

    def __init__(
        self,
        channel: socket.socket,
        *,
        local_node_id: str,
        peer_node_id: str,
    ) -> None:
        for name, value in (("local_node_id", local_node_id), ("peer_node_id", peer_node_id)):
            _identifier(value, name)
        if local_node_id == peer_node_id:
            raise ValueError("PD channel peers must have distinct node ids")
        self._channel = channel
        self._local_node_id = local_node_id
        self._peer_node_id = peer_node_id
        self._send_sequence = 0
        self._recv_sequence = 0

    def send(self, message: ControlMessage) -> None:
        payload = encode_message(message)
        self._send_sequence += 1
        frame = _Frame(
            sender=self._local_node_id,
            sequence=self._send_sequence,
            payload=payload,
        )
        wire = _frame_encoder.encode(frame)
        if len(wire) > MAX_CONTROL_MESSAGE_BYTES:
            raise ValueError("PD frame exceeds the bounded wire size")
        self._channel.sendall(struct.pack("!I", len(wire)) + wire)

    def receive(self) -> ControlMessage:
        length = struct.unpack("!I", self._read_exact(4))[0]
        if not 0 < length <= MAX_CONTROL_MESSAGE_BYTES:
            raise ValueError("invalid PD frame size")
        frame = _frame_decoder.decode(self._read_exact(length))
        expected_sequence = self._recv_sequence + 1
        if frame.sender != self._peer_node_id or frame.sequence != expected_sequence:
            raise ValueError("stale, replayed, or misrouted PD control frame")
        message = decode_message(frame.payload)
        self._recv_sequence = frame.sequence
        return message

    def _read_exact(self, length: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < length:
            part = self._channel.recv(length - len(chunks))
            if not part:
                raise EOFError("PD control channel closed")
            chunks.extend(part)
        return bytes(chunks)


def make_hello(
    *,
    node_id: str,
    role: PDRole,
    run_id: str,
    capabilities: PDCapabilities,
    endpoint_generation: int = 1,
    control_incarnation: int = 1,
) -> Hello:
    return Hello(
        node_id=node_id,
        role=role.value,
        run_id=run_id,
        nonce=os.urandom(16),
        capabilities=CapabilityWire.from_capabilities(capabilities),
        endpoint_generation=endpoint_generation,
        control_incarnation=control_incarnation,
    )


def exchange_and_validate_hello(
    channel: FramedChannel,
    hello: Hello,
    *,
    expected_peer_node_id: str,
    expected_peer_role: PDRole,
) -> Hello:
    """Symmetric send-then-receive handshake with strict peer/capability checks."""
    channel.send(hello)
    peer = channel.receive()
    if not isinstance(peer, Hello):
        raise ValueError("the first PD peer message must be hello")
    if peer.node_id != expected_peer_node_id or peer.role != expected_peer_role.value:
        raise ValueError("PD peer identity or role mismatch")
    if peer.run_id != hello.run_id:
        raise ValueError("PD peer run_id mismatch")
    if peer.nonce == hello.nonce or len(peer.nonce) != 16:
        raise ValueError("invalid PD peer handshake nonce")
    if peer.endpoint_generation < 1 or peer.control_incarnation < 1:
        raise ValueError("invalid PD peer generation")
    local_capabilities = hello.capabilities.to_capabilities()
    error = local_capabilities.compatibility_error(peer.capabilities.to_capabilities())
    if error is not None:
        raise ValueError(error)
    return peer
