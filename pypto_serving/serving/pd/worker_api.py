# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Typed payloads carried by the generic engine↔worker PD command."""

from __future__ import annotations

from typing import TypeVar

import msgspec

from pypto_serving.model.deepseek.transfer_layout import ComponentLayout, DSV4Registry

from .protocol import ChunkManifest, RankRegistration, TransferResult


OP_PREPARE_REGISTRY = "prepare_registry"
OP_INSTALL_PEER = "install_peer"
OP_TRANSFER_CHUNK = "transfer_chunk"
OP_SUBMIT_TRANSFER_CHUNK = "submit_transfer_chunk"
OP_POLL_TRANSFER_CHUNK = "poll_transfer_chunk"
OP_INSPECT_REGISTRY = "inspect_registry"
OP_INSPECT_RUNTIME_METRICS = "inspect_runtime_metrics"

# Must cover the same expanded final-chunk manifest accepted by the Host
# control channel.  This remains independently bounded because the
# multiprocessing queue crosses a trust/process boundary as well.
MAX_WORKER_PD_PAYLOAD_BYTES = 16 << 20


class ComponentGeometry(msgspec.Struct, frozen=True):
    component_id: str
    dtype: str
    item_bytes: int
    layers: tuple[int, ...]
    blocks_per_layer: int
    block_tokens: int
    token_stride_bytes: int
    extent: int


class WorkerRegistryBundle(msgspec.Struct, frozen=True):
    model_revision: str
    topology: tuple[int, ...]
    registry_fingerprint: str
    layout_fingerprint: str
    components: tuple[ComponentGeometry, ...]
    ranks: tuple[RankRegistration, ...]

    def registry(self) -> DSV4Registry:
        return DSV4Registry(
            model_revision=self.model_revision,
            topology=self.topology,
            components=tuple(
                ComponentLayout(
                    component_id=component.component_id,
                    dtype=component.dtype,
                    item_bytes=component.item_bytes,
                    layers=component.layers,
                    blocks_per_layer=component.blocks_per_layer,
                    block_tokens=component.block_tokens,
                    token_stride_bytes=component.token_stride_bytes,
                )
                for component in self.components
            ),
        )


class InstallPeerRequest(msgspec.Struct, frozen=True):
    registry_fingerprint: str
    layout_fingerprint: str
    ranks: tuple[RankRegistration, ...]


class TransferChunkRequest(msgspec.Struct, frozen=True):
    manifest: ChunkManifest
    destination_fingerprint: str
    timeout_seconds: float = 60.0
    attempt_ordinal: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.timeout_seconds, (int, float)) or self.timeout_seconds <= 0:
            raise ValueError("transfer timeout must be positive")
        if type(self.attempt_ordinal) is not int or not 1 <= self.attempt_ordinal <= 8:
            raise ValueError("transfer attempt ordinal must be in [1, 8]")


class TransferChunkResponse(msgspec.Struct, frozen=True):
    results: tuple[TransferResult, ...]


class TransferSubmission(msgspec.Struct, frozen=True):
    job_id: str


class TransferPollRequest(msgspec.Struct, frozen=True):
    job_id: str


class TransferPollResponse(msgspec.Struct, frozen=True):
    job_id: str
    complete: bool
    results: tuple[TransferResult, ...] = ()


class WorkerRuntimeMetrics(msgspec.Struct, frozen=True):
    values: dict[str, float]


_encoder = msgspec.msgpack.Encoder()
T = TypeVar("T")


def encode_worker_payload(value: object) -> bytes:
    return _encoder.encode(value)


def decode_worker_payload(wire: bytes, value_type: type[T]) -> T:
    if not isinstance(wire, bytes) or len(wire) > MAX_WORKER_PD_PAYLOAD_BYTES:
        raise ValueError("invalid worker PD payload size")
    return msgspec.msgpack.decode(wire, type=value_type)
