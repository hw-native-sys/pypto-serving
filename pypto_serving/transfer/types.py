# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Immutable transfer identities and address-free segment contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

SCHEMA_VERSION = 1
MAX_EXTENT = (1 << 63) - 1


def integer(value: int, name: str, minimum: int = 0) -> None:
    if type(value) is not int or not minimum <= value <= MAX_EXTENT:
        raise ValueError(f"{name} must be an integer in [{minimum}, {MAX_EXTENT}]")


def identifier(value: str) -> None:
    if not isinstance(value, str) or not value or len(value) > 256:
        raise ValueError("identity must be a nonempty string of at most 256 characters")


class CompletionCertainty(str, Enum):
    NOT_SUBMITTED = "NOT_SUBMITTED"
    COMPLETED = "COMPLETED"
    FAILED_DEFINITE = "FAILED_DEFINITE"
    UNKNOWN = "UNKNOWN"


class Stage(str, Enum):
    CREATED = "CREATED"
    VALIDATED = "VALIDATED"
    QUEUED = "QUEUED"
    WAITING_FENCE = "WAITING_FENCE"
    READY = "READY"
    SUBMITTED = "SUBMITTED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

    @property
    def terminal(self) -> bool:
        return self in (Stage.COMPLETED, Stage.FAILED)


@dataclass(frozen=True)
class OwnerRef:
    run_id: str
    rank_id: int
    generation: int
    endpoint_generation: int
    worker_id: str

    def __post_init__(self):
        identifier(self.run_id)
        identifier(self.worker_id)
        for name in ("rank_id", "generation", "endpoint_generation"):
            integer(getattr(self, name), name)


@dataclass(frozen=True)
class RegionLease:
    owner: OwnerRef
    region_id: str
    lease: int
    extent: int

    def __post_init__(self):
        if not isinstance(self.owner, OwnerRef):
            raise TypeError("owner must be OwnerRef")
        identifier(self.region_id)
        integer(self.lease, "lease")
        integer(self.extent, "extent", 1)


@dataclass(frozen=True)
class TransferAttemptRef:
    request_id: str
    plan_id: str
    handoff_id: str
    attempt_id: str
    data_generation: int
    route_epoch: int
    chunk_id: int
    source: OwnerRef
    destination: OwnerRef
    component_manifest_hash: str
    attempt_sequence: int = 0

    def __post_init__(self):
        for name in ("request_id", "plan_id", "handoff_id", "attempt_id", "component_manifest_hash"):
            identifier(getattr(self, name))
        for name in ("data_generation", "route_epoch", "chunk_id", "attempt_sequence"):
            integer(getattr(self, name), name)
        if not isinstance(self.source, OwnerRef) or not isinstance(self.destination, OwnerRef):
            raise TypeError("attempt endpoints must be OwnerRef")
        if self.source.run_id != self.destination.run_id or self.source.rank_id != self.destination.rank_id:
            raise ValueError("only same-run, same-rank transfers are supported")


@dataclass(frozen=True)
class ProviderCapabilities:
    registered_device_memory: bool = True
    batch_scatter_gather_write: bool = True
    async_completion: bool = False
    supports_cancel: bool = False
    supports_drain_fence: bool = False
    supports_query_status: bool = False
    min_offset_alignment: int = 64
    min_length_alignment: int = 64
    max_segments_per_batch: int | None = None

    def __post_init__(self):
        integer(self.min_offset_alignment, "offset alignment", 1)
        integer(self.min_length_alignment, "length alignment", 1)
        if self.max_segments_per_batch is not None:
            integer(self.max_segments_per_batch, "segment limit", 1)


@dataclass(frozen=True)
class Segment:
    component_id: str
    source: RegionLease
    destination: RegionLease
    source_offset: int
    destination_offset: int
    length: int

    def __post_init__(self):
        identifier(self.component_id)
        for name in ("source_offset", "destination_offset"):
            integer(getattr(self, name), name)
        integer(self.length, "length", 1)
        for region, offset in ((self.source, self.source_offset), (self.destination, self.destination_offset)):
            if not isinstance(region, RegionLease) or offset + self.length > region.extent:
                raise ValueError("segment exceeds region extent")


@dataclass(frozen=True)
class ProviderTransferTask:
    attempt: TransferAttemptRef
    segments: tuple[Segment, ...]

    def __post_init__(self):
        if not isinstance(self.attempt, TransferAttemptRef):
            raise TypeError("attempt must be TransferAttemptRef")
        if not isinstance(self.segments, tuple) or not self.segments:
            raise ValueError("segments must be a nonempty immutable tuple")
        for segment in self.segments:
            if segment.source.owner != self.attempt.source or segment.destination.owner != self.attempt.destination:
                raise ValueError("segment owner does not match attempt")
        integer(self.nbytes, "total bytes", 1)

    @property
    def nbytes(self) -> int:
        return sum(segment.length for segment in self.segments)

    def validate(self, capabilities: ProviderCapabilities) -> None:
        if capabilities.max_segments_per_batch is not None:
            if len(self.segments) > capabilities.max_segments_per_batch:
                raise ValueError("batch exceeds provider segment limit")
        for segment in self.segments:
            if (segment.source_offset % capabilities.min_offset_alignment
                    or segment.destination_offset % capabilities.min_offset_alignment
                    or segment.length % capabilities.min_length_alignment):
                raise ValueError("segment alignment does not match provider")
