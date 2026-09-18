# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable


@dataclass(frozen=True, slots=True)
class ExternalKVBuffer:
    """One registered memory range participating in a cache object transfer."""

    address: int
    size_bytes: int

    def __post_init__(self) -> None:
        if self.address <= 0:
            raise ValueError("external KV buffer address must be positive")
        if self.size_bytes <= 0:
            raise ValueError("external KV buffer size must be positive")


@dataclass(frozen=True, slots=True)
class ExternalKVTransfer:
    """Scatter-gather buffers containing one external cache object."""

    key: str
    buffers: tuple[ExternalKVBuffer, ...]

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("external KV transfer key must not be empty")
        if not self.buffers:
            raise ValueError("external KV transfer requires at least one buffer")

    @property
    def size_bytes(self) -> int:
        return sum(buffer.size_bytes for buffer in self.buffers)


@runtime_checkable
class ExternalKVBackend(Protocol):
    """Synchronous object backend used by an asynchronous worker connector."""

    def register_buffers(self, buffers: Sequence[ExternalKVBuffer]) -> None:
        """Register long-lived device buffers with the transfer engine."""

    def exists(self, keys: Sequence[str]) -> tuple[bool, ...]:
        """Return object presence in input order."""

    def put(self, transfers: Sequence[ExternalKVTransfer]) -> tuple[int, ...]:
        """Store objects and return one backend status code per object."""

    def put_bytes(self, key: str, payload: bytes) -> int:
        """Store one small host-resident metadata object."""

    def get(self, transfers: Sequence[ExternalKVTransfer]) -> tuple[int, ...]:
        """Load objects and return one backend status code per object."""
