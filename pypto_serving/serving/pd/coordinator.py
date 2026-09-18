# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Single-writer routing facts for the first fixed 1P1D implementation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import uuid

from .config import PDConfig, PDRole
from .protocol import HandoffKey


class CoordinatorState(str, Enum):
    CREATED = "CREATED"
    RESERVED = "RESERVED"
    READY = "READY"
    RELEASED = "RELEASED"
    FAILED = "FAILED"


@dataclass
class HandoffRecord:
    key: HandoffKey
    prefill_node_id: str
    decode_node_id: str
    state: CoordinatorState = CoordinatorState.CREATED
    reservation_id: str = ""
    manifest_hash: str = ""
    error_code: str = ""


class FixedCoordinator:
    """Logical coordinator for one configured P/D pair; no dynamic routing."""

    def __init__(self, config: PDConfig) -> None:
        if not config.enabled:
            raise ValueError("FixedCoordinator requires an enabled PD config")
        self.config = config
        self._records: dict[HandoffKey, HandoffRecord] = {}
        self._active_request: dict[str, HandoffKey] = {}

    def create_handoff(self, request_id: str, *, data_generation: int = 1) -> HandoffRecord:
        if request_id in self._active_request:
            raise ValueError(f"request {request_id!r} already has an active handoff")
        if type(data_generation) is not int or data_generation < 1:
            raise ValueError("data_generation must be a positive integer")
        key = HandoffKey(
            request_id=request_id,
            handoff_id=uuid.uuid4().hex,
            data_generation=data_generation,
            route_epoch=self.config.route_epoch,
            control_incarnation=self.config.control_incarnation,
        )
        if self.config.role is PDRole.PREFILL:
            prefill, decode = self.config.node_id, self.config.peer_node_id
        else:
            prefill, decode = self.config.peer_node_id, self.config.node_id
        record = HandoffRecord(key=key, prefill_node_id=prefill, decode_node_id=decode)
        self._records[key] = record
        self._active_request[request_id] = key
        return record

    def register_handoff(
        self,
        key: HandoffKey,
        *,
        prefill_node_id: str,
        decode_node_id: str,
    ) -> HandoffRecord:
        """Register the immutable identity selected by an external Router."""
        if (
            key.route_epoch != self.config.route_epoch
            or key.control_incarnation != self.config.control_incarnation
        ):
            raise ValueError("handoff route epoch or control incarnation is stale")
        existing = self._records.get(key)
        if existing is not None:
            if (
                existing.prefill_node_id != prefill_node_id
                or existing.decode_node_id != decode_node_id
            ):
                raise ValueError("handoff identity was replayed with different nodes")
            return existing
        active = self._active_request.get(key.request_id)
        if active is not None and active != key:
            raise ValueError(f"request {key.request_id!r} already has an active handoff")
        if prefill_node_id == decode_node_id:
            raise ValueError("handoff P and D nodes must differ")
        record = HandoffRecord(
            key=key,
            prefill_node_id=prefill_node_id,
            decode_node_id=decode_node_id,
        )
        self._records[key] = record
        self._active_request[key.request_id] = key
        return record

    def mark_reserved(self, key: HandoffKey, reservation_id: str) -> HandoffRecord:
        record = self._require(key)
        if record.state is CoordinatorState.RESERVED:
            if record.reservation_id != reservation_id:
                raise ValueError("reservation acknowledgement changed under one handoff")
            return record
        if record.state is not CoordinatorState.CREATED or not reservation_id:
            raise ValueError("handoff cannot enter RESERVED")
        record.state = CoordinatorState.RESERVED
        record.reservation_id = reservation_id
        return record

    def mark_ready(self, key: HandoffKey, manifest_hash: str) -> HandoffRecord:
        record = self._require(key)
        if record.state is CoordinatorState.READY:
            if record.manifest_hash != manifest_hash:
                raise ValueError("READY acknowledgement changed under one handoff")
            return record
        if record.state is not CoordinatorState.RESERVED or not manifest_hash:
            raise ValueError("handoff cannot enter READY")
        record.state = CoordinatorState.READY
        record.manifest_hash = manifest_hash
        return record

    def release(self, key: HandoffKey) -> HandoffRecord:
        record = self._require(key)
        if record.state is CoordinatorState.RELEASED:
            return record
        if record.state is not CoordinatorState.READY:
            raise ValueError("only a READY handoff can release retained P state")
        record.state = CoordinatorState.RELEASED
        self._active_request.pop(key.request_id, None)
        return record

    def fail(self, key: HandoffKey, error_code: str) -> HandoffRecord:
        record = self._require(key)
        if record.state is CoordinatorState.RELEASED:
            raise ValueError("a released handoff cannot fail retroactively")
        record.state = CoordinatorState.FAILED
        record.error_code = error_code
        self._active_request.pop(key.request_id, None)
        return record

    def query(self, key: HandoffKey) -> HandoffRecord | None:
        return self._records.get(key)

    def _require(self, key: HandoffKey) -> HandoffRecord:
        try:
            return self._records[key]
        except KeyError as exc:
            raise KeyError("unknown coordinator handoff") from exc
