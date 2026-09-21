# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""D-side manifest aggregation and at-most-once Decode admission."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from pypto_serving.transfer.types import CompletionCertainty

from .protocol import (
    ChunkManifest,
    CommitRequest,
    HandoffKey,
    HandoffStatus,
    ReadyAck,
    TransferResult,
    chunk_payload_hash,
    continuation_metadata_hash,
)


class CompletionState(str, Enum):
    RESERVED = "RESERVED"
    TRANSFERRING = "TRANSFERRING"
    READY = "READY"
    IN_USE = "IN_USE"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"
    QUARANTINED = "QUARANTINED"


@dataclass
class _UnitState:
    attempts: dict[str, CompletionCertainty] = field(default_factory=dict)
    completed_attempt: str = ""


@dataclass
class _ChunkState:
    manifest: ChunkManifest
    units: dict[tuple[int, str], _UnitState]

    @property
    def complete(self) -> bool:
        return all(unit.completed_attempt for unit in self.units.values())


class CompletionTracker:
    """Stable D authority for one reservation/handoff identity."""

    def __init__(self, key: HandoffKey, reservation_id: str) -> None:
        if not reservation_id:
            raise ValueError("reservation_id must not be empty")
        self.key = key
        self.reservation_id = reservation_id
        self.state = CompletionState.RESERVED
        self._chunks: dict[int, _ChunkState] = {}
        self._final_chunk_id: int | None = None
        self._ready_ack: ReadyAck | None = None
        self._error_code = ""
        self._admitted = False

    @property
    def source_prefix_hit_tokens(self) -> int:
        if not self._chunks:
            return 0
        return self._chunks[min(self._chunks)].manifest.source_prefix_hit_tokens

    def register_chunk(self, manifest: ChunkManifest) -> None:
        if manifest.key != self.key:
            raise ValueError("chunk manifest handoff identity mismatch")
        if type(manifest.chunk_id) is not int or manifest.chunk_id < 0:
            raise ValueError("chunk_id must be a non-negative integer")
        if not 0 <= manifest.start_token < manifest.end_token:
            raise ValueError("chunk token interval must be nonempty and increasing")
        if not manifest.manifest_hash:
            raise ValueError("chunk manifest hash must not be empty")
        if manifest.manifest_hash != chunk_payload_hash(
            manifest.key,
            chunk_id=manifest.chunk_id,
            start_token=manifest.start_token,
            end_token=manifest.end_token,
            final=manifest.final,
            expected_units=manifest.expected_units,
            copies_by_rank=manifest.copies_by_rank,
            source_prefix_hit_tokens=manifest.source_prefix_hit_tokens,
        ):
            raise ValueError("chunk manifest hash does not match its physical write set")
        existing = self._chunks.get(manifest.chunk_id)
        if existing is not None:
            if existing.manifest != manifest:
                raise ValueError("chunk_id was replayed with a different manifest")
            return
        if self.state in (
            CompletionState.READY,
            CompletionState.IN_USE,
            CompletionState.COMPLETED,
            CompletionState.ABORTED,
            CompletionState.QUARANTINED,
        ):
            raise ValueError(f"cannot register a chunk in state {self.state.value}")
        if self._final_chunk_id is not None:
            raise ValueError("no chunk may follow the final manifest")
        if self._chunks and manifest.chunk_id != max(self._chunks) + 1:
            raise ValueError("chunk manifests must be registered in contiguous order")
        if not self._chunks and manifest.chunk_id != 0:
            raise ValueError("the first chunk_id must be zero")
        expected_start = (
            0
            if not self._chunks
            else self._chunks[max(self._chunks)].manifest.end_token
        )
        if manifest.start_token != expected_start:
            raise ValueError("chunk token intervals must be contiguous")
        if self._chunks:
            first_manifest = self._chunks[min(self._chunks)].manifest
            if (
                manifest.source_prefix_hit_tokens
                != first_manifest.source_prefix_hit_tokens
            ):
                raise ValueError("P prefix hit must remain stable across chunks")

        units: dict[tuple[int, str], _UnitState] = {}
        for unit in manifest.expected_units:
            if type(unit.rank_id) is not int or unit.rank_id < 0:
                raise ValueError("completion rank_id must be a non-negative integer")
            if not unit.component_id or len(unit.component_id.encode()) > 256:
                raise ValueError("completion component_id must be a bounded identifier")
            if type(unit.nbytes) is not int or unit.nbytes < 0:
                raise ValueError("completion unit nbytes must be non-negative")
            identity = (unit.rank_id, unit.component_id)
            if identity in units:
                raise ValueError("chunk manifest contains a duplicate completion unit")
            units[identity] = _UnitState()
        if not units:
            raise ValueError("chunk manifest must enumerate its expected completion units")
        ranks = set(manifest.copies_by_rank)
        if not ranks.issubset({rank for rank, _ in units}):
            raise ValueError("chunk copies contain a rank absent from expected completion units")

        self._chunks[manifest.chunk_id] = _ChunkState(manifest, units)
        if manifest.final:
            if (
                manifest.first_token is None
                or not manifest.metadata_hash
                or manifest.continuation is None
            ):
                raise ValueError("final chunk requires first token and continuation metadata")
            if continuation_metadata_hash(manifest.continuation) != manifest.metadata_hash:
                raise ValueError("final chunk continuation metadata hash mismatch")
            self._final_chunk_id = manifest.chunk_id
        elif (
            manifest.first_token is not None
            or manifest.metadata_hash
            or manifest.continuation is not None
        ):
            raise ValueError("non-final chunks cannot carry continuation metadata")
        self.state = CompletionState.TRANSFERRING

    def record_transfer(self, result: TransferResult) -> None:
        if result.key != self.key:
            raise ValueError("transfer result handoff identity mismatch")
        if self.state is CompletionState.QUARANTINED:
            # Late native success cannot overwrite an UNKNOWN terminal fact.
            return
        if self.state in (
            CompletionState.READY,
            CompletionState.IN_USE,
            CompletionState.COMPLETED,
            CompletionState.ABORTED,
        ):
            raise ValueError(f"cannot record transfer in state {self.state.value}")
        try:
            chunk = self._chunks[result.chunk_id]
        except KeyError as exc:
            raise ValueError("transfer result references an unknown chunk") from exc
        try:
            unit = chunk.units[(result.rank_id, result.component_id)]
        except KeyError as exc:
            raise ValueError("transfer result is absent from the expected completion set") from exc
        if not result.attempt_id:
            raise ValueError("transfer attempt_id must not be empty")
        certainty = CompletionCertainty(result.certainty)
        previous = unit.attempts.get(result.attempt_id)
        if previous is not None:
            if previous is not certainty:
                raise ValueError("one transfer attempt returned conflicting terminal facts")
            return
        unit.attempts[result.attempt_id] = certainty
        if certainty is CompletionCertainty.COMPLETED:
            if not unit.completed_attempt:
                unit.completed_attempt = result.attempt_id
            return
        if certainty is CompletionCertainty.UNKNOWN:
            self.state = CompletionState.QUARANTINED
            self._error_code = result.error_code or "UNKNOWN_TRANSFER"
            return
        # NOT_SUBMITTED and FAILED_DEFINITE remain safely retryable under a new
        # attempt. They are visible through status but do not poison the owner.
        self._error_code = result.error_code

    def commit(self, request: CommitRequest) -> ReadyAck:
        if request.key != self.key:
            raise ValueError("commit handoff identity mismatch")
        if self._ready_ack is not None:
            if (
                self._ready_ack.manifest_hash != request.manifest_hash
                or self._final_manifest().first_token != request.first_token
                or self._final_manifest().metadata_hash != request.metadata_hash
            ):
                raise ValueError("committed handoff was replayed with different metadata")
            return self._ready_ack
        if self.state is CompletionState.QUARANTINED:
            raise RuntimeError("uncertain transfer cannot be committed")
        final = self._final_manifest()
        if request.manifest_hash != final.manifest_hash:
            raise ValueError("commit manifest hash mismatch")
        if request.first_token != final.first_token or request.metadata_hash != final.metadata_hash:
            raise ValueError("commit continuation metadata mismatch")
        if not self._chunks or not all(chunk.complete for chunk in self._chunks.values()):
            raise RuntimeError("handoff completion set is incomplete")
        self.state = CompletionState.READY
        self._error_code = ""
        self._ready_ack = ReadyAck(
            key=self.key,
            reservation_id=self.reservation_id,
            manifest_hash=request.manifest_hash,
            admitted=False,
        )
        return self._ready_ack

    def admit_decode(self) -> bool:
        """Return true exactly once for the scheduler admission side effect."""
        if self.state is CompletionState.IN_USE:
            return False
        if self.state is not CompletionState.READY:
            raise RuntimeError(f"handoff is not ready for Decode admission: {self.state.value}")
        self.state = CompletionState.IN_USE
        self._admitted = True
        assert self._ready_ack is not None
        self._ready_ack = ReadyAck(
            key=self._ready_ack.key,
            reservation_id=self._ready_ack.reservation_id,
            manifest_hash=self._ready_ack.manifest_hash,
            admitted=True,
        )
        return True

    def mark_completed(self) -> None:
        if self.state is CompletionState.COMPLETED:
            return
        if self.state is not CompletionState.IN_USE:
            raise RuntimeError("only an admitted Decode request can complete")
        self.state = CompletionState.COMPLETED

    def mark_cancelled(self, *, error_code: str) -> None:
        """Publish a deterministic terminal fact after scheduler-side abort."""
        if self.state is CompletionState.ABORTED:
            return
        if self.state not in (CompletionState.READY, CompletionState.IN_USE):
            raise RuntimeError("only a committed Decode request can be cancelled")
        self.state = CompletionState.ABORTED
        self._error_code = error_code

    def abort(self, *, deterministic: bool, error_code: str) -> None:
        if self.state in (CompletionState.COMPLETED, CompletionState.ABORTED):
            return
        if self.state is CompletionState.QUARANTINED:
            # Once any native attempt is UNKNOWN, a later request-level
            # cancellation cannot prove that writer stopped and must never
            # downgrade quarantine to a reusable reservation.
            return
        if self.state in (CompletionState.READY, CompletionState.IN_USE):
            raise RuntimeError("a committed handoff cannot return to the abort path")
        if deterministic:
            self.state = CompletionState.ABORTED
        else:
            self.state = CompletionState.QUARANTINED
        self._error_code = error_code

    def query(self) -> HandoffStatus:
        manifest_hash = self._ready_ack.manifest_hash if self._ready_ack is not None else ""
        return HandoffStatus(
            key=self.key,
            state=self.state.value,
            reservation_id=self.reservation_id,
            manifest_hash=manifest_hash,
            admitted=self._admitted,
            error_code=self._error_code,
        )

    def replay_ready_ack(self) -> ReadyAck:
        if self._ready_ack is None:
            raise RuntimeError("handoff has no committed READY fact")
        return self._ready_ack

    def final_manifest(self) -> ChunkManifest:
        """Return the immutable committed continuation contract."""
        if self.state not in (
            CompletionState.READY,
            CompletionState.IN_USE,
            CompletionState.COMPLETED,
        ):
            raise RuntimeError("handoff continuation is not committed")
        return self._final_manifest()

    def _final_manifest(self) -> ChunkManifest:
        if self._final_chunk_id is None:
            raise RuntimeError("handoff has no final chunk manifest")
        return self._chunks[self._final_chunk_id].manifest
