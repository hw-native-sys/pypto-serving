# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded causal status journal with immutable terminal observations."""

from __future__ import annotations

import threading
import time
import hashlib
import json
from dataclasses import asdict, dataclass

from .errors import ErrorCode, TransferError, TransferFailure
from .types import CompletionCertainty, OwnerRef, Stage, TransferAttemptRef, integer

_NEXT = {
    Stage.CREATED: Stage.VALIDATED,
    Stage.VALIDATED: Stage.QUEUED,
    Stage.QUEUED: Stage.WAITING_FENCE,
    Stage.WAITING_FENCE: Stage.READY,
    Stage.READY: Stage.SUBMITTED,
    Stage.SUBMITTED: Stage.COMPLETED,
}


@dataclass(frozen=True)
class TransferEvent:
    event_id: str
    parent_event_id: str | None
    owner_sequence: int
    monotonic_ns: int
    attempt: TransferAttemptRef
    stage: Stage
    certainty: CompletionCertainty
    error: TransferError | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class StatusReply:
    kind: str
    event: TransferEvent | None = None


class StatusIndex:
    """Capacity is reserved before admission; only explicitly released terminals can be evicted."""

    def __init__(self, owner: OwnerRef, capacity: int = 1024):
        integer(capacity, "status capacity", 1)
        self.owner = owner
        self._event_prefix = hashlib.sha256(json.dumps(asdict(owner), sort_keys=True).encode()).hexdigest()
        self.capacity = capacity
        self._events: dict[str, tuple[TransferEvent, ...]] = {}
        self._released: set[str] = set()
        self._sequence = 0
        self._last_attempt_sequence = -1
        self._lock = threading.RLock()

    def create(self, attempt: TransferAttemptRef) -> TransferEvent:
        with self._lock:
            if attempt.source != self.owner:
                raise TransferFailure(TransferError(ErrorCode.STALE_GENERATION, CompletionCertainty.NOT_SUBMITTED))
            if attempt.attempt_id in self._events:
                raise ValueError("attempt identity cannot be replayed")
            if attempt.attempt_sequence <= self._last_attempt_sequence:
                raise ValueError("attempt sequence was already admitted or retired")
            if len(self._events) == self.capacity:
                if not self._released:
                    raise TransferFailure(TransferError(
                        ErrorCode.BACKPRESSURE, CompletionCertainty.NOT_SUBMITTED, True))
                victim = next(key for key in self._events if key in self._released)
                del self._events[victim]
                self._released.remove(victim)
            self._events[attempt.attempt_id] = ()
            self._last_attempt_sequence = attempt.attempt_sequence
            return self._append(attempt, Stage.CREATED, CompletionCertainty.NOT_SUBMITTED, None)

    def _append(self, attempt, stage, certainty, error):
        history = self._events[attempt.attempt_id]
        self._sequence += 1
        event = TransferEvent(
            f"{self._event_prefix}:{self._sequence}",
            history[-1].event_id if history else None,
            self._sequence, time.monotonic_ns(), attempt, stage, certainty, error,
        )
        self._events[attempt.attempt_id] = (*history, event)
        return event

    def advance(self, attempt: TransferAttemptRef, stage: Stage,
                error: TransferError | None = None) -> TransferEvent:
        with self._lock:
            old = self._events[attempt.attempt_id][-1]
            if old.attempt != attempt:
                raise ValueError("attempt identity mismatch")
            if old.stage.terminal:
                raise ValueError("terminal observation is immutable")
            if stage == Stage.FAILED:
                if error is None:
                    raise ValueError("failure requires an error")
                certainty = error.certainty
                if old.stage == Stage.SUBMITTED:
                    if certainty == CompletionCertainty.NOT_SUBMITTED:
                        raise ValueError("submitted attempt cannot become not-submitted")
                elif certainty != CompletionCertainty.NOT_SUBMITTED:
                    raise ValueError("pre-submit failure must be not-submitted")
            else:
                if error is not None or _NEXT.get(old.stage) != stage:
                    raise ValueError("invalid transition")
                certainty = (CompletionCertainty.COMPLETED if stage == Stage.COMPLETED
                             else CompletionCertainty.UNKNOWN if stage == Stage.SUBMITTED
                             else CompletionCertainty.NOT_SUBMITTED)
            return self._append(attempt, stage, certainty, error)

    def query(self, attempt: TransferAttemptRef) -> StatusReply:
        with self._lock:
            history = self._events.get(attempt.attempt_id)
            if history and history[-1].attempt == attempt:
                event = history[-1]
                return StatusReply("terminal" if event.stage.terminal else "in_flight", event)
            if attempt.source != self.owner:
                return StatusReply("stale_owner")
            return StatusReply("unknown_not_observed")

    def release(self, attempt: TransferAttemptRef) -> None:
        with self._lock:
            if self.query(attempt).kind != "terminal":
                raise ValueError("only observed terminal attempts may pass the release watermark")
            self._released.add(attempt.attempt_id)

    def history(self, attempt: TransferAttemptRef) -> tuple[TransferEvent, ...]:
        with self._lock:
            if self.query(attempt).event is None:
                return ()
            return self._events[attempt.attempt_id]
