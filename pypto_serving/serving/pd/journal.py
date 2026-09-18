# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Durable, address-free recovery facts for the fixed 1P1D control plane."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import PDConfig
    from .protocol import HandoffKey


JOURNAL_SCHEMA_VERSION = 1
_GENESIS_HASH = "0" * 64
_TERMINAL_EVENTS = frozenset(
    {
        "HANDOFF_COMPLETED",
        "HANDOFF_ABORTED",
        "HANDOFF_FAILED",
        "RECOVERY_REQUIRED",
    }
)


def _canonical(value: dict[str, object]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


class DurablePDJournal:
    """Append-only JSONL with fsync and a corruption-detecting hash chain.

    Records intentionally exclude prompts, provider envelopes, native
    addresses and secrets.  A journal is scoped to one node and one control
    incarnation; reusing it with unresolved handoffs is rejected at startup.
    """

    def __init__(self, path: str, config: "PDConfig") -> None:
        if not path:
            raise ValueError("PD recovery journal path must not be empty")
        if len(os.fsencode(path)) > 4096:
            raise ValueError("PD recovery journal path is too long")
        self.path = Path(path)
        self.config = config
        self._lock = threading.Lock()
        self._sequence = 0
        self._last_hash = _GENESIS_HASH
        self._active: set[tuple[str, str]] = set()
        self._load()
        if self._active:
            raise RuntimeError(
                "PD journal contains unresolved handoffs; start a new control "
                "incarnation after recovery"
            )
        if self._sequence:
            raise RuntimeError(
                "PD journal already belongs to a completed control incarnation; "
                "use a new path and control incarnation"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(
            self.path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o600,
        )
        os.chmod(self.path, 0o600)

    @classmethod
    def audit(cls, path: str, config: "PDConfig") -> dict[str, object]:
        """Validate an existing chain without opening it for mutation."""
        journal = object.__new__(cls)
        journal.path = Path(path)
        journal.config = config
        journal._lock = threading.Lock()
        journal._sequence = 0
        journal._last_hash = _GENESIS_HASH
        journal._active = set()
        journal._load()
        return {
            "records": journal._sequence,
            "last_hash": journal._last_hash,
            "unresolved_handoffs": len(journal._active),
        }

    def close(self) -> None:
        with self._lock:
            fd = getattr(self, "_fd", None)
            if fd is None:
                return
            os.fsync(fd)
            os.close(fd)
            self._fd = None

    def append(
        self,
        event: str,
        *,
        key: "HandoffKey | None" = None,
        reservation_id: str = "",
        manifest_hash: str = "",
        chunk_id: int = -1,
        state: str = "",
        certainty: str = "",
        error_code: str = "",
    ) -> None:
        for name, value in (
            ("event", event),
            ("reservation_id", reservation_id),
            ("manifest_hash", manifest_hash),
            ("state", state),
            ("certainty", certainty),
            ("error_code", error_code),
        ):
            if not isinstance(value, str) or len(value.encode()) > 256:
                raise ValueError(f"PD journal {name} must be a string of at most 256 bytes")
        if not event:
            raise ValueError("PD journal event must not be empty")
        if type(chunk_id) is not int or chunk_id < -1:
            raise ValueError("PD journal chunk_id must be -1 or a non-negative integer")

        with self._lock:
            self._sequence += 1
            record: dict[str, object] = {
                "schema_version": JOURNAL_SCHEMA_VERSION,
                "sequence": self._sequence,
                "timestamp_ns": time.time_ns(),
                "run_id": self.config.run_id,
                "node_id": self.config.node_id,
                "role": self.config.role.value,
                "control_incarnation": self.config.control_incarnation,
                "event": event,
                "request_id": key.request_id if key is not None else "",
                "handoff_id": key.handoff_id if key is not None else "",
                "data_generation": key.data_generation if key is not None else 0,
                "route_epoch": key.route_epoch if key is not None else 0,
                "reservation_id": reservation_id,
                "manifest_hash": manifest_hash,
                "chunk_id": chunk_id,
                "state": state,
                "certainty": certainty,
                "error_code": error_code,
                "previous_hash": self._last_hash,
            }
            record_hash = hashlib.sha256(
                self._last_hash.encode() + _canonical(record)
            ).hexdigest()
            record["record_hash"] = record_hash
            wire = _canonical(record) + b"\n"
            written = os.write(self._fd, wire)
            if written != len(wire):
                raise OSError("short write while appending the PD recovery journal")
            os.fsync(self._fd)
            self._last_hash = record_hash
            self._update_active(record)

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open("rb") as stream:
                lines = stream.readlines()
        except OSError as exc:
            raise RuntimeError("cannot read the PD recovery journal") from exc
        previous_hash = _GENESIS_HASH
        for sequence, line in enumerate(lines, 1):
            if not line.endswith(b"\n"):
                raise RuntimeError("PD recovery journal ends with a partial record")
            try:
                record = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError("PD recovery journal contains invalid JSON") from exc
            if not isinstance(record, dict):
                raise RuntimeError("PD recovery journal record is not an object")
            record_hash = record.pop("record_hash", None)
            if (
                record.get("schema_version") != JOURNAL_SCHEMA_VERSION
                or record.get("sequence") != sequence
                or record.get("previous_hash") != previous_hash
                or record.get("run_id") != self.config.run_id
                or record.get("node_id") != self.config.node_id
                or record.get("role") != self.config.role.value
                or record.get("control_incarnation")
                != self.config.control_incarnation
            ):
                raise RuntimeError("PD recovery journal identity or sequence mismatch")
            expected = hashlib.sha256(
                previous_hash.encode() + _canonical(record)
            ).hexdigest()
            if record_hash != expected:
                raise RuntimeError("PD recovery journal hash chain mismatch")
            record["record_hash"] = record_hash
            previous_hash = record_hash
            self._sequence = sequence
            self._last_hash = record_hash
            self._update_active(record)

    def _update_active(self, record: dict[str, object]) -> None:
        request_id = record.get("request_id")
        handoff_id = record.get("handoff_id")
        if not isinstance(request_id, str) or not isinstance(handoff_id, str):
            raise RuntimeError("PD recovery journal handoff identity is invalid")
        if not request_id and not handoff_id:
            return
        if not request_id or not handoff_id:
            raise RuntimeError("PD recovery journal has a partial handoff identity")
        identity = (request_id, handoff_id)
        if record.get("event") == "HANDOFF_CREATED":
            self._active.add(identity)
        elif record.get("event") in _TERMINAL_EVENTS:
            self._active.discard(identity)
