# Copyright (c) PyPTO Contributors.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""Minimal hash-chained Router intent journal without prompts or addresses."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading
import time

from pypto_serving.serving.pd.protocol import HandoffKey


class RouterJournal:
    def __init__(self, path: str, run_id: str) -> None:
        self.path = Path(path)
        self.run_id = run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        os.chmod(self.path, 0o600)
        self._lock = threading.Lock()
        self._sequence, self._last_hash, self._active = self._audit()

    @property
    def unresolved(self) -> tuple[HandoffKey, ...]:
        return tuple(self._active.values())

    def append(self, event: str, key: HandoffKey, *, error_code: str = "") -> None:
        with self._lock:
            self._sequence += 1
            record = {
                "schema_version": 1,
                "sequence": self._sequence,
                "timestamp_ns": time.time_ns(),
                "run_id": self.run_id,
                "event": event,
                "request_id": key.request_id,
                "handoff_id": key.handoff_id,
                "data_generation": key.data_generation,
                "route_epoch": key.route_epoch,
                "control_incarnation": key.control_incarnation,
                "error_code": error_code,
                "previous_hash": self._last_hash,
            }
            canonical = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
            record_hash = hashlib.sha256(self._last_hash.encode() + canonical).hexdigest()
            record["record_hash"] = record_hash
            wire = json.dumps(record, sort_keys=True, separators=(",", ":")).encode() + b"\n"
            if os.write(self._fd, wire) != len(wire):
                raise OSError("short write to Router journal")
            os.fsync(self._fd)
            self._last_hash = record_hash
            identity = (key.request_id, key.handoff_id)
            if event == "HANDOFF_CREATED":
                self._active[identity] = key
            elif event in ("HANDOFF_COMPLETED", "HANDOFF_FAILED", "RECOVERY_REQUIRED"):
                self._active.pop(identity, None)

    def close(self) -> None:
        with self._lock:
            if self._fd is not None:
                os.fsync(self._fd)
                os.close(self._fd)
                self._fd = None

    def _audit(self) -> tuple[int, str, dict[tuple[str, str], HandoffKey]]:
        last_hash = "0" * 64
        sequence = 0
        active: dict[tuple[str, str], HandoffKey] = {}
        with self.path.open("rb") as stream:
            for sequence, line in enumerate(stream, 1):
                record = json.loads(line)
                record_hash = record.pop("record_hash", "")
                if (
                    record.get("sequence") != sequence
                    or record.get("run_id") != self.run_id
                    or record.get("previous_hash") != last_hash
                ):
                    raise RuntimeError("Router journal identity or sequence mismatch")
                canonical = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
                expected = hashlib.sha256(last_hash.encode() + canonical).hexdigest()
                if record_hash != expected:
                    raise RuntimeError("Router journal hash chain mismatch")
                last_hash = record_hash
                key = HandoffKey(
                    request_id=record["request_id"],
                    handoff_id=record["handoff_id"],
                    data_generation=record["data_generation"],
                    route_epoch=record["route_epoch"],
                    control_incarnation=record["control_incarnation"],
                )
                identity = (key.request_id, key.handoff_id)
                if record["event"] == "HANDOFF_CREATED":
                    active[identity] = key
                elif record["event"] in (
                    "HANDOFF_COMPLETED",
                    "HANDOFF_FAILED",
                    "RECOVERY_REQUIRED",
                ):
                    active.pop(identity, None)
        return sequence, last_hash, active
