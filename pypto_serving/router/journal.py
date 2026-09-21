# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Minimal hash-chained Router intent journal without prompts or addresses."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import threading
import time

from pypto_serving.serving.pd.protocol import HandoffKey


@dataclass(frozen=True)
class JournalRouteBinding:
    key: HandoffKey
    prefill_node_id: str = ""
    decode_node_id: str = ""
    prefill_endpoint_generation: int = 0
    decode_endpoint_generation: int = 0


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
        return tuple(binding.key for binding in self._active.values())

    @property
    def unresolved_bindings(self) -> tuple[JournalRouteBinding, ...]:
        return tuple(self._active.values())

    def append(
        self,
        event: str,
        key: HandoffKey,
        *,
        error_code: str = "",
        prefill_node_id: str = "",
        decode_node_id: str = "",
        prefill_endpoint_generation: int = 0,
        decode_endpoint_generation: int = 0,
    ) -> None:
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
                "prefill_node_id": prefill_node_id,
                "decode_node_id": decode_node_id,
                "prefill_endpoint_generation": prefill_endpoint_generation,
                "decode_endpoint_generation": decode_endpoint_generation,
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
                self._active[identity] = JournalRouteBinding(
                    key,
                    prefill_node_id,
                    decode_node_id,
                    prefill_endpoint_generation,
                    decode_endpoint_generation,
                )
            elif event in ("HANDOFF_COMPLETED", "HANDOFF_FAILED", "RECOVERY_REQUIRED"):
                self._active.pop(identity, None)
            elif identity in self._active:
                current = self._active[identity]
                self._active[identity] = JournalRouteBinding(
                    key,
                    prefill_node_id or current.prefill_node_id,
                    decode_node_id or current.decode_node_id,
                    prefill_endpoint_generation
                    or current.prefill_endpoint_generation,
                    decode_endpoint_generation or current.decode_endpoint_generation,
                )

    def close(self) -> None:
        with self._lock:
            if self._fd is not None:
                os.fsync(self._fd)
                os.close(self._fd)
                self._fd = None

    def _audit(
        self,
    ) -> tuple[int, str, dict[tuple[str, str], JournalRouteBinding]]:
        last_hash = "0" * 64
        sequence = 0
        active: dict[tuple[str, str], JournalRouteBinding] = {}
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
                    active[identity] = JournalRouteBinding(
                        key,
                        record.get("prefill_node_id", ""),
                        record.get("decode_node_id", ""),
                        record.get("prefill_endpoint_generation", 0),
                        record.get("decode_endpoint_generation", 0),
                    )
                elif record["event"] in (
                    "HANDOFF_COMPLETED",
                    "HANDOFF_FAILED",
                    "RECOVERY_REQUIRED",
                ):
                    active.pop(identity, None)
                elif identity in active:
                    current = active[identity]
                    active[identity] = JournalRouteBinding(
                        key,
                        record.get("prefill_node_id", "")
                        or current.prefill_node_id,
                        record.get("decode_node_id", "") or current.decode_node_id,
                        record.get("prefill_endpoint_generation", 0)
                        or current.prefill_endpoint_generation,
                        record.get("decode_endpoint_generation", 0)
                        or current.decode_endpoint_generation,
                    )
        return sequence, last_hash, active
