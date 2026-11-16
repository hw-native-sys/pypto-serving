# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Bounded, opt-in DPLB diagnostics. All timestamps are process-local host clocks."""

from __future__ import annotations

import json
import logging
import math
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DplbDfxConfig:
    level: str = "off"
    sample_every: int = 32
    max_steps: int = 256
    max_seconds: float = 60
    queue_size: int = 1024
    max_event_bytes: int = 262144
    max_bytes: int = 67108864

    def __post_init__(self):
        if self.level not in {"off", "load", "timing"}:
            raise ValueError("DPLB DFX level must be off, load, or timing")
        if self.level == "timing":
            raise ValueError("DPLB DFX timing is unsupported: semantic device timing is not connected")
        if not math.isfinite(self.max_seconds):
            raise ValueError("DPLB DFX duration must be finite")
        if min(self.sample_every, self.max_steps, self.max_seconds, self.queue_size,
               self.max_event_bytes, self.max_bytes) <= 0:
            raise ValueError("DPLB DFX limits must be positive")


class _Sink:
    def __init__(self, path: Path, config: DplbDfxConfig):
        self.path = path
        self.config = config
        self.queue = queue.Queue(maxsize=config.queue_size)
        self.lock = threading.Lock()
        self.closed = False
        self.dropped = 0
        self.written = 0
        self.bytes_written = 0
        self.error = None
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._write, name="dplb-dfx-writer", daemon=True)
        self.thread.start()

    def put(self, event: dict) -> None:
        # Serialization bounds queued bytes as well as queued event count.
        try:
            data = (json.dumps(event, separators=(",", ":"), allow_nan=False) + "\n").encode()
        except (TypeError, ValueError):
            with self.lock:
                self.dropped += 1
            return
        with self.lock:
            if self.closed or self.error or len(data) > self.config.max_event_bytes:
                self.dropped += 1
                return
            try:
                self.queue.put_nowait(data)
            except queue.Full:
                self.dropped += 1

    def close(self, timeout: float = 2) -> bool:
        with self.lock:
            self.closed = True
        return self.done.wait(timeout)

    def _write(self):
        stream = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            stream = self.path.open("xb")
            while True:
                try:
                    data = self.queue.get(timeout=0.05)
                except queue.Empty:
                    with self.lock:
                        if self.closed:
                            break
                    continue
                if self.bytes_written + len(data) > self.config.max_bytes:
                    with self.lock:
                        self.dropped += 1
                    continue
                stream.write(data)
                stream.flush()
                self.bytes_written += len(data)
                self.written += 1
        except Exception as exc:
            self.error = str(exc)
            logger.warning("DPLB DFX sink failed: %s", exc)
        finally:
            if stream is not None:
                stream.close()
            try:
                manifest = dict(written=self.written, dropped=self.dropped, error=self.error,
                                bytes_written=self.bytes_written, closed=True)
                self.path.with_suffix(".manifest.json").write_text(json.dumps(manifest))
            except OSError:
                logger.warning("Cannot write DPLB DFX manifest: %s", self.path)
            self.done.set()


class DplbDfxRecorder:
    """One recorder per process/replica; explicit tokens retain capture ownership.

    Stop ends sampling, while already sampled steps may still append outcomes.
    A subsequent start or process shutdown seals the previous capture. Readers
    must treat a missing final manifest as an open or interrupted capture.
    """

    def __init__(self, config: DplbDfxConfig, output: Path, role: str, replica: int):
        self.config = config
        self.output = output
        self.role = role
        self.replica = replica
        self.instance = uuid.uuid4().hex
        self.sink = None
        self.capture_id = None
        self.active = False
        self.started = 0
        self.seen_steps = 0
        self.sampled_steps = 0
        self.epochs = {}
        self.decisions = {}
        self.decision_counter = 0
        self.step_tokens = {}

    @property
    def capturing(self) -> bool:
        return self.active and time.monotonic() - self.started < self.config.max_seconds

    def start(self, capture_id: str, **baseline):
        if self.active:
            return
        self.close()
        # IDs originate from UUIDs in the API process, never paths from clients.
        if not capture_id or any(c not in "0123456789abcdef-" for c in capture_id):
            raise ValueError("Invalid DPLB capture ID")
        self.capture_id = capture_id
        self.started = time.monotonic()
        self.seen_steps = self.sampled_steps = 0
        self.epochs = {}
        self.decisions = {}
        self.decision_counter = 0
        self.step_tokens = {}
        self.sink = _Sink(self.output / "dplb" / capture_id /
                          f"{self.role}-{self.replica}-{os.getpid()}-{self.instance}.jsonl", self.config)
        self.active = True
        self.emit("capture_start", device_timing="unavailable", **baseline)

    def emit(self, kind: str, *, token: dict | None = None, **fields):
        if self.sink is None:
            return
        if token is None:
            if not self.capturing:
                return
        elif token.get("capture_id") != self.capture_id:
            return
        self.sink.put(dict(schema_version=1, capture_id=self.capture_id, instance=self.instance,
                           role=self.role, replica=self.replica, clock_domain=f"host-{os.getpid()}",
                           monotonic_ns=time.monotonic_ns(), kind=kind, **fields))

    def begin_decision(self, request_id: str | None) -> int:
        self.decision_counter += 1
        self.decisions[request_id] = self.decision_counter
        return self.decision_counter

    def sample(self, step_id: int) -> dict | None:
        if not self.capturing:
            return None
        ordinal = self.seen_steps
        self.seen_steps += 1
        if ordinal % self.config.sample_every or self.sampled_steps >= self.config.max_steps:
            return None
        self.sampled_steps += 1
        token = dict(capture_id=self.capture_id, step_id=step_id)
        self.step_tokens[step_id] = token
        return token

    def outcome(self, step_id: int, **fields):
        token = self.step_tokens.pop(step_id, None)
        if token is not None:
            self.emit("step_outcome", token=token, step_id=step_id, **fields)

    def stop(self, **counters):
        if self.active:
            self.emit("capture_stop", token=dict(capture_id=self.capture_id),
                      sampled_steps=self.sampled_steps, seen_steps=self.seen_steps,
                      pending_steps=list(self.step_tokens), **counters)
            self.active = False

    def close(self):
        self.stop()
        if self.sink is not None:
            self.sink.close()
            self.sink = None


def command_load(command, group_count: int, epochs: dict | None = None) -> list[dict]:
    """Snapshot the command's host view, including empty logical DP groups."""
    groups = [dict(group=g, requests=[], decode_count=0, sum_seq_len=0, max_seq_len=0,
                   prefill_tokens=0) for g in range(group_count)]
    for stage, requests in (("prefill", command.prefill_requests), ("decode", command.decode_requests)):
        for request in requests:
            partition = request.cache_partition
            if partition is None or not 0 <= partition < group_count:
                continue
            group = groups[partition]
            length = (request.seq_len if stage == "decode" else
                      request.num_computed_tokens + len(request.chunk_tokens))
            group["requests"].append(dict(request_id=request.request_id, stage=stage, seq_len=length,
                                          admission_epoch=(epochs or {}).get(request.request_id)))
            if stage == "decode":
                group["decode_count"] += 1
                group["sum_seq_len"] += length
                group["max_seq_len"] = max(group["max_seq_len"], length)
            else:
                group["prefill_tokens"] += len(request.chunk_tokens)
    return groups
