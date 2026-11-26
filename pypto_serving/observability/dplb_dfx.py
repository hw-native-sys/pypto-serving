# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded Host placement diagnostics, adapted from the initial #336 collector.

The allocator serializes immutable observations into a bounded queue; only the
writer touches disk. This phase records policy load, not device execution time.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import queue
import re
import subprocess
import threading
import time
import uuid

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _source_info() -> dict:
    root = Path(__file__).resolve().parents[2]
    if not (root / ".git").exists():
        return dict(status="source_revision_unavailable")
    try:

        def git(*args):
            return subprocess.run(
                ["git", "-C", str(root), *args],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=2,
            ).stdout

        revision = git("rev-parse", "HEAD").decode().strip()
        patch = git("diff", "HEAD", "--")
        status = git("status", "--porcelain", "--untracked-files=normal")
        return dict(
            revision=revision,
            dirty=bool(status),
            tracked_diff_sha256=hashlib.sha256(patch).hexdigest(),
            untracked_files=sum(line.startswith(b"??") for line in status.splitlines()),
        )
    except (OSError, subprocess.SubprocessError):
        return dict(status="source_revision_unavailable")


@dataclass(frozen=True)
class DplbDfxConfig:
    level: str = "off"
    max_seconds: float = 60.0
    queue_size: int = 1024
    queue_bytes: int = 8 * 1024 * 1024
    max_event_bytes: int = 256 * 1024
    max_bytes: int = 64 * 1024 * 1024  # Per replica; the coordinator divides the capture budget.
    max_requests: int = 65536

    def __post_init__(self) -> None:
        if self.level not in {"off", "load", "timing"}:
            raise ValueError("DPLB DFX level must be off, load, or timing")
        if self.level == "timing":
            raise ValueError("DPLB DFX timing is unavailable: semantic device timing is not connected")
        if isinstance(self.max_seconds, bool) or not math.isfinite(self.max_seconds) or self.max_seconds <= 0:
            raise ValueError("DPLB DFX max_seconds must be finite and positive")
        for name in ("queue_size", "queue_bytes", "max_event_bytes", "max_bytes", "max_requests"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"DPLB DFX {name} must be a positive integer")


class _Sink:
    """Byte- and count-bounded output with a final status outside the data queue."""

    def __init__(self, path: Path, config: DplbDfxConfig, metadata: dict):
        self.path = path
        self.config = config
        self.metadata = metadata
        self.queue: queue.Queue = queue.Queue(maxsize=config.queue_size)
        self.lock = threading.Lock()
        self.closed = False
        self.done = threading.Event()
        self.error: str | None = None
        self.stop_reason: str | None = None
        self.attempted: Counter = Counter()
        self.written: Counter = Counter()
        self.dropped: Counter = Counter()
        self.bytes_written = 0
        self.queued_bytes = 0
        self.peak_queued_bytes = 0
        self.final_fields: dict = {}
        self.thread = threading.Thread(target=self._write, name="dplb-dfx-writer", daemon=True)
        self.thread.start()

    def put(self, event: dict) -> bool:
        kind = event["event"]
        try:
            data = (json.dumps(event, separators=(",", ":"), allow_nan=False) + "\n").encode()
        except (TypeError, ValueError, OverflowError):
            with self.lock:
                self.attempted[kind] += 1
                self.dropped["serialization"] += 1
            return False
        with self.lock:
            self.attempted[kind] += 1
            if self.closed or self.error:
                self.dropped["closed_or_io_error"] += 1
            elif len(data) > self.config.max_event_bytes:
                self.dropped["event_bytes"] += 1
            elif self.bytes_written + self.queued_bytes + len(data) > self.config.max_bytes:
                self.dropped["output_bytes"] += 1
                self.stop_reason = "output_bytes"
            elif self.queued_bytes + len(data) > self.config.queue_bytes:
                self.dropped["queue_bytes"] += 1
            else:
                try:
                    self.queue.put_nowait((kind, data))
                except queue.Full:
                    self.dropped["queue_full"] += 1
                else:
                    self.queued_bytes += len(data)
                    self.peak_queued_bytes = max(self.peak_queued_bytes, self.queued_bytes)
                    return True
        return False

    def seal(self, reason: str, **fields) -> None:
        with self.lock:
            if not self.closed:
                if self.stop_reason is None or reason == "profile_control_failed":
                    self.stop_reason = reason
                self.final_fields = fields
                self.closed = True

    def _write(self) -> None:
        created = False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("xb") as stream:
                created = True
                while True:
                    try:
                        kind, data = self.queue.get(timeout=0.05)
                    except queue.Empty:
                        with self.lock:
                            if self.closed and self.queued_bytes == 0:
                                break
                        continue
                    stream.write(data)
                    with self.lock:
                        self.queued_bytes -= len(data)
                        self.bytes_written += len(data)
                        self.written[kind] += 1
        except Exception as exc:
            with self.lock:
                self.error = f"{type(exc).__name__}: {exc}"
                self.closed = True
                self.stop_reason = "io_error"
            logger.warning("DPLB DFX output failed: %s", exc)
        finally:
            try:
                if created:
                    self._write_manifest()
            finally:
                self.done.set()

    def _write_manifest(self) -> None:
        with self.lock:
            status = dict(
                self.metadata | self.final_fields,
                stop_reason=self.stop_reason,
                attempted=dict(self.attempted),
                written=dict(self.written),
                dropped=dict(self.dropped),
                bytes_written=self.bytes_written,
                queued_bytes=self.queued_bytes,
                peak_queued_bytes=self.peak_queued_bytes,
                error=self.error,
                complete=not self.error
                and not self.dropped
                and self.queued_bytes == 0
                and self.attempted == self.written
                and self.stop_reason != "profile_control_failed"
                and not self.final_fields.get("untracked_requests", 0),
            )
        try:
            target = self.path.with_suffix(".manifest.json")
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps(status, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            temporary.replace(target)
        except (OSError, TypeError, ValueError, OverflowError) as exc:
            logger.warning("DPLB DFX manifest could not be written: %s", exc)


class DplbDfxRecorder:
    """One capture per replica, with bounded identities for live requests.

    Admission epochs survive nonterminal owner releases (preemption/fallback).
    Terminal events retire identities, so reusing a client request ID is safe.
    No worker command mirrors or device durations are claimed by this collector.
    """

    def __init__(self, config: DplbDfxConfig, output: Path, replica: int = 0):
        self.config = config
        self.output = Path(output)
        self.replica = replica
        self.instance = uuid.uuid4().hex
        self.source = _source_info()
        try:
            self.clock_instance = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        except OSError:
            self.clock_instance = self.instance
        self.capture_id: str | None = None
        self.sink: _Sink | None = None
        self._active = False
        self._profile_start_pending = False
        self._pending_end: tuple[str, dict] | None = None
        self._lock = threading.RLock()
        self._timer: threading.Timer | None = None
        self._requests: dict[str, dict] = {}
        self._request_counter = 0
        self._decision_counter = 0
        self._sequence = 0
        self._started_ns = 0
        self._untracked_requests = 0

    @property
    def capturing(self) -> bool:
        with self._lock:
            if self._active and self.sink is not None:
                if self.sink.error or self.sink.stop_reason:
                    self._seal(self.sink.stop_reason or "io_error")
                elif time.monotonic_ns() - self._started_ns >= self.config.max_seconds * 1e9:
                    self._seal("duration")
            return self._active

    @property
    def ready_for_capture(self) -> bool:
        with self._lock:
            return (not self.capturing and self.config.level != "off"
                    and self._pending_end is None and (self.sink is None or self.sink.done.is_set()))

    def start(self, capture_id: str | None = None, **metadata) -> bool:
        with self._lock:
            if not self.ready_for_capture:
                return False
            capture_id = capture_id or uuid.uuid4().hex
            if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", capture_id):
                raise ValueError("Invalid DPLB DFX capture ID")
            self.capture_id = capture_id
            self._requests.clear()
            self._sequence = self._decision_counter = self._untracked_requests = 0
            self._started_ns = time.monotonic_ns()
            manifest = dict(
                schema_version=1,
                capture_id=capture_id,
                replica=self.replica,
                replica_instance_id=self.instance,
                device_timing="unavailable",
                observation_level="placement",
                serving_source=self.source,
                output_byte_limit=self.config.max_bytes,
                clock_instance_id=self.clock_instance,
                **metadata,
            )
            path = self.output / "dplb" / capture_id / f"engine-{self.replica}-{self.instance}.jsonl"
            self.sink = _Sink(path, self.config, manifest)
            self._active = True
            self._timer = threading.Timer(
                self.config.max_seconds,
                self.stop,
                kwargs={"reason": "duration", "expected_capture_id": capture_id},
            )
            self._timer.daemon = True
            self.emit("capture_start", **metadata)
            if self.capturing and self._timer is not None:
                self._timer.start()
            return True

    def emit(self, event: str, **fields) -> bool:
        with self._lock:
            if not self.capturing:
                return False
            return self._emit(event, **fields)

    def _emit(self, event: str, **fields) -> bool:
        self._sequence += 1
        return self.sink.put(
            dict(
                schema_version=1,
                event=event,
                event_seq=self._sequence,
                capture_id=self.capture_id,
                replica=self.replica,
                replica_instance_id=self.instance,
                pid=os.getpid(),
                clock_domain="host_monotonic_ns",
                clock_instance_id=self.clock_instance,
                monotonic_ns=time.monotonic_ns(),
                **fields,
            )
        )

    def request_fields(self, request_id: str, *, acquire: bool = False, preexisting: bool = False) -> dict:
        with self._lock:
            state = self._requests.get(request_id)
            if state is None:
                if len(self._requests) >= self.config.max_requests:
                    self._untracked_requests += 1
                    return dict(
                        request_id=request_id,
                        request_uid=None,
                        admission_epoch=None,
                        identity_status="request_limit",
                    )
                self._request_counter += 1
                state = dict(
                    request_id=request_id,
                    request_uid=f"{self.instance}:{self._request_counter}",
                    admission_epoch=0,
                    identity_status="preexisting" if preexisting else "captured",
                )
                self._requests[request_id] = state
            if acquire:
                state["admission_epoch"] += 1
            return dict(state)

    def begin_decision(self, request_id: str) -> dict:
        with self._lock:
            self._decision_counter += 1
            return dict(**self.request_fields(request_id), decision_id=self._decision_counter)

    def terminal(self, request_id: str, reason: str) -> None:
        with self._lock:
            if self.capturing and request_id in self._requests:
                self.emit("request_terminal", **self.request_fields(request_id), reason=reason)
            self._requests.pop(request_id, None)

    def _end_collection(self, **fields) -> dict:
        self._active = False
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        final_fields = dict(
            ended_ns=time.monotonic_ns(),
            untracked_requests=self._untracked_requests,
            open_request_count=len(self._requests),
            **fields,
        )
        self._requests.clear()
        return final_fields

    def _seal(self, reason: str, **fields) -> None:
        final_fields = self._end_collection(**fields)
        if self._profile_start_pending:
            self._pending_end = (reason, final_fields)
        else:
            self.sink.seal(reason, **final_fields)

    def stop(self, *, reason: str = "requested", expected_capture_id: str | None = None, **fields) -> None:
        with self._lock:
            if expected_capture_id is not None and expected_capture_id != self.capture_id:
                return
            if self.capturing:
                self._emit("capture_end", reason=reason, **fields)
                self._seal(reason, **fields)

    def begin_profile_start(self) -> None:
        """Defer the manifest if Host collection ends before the worker replies."""
        with self._lock:
            self._profile_start_pending = True

    def finish_profile_start(self, *, failed: bool = False) -> None:
        with self._lock:
            if failed:
                self.stop(reason="profile_control_failed")
            else:
                self.emit("profile_control_ack", active=True)
            self._profile_start_pending = False
            self._finish_pending_end(failed)

    def begin_profile_stop(self, **fields) -> None:
        """End Host collection now and defer the manifest until the worker reply."""
        with self._lock:
            if self.capturing:
                self._emit("capture_end", reason="requested", **fields)
                self._pending_end = ("requested", self._end_collection(**fields))

    def finish_profile_stop(self, *, failed: bool = False) -> None:
        with self._lock:
            self._finish_pending_end(failed)

    def _finish_pending_end(self, failed: bool) -> None:
        if self._pending_end is not None:
            reason, fields = self._pending_end
            self.sink.seal("profile_control_failed" if failed else reason, **fields)
            self._pending_end = None

    def close(self, *, reason: str = "shutdown", timeout: float = 2.0) -> bool:
        with self._lock:
            self.stop(reason=reason)
            sink = self.sink
        return sink is None or sink.done.wait(timeout)
