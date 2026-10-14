# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Startup records and bounded PD metrics without prompt or address payloads."""

from __future__ import annotations

from pypto_serving.observability.tokens import token_ids_sha256

from collections import deque
import json
import os
from pathlib import Path
import threading
import time


def write_startup_record(log_dir: str, *, enabled: bool, values: dict[str, object]) -> None:
    if not enabled:
        return
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp_ns": time.time_ns(), **values}
    (directory / "startup.json").write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (directory / "pd.log").touch(exist_ok=True)
    (directory / "events.jsonl").touch(exist_ok=True)




class PDMetrics:
    """Small in-process registry used by Router and P/D internal endpoints."""

    def __init__(self, component: str, identity: str) -> None:
        self.component = component
        self.identity = identity
        self.started_ns = time.time_ns()
        self._lock = threading.RLock()
        self._counters: dict[str, int] = {}
        self._gauges: dict[str, int | float] = {}
        self._durations: dict[str, list[int]] = {}
        self._recent: deque[dict[str, object]] = deque(maxlen=128)

    def increment(self, name: str, value: int = 1) -> None:
        if type(value) is not int or value < 0:
            raise ValueError("PD counter increment must be a non-negative integer")
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + value

    def set_gauge(self, name: str, value: int | float) -> None:
        if not isinstance(value, (int, float)):
            raise TypeError("PD gauge must be numeric")
        with self._lock:
            self._gauges[name] = value

    def set_peak_gauge(self, name: str, value: int | float) -> None:
        """Keep a high-water mark without exposing the caller to locking."""
        if not isinstance(value, (int, float)):
            raise TypeError("PD gauge must be numeric")
        with self._lock:
            self._gauges[name] = max(self._gauges.get(name, value), value)

    def observe_ns(self, name: str, duration_ns: int) -> None:
        if type(duration_ns) is not int or duration_ns < 0:
            raise ValueError("PD duration must be a non-negative integer")
        with self._lock:
            aggregate = self._durations.setdefault(name, [0, 0, 0])
            aggregate[0] += 1
            aggregate[1] += duration_ns
            aggregate[2] = max(aggregate[2], duration_ns)

    def record_terminal(
        self,
        *,
        request_id: str,
        state: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        token_ids: tuple[int, ...] = (),
        error_code: str = "",
    ) -> None:
        digest = token_ids_sha256(token_ids)
        with self._lock:
            self._recent.append(
                {
                    "request_id": request_id,
                    "state": state,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "token_ids_sha256": digest,
                    "error_code": error_code,
                    "timestamp_ns": time.time_ns(),
                }
            )

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            durations = {
                name: {
                    "count": aggregate[0],
                    "sum_ns": aggregate[1],
                    "max_ns": aggregate[2],
                }
                for name, aggregate in sorted(self._durations.items())
            }
            return {
                "schema_version": 1,
                "component": self.component,
                "identity": self.identity,
                "started_ns": self.started_ns,
                "timestamp_ns": time.time_ns(),
                "counters": dict(sorted(self._counters.items())),
                "gauges": dict(sorted(self._gauges.items())),
                "durations": durations,
                "recent_terminal": list(self._recent),
                "process": _process_snapshot(),
            }


def _process_snapshot() -> dict[str, int]:
    rss_bytes = 0
    try:
        pages = int(Path("/proc/self/statm").read_text().split()[1])
        rss_bytes = pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        pass
    fd_count = 0
    try:
        fd_count = len(tuple(Path("/proc/self/fd").iterdir()))
    except OSError:
        pass
    return {
        "pid": os.getpid(),
        "rss_bytes": rss_bytes,
        "threads": threading.active_count(),
        "fds": fd_count,
    }
