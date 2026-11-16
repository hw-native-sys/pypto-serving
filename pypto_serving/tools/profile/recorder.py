# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import atexit
import asyncio
import json
import multiprocessing
import os
import sys
import threading
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator

from .env import ProfileConfig, load_profile_config
from .merge import merge_fragments

_MAIN_PID_ENV = "SA_PROFILE_MAIN_PID"
_profiler: "ProfileRecorder | None" = None
_profiler_pid: int | None = None


class ProfileRecorder:
    """Chrome trace-event recorder backed by one JSONL fragment per process."""

    def __init__(
        self,
        config: ProfileConfig,
        *,
        process_name: str | None = None,
        initially_active: bool = True,
    ) -> None:
        self.config = config
        self.enabled = config.enabled
        self.active = False
        self.pid = os.getpid()
        self.process_name = process_name or _default_process_name()
        self._lock = threading.Lock()
        self._thread_names: set[int] = set()
        self._fragment_file: Path | None = None
        self._fh = None
        if self.enabled:
            self.config.fragments_dir.mkdir(parents=True, exist_ok=True)
            self._fragment_file = self.config.fragments_dir / f"trace.{self.pid}.jsonl"
            atexit.register(self.stop)
            if initially_active:
                self.start()

    def start(self) -> bool:
        """Start recording in this process.

        Returns ``True`` when a new recording session was started and ``False``
        when profiling is unavailable or was already active.
        """
        if not self.enabled or self._fragment_file is None:
            return False
        with self._lock:
            if self._fh is not None:
                return False
            self._thread_names.clear()
            self._fh = self._fragment_file.open("a", encoding="utf-8")
            self.active = True
        self.metadata("process_name", self.process_name, tid=0)
        return True

    def stop(self) -> bool:
        """Flush and stop recording in this process."""
        with self._lock:
            if self._fh is None:
                self.active = False
                return False
            self._fh.flush()
            self._fh.close()
            self._fh = None
            self.active = False
        return True

    close = stop

    def includes(self, level: str) -> bool:
        return self.active and self.config.includes(level)

    @contextmanager
    def span(
        self,
        name: str,
        *,
        cat: str = "e2e",
        level: str = "e2e",
        args: dict[str, Any] | None = None,
    ) -> Iterator[None]:
        if not self.includes(level):
            yield
            return
        tid, tid_name = self._get_tid_and_name()
        self._ensure_thread_metadata(tid, tid_name)
        start_us = _now_us()
        try:
            yield
        finally:
            dur_us = max(0, _now_us() - start_us)
            self._write(
                {
                    "name": name,
                    "cat": cat,
                    "ph": "X",
                    "ts": start_us,
                    "dur": dur_us,
                    "pid": self.pid,
                    "tid": tid,
                    "args": args or {},
                }
            )

    def instant(
        self,
        name: str,
        *,
        cat: str = "e2e",
        level: str = "e2e",
        args: dict[str, Any] | None = None,
    ) -> None:
        if not self.includes(level):
            return
        tid, tid_name = self._get_tid_and_name()
        self._ensure_thread_metadata(tid, tid_name)
        self._write(
            {
                "name": name,
                "cat": cat,
                "ph": "i",
                "s": "t",
                "ts": _now_us(),
                "pid": self.pid,
                "tid": tid,
                "args": args or {},
            }
        )

    def duration(
        self,
        name: str,
        *,
        dur_us: float,
        cat: str = "e2e",
        level: str = "e2e",
        args: dict[str, Any] | None = None,
        ts_us: float | None = None,
    ) -> None:
        if not self.includes(level):
            return
        tid, tid_name = self._get_tid_and_name()
        self._ensure_thread_metadata(tid, tid_name)
        dur_us_i = max(0, int(dur_us))
        ts_us_i = int(_now_us() - dur_us_i if ts_us is None else ts_us)
        self._write(
            {
                "name": name,
                "cat": cat,
                "ph": "X",
                "ts": ts_us_i,
                "dur": dur_us_i,
                "pid": self.pid,
                "tid": tid,
                "args": args or {},
            }
        )

    def metadata(self, name: str, value: str, *, tid: int | None = None) -> None:
        if not self.enabled:
            return
        self._write(
            {
                "name": name,
                "ph": "M",
                "pid": self.pid,
                "tid": 0 if tid is None else tid,
                "args": {"name": value},
            }
        )

    def _get_tid_and_name(self) -> tuple[int, str]:
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        if task is not None:
            return id(task), task.get_name()
        return threading.get_ident(), threading.current_thread().name

    def _ensure_thread_metadata(self, tid: int, name: str) -> None:
        if tid in self._thread_names:
            return
        with self._lock:
            if tid in self._thread_names:
                return
            self._thread_names.add(tid)
        self.metadata("thread_name", name, tid=tid)

    def _write(self, event: dict[str, Any]) -> None:
        if self._fh is None:
            return
        try:
            line = json.dumps(event, separators=(",", ":"), default=str)
            with self._lock:
                if self._fh is not None:
                    self._fh.write(line)
                    self._fh.write("\n")
        except Exception:
            pass


def get_profiler(
    *,
    process_name: str | None = None,
    initially_active: bool = True,
) -> ProfileRecorder:
    """Return the process-local recorder configured from SA_PROFILE_* envs."""
    global _profiler, _profiler_pid
    pid = os.getpid()
    if _profiler is not None and _profiler_pid == pid:
        if process_name is not None and _profiler.process_name != process_name:
            _profiler.process_name = process_name
            _profiler.metadata("process_name", process_name, tid=0)
        return _profiler

    config = load_profile_config()
    is_main_process = multiprocessing.current_process().name == "MainProcess"
    if config.enabled and is_main_process and _MAIN_PID_ENV not in os.environ:
        os.environ[_MAIN_PID_ENV] = str(pid)
        _prepare_run(config)
    _profiler = ProfileRecorder(
        config,
        process_name=process_name,
        initially_active=initially_active,
    )
    _profiler_pid = pid
    return _profiler


def configure_profiler(
    config: ProfileConfig,
    *,
    process_name: str | None = None,
    initially_active: bool = True,
) -> ProfileRecorder:
    """Replace the process-local recorder with an explicit configuration."""
    global _profiler, _profiler_pid
    if _profiler is not None and _profiler_pid == os.getpid():
        _profiler.stop()

    is_main_process = multiprocessing.current_process().name == "MainProcess"
    if config.enabled and is_main_process:
        _prepare_run(config)

    _profiler = ProfileRecorder(
        config,
        process_name=process_name,
        initially_active=initially_active,
    )
    _profiler_pid = os.getpid()
    return _profiler


def is_enabled(level: str = "e2e") -> bool:
    return get_profiler().includes(level)


def profile_span(
    name: str,
    *,
    cat: str = "e2e",
    level: str = "e2e",
    args: dict[str, Any] | None = None,
):
    from . import request
    if request.current.get():
        return request.scope(request.current.get(), name, cat=cat, **(args or {}))
    profiler = get_profiler()
    if not profiler.includes(level):
        return nullcontext()
    return profiler.span(name, cat=cat, level=level, args=args)


def profile_instant(
    name: str,
    *,
    cat: str = "e2e",
    level: str = "e2e",
    args: dict[str, Any] | None = None,
) -> None:
    from . import request
    if request.current.get():
        request.event(name, **(args or {}))
        return
    get_profiler().instant(name, cat=cat, level=level, args=args)


def profile_duration(
    name: str,
    *,
    dur_us: float,
    cat: str = "e2e",
    level: str = "e2e",
    args: dict[str, Any] | None = None,
    ts_us: float | None = None,
) -> None:
    from . import request
    if request.duration(name, dur_us, cat=cat, args=args, ts_us=ts_us):
        return
    get_profiler().duration(name, dur_us=dur_us, cat=cat, level=level, args=args, ts_us=ts_us)


def start_profile() -> bool:
    """Start a fresh profile session in the current process."""
    from . import request
    if request.enabled():
        raise ValueError("request profiling is active")
    profiler = get_profiler(initially_active=False)
    if not profiler.enabled or profiler.active:
        return False
    if multiprocessing.current_process().name == "MainProcess":
        _prepare_run(profiler.config)
    return profiler.start()


def stop_profile() -> bool:
    """Flush and stop the current process profile session."""
    return get_profiler(initially_active=False).stop()


def merge_profile() -> int:
    profiler = get_profiler(initially_active=False)
    profiler.stop()
    if not profiler.enabled:
        return 0
    return merge_fragments(profiler.config.fragments_dir, profiler.config.trace_file)


def _now_us() -> int:
    return time.perf_counter_ns() // 1000


def _default_process_name() -> str:
    if sys.argv and sys.argv[0]:
        return Path(sys.argv[0]).stem
    return Path(os.environ.get("_", "python")).name or "python"


def _prepare_run(config: ProfileConfig) -> None:
    config.fragments_dir.mkdir(parents=True, exist_ok=True)
    for fragment in config.fragments_dir.glob("trace.*.jsonl"):
        fragment.unlink()
    if config.trace_file.is_file():
        config.trace_file.unlink()
