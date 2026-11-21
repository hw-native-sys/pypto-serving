# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Opt-in DDR diagnostics; producers never perform file I/O or wait for space."""

import json
import logging
import os
from pathlib import Path
from queue import Empty, Full, Queue
import threading
import time
import uuid

logger = logging.getLogger(__name__)


class OffloadTrace:
    def __init__(self, directory):
        self.session = uuid.uuid4().hex
        self.pid = os.getpid()
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / f"offload.{self.pid}.{self.session}.jsonl"
        self.stream = self.path.open("x", encoding="utf-8")
        self.queue = Queue(maxsize=4096)
        self.lock = threading.Lock()
        self.closed, self.failed, self.dropped = False, False, 0
        self.last_snapshot_ns = 0
        self.thread = threading.Thread(target=self._write, name="cache-trace-writer", daemon=True)
        self.thread.start()
        self.emit("start", wall_ns=time.time_ns())

    @classmethod
    def from_env(cls):
        directory = os.environ.get("PYPTO_SERVING_KV_OFFLOAD_TRACE_DIR")
        if not directory:
            return None
        try:
            return cls(directory)
        except Exception:
            logger.warning("DDR trace unavailable; cache execution is unchanged", exc_info=True)
            return None

    def emit(self, event, **fields):
        try:
            with self.lock:
                if self.closed or self.failed:
                    return
                record = dict(schema=1, session=self.session, pid=self.pid,
                              event=event, ts_ns=time.monotonic_ns(), **fields)
                try:
                    self.queue.put_nowait(record)
                except Full:
                    self.dropped += 1
        except Exception:
            # Diagnostics must not change native proof, references or error handling.
            self.failed = True

    def snapshot(self, cache, *, force=False):
        try:
            self._snapshot(cache, force=force)
        except Exception:
            self.failed = True

    def _snapshot(self, cache, *, force=False):
        now = time.monotonic_ns()
        if not force and now - self.last_snapshot_ns < 1_000_000_000:
            return
        self.last_snapshot_ns = now
        tier = cache.tier
        self.emit("state", **cache.sources[0].snapshot(), primary_tier=cache.sources[0].name,
                  tiers={source.name: source.snapshot() for source in cache.sources[1:]},
                  checkpoint_candidates=len(cache.checkpoints.units),
                  restore_requests=len(cache.pending), restore_pages=len(cache.loads),
                  quarantined=bool(tier.error), dropped_events=self.dropped)

    def _write(self):
        try:
            while True:
                try:
                    record = self.queue.get(timeout=1)
                except Empty:
                    self.stream.flush()
                    if self.closed:
                        break
                    continue
                if record is None:
                    break
                self.stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            self.stream.write(json.dumps(dict(schema=1, session=self.session, pid=self.pid,
                event="end", ts_ns=time.monotonic_ns(), dropped_events=self.dropped,
                complete=not self.failed and self.dropped == 0)) + "\n")
        except Exception:
            self.failed = True
            logger.warning("DDR trace write failed; cache execution is unchanged", exc_info=True)
        finally:
            try:
                self.stream.close()
            except Exception:
                self.failed = True

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
        try:
            self.queue.put(None, timeout=1)
        except Full:
            self.failed = True
        self.thread.join(2)
        if self.thread.is_alive() or self.failed or self.dropped:
            logger.warning("DDR trace incomplete: dropped=%d writer_active=%s",
                           self.dropped, self.thread.is_alive())
