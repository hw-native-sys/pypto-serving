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
        ssd = {}
        if cache.archive is not None:
            archive = cache.archive
            files = archive.files
            with files.lock:
                ssd = dict(capacity_bytes=files.capacity, used_bytes=files.used, pages=len(files.entries),
                           enqueued_operations=files.queue.qsize(), waiting_writes=len(files.waiting),
                           waiting_reads=len(files.read_waiting), active_operations=len(files.active),
                           counters=dict(files.counters))
            with archive.lock:
                ssd.update(work_bytes=archive.work_bytes, window_bytes=archive.window_bytes,
                           window_states=[window.state for window in archive.windows],
                           waiting_loads=sum(map(len, archive.pending)), running_handoffs=archive.running,
                           write_readers=len(archive.writers))
        self.emit("state", capacity_bytes=tier.capacity, used_bytes=tier.used,
                  pages=len(tier.pages), queued_pages=len(tier.queued),
                  store_inflight=len(tier.inflight), retiring_pages=len(tier.retiring),
                  joint_checkpoints=len(tier.units), pending_checkpoints=len(tier.pending_units),
                  restore_requests=len(cache.pending), restore_pages=len(cache.loads),
                  quarantined=bool(tier.error), counters=dict(tier.counters), dropped_events=self.dropped,
                  ssd=ssd,
                  oldest_store_age_ns=now-tier.queued[0].queued_ns if tier.queued else 0)

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
