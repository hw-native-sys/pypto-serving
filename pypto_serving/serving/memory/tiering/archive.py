# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""SSD locations and Host-to-owner handoff, using the existing cache lifecycle."""

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
import ctypes
import logging
from mmap import PAGESIZE
from queue import SimpleQueue, Empty
import threading

from ..kv_cache import KVCacheRestoreError
from .manager import CacheTier
from .pool import host_views
from .ssd import FileStore

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class Window:
    partition: int
    extents: tuple
    state: str = "free"


@dataclass
class Restore:
    page: object
    destination: object
    entry: object
    future: Future
    extents: tuple = ()
    release_reader: bool = True
    request: str | None = None


class SSDArchive:
    def __init__(self, tier, pool, root, capacity):
        self.tier, self.backend, self.pool = tier, tier.backend, pool
        geometry, tp = self.backend.geometry, self.backend.tp
        ranks = len(self.backend.owners)
        maximum = max(self.backend.footprint((name, 0, 0)) // tp for name in geometry["groups"])
        if capacity < maximum * tp:
            raise ValueError("SSD budget is smaller than one complete TP page")
        self.window_bytes = max(maximum, min(8 << 20, tier.capacity // (4 * ranks))) // PAGESIZE * PAGESIZE
        work_bytes = 2 * ranks * self.window_bytes
        if work_bytes + maximum * tp > tier.capacity:
            raise ValueError("DDR budget cannot hold SSD work windows and one cache page")
        self.windows, self.allocations = [], []
        try:
            for _ in range(2):
                handle = object()
                extents = self.backend.extents.reserve(handle, ranks, self.window_bytes)
                if extents is None:
                    raise ValueError("cannot reserve SSD work windows atomically")
                self.allocations.append(handle)
                self.windows.extend(Window(partition, extents[partition * tp:(partition + 1) * tp])
                                    for partition in range(ranks // tp))
            self.files = FileStore(root, capacity, notify=self._notify)
        except BaseException:
            for handle in self.allocations:
                self.backend.extents.release(handle)
            raise
        tier.capacity -= work_bytes
        self.work_bytes = work_bytes
        self.pages, self.writers, self.units, self.members = {}, {}, {}, {}
        self.queue, self.promotions, self.completed = SimpleQueue(), SimpleQueue(), SimpleQueue()
        self.restored = SimpleQueue()
        self.pending = [deque() for _ in range(ranks // tp)]
        self.lock, self.wake = threading.RLock(), threading.Event()
        self.closing = self.stopping = False
        self.running = 0
        self.executor = ThreadPoolExecutor(max_workers=min(8, 2 * ranks // tp), thread_name_prefix="cache-ssd-handoff")
        self.thread = threading.Thread(target=self._run, name="cache-ssd-progress", daemon=False)
        tier.lower = self
        self.thread.start()

    def _notify(self):
        notify = self.tier.notify
        if notify is not None:
            notify()

    def _stored(self, key):
        self.completed.put(key)
        self._notify()

    def pop_restored(self):
        while True:
            try:
                yield from self.restored.get_nowait()
            except Empty:
                return

    def stored(self, page):
        if self.closing or self.files.contains(page.key):
            return
        reader, = self.tier.acquire((page.key,))
        try:
            entry = self.files.write(page.key, self.backend.host_ranges(page.handle), page.size)
        except Exception as error:
            self.tier.release((reader,))
            logger.warning("kv_ssd_store_failed error=%s", error)
            return
        self.pages[page.key] = page
        self.writers[page.key] = reader, entry
        entry.future.add_done_callback(lambda _, key=page.key: self._stored(key))

    def publish(self, identity, keys):
        if self.closing or identity in self.units or not keys:
            return
        self.units[identity] = keys  # Share the existing immutable member tuple.
        for key in keys:
            self.members.setdefault(key, set()).add(identity)

    def removed(self, key):
        if key in self.tier.pages or self.files.contains(key):
            return
        self.pages.pop(key, None)
        for identity in tuple(self.members.get(key, ())):
            for member in self.units.pop(identity):
                owners = self.members[member]
                owners.remove(identity)
                if not owners:
                    del self.members[member]

    def lookup(self, key):
        return self.pages.get(key) if self.files.lookup(key) is not None else None

    def touch(self, key):
        self.files.touch(key)

    def acquire(self, keys):
        """Pin selected SSD sources atomically before any destination or native work."""
        with self.files.lock:
            if any(self.files.lookup(key) is None for key in keys):
                return None
            return {key: self.files.hold(key) for key in keys}

    def load(self, page, destination, entry=None, *, request=None):
        future = Future()
        future.set_running_or_notify_cancel()  # A waiter cancellation does not cancel native work.
        release_reader = entry is None
        try:
            if entry is None:
                entry = self.files.hold(page.key)
            expected = self.backend.footprint(destination) // self.backend.tp
            if entry.lengths != (expected,) * self.backend.tp:
                if release_reader:
                    self.files.release(entry)
                raise OSError("SSD page layout does not match owner geometry")
        except OSError as error:
            future.set_exception(KVCacheRestoreError(str(error)))
            return future
        self.queue.put(Restore(page, destination, entry, future,
                               release_reader=release_reader, request=request))
        self.wake.set()
        return future

    def _run(self):
        while not self.stopping:
            self.wake.clear()
            with self.lock:
                while True:
                    try:
                        job = self.queue.get_nowait()
                    except Empty:
                        break
                    self.pending[job.destination[1]].append(job)
                for window in self.windows:
                    waiting = self.pending[window.partition]
                    if window.state != "free" or not waiting:
                        continue
                    batch, cursor = [], 0
                    while waiting and cursor + waiting[0].entry.lengths[0] <= self.window_bytes:
                        job = waiting.popleft()
                        length = job.entry.lengths[0]
                        job.extents = tuple((offset + cursor, length) for offset, _ in window.extents)
                        batch.append(job)
                        cursor += length
                    window.state = "transfer"
                    self.running += 1
                    self.executor.submit(self._stage, window, batch)
            self.wake.wait()

    def _stage(self, window, batch):
        good, reads = [], []
        poisoned = False
        try:
            for job in batch:
                views = host_views(self.pool, job.extents, writable=True)
                try:
                    reads.append((job, views, self.files.read(job.entry, views)))
                except OSError as error:
                    if job.release_reader:
                        self.files.release(job.entry)
                    job.future.set_exception(KVCacheRestoreError(str(error)))
            wait([future for _, _, future in reads])
            for job, views, future in reads:
                try:
                    proof = future.result()
                    if not proof.get("file_completed") or proof.get("bytes") != job.page.size:
                        raise OSError("SSD read has no complete file proof")
                    good.append((job, views))
                except Exception as error:
                    job.future.set_exception(KVCacheRestoreError(str(error)))
                finally:
                    if job.release_reader:
                        self.files.release(job.entry)
            if good:
                native = self.backend.load_host_batch(tuple(
                    (job.page.handle, job.destination, job.extents, job.request) for job, _ in good))
                wait(native)
                for (job, _), future in zip(good, native):
                    proof = CacheTier.completed(future, job.page.size)
                    job.future.set_result(proof)
            with self.lock:
                if (good and not self.closing and not any(
                        other.partition == window.partition and other.state == "promotion"
                        for other in self.windows)):
                    window.state = "promotion"
                    self.promotions.put((window, good))
                else:
                    window.state = "free"
        except BaseException as error:
            # Unknown native completion keeps the work ranges and HBM targets isolated.
            poisoned = True
            for job in batch:
                if not job.future.done():
                    job.future.set_exception(error)
        finally:
            with self.lock:
                if poisoned:
                    window.state = "quarantined"
                self.running -= 1
            keys = tuple(job.page.key for job in batch if not job.release_reader)
            if keys:
                self.restored.put(keys)
            self._notify()
            self.wake.set()

    def _copy_host(self, handle, views, size):
        targets = self.backend.host_ranges(handle, writable=True)
        libc = ctypes.CDLL(None)
        libc.memmove.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        libc.memmove.restype = ctypes.c_void_p
        for target, source in zip(targets, views):
            libc.memmove(ctypes.addressof(ctypes.c_ubyte.from_buffer(target)),
                         ctypes.addressof(ctypes.c_ubyte.from_buffer(source)), len(target))
        self.backend.host_ready(handle)
        return dict(host_completed=True, bytes=size)

    def pump(self):
        while True:
            try:
                key = self.completed.get_nowait()
            except Empty:
                break
            page, entry = self.writers.pop(key)
            try:
                entry.future.result()
            except Exception as error:
                logger.warning("kv_ssd_store_failed error=%s", error)
            self.tier.release((page,))
            self.removed(key)
        while True:
            try:
                key = self.files.evicted.get_nowait()
            except Empty:
                break
            self.removed(key)
        while True:
            try:
                window, jobs = self.promotions.get_nowait()
            except Empty:
                break
            futures = []
            if not self.closing:
                for job, views in jobs:
                    future = self.tier.warm(job.page.key, job.destination,
                        lambda handle, views=views, size=job.page.size:
                            self.executor.submit(self._copy_host, handle, views, size))
                    if future is not None:
                        futures.append(future)
            remaining = [len(futures)]
            def release(_, window=window, remaining=remaining):
                with self.lock:
                    remaining[0] -= 1
                    if remaining[0] == 0:
                        window.state = "free"
                self.wake.set()
            if not futures:
                with self.lock:
                    window.state = "free"
                self.wake.set()
            for future in futures:
                future.add_done_callback(release)

    def begin_close(self):
        self.closing = True
        self.files.begin_close()
        self.wake.set()

    def idle(self):
        with self.lock:
            return (not self.writers and not self.running and self.queue.empty() and self.promotions.empty()
                    and self.restored.empty() and not any(self.pending)
                    and all(w.state == "free" for w in self.windows)
                    and self.files.idle())

    def close(self):
        if not self.closing or not self.idle():
            raise RuntimeError("SSD handoff and readers must drain before owner retirement")
        self.stopping = True
        self.wake.set()
        self.thread.join()
        self.executor.shutdown(wait=True)
        try:
            self.files.close()
        except OSError as error:
            # File I/O threads have joined. A disk cleanup error is not unknown DMA.
            logger.warning("kv_ssd_cleanup_failed error=%s", error)
        for handle in self.allocations:
            self.backend.extents.release(handle)
        self.pages.clear()
        self.units.clear()
        self.members.clear()
