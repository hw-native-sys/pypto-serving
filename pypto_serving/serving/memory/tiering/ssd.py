# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Private fixed-slot files and asynchronous Host I/O; no model or device ownership."""

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import ctypes
from itertools import islice
from mmap import PAGESIZE, mmap
import os
from pathlib import Path
from queue import SimpleQueue, Empty
from tempfile import TemporaryDirectory
import threading
import time

from .policy import LRU

IOV_MAX = os.sysconf("SC_IOV_MAX")


@dataclass(eq=False)
class FilePage:
    key: object
    lengths: tuple
    valid_bytes: int
    future: Future
    slot: int = -1
    fd: int = -1
    ready: bool = False
    readers: int = 0
    retiring: bool = False

    @property
    def charge(self):
        return sum(self.lengths)


class FileStore:
    def __init__(self, root, capacity, *, policy=None, notify=None, trace=None):
        if type(capacity) is not int or capacity < PAGESIZE:
            raise ValueError("SSD capacity must hold a physical page")
        if not Path(root).is_dir():
            raise ValueError("SSD directory must already exist")
        self.capacity, self.used = capacity, 0
        self.policy = LRU() if policy is None else policy
        self.notify = notify
        self.trace = trace
        self.entries, self.classes = {}, {}
        self.failed = set()
        self.queue, self.evicted = SimpleQueue(), SimpleQueue()
        self.waiting, self.read_waiting, self.active = deque(), deque(), []
        self.lock, self.wake, self.started = threading.RLock(), threading.Event(), threading.Event()
        self.closing = self.stopping = False
        self.error = None
        self.counters = dict(written_bytes=0, read_bytes=0, evicted_bytes=0, writes=0, reads=0)
        self.root = root
        self.thread = threading.Thread(target=self._run, name="cache-file-progress", daemon=False)
        self.thread.start()
        self.started.wait()
        if self.error is not None:
            self.thread.join()
            raise self.error

    def lookup(self, key):
        with self.lock:
            entry = self.entries.get(key)
            return entry if entry is not None and entry.ready and not entry.retiring else None

    def contains(self, key):
        with self.lock:
            return key in self.entries

    def touch(self, key):
        with self.lock:
            self.policy.on_access(key, time.monotonic())

    def hold(self, key):
        with self.lock:
            entry = self.lookup(key)
            if self.closing or entry is None:
                raise OSError("SSD content is no longer available")
            entry.readers += 1
            self.policy.on_access(key, time.monotonic())
            return entry

    def release(self, entry):
        with self.lock:
            if self.entries.get(entry.key) is not entry or entry.readers <= 0:
                raise RuntimeError("stale SSD reader")
            entry.readers -= 1
        self.wake.set()

    def write(self, key, views, valid_bytes, *, trace_fields=None):
        lengths = tuple(len(view) for view in views)
        if (not lengths or any(length <= 0 or length % PAGESIZE for length in lengths)
                or not 0 < valid_bytes <= sum(lengths) <= self.capacity):
            raise ValueError("invalid SSD page geometry or capacity")
        with self.lock:
            if self.closing:
                raise RuntimeError("SSD store is closing")
            if self.error is not None:
                raise OSError(f"SSD allocator is unavailable: {self.error}")
            if key in self.entries:
                return self.entries[key]
            entry = FilePage(key, lengths, valid_bytes, Future())
            entry.future.add_done_callback(lambda _: self.wake.set())
            self.entries[key] = entry
            context = dict(trace_fields or {}, enqueued_ns=time.monotonic_ns()) if self.trace else None
            self.queue.put(("write", entry, tuple(views), entry.future, context))
        self.wake.set()
        return entry

    def read(self, entry, views, *, trace_fields=None):
        with self.lock:
            if (self.entries.get(entry.key) is not entry or not entry.ready or entry.readers <= 0
                    or tuple(map(len, views)) != entry.lengths):
                raise OSError("SSD read requires a current reader and matching layout")
            future = Future()
            future.add_done_callback(lambda _: self.wake.set())
            context = dict(trace_fields or {}, enqueued_ns=time.monotonic_ns()) if self.trace else None
            self.queue.put(("read", entry, tuple(views), future, context))
        self.wake.set()
        return future

    @staticmethod
    def _punch(fd, offset, size):
        libc = ctypes.CDLL(None, use_errno=True)
        libc.fallocate.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_longlong, ctypes.c_longlong]
        libc.fallocate.restype = ctypes.c_int
        if libc.fallocate(fd, 3, offset, size):  # KEEP_SIZE | PUNCH_HOLE
            raise OSError(ctypes.get_errno(), "SSD hole-punch failed")

    @staticmethod
    def _io(operation, entry, views):
        offset = entry.slot * entry.charge
        syscall = os.pwritev if operation == "write" else os.preadv
        pending, calls = deque(views), 0
        while pending:
            vectors = list(islice(pending, IOV_MAX))
            try:
                calls += 1
                count = syscall(entry.fd, vectors, offset)
            except InterruptedError:
                continue
            if count <= 0 or count % PAGESIZE or count > sum(map(len, vectors)):
                raise OSError("SSD I/O made no aligned progress or reached EOF")
            offset += count
            while pending and count >= len(pending[0]):
                count -= len(pending.popleft())
            if count:
                pending[0] = pending[0][count:]
        return dict(bytes=entry.valid_bytes, physical_bytes=entry.charge,
                    file_completed=True, stopped=True, syscalls=calls)

    def _execute(self, operation, entry, views, context):
        if context is None:
            return self._io(operation, entry, views)
        started, result, error = time.monotonic_ns(), None, None
        try:
            result = self._io(operation, entry, views)
            return result
        except BaseException as failure:
            error = type(failure).__name__
            raise
        finally:
            self.trace.emit("ssd_file", **context, operation=operation, started_ns=started,
                completed_ns=time.monotonic_ns(), queue_ns=started-context["enqueued_ns"],
                executor_queue_ns=started-context["dispatched_ns"], requested_bytes=entry.charge,
                bytes=entry.valid_bytes if result else 0, physical_bytes=entry.charge if result else 0,
                syscalls=result.get("syscalls") if result else None,
                file_completed=bool(result and result.get("file_completed")), error_type=error)

    def _reserve(self, entry):
        while True:
            with self.lock:
                if self.used + entry.charge <= self.capacity:
                    break
                victim = next((self.entries[key] for key in self.policy.candidates(time.monotonic())
                               if self.entries[key].ready and not self.entries[key].readers), None)
                if victim is None:
                    return False
                victim.retiring = True
                del self.entries[victim.key]
                self.policy.on_remove(victim.key)
                self.evicted.put(victim.key)
                if self.notify is not None:
                    self.notify()
            try:
                self._punch(victim.fd, victim.slot * victim.charge, victim.charge)
            except OSError as error:
                self.error = error
                raise
            with self.lock:
                self.classes[victim.lengths][1].append(victim.slot)
                self.used -= victim.charge
                self.counters["evicted_bytes"] += victim.charge
        if entry.lengths not in self.classes:
            fd = os.open(Path(self.directory.name) / f"cache-{len(self.classes)}.data",
                         os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_DIRECT, 0o600)
            self.classes[entry.lengths] = [fd, [], 0]
        allocation = self.classes[entry.lengths]
        entry.fd = allocation[0]
        if allocation[1]:
            entry.slot = allocation[1].pop()
        else:
            entry.slot = allocation[2]
            allocation[2] += 1
        # Reserve accounting before physical work, without holding the public metadata lock.
        with self.lock:
            self.used += entry.charge
        os.posix_fallocate(entry.fd, entry.slot * entry.charge, entry.charge)
        return True

    def _discard(self, entry):
        with self.lock:
            if self.entries.get(entry.key) is entry:
                del self.entries[entry.key]
            self.policy.on_remove(entry.key)
            entry.retiring = True
            self.failed.discard(entry)
            self.evicted.put(entry.key)
            if self.notify is not None:
                self.notify()
        if entry.slot >= 0:
            self._punch(entry.fd, entry.slot * entry.charge, entry.charge)
            with self.lock:
                self.classes[entry.lengths][1].append(entry.slot)
                self.used -= entry.charge

    def _probe(self):
        self.directory = TemporaryDirectory(prefix=".pypto-cache-", dir=self.root)
        fd = os.open(Path(self.directory.name) / "probe", os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_DIRECT, 0o600)
        try:
            with mmap(-1, PAGESIZE) as buffer:
                buffer[:] = b"p" * PAGESIZE
                os.posix_fallocate(fd, 0, PAGESIZE)
                if os.pwritev(fd, [buffer], 0) != PAGESIZE:
                    raise OSError("SSD direct-I/O probe was incomplete")
                buffer[:] = b"\0" * PAGESIZE
                if os.preadv(fd, [buffer], 0) != PAGESIZE or buffer[:] != b"p" * PAGESIZE:
                    raise OSError("SSD direct-I/O probe contents do not match")
                self._punch(fd, 0, PAGESIZE)
        finally:
            os.close(fd)
            os.unlink(Path(self.directory.name) / "probe")

    def _run(self):
        try:
            self._probe()
        except BaseException as error:
            self.error = error
            if hasattr(self, "directory"):
                self.directory.cleanup()
            self.started.set()
            return
        self.started.set()
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="cache-file-io") as executor:
            while True:
                self.wake.clear()
                try:
                    self._progress(executor)
                except BaseException as error:
                    self.error = error
                    # A failed allocator cannot admit writes; already submitted I/O still drains.
                    with self.lock:
                        for key, entry in tuple(self.entries.items()):
                            if not entry.future.done() and not entry.future.running():
                                entry.future.set_exception(error)
                                del self.entries[key]
                    self.waiting.clear()
                if self.stopping and self.idle():
                    break
                self.wake.wait()
        released, removed = self.used, False
        try:
            for fd, _, _ in self.classes.values():
                os.close(fd)
            self.directory.cleanup()
            removed = not Path(self.directory.name).exists()
            if not removed:
                raise OSError("SSD private directory remains after cleanup")
            with self.lock:
                for key in self.entries:
                    self.policy.on_remove(key)
                self.entries.clear()
                self.classes.clear()
                self.used = 0
        except OSError as error:
            self.error = error
        if self.trace is not None:
            self.trace.emit("ssd_closed", files_removed=removed,
                            released_bytes=released if removed else 0, used_bytes=self.used,
                            error_type=type(self.error).__name__ if self.error else None)

    def _progress(self, executor):
        for operation, entry, future, native in tuple(self.active):
            if not native.done():
                continue
            self.active.remove((operation, entry, future, native))
            try:
                result = native.result()
                with self.lock:
                    if operation == "write":
                        entry.ready = True
                        self.policy.on_publish(entry.key, time.monotonic())
                    self.counters["written_bytes" if operation == "write" else "read_bytes"] += entry.charge
                    self.counters["writes" if operation == "write" else "reads"] += 1
                future.set_result(result)  # Index is ready before write completion is visible.
            except BaseException as error:
                if operation == "write":
                    try:
                        self._discard(entry)
                    except OSError as cleanup_error:
                        self.error = cleanup_error
                else:
                    with self.lock:
                        entry.ready = False
                        self.failed.add(entry)
                        self.policy.on_remove(entry.key)
                future.set_exception(error)
        for entry in tuple(self.failed):
            if not entry.readers:
                self._discard(entry)
        while True:
            try:
                operation, entry, views, future, context = self.queue.get_nowait()
            except Empty:
                break
            if operation == "read":
                self.read_waiting.append((entry, views, future, context))
            else:
                self.waiting.append((entry, views, future, context))
        while self.read_waiting and len(self.active) < 8:
            entry, views, future, context = self.read_waiting.popleft()
            if future.set_running_or_notify_cancel():
                if context is not None:
                    context["dispatched_ns"] = time.monotonic_ns()
                native = executor.submit(self._execute, "read", entry, views, context)
                native.add_done_callback(lambda _: self.wake.set())
                self.active.append(("read", entry, future, native))
        while self.waiting and len(self.active) < 8:
            entry, views, future, context = self.waiting[0]
            if future.cancelled():
                self.waiting.popleft()
                self._discard(entry)
                continue
            if self.error is not None:
                self.waiting.popleft()
                self._discard(entry)
                future.set_exception(self.error)
                continue
            try:
                if not self._reserve(entry):
                    break
            except OSError as error:
                self.waiting.popleft()
                try:
                    self._discard(entry)
                except OSError as cleanup_error:
                    self.error = cleanup_error
                if not future.done():
                    future.set_exception(error)
                continue
            self.waiting.popleft()
            if not future.set_running_or_notify_cancel():
                self._discard(entry)
                continue
            if context is not None:
                context["dispatched_ns"] = time.monotonic_ns()
            native = executor.submit(self._execute, "write", entry, views, context)
            native.add_done_callback(lambda _: self.wake.set())
            self.active.append(("write", entry, future, native))

    def begin_close(self):
        with self.lock:
            self.closing = True
            for entry in self.entries.values():
                entry.future.cancel()
        self.wake.set()

    def idle(self):
        return not self.waiting and not self.read_waiting and not self.active and self.queue.empty()

    def close(self):
        self.begin_close()
        with self.lock:
            if any(entry.readers for entry in self.entries.values()):
                raise RuntimeError("SSD readers must drain before close")
        self.stopping = True
        self.wake.set()
        self.thread.join()
        if self.error is not None:
            raise self.error
