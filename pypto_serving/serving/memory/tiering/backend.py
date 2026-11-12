# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent owner endpoints and pooled asynchronous cache transfers."""

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from itertools import count
from mmap import PAGESIZE
from queue import Empty, SimpleQueue
import logging
import threading
import time
from typing import Protocol

from .pool import Extents
from .pool import host_views

BATCH_BYTES = 256 << 20
BATCH_SPANS = 32768
logger = logging.getLogger(__name__)


class StorageBackend(Protocol):
    def size(self, source: object) -> int: ...
    def footprint(self, source: object) -> int: ...
    def reserve(self, handle: int, source: object) -> bool: ...
    def store(self, handle: int, source: object) -> Future: ...
    def load(self, handle: int, destination: object, *, request=None) -> Future: ...
    def release(self, handle: int) -> Future: ...
    def pop_completed(self, operation: str): ...
    def set_notify(self, notify): ...
    def close(self) -> None: ...


@dataclass
class Endpoint:
    parent: object
    child: object
    sequence: object
    lock: object
    stopped: object
    failed: object

    def call(self, operation, payload=None):
        with self.lock:
            if self.stopped.value:
                raise RuntimeError("cache owner already stopped")
            if self.failed.value:
                raise RuntimeError("cache owner endpoint is quarantined")
            try:
                self.sequence.value += 1
                sequence = self.sequence.value
                self.parent.send((sequence, operation, payload))
                if not self.parent.poll(60):
                    raise RuntimeError("cache owner completion unknown: response timeout")
                received, success, result = self.parent.recv()
                if received != sequence:
                    raise RuntimeError("cache owner completion identity mismatch")
                if not success:
                    raise RuntimeError(f"cache owner failed: {result}")
            except Exception:
                self.failed.value = True
                raise
            if operation == "stop":
                self.stopped.value = True
            return result


def endpoints(context, ranks):
    result = []
    for _ in range(ranks):
        parent, child = context.Pipe()
        result.append(Endpoint(parent, child, context.Value("Q", 0), context.Lock(),
                               context.Value("b", False), context.Value("b", False)))
    return tuple(result)


@dataclass
class Storage:
    ranks: tuple
    size: int
    extents: tuple
    active: int = 0
    complete: bool = False
    uncertain: bool = False


@dataclass
class Job:
    future: Future
    operation: str
    handle: int
    spans: tuple
    storage: Storage
    enqueued: int
    group: str
    request: str | None
    work: bool = False


class SimplerBackend:
    def __init__(self, owners, geometry, capacity, *, pool=None, trace=None):
        self.owners, self.geometry = owners, geometry
        self.pool = pool
        self.tp = geometry["tp"]
        minimum = max(self.footprint((name, 0, 0)) for name in geometry["groups"])
        if capacity < minimum:
            raise ValueError(f"DDR pool capacity {capacity} is smaller than the largest page footprint {minimum}")
        self.extents = Extents(capacity)
        self.queue = SimpleQueue()
        self.completed = {operation: SimpleQueue() for operation in ("store", "load")}
        self.notify = None
        self.records = {}
        self.work = {}
        self.lock = threading.RLock()
        self.wake = threading.Event()
        self.closing = False
        self.close_error = None
        self.trace = trace
        self.batch_ids = count(1)
        self.thread = threading.Thread(target=self._run, name="cache-transfer-progress", daemon=False)
        self.thread.start()

    def set_notify(self, notify):
        self.notify = notify

    def pop_completed(self, operation):
        queue = self.completed[operation]
        while True:
            try:
                yield from queue.get_nowait()
            except Empty:
                return

    def size(self, source):
        group, _, _ = source
        return self.tp * sum(layers * page_bytes for _, layers, _, page_bytes in self.geometry["groups"][group])

    def footprint(self, source):
        size = self.size(source) // self.tp
        return self.tp * ((size + PAGESIZE - 1) // PAGESIZE * PAGESIZE)

    def plan(self, source):
        group, partition, page = source
        if not 0 <= partition < len(self.owners) // self.tp:
            raise ValueError("offload cache partition out of range")
        spans, offset = [], 0
        for name, layers, blocks, page_bytes in self.geometry["groups"][group]:
            if not 0 <= page < blocks:
                raise ValueError("offload page outside physical region")
            spans.append((name, page * page_bytes, offset, page_bytes, layers, blocks * page_bytes))
            offset += layers * page_bytes
        return tuple(range(partition * self.tp, (partition + 1) * self.tp)), tuple(spans), offset

    def reserve(self, handle, source):
        ranks, spans, size = self.plan(source)
        if size > BATCH_BYTES or len(spans) > BATCH_SPANS:
            raise ValueError("cache page exceeds native batch capacity")
        with self.lock:
            if self.closing:
                raise RuntimeError("cache backend is closing")
            extents = self.extents.reserve(handle, len(ranks), size)
            if extents is None:
                return False
            self.records[handle] = Storage(ranks, size, extents)
            return True

    def _submit(self, operation, handle, source, *, request=None):
        ranks, spans, size = self.plan(source)
        with self.lock:
            if self.closing:
                raise RuntimeError("cache backend is closing")
            storage = self.records[handle]
            if (ranks != storage.ranks or size != storage.size or storage.uncertain
                    or (operation == "load" and not storage.complete)
                    or (operation == "store" and (storage.complete or storage.active))):
                raise ValueError("cache storage layout or generation mismatch")
            future = Future()
            job = Job(future, operation, handle, spans, storage, time.monotonic_ns(), source[0], request)
            storage.active += 1
            self.queue.put_nowait((job,))
            self.wake.set()
            return future

    def store(self, handle, source):
        return self._submit("store", handle, source)

    def load(self, handle, destination, *, request=None):
        return self._submit("load", handle, destination, request=request)

    def host_ranges(self, handle, *, writable=False):
        with self.lock:
            storage = self.records[handle]
            if storage.uncertain or (not writable and not storage.complete):
                raise RuntimeError("DDR range has no confirmed contents")
            return host_views(self.pool, storage.extents, writable=writable)

    def host_ready(self, handle):
        with self.lock:
            storage = self.records[handle]
            if storage.active or storage.complete or storage.uncertain:
                raise RuntimeError("DDR host initialization identity mismatch")
            storage.complete = True

    def discard_host(self, handle):
        with self.lock:
            storage = self.records[handle]
            if storage.active or storage.uncertain:
                raise RuntimeError("cannot discard a Host range used by native work")
            self.extents.release(handle)
            del self.records[handle]

    def load_host_batch(self, pages):
        """Submit preauthorized, initialized work ranges through the existing copy queue."""
        jobs = []
        handles = set()
        with self.lock:
            if self.closing:
                raise RuntimeError("cache backend is closing")
            for handle, destination, extents, request in pages:
                if handle in self.work or handle in handles:
                    raise ValueError("work handle is already in flight")
                handles.add(handle)
                ranks, spans, size = self.plan(destination)
                if len(extents) != len(ranks) or any(length < size for _, length in extents):
                    raise ValueError("work range does not match native geometry")
                storage = Storage(ranks, size, extents, active=1, complete=True)
                jobs.append(Job(Future(), "load", handle, spans, storage, time.monotonic_ns(),
                                destination[0], request, work=True))
            for job in jobs:
                self.work[job.handle] = job.storage
            self.queue.put_nowait(tuple(jobs))
        self.wake.set()
        return tuple(job.future for job in jobs)

    def release(self, handle):
        future = Future()
        with self.lock:
            storage = self.records[handle]
            if storage.active or storage.uncertain or not storage.complete:
                raise RuntimeError("cache storage still has native users")
            self.extents.release(handle)
            del self.records[handle]
        future.set_result(None)
        return future

    def _finish(self, jobs, calls, dispatched, batch_id):
        try:
            replies = [call.result() for call in calls]
            for reply in replies:
                if tuple(row["handle"] for row in reply["pages"]) != tuple(job.handle for job in jobs):
                    raise RuntimeError("cache batch completion identity mismatch")
            for index, job in enumerate(jobs):
                results = [reply["pages"][index] for reply in replies]
                if any(not row["stopped"] or not row["native_completed"]
                       or row["bytes"] != job.storage.size for row in results):
                    raise RuntimeError("cache batch has incomplete native proof")
            for job in jobs:
                with self.lock:
                    job.storage.active -= 1
                    job.storage.complete = True
                    if self.work.get(job.handle) is job.storage:
                        del self.work[job.handle]
                job.future.set_result(dict(bytes=self.tp * job.storage.size,
                                           stopped=True, native_completed=True))
            logger.debug("kv_offload_batch direction=%s pages=%d owners=%d bytes=%d rpc_ms=%.3f",
                         jobs[0].operation, len(jobs), len(replies),
                         self.tp * sum(job.storage.size for job in jobs),
                         (time.monotonic_ns() - dispatched) / 1e6)
        except BaseException as error:
            # An uncertain batch retains all ranges and the caller's HBM pins.
            for job in jobs:
                job.storage.uncertain = True
                if not job.future.done():
                    job.future.set_exception(error)
            if self.trace is not None:
                self.trace.emit("batch_failed", batch_id=batch_id, direction=jobs[0].operation,
                    requests=sorted({job.request for job in jobs if job.request is not None}),
                    pages=len(jobs), native_completed=False, error_type=type(error).__name__)
            return
        finally:
            # One notification per batch, after every TP owner has returned.
            handles = tuple(job.handle for job in jobs if not job.work)
            if handles:
                self.completed[jobs[0].operation].put(handles)
                notify = self.notify
                if notify is not None:
                    notify()
        if self.trace is not None:
            try:
                ended = time.monotonic_ns()
                groups = {}
                for job in jobs:
                    groups[job.group] = groups.get(job.group, 0) + 1
                common = dict(batch_id=batch_id, direction=jobs[0].operation, pages=len(jobs),
                    bytes=sum(job.storage.size for job in jobs), groups=groups,
                    requests=sorted({job.request for job in jobs if job.request is not None}),
                    queue_ns=sum(dispatched-job.enqueued for job in jobs),
                    queue_max_ns=max(dispatched-job.enqueued for job in jobs),
                    rpc_start_ns=dispatched, rpc_end_ns=ended)
                for rank, reply in zip(jobs[0].storage.ranks, replies):
                    self.trace.emit("batch", **common, partition=rank // self.tp, owner=rank,
                        prepare_start_ns=reply["prepare_start_ns"], prepare_end_ns=reply["prepare_end_ns"],
                        native_start_ns=reply["started_ns"], submitted_ns=reply["submitted_ns"],
                        completed_ns=reply["completed_ns"], native_commands=reply["native_commands"])
            except Exception:
                self.trace.emit("diagnostic_failed", batch_id=batch_id, reason="batch_metadata")
                logger.warning("DDR batch diagnostics unavailable; native result is unchanged", exc_info=True)

    def _run(self):
        pending = [dict(store=deque(), load=deque()) for _ in range(len(self.owners) // self.tp)]
        previous_direction = {}
        active = {}
        with ThreadPoolExecutor(max_workers=len(self.owners), thread_name_prefix="cache-owner-rpc") as executor:
            while True:
                self.wake.clear()
                for partition, (jobs, calls, dispatched, batch_id) in tuple(active.items()):
                    if all(call.done() for call in calls):
                        self._finish(jobs, calls, dispatched, batch_id)
                        del active[partition]
                while True:
                    try:
                        jobs = self.queue.get_nowait()
                    except Empty:
                        break
                    for job in jobs:
                        pending[job.storage.ranks[0] // self.tp][job.operation].append(job)
                for partition, waiting in enumerate(pending):
                    if partition in active or not any(waiting.values()):
                        continue
                    direction = ("load" if waiting["load"] and
                                 (previous_direction.get(partition) != "load" or not waiting["store"])
                                 else "store")
                    queue = waiting[direction]
                    jobs, payload_bytes, descriptors = [], 0, 0
                    while queue:
                        job = queue.popleft()
                        size, count = job.storage.size, len(job.spans)
                        if jobs and (payload_bytes + size > BATCH_BYTES
                                     or descriptors + count > BATCH_SPANS):
                            queue.appendleft(job)
                            break
                        if not job.future.set_running_or_notify_cancel():
                            with self.lock:
                                job.storage.active -= 1
                                if self.work.get(job.handle) is job.storage:
                                    del self.work[job.handle]
                                if job.operation == "store":
                                    self.extents.release(job.handle)
                                    del self.records[job.handle]
                            continue
                        jobs.append(job)
                        payload_bytes += size
                        descriptors += count
                    if not jobs:
                        continue
                    dispatched = time.monotonic_ns()
                    calls = []
                    for copy_index, rank in enumerate(jobs[0].storage.ranks):
                        payload = tuple((job.handle, *job.storage.extents[copy_index], job.spans) for job in jobs)
                        call = executor.submit(self.owners[rank].call, "copy_batch", (direction, payload))
                        call.add_done_callback(lambda _: self.wake.set())
                        calls.append(call)
                    active[partition] = jobs, calls, dispatched, next(self.batch_ids)
                    previous_direction[partition] = direction
                if self.closing and not active and self.queue.empty() and all(not any(q.values()) for q in pending):
                    try:
                        if any(storage.active or storage.uncertain
                               for storage in (*self.records.values(), *self.work.values())):
                            raise RuntimeError("cache owner work is still uncertain")
                        calls = [executor.submit(owner.call, "stop") for owner in self.owners]
                        for call in calls:
                            call.result()
                        self.records.clear()
                        self.extents.allocations.clear()
                    except BaseException as error:
                        self.close_error = error
                    return
                self.wake.wait(.01)

    def close(self):
        with self.lock:
            self.closing = True
        self.wake.set()
        self.thread.join()
        if self.close_error is not None:
            raise self.close_error
        for owner in self.owners:
            owner.parent.close()
