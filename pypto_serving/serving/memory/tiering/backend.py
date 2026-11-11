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
from queue import Empty, PriorityQueue
import logging
import threading
import time
from typing import Protocol

from .pool import Extents

BATCH_BYTES = 256 << 20
BATCH_SPANS = 32768
logger = logging.getLogger(__name__)


class StorageBackend(Protocol):
    def size(self, source: object) -> int: ...
    def footprint(self, source: object) -> int: ...
    def reserve(self, handle: int, source: object) -> bool: ...
    def store(self, handle: int, source: object) -> Future: ...
    def load(self, handle: int, destination: object) -> Future: ...
    def release(self, handle: int) -> Future: ...
    def close(self) -> None: ...


@dataclass
class Endpoint:
    parent: object
    child: object
    sequence: object
    lock: object
    stopped: object

    def call(self, operation, payload=None):
        with self.lock:
            if self.stopped.value:
                raise RuntimeError("cache owner already stopped")
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
            if operation == "stop":
                self.stopped.value = True
            return result


def endpoints(context, ranks):
    result = []
    for _ in range(ranks):
        parent, child = context.Pipe()
        result.append(Endpoint(parent, child, context.Value("Q", 0), context.Lock(), context.Value("b", False)))
    return tuple(result)


@dataclass
class Storage:
    ranks: tuple
    spans: tuple
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


class SimplerBackend:
    def __init__(self, owners, geometry, capacity):
        self.owners, self.geometry = owners, geometry
        self.tp = geometry["tp"]
        self.extents = Extents(capacity)
        self.queue = PriorityQueue()
        self.sequence = count()
        self.records = {}
        self.lock = threading.RLock()
        self.wake = threading.Event()
        self.accepting = True
        self.closing = False
        self.close_error = None
        self.metrics = {direction: dict(batches=0, pages=0, bytes=0, rpc_ns=0, native_ns=0,
                                       submit_ns=0, wait_ns=0, queue_ns=0)
                        for direction in ("store", "load")}
        self.thread = threading.Thread(target=self._run, name="cache-transfer-progress", daemon=False)
        self.thread.start()

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
            if not self.accepting:
                raise RuntimeError("cache backend is closing")
            extents = self.extents.reserve(handle, len(ranks), size)
            if extents is None:
                return False
            self.records[handle] = Storage(ranks, spans, size, extents)
            return True

    def _submit(self, operation, handle, source):
        ranks, spans, size = self.plan(source)
        with self.lock:
            if not self.accepting:
                raise RuntimeError("cache backend is closing")
            storage = self.records[handle]
            if (ranks != storage.ranks or size != storage.size or storage.uncertain
                    or (operation == "load" and not storage.complete)
                    or (operation == "store" and (storage.complete or storage.active))):
                raise ValueError("cache storage layout or generation mismatch")
            future = Future()
            job = Job(future, operation, handle, spans, storage, time.monotonic_ns())
            storage.active += 1
            self.queue.put_nowait((0 if operation == "load" else 1, next(self.sequence), job))
            self.wake.set()
            return future

    def store(self, handle, source):
        return self._submit("store", handle, source)

    def load(self, handle, destination):
        return self._submit("load", handle, destination)

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

    def _finish(self, jobs, calls, dispatched):
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
                job.future.set_result(dict(bytes=self.tp * job.storage.size,
                                           stopped=True, native_completed=True))
            logger.debug("kv_offload_batch direction=%s pages=%d owners=%d bytes=%d rpc_ms=%.3f",
                         jobs[0].operation, len(jobs), len(replies),
                         self.tp * sum(job.storage.size for job in jobs),
                         (time.monotonic_ns() - dispatched) / 1e6)
            stats = self.metrics[jobs[0].operation]
            stats["batches"] += len(replies)
            stats["pages"] += len(jobs)
            stats["bytes"] += self.tp * sum(job.storage.size for job in jobs)
            stats["rpc_ns"] += (time.monotonic_ns() - dispatched) * len(replies)
            stats["queue_ns"] += sum(dispatched - job.enqueued for job in jobs)
            for reply in replies:
                stats["native_ns"] += reply["completed_ns"] - reply["started_ns"]
                stats["submit_ns"] += reply["submitted_ns"] - reply["started_ns"]
                stats["wait_ns"] += reply["completed_ns"] - reply["submitted_ns"]
        except BaseException as error:
            # An uncertain batch retains all ranges and the caller's HBM pins.
            for job in jobs:
                job.storage.uncertain = True
                if not job.future.done():
                    job.future.set_exception(error)

    def _run(self):
        pending = [dict(store=deque(), load=deque()) for _ in range(len(self.owners) // self.tp)]
        previous_direction = {}
        active = {}
        with ThreadPoolExecutor(max_workers=len(self.owners), thread_name_prefix="cache-owner-rpc") as executor:
            while True:
                self.wake.clear()
                for partition, (jobs, calls, dispatched) in tuple(active.items()):
                    if all(call.done() for call in calls):
                        self._finish(jobs, calls, dispatched)
                        del active[partition]
                while True:
                    try:
                        item = self.queue.get_nowait()
                    except Empty:
                        break
                    job = item[2]
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
                    active[partition] = jobs, calls, dispatched
                    previous_direction[partition] = direction
                if self.closing and not active and self.queue.empty() and all(not any(q.values()) for q in pending):
                    try:
                        if any(storage.active or storage.uncertain for storage in self.records.values()):
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
        self.accepting = False
        self.closing = True
        self.wake.set()
        self.thread.join()
        if self.close_error is not None:
            raise self.close_error
        for owner in self.owners:
            owner.parent.close()
