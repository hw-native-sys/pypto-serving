# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent owner endpoints and asynchronous cache-storage backend."""

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from itertools import count
from mmap import PAGESIZE
from queue import PriorityQueue
import threading
from typing import Protocol


class StorageBackend(Protocol):
    def size(self, source: object) -> int: ...
    def footprint(self, source: object) -> int: ...
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


class SimplerBackend:
    def __init__(self, owners, geometry):
        self.owners, self.geometry = owners, geometry
        self.tp = geometry["tp"]
        self.queue = PriorityQueue()
        self.sequence = count()
        self.records = {}
        self.accepting = True
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

    def _submit(self, priority, operation, handle=None, payload=None):
        if not self.accepting:
            raise RuntimeError("cache backend is closing")
        future = Future()
        self.queue.put_nowait((priority, next(self.sequence), future, operation, handle, payload))
        return future

    def store(self, handle, source):
        return self._submit(2, "store", handle, source)

    def load(self, handle, destination):
        return self._submit(0, "load", handle, destination)

    def release(self, handle):
        return self._submit(0, "release", handle)

    def _run(self):
        with ThreadPoolExecutor(max_workers=len(self.owners), thread_name_prefix="cache-owner-rpc") as pool:
            while True:
                _, _, future, operation, handle, source = self.queue.get()
                if not future.set_running_or_notify_cancel():
                    continue
                try:
                    if operation == "stop":
                        if self.records:
                            raise RuntimeError("DDR allocations remain during backend shutdown")
                        calls = [pool.submit(owner.call, "stop") for owner in self.owners]
                    elif operation == "release":
                        ranks, _ = self.records[handle]
                        calls = [pool.submit(self.owners[rank].call, "release", handle) for rank in ranks]
                    else:
                        ranks, spans, size = self.plan(source)
                        if operation == "store":
                            if handle in self.records:
                                raise ValueError("duplicate DDR storage generation")
                            self.records[handle] = (ranks, size)
                            allocation = (size + PAGESIZE - 1) // PAGESIZE * PAGESIZE
                            reservations = [pool.submit(self.owners[rank].call, "reserve", (handle, allocation))
                                            for rank in ranks]
                            for reservation in reservations:
                                reservation.result()
                        elif self.records[handle] != (ranks, size):
                            raise ValueError("LOAD layout differs from stored content")
                        calls = [pool.submit(self.owners[rank].call, operation, (handle, spans)) for rank in ranks]
                    results = [call.result() for call in calls]
                    if operation == "release":
                        del self.records[handle]
                    if operation in ("store", "load"):
                        result = dict(bytes=sum(item["bytes"] for item in results),
                                      stopped=all(item["stopped"] for item in results),
                                      native_completed=all(item["native_completed"] for item in results))
                    else:
                        result = None
                    future.set_result(result)
                except BaseException as error:
                    future.set_exception(error)
                if operation == "stop":
                    return

    def close(self):
        future = self._submit(3, "stop")
        self.accepting = False
        future.result(timeout=120)
        self.thread.join(5)
        if self.thread.is_alive():
            raise RuntimeError("cache progress thread is still active")
        for owner in self.owners:
            owner.parent.close()
