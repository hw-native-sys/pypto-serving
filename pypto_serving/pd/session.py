# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Peer-indexed sessions with correlated, bounded concurrent control calls."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass

from .protocol import receive, send


class Session:
    def __init__(self, reader, writer, capacity=32):
        self.reader, self.writer = reader, writer
        self.pending = {}
        self.sequence = 0
        self.lock = asyncio.Lock()
        self.slots = asyncio.Semaphore(capacity)
        self.task = asyncio.create_task(self._read())

    async def _read(self):
        error = ConnectionError("peer control session closed")
        try:
            while True:
                reply = await receive(self.reader)
                future = self.pending.get(reply["id"])
                if future is None or future.done():
                    raise ValueError("unexpected control reply")
                if "error" in reply:
                    future.set_exception(RuntimeError(reply["error"]))
                else:
                    future.set_result(reply["result"])
        except (Exception, asyncio.CancelledError) as exc:
            error = exc if isinstance(exc, Exception) else error
        finally:
            self.writer.close()
            for future in self.pending.values():
                if not future.done():
                    future.set_exception(error)

    async def call(self, operation, payload):
        await self.slots.acquire()
        future = None
        try:
            async with self.lock:
                if self.task.done():
                    raise ConnectionError("closed peer session")
                self.sequence += 1
                sequence = self.sequence
                future = asyncio.get_running_loop().create_future()
                self.pending[sequence] = future
                try:
                    await send(self.writer, {"id": sequence, "op": operation, "payload": payload})
                except BaseException:
                    self.pending.pop(sequence, None)
                    future.cancel()
                    raise
            return await asyncio.shield(future)
        finally:
            # Keep the slot until its reply arrives, even after caller cancellation.
            if future is None:
                self.slots.release()
            else:
                def consume(done):
                    self.pending.pop(sequence, None)
                    self.slots.release()
                    if not done.cancelled():
                        done.exception()
                future.add_done_callback(consume)

    async def close(self):
        self.writer.close()
        self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)
        await self.writer.wait_closed()


@dataclass
class Entry:
    value: object
    users: int = 0
    retired: bool = False


class PeerPool:
    def __init__(self, capacity):
        self.capacity = capacity
        self.entries = {}
        self.lock = asyncio.Lock()

    @asynccontextmanager
    async def use(self, key, connect):
        async with self.lock:
            entry = self.entries.get(key)
            if entry is None:
                if len(self.entries) >= self.capacity:
                    raise RuntimeError("peer session capacity exhausted")
                entry = self.entries[key] = Entry(asyncio.create_task(connect()))
            if entry.retired:
                raise ValueError("peer identity has retired")
            entry.users += 1
        try:
            yield await asyncio.shield(entry.value)
        finally:
            async with self.lock:
                entry.users -= 1
                failed = entry.value.done() and (entry.value.cancelled() or entry.value.exception() is not None)
                if failed and not entry.users:
                    self.entries.pop(key)
                elif entry.retired and not entry.users:
                    await (await entry.value).close()
                    self.entries.pop(key)

    async def retire(self, key):
        async with self.lock:
            entry = self.entries.get(key)
            if entry is not None:
                entry.retired = True
                if not entry.users:
                    await (await entry.value).close()
                    self.entries.pop(key)

    async def close(self):
        if any(e.users for e in self.entries.values()):
            raise RuntimeError("cannot close sessions with active handoffs")
        values = await asyncio.gather(*(e.value for e in self.entries.values()))
        await asyncio.gather(*(value.close() for value in values))
        self.entries.clear()
