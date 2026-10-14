# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded completion observation, independent of the device command FIFO."""

from __future__ import annotations

import asyncio
import threading
import time

import msgspec

from pypto_serving.tools.profile import profile_span

from .worker_api import (
    OP_POLL_TRANSFER_CHUNK, TransferPollRequest, TransferPollResponse,
    MAX_WORKER_PD_PAYLOAD_BYTES, decode_worker_payload, encode_worker_payload,
)


class _ProgressReply(msgspec.Struct, frozen=True):
    payload: bytes = b""
    error: str = ""


class ProgressServer:
    """Worker-local terminal fact reader; never submits work or touches tensors."""

    def __init__(self, channel, poll, *, limit):
        self.channel, self.poll, self.limit = channel, poll, limit
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, name="pd-progress", daemon=False)
        self._thread.start()

    def _serve(self):
        try:
            while not self._stop.is_set():
                if not self.channel.poll(.05):
                    continue
                try:
                    requests = decode_worker_payload(
                        self.channel.recv_bytes(MAX_WORKER_PD_PAYLOAD_BYTES), tuple[TransferPollRequest, ...],
                    )
                    jobs = tuple(request.job_id for request in requests)
                    if not jobs or len(jobs) > self.limit or len(set(jobs)) != len(jobs):
                        raise ValueError("invalid transfer completion poll batch")
                    with profile_span("pd.worker.poll_progress", cat="pd", args={"jobs": len(jobs)}):
                        reply = _ProgressReply(encode_worker_payload(tuple(self.poll(job) for job in jobs)))
                except (EOFError, OSError):
                    break
                except Exception as exc:
                    reply = _ProgressReply(error=str(exc))
                self.channel.send_bytes(encode_worker_payload(reply))
        except (EOFError, OSError):
            # Channel loss is not native cancellation or proof of non-submission.
            pass
        finally:
            self.channel.close()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2)
        if self._thread.is_alive():
            raise RuntimeError("transfer progress reader did not exit")


class ProgressClient:
    """One serialized, bounded local pipe; cancellation consumes the in-flight reply."""

    def __init__(self, channel, *, timeout):
        self.channel, self.timeout = channel, timeout
        self._lock = asyncio.Lock()
        self._closed = False

    async def call(self, operation, payload):
        if operation != OP_POLL_TRANSFER_CHUNK:
            raise ValueError("progress channel accepts completion queries only")
        async with self._lock:
            if self._closed:
                raise RuntimeError("transfer progress channel is closed")
            task = asyncio.create_task(asyncio.to_thread(self._request, payload))
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                # Retain correlation: the next caller must not consume this reply.
                while not task.done():
                    try:
                        await asyncio.shield(task)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
                if task.done() and not task.cancelled():
                    task.exception()
                raise

    def _request(self, payload):
        if not isinstance(payload, bytes) or len(payload) > MAX_WORKER_PD_PAYLOAD_BYTES:
            raise ValueError("invalid progress payload size")
        try:
            self.channel.send_bytes(payload)
            if not self.channel.poll(self.timeout):
                raise TimeoutError("transfer completion observation timed out")
            reply = decode_worker_payload(
                self.channel.recv_bytes(MAX_WORKER_PD_PAYLOAD_BYTES), _ProgressReply,
            )
        except BaseException:
            self._closed = True
            self.channel.close()
            raise
        if reply.error:
            raise RuntimeError(reply.error)
        return reply.payload

    async def close(self):
        async with self._lock:
            self._closed = True
            self.channel.close()


class TransferProgress:
    def __init__(self, call_worker, *, interval, limit, metrics):
        self.call_worker, self.interval, self.limit, self.metrics = call_worker, interval, limit, metrics
        self._pending = {}
        self._task = None

    async def wait(self, job_id, *, timeout):
        if job_id in self._pending or len(self._pending) >= self.limit:
            raise RuntimeError("duplicate or excessive active transfer completion wait")
        future = asyncio.get_running_loop().create_future()
        self._pending[job_id] = future
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._poll())
        try:
            return await asyncio.wait_for(asyncio.shield(future), timeout)
        finally:
            self._pending.pop(job_id, None)
            if not future.done():
                future.cancel()

    async def close(self):
        if self._task is not None:
            # Never abandon an in-flight completion query and its correlated reply.
            await self._task

    async def _poll(self):
        try:
            while self._pending:
                await asyncio.sleep(self.interval)
                jobs = tuple(self._pending)
                if not jobs:
                    break
                started = time.monotonic_ns()
                raw = await self.call_worker(OP_POLL_TRANSFER_CHUNK, encode_worker_payload(
                    tuple(TransferPollRequest(job) for job in jobs),
                ))
                self.metrics.observe_ns("transfer.poll_rpc", time.monotonic_ns() - started)
                self.metrics.increment("transfer.poll_commands")
                self.metrics.increment("transfer.poll_jobs", len(jobs))
                replies = decode_worker_payload(raw, tuple[TransferPollResponse, ...])
                if tuple(reply.job_id for reply in replies) != jobs:
                    raise RuntimeError("worker returned a different transfer completion batch")
                for reply in replies:
                    future = self._pending.get(reply.job_id)
                    if reply.complete and future is not None and not future.done():
                        future.set_result(reply)
        except BaseException as exc:
            for future in tuple(self._pending.values()):
                if not future.done():
                    future.set_exception(exc)
