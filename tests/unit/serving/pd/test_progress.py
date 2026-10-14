# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import asyncio
import multiprocessing
import threading

import pytest

from pypto_serving.serving.pd.observability import PDMetrics
from pypto_serving.serving.pd.progress import ProgressClient, ProgressServer, TransferProgress
from pypto_serving.serving.pd.worker_api import (
    TransferPollRequest, TransferPollResponse, decode_worker_payload, encode_worker_payload,
)


def _spawn_progress(channel, ready, finished):
    server = ProgressServer(channel, lambda job: TransferPollResponse(job, True), limit=4)
    ready.set()
    try:
        if not finished.wait(10):
            raise RuntimeError("test progress client did not finish")
    finally:
        server.close()


def test_progress_pipe_crosses_a_spawn_boundary_without_sharing_runtime_state():
    async def run():
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        ready, finished = context.Event(), context.Event()
        process = context.Process(target=_spawn_progress, args=(child, ready, finished))
        process.start()
        child.close()
        client = ProgressClient(parent, timeout=2)
        try:
            assert await asyncio.to_thread(ready.wait, 5)
            raw = await client.call("poll_transfer_chunk", encode_worker_payload((TransferPollRequest("spawn"),)))
            assert decode_worker_payload(raw, tuple[TransferPollResponse, ...])[0].job_id == "spawn"
        finally:
            await client.close()
            finished.set()
            await asyncio.to_thread(process.join, 5)
        assert process.exitcode == 0
    asyncio.run(run())


def test_active_jobs_share_a_poll_and_cancellation_does_not_abandon_worker_reply():
    async def run():
        commands = []
        entered, complete = asyncio.Event(), asyncio.Event()
        async def worker(operation, payload):
            jobs = decode_worker_payload(payload, tuple[TransferPollRequest, ...])
            commands.append(tuple(job.job_id for job in jobs))
            entered.set()
            await complete.wait()
            return encode_worker_payload(tuple(TransferPollResponse(job.job_id, True) for job in jobs))
        progress = TransferProgress(worker, interval=.001, limit=4, metrics=PDMetrics("test", "worker"))
        waits = [asyncio.create_task(progress.wait(str(i), timeout=1)) for i in range(4)]
        await entered.wait()
        waits[0].cancel()
        complete.set()
        results = await asyncio.gather(*waits, return_exceptions=True)
        assert isinstance(results[0], asyncio.CancelledError)
        assert all(result.complete for result in results[1:])
        await progress.close()
        assert commands == [("0", "1", "2", "3")]
        assert not progress._pending
    asyncio.run(run())


def test_local_progress_channel_observes_facts_without_running_ordered_work():
    async def run():
        parent, child = multiprocessing.Pipe()
        server = ProgressServer(child, lambda job: TransferPollResponse(job, True), limit=4)
        client = ProgressClient(parent, timeout=1)
        try:
            raw = await client.call("poll_transfer_chunk", encode_worker_payload((TransferPollRequest("a"),)))
            assert decode_worker_payload(raw, tuple[TransferPollResponse, ...]) == (TransferPollResponse("a", True),)
            with pytest.raises(ValueError, match="queries only"):
                await client.call("submit_transfer_chunk", b"")
            for jobs in ((), ("a", "a"), tuple(str(i) for i in range(5))):
                with pytest.raises(RuntimeError, match="invalid"):
                    await client.call("poll_transfer_chunk", encode_worker_payload(
                        tuple(TransferPollRequest(job) for job in jobs),
                    ))
            # An invalid read cannot poison or mutate native work.
            raw = await client.call("poll_transfer_chunk", encode_worker_payload((TransferPollRequest("b"),)))
            assert decode_worker_payload(raw, tuple[TransferPollResponse, ...])[0].job_id == "b"
        finally:
            await client.close()
            server.close()
        assert not server._thread.is_alive()
    asyncio.run(run())


def test_progress_cancellation_drains_the_response_before_another_reader():
    async def run():
        parent, child = multiprocessing.Pipe()
        entered, release = threading.Event(), threading.Event()
        def poll(job):
            entered.set()
            assert release.wait(1)
            return TransferPollResponse(job, True)
        server = ProgressServer(child, poll, limit=4)
        client = ProgressClient(parent, timeout=2)
        try:
            first = asyncio.create_task(client.call("poll_transfer_chunk", encode_worker_payload(
                (TransferPollRequest("first"),),
            )))
            assert await asyncio.to_thread(entered.wait, 1)
            first.cancel()
            second = asyncio.create_task(client.call("poll_transfer_chunk", encode_worker_payload(
                (TransferPollRequest("second"),),
            )))
            await asyncio.sleep(.01)
            assert not first.done() and not second.done()
            first.cancel()
            await asyncio.sleep(.01)
            assert not first.done() and not second.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await first
            raw = await second
            assert decode_worker_payload(raw, tuple[TransferPollResponse, ...])[0].job_id == "second"
        finally:
            release.set()
            await client.close()
            server.close()
    asyncio.run(run())


def test_progress_eof_is_failure_not_a_completed_or_cancelled_transfer():
    async def run():
        parent, child = multiprocessing.Pipe()
        client = ProgressClient(parent, timeout=.05)
        child.close()
        try:
            with pytest.raises((EOFError, OSError)):
                await client.call("poll_transfer_chunk", encode_worker_payload((TransferPollRequest("live"),)))
            with pytest.raises(RuntimeError, match="closed"):
                await client.call("poll_transfer_chunk", b"")
        finally:
            await client.close()
    asyncio.run(run())


def test_bad_completion_batch_fails_all_waiters_without_claiming_native_cancellation():
    async def run():
        async def worker(operation, payload):
            return encode_worker_payload((TransferPollResponse("wrong", True),))
        progress = TransferProgress(worker, interval=.001, limit=4, metrics=PDMetrics("test", "worker"))
        result = await asyncio.gather(
            progress.wait("a", timeout=1), progress.wait("b", timeout=1), return_exceptions=True,
        )
        assert all(isinstance(error, RuntimeError) for error in result)
        await progress.close()
        assert not progress._pending
    asyncio.run(run())
