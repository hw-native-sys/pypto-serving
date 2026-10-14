# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""HTTP thread/socket ownership under local cancellation, with no device work."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from pypto_serving.router.client import GenerationResponse, NodeClient


class _Response:
    status = 200
    headers = {"Content-Type": "text/event-stream"}
    def __init__(self):
        self.closed = False
        self.read_started = threading.Event()
        self.release_read = threading.Event()
        self.fp = None
    def read(self, limit):
        self.read_started.set()
        assert self.release_read.wait(2)
        return b"{}"
    def close(self):
        self.closed = True
        self.release_read.set()


def test_cancellation_during_open_drains_late_response_and_closes_socket():
    async def exercise():
        started, release = threading.Event(), threading.Event()
        response = _Response()
        client = NodeClient("http://localhost", 1)
        def open_(*args):
            started.set()
            assert release.wait(2)
            return response
        client._open = open_
        task = asyncio.create_task(client.open_generation("/v1/completions", b"{}"))
        assert await asyncio.to_thread(started.wait, 1)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()  # Worker still owns a possible future socket.
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert response.closed
    asyncio.run(exercise())


def test_cancellation_during_read_interrupts_and_drains_worker_off_loop():
    async def exercise():
        response = _Response()
        owned = GenerationResponse(response)
        task = asyncio.create_task(owned.json())
        assert await asyncio.to_thread(response.read_started.wait, 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert response.closed and owned._closed
        await owned.aclose()
        assert owned.headers["content-type"] == "text/event-stream"
    asyncio.run(exercise())


def test_stream_frames_are_forwarded_byte_for_byte_and_eof_is_not_done():
    async def exercise():
        lines = iter((b'data: {"id":"request","choices":[]}\r\n', b"\r\n", b""))
        response = SimpleNamespace(status=200, headers={}, readline=lambda limit: next(lines), close=lambda: None)
        owned = GenerationResponse(response)
        frames = owned.frames()
        assert await anext(frames) == b'data: {"id":"request","choices":[]}\r\n\r\n'
        with pytest.raises(EOFError):
            await anext(frames)
        await owned.aclose()
    asyncio.run(exercise())
