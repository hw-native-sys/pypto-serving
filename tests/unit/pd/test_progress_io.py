# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Frame deadlines must bound blocked I/O as well as waiting for a reply."""

import asyncio
import multiprocessing
import socket
import struct
import threading
import time
from types import SimpleNamespace

import msgspec
import pytest

from pypto_serving.pd import worker
from pypto_serving.pd.transfer import TransferError
from pypto_serving.pd.worker import Progress, ProgressServer, _receive_frame, _send_frame, _shutdown
from tests.unit.test_pd_metadata import peer, plan, runtime as runtime


@pytest.mark.parametrize("cancel", [False, True])
def test_stalled_send_and_cancellation_drain_within_deadline(cancel):
    async def run():
        parent, child = socket.socketpair()
        parent.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
        child.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
        progress = Progress(parent, .1, 2)
        entered = threading.Event()
        exchange = progress._exchange
        def observed(*args):
            entered.set()
            return exchange(*args)
        progress._exchange = observed
        # A broken implementation must fail the test rather than hang pytest.
        guard = threading.Timer(2, _shutdown, args=(child,))
        guard.start()
        started = time.monotonic()
        try:
            task = asyncio.create_task(progress.call("submit", b"x" * (1024 * 1024)))
            assert await asyncio.to_thread(entered.wait, 1)
            if cancel:
                task.cancel()
            with pytest.raises(asyncio.CancelledError if cancel else TransferError):
                await task
            assert time.monotonic() - started < 1
            assert progress.error.uncertain and not progress.calls and not progress.lock.locked()
            with pytest.raises(TransferError) as error:
                await progress.call("source_state", msgspec.msgpack.encode("request"))
            assert error.value.uncertain
        finally:
            guard.cancel()
            _shutdown(child)
            child.close()
            await progress.close()
            guard.join(2)
    asyncio.run(run())


def test_partial_reply_after_submit_keeps_unknown_and_cannot_unregister(runtime):
    remote = peer(runtime)
    runtime.install(remote)
    parent, child = socket.socketpair()
    partial, release = threading.Event(), threading.Event()
    def respond():
        try:
            sequence, operation, payload = msgspec.msgpack.decode(_receive_frame(child, time.monotonic() + 2))
            result = runtime.handle_progress(operation, payload)
            reply = msgspec.msgpack.encode((sequence, result))
            child.sendall(struct.pack("!I", len(reply)) + reply[:1])
            partial.set()
            release.wait(2)
        finally:
            child.close()
    thread = threading.Thread(target=respond)
    thread.start()
    async def run():
        progress = Progress(parent, .1, 2)
        started = time.monotonic()
        try:
            with pytest.raises(TransferError) as error:
                await progress.call("submit", msgspec.msgpack.encode(plan(runtime, remote)))
            assert partial.is_set() and not release.is_set() and runtime.jobs
            assert error.value.uncertain and time.monotonic() - started < 1
            with pytest.raises(TransferError, match="unsafe or active"):
                runtime.handle("release", b"")
        finally:
            release.set()
            await progress.close()
    try:
        asyncio.run(run())
    finally:
        release.set()
        thread.join(2)
        parent.close()
    assert not thread.is_alive()


def test_fragmented_reads_share_one_absolute_deadline(monkeypatch):
    now, timeouts = [0.0], []
    packet = bytearray(struct.pack("!I", 4) + b"data")
    class Trickle:
        def settimeout(self, value):
            timeouts.append(value)
        def recv(self, count):
            now[0] += .02  # Every fragment is faster than the configured .05s.
            data = bytes(packet[:min(count, 2)])
            del packet[:len(data)]
            return data
    monkeypatch.setattr(worker, "time", SimpleNamespace(monotonic=lambda: now[0]))
    with pytest.raises(TimeoutError):
        _receive_frame(Trickle(), .05)
    assert timeouts == pytest.approx([.05, .03, .01])
    assert packet  # Total frame time expired instead of resetting on every fragment.


def test_server_close_interrupts_partial_request(monkeypatch):
    entered, handled = threading.Event(), []
    receive = worker._receive_frame
    def observed(*args):
        entered.set()
        return receive(*args)
    monkeypatch.setattr(worker, "_receive_frame", observed)
    parent, child = socket.socketpair()
    server = ProgressServer(child, lambda *args: handled.append(args), timeout=10)
    try:
        parent.sendall(struct.pack("!I", 1024) + b"partial")
        assert entered.wait(1)
        started = time.monotonic()
        server.close()
        assert time.monotonic() - started < 1
        assert not server.thread.is_alive() and not handled
    finally:
        parent.close()
        server.close()


@pytest.mark.parametrize("size", [0, 16 * 1024 * 1024 + 1])
def test_invalid_frame_closes_channel_before_handler(size):
    parent, child = socket.socketpair()
    handled = []
    server = ProgressServer(child, lambda *args: handled.append(args))
    try:
        parent.settimeout(1)
        parent.sendall(struct.pack("!I", size))
        assert parent.recv(1) == b"" and not handled
    finally:
        parent.close()
        server.close()


def _echo_in_spawned_process(channel):
    try:
        sequence, _, payload = msgspec.msgpack.decode(_receive_frame(channel, time.monotonic() + 5))
        _send_frame(channel, msgspec.msgpack.encode((sequence, payload)), time.monotonic() + 5)
    finally:
        channel.close()


def test_socket_channel_survives_worker_spawn_and_large_frame():
    parent, child = socket.socketpair()
    process = multiprocessing.get_context("spawn").Process(target=_echo_in_spawned_process, args=(child,))
    process.start()
    child.close()
    async def run():
        progress = Progress(parent, 5, 2)
        try:
            payload = msgspec.msgpack.encode(b"x" * (1024 * 1024))
            assert await progress.call("source_state", payload) == payload
        finally:
            await progress.close()
    try:
        asyncio.run(run())
        process.join(5)
        assert process.exitcode == 0
    finally:
        parent.close()
        if process.is_alive():
            process.terminate()
        process.join(5)
