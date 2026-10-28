# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise HTTP disconnects through native and AgentInfer request cleanup."""

import asyncio
import contextlib
import json
import socket
import threading
import time
from types import SimpleNamespace

import pytest
import uvicorn

from pypto_serving.config.types import GenerateConfig
from pypto_serving.serving.engine.async_engine import EngineConfig, ReplicaEngineCore, TokenOutput
from pypto_serving.serving.memory.kv_cache import KvCacheManager
from pypto_serving.serving.sched.scheduler import Scheduler, SchedulerConfig
from pypto_serving.serving.server.server import CompletionRequest, ServingServer


class BlockingEngine:
    tokenizer = SimpleNamespace(
        eos_token_id=0,
        apply_chat_template=lambda messages, **kwargs: " ".join(m["content"] for m in messages),
    )

    def __init__(self, state):
        self.state = state
        self.started = threading.Event()
        self.closed = threading.Event()
        self.retained = set()
        self.cancelled = []
        self.core = ReplicaEngineCore.__new__(ReplicaEngineCore)
        self.core.config = EngineConfig()
        self.core.tokenizer = self.tokenizer
        self.core.scheduler = Scheduler(SchedulerConfig(), KvCacheManager(num_blocks=16, block_size=2))
        self.core._request_contexts = {}
        self.core._pending_free_ids = []
        if state == "retained":
            def arrive(request, session):
                self.retained.add(request.request_id)
                return False

            def cancel(request_id):
                self.cancelled.append(request_id)
                self.retained.remove(request_id)
                return True

            self.core._agentinfer_admission = SimpleNamespace(arrive=arrive, cancel=cancel)

    async def add_request(self, request_id, prompt, config, **kwargs):
        if prompt == "healthy":
            yield TokenOutput(text="OK", finished=True, finish_reason="FINISHED_EOS",
                              prompt_tokens=1, completion_tokens=1)
            return

        def on_queued():
            if self.state == "running":
                self.core.scheduler.schedule()
            self.started.set()

        outputs = self.core.add_request(request_id, prompt, config, on_queued=on_queued,
                                        prompt_token_ids=[1, 2, 3, 4],
                                        agent_session_id=kwargs.get("agent_session_id", "session"))
        try:
            async with contextlib.aclosing(outputs):
                async for output in outputs:
                    yield output
        finally:
            self.closed.set()


@pytest.mark.parametrize("state", ["waiting", "running", "retained"])
@pytest.mark.parametrize("chat", [False, True])
def test_tcp_disconnect_releases_core_and_next_request_works(state, chat):
    engine = BlockingEngine(state)
    serving = ServingServer(engine, "test", GenerateConfig())
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(16)
    address = listener.getsockname()
    server = uvicorn.Server(uvicorn.Config(serving.app, log_level="error", lifespan="off"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while not server.started:
            assert thread.is_alive() and time.monotonic() < deadline
            time.sleep(0.01)
        path = "/v1/chat/completions" if chat else "/v1/completions"
        body = {"messages": [{"role": "user", "content": "blocked"}],
                "agent_hint": {"session_id": "session"}} if chat else {"prompt": "blocked"}
        data = json.dumps(body).encode()
        client = socket.create_connection(address, timeout=2)
        client.sendall((f"POST {path} HTTP/1.1\r\nHost: localhost\r\n"
                        f"Content-Type: application/json\r\nContent-Length: {len(data)}\r\n\r\n").encode() + data)
        assert engine.started.wait(2)
        client.close()
        assert engine.closed.wait(2), "Disconnected client left its core request alive"
        assert not engine.core._request_contexts
        assert not engine.retained
        assert engine.core.scheduler.kv_cache_manager.num_free_blocks == 16
        if state == "retained":
            assert len(engine.cancelled) == 1
            assert engine.core._pending_free_ids == []
        else:
            assert len(engine.core._pending_free_ids) == 1
        import httpx
        payload = {"messages": [{"role": "user", "content": "healthy"}]} if chat else {"prompt": "healthy"}
        with httpx.Client(trust_env=False) as client:
            response = client.post(f"http://{address[0]}:{address[1]}{path}", json=payload, timeout=2)
        assert response.status_code == 200
        assert response.json()["usage"]["completion_tokens"] == 1
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        listener.close()
        assert not thread.is_alive()


def test_cancelling_handler_closes_pending_engine_generator():
    async def run():
        engine = BlockingEngine("running")
        serving = ServingServer(engine, "test", GenerateConfig())
        http_request = SimpleNamespace(receive=lambda: asyncio.Future())
        task = asyncio.create_task(serving._completions(CompletionRequest(prompt="blocked"), http_request))
        deadline = asyncio.get_running_loop().time() + 2
        while not engine.started.is_set():
            if task.done():
                await task
            assert asyncio.get_running_loop().time() < deadline
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert engine.closed.is_set()
        assert not engine.core._request_contexts
        assert len(engine.core._pending_free_ids) == 1
    asyncio.run(run())


def test_normal_completion_cancels_disconnect_watcher():
    async def run():
        engine = BlockingEngine("waiting")
        serving = ServingServer(engine, "test", GenerateConfig())
        http_request = SimpleNamespace(receive=lambda: asyncio.Future())
        for _ in range(20):
            response = await asyncio.wait_for(
                serving._completions(CompletionRequest(prompt="healthy"), http_request), timeout=0.2,
            )
            assert response.status_code == 200
        assert not [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
    asyncio.run(run())
