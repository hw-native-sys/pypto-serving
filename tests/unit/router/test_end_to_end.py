# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The router over real HTTP, against two stand-in replicas.

The other router tests inject a fake client; this one runs the real stack --
uvicorn, httpx, chunked SSE relay, health polling -- because the failure modes
that matter here (a stream that is not relayed byte-for-byte, a replica that is
never taken out of rotation) only appear on a socket. No NPU is involved.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

httpx = pytest.importorskip("httpx", reason="the router needs httpx to reach a replica")
pytest.importorskip("uvicorn", reason="the stub replicas need uvicorn")

REPO_ROOT = Path(__file__).resolve().parents[3]
STUB = Path(__file__).with_name("stub_replica.py")
# Proxy settings in the environment must not intercept loopback traffic.
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
CHAT = {"model": "m", "messages": [{"role": "user", "content": "Hello"}]}


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _get(url: str):
    """GET returning (status, payload), treating 503 as an answer rather than an error."""
    try:
        with OPENER.open(url, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _post(url: str, payload: dict, session: str | None = None):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    if session is not None:
        request.add_header("X-Session-Id", session)
    with OPENER.open(request, timeout=30) as response:
        headers = {key.lower(): value for key, value in response.headers.items()}
        return response.status, headers, response.read()


def _await_ready(url: str, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if _get(url)[0] == 200:
                return
        except Exception:
            pass
        time.sleep(0.2)
    raise AssertionError(f"{url} never became ready within {timeout:g}s")


def _await(predicate, timeout: float = 20.0) -> bool:
    """Poll until the predicate holds; a predicate that raises counts as not yet."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if predicate():
                return True
        except Exception:
            pass
        time.sleep(0.2)
    return False


def _served_by(body: bytes) -> str:
    text = body.decode()
    for name in ("alpha", "beta"):
        if f'"served_by": "{name}"' in text or f"[{name}]" in text:
            return name
    return "?"


class _Cluster:
    """Two stub replicas behind a router, all as real processes."""

    def __init__(self, tmp_path: Path) -> None:
        self.ports = {"alpha": _free_port(), "beta": _free_port()}
        self.router_port = _free_port()
        table = tmp_path / "replicas.json"
        table.write_text(
            json.dumps({
                "replicas": [
                    {"name": name, "host": "127.0.0.1", "port": port}
                    for name, port in self.ports.items()
                ]
            }),
            encoding="utf-8",
        )
        env_path = str(REPO_ROOT)
        self.procs = [
            subprocess.Popen(
                [sys.executable, str(STUB), "--port", str(port), "--name", name],
                cwd=env_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            for name, port in self.ports.items()
        ]
        self.procs.append(
            subprocess.Popen(
                [
                    sys.executable, "-m", "pypto_serving.router.app",
                    "--replicas", str(table),
                    "--port", str(self.router_port),
                    # Fast enough that failover is observable inside a test.
                    "--health-interval", "0.5",
                ],
                cwd=env_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        )
        for port in self.ports.values():
            _await_ready(f"http://127.0.0.1:{port}/health")
        _await_ready(self.url("/health"))
        assert _await(lambda: self.health()["replicas_ready"] == 2), "router never saw both replicas"

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.router_port}{path}"

    def replica_url(self, name: str, path: str) -> str:
        return f"http://127.0.0.1:{self.ports[name]}{path}"

    def health(self) -> dict:
        return _get(self.url("/health"))[1]

    def toggle(self, name: str) -> None:
        _post(self.replica_url(name, "/toggle"), {})

    def close(self) -> None:
        for proc in self.procs:
            proc.terminate()
        for proc in self.procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


@pytest.fixture(scope="module")
def cluster(tmp_path_factory):
    started = _Cluster(tmp_path_factory.mktemp("router-e2e"))
    try:
        yield started
    finally:
        started.close()


def test_a_new_conversation_is_minted_a_session_id(cluster):
    status, headers, body = _post(cluster.url("/v1/chat/completions"), CHAT)
    assert status == 200
    assert len(headers["x-session-id"]) == 32
    assert _served_by(body) in {"alpha", "beta"}


def test_a_conversation_stays_on_its_replica(cluster):
    _, headers, body = _post(cluster.url("/v1/chat/completions"), CHAT)
    session, home = headers["x-session-id"], _served_by(body)

    landed = []
    for _ in range(6):
        _, headers, body = _post(cluster.url("/v1/chat/completions"), CHAT, session=session)
        assert headers["x-session-id"] == session
        landed.append(_served_by(body))
    assert set(landed) == {home}, landed


def test_streaming_relays_the_sse_stream_and_the_session(cluster):
    _, headers, body = _post(cluster.url("/v1/chat/completions"), CHAT)
    session, home = headers["x-session-id"], _served_by(body)

    status, headers, body = _post(
        cluster.url("/v1/chat/completions"), dict(CHAT, stream=True), session=session
    )
    assert status == 200
    assert headers["x-session-id"] == session
    assert "text/event-stream" in headers["content-type"]
    assert _served_by(body) == home

    events = [line for line in body.decode().splitlines() if line.startswith("data:")]
    assert len(events) == 3
    assert events[-1] == "data: [DONE]"
    assert json.loads(events[0][len("data:"):])["choices"][0]["delta"]["content"] == "hello"


def test_a_body_session_id_needs_no_header(cluster):
    _, headers, body = _post(cluster.url("/v1/chat/completions"), CHAT)
    session, home = headers["x-session-id"], _served_by(body)

    _, headers, body = _post(cluster.url("/v1/chat/completions"), dict(CHAT, session_id=session))
    assert headers["x-session-id"] == session
    assert _served_by(body) == home


def test_distinct_conversations_reach_both_replicas(cluster):
    seen = {_served_by(_post(cluster.url("/v1/chat/completions"), CHAT)[2]) for _ in range(8)}
    assert seen == {"alpha", "beta"}


def test_models_are_listed_through_the_router(cluster):
    status, payload = _get(cluster.url("/v1/models"))
    assert status == 200
    assert payload["object"] == "list"
    assert payload["data"]


def test_an_unroutable_replica_loses_its_sessions(cluster):
    _, headers, body = _post(cluster.url("/v1/chat/completions"), CHAT)
    session, home = headers["x-session-id"], _served_by(body)

    cluster.toggle(home)
    try:
        assert _await(lambda: cluster.health()["replicas_ready"] == 1), "replica stayed routable"
        _, _, body = _post(cluster.url("/v1/chat/completions"), CHAT, session=session)
        assert _served_by(body) != home
    finally:
        cluster.toggle(home)
    assert _await(lambda: cluster.health()["replicas_ready"] == 2), "replica never recovered"


def test_with_no_routable_replica_requests_are_rejected_not_hung(cluster):
    for name in cluster.ports:
        cluster.toggle(name)
    try:
        assert _await(
            lambda: _get(cluster.url("/health"))[0] == 503
        ), "router kept reporting healthy"
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            _post(cluster.url("/v1/chat/completions"), CHAT)
        assert excinfo.value.code == 503
    finally:
        for name in cluster.ports:
            cluster.toggle(name)
    assert _await(lambda: cluster.health()["replicas_ready"] == 2)
