# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A replica that dies, rather than one that merely says it is not ready.

The existing end-to-end tests take a replica out of rotation with ``/toggle``,
which leaves the process up and the socket listening. Nothing covered a replica
whose process is gone -- and that is the case a deployment actually meets, with
a different signature at every layer: the connection is refused rather than
answered, a stream already in flight is cut rather than finished, and the
session that was pinned there has nowhere to go back to.

Each cluster here is built per test, because a test that kills a replica cannot
hand it back to the next one.
"""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.error
import urllib.request

import pytest

from tests.unit.router.test_end_to_end import (  # noqa: F401 - the importorskip runs on import
    CHAT,
    OPENER,
    REPO_ROOT,
    STUB,
    _await,
    _await_ready,
    _free_port,
    _get,
    _post,
    _served_by,
)


class _Cluster:
    """Two stub replicas behind a router, all real processes, all disposable.

    ``health_interval`` is the dial these tests turn: fast when the point is
    that the poller notices a death, and slow when the point is what happens
    to a request that reaches a replica before anything has noticed.
    """

    def __init__(self, tmp_path, *, health_interval: float) -> None:
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
        self.procs: dict[str, subprocess.Popen] = {
            name: subprocess.Popen(
                [sys.executable, str(STUB), "--port", str(port), "--name", name],
                cwd=str(REPO_ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            for name, port in self.ports.items()
        }
        self.router = subprocess.Popen(
            [
                sys.executable, "-m", "pypto_serving.router.app",
                "--replicas", str(table),
                "--port", str(self.router_port),
                "--health-interval", str(health_interval),
            ],
            cwd=str(REPO_ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        for port in self.ports.values():
            _await_ready(f"http://127.0.0.1:{port}/health")
        _await_ready(self.url("/health"))
        assert _await(lambda: self.health()["replicas_ready"] == 2), "the router never saw both"

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.router_port}{path}"

    def health(self) -> dict:
        return _get(self.url("/health"))[1]

    def kill(self, name: str) -> None:
        """Kill a replica outright. No shutdown hook, no socket close."""
        proc = self.procs.pop(name)
        proc.kill()
        proc.wait(timeout=10)

    def ask(self, session: str | None = None, timeout: float = 30.0):
        request = urllib.request.Request(
            self.url("/v1/chat/completions"),
            data=json.dumps(CHAT).encode(),
            headers={"Content-Type": "application/json"},
        )
        if session is not None:
            request.add_header("X-Session-Id", session)
        try:
            with OPENER.open(request, timeout=timeout) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()

    def close(self) -> None:
        for proc in [*self.procs.values(), self.router]:
            proc.terminate()
        for proc in [*self.procs.values(), self.router]:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


@pytest.fixture
def cluster(tmp_path):
    """A cluster whose poller is fast enough to notice a death inside a test."""
    started = _Cluster(tmp_path, health_interval=0.5)
    try:
        yield started
    finally:
        started.close()


@pytest.fixture
def unwatched_cluster(tmp_path):
    """A cluster whose poller will not notice anything during the test.

    What is left is the proxy's own view of a dead replica, which is the only
    thing standing between a client and a connection error.
    """
    started = _Cluster(tmp_path, health_interval=600.0)
    try:
        yield started
    finally:
        started.close()


def test_a_dead_replica_leaves_rotation_and_the_fleet_keeps_serving(cluster):
    """The fleet is only useful if losing one replica is not losing the service."""
    cluster.kill("alpha")

    assert _await(lambda: cluster.health()["replicas_ready"] == 1), cluster.health()
    for _ in range(4):
        status, body = cluster.ask()
        assert status == 200
        assert _served_by(body) == "beta"


def test_a_request_that_reaches_a_dead_replica_is_refused_not_left_hanging(unwatched_cluster):
    """The poller is not the only thing that has to cope with a dead replica.

    A request dispatched before anything noticed cannot be rescued, but it must
    come back as an error rather than hold the caller for as long as a
    connection takes to fail. The proxy also demotes the replica there and then,
    without waiting for two failed probes.
    """
    cluster = unwatched_cluster
    status, body = cluster.ask(session="pinned")
    assert status == 200
    home = _served_by(body)
    assert home in cluster.ports, "the session has to be pinned for this to be a test"

    cluster.kill(home)

    status, body = cluster.ask(session="pinned", timeout=20.0)
    assert status == 502, body[:200]
    assert b"unreachable" in body

    # Demoted by that one failure, so the next turn is served rather than
    # dispatched into the same hole.
    status, body = cluster.ask(session="pinned", timeout=20.0)
    assert status == 200
    assert _served_by(body) != home


def test_a_session_pinned_to_a_dead_replica_moves_on_its_next_turn(cluster):
    """Affinity is a preference. A pin to something gone must not be honoured."""
    status, body = cluster.ask(session="stays-put")
    assert status == 200
    home = _served_by(body)

    cluster.kill(home)
    assert _await(lambda: cluster.health()["replicas_ready"] == 1), cluster.health()

    status, body = cluster.ask(session="stays-put")
    assert status == 200
    assert _served_by(body) != home


def test_a_stream_whose_replica_dies_mid_flight_ends_instead_of_hanging(unwatched_cluster):
    """The one failure the relay cannot answer with a status code.

    Headers are out and a chunk has already been delivered, so there is no 502
    left to send. What matters is that the client is released rather than left
    reading a stream nobody will ever finish.
    """
    cluster = unwatched_cluster
    payload = {**CHAT, "stream": True, "stream_crash_after": 1}
    request = urllib.request.Request(
        cluster.url("/v1/chat/completions"),
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with OPENER.open(request, timeout=30.0) as response:
            assert response.status == 200, "the first chunk went out before the replica died"
            received = b"".join(response)
    except Exception as exc:  # noqa: BLE001 - a reset is one of the two honest outcomes
        received = getattr(exc, "partial", b"")

    assert b"hello" in received, "the replica died before the stream started; nothing mid-flight"
    assert b"data: [DONE]" not in received, "a truncated stream must not look complete"


def test_losing_every_replica_is_a_503_rather_than_a_hang(cluster):
    """With nothing left to route to, say so. A caller can retry a 503."""
    cluster.kill("alpha")
    cluster.kill("beta")

    assert _await(lambda: cluster.health()["replicas_ready"] == 0), cluster.health()
    assert _get(cluster.url("/health"))[0] == 503

    status, _body = cluster.ask(timeout=20.0)
    assert status == 503


def test_a_dead_replica_is_kept_but_reported_as_failed(cluster):
    """These replicas are listed in the config, so the router never stops them.

    It keeps the entry and keeps counting it, and capacity quietly drops by
    one. What it must not do is leave that indistinguishable from a replica
    still loading its model -- both are unroutable, and only one of them is
    coming back on its own.
    """
    cluster.kill("alpha")
    assert _await(lambda: cluster.health()["replicas_ready"] == 1), cluster.health()

    health = cluster.health()
    assert health["replicas_total"] == 2, "the dead replica is still in the table"
    assert health["replicas_failed"] == 1
    dead = [r for r in health["replicas"] if r["name"] == "alpha"]
    assert dead and dead[0]["ready"] is False
    assert dead[0]["phase"] == "failed"
    alive = [r for r in health["replicas"] if r["name"] == "beta"]
    assert alive and alive[0]["phase"] == "ready"
