# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The replica table, session pinning, replica selection, and health polling."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from pypto_serving.router.config import ReplicaSpec, RouterConfig, load_replica_table
from pypto_serving.router.routing import (
    UNHEALTHY_THRESHOLD,
    HealthMonitor,
    NoReplicaAvailable,
    ReplicaRegistry,
    SessionDirectory,
    new_session_id,
)


class _Clock:
    """Manual monotonic clock, so expiry is testable without sleeping."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _config(count: int = 2, **kwargs) -> RouterConfig:
    replicas = tuple(ReplicaSpec(name=f"r{i}", host="h", port=8000 + i) for i in range(count))
    return RouterConfig(replicas=replicas, **kwargs)


def _registry(config: RouterConfig, clock=None):
    sessions = SessionDirectory(config.session_ttl_seconds, **({"clock": clock} if clock else {}))
    return ReplicaRegistry(config, sessions), sessions


# --- the replica table ---

def _write(tmp_path, payload) -> str:
    path = tmp_path / "replicas.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_replica_table_round_trips(tmp_path):
    path = _write(tmp_path, {"replicas": [{"name": "node0", "host": "127.0.0.1", "port": 8001}]})
    assert load_replica_table(path) == (ReplicaSpec(name="node0", host="127.0.0.1", port=8001),)


def test_replica_scheme_defaults_to_http_and_can_be_https(tmp_path):
    """Prompts cross this hop in the clear unless the deployment terminates TLS."""
    (plain,) = load_replica_table(_write(tmp_path, {"replicas": [{"host": "h", "port": 8001}]}))
    assert plain.base_url == "http://h:8001"

    (secure,) = load_replica_table(
        _write(tmp_path, {"replicas": [{"host": "h", "port": 8443, "scheme": "https"}]})
    )
    assert secure.base_url == "https://h:8443"

    with pytest.raises(ValueError, match="invalid scheme"):
        load_replica_table(_write(tmp_path, {"replicas": [{"host": "h", "port": 8001, "scheme": "ftp"}]}))


def test_replica_name_defaults_to_host_port(tmp_path):
    (replica,) = load_replica_table(_write(tmp_path, {"replicas": [{"host": "10.0.0.2", "port": 8001}]}))
    assert replica.name == "10.0.0.2:8001"
    assert replica.base_url == "http://10.0.0.2:8001"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"nodes": []}, "unknown top-level keys: nodes"),
        ({"replicas": []}, "neither a replica nor a launchable host"),
        ({"replicas": [], "hosts": []}, "neither a replica nor a launchable host"),
        ({"replicas": [{"host": "h"}]}, "needs 'host' and 'port'"),
        ({"replicas": [{"port": 8000}]}, "needs 'host' and 'port'"),
        ({"replicas": ["nope"]}, "needs 'host' and 'port'"),
        ({"replicas": [{"host": "h", "port": "8000"}]}, "invalid port"),
        ({"replicas": [{"host": "h", "port": True}]}, "invalid port"),
        ({"replicas": [{"host": "h", "port": 70000}]}, "invalid port"),
        ({"replicas": [{"host": "h", "port": 8000, "device": 4}]}, "unknown fields: device"),
    ],
)
def test_replica_table_rejects_malformed_entries(tmp_path, payload, message):
    with pytest.raises(ValueError, match=message):
        load_replica_table(_write(tmp_path, payload))


def test_replica_table_reports_bad_json_and_missing_file(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="is not valid JSON"):
        load_replica_table(bad)
    with pytest.raises(ValueError, match="cannot read replica file"):
        load_replica_table(tmp_path / "missing.json")


def test_router_config_validates_its_settings():
    spec = ReplicaSpec(name="a", host="h", port=8000)
    with pytest.raises(ValueError, match="at least one replica"):
        RouterConfig(replicas=())
    with pytest.raises(ValueError, match="names must be unique"):
        RouterConfig(replicas=(spec, ReplicaSpec(name="a", host="h2", port=8000)))
    with pytest.raises(ValueError, match="affinity_slack must be non-negative"):
        RouterConfig(replicas=(spec,), affinity_slack=-1)
    with pytest.raises(ValueError, match="session_ttl_seconds must be positive"):
        RouterConfig(replicas=(spec,), session_ttl_seconds=0)


# --- session pinning ---

def test_pin_lookup_and_forget():
    sessions = SessionDirectory(60.0, clock=_Clock())
    assert sessions.lookup("s1") is None
    sessions.pin("s1", "r0")
    assert sessions.lookup("s1") == "r0"
    assert len(sessions) == 1
    sessions.forget("s1")
    assert sessions.lookup("s1") is None


def test_lookup_expires_a_stale_pin():
    clock = _Clock()
    sessions = SessionDirectory(60.0, clock=clock)
    sessions.pin("s1", "r0")
    clock.advance(59.0)
    assert sessions.lookup("s1") == "r0"
    clock.advance(2.0)
    assert sessions.lookup("s1") is None
    assert len(sessions) == 0


def test_pin_refreshes_last_seen():
    clock = _Clock()
    sessions = SessionDirectory(60.0, clock=clock)
    sessions.pin("s1", "r0")
    clock.advance(50.0)
    sessions.pin("s1", "r0")
    clock.advance(50.0)
    assert sessions.lookup("s1") == "r0"


def test_sweep_drops_only_expired_pins():
    clock = _Clock()
    sessions = SessionDirectory(60.0, clock=clock)
    sessions.pin("old", "r0")
    clock.advance(61.0)
    sessions.pin("fresh", "r1")
    assert sessions.sweep() == 1
    assert sessions.lookup("fresh") == "r1"


def test_the_directory_evicts_least_recently_used_pins_when_full():
    """Session ids are client-supplied, so the map needs a cardinality bound."""
    sessions = SessionDirectory(60.0, clock=_Clock(), max_sessions=3)
    for i in range(3):
        sessions.pin(f"s{i}", "r0")
    sessions.pin("s0", "r0")          # refresh s0, making s1 the oldest
    sessions.pin("s3", "r0")          # overflows

    assert len(sessions) == 3
    assert sessions.lookup("s1") is None, "the least-recently pinned entry is evicted"
    assert sessions.lookup("s0") == "r0"
    assert sessions.lookup("s3") == "r0"


def test_expired_pins_are_reclaimed_before_anything_is_evicted():
    clock = _Clock()
    sessions = SessionDirectory(60.0, clock=clock, max_sessions=2)
    sessions.pin("old", "r0")
    clock.advance(61.0)
    sessions.pin("fresh", "r0")
    sessions.pin("newest", "r0")
    assert sessions.lookup("fresh") == "r0", "a live pin survives when an expired one can go"
    assert sessions.lookup("newest") == "r0"


def test_session_ids_are_unique_and_ttl_must_be_positive():
    assert new_session_id() != new_session_id()
    with pytest.raises(ValueError, match="ttl_seconds must be positive"):
        SessionDirectory(0)
    with pytest.raises(ValueError, match="max_sessions must be positive"):
        SessionDirectory(60.0, max_sessions=0)


# --- replica selection ---

def test_same_session_returns_to_the_same_replica():
    registry, _ = _registry(_config())
    first = registry.select("s1").replica.name
    for _ in range(5):
        decision = registry.select("s1")
        assert decision.replica.name == first
        assert decision.affinity_hit
    assert registry.affinity_hits == 5


def test_new_sessions_spread_across_replicas():
    registry, _ = _registry(_config(count=3))
    assert sorted(registry.select(f"s{i}").replica.name for i in range(3)) == ["r0", "r1", "r2"]


def test_expired_pin_re_routes_by_load():
    clock = _Clock()
    registry, sessions = _registry(_config(session_ttl_seconds=60.0), clock=clock)
    registry.select("s1")
    clock.advance(61.0)
    assert registry.select("s1").affinity_hit is False
    assert sessions.lookup("s1") is not None


def test_unroutable_pinned_replica_falls_back():
    registry, _ = _registry(_config())
    pinned = registry.select("s1").replica.name
    registry.set_ready(pinned, False)
    decision = registry.select("s1")
    assert decision.replica.name != pinned
    assert decision.affinity_hit is False
    # The session follows its new home.
    assert registry.select("s1").replica.name == decision.replica.name


def test_affinity_survives_moderate_load_but_yields_past_the_slack():
    registry, _ = _registry(_config(affinity_slack=8))
    pinned = registry.select("s1").replica.name
    for _ in range(8):
        registry.acquire(pinned)
    assert registry.select("s1").affinity_hit is True

    registry.acquire(pinned)
    decision = registry.select("s1")
    assert decision.affinity_hit is False
    assert decision.replica.name != pinned


def test_zero_slack_disables_affinity_under_any_load():
    registry, _ = _registry(_config(affinity_slack=0))
    pinned = registry.select("s1").replica.name
    registry.acquire(pinned)
    assert registry.select("s1").replica.name != pinned


def test_no_routable_replica_raises_and_counts():
    registry, _ = _registry(_config())
    for state in registry.states:
        registry.set_ready(state.name, False)
    with pytest.raises(NoReplicaAvailable):
        registry.select("s1")
    assert registry.rejected == 1


def test_counters_track_where_requests_went():
    registry, _ = _registry(_config())
    for i in range(4):
        registry.select(f"s{i}")
    assert registry.total_routed() == 4
    assert sum(state.routed for state in registry.states) == 4


def test_release_floors_at_zero_and_unknown_names_are_ignored():
    registry, _ = _registry(_config())
    registry.release("r0")
    assert registry.state("r0").outstanding == 0
    registry.acquire("ghost")
    registry.release("ghost")
    registry.set_ready("ghost", False)
    assert registry.ready_count() == 2


# --- health polling ---

class _ProbeClient:
    """Answers /health per replica URL from a scripted table."""

    def __init__(self, answers: dict) -> None:
        self.answers = answers
        self.probed: list[str] = []

    async def get(self, url, timeout=None):
        self.probed.append(url)
        answer = self.answers.get(url, 200)
        if isinstance(answer, Exception):
            raise answer
        return SimpleNamespace(status_code=answer)


def _monitor(count: int = 2, answers: dict | None = None):
    config = _config(count, health_interval_seconds=0.01)
    sessions = SessionDirectory(config.session_ttl_seconds)
    registry = ReplicaRegistry(config, sessions)
    client = _ProbeClient(answers or {})
    return HealthMonitor(config, registry, sessions, client), registry, client


def test_all_replicas_are_probed_each_cycle():
    monitor, _, client = _monitor()
    asyncio.run(monitor.probe_once())
    assert sorted(client.probed) == ["http://h:8000/health", "http://h:8001/health"]


@pytest.mark.parametrize("answer", [503, OSError("connection refused")])
def test_repeated_failures_take_a_replica_out_but_one_does_not(answer):
    monitor, registry, _ = _monitor(answers={"http://h:8000/health": answer})
    asyncio.run(monitor.probe_once())
    assert registry.state("r0").ready is True, "a single blip must not depin sessions"

    async def rest():
        for _ in range(UNHEALTHY_THRESHOLD - 1):
            await monitor.probe_once()

    asyncio.run(rest())
    assert registry.state("r0").ready is False
    assert registry.state("r1").ready is True, "other replicas are unaffected"


def test_one_success_restores_a_replica_immediately():
    answers = {"http://h:8000/health": 503}
    monitor, registry, _ = _monitor(answers=answers)

    async def check():
        for _ in range(UNHEALTHY_THRESHOLD):
            await monitor.probe_once()
        assert registry.state("r0").ready is False
        answers["http://h:8000/health"] = 200
        await monitor.probe_once()

    asyncio.run(check())
    assert registry.state("r0").ready is True
    assert registry.state("r0").failures == 0


def test_monitor_starts_and_stops_cleanly():
    monitor, _, client = _monitor()

    async def check():
        await monitor.start()
        await asyncio.sleep(0.05)  # several cycles at a 10ms interval
        await monitor.stop()
        return len(client.probed)

    assert asyncio.run(check()) > 0
    asyncio.run(monitor.stop())  # safe without a start
