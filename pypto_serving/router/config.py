# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Replica table, host pool, and router settings.

Two kinds of replica appear here. Entries under ``replicas`` already exist: the
router routes to them and never stops them. Entries under ``hosts`` describe
capacity the router may *launch* into -- one slot per declared device -- and a
replica launched from a slot is owned by the router and stopped with it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# A replica whose pinned load exceeds the least-loaded one by more than this many
# outstanding requests loses the pin. Affinity saves at most one prefill;
# queueing behind a saturated replica costs an unbounded wait.
DEFAULT_AFFINITY_SLACK = 8
DEFAULT_SESSION_TTL_SECONDS = 600.0
DEFAULT_HEALTH_INTERVAL_SECONDS = 5.0
# Generation is slow by design, so there is no useful read timeout; this bounds
# the whole exchange instead.
DEFAULT_REQUEST_TIMEOUT_SECONDS = 3600.0
DEFAULT_CONNECT_TIMEOUT_SECONDS = 5.0
# A model load is minutes, not seconds, and the spread is wide: Qwen3-14B
# reports ready in ~130s with a warm kernel cache, DeepSeek V4 on eight cards in
# ~370s as measured here. Older measurements of DeepSeek on a different
# checkpoint and an earlier stack ran to 1198s, so the figure moves with both --
# size this for the model and the stack the deployment actually runs, and raise
# it rather than have the router stop a replica that was still loading.
DEFAULT_LAUNCH_TIMEOUT_SECONDS = 600.0
# Aborting a half-finished generation to reclaim a device is worse than waiting.
DEFAULT_DRAIN_TIMEOUT_SECONDS = 300.0


@dataclass(frozen=True)
class ReplicaSpec:
    """One serving replica reachable over HTTP.

    The scheme is per replica so a deployment whose replicas are not on a
    trusted network can terminate TLS in front of them; prompts and generated
    text cross this hop in the clear otherwise.
    """

    name: str
    host: str
    port: int
    scheme: str = "http"

    @property
    def base_url(self) -> str:
        return f"{self.scheme}://{self.host}:{self.port}"


@dataclass(frozen=True)
class HostSpec:
    """A machine the router may launch replicas on.

    ``devices`` is declared, not discovered: plenty of deployments have no
    scheduler to ask, and a wrong answer that over-subscribes a card is worse
    than one the operator wrote down.

    ``devices_per_replica`` is how many of those cards one replica needs, which
    is a property of the model and its kernels rather than anything the router
    can measure -- 1 for Qwen3-14B, exactly 8 for DeepSeek V4, 16 for its DSpark
    variant, where the expert-parallel width is compiled in. Declared for the
    same reason: deriving it would mean reading a checkpoint on a machine the
    router has no business being able to see. A wrong value fails on the replica
    at startup, where the model topology is validated and the error names the
    number it wanted.

    Together they give the host's ceiling: ``len(devices) // devices_per_replica``.
    """

    name: str
    # None means this machine; anything else is an ssh destination.
    ssh: str | None = None
    # Path to a private key. Key material never appears in the config itself.
    identity_file: str | None = None
    devices: tuple[int, ...] = ()
    devices_per_replica: int = 1
    port_base: int = 8001
    model: str = ""
    served_model_name: str | None = None
    # Extra pypto-serving flags, passed through verbatim.
    serve_args: tuple[str, ...] = ()
    # Environment for the launched process. "{device}" expands per slot.
    env: dict[str, str] = field(default_factory=dict)
    # Optional prefix, e.g. ["task-submit", "--device", "{device}", "--run"].
    # Absent, the serving command runs bare -- deployments with no broker are
    # exactly why this is configuration rather than a hardcoded wrapper.
    launch_wrapper: tuple[str, ...] = ()
    # How to stop a replica; "{port}" and "{device}" expand.
    stop_command: tuple[str, ...] = ()
    python: str = "python3"
    workdir: str | None = None
    # Where a launched replica's stdout lands. A launch is detached, so this is
    # the only place a failed model load can be diagnosed from.
    log_dir: str = "/tmp/pypto-serving-router"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("host name must not be empty")
        if not self.devices:
            raise ValueError(f"host {self.name!r} declares no devices")
        if len(set(self.devices)) != len(self.devices):
            raise ValueError(f"host {self.name!r} repeats a device id")
        if not self.model:
            raise ValueError(f"host {self.name!r} must declare a model path")
        if self.devices_per_replica < 1:
            raise ValueError(f"host {self.name!r} devices_per_replica must be positive")
        # Refused here rather than leaving a partial group to fail on the device:
        # a leftover card is either a typo or a model the operator has misread.
        if len(self.devices) % self.devices_per_replica:
            raise ValueError(
                f"host {self.name!r} declares {len(self.devices)} devices, which is not a "
                f"multiple of devices_per_replica={self.devices_per_replica}"
            )
        for group in self.device_groups():
            port = self.port_for(group)
            if not 1 <= port <= 65535:
                raise ValueError(
                    f"host {self.name!r} device group {list(group)} maps to invalid port {port}"
                )

    @property
    def is_local(self) -> bool:
        return self.ssh is None

    @property
    def replica_capacity(self) -> int:
        """How many replicas fit on this host."""
        return len(self.devices) // self.devices_per_replica

    def device_groups(self) -> tuple[tuple[int, ...], ...]:
        """The declared devices cut into one group per replica, in order."""
        width = self.devices_per_replica
        return tuple(
            tuple(self.devices[start:start + width])
            for start in range(0, len(self.devices), width)
        )

    def port_for(self, group: tuple[int, ...]) -> int:
        """Port a replica on this device group listens on.

        Keyed on the group's first device, so it is derived rather than
        allocated: two launches on one host can never pick the same port, and a
        relaunch into the same group is predictable.
        """
        return self.port_base + group[0]

    def address(self) -> str:
        """Address the router reaches this host's replicas on."""
        if self.ssh is None:
            return "127.0.0.1"
        return self.ssh.rpartition("@")[2]


@dataclass(frozen=True)
class RouterConfig:
    """Everything the router needs to run."""

    replicas: tuple[ReplicaSpec, ...] = field(default_factory=tuple)
    hosts: tuple[HostSpec, ...] = field(default_factory=tuple)
    session_ttl_seconds: float = DEFAULT_SESSION_TTL_SECONDS
    affinity_slack: int = DEFAULT_AFFINITY_SLACK
    health_interval_seconds: float = DEFAULT_HEALTH_INTERVAL_SECONDS
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS
    connect_timeout_seconds: float = DEFAULT_CONNECT_TIMEOUT_SECONDS
    launch_timeout_seconds: float = DEFAULT_LAUNCH_TIMEOUT_SECONDS
    drain_timeout_seconds: float = DEFAULT_DRAIN_TIMEOUT_SECONDS
    # Zero by default: a config with only static replicas launches nothing.
    # The CLI raises this to 1 when the file declares launchable hosts.
    initial_replicas: int = 0
    max_replicas: int | None = None

    def __post_init__(self) -> None:
        if not self.replicas and not self.hosts:
            raise ValueError("router requires at least one replica or one host")
        names = [replica.name for replica in self.replicas]
        if len(set(names)) != len(names):
            raise ValueError(f"replica names must be unique: {sorted(names)}")
        host_names = [host.name for host in self.hosts]
        if len(set(host_names)) != len(host_names):
            raise ValueError(f"host names must be unique: {sorted(host_names)}")
        if self.session_ttl_seconds <= 0:
            raise ValueError("session_ttl_seconds must be positive")
        if self.affinity_slack < 0:
            raise ValueError("affinity_slack must be non-negative")
        if self.health_interval_seconds <= 0:
            raise ValueError("health_interval_seconds must be positive")
        if self.launch_timeout_seconds <= 0:
            raise ValueError("launch_timeout_seconds must be positive")
        if self.drain_timeout_seconds <= 0:
            raise ValueError("drain_timeout_seconds must be positive")
        if self.initial_replicas < 0:
            raise ValueError("initial_replicas must be non-negative")
        if self.max_replicas is not None and self.max_replicas < 1:
            raise ValueError("max_replicas must be positive")
        # Caught at startup rather than at the first launch, so a configuration
        # that can never satisfy itself fails the process that asked for it.
        if self.initial_replicas > self.pool_ceiling:
            capped = f", capped by max_replicas={self.max_replicas}" if self.max_replicas else ""
            raise ValueError(
                f"initial_replicas={self.initial_replicas} exceeds the pool ceiling "
                f"{self.pool_ceiling} (declared devices{capped})"
            )

    @property
    def pool_ceiling(self) -> int:
        """How many replicas the router may own at once.

        Cards, divided by the cards one replica of this model needs -- so eight
        cards serve eight Qwen replicas or exactly one DeepSeek V4.
        """
        declared = sum(host.replica_capacity for host in self.hosts)
        if self.max_replicas is None:
            return declared
        return min(declared, self.max_replicas)


def _parse_replica(entry: object, index: int) -> ReplicaSpec:
    if not isinstance(entry, dict) or not {"host", "port"} <= set(entry):
        raise ValueError(f"replica {index} needs 'host' and 'port'")
    unknown = sorted(set(entry) - {"name", "host", "port", "scheme"})
    if unknown:
        raise ValueError(f"replica {index} has unknown fields: {', '.join(unknown)}")
    host, port = entry["host"], entry["port"]
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError(f"replica {index} has an invalid port: {port!r}")
    scheme = entry.get("scheme", "http")
    if scheme not in ("http", "https"):
        raise ValueError(f"replica {index} has an invalid scheme: {scheme!r}")
    return ReplicaSpec(
        name=entry.get("name") or f"{host}:{port}", host=host, port=port, scheme=scheme,
    )


_HOST_FIELDS = frozenset({
    "name", "ssh", "identity_file", "devices", "devices_per_replica", "port_base",
    "model", "served_model_name", "serve_args", "env", "launch_wrapper",
    "stop_command", "python", "workdir", "log_dir",
})


def _parse_host(entry: object, index: int) -> HostSpec:
    if not isinstance(entry, dict):
        raise ValueError(f"host {index} must be a JSON object")
    unknown = sorted(set(entry) - _HOST_FIELDS)
    if unknown:
        raise ValueError(f"host {index} has unknown fields: {', '.join(unknown)}")

    devices = entry.get("devices")
    if not isinstance(devices, list) or not devices:
        raise ValueError(f"host {index} needs a non-empty 'devices' list")
    for device in devices:
        if isinstance(device, bool) or not isinstance(device, int) or device < 0:
            raise ValueError(f"host {index} has an invalid device id: {device!r}")

    env = entry.get("env", {})
    if not isinstance(env, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in env.items()
    ):
        raise ValueError(f"host {index} 'env' must map strings to strings")

    def string_list(key: str) -> tuple[str, ...]:
        value = entry.get(key, [])
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"host {index} {key!r} must be a list of strings")
        return tuple(value)

    port_base = entry.get("port_base", 8001)
    if isinstance(port_base, bool) or not isinstance(port_base, int):
        raise ValueError(f"host {index} 'port_base' must be an integer")

    per_replica = entry.get("devices_per_replica", 1)
    if isinstance(per_replica, bool) or not isinstance(per_replica, int) or per_replica < 1:
        raise ValueError(
            f"host {index} 'devices_per_replica' must be a positive integer"
        )

    return HostSpec(
        name=entry.get("name") or f"host{index}",
        ssh=entry.get("ssh"),
        identity_file=entry.get("identity_file"),
        devices=tuple(devices),
        devices_per_replica=per_replica,
        port_base=port_base,
        model=entry.get("model", ""),
        served_model_name=entry.get("served_model_name"),
        serve_args=string_list("serve_args"),
        env=dict(env),
        launch_wrapper=string_list("launch_wrapper"),
        stop_command=string_list("stop_command"),
        python=entry.get("python", "python3"),
        workdir=entry.get("workdir"),
        log_dir=entry.get("log_dir", "/tmp/pypto-serving-router"),
    )


def load_fleet_file(path: str | Path) -> tuple[tuple[ReplicaSpec, ...], tuple[HostSpec, ...]]:
    """Read ``replicas`` and ``hosts`` from a JSON file.

    Either key may be omitted, but the file must describe at least one of them:
    a router with neither has nothing to route to and no way to get it.
    """
    file_path = Path(path)
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read replica file {file_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"replica file {file_path} is not valid JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"replica file {file_path} must be a JSON object")

    unknown = sorted(set(data) - {"replicas", "hosts"})
    if unknown:
        raise ValueError(f"replica file has unknown top-level keys: {', '.join(unknown)}")

    raw_replicas = data.get("replicas", [])
    raw_hosts = data.get("hosts", [])
    if not isinstance(raw_replicas, list) or not isinstance(raw_hosts, list):
        raise ValueError("'replicas' and 'hosts' must be lists")
    if not raw_replicas and not raw_hosts:
        raise ValueError(
            f"replica file {file_path} declares neither a replica nor a launchable host"
        )

    replicas = tuple(_parse_replica(entry, i) for i, entry in enumerate(raw_replicas))
    hosts = tuple(_parse_host(entry, i) for i, entry in enumerate(raw_hosts))
    return replicas, hosts


def load_replica_table(path: str | Path) -> tuple[ReplicaSpec, ...]:
    """Read just the static replica table, for callers that never launch."""
    return load_fleet_file(path)[0]
