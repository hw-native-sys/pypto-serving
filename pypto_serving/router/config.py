# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Replica table and router settings.

The replica table is the flat routing table: one entry per serving replica,
wherever it runs. A replica is exactly one serving process owning one engine
core, so co-located and remote replicas are addressed identically.
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


@dataclass(frozen=True)
class ReplicaSpec:
    """One serving replica reachable over HTTP."""

    name: str
    host: str
    port: int

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass(frozen=True)
class RouterConfig:
    """Everything the router needs to run."""

    replicas: tuple[ReplicaSpec, ...] = field(default_factory=tuple)
    session_ttl_seconds: float = DEFAULT_SESSION_TTL_SECONDS
    affinity_slack: int = DEFAULT_AFFINITY_SLACK
    health_interval_seconds: float = DEFAULT_HEALTH_INTERVAL_SECONDS
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS
    connect_timeout_seconds: float = DEFAULT_CONNECT_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if not self.replicas:
            raise ValueError("router requires at least one replica")
        names = [replica.name for replica in self.replicas]
        if len(set(names)) != len(names):
            raise ValueError(f"replica names must be unique: {sorted(names)}")
        if self.session_ttl_seconds <= 0:
            raise ValueError("session_ttl_seconds must be positive")
        if self.affinity_slack < 0:
            raise ValueError("affinity_slack must be non-negative")
        if self.health_interval_seconds <= 0:
            raise ValueError("health_interval_seconds must be positive")


def load_replica_table(path: str | Path) -> tuple[ReplicaSpec, ...]:
    """Read the replica table from a JSON file: ``{"replicas": [{host, port, name?}]}``."""
    file_path = Path(path)
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read replica file {file_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"replica file {file_path} is not valid JSON: {exc.msg}") from exc

    entries = data.get("replicas") if isinstance(data, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"replica file {file_path} needs a non-empty 'replicas' list")

    replicas = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not {"host", "port"} <= set(entry):
            raise ValueError(f"replica {index} needs 'host' and 'port'")
        unknown = sorted(set(entry) - {"name", "host", "port"})
        if unknown:
            raise ValueError(f"replica {index} has unknown fields: {', '.join(unknown)}")
        host, port = entry["host"], entry["port"]
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError(f"replica {index} has an invalid port: {port!r}")
        replicas.append(ReplicaSpec(name=entry.get("name") or f"{host}:{port}", host=host, port=port))
    return tuple(replicas)
