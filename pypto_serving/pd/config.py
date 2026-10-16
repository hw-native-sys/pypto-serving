# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Small deployment configuration; model and compute options remain Serving options."""

from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager
import json
import logging
from pathlib import Path
import uuid

PROVIDERS = {"mooncake": ("pypto_serving.pd.transfer", "Mooncake")}


@dataclass(frozen=True)
class Endpoint:
    host: str
    port: int
    node_id: str = ""
    control_port: int = 0
    transfer_hostname: str = ""

    def __post_init__(self):
        if not self.host or any(c in self.host for c in "/?#@"):
            raise ValueError("endpoint host must be a hostname or IP address")
        for port in (self.port, self.control_port or self.port + 10000):
            if type(port) is not int or not 0 < port < 65536:
                raise ValueError("endpoint port outside 1..65535")
        object.__setattr__(self, "node_id", self.node_id or f"{self.host}:{self.port}")
        object.__setattr__(self, "control_port", self.control_port or self.port + 10000)

    @property
    def url(self):
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{host}:{self.port}"


@dataclass(frozen=True)
class Config:
    prefill: tuple[Endpoint, ...]
    decode: tuple[Endpoint, ...]
    provider: str = "mooncake"
    policy: str = "round_robin"
    capacity: int = 8
    handoff_timeout: float = 300.0
    request_timeout: float = 3600.0
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8])
    log_enabled: bool = True
    log_root: str = "serving_log/pd_disag"

    def __post_init__(self):
        if self.provider not in PROVIDERS:
            raise ValueError(f"unsupported PD provider: {self.provider}")
        if self.policy != "round_robin":
            raise ValueError(f"unsupported PD policy: {self.policy}")
        nodes = self.prefill + self.decode
        if not self.prefill or not self.decode or len({n.node_id for n in nodes}) != len(nodes):
            raise ValueError("PD requires nonempty P/D pools with unique node IDs")
        if len({(n.host, n.port) for n in nodes}) != len(nodes):
            raise ValueError("node HTTP endpoints must be unique")
        if len({(n.host, n.control_port) for n in nodes}) != len(nodes):
            raise ValueError("node control endpoints must be unique")
        if type(self.capacity) is not int or self.capacity < 1:
            raise ValueError("capacity must be positive")
        if not 0 < self.handoff_timeout <= self.request_timeout < float("inf"):
            raise ValueError("timeouts must be finite and request >= handoff > 0")
        if not self.run_id or Path(self.run_id).name != self.run_id or self.run_id in (".", ".."):
            raise ValueError("run_id must be one path component")

    @classmethod
    def load(cls, path):
        document = json.loads(Path(path).read_text())
        if set(document) - {"runtime", "observability"}:
            raise ValueError("unknown PD document section")
        runtime = dict(document["runtime"])
        logs = document.get("observability", {})
        if set(logs) - {"enabled", "root"} or type(logs.get("enabled", True)) is not bool:
            raise ValueError("invalid observability options")
        try:
            return cls(
                **{**runtime, "prefill": tuple(Endpoint(**n) for n in runtime["prefill"]),
                   "decode": tuple(Endpoint(**n) for n in runtime["decode"])},
                log_enabled=logs.get("enabled", True), log_root=logs.get("root", "serving_log/pd_disag"),
            )
        except (TypeError, KeyError) as exc:
            raise ValueError(f"invalid PD configuration: {exc}") from exc

    def endpoint(self, role, node_id=None):
        if role not in ("prefill", "decode"):
            raise ValueError("role must be prefill or decode")
        nodes = getattr(self, role)
        selected = [n for n in nodes if node_id is None or n.node_id == node_id]
        if len(selected) != 1:
            raise ValueError("select one configured node with --pd-node-id")
        return selected[0]


@contextmanager
def log_context(config, role, node_id="router"):
    handler = None
    logger = logging.getLogger("pypto_serving.pd")
    if config.log_enabled:
        role = {"prefill": "p", "decode": "d"}.get(role, role)
        directory = Path(config.log_root) / config.run_id / role
        directory.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(directory / (uuid.uuid5(uuid.NAMESPACE_URL, node_id).hex[:12] + ".log"))
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
    try:
        yield
    finally:
        if handler is not None:
            logger.removeHandler(handler)
            handler.close()
