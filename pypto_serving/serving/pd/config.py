# Copyright (c) PyPTO Contributors.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""Strict product configuration for the external-router PD engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .adapter import ModelPDAdapter
from .contracts import RuntimeLayoutDescriptor


PD_SCHEMA_VERSION = 4
def _identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not value or len(value.encode()) > 256:
        raise ValueError(f"{name} must be a nonempty string of at most 256 bytes")
    return value


def _strict_keys(value: dict[str, Any], allowed: set[str], name: str) -> None:
    extra = set(value) - allowed
    if extra:
        raise ValueError(f"unknown {name} fields: {sorted(extra)}")


class PDRole(str, Enum):
    DISABLED = "disabled"
    PREFILL = "prefill"
    DECODE = "decode"

    @property
    def peer(self) -> "PDRole":
        if self is PDRole.PREFILL:
            return PDRole.DECODE
        if self is PDRole.DECODE:
            return PDRole.PREFILL
        return PDRole.DISABLED


@dataclass(frozen=True)
class PDEndpoint:
    host: str
    port: int
    node_id: str = ""
    control_host: str = ""
    control_port: int = 29831
    transfer_hostname: str = ""

    def __post_init__(self) -> None:
        _identifier(self.host, "endpoint.host")
        for name in ("port", "control_port"):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= 65535:
                raise ValueError(f"endpoint.{name} must be in [1, 65535]")
        for name in ("node_id", "control_host", "transfer_hostname"):
            value = getattr(self, name)
            if value:
                _identifier(value, f"endpoint.{name}")

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass(frozen=True)
class PDRuntimeConfig:
    prefill: tuple[PDEndpoint, ...]
    decode: tuple[PDEndpoint, ...]
    provider: str = "mooncake"
    policy: str = "round_robin"
    run_id: str = ""

    def __post_init__(self) -> None:
        if not self.prefill or not self.decode:
            raise ValueError("runtime requires at least one prefill and one decode endpoint")
        for role, endpoints in (("prefill", self.prefill), ("decode", self.decode)):
            identities = tuple((item.host, item.port) for item in endpoints)
            if len(identities) != len(set(identities)):
                raise ValueError(f"runtime.{role} contains duplicate endpoints")
            node_ids = tuple(item.node_id for item in endpoints if item.node_id)
            if len(node_ids) != len(set(node_ids)):
                raise ValueError(f"runtime.{role} contains duplicate node ids")
        _identifier(self.provider, "runtime.provider")
        _identifier(self.policy, "runtime.policy")
        if self.run_id:
            _identifier(self.run_id, "runtime.run_id")


@dataclass(frozen=True)
class PDObservabilityConfig:
    enabled: bool = True
    root: str = ""

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("observability.enabled must be boolean")
        if not isinstance(self.root, str) or len(os.fsencode(self.root)) > 4096:
            raise ValueError("observability.root must be a bounded path")


@dataclass(frozen=True)
class PDDocument:
    runtime: PDRuntimeConfig
    observability: PDObservabilityConfig

    @property
    def run_id(self) -> str:
        if self.runtime.run_id:
            return self.runtime.run_id
        canonical = {
            "prefill": [(item.host, item.port) for item in self.runtime.prefill],
            "decode": [(item.host, item.port) for item in self.runtime.decode],
            "provider": self.runtime.provider,
            "policy": self.runtime.policy,
        }
        digest = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        return f"pd-{digest}"

    def endpoint(self, role: PDRole, node_id: str = "") -> PDEndpoint:
        endpoints = self.runtime.prefill if role is PDRole.PREFILL else self.runtime.decode
        if node_id:
            matches = tuple(item for item in endpoints if item.node_id == node_id)
            if len(matches) != 1:
                raise ValueError(f"node_id {node_id!r} must select exactly one {role.value} endpoint")
            return matches[0]
        if len(endpoints) != 1:
            raise ValueError(f"--pd-node-id is required when {role.value} has multiple endpoints")
        return endpoints[0]


def _endpoint(value: Any, name: str) -> PDEndpoint:
    if not isinstance(value, dict):
        raise ValueError(f"{name} endpoint must be an object")
    _strict_keys(
        value,
        {"host", "port", "node_id", "control_host", "control_port", "transfer_hostname"},
        name,
    )
    return PDEndpoint(**value)


def load_pd_document(path: str | os.PathLike[str]) -> PDDocument:
    """Load one strict JSON document shared by Router, P and D."""
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load PD config {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("PD config root must be an object")
    _strict_keys(raw, {"runtime", "observability"}, "PD config")
    runtime = raw.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("PD config requires a runtime object")
    _strict_keys(runtime, {"prefill", "decode", "provider", "policy", "run_id"}, "runtime")
    prefill = runtime.get("prefill")
    decode = runtime.get("decode")
    if not isinstance(prefill, list) or not isinstance(decode, list):
        raise ValueError("runtime.prefill and runtime.decode must be arrays")
    runtime_config = PDRuntimeConfig(
        prefill=tuple(_endpoint(value, "prefill") for value in prefill),
        decode=tuple(_endpoint(value, "decode") for value in decode),
        provider=runtime.get("provider", "mooncake"),
        policy=runtime.get("policy", "round_robin"),
        run_id=runtime.get("run_id", ""),
    )
    observability = raw.get("observability", {})
    if not isinstance(observability, dict):
        raise ValueError("observability must be an object")
    _strict_keys(observability, {"enabled", "root"}, "observability")
    return PDDocument(
        runtime=runtime_config,
        observability=PDObservabilityConfig(
            enabled=observability.get("enabled", True),
            root=observability.get("root", ""),
        ),
    )


@dataclass(frozen=True)
class PDConfig:
    """Resolved node config; values absent from JSON are Serving defaults."""

    role: PDRole
    node_id: str
    run_id: str
    control_host: str
    control_port: int
    control_advertise_host: str
    transfer_hostname: str
    model_revision: str
    model_adapter: ModelPDAdapter
    provider: str = "mooncake"
    generation: int = 1
    route_epoch: int = 1
    control_incarnation: int = 1
    connect_timeout_seconds: float = 30.0
    request_timeout_seconds: float = 600.0
    transfer_poll_interval_seconds: float = 0.005
    max_active_handoffs: int = 4
    max_pending_handoffs: int = 8
    max_transfer_attempts: int = 2
    enable_chunk_overlap: bool = False
    prepared_request_ttl_seconds: float = 300.0
    journal_path: str = ""
    log_dir: str = ""
    observability_enabled: bool = True

    def __post_init__(self) -> None:
        role = self.role if isinstance(self.role, PDRole) else PDRole(self.role)
        object.__setattr__(self, "role", role)
        if role is PDRole.DISABLED:
            raise ValueError("PDConfig cannot use the disabled role")
        for name in (
            "node_id", "run_id", "control_host", "control_advertise_host",
            "transfer_hostname", "model_revision", "provider",
        ):
            _identifier(getattr(self, name), name)
        if self.model_adapter.contract.model_family == "" or self.model_adapter.contract.version < 1:
            raise ValueError("PD model contract is invalid")
        if type(self.control_port) is not int or not 1 <= self.control_port <= 65535:
            raise ValueError("PD control_port must be in [1, 65535]")
        if self.max_active_handoffs > self.max_pending_handoffs:
            raise ValueError("PD active handoffs must not exceed pending handoffs")

    @property
    def enabled(self) -> bool:
        return True

    @property
    def model_contract(self):
        return self.model_adapter.contract

    def worker_config(self) -> "PDWorkerConfig":
        return PDWorkerConfig(
            role=self.role,
            run_id=self.run_id,
            worker_id=f"{self.node_id}-{self.role.value}",
            transfer_hostname=self.transfer_hostname,
            generation=self.generation,
            endpoint_generation=self.generation,
            model_revision=self.model_revision,
            max_active_handoffs=self.max_active_handoffs,
        )


def resolve_pd_config(
    document: PDDocument,
    *,
    role: PDRole,
    model_revision: str,
    model_adapter: ModelPDAdapter,
    node_id: str = "",
) -> PDConfig:
    endpoint = document.endpoint(role, node_id)
    resolved_node_id = endpoint.node_id or (
        f"{role.value}-{hashlib.sha256(f'{endpoint.host}:{endpoint.port}'.encode()).hexdigest()[:8]}"
    )
    root = (
        Path(document.observability.root)
        if document.observability.root
        else Path.cwd() / "serving_log" / "pd_disag"
    )
    role_dir = root / document.run_id / role.value / resolved_node_id
    state_dir = role_dir / "state"
    # Correctness state remains durable when optional trace/log output is disabled.
    state_dir.mkdir(parents=True, exist_ok=True)
    return PDConfig(
        role=role,
        node_id=resolved_node_id,
        run_id=document.run_id,
        control_host=endpoint.control_host or endpoint.host,
        control_port=endpoint.control_port,
        control_advertise_host=endpoint.control_host or endpoint.host,
        transfer_hostname=endpoint.transfer_hostname or endpoint.host,
        model_revision=model_revision,
        model_adapter=model_adapter,
        provider=document.runtime.provider,
        journal_path=str(state_dir / "journal.jsonl"),
        log_dir=str(role_dir),
        observability_enabled=document.observability.enabled,
    )


@dataclass(frozen=True)
class PDWorkerConfig:
    role: PDRole
    run_id: str
    worker_id: str
    transfer_hostname: str
    generation: int
    endpoint_generation: int
    model_revision: str
    max_active_handoffs: int = 1


@dataclass(frozen=True)
class PDCapabilities:
    adapter_id: str
    contract_version: int
    contract_digest: str
    continuation_schema: str
    model_revision: str
    registry_fingerprint: str
    layout_fingerprint: str
    topology: tuple[int, ...]
    provider: str = "mooncake"
    schema_version: int = PD_SCHEMA_VERSION
    logical_groups: tuple[str, ...] = ()
    physical_regions: tuple[str, ...] = ()

    @classmethod
    def from_layout(cls, layout: RuntimeLayoutDescriptor) -> "PDCapabilities":
        return cls(
            adapter_id=layout.adapter_id,
            contract_version=layout.contract_version,
            contract_digest=layout.contract_digest,
            continuation_schema=layout.continuation_schema,
            model_revision=layout.model_revision,
            registry_fingerprint=layout.registry_fingerprint,
            layout_fingerprint=layout.layout_fingerprint,
            topology=layout.topology,
            provider=layout.provider,
            logical_groups=layout.logical_groups,
            physical_regions=layout.physical_regions,
        )

    def compatibility_error(self, peer: "PDCapabilities") -> str | None:
        for name in (
            "schema_version", "adapter_id", "contract_version", "contract_digest",
            "continuation_schema", "model_revision", "layout_fingerprint", "topology",
            "provider", "logical_groups", "physical_regions",
        ):
            if getattr(self, name) != getattr(peer, name):
                return f"PD capability mismatch: {name}"
        return None
