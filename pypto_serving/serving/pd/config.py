# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validated, pickle-safe configuration for embedded and external-router PD."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os


PD_SCHEMA_VERSION = 4
PD_DSPARK_SPECULATIVE_TOKENS = 7
PD_PHYSICAL_REGIONS = (
    "ori",
    "hca_cmp",
    "csa_cmp",
    "idx_k",
    "idx_scale",
    "hca_state",
    "csa_state",
    "csa_inner_state",
)
PD_LOGICAL_GROUPS = (
    "ori",
    "cmp_c128",
    "cmp_c4",
    "idx",
    "hca_state",
    "csa_state",
    "csa_inner_state",
)


def _identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 256:
        raise ValueError(f"{name} must be a nonempty string of at most 256 characters")
    return value


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


class PDDeploymentMode(str, Enum):
    EMBEDDED_FIXED_PEER = "embedded-fixed-peer"
    EXTERNAL_ROUTER = "external-router"


@dataclass(frozen=True)
class PDConfig:
    """Host configuration; secrets stay in the named environment variable."""

    role: PDRole = PDRole.DISABLED
    deployment_mode: PDDeploymentMode = PDDeploymentMode.EMBEDDED_FIXED_PEER
    node_id: str = ""
    peer_node_id: str = ""
    run_id: str = ""
    control_host: str = "127.0.0.1"
    control_port: int = 29831
    control_advertise_host: str = ""
    peer_host: str = ""
    auth_secret_env: str = "PYPTO_PD_AUTH_SECRET"
    router_auth_secret_env: str = "PYPTO_PD_ROUTER_SECRET"
    transfer_hostname: str = ""
    model_revision: str = ""
    generation: int = 1
    route_epoch: int = 1
    control_incarnation: int = 1
    provider: str = "mooncake"
    decode_speculative_tokens: int = PD_DSPARK_SPECULATIVE_TOKENS
    connect_timeout_seconds: float = 30.0
    request_timeout_seconds: float = 60.0
    transfer_poll_interval_seconds: float = 0.005
    max_active_handoffs: int = 4
    max_pending_handoffs: int = 8
    max_inflight_transfer_bytes: int = 1 << 30
    max_transfer_attempts: int = 2
    enable_chunk_overlap: bool = False
    prepared_request_ttl_seconds: float = 300.0
    journal_path: str = ""

    def __post_init__(self) -> None:
        role = self.role if isinstance(self.role, PDRole) else PDRole(self.role)
        object.__setattr__(self, "role", role)
        mode = (
            self.deployment_mode
            if isinstance(self.deployment_mode, PDDeploymentMode)
            else PDDeploymentMode(self.deployment_mode)
        )
        object.__setattr__(self, "deployment_mode", mode)
        if role is PDRole.DISABLED:
            return
        for name in (
            "node_id",
            "run_id",
            "control_host",
            "auth_secret_env",
            "router_auth_secret_env",
            "transfer_hostname",
            "model_revision",
            "provider",
        ):
            _identifier(getattr(self, name), name)
        if mode is PDDeploymentMode.EMBEDDED_FIXED_PEER:
            _identifier(self.peer_node_id, "peer_node_id")
            _identifier(self.peer_host, "peer_host")
            if self.node_id == self.peer_node_id:
                raise ValueError("PD node_id and peer_node_id must differ")
        else:
            _identifier(self.control_advertise_host, "control_advertise_host")
            if self.peer_node_id:
                _identifier(self.peer_node_id, "peer_node_id")
                if self.node_id == self.peer_node_id:
                    raise ValueError("PD node_id and peer_node_id must differ")
            if self.peer_host:
                _identifier(self.peer_host, "peer_host")
        if type(self.control_port) is not int or not 1 <= self.control_port <= 65535:
            raise ValueError("PD control_port must be in [1, 65535]")
        if self.decode_speculative_tokens != PD_DSPARK_SPECULATIVE_TOKENS:
            raise ValueError(
                "the first PD version requires DeepSeek V4 DSpark K7 Decode"
            )
        for name in ("generation", "route_epoch", "control_incarnation"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"PD {name} must be a positive integer")
        for name in (
            "connect_timeout_seconds",
            "request_timeout_seconds",
            "transfer_poll_interval_seconds",
            "prepared_request_ttl_seconds",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"PD {name} must be positive")
        for name in (
            "max_active_handoffs",
            "max_pending_handoffs",
            "max_inflight_transfer_bytes",
            "max_transfer_attempts",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"PD {name} must be a positive integer")
        if self.max_active_handoffs > self.max_pending_handoffs:
            raise ValueError("PD active handoffs must not exceed total pending handoffs")
        if self.max_transfer_attempts > 8:
            raise ValueError("PD max_transfer_attempts must not exceed 8")
        if type(self.enable_chunk_overlap) is not bool:
            raise ValueError("PD enable_chunk_overlap must be a boolean")
        if not isinstance(self.journal_path, str) or len(os.fsencode(self.journal_path)) > 4096:
            raise ValueError("PD journal_path must be a string of at most 4096 bytes")

    @property
    def enabled(self) -> bool:
        return self.role is not PDRole.DISABLED

    @property
    def external_router(self) -> bool:
        return self.deployment_mode is PDDeploymentMode.EXTERNAL_ROUTER

    def auth_secret(self) -> bytes:
        """Resolve the pre-shared secret without storing it in logs/config dumps."""
        if not self.enabled:
            raise RuntimeError("PD authentication is unavailable when PD is disabled")
        value = os.environ.get(self.auth_secret_env, "")
        if len(value.encode()) < 16:
            raise RuntimeError(
                f"{self.auth_secret_env} must contain at least 16 bytes for PD authentication"
            )
        return value.encode()

    def router_auth_secret(self) -> bytes:
        """Resolve the Router signing secret used only by Router and D."""
        if not self.enabled or self.role is not PDRole.DECODE:
            raise RuntimeError("PD Router ticket verification is available only on D")
        value = os.environ.get(self.router_auth_secret_env, "")
        if len(value.encode()) < 16:
            raise RuntimeError(
                f"{self.router_auth_secret_env} must contain at least 16 bytes "
                "for PD route-ticket verification"
            )
        return value.encode()

    def worker_config(self) -> "PDWorkerConfig | None":
        if not self.enabled:
            return None
        return PDWorkerConfig(
            role=self.role,
            run_id=self.run_id,
            worker_id=f"{self.node_id}-{self.role.value}",
            transfer_hostname=self.transfer_hostname,
            generation=self.generation,
            endpoint_generation=self.generation,
            model_revision=self.model_revision,
            max_active_handoffs=self.max_active_handoffs,
            max_inflight_transfer_bytes=self.max_inflight_transfer_bytes,
        )


@dataclass(frozen=True)
class PDWorkerConfig:
    """Minimal address-free config copied into the spawned serving worker."""

    role: PDRole
    run_id: str
    worker_id: str
    transfer_hostname: str
    generation: int
    endpoint_generation: int
    model_revision: str
    max_active_handoffs: int = 1
    max_inflight_transfer_bytes: int = 1 << 30

    def __post_init__(self) -> None:
        role = self.role if isinstance(self.role, PDRole) else PDRole(self.role)
        object.__setattr__(self, "role", role)
        if role is PDRole.DISABLED:
            raise ValueError("PDWorkerConfig cannot use the disabled role")
        for name in ("run_id", "worker_id", "transfer_hostname", "model_revision"):
            _identifier(getattr(self, name), name)
        for name in ("generation", "endpoint_generation"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("max_active_handoffs", "max_inflight_transfer_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class PDCapabilities:
    """Host-visible compatibility contract; it intentionally contains no address."""

    model_revision: str
    registry_fingerprint: str
    layout_fingerprint: str
    topology: tuple[int, ...]
    provider: str = "mooncake"
    schema_version: int = PD_SCHEMA_VERSION
    logical_groups: tuple[str, ...] = PD_LOGICAL_GROUPS
    physical_regions: tuple[str, ...] = PD_PHYSICAL_REGIONS
    chunk_transfer: bool = True
    target_cache_only: bool = True
    decode_speculative_tokens: int = PD_DSPARK_SPECULATIVE_TOKENS

    def __post_init__(self) -> None:
        for name in (
            "model_revision",
            "registry_fingerprint",
            "layout_fingerprint",
            "provider",
        ):
            _identifier(getattr(self, name), name)
        if self.schema_version != PD_SCHEMA_VERSION:
            raise ValueError(f"unsupported PD schema version {self.schema_version}")
        if not self.topology or any(type(value) is not int or value <= 0 for value in self.topology):
            raise ValueError("PD topology must contain positive integer dimensions")
        if self.logical_groups != PD_LOGICAL_GROUPS:
            raise ValueError("PD logical cache groups do not match the first-version contract")
        if self.physical_regions != PD_PHYSICAL_REGIONS:
            raise ValueError("PD physical regions do not match the first-version contract")
        if not self.chunk_transfer or not self.target_cache_only:
            raise ValueError(
                "the first PD version requires chunk transfer of target cache only"
            )
        if self.decode_speculative_tokens != PD_DSPARK_SPECULATIVE_TOKENS:
            raise ValueError(
                "the first PD version requires DeepSeek V4 DSpark K7 Decode"
            )

    def compatibility_error(self, peer: "PDCapabilities") -> str | None:
        for name in (
            "schema_version",
            "model_revision",
            "layout_fingerprint",
            "topology",
            "provider",
            "logical_groups",
            "physical_regions",
            "chunk_transfer",
            "target_cache_only",
            "decode_speculative_tokens",
        ):
            if getattr(self, name) != getattr(peer, name):
                return f"PD capability mismatch: {name}"
        return None
