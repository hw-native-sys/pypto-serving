# Copyright (c) PyPTO Contributors.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""Validated configuration for the fixed 1P1D external Router."""

from __future__ import annotations

from dataclasses import dataclass
import os
from urllib.parse import urlparse


@dataclass(frozen=True)
class RouterConfig:
    prefill_url: str
    decode_url: str
    run_id: str
    model_id: str = ""
    auth_secret_env: str = "PYPTO_PD_AUTH_SECRET"
    route_secret_env: str = "PYPTO_PD_ROUTER_SECRET"
    route_epoch: int = 1
    control_incarnation: int = 1
    request_timeout_seconds: float = 600.0
    ticket_ttl_seconds: float = 300.0
    journal_path: str = ""
    max_request_bytes: int = 4 << 20
    max_active_handoffs: int = 4
    max_pending_handoffs: int = 8

    def __post_init__(self) -> None:
        for name in ("prefill_url", "decode_url"):
            value = getattr(self, name).rstrip("/")
            parsed = urlparse(value)
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                raise ValueError(f"{name} must be an absolute HTTP(S) URL")
            object.__setattr__(self, name, value)
        if self.prefill_url == self.decode_url:
            raise ValueError("Router P and D service endpoints must differ")
        for name in ("run_id", "auth_secret_env", "route_secret_env"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or len(value.encode()) > 256:
                raise ValueError(f"{name} must be a nonempty bounded string")
        for name in (
            "route_epoch",
            "control_incarnation",
            "max_request_bytes",
            "max_active_handoffs",
            "max_pending_handoffs",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_active_handoffs > self.max_pending_handoffs:
            raise ValueError("Router active handoffs must not exceed total pending handoffs")
        for name in ("request_timeout_seconds", "ticket_ttl_seconds"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if not self.journal_path:
            raise ValueError("Router requires an append-only journal path")

    def auth_secret(self) -> str:
        value = os.environ.get(self.auth_secret_env, "")
        if len(value.encode()) < 16:
            raise RuntimeError(f"{self.auth_secret_env} must contain at least 16 bytes")
        return value

    def route_secret(self) -> bytes:
        value = os.environ.get(self.route_secret_env, "")
        if len(value.encode()) < 16:
            raise RuntimeError(f"{self.route_secret_env} must contain at least 16 bytes")
        return value.encode()
