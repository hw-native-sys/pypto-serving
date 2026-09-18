# Copyright (c) PyPTO Contributors.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""Resolved configuration for one external PD Router process."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypto_serving.serving.pd.config import PDDocument


@dataclass(frozen=True)
class RouterConfig:
    prefill_url: str
    decode_url: str
    run_id: str
    policy: str
    provider: str
    journal_path: str
    log_dir: str
    observability_enabled: bool = True
    route_epoch: int = 1
    control_incarnation: int = 1
    request_timeout_seconds: float = 600.0
    max_request_bytes: int = 4 << 20
    max_active_handoffs: int = 4
    max_pending_handoffs: int = 8

    @classmethod
    def from_document(cls, document: PDDocument) -> "RouterConfig":
        if len(document.runtime.prefill) != 1 or len(document.runtime.decode) != 1:
            raise ValueError("F1 Router requires exactly one P and one D endpoint")
        root = (
            Path(document.observability.root)
            if document.observability.root
            else Path.cwd() / "serving_log" / "pd_disag"
        )
        log_dir = root / document.run_id / "router"
        state_dir = log_dir / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            prefill_url=document.runtime.prefill[0].url,
            decode_url=document.runtime.decode[0].url,
            run_id=document.run_id,
            policy=document.runtime.policy,
            provider=document.runtime.provider,
            journal_path=str(state_dir / "journal.jsonl"),
            log_dir=str(log_dir),
            observability_enabled=document.observability.enabled,
        )
