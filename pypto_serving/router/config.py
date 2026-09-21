# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Resolved configuration for one external PD Router process."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypto_serving.serving.pd.config import PDDocument


@dataclass(frozen=True)
class RouterConfig:
    prefill_urls: tuple[str, ...]
    decode_urls: tuple[str, ...]
    run_id: str
    policy: str
    provider: str
    journal_path: str
    log_dir: str
    observability_enabled: bool = True
    data_generation: int = 1
    route_epoch: int = 1
    control_incarnation: int = 1
    request_timeout_seconds: float = 600.0
    max_request_bytes: int = 4 << 20
    max_active_handoffs: int = 4
    max_pending_handoffs: int = 8

    @classmethod
    def from_document(cls, document: PDDocument) -> "RouterConfig":
        root = (
            Path(document.observability.root)
            if document.observability.root
            else Path.cwd() / "serving_log" / "pd_disag"
        )
        log_dir = root / document.run_id / "router"
        state_dir = log_dir / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            prefill_urls=tuple(endpoint.url for endpoint in document.runtime.prefill),
            decode_urls=tuple(endpoint.url for endpoint in document.runtime.decode),
            run_id=document.run_id,
            policy=document.runtime.policy,
            provider=document.runtime.provider,
            journal_path=str(state_dir / "journal.jsonl"),
            log_dir=str(log_dir),
            observability_enabled=document.observability.enabled,
            data_generation=document.runtime.generation,
            route_epoch=document.runtime.route_epoch,
            control_incarnation=document.runtime.control_incarnation,
            max_active_handoffs=document.runtime.max_active_handoffs,
            max_pending_handoffs=document.runtime.max_pending_handoffs,
        )
