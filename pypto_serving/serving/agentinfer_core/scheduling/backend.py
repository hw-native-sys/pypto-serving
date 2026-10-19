# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
# Modified by PyPTO Serving: internal imports use the bundled scheduling package.
"""Backend, DP-rank capacity, and dispatch types shared by host adapters.

Capacities are expressed as logical KV-token slots and belong to the concrete DP rank that owns the corresponding
scheduling queue. HBM is rank-local physical capacity. Machine-local DRAM or SSD shared by several ranks is recorded
as an adapter-assigned per-rank quota, never duplicated as the full machine capacity. A shared remote cache remains a
backend-pool fact because it is not owned by one rank.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DpRankInfo:
    """Health, allocatable cache quotas, and request load for one data-parallel rank.

    ``total_hbm_kv_tokens`` is the rank's directly observed HBM capacity. DRAM and SSD fields have the same per-rank
    scheduling semantics: when their physical pool is shared by local ranks, the host Adapter must partition the
    machine total into enforceable or estimated rank quotas before constructing this record. Equal division is the
    default approximation for balanced, rank-pinned traffic. An unknown or unpartitionable capacity remains ``None``.
    """

    dp_rank: int
    healthy: bool
    schedulable: bool
    running_requests: int | None = None
    waiting_requests: int | None = None
    total_hbm_kv_tokens: int | None = None
    used_hbm_kv_tokens: int | None = None
    waiting_hbm_kv_tokens: int | None = None
    total_dram_kv_tokens: int | None = None
    total_ssd_kv_tokens: int | None = None
    expected_reasoning_agent_nums: int | None = None

    def __post_init__(self) -> None:
        """Validate rank, local capacities, and optional request counts."""
        if self.dp_rank < 0:
            raise ValueError("dp_rank must be non-negative")
        if any(value is not None and value < 0 for value in (self.running_requests, self.waiting_requests)):
            raise ValueError("DP request counts must be non-negative")
        capacities = (
            self.total_hbm_kv_tokens,
            self.used_hbm_kv_tokens,
            self.waiting_hbm_kv_tokens,
            self.total_dram_kv_tokens,
            self.total_ssd_kv_tokens,
            self.expected_reasoning_agent_nums,
        )
        if any(value is not None and value < 0 for value in capacities):
            raise ValueError("DP-rank capacities and expected agent count must be non-negative")
        if (
            self.total_hbm_kv_tokens is not None
            and self.used_hbm_kv_tokens is not None
            and self.used_hbm_kv_tokens > self.total_hbm_kv_tokens
        ):
            raise ValueError("used_hbm_kv_tokens must not exceed total_hbm_kv_tokens")


@dataclass(frozen=True)
class BackendInfo:
    """One backend and the concrete DP-rank scheduling domains observed beneath it."""

    backend_id: str
    backend_url: str
    healthy: bool
    dp_ranks: tuple[DpRankInfo, ...] = ()
    observed_at_monotonic_s: float = 0.0

    def __post_init__(self) -> None:
        """Validate identity, observation time, and unique rank membership."""
        object.__setattr__(self, "dp_ranks", tuple(self.dp_ranks))
        if not self.backend_id or not self.backend_url:
            raise ValueError("backend identity and URL must not be empty")
        if not math.isfinite(self.observed_at_monotonic_s) or self.observed_at_monotonic_s < 0:
            raise ValueError("backend observation time must be finite and non-negative")
        ranks = [rank.dp_rank for rank in self.dp_ranks]
        if len(ranks) != len(set(ranks)):
            raise ValueError("DP ranks must be unique within one backend observation")


@dataclass(frozen=True)
class BackendPoolInfo:
    """One observation of all backends and their shared remote KV pool."""

    backends: tuple[BackendInfo, ...]
    total_remote_store_kv_tokens: int | None = None
    observed_at_monotonic_s: float = 0.0

    def __post_init__(self) -> None:
        """Validate unique backend ids and shared capacity."""
        object.__setattr__(self, "backends", tuple(self.backends))
        ids = [backend.backend_id for backend in self.backends]
        if len(ids) != len(set(ids)):
            raise ValueError("backend ids must be unique")
        if self.total_remote_store_kv_tokens is not None and self.total_remote_store_kv_tokens < 0:
            raise ValueError("remote KV capacity must be non-negative")
        if not math.isfinite(self.observed_at_monotonic_s) or self.observed_at_monotonic_s < 0:
            raise ValueError("backend-pool observation time must be finite and non-negative")


@dataclass(frozen=True)
class DispatchTarget:
    """Backend and optional DP rank selected for one request."""

    backend_id: str
    dp_rank: int | None = None

    def __post_init__(self) -> None:
        """Validate target identity and optional rank."""
        if not self.backend_id:
            raise ValueError("backend_id must not be empty")
        if self.dp_rank is not None and self.dp_rank < 0:
            raise ValueError("dp_rank must be non-negative")
