# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Per-executor load-window ownership and placement publication."""

from __future__ import annotations

from collections import deque
from dataclasses import replace
import logging
from threading import Lock
from time import monotonic

import numpy as np

from pypto_serving.config.types import EplbConfig
from .plan import RebalancePlan, validate_plan
from .policy import plan_rebalance
from .state import LoadSample, Placement

logger = logging.getLogger(__name__)


class EplbController:
    """Consume FIFO reclaims once; never hold a lock while planning or running a kernel."""

    def __init__(self, placement: Placement, config: EplbConfig, *, expert_bytes: int,
                 execution_role: str = "mixed") -> None:
        if not config.enabled or expert_bytes <= 0:
            raise ValueError("EPLB controller requires enabled configuration and a positive expert size")
        self.config = config
        self.phase = config.phase_for_role(execution_role)
        self.expert_bytes = expert_bytes
        self._placement = placement
        self._lock = Lock()
        self._window: deque[np.ndarray] = deque(maxlen=config.window_size)
        self._selected_steps = 0
        self._last_dispatch_id = -1
        self._commit_index = 0
        self._layer_commit: dict[int, int] = {}
        self._ready_history: np.ndarray | None = None
        self._ready_at = 0.0
        self._failure: BaseException | None = None

    def _check_healthy(self) -> None:
        if self._failure is not None:
            raise RuntimeError("EPLB executor is in a fatal maintenance state") from self._failure

    def current(self) -> Placement:
        with self._lock:
            self._check_healthy()
            return self._placement

    def record(self, sample: LoadSample) -> bool:
        """Called in output-queue order, before releasing the dispatch's output slot."""
        with self._lock:
            self._check_healthy()
            if sample.placement.generation < self._placement.generation:
                return False
            if sample.placement is not self._placement:
                raise ValueError("EPLB sample must own the submitted placement snapshot")
            if sample.dispatch_id <= self._last_dispatch_id:
                return False
            self._last_dispatch_id = sample.dispatch_id
            if sample.phase != self.phase:
                return False
            self._window.append(sample.logical_counts())
            self._selected_steps += 1
            if self._selected_steps % self.config.step_interval == 0:
                self._ready_history = np.stack(self._window)
                self._ready_at = monotonic()
                # Per-layer expert concentration: an aggregate that looks uniform
                # can still hide a different hot set in every layer.
                per_layer = self._ready_history.sum(axis=0)
                totals = per_layer.sum(axis=1)
                means = totals / per_layer.shape[1]
                safe_means = np.where(means > 0, means, 1.0)
                skew_l = np.where(totals > 0, per_layer.max(axis=1) / safe_means, 0.0)
                top10_l = np.sort(per_layer, axis=1)[:, -10:].sum(axis=1) / np.where(totals > 0, totals, 1.0)
                aggregate = per_layer.sum(axis=0)
                top_experts = np.argsort(aggregate)[-min(16, aggregate.size):][::-1]
                logger.info("[eplb] window phase=%s samples=%d skew=%.3f top10_share=%.3f top_experts=%s",
                            self.phase, self._ready_history.shape[0],
                            float(skew_l.mean()), float(top10_l.mean()),
                            ",".join(str(int(expert)) for expert in top_experts))
            return True

    def take_ready_plan(self) -> RebalancePlan | None:
        with self._lock:
            self._check_healthy()
            history = self._ready_history
            sampled_at = self._ready_at
            self._ready_history = None
            base = self._placement
            cooldown = self.config.layer_cooldown_rebalances
            blocked = frozenset(
                layer for layer, index in self._layer_commit.items()
                if cooldown and self._commit_index - index <= cooldown
            )
        if history is None:
            return None
        plan = plan_rebalance(base, history, self.config, expert_bytes=self.expert_bytes,
                              blocked_layers=blocked)
        with self._lock:
            self._check_healthy()
            if self._placement is not base:
                return None
        return replace(plan, sampled_at=sampled_at) if plan is not None else None

    def has_ready_history(self) -> bool:
        """Inspect readiness without running the placement policy on the caller."""
        with self._lock:
            self._check_healthy()
            return self._ready_history is not None

    def _layer_limit(self) -> int:
        limit = self.config.max_layers_per_rebalance
        if self._placement.generation == 0 and self.config.first_rebalance_layers > limit:
            return min(self.config.first_rebalance_layers, self._placement.layers)
        return limit

    def validate_base(self, plan: RebalancePlan) -> None:
        with self._lock:
            self._check_healthy()
            validate_plan(self._placement, plan, expert_bytes=self.expert_bytes,
                          max_layers=self._layer_limit(),
                          max_migration_bytes=self.config.max_migration_bytes)

    def committed(self, plan: RebalancePlan) -> None:
        with self._lock:
            self._check_healthy()
            validate_plan(self._placement, plan, expert_bytes=self.expert_bytes,
                          max_layers=self._layer_limit(),
                          max_migration_bytes=self.config.max_migration_bytes)
            self._placement = plan.target
            self._commit_index += 1
            for move in plan.layers:
                self._layer_commit[move.layer_id] = self._commit_index
            self._window.clear()
            self._ready_history = None
            self._selected_steps = 0

    def failed(self, error: BaseException) -> None:
        with self._lock:
            self._failure = error
            self._ready_history = None
