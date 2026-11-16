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
from threading import Lock

import numpy as np

from pypto_serving.config.types import EplbConfig
from .plan import RebalancePlan, validate_plan
from .policy import plan_rebalance
from .state import LoadSample, Placement


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
        self._ready_history: np.ndarray | None = None
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
            return True

    def take_ready_plan(self) -> RebalancePlan | None:
        with self._lock:
            self._check_healthy()
            history = self._ready_history
            self._ready_history = None
            base = self._placement
        if history is None:
            return None
        plan = plan_rebalance(base, history, self.config, expert_bytes=self.expert_bytes)
        with self._lock:
            self._check_healthy()
            if self._placement is not base:
                return None
        return plan

    def has_ready_history(self) -> bool:
        """Inspect readiness without running the placement policy on the caller."""
        with self._lock:
            self._check_healthy()
            return self._ready_history is not None

    def validate_base(self, plan: RebalancePlan) -> None:
        with self._lock:
            self._check_healthy()
            validate_plan(self._placement, plan, expert_bytes=self.expert_bytes,
                          max_layers=self.config.max_layers_per_rebalance,
                          max_migration_bytes=self.config.max_migration_bytes)

    def committed(self, plan: RebalancePlan) -> None:
        with self._lock:
            self._check_healthy()
            validate_plan(self._placement, plan, expert_bytes=self.expert_bytes,
                          max_layers=self.config.max_layers_per_rebalance,
                          max_migration_bytes=self.config.max_migration_bytes)
            self._placement = plan.target
            self._window.clear()
            self._ready_history = None
            self._selected_steps = 0

    def failed(self, error: BaseException) -> None:
        with self._lock:
            self._failure = error
            self._ready_history = None
