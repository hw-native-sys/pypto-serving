# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Device tables, dispatch-owned statistics and quiescent maintenance transactions."""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import Future, ThreadPoolExecutor
from collections import deque
import logging
from threading import Lock
from time import monotonic, perf_counter
from typing import Any

import numpy as np
import torch

from pypto_serving.config.types import EplbConfig
from pypto_serving.model.common.runner.buffer_set import shared_empty
from .controller import EplbController
from .plan import RebalancePlan
from .state import LoadSample, Placement
from .transfer import ExpertBankLayout, ExpertBankSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DispatchContext:
    dispatch_id: int
    phase: str
    placement: Placement
    output: torch.Tensor
    record_enabled: bool
    submitted_at: float


class EplbRuntime:
    """One executor's expert tables; maintenance is serialized by its execution lane."""

    def __init__(self, placement: Placement, config: EplbConfig, *, execution_role: str = "mixed",
                 bank: ExpertBankSpec | None = None):
        self.placement = placement
        self.config = config
        self.execution_role = execution_role
        # None keeps the model family's default resident bank convention.
        self.bank = bank
        config.phase_for_role(execution_role)
        self.controller: EplbController | None = None
        self.bank_layout: ExpertBankLayout | None = None
        self.tables: dict[str, Any] = {}
        self._lock = Lock()
        self._next_dispatch = 1
        self._sample_dispatches = 0
        self._next_reclaim = 1
        self._completed: dict[int, LoadSample | None] = {}
        self._inflight: set[int] = set()
        # Serving dispatches every migrated layer on one persistent L3 worker
        # whose retained CommDomain signal windows are not reset, so the
        # notification counters accumulate across calls. Every transfer call
        # increments this 1-based epoch and both barriers wait on it.
        self._epoch = 0
        self._failure: BaseException | None = None
        self._planner: ThreadPoolExecutor | None = None
        self._planning: Future | None = None
        self._candidate: RebalancePlan | None = None
        self._retry_at = 0.0
        self._last_commit = -float("inf")
        self._dispatch_seconds = 0.0
        self._pause_seconds = 0.0
        self._last_payload_bytes = 0
        self._commits = 0
        self._commit_times: deque[float] = deque()
        ranks, slots = placement.ranks, placement.local_slots
        self._sources = shared_empty((ranks, ranks, slots, 2), torch.int32, name="eplb_sources")
        self._changed = shared_empty((ranks, ranks, slots), torch.int32, name="eplb_changed")
        self._sources.zero_()
        self._changed.zero_()

    @property
    def run_config(self):
        from pypto.runtime import RunConfig

        # The transfer kernel borrows every local expert bank view.  EP16 W8A8
        # migration also needs a roughly 2.6 GiB static gm_sm workspace, so the
        # maintenance lane uses the 4 GiB decode ring budget from the serving
        # launch configuration.  The larger heap is required by apply(), even
        # though startup no longer runs a zero-migration transfer.
        # Keep the first three shared L3 rings at the serving baseline and give
        # only the maintenance lane the larger transfer budget.  Passing one
        # integer would be expanded to four 4-GiB rings by the shared worker,
        # reserving 16 GiB before the model cache is allocated.
        return RunConfig(ring_heap=[1 << 29, 1 << 29, 1 << 29, 1 << 32],
                         ring_task_window=131072, ring_dep_pool=131072)

    @property
    def workspace_reserve_bytes(self) -> int:
        """Per-rank HBM reserved for the transfer program's static gm_sm arena.

        The arena and its persistent windows are allocated at the first real
        migration, after the KV cache exists. Reserving the space from the KV
        budget keeps that first migration from OOMing at high memory
        utilization; measured demand is ~2.5 GiB (decode) to >4.4 GiB
        (prefill) per rank for EP16 W8A8. No reserve is needed while migrations
        are disabled.
        """
        if not self.config.enabled or self.config.max_migration_bytes <= 0:
            return 0
        return self.config.maintenance_workspace_bytes

    def inherited_tensors(self) -> list[torch.Tensor]:
        return [self._sources, self._changed]

    def check_healthy(self) -> None:
        if self._failure is not None:
            raise RuntimeError("EPLB executor stopped after a failed EPLB operation") from self._failure

    def table(self, worker, name):
        self.check_healthy()
        if not self.tables:
            mapping, replicas = self._host_tables(self.placement)
            self.tables["eplb_mapping"] = worker.alloc_stacked_tensor(mapping)
            self.tables["eplb_replicas"] = worker.alloc_stacked_tensor(replicas)
        return self.tables[name]

    @staticmethod
    def _host_tables(placement):
        ranks = placement.ranks
        mapping = torch.from_numpy(placement.logical_to_physical.astype(np.int32)).reshape(-1, ranks)
        replicas = torch.from_numpy(placement.replica_count.astype(np.int32)).reshape(-1)
        return (mapping.unsqueeze(0).repeat(ranks, 1, 1), replicas.unsqueeze(0).repeat(ranks, 1))

    def prepare(self, worker, banks, transfer):
        self.check_healthy()
        if self.controller is not None:
            return
        if transfer is None:
            raise RuntimeError("Enabled EPLB requires a compiled device maintenance program")
        layout = ExpertBankLayout.inspect(banks, self.placement, self.bank)
        self.table(worker, "eplb_mapping")
        # Do not execute a zero-migration transfer during startup.  The transfer
        # program still allocates cross-rank windows and waits on barriers even when
        # every slot is unchanged; that no-op path can poison device startup.  The
        # real transfer remains in apply(), after a non-empty plan has been admitted.
        logger.info("[eplb] prepared transfer views without zero-migration warmup")
        # The first real migration publishes one increment per peer.  With no
        # startup warmup, the signal epoch therefore starts at zero.
        self._epoch = 0
        self.bank_layout = layout
        self.controller = EplbController(self.placement, self.config, expert_bytes=layout.expert_bytes,
                                         execution_role=self.execution_role)
        logger.info("[eplb] prepared role=%s generation=0 physical_experts=%d local_slots=%d receive_bytes_per_rank=%d",
                    self.execution_role, self.placement.physical_to_logical.shape[1], self.placement.local_slots,
                    layout.receive_bytes_per_rank)

    def begin(self, phase: str, output: torch.Tensor, control: torch.Tensor, *, record: bool = True):
        self.check_healthy()
        if self.controller is None:
            raise RuntimeError("EPLB dispatch before maintenance preflight")
        with self._lock:
            dispatch_id = self._next_dispatch
            if dispatch_id >= np.iinfo(np.int32).max:
                raise RuntimeError("EPLB dispatch salt exhausted")
            self._next_dispatch += 1
            placement = self.controller.current()
            self._inflight.add(dispatch_id)
            record = (record and phase == self.controller.phase
                      and (not self.config.max_rebalances or self._commits < self.config.max_rebalances))
            if record:
                self._sample_dispatches += 1
                record = self._sample_dispatches % self.config.sample_stride == 0
        control[:, 0].fill_(dispatch_id)
        control[:, 1].fill_(int(record and phase == self.controller.phase))
        return DispatchContext(dispatch_id, phase, placement, output, record, monotonic())

    def complete(self, context: DispatchContext, *, success: bool = True):
        """Retire each context once, forwarding samples in submission order."""
        with self._lock:
            if context.dispatch_id not in self._inflight:
                return
            self.check_healthy()
            try:
                sample = None
                if success and context.record_enabled:
                    slots = context.placement.local_slots
                    counts = context.output[:, :, :slots].numpy()
                    sample = LoadSample(context.dispatch_id, context.phase, context.placement, counts)
                    if context.phase == self.controller.phase:
                        elapsed = monotonic() - context.submitted_at
                        self._dispatch_seconds = (elapsed if not self._dispatch_seconds
                                                  else .9 * self._dispatch_seconds + .1 * elapsed)
                self._inflight.remove(context.dispatch_id)
                self._completed[context.dispatch_id] = sample
                while self._next_reclaim in self._completed:
                    sample = self._completed.pop(self._next_reclaim)
                    if sample is not None:
                        self.controller.record(sample)
                    self._next_reclaim += 1
            except BaseException as error:
                self._failure = error
                self.controller.failed(error)
                raise

    def take_ready_plan(self):
        """Poll one bounded CPU planning job without delaying native submissions."""
        self.check_healthy()
        if self.controller is None:
            return None
        if self.config.max_rebalances and self._commits >= self.config.max_rebalances:
            return None
        now = monotonic()
        if now < self._retry_at or now - self._last_commit < self.config.cooldown_seconds:
            return None
        if not self._maintenance_rate_allows(now):
            logger.debug("[eplb] rate limited role=%s cap=%d/min",
                         self.execution_role, self.config.max_maintenances_per_minute)
            return None
        if self._candidate is not None:
            plan, self._candidate = self._candidate, None
            return plan if self._admit(plan, now) else None
        if self._planning is None and self.controller.has_ready_history():
            if self._planner is None:
                # The first sample arrives after all native workers have forked.
                self._planner = ThreadPoolExecutor(max_workers=1, thread_name_prefix="eplb-planner")
            self._planning = self._planner.submit(self._plan)
        if self._planning is None or not self._planning.done():
            return None
        completed, self._planning = self._planning, None
        try:
            plan, elapsed_ms = completed.result()
        except BaseException as error:
            self._failure = error
            self.controller.failed(error)
            raise
        if plan is not None:
            if not self._admit(plan, monotonic()):
                return None
            logger.info("[eplb] plan ready base_generation=%d planning_ms=%.3f balance=%s",
                        plan.base_generation, elapsed_ms,
                        "n/a" if plan.target_balance is None else f"{plan.target_balance:.4f}")
        return plan

    def _admit(self, plan, now):
        if (plan.base_generation != self.placement.generation
                or now - plan.sampled_at > self.config.max_sample_age_seconds):
            return False
        # Dispatch wall time is a proxy, not a measured MoE kernel speedup.
        savings = self._dispatch_seconds * plan.predicted_gain * self.config.amortization_steps
        cost = self._pause_seconds * max(1.0, plan.payload_bytes / max(1, self._last_payload_bytes))
        if cost and savings < cost * self.config.maintenance_cost_margin:
            logger.info("[eplb] deferred role=%s reason=amortization predicted_savings_ms=%.3f cost_ms=%.3f",
                        self.execution_role, savings * 1000, cost * 1000)
            return False
        return True

    def _maintenance_rate_allows(self, now: float) -> bool:
        """Apply the hard per-minute ceiling independent of the policy cooldown.

        Extreme maintenance rates (hundreds to ~1000 applied migrations within
        minutes) have wedged the PD peer link; the ceiling is a safety bound,
        not a tuning knob. The default policy's cooldown keeps real deployments
        far below it.
        """
        limit = self.config.max_maintenances_per_minute
        if limit <= 0:
            return True
        cutoff = now - 60.0
        while self._commit_times and self._commit_times[0] <= cutoff:
            self._commit_times.popleft()
        return len(self._commit_times) < limit

    def defer(self, plan):
        self.check_healthy()
        self._candidate = plan
        self._retry_at = monotonic() + .1

    def record_pause(self, plan, seconds):
        self._pause_seconds = seconds
        self._last_payload_bytes = plan.payload_bytes

    def _plan(self):
        started = perf_counter()
        plan = self.controller.take_ready_plan()
        return plan, (perf_counter() - started) * 1000

    def close(self):
        """Join CPU planning before the runner releases its native worker."""
        if self._planner is not None:
            self._planner.shutdown(wait=True, cancel_futures=True)
            self._planner = None
            self._planning = None

    def validate(self, plan):
        self.check_healthy()
        if self.controller is None:
            raise RuntimeError("EPLB controller is not prepared")
        self.controller.validate_base(plan)

    def assert_quiescent(self):
        self.check_healthy()
        with self._lock:
            if self._inflight or self._completed:
                raise RuntimeError("EPLB maintenance requires every submitted output to be reclaimed")

    def apply(self, worker, banks, transfer, plan: RebalancePlan):
        """Publish a new layout only after all writes and every table upload complete."""
        self.validate(plan)
        self.assert_quiescent()
        if self._epoch + len(plan.layers) >= np.iinfo(np.int32).max:
            raise RuntimeError("EPLB maintenance epoch exhausted")
        started = perf_counter()
        replica_changes = int(np.count_nonzero(self.placement.replica_count != plan.target.replica_count))
        changed_slots = sum(int(move.changed.sum()) for move in plan.layers)
        destination_ranks = np.arange(self.placement.ranks).reshape(-1, 1)
        remote_slots = sum(int(np.count_nonzero(move.changed & (move.sources[:, :, 0] != destination_ranks)))
                           for move in plan.layers)
        submitted = False
        try:
            for move in plan.layers:
                views = self.bank_layout.layer_views(worker, banks, move.layer_id)
                for destination, source in zip((self._sources, self._changed), self.bank_layout.controls(move)):
                    destination.copy_(source)
                self._epoch += 1
                submitted = True
                worker.run(transfer.compiled, *views, self._sources, self._changed, self._epoch,
                           config=self.run_config)
            mapping, replicas = self._host_tables(plan.target)
            for name, host in (("eplb_mapping", mapping), ("eplb_replicas", replicas)):
                table = self.tables[name]
                for rank, shard in enumerate(table.shards):
                    source = host[rank].contiguous()
                    worker.copy_to(shard.data_ptr, source.data_ptr(), source.numel() * source.element_size(),
                                   worker_id=rank)
            self.controller.committed(plan)
            self.placement = plan.target
            self._last_commit = monotonic()
            self._commit_times.append(self._last_commit)
            self._commits += 1
            logger.info("[eplb] committed role=%s generation=%d layers=%s payload_bytes=%d predicted_gain=%.6f "
                        "changed_slots=%d replica_changes=%d remote_bytes=%d elapsed_ms=%.3f balance=%s",
                        self.execution_role, plan.target.generation, [move.layer_id for move in plan.layers],
                        plan.payload_bytes, plan.predicted_gain, changed_slots, replica_changes,
                        remote_slots * self.bank_layout.expert_bytes, (perf_counter() - started) * 1000,
                        "n/a" if plan.target_balance is None else f"{plan.target_balance:.4f}")
        except BaseException as error:
            if submitted:
                self._failure = error
                self.controller.failed(error)
            raise
