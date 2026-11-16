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
import logging
from threading import Lock
from time import perf_counter
from typing import Any

import numpy as np
import torch

from pypto_serving.config.types import EplbConfig
from pypto_serving.model.common.runner.buffer_set import shared_empty
from .controller import EplbController
from .plan import RebalancePlan
from .state import LoadSample, Placement
from .transfer import ExpertBankLayout

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DispatchContext:
    dispatch_id: int
    phase: str
    placement: Placement
    output: torch.Tensor
    record_enabled: bool


class EplbRuntime:
    """One executor's expert tables; maintenance is serialized by its execution lane."""

    def __init__(self, placement: Placement, config: EplbConfig):
        self.placement = placement
        self.config = config
        self.controller: EplbController | None = None
        self.bank_layout: ExpertBankLayout | None = None
        self.tables: dict[str, Any] = {}
        self._lock = Lock()
        self._next_dispatch = 1
        self._next_reclaim = 1
        self._completed: dict[int, LoadSample | None] = {}
        self._inflight: set[int] = set()
        self._epoch = 0
        self._failure: BaseException | None = None
        self._planner: ThreadPoolExecutor | None = None
        self._planning: Future | None = None
        ranks, slots = placement.ranks, placement.local_slots
        self._sources = shared_empty((ranks, ranks, slots, 2), torch.int32, name="eplb_sources")
        self._changed = shared_empty((ranks, ranks, slots), torch.int32, name="eplb_changed")
        self._sources.zero_()
        self._changed.zero_()

    @property
    def run_config(self):
        from pypto.runtime import RunConfig

        return RunConfig(ring_heap=268435456, ring_task_window=32768, ring_dep_pool=32768)

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
        layout = ExpertBankLayout.inspect(banks, self.placement)
        self.table(worker, "eplb_mapping")
        views = layout.layer_views(worker, banks, 0)
        # Warm persistent receive/signal windows before the KV capacity snapshot.
        worker.run(transfer.compiled, *views, self._sources, self._changed, 1, config=self.run_config)
        self._epoch = 1
        self.bank_layout = layout
        self.controller = EplbController(self.placement, self.config, expert_bytes=layout.expert_bytes)
        logger.info("[eplb] prepared generation=0 physical_experts=%d local_slots=%d receive_bytes_per_rank=%d",
                    self.placement.physical_to_logical.shape[1], self.placement.local_slots,
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
        control[:, 0].fill_(dispatch_id)
        control[:, 1].fill_(int(record and phase == self.controller.phase))
        return DispatchContext(dispatch_id, phase, placement, output, record)

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
            if plan.base_generation != self.placement.generation:
                return None
            logger.info("[eplb] plan ready base_generation=%d planning_ms=%.3f",
                        plan.base_generation, elapsed_ms)
        return plan

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
            logger.info("[eplb] committed generation=%d layers=%s payload_bytes=%d predicted_gain=%.6f "
                        "changed_slots=%d replica_changes=%d remote_bytes=%d elapsed_ms=%.3f",
                        plan.target.generation, [move.layer_id for move in plan.layers],
                        plan.payload_bytes, plan.predicted_gain, changed_slots, replica_changes,
                        remote_slots * self.bank_layout.expert_bytes, (perf_counter() - started) * 1000)
        except BaseException as error:
            if submitted:
                self._failure = error
                self.controller.failed(error)
            raise
