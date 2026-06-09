# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

import torch


class KVBlockLocation(Enum):
    """Residency state for one logical KV cache block."""

    NPU = "npu"
    CPU = "cpu"
    SSD = "ssd"
    MOVING_TO_CPU = "moving_to_cpu"
    MOVING_TO_SSD = "moving_to_ssd"
    MOVING_TO_NPU = "moving_to_npu"
    INVALID = "invalid"


@dataclass(frozen=True)
class OffloadKey:
    """Stable identifier for offloaded KV data, independent of NPU page ids."""

    block_hash: int
    group_id: int = 0
    layout_version: int = 0


@dataclass(frozen=True)
class LoadStoreSpec:
    """Base transfer endpoint descriptor."""

    block_ids: tuple[int, ...]
    medium: str


@dataclass(frozen=True)
class NPULoadStoreSpec(LoadStoreSpec):
    """NPU KV page transfer endpoint."""

    def __init__(self, page_ids: list[int] | tuple[int, ...]) -> None:
        super().__init__(tuple(page_ids), "NPU")


@dataclass(frozen=True)
class SSDLoadStoreSpec(LoadStoreSpec):
    """On-card SSD slot transfer endpoint."""

    def __init__(self, slot_ids: list[int] | tuple[int, ...]) -> None:
        super().__init__(tuple(slot_ids), "SSD")


@dataclass(frozen=True)
class CPULoadStoreSpec(LoadStoreSpec):
    """CPU KV offload slot transfer endpoint."""

    def __init__(self, slot_ids: list[int] | tuple[int, ...]) -> None:
        super().__init__(tuple(slot_ids), "CPU")


@dataclass(frozen=True)
class TransferJob:
    """One asynchronous KV cache transfer job."""

    job_id: int
    request_id: str | None
    src: LoadStoreSpec
    dst: LoadStoreSpec
    keys: tuple[OffloadKey, ...] = ()

    @property
    def direction(self) -> tuple[str, str]:
        """Return the source and destination media for handler dispatch."""
        return (self.src.medium, self.dst.medium)


@dataclass(frozen=True)
class TransferResult:
    """Completion status for one transfer job."""

    job_id: int
    success: bool = True
    error: str | None = None


class KvOffloadBackend(Protocol):
    """Worker-side transfer backend interface."""

    def submit(self, job: TransferJob) -> bool:
        """Submit an asynchronous transfer job."""
        ...

    def poll(self) -> list[TransferResult]:
        """Return transfer jobs completed since the previous poll."""
        ...

    def wait(self, job_ids: set[int]) -> list[TransferResult]:
        """Block until the selected jobs are complete."""
        ...

    def shutdown(self) -> None:
        """Release backend resources."""
        ...


class NoopKvOffloadBackend:
    """Backend used when KV offload is disabled."""

    def submit(self, job: TransferJob) -> bool:
        return False

    def poll(self) -> list[TransferResult]:
        return []

    def wait(self, job_ids: set[int]) -> list[TransferResult]:
        return []

    def shutdown(self) -> None:
        return None


class UnavailableKvOffloadBackend:
    """Backend placeholder for transfer media whose runtime API is not wired yet."""

    def __init__(self, medium: str, *, reason: str | None = None) -> None:
        self.medium = medium
        self.reason = reason or f"{medium} KV offload backend is not configured"
        self.submitted_jobs: dict[int, TransferJob] = {}
        self._finished: dict[int, TransferResult] = {}

    def submit(self, job: TransferJob) -> bool:
        self.submitted_jobs[job.job_id] = job
        self._finished[job.job_id] = TransferResult(
            job_id=job.job_id,
            success=False,
            error=self.reason,
        )
        return False

    def poll(self) -> list[TransferResult]:
        results = list(self._finished.values())
        self._finished.clear()
        return results

    def wait(self, job_ids: set[int]) -> list[TransferResult]:
        results: list[TransferResult] = []
        for job_id in job_ids:
            result = self._finished.pop(job_id, None)
            if result is not None:
                results.append(result)
        return results

    def shutdown(self) -> None:
        self.submitted_jobs.clear()
        self._finished.clear()


class MockKvOffloadBackend:
    """Deterministic in-memory backend for scheduler and state-machine tests."""

    def __init__(self, *, complete_immediately: bool = True) -> None:
        self.complete_immediately = complete_immediately
        self.submitted_jobs: dict[int, TransferJob] = {}
        self._finished: dict[int, TransferResult] = {}
        self._pending: set[int] = set()

    def submit(self, job: TransferJob) -> bool:
        self.submitted_jobs[job.job_id] = job
        if self.complete_immediately:
            self._finished[job.job_id] = TransferResult(job_id=job.job_id)
        else:
            self._pending.add(job.job_id)
        return True

    def complete(self, job_id: int, *, success: bool = True, error: str | None = None) -> None:
        """Mark one pending job complete."""
        if job_id not in self.submitted_jobs:
            raise KeyError(f"Unknown transfer job {job_id}")
        self._pending.discard(job_id)
        self._finished[job_id] = TransferResult(job_id=job_id, success=success, error=error)

    def poll(self) -> list[TransferResult]:
        results = list(self._finished.values())
        self._finished.clear()
        return results

    def wait(self, job_ids: set[int]) -> list[TransferResult]:
        for job_id in list(job_ids):
            if job_id in self._pending:
                self.complete(job_id)
        results: list[TransferResult] = []
        for job_id in job_ids:
            result = self._finished.pop(job_id, None)
            if result is not None:
                results.append(result)
        return results

    def shutdown(self) -> None:
        self.submitted_jobs.clear()
        self._finished.clear()
        self._pending.clear()


@dataclass
class OffloadStats:
    """Counters used by mock and future real backends."""

    stores_submitted: int = 0
    loads_submitted: int = 0
    stores_completed: int = 0
    loads_completed: int = 0
    failures: int = 0
    bytes_moved: int = 0
    extra: dict[str, int] = field(default_factory=dict)


class TorchKVPageView:
    """Canonical page view over torch KV tensors.

    Each component tensor must expose pages on dimension 0. For the serving
    manager this is typically one component per K/V tensor and layer, shaped
    ``[num_pages, ...]``. Transfers copy the flattened bytes of every component
    for each selected page.
    """

    def __init__(self, components: list[torch.Tensor]) -> None:
        if not components:
            raise ValueError("components must not be empty")
        num_pages = int(components[0].shape[0])
        if num_pages <= 0:
            raise ValueError("components must contain at least one page")
        page_bytes = 0
        normalized: list[torch.Tensor] = []
        for component in components:
            if int(component.shape[0]) != num_pages:
                raise ValueError("all components must have the same num_pages")
            if not component.is_contiguous():
                raise ValueError("canonical KV page components must be contiguous")
            normalized.append(component)
            page_bytes += int(component[0].nbytes)
        self.components = normalized
        self.num_pages = num_pages
        self.page_size_bytes = page_bytes

    def copy_page_to(self, page_id: int, dst: torch.Tensor) -> None:
        """Copy one page from this view into one flat uint8 destination row."""
        self._validate_page_id(page_id)
        self._validate_cpu_row(dst)
        offset = 0
        for component in self.components:
            src = component[page_id].view(torch.uint8).reshape(-1)
            end = offset + src.numel()
            dst[offset:end].copy_(src.cpu())
            offset = end

    def copy_page_from(self, page_id: int, src: torch.Tensor) -> None:
        """Copy one flat uint8 source row into one page in this view."""
        self._validate_page_id(page_id)
        self._validate_cpu_row(src)
        offset = 0
        for component in self.components:
            dst = component[page_id].view(torch.uint8).reshape(-1)
            end = offset + dst.numel()
            dst.copy_(src[offset:end].to(dst.device))
            offset = end

    def _validate_page_id(self, page_id: int) -> None:
        if page_id < 0 or page_id >= self.num_pages:
            raise IndexError(f"page_id {page_id} is out of range for {self.num_pages} pages")

    def _validate_cpu_row(self, row: torch.Tensor) -> None:
        if row.dtype != torch.uint8 or row.device.type != "cpu" or row.dim() != 1:
            raise ValueError("CPU offload rows must be 1-D CPU torch.uint8 tensors")
        if row.numel() != self.page_size_bytes:
            raise ValueError(f"CPU offload row has {row.numel()} bytes, expected {self.page_size_bytes}")


class WorkerKVPageView:
    """Canonical page view over runner-owned WorkerTensor KV caches."""

    def __init__(
        self,
        *,
        worker: Any,
        key_pages: Any,
        value_pages: Any,
        num_layers: int,
        num_pages: int,
        num_kv_heads: int,
        page_size: int,
        head_dim: int,
    ) -> None:
        if num_layers <= 0 or num_pages <= 0 or num_kv_heads <= 0 or page_size <= 0 or head_dim <= 0:
            raise ValueError("KV page view dimensions must be positive")
        self.worker = worker
        self.key_pages = key_pages
        self.value_pages = value_pages
        self.num_layers = num_layers
        self.num_pages = num_pages
        self.num_kv_heads = num_kv_heads
        self.page_size = page_size
        self.head_dim = head_dim
        self.rows_per_page = num_kv_heads * page_size
        element_size = torch.empty((), dtype=key_pages.torch_dtype).element_size()
        if torch.empty((), dtype=value_pages.torch_dtype).element_size() != element_size:
            raise ValueError("key_pages and value_pages must use dtypes with the same element size")
        self._element_size = element_size
        self.component_size_bytes = self.rows_per_page * head_dim * element_size
        self.page_size_bytes = 2 * num_layers * self.component_size_bytes

    def copy_page_to(self, page_id: int, dst: torch.Tensor) -> None:
        """Copy one worker-resident KV page into one flat CPU uint8 row."""
        self._validate_page_id(page_id)
        self._validate_cpu_row(dst)
        offset = 0
        for layer_idx in range(self.num_layers):
            for tensor in (self.key_pages, self.value_pages):
                src_ptr = tensor.data_ptr + self._component_offset(layer_idx, page_id)
                end = offset + self.component_size_bytes
                self.worker.copy_from(
                    dst[offset:end].data_ptr(),
                    src_ptr,
                    self.component_size_bytes,
                    worker_id=tensor.worker_id,
                )
                offset = end

    def copy_page_from(self, page_id: int, src: torch.Tensor) -> None:
        """Copy one flat CPU uint8 row into one worker-resident KV page."""
        self._validate_page_id(page_id)
        self._validate_cpu_row(src)
        offset = 0
        for layer_idx in range(self.num_layers):
            for tensor in (self.key_pages, self.value_pages):
                dst_ptr = tensor.data_ptr + self._component_offset(layer_idx, page_id)
                end = offset + self.component_size_bytes
                self.worker.copy_to(
                    dst_ptr,
                    src[offset:end].data_ptr(),
                    self.component_size_bytes,
                    worker_id=tensor.worker_id,
                )
                offset = end

    def _component_offset(self, layer_idx: int, page_id: int) -> int:
        row_offset = (layer_idx * self.num_pages + page_id) * self.rows_per_page
        return row_offset * self.head_dim * self._element_size

    def _validate_page_id(self, page_id: int) -> None:
        if page_id < 0 or page_id >= self.num_pages:
            raise IndexError(f"page_id {page_id} is out of range for {self.num_pages} pages")

    def _validate_cpu_row(self, row: torch.Tensor) -> None:
        if row.dtype != torch.uint8 or row.device.type != "cpu" or row.dim() != 1:
            raise ValueError("CPU offload rows must be 1-D CPU torch.uint8 tensors")
        if row.numel() != self.page_size_bytes:
            raise ValueError(f"CPU offload row has {row.numel()} bytes, expected {self.page_size_bytes}")


class CPUKvOffloadBackend:
    """Synchronous CPU offload backend for canonical torch KV page views."""

    def __init__(self, page_view: Any, *, num_cpu_slots: int) -> None:
        if num_cpu_slots <= 0:
            raise ValueError("num_cpu_slots must be positive")
        self.page_view = page_view
        self.cpu_slots = torch.empty(
            (num_cpu_slots, page_view.page_size_bytes),
            dtype=torch.uint8,
            device="cpu",
        )
        self.submitted_jobs: dict[int, TransferJob] = {}
        self._finished: dict[int, TransferResult] = {}

    @property
    def num_cpu_slots(self) -> int:
        """Return the number of CPU slots currently allocated."""
        return int(self.cpu_slots.shape[0])

    def ensure_num_cpu_slots(self, num_cpu_slots: int) -> None:
        """Grow the CPU slot tensor if a later transfer references more slots."""
        if num_cpu_slots <= self.num_cpu_slots:
            return
        new_slots = torch.empty(
            (num_cpu_slots, self.page_view.page_size_bytes),
            dtype=torch.uint8,
            device="cpu",
        )
        new_slots[: self.num_cpu_slots].copy_(self.cpu_slots)
        self.cpu_slots = new_slots

    def submit(self, job: TransferJob) -> bool:
        self.submitted_jobs[job.job_id] = job
        try:
            if job.direction == ("NPU", "CPU"):
                self._copy_to_cpu(job)
            elif job.direction == ("CPU", "NPU"):
                self._copy_from_cpu(job)
            else:
                raise ValueError(f"unsupported CPU offload transfer direction: {job.direction}")
        except Exception as exc:
            self._finished[job.job_id] = TransferResult(job_id=job.job_id, success=False, error=str(exc))
            return False
        self._finished[job.job_id] = TransferResult(job_id=job.job_id)
        return True

    def poll(self) -> list[TransferResult]:
        results = list(self._finished.values())
        self._finished.clear()
        return results

    def wait(self, job_ids: set[int]) -> list[TransferResult]:
        results: list[TransferResult] = []
        for job_id in job_ids:
            result = self._finished.pop(job_id, None)
            if result is not None:
                results.append(result)
        return results

    def shutdown(self) -> None:
        self.submitted_jobs.clear()
        self._finished.clear()

    def _copy_to_cpu(self, job: TransferJob) -> None:
        if not isinstance(job.src, NPULoadStoreSpec) or not isinstance(job.dst, CPULoadStoreSpec):
            raise TypeError("NPU -> CPU transfers require NPULoadStoreSpec and CPULoadStoreSpec")
        self._validate_same_length(job)
        for page_id, slot_id in zip(job.src.block_ids, job.dst.block_ids):
            self._validate_slot_id(slot_id)
            self.page_view.copy_page_to(page_id, self.cpu_slots[slot_id])

    def _copy_from_cpu(self, job: TransferJob) -> None:
        if not isinstance(job.src, CPULoadStoreSpec) or not isinstance(job.dst, NPULoadStoreSpec):
            raise TypeError("CPU -> NPU transfers require CPULoadStoreSpec and NPULoadStoreSpec")
        self._validate_same_length(job)
        for slot_id, page_id in zip(job.src.block_ids, job.dst.block_ids):
            self._validate_slot_id(slot_id)
            self.page_view.copy_page_from(page_id, self.cpu_slots[slot_id])

    @staticmethod
    def _validate_same_length(job: TransferJob) -> None:
        if len(job.src.block_ids) != len(job.dst.block_ids):
            raise ValueError("source and destination specs must have the same number of blocks")

    def _validate_slot_id(self, slot_id: int) -> None:
        if slot_id < 0 or slot_id >= self.cpu_slots.shape[0]:
            raise IndexError(f"CPU slot {slot_id} is out of range for {self.cpu_slots.shape[0]} slots")
