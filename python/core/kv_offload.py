# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import itertools
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import torch
from pypto.runtime import DeviceTensor


class KVBlockLocation(Enum):
    """Current residency of one logical KV block."""

    NPU = auto()
    CPU = auto()
    TRANSFERRING = auto()


@dataclass(frozen=True)
class NPULoadStoreSpec:
    """A list of physical NPU page IDs used as a transfer endpoint."""

    page_ids: list[int]


@dataclass(frozen=True)
class CPULoadStoreSpec:
    """A list of CPU offload slot IDs used as a transfer endpoint."""

    slot_ids: list[int]


@dataclass(frozen=True)
class TransferJob:
    """One page-level KV transfer between NPU cache pages and CPU slots."""

    job_id: int
    request_id: str | None
    src: NPULoadStoreSpec | CPULoadStoreSpec
    dst: NPULoadStoreSpec | CPULoadStoreSpec


@dataclass(frozen=True)
class TransferResult:
    """Completion status for one KV transfer job."""

    job_id: int
    request_id: str | None
    src: NPULoadStoreSpec | CPULoadStoreSpec
    dst: NPULoadStoreSpec | CPULoadStoreSpec
    success: bool
    error: str | None = None


class WorkerKVPageView:
    """Byte-level view over runner-owned paged K/V DeviceTensors.

    The runner stores KV as a flat view of
    ``[num_layers, num_pages, num_kv_heads, page_size, head_dim]``. A logical
    page is not one contiguous range across layers, so copies are split into
    one contiguous K segment and one contiguous V segment per layer.
    """

    def __init__(
        self,
        *,
        worker: Any,
        key_pages: DeviceTensor,
        value_pages: DeviceTensor,
        num_layers: int,
        num_pages: int,
        num_kv_heads: int,
        page_size: int,
        head_dim: int,
    ) -> None:
        self.worker = worker
        self.key_pages = key_pages
        self.value_pages = value_pages
        self.num_layers = int(num_layers)
        self.num_pages = int(num_pages)
        self.num_kv_heads = int(num_kv_heads)
        self.page_size = int(page_size)
        self.head_dim = int(head_dim)
        self.dtype = key_pages.dtype
        if value_pages.dtype != self.dtype:
            raise ValueError("key_pages and value_pages must have the same dtype")
        expected_shape = (
            self.num_layers * self.num_pages * self.num_kv_heads * self.page_size,
            self.head_dim,
        )
        if tuple(key_pages.shape) != expected_shape or tuple(value_pages.shape) != expected_shape:
            raise ValueError(
                "KV page tensors do not match the expected flat paged layout: "
                f"expected={expected_shape}, key={key_pages.shape}, value={value_pages.shape}"
            )

    @property
    def _element_size(self) -> int:
        return torch.empty((), dtype=self.dtype).element_size()

    @property
    def _layer_page_bytes(self) -> int:
        return self.num_kv_heads * self.page_size * self.head_dim * self._element_size

    @property
    def page_size_bytes(self) -> int:
        """Total bytes for one logical page, including K and V for all layers."""

        return 2 * self.num_layers * self._layer_page_bytes

    def copy_page_to(self, page_id: int, dst: torch.Tensor) -> None:
        """Copy one NPU page into a contiguous uint8 CPU tensor."""

        dst_u8 = self._validate_host_buffer(dst)
        offset = 0
        for tensor in (self.key_pages, self.value_pages):
            for layer_idx in range(self.num_layers):
                self.worker.copy_from(
                    dst_u8.data_ptr() + offset,
                    self._device_segment_ptr(tensor, page_id, layer_idx),
                    self._layer_page_bytes,
                )
                offset += self._layer_page_bytes

    def copy_page_from(self, page_id: int, src: torch.Tensor) -> None:
        """Copy one contiguous uint8 CPU tensor into an NPU page."""

        src_u8 = self._validate_host_buffer(src)
        offset = 0
        for tensor in (self.key_pages, self.value_pages):
            for layer_idx in range(self.num_layers):
                self.worker.copy_to(
                    self._device_segment_ptr(tensor, page_id, layer_idx),
                    src_u8.data_ptr() + offset,
                    self._layer_page_bytes,
                )
                offset += self._layer_page_bytes

    def _device_segment_ptr(self, tensor: DeviceTensor, page_id: int, layer_idx: int) -> int:
        if page_id < 0 or page_id >= self.num_pages:
            raise ValueError(f"page_id {page_id} is outside [0, {self.num_pages})")
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"layer_idx {layer_idx} is outside [0, {self.num_layers})")
        row = (layer_idx * self.num_pages + page_id) * self.num_kv_heads * self.page_size
        return tensor.data_ptr + row * self.head_dim * self._element_size

    def _validate_host_buffer(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device.type != "cpu":
            raise ValueError("KV offload host buffer must be on CPU")
        if tensor.dtype != torch.uint8:
            raise ValueError("KV offload host buffer must be torch.uint8")
        if not tensor.is_contiguous():
            raise ValueError("KV offload host buffer must be contiguous")
        if tensor.numel() != self.page_size_bytes:
            raise ValueError(
                f"KV offload host buffer has {tensor.numel()} bytes, expected {self.page_size_bytes}"
            )
        return tensor


class CPUKvOffloadBackend:
    """Synchronous CPU offload backend for runner-owned NPU KV pages."""

    def __init__(
        self,
        page_view: WorkerKVPageView,
        *,
        num_cpu_slots: int,
        cpu_slots: torch.Tensor | None = None,
    ) -> None:
        self.page_view = page_view
        if cpu_slots is None:
            self.cpu_slots = torch.empty(
                (int(num_cpu_slots), page_view.page_size_bytes),
                dtype=torch.uint8,
                device="cpu",
            ).share_memory_()
        else:
            if tuple(cpu_slots.shape) != (int(num_cpu_slots), page_view.page_size_bytes):
                raise ValueError(
                    "Preallocated CPU KV slots have wrong shape: "
                    f"got={tuple(cpu_slots.shape)}, expected={(int(num_cpu_slots), page_view.page_size_bytes)}"
                )
            if cpu_slots.dtype != torch.uint8 or cpu_slots.device.type != "cpu" or not cpu_slots.is_contiguous():
                raise ValueError("Preallocated CPU KV slots must be contiguous CPU uint8")
            if not cpu_slots.is_shared():
                raise ValueError("Preallocated CPU KV slots must be in shared memory")
            self.cpu_slots = cpu_slots
        self._completed: dict[int, TransferResult] = {}
        self._job_ids = itertools.count()

    def ensure_num_cpu_slots(self, num_cpu_slots: int) -> None:
        """Validate that the fixed shared CPU slot pool is large enough."""

        num_cpu_slots = int(num_cpu_slots)
        if num_cpu_slots <= self.cpu_slots.shape[0]:
            return
        raise RuntimeError(
            "CPU KV offload slots cannot be resized after initialization: "
            f"available={self.cpu_slots.shape[0]}, requested={num_cpu_slots}"
        )

    def next_job_id(self) -> int:
        """Return a backend-local monotonically increasing job ID."""

        return next(self._job_ids)

    def submit(self, job: TransferJob) -> bool:
        """Run one transfer synchronously and record its result."""

        try:
            if isinstance(job.src, NPULoadStoreSpec) and isinstance(job.dst, CPULoadStoreSpec):
                self._copy_npu_to_cpu(job.src.page_ids, job.dst.slot_ids)
            elif isinstance(job.src, CPULoadStoreSpec) and isinstance(job.dst, NPULoadStoreSpec):
                self._copy_cpu_to_npu(job.src.slot_ids, job.dst.page_ids)
            else:
                raise TypeError("KV offload only supports NPU<->CPU transfers")
        except Exception as exc:
            self._completed[job.job_id] = TransferResult(
                job.job_id,
                job.request_id,
                job.src,
                job.dst,
                False,
                str(exc),
            )
            return False
        self._completed[job.job_id] = TransferResult(
            job.job_id,
            job.request_id,
            job.src,
            job.dst,
            True,
        )
        return True

    def wait(self, job_ids: set[int]) -> list[TransferResult]:
        """Return completed results for the requested job IDs."""

        missing = job_ids - self._completed.keys()
        if missing:
            raise RuntimeError(f"Unknown or incomplete KV offload job IDs: {sorted(missing)}")
        return [self._completed.pop(job_id) for job_id in job_ids]

    def _copy_npu_to_cpu(self, page_ids: list[int], slot_ids: list[int]) -> None:
        if len(page_ids) != len(slot_ids):
            raise ValueError("NPU page ids and CPU slot ids must have the same length")
        self.ensure_num_cpu_slots(max(slot_ids, default=-1) + 1)
        for page_id, slot_id in zip(page_ids, slot_ids, strict=True):
            self.page_view.copy_page_to(page_id, self.cpu_slots[slot_id])

    def _copy_cpu_to_npu(self, slot_ids: list[int], page_ids: list[int]) -> None:
        if len(slot_ids) != len(page_ids):
            raise ValueError("CPU slot ids and NPU page ids must have the same length")
        self.ensure_num_cpu_slots(max(slot_ids, default=-1) + 1)
        for slot_id, page_id in zip(slot_ids, page_ids, strict=True):
            self.page_view.copy_page_from(page_id, self.cpu_slots[slot_id])
