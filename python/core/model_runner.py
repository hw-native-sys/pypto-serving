# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from python.runtime.worker import WorkerTensor

from .kv_offload import WorkerKVPageView
from .types import (
    DecodeBatch,
    DecodeResult,
    ModelConfig,
    PrefillBatch,
    PrefillResult,
    RuntimeConfig,
    RuntimeModel,
)


@dataclass
class _KvCachePool:
    """Worker-resident flat all-layer KV cache for one model."""

    key_pages: WorkerTensor
    value_pages: WorkerTensor
    num_layers: int
    num_pages: int
    num_kv_heads: int
    page_size: int
    head_dim: int


class ModelRunner(ABC):
    """Runtime interface for compiled kernels registered to one model."""

    def __init__(self) -> None:
        self._kv_caches: dict[str, _KvCachePool] = {}

    def init_kv_cache(
        self, model_id: str, config: ModelConfig, runtime: RuntimeConfig
    ) -> None:
        """Create the paged KV cache directly in runner-owned device memory."""
        if model_id in self._kv_caches:
            return
        max_blocks_per_seq = math.ceil(runtime.max_seq_len / runtime.page_size)
        num_pages = runtime.total_kv_pages
        if num_pages is None:
            num_pages = runtime.max_batch_size * max_blocks_per_seq
        kv_dtype = getattr(torch, runtime.kv_dtype)
        cache_rows = config.num_hidden_layers * num_pages * config.num_key_value_heads * runtime.page_size
        cache_shape = (
            cache_rows,
            config.head_dim,
        )
        key_pages = self._alloc_kv_cache_tensor(cache_shape, kv_dtype)
        try:
            value_pages = self._alloc_kv_cache_tensor(cache_shape, kv_dtype)
        except Exception:
            self._free_kv_cache_tensor(key_pages)
            raise
        self._kv_caches[model_id] = _KvCachePool(
            key_pages=key_pages,
            value_pages=value_pages,
            num_layers=config.num_hidden_layers,
            num_pages=num_pages,
            num_kv_heads=config.num_key_value_heads,
            page_size=runtime.page_size,
            head_dim=config.head_dim,
        )

    def close_kv_cache(self) -> None:
        """Release all runner-owned KV cache tensors."""
        for pool in list(self._kv_caches.values()):
            self._free_kv_cache_tensor(pool.key_pages)
            self._free_kv_cache_tensor(pool.value_pages)
        self._kv_caches.clear()

    def materialize_worker_page_view(self, model_id: str, worker) -> WorkerKVPageView:
        """Return a page-contiguous worker KV view for CPU offload transfers."""
        if model_id not in self._kv_caches:
            raise KeyError(f"KV cache for model {model_id!r} is not initialized")
        pool = self._kv_caches[model_id]
        return WorkerKVPageView(
            worker=worker,
            key_pages=pool.key_pages,
            value_pages=pool.value_pages,
            num_layers=pool.num_layers,
            num_pages=pool.num_pages,
            num_kv_heads=pool.num_kv_heads,
            page_size=pool.page_size,
            head_dim=pool.head_dim,
        )

    @abstractmethod
    def _alloc_kv_cache_tensor(self, shape: tuple[int, ...], dtype: torch.dtype) -> WorkerTensor:
        """Allocate one worker-resident KV cache tensor."""
        raise NotImplementedError

    @abstractmethod
    def _free_kv_cache_tensor(self, tensor: WorkerTensor) -> None:
        """Free one worker-resident KV cache tensor."""
        raise NotImplementedError

    @abstractmethod
    def run_prefill(self, model: RuntimeModel, batch: PrefillBatch) -> PrefillResult:
        """Run the compiled prefill path for one batch."""
        raise NotImplementedError

    @abstractmethod
    def run_decode(self, model: RuntimeModel, batch: DecodeBatch) -> DecodeResult:
        """Run the compiled decode path for one batch."""
        raise NotImplementedError
