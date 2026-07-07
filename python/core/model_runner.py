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

from pypto.runtime import DeviceTensor

from .types import (
    DecodeBatch,
    DecodeResult,
    ModelConfig,
    ModelRecord,
    PrefillBatch,
    PrefillResult,
    RuntimeConfig,
    RuntimeModel,
)


@dataclass
class _KvCachePool:
    """Worker-resident flat all-layer KV cache for one model."""

    key_pages: DeviceTensor
    value_pages: DeviceTensor


class ModelRunner(ABC):
    """Runtime interface for compiled kernels registered to one model."""

    def __init__(self) -> None:
        self._kv_caches: dict[str, _KvCachePool] = {}

    def init_kv_cache(
        self, model_id: str, config: ModelConfig, runtime: RuntimeConfig,
        *, num_pages: int | None = None,
    ) -> int:
        """Create the paged KV cache directly in runner-owned device memory.

        When *num_pages* is ``None`` (default), the total page count is derived
        from ``runtime.total_kv_pages`` or the static capacity formula.  NPU
        runners should pass an explicit value computed from available device
        memory after model weights have been uploaded.

        Returns the number of pages allocated.
        """
        if model_id in self._kv_caches:
            cache_rows = self._kv_caches[model_id].key_pages.shape[0]
            pages = cache_rows // (
                config.num_hidden_layers * config.num_key_value_heads * runtime.page_size
            )
            return pages
        max_blocks_per_seq = math.ceil(runtime.max_seq_len / runtime.page_size)
        if num_pages is None:
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
        )
        return num_pages

    def close_kv_cache(self) -> None:
        """Release all runner-owned KV cache tensors."""
        for pool in list(self._kv_caches.values()):
            self._free_kv_cache_tensor(pool.key_pages)
            self._free_kv_cache_tensor(pool.value_pages)
        self._kv_caches.clear()

    def preflight(self, record: ModelRecord) -> None:
        """Eagerly materialize weights and shared buffers before the worker signals ready.

        Called once by ``PyptoExecutor.register_model`` after ``init_kv_cache``,
        before the serving worker sets its ready event. Runners whose weights are
        already loaded by the model loader (e.g. HuggingFace eager load) inherit
        this no-op; runners that defer weight reads to first inference override
        it so the serving "ready" contract means weights are fully loaded.
        """
        del record

    @abstractmethod
    def _alloc_kv_cache_tensor(self, shape: tuple[int, ...], dtype: torch.dtype) -> DeviceTensor:
        """Allocate one worker-resident KV cache tensor."""
        raise NotImplementedError

    @abstractmethod
    def _free_kv_cache_tensor(self, tensor: DeviceTensor) -> None:
        """Free one worker-resident KV cache tensor."""
        raise NotImplementedError

    def warmup(self, model: RuntimeModel) -> None:
        """Run a minimal prefill + decode to warm up device kernels.

        The default implementation is a no-op.  NPU runners should override
        this to dispatch one dummy prefill and one dummy decode so that all
        device kernels are compiled and cached before the first real request.
        """
        del model  # unused in the default no-op

    @abstractmethod
    def run_prefill(self, model: RuntimeModel, batch: PrefillBatch) -> PrefillResult:
        """Run the compiled prefill path for one batch."""
        raise NotImplementedError

    @abstractmethod
    def run_decode(self, model: RuntimeModel, batch: DecodeBatch) -> DecodeResult:
        """Run the compiled decode path for one batch."""
        raise NotImplementedError
