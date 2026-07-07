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
    PrefillBatch,
    PrefillResult,
    RuntimeConfig,
    RuntimeModel,
)


@dataclass
class _KvCachePool:
    """Worker-resident flat all-layer KV cache for one model.

    For TurboQuant (TQ) mode the quantized caches and scales are populated
    instead of (or in addition to) the BF16 key_pages / value_pages.
    """

    key_pages: DeviceTensor | None = None
    value_pages: DeviceTensor | None = None
    # TurboQuant compressed caches.
    quant_k_pages: DeviceTensor | None = None
    quant_v_pages: DeviceTensor | None = None
    k_scales_pages: DeviceTensor | None = None
    v_scales_pages: DeviceTensor | None = None


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
            num_pages = runtime.max_batch_size * max_blocks_per_seq
            num_pages = runtime.total_kv_pages
            if num_pages is None:
                num_pages = runtime.max_batch_size * max_blocks_per_seq
        kv_dtype = getattr(torch, runtime.kv_dtype)
        cache_rows = config.num_hidden_layers * num_pages * config.num_key_value_heads * runtime.page_size
        cache_shape = (
            cache_rows,
            config.head_dim,
        )

        qcfg = runtime.kv_quant_config
        if qcfg is not None and qcfg.enabled:
            # TurboQuant: UINT8 quantized K/V caches + FP32 per-row scales,
            # allocated in place of the BF16 key/value pages.
            scales_shape = (cache_rows, 1)
            quant_k = self._alloc_kv_cache_tensor(cache_shape, torch.uint8)
            try:
                quant_v = self._alloc_kv_cache_tensor(cache_shape, torch.uint8)
            except Exception:
                self._free_kv_cache_tensor(quant_k)
                raise
            try:
                k_scales = self._alloc_kv_cache_tensor(scales_shape, torch.float32)
            except Exception:
                self._free_kv_cache_tensor(quant_k)
                self._free_kv_cache_tensor(quant_v)
                raise
            try:
                v_scales = self._alloc_kv_cache_tensor(scales_shape, torch.float32)
            except Exception:
                self._free_kv_cache_tensor(quant_k)
                self._free_kv_cache_tensor(quant_v)
                self._free_kv_cache_tensor(k_scales)
                raise
            self._kv_caches[model_id] = _KvCachePool(
                quant_k_pages=quant_k,
                quant_v_pages=quant_v,
                k_scales_pages=k_scales,
                v_scales_pages=v_scales,
            )
            return

        kv_dtype = getattr(torch, runtime.kv_dtype)
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
            if pool.key_pages is not None:
                self._free_kv_cache_tensor(pool.key_pages)
            if pool.value_pages is not None:
                self._free_kv_cache_tensor(pool.value_pages)
            if pool.quant_k_pages is not None:
                self._free_kv_cache_tensor(pool.quant_k_pages)
            if pool.quant_v_pages is not None:
                self._free_kv_cache_tensor(pool.quant_v_pages)
            if pool.k_scales_pages is not None:
                self._free_kv_cache_tensor(pool.k_scales_pages)
            if pool.v_scales_pages is not None:
                self._free_kv_cache_tensor(pool.v_scales_pages)
        self._kv_caches.clear()

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
