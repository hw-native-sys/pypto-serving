# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from pypto_serving.config.types import (
    DecodeBatch,
    DecodeResult,
    GenerateConfig,
    GenerateResult,
    ModelRecord,
    PrefillBatch,
    PrefillResult,
    RequestState,
    RuntimeModel,
    SamplingParams,
)
from pypto_serving.serving.memory.kv_cache import KvCacheManager


class ModelExecutor(ABC):
    """Backend-neutral interface used by the serving worker to execute generation."""

    def __init__(self, kv_cache_manager: KvCacheManager | None = None) -> None:
        """Store the KV cache manager shared with the engine (optional for serving path)."""
        self._kv_cache_manager = kv_cache_manager

    @property
    def supports_device_sampling(self) -> bool:
        """Return whether executor results may include already-sampled token IDs."""
        return False

    @property
    def supports_device_stochastic_sampling(self) -> bool:
        """Return whether sampled IDs may use per-request temperature and top-k."""
        return False

    @property
    def device_topk_sampling_k(self) -> int:
        """Return the max top-k candidate width the executor can produce on device."""
        return 0

    @property
    def supports_device_embedding(self) -> bool:
        """Return whether token embedding can be handled inside the device kernels.

        When true, callers may omit prefill and decode hidden states because the
        executor gathers token embeddings from the batch token ids.
        """
        return False

    @property
    def supports_device_decode_embedding(self) -> bool:
        """Return whether decode kernels gather embeddings from token IDs."""
        return self.supports_device_embedding

    @property
    def max_prefill_batch_size(self) -> int | None:
        """Return an executor-specific prefill dispatch limit, if any."""
        return None

    @property
    def max_prefill_batch_size_per_partition(self) -> int:
        """Return the per-cache-partition prefill width for one dispatch."""
        return 1

    @property
    def supports_async_decode_prepare(self) -> bool:
        """Return whether decode metadata can be prepared ahead of execution.

        Implementations advertising this capability must keep the returned
        preparation independent from any decode invocation currently executing.
        The worker may call :meth:`prepare_decode` on its command thread while
        :meth:`run_prepared_decode` is running on the device thread.
        """
        return False

    def prepare_decode(
        self,
        model: RuntimeModel,
        batch: DecodeBatch,
        *,
        buffer_slot: int,
    ) -> object:
        """Prepare one decode execution snapshot without dispatching the model."""
        raise NotImplementedError(f"{type(self).__name__} does not support async decode prepare")

    def run_prepared_decode(
        self,
        model: RuntimeModel,
        batch: DecodeBatch,
        prepared: object,
    ) -> DecodeResult:
        """Late-bind and execute a snapshot returned by :meth:`prepare_decode`."""
        raise NotImplementedError(f"{type(self).__name__} does not support prepared decode")

    @property
    def supports_async_decode_reclaim(self) -> bool:
        """Return whether dispatch and host output reclaim can run independently.

        The device lane calls :meth:`dispatch_prepared_decode`; a separate output
        lane waits for completion through :meth:`reclaim_prepared_decode`.
        Implementations must keep every mutable binding alive until completion
        and every host-visible output alive until reclaim.
        """
        return False

    def dispatch_prepared_decode(
        self,
        model: RuntimeModel,
        batch: DecodeBatch,
        prepared: object,
    ) -> object:
        """Submit device work and return an executor-owned completion/reclaim ticket."""
        raise NotImplementedError(f"{type(self).__name__} does not support split decode dispatch")

    def reclaim_prepared_decode(self, pending: object) -> DecodeResult:
        """Materialize host output from a completed dispatch ticket."""
        raise NotImplementedError(f"{type(self).__name__} does not support split decode reclaim")

    def prepared_decode_requires_token(self, prepared: object) -> bool:
        """Return whether dispatch still needs the prior host-sampled token."""
        return True

    def lookup_embeddings(self, model: RuntimeModel, token_ids: torch.Tensor) -> torch.Tensor:
        """Return embedding rows for ``token_ids`` on the model runtime device."""
        token_ids = token_ids.to(device=model.runtime.device, dtype=torch.long)
        return model.embed_tokens.index_select(0, token_ids.view(-1)).view(
            *token_ids.shape,
            model.config.hidden_size,
        )

    def validate_generate_batch(
        self,
        record: ModelRecord,
        batch_size: int,
        config: GenerateConfig,
    ) -> None:
        """Validate executor-specific limits before KV allocation begins."""
        return None

    def prompt_allocation_length(
        self,
        record: ModelRecord,
        prompt_len: int,
        config: GenerateConfig,
    ) -> int:
        """Return the initial KV allocation size for one prompt."""
        return prompt_len

    def try_generate_batch(
        self,
        record: ModelRecord,
        requests: list[RequestState],
        prefill_batch: PrefillBatch,
        config: GenerateConfig,
    ) -> list[GenerateResult] | None:
        """Optionally handle generation with an executor-specific fast path."""
        return None

    @abstractmethod
    def run_prefill(self, model: RuntimeModel, batch: PrefillBatch) -> PrefillResult:
        """Run prompt prefill and return logits for the next token."""
        raise NotImplementedError

    def finalize_prefill(
        self,
        model: RuntimeModel,
        request_ids: list[str],
        sampled_token_ids: list[int],
        sampling_params: list[SamplingParams] | None = None,
    ) -> None:
        """Finalize model-specific state after terminal-prefill sampling.

        This hook remains part of the prefill command. Model integrations may
        use the sampled first output token to seed decode-only persistent state
        before the command is made visible to the scheduler.
        """
        return None

    @abstractmethod
    def run_decode(self, model: RuntimeModel, batch: DecodeBatch) -> DecodeResult:
        """Run one decode step for active requests and return next-token logits."""
        raise NotImplementedError
