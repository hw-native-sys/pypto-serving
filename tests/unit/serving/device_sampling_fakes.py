# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch

from pypto_serving.config.types import (
    DecodeBatch,
    DecodeResult,
    ModelConfig,
    PrefillBatch,
    PrefillResult,
    RuntimeConfig,
    RuntimeModel,
    SamplingCandidates,
    SamplingParams,
)
from pypto_serving.model.common.executor.executor import ModelExecutor
from pypto_serving.serving.memory.kv_cache import KvCacheManager


class _Tokenizer:
    def encode(self, text: str) -> list[int]:
        return [max(1, len(text))]

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(token_id) for token_id in token_ids)


class _VariableLengthTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(range(1, len(text) + 1))

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(token_id) for token_id in token_ids)


def _model(
    max_batch_size: int,
    max_seq_len: int = 128,
    page_size: int = 64,
    eos_token_id: int | None = None,
    max_num_batched_tokens: int = 4096,
) -> RuntimeModel:
    config = ModelConfig(
        model_id="test-model",
        architecture="qwen3",
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=max_seq_len,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        bos_token_id=None,
        eos_token_id=eos_token_id,
        pad_token_id=None,
        torch_dtype="bfloat16",
    )
    runtime = RuntimeConfig(
        page_size=page_size,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        device="cpu",
        max_num_batched_tokens=max_num_batched_tokens,
    )
    return RuntimeModel(
        config=config,
        runtime=runtime,
        embed_tokens=torch.zeros(config.vocab_size, config.hidden_size),
        final_norm_weight=torch.ones(config.hidden_size),
        lm_head=torch.zeros(config.vocab_size, config.hidden_size),
    )


class _ImmediateEosExecutor(ModelExecutor):
    def __init__(self, kv_cache_manager: KvCacheManager) -> None:
        super().__init__(kv_cache_manager)
        self.prefill_batches: list[PrefillBatch] = []
        self.finalized_prefills: list[tuple[list[str], list[int]]] = []
        self.embedding_lookup_shapes: list[tuple[int, ...]] = []

    def lookup_embeddings(self, model: RuntimeModel, token_ids: torch.Tensor) -> torch.Tensor:
        self.embedding_lookup_shapes.append(tuple(token_ids.shape))
        return super().lookup_embeddings(model, token_ids)

    def run_prefill(self, model: RuntimeModel, batch: PrefillBatch) -> PrefillResult:
        self.prefill_batches.append(batch)
        logits = torch.full((len(batch.request_ids), model.config.vocab_size), -1.0)
        logits[:, 0] = 1.0
        hidden = torch.zeros(len(batch.request_ids), model.config.hidden_size)
        return PrefillResult(last_hidden=hidden, logits=logits)

    def finalize_prefill(
        self,
        model: RuntimeModel,
        request_ids: list[str],
        sampled_token_ids: list[int],
        sampling_params: list[SamplingParams] | None = None,
    ) -> None:
        del sampling_params
        self.finalized_prefills.append((list(request_ids), list(sampled_token_ids)))

    def run_decode(self, model: RuntimeModel, batch: DecodeBatch) -> DecodeResult:
        logits = torch.full((len(batch.request_ids), model.config.vocab_size), -1.0)
        logits[:, 0] = 1.0
        hidden = torch.zeros(len(batch.request_ids), model.config.hidden_size)
        return DecodeResult(hidden_states=hidden, logits=logits)


class _FailingSampler:
    def __init__(self) -> None:
        self.sample_calls = 0

    def from_generate_config(self, config):
        return None

    def sample(self, logits, params) -> int:
        self.sample_calls += 1
        raise AssertionError("host sampler should not be used when device sampled ids are available")


class _FixedSampler:
    def __init__(self, token_id: int) -> None:
        self.token_id = token_id
        self.sample_calls = 0

    def from_generate_config(self, config):
        return None

    def sample(self, logits, params) -> int:
        self.sample_calls += 1
        return self.token_id


class _CandidateSampler(_FixedSampler):
    def __init__(self, token_id: int) -> None:
        super().__init__(token_id)
        self.candidate_calls = 0

    def sample_from_candidates(self, candidates, row_idx, params) -> int:
        self.candidate_calls += 1
        return self.token_id


class _RoutingSampler(_FixedSampler):
    def __init__(self, host_token_id: int, candidate_token_id: int) -> None:
        super().__init__(host_token_id)
        self.candidate_token_id = candidate_token_id
        self.candidate_calls = 0

    def sample_from_candidates(self, candidates, row_idx, params) -> int:
        self.candidate_calls += 1
        return self.candidate_token_id


class _DeviceSamplingExecutor(ModelExecutor):
    def __init__(
        self,
        kv_cache_manager: KvCacheManager,
        *,
        first_token: int,
        second_token: int,
        return_next_hidden: bool = True,
    ) -> None:
        super().__init__(kv_cache_manager)
        self.first_token = first_token
        self.second_token = second_token
        self.return_next_hidden = return_next_hidden
        self.prefill_calls = 0
        self.decode_calls = 0
        self.lookup_calls = 0
        self.decode_hidden_seen: list[torch.Tensor | None] = []

    @property
    def supports_device_sampling(self) -> bool:
        return True

    @property
    def supports_device_embedding(self) -> bool:
        return True

    def lookup_embeddings(self, model: RuntimeModel, token_ids: torch.Tensor) -> torch.Tensor:
        self.lookup_calls += 1
        raise AssertionError("device-embedding prefill/decode should not use host lookup")

    def run_prefill(self, model: RuntimeModel, batch: PrefillBatch) -> PrefillResult:
        self.prefill_calls += 1
        assert batch.input_embeddings is None
        token = torch.tensor([self.first_token], dtype=torch.int64)
        return PrefillResult(
            last_hidden=None,
            logits=torch.zeros(1, model.config.vocab_size),
            sampled_token_ids=token.to(torch.int32),
            next_hidden_states=model.embed_tokens.index_select(0, token) if self.return_next_hidden else None,
        )

    def run_decode(self, model: RuntimeModel, batch: DecodeBatch) -> DecodeResult:
        self.decode_calls += 1
        hidden = batch.hidden_states
        self.decode_hidden_seen.append(None if hidden is None else hidden[0].detach().clone())
        token = torch.tensor([self.second_token], dtype=torch.int64)
        return DecodeResult(
            hidden_states=batch.hidden_states,
            logits=torch.zeros(1, model.config.vocab_size),
            sampled_token_ids=token.to(torch.int32),
            next_hidden_states=model.embed_tokens.index_select(0, token) if self.return_next_hidden else None,
        )


class _DeviceTopkExecutor(ModelExecutor):
    def __init__(
        self,
        kv_cache_manager: KvCacheManager,
        token_id: int = 7,
        *,
        always_return_candidates: bool = False,
    ) -> None:
        super().__init__(kv_cache_manager)
        self.token_id = token_id
        self.always_return_candidates = always_return_candidates
        self.prefill_allow_topk = False
        self.decode_allow_topk = False

    @property
    def device_topk_sampling_k(self) -> int:
        return 4

    @property
    def supports_device_embedding(self) -> bool:
        return True

    def _result_tensors(
        self,
        batch_size: int,
        vocab_size: int,
        allow_device_topk_sampling: bool,
    ) -> tuple[torch.Tensor, SamplingCandidates | None]:
        logits = torch.zeros(batch_size, vocab_size)
        candidates = None
        if allow_device_topk_sampling or self.always_return_candidates:
            values = torch.tensor([[4.0, 3.0, 2.0, 1.0]], dtype=torch.float32)
            token_ids = torch.tensor(
                [[self.token_id, 6, 5, 4]],
                dtype=torch.int32,
            )
            candidates = SamplingCandidates(
                values=values.expand(batch_size, -1).clone(),
                token_ids=token_ids.expand(batch_size, -1).clone(),
            )
        if allow_device_topk_sampling:
            logits = torch.empty(batch_size, 0)
        return logits, candidates

    def run_prefill(self, model: RuntimeModel, batch: PrefillBatch) -> PrefillResult:
        self.prefill_allow_topk = batch.allow_device_topk_sampling
        logits, candidates = self._result_tensors(
            len(batch.request_ids),
            model.config.vocab_size,
            batch.allow_device_topk_sampling,
        )
        return PrefillResult(
            last_hidden=None,
            logits=logits,
            sampling_candidates=candidates,
        )

    def run_decode(self, model: RuntimeModel, batch: DecodeBatch) -> DecodeResult:
        self.decode_allow_topk = batch.allow_device_topk_sampling
        logits, candidates = self._result_tensors(
            len(batch.request_ids),
            model.config.vocab_size,
            batch.allow_device_topk_sampling,
        )
        return DecodeResult(
            hidden_states=batch.hidden_states,
            logits=logits,
            sampling_candidates=candidates,
        )
