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
from typing import Literal

import torch

from pypto_serving.model.tokenizer import TokenizerAdapter


@dataclass(frozen=True)
class GenerateConfig:
    """User-facing options that control text generation."""

    max_new_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 1.0
    top_k: int | None = None
    seed: int | None = None
    stop: tuple[str, ...] = ()
    stream: bool = False
    ignore_eos: bool = False


@dataclass(frozen=True)
class ModelConfig:
    """Static architecture metadata parsed from model config."""

    model_id: str
    architecture: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    bos_token_id: int | None
    eos_token_id: int | None
    pad_token_id: int | None
    torch_dtype: str


@dataclass(frozen=True)
class KVCacheSpec:
    """Source-token and physical-storage layout of one cache block."""

    block_size: int
    page_size_bytes: int
    compress_ratio: int = 1

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError("KV cache block_size must be positive")
        if self.page_size_bytes <= 0:
            raise ValueError("KV cache page_size_bytes must be positive")
        if self.compress_ratio <= 0:
            raise ValueError("KV cache compress_ratio must be positive")
        if self.block_size % self.compress_ratio:
            raise ValueError("KV cache block_size must be divisible by compress_ratio")

    @property
    def token_capacity(self) -> int:
        """Return the number of source tokens represented by one block."""
        return self.block_size

    @property
    def storage_block_size(self) -> int:
        """Return the number of physical rows stored for one source-token block."""
        return self.block_size // self.compress_ratio


@dataclass(frozen=True)
class KVCacheGroupSpec:
    """Describe one independently allocated model-specific cache family."""

    name: str
    layer_indices: tuple[int, ...]
    spec: KVCacheSpec
    max_blocks_per_seq: int
    # Fixed kernel layouts may expose a physical pool smaller than the serving
    # configuration's generic max batch size. In that case this is the source
    # of truth for scheduler-visible block IDs in one cache partition.
    num_blocks: int | None = None
    # Some distributed kernels combine rank-local data parallelism with expert
    # parallelism. Each rank then owns an independent cache namespace whose
    # physical block IDs start at zero. ``num_partitions`` describes those
    # namespaces without flattening them into one conflicting block-ID space.
    num_partitions: int = 1
    # Source-token tail needed to resume a rolling cache group. ``None``
    # denotes a full-history group whose pages are append-only. Physical ring
    # capacity remains bounded independently by ``max_blocks_per_seq``.
    sliding_window: int | None = None
    # EAGLE/MTP cache groups are shifted by one token: the last KV row in a
    # matched page depends on the first token after that page. Its cache hash
    # includes that boundary token, and publication waits until it is known.
    is_eagle_group: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("KV cache group name must not be empty")
        if self.max_blocks_per_seq <= 0:
            raise ValueError("KV cache max_blocks_per_seq must be positive")
        if self.num_blocks is not None and self.num_blocks <= 0:
            raise ValueError("KV cache num_blocks must be positive when specified")
        if self.num_partitions <= 0:
            raise ValueError("KV cache num_partitions must be positive")
        if self.sliding_window is not None:
            if self.sliding_window <= 0:
                raise ValueError("KV cache sliding_window must be positive")
            if self.sliding_window % self.spec.token_capacity:
                raise ValueError(
                    "KV cache sliding_window must be a multiple of the block token capacity"
                )
            if self.sliding_window // self.spec.token_capacity > self.max_blocks_per_seq:
                raise ValueError(
                    "KV cache sliding_window requires more blocks than max_blocks_per_seq"
                )


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime limits and device placement for one loaded model."""

    page_size: int = 64
    max_batch_size: int = 1
    max_seq_len: int = 4096
    # Host-side tensor placement.  NPU executors manage device memory through
    # the DistributedWorker internally — keep this as ``"cpu"``.
    device: str = "cpu"
    kv_dtype: str = "bfloat16"
    weight_dtype: str = "bfloat16"
    total_kv_pages: int | None = None
    # Fraction of total NPU HBM the server is allowed to use (weights + activations + KV).
    npu_memory_utilization: float = 0.90
    # Max tokens processed per scheduling step (chunked-prefill granularity).
    max_num_batched_tokens: int = 4096
    # Optional model/kernel limit for one request's active prefill tokens in a
    # single dispatch. ``None`` means the model has no stricter per-request
    # limit than ``max_num_batched_tokens``.
    max_prefill_tokens_per_request: int | None = None
    # Whether speculative decoding can safely consume a prompt produced by
    # more than one prefill dispatch.
    supports_chunked_prefill_with_speculation: bool = True
    # Compile-time generation limit used by model-specific runners.
    max_new_tokens: int = 256
    # Extra cache positions and execution tokens reserved for speculative decode.
    num_speculative_tokens: int = 0
    # Model-specific cache families. Empty means the generic single KV pool.
    kv_cache_groups: tuple[KVCacheGroupSpec, ...] = ()
    # Simpler ring sizing forwarded per-dispatch through pypto's RunConfig
    # (CallConfig.runtime_env) on every l3_worker.run/submit — replacing the
    # process-wide PTO2_RING_* envs, which sized rings for every worker in the
    # process, not just L3 dispatches. Each field takes a scalar (broadcast to
    # all scope-depth rings) or a list of exactly 4 ints sizing rings 0..3;
    # ``None`` leaves the field unset so the pypto runtime's own default
    # applies.
    ring_dep_pool: int | list[int] | None = None
    ring_task_window: int | list[int] | None = None
    ring_heap: int | list[int] | None = None


@dataclass(frozen=True)
class LayerSpec:
    """Shape metadata for one transformer layer."""

    layer_idx: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int


@dataclass
class RuntimeModel:
    """Loaded model tensors plus runtime and architecture metadata."""

    config: ModelConfig
    runtime: RuntimeConfig
    embed_tokens: torch.Tensor
    final_norm_weight: torch.Tensor
    lm_head: torch.Tensor
    extra: dict[str, object] = field(default_factory=dict)


@dataclass
class ModelRecord:
    """Engine registry entry for one initialized model."""

    config: ModelConfig
    runtime: RuntimeConfig
    tokenizer: TokenizerAdapter
    layer_specs: list[LayerSpec]
    runtime_model: RuntimeModel


@dataclass
class LoadedModel:
    """Model-loader result before registration with the engine."""

    model_id: str
    model_dir: str
    config: ModelConfig
    tokenizer: TokenizerAdapter
    layer_specs: list[LayerSpec]
    runtime_model: RuntimeModel


@dataclass
class SamplingParams:
    """Internal sampling parameters derived from generation config."""

    temperature: float
    top_p: float
    top_k: int | None = None
    seed: int | None = None


@dataclass
class SamplingCandidates:
    """Device-selected sampling candidates for host-side final sampling."""

    values: torch.Tensor
    token_ids: torch.Tensor


@dataclass
class RequestState:
    """Mutable per-request state tracked during generation."""

    request_id: str
    model_id: str
    prompt: str
    prompt_token_ids: list[int]
    generated_token_ids: list[int] = field(default_factory=list)
    sampling_params: SamplingParams | None = None
    status: Literal["waiting", "prefill", "decode", "finished", "aborted", "error"] = "waiting"
    max_new_tokens: int = 0
    stop_strings: tuple[str, ...] = ()
    eos_token_id: int | None = None
    seq_len: int = 0
    num_prompt_tokens: int = 0
    kv_allocation: "KvAllocation | None" = None
    output_text: str = ""


@dataclass
class KvAllocation:
    """Paged KV-cache allocation assigned to one request."""

    request_id: str
    model_id: str
    page_ids: list[int]
    tokens_capacity: int
    tokens_used: int = 0


@dataclass
class PrefillBatch:
    """Packed inputs for a batched prompt prefill call."""

    request_ids: list[str]
    token_ids: torch.Tensor
    input_embeddings: torch.Tensor | None
    seq_lens: list[int]
    chunk_lens: list[int]
    chunk_offsets: list[int]
    chunk_starts: list[int]
    allow_device_greedy_sampling: bool = False
    allow_device_topk_sampling: bool = False
    kv_allocations: list[KvAllocation] = field(default_factory=list)
    block_ids: list[list[int]] = field(default_factory=list)
    block_ids_by_group: list[dict[str, list[int]]] = field(default_factory=list)
    cache_partitions: list[int | None] = field(default_factory=list)
    # Optional prompt lookahead used by parallel block drafters during chunked
    # prefill. ``None`` marks a chunk that reaches the end of the prompt; its
    # anchor becomes the target model's sampled token in ``finalize_prefill``.
    next_prefill_token_ids: list[int | None] = field(default_factory=list)


@dataclass
class PrefillResult:
    """Outputs from prompt prefill."""

    last_hidden: torch.Tensor | None
    logits: torch.Tensor
    sampled_token_ids: torch.Tensor | None = None
    sampling_candidates: SamplingCandidates | None = None
    next_hidden_states: torch.Tensor | None = None


@dataclass
class DecodeBatch:
    """Inputs for one batched decode step."""

    request_ids: list[str]
    token_ids: torch.Tensor
    # Host-embedding executors (for example DeepSeek) consume hidden_states.
    # Device-embedding executors (for example Qwen) gather embeddings from
    # token_ids inside the decode kernel and leave this as None.
    hidden_states: torch.Tensor | None
    seq_lens: torch.Tensor
    allow_device_greedy_sampling: bool = False
    allow_device_topk_sampling: bool = False
    sampling_params: list[SamplingParams] = field(default_factory=list)
    kv_allocations: list[KvAllocation] = field(default_factory=list)
    block_ids: list[list[int]] = field(default_factory=list)
    block_ids_by_group: list[dict[str, list[int]]] = field(default_factory=list)
    cache_partitions: list[int | None] = field(default_factory=list)
    # Optional MTP context for models (e.g. DeepSeek V4) that decode two real
    # trailing tokens per step. ``prev_token_ids`` holds the token id at absolute
    # position ``seq_len-2`` per request (shape ``[B]``) and ``prev_hidden_states``
    # its embedding (shape ``[B, hidden]``). Left ``None`` for single-token decoders.
    prev_token_ids: torch.Tensor | None = None
    prev_hidden_states: torch.Tensor | None = None


@dataclass
class DecodeResult:
    """Outputs from one decode step."""

    hidden_states: torch.Tensor | None
    # None on an integrated device-sampling path: the host consumes sampled_token_ids and
    # the logits buffer stays device-resident (never copied back).
    logits: torch.Tensor | None
    sampled_token_ids: torch.Tensor | None = None
    sampling_candidates: SamplingCandidates | None = None
    next_hidden_states: torch.Tensor | None = None
    accepted_token_ids: list[list[int]] | None = None


@dataclass
class GenerateResult:
    """Final text, generated IDs, and stop reason for one request."""

    text: str
    token_ids: list[int]
    finish_reason: str
