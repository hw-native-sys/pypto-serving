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

from .tokenizer import TokenizerAdapter


@dataclass(frozen=True)
class GenerateConfig:
    """User-facing options that control text generation."""

    max_new_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int | None = None
    stop: tuple[str, ...] = ()
    stream: bool = False


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
    # Compile-time generation limit used by model-specific runners.
    max_new_tokens: int = 256


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
class LayerWeights:
    """Loaded weights for one transformer layer in framework orientation."""

    input_rms_weight: torch.Tensor
    wq: torch.Tensor
    wk: torch.Tensor
    wv: torch.Tensor
    q_norm_weight: torch.Tensor
    k_norm_weight: torch.Tensor
    wo: torch.Tensor
    post_rms_weight: torch.Tensor
    w_gate: torch.Tensor
    w_up: torch.Tensor
    w_down: torch.Tensor


@dataclass
class RuntimeModel:
    """Loaded model tensors plus runtime and architecture metadata."""

    config: ModelConfig
    runtime: RuntimeConfig
    embed_tokens: torch.Tensor
    final_norm_weight: torch.Tensor
    lm_head: torch.Tensor
    layers: list[LayerWeights]
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
    """Inputs for a batched prompt prefill call."""

    request_ids: list[str]
    token_ids: torch.Tensor
    input_embeddings: torch.Tensor
    seq_lens: torch.Tensor
    allow_device_greedy_sampling: bool = False
    kv_allocations: list[KvAllocation] = field(default_factory=list)
    positions: torch.Tensor | None = None
    block_ids: list[list[int]] = field(default_factory=list)


@dataclass
class PrefillResult:
    """Outputs from prompt prefill."""

    last_hidden: torch.Tensor | None
    logits: torch.Tensor
    sampled_token_ids: torch.Tensor | None = None
    next_hidden_states: torch.Tensor | None = None


@dataclass
class DecodeBatch:
    """Inputs for one batched decode step."""

    request_ids: list[str]
    token_ids: torch.Tensor
    hidden_states: torch.Tensor
    seq_lens: torch.Tensor
    allow_device_greedy_sampling: bool = False
    kv_allocations: list[KvAllocation] = field(default_factory=list)
    block_ids: list[list[int]] = field(default_factory=list)
    # Optional MTP context for models (e.g. DeepSeek V4) that decode two real
    # trailing tokens per step. ``prev_token_ids`` holds the token id at absolute
    # position ``seq_len-2`` per request (shape ``[B]``) and ``prev_hidden_states``
    # its embedding (shape ``[B, hidden]``). Left ``None`` for single-token decoders.
    prev_token_ids: torch.Tensor | None = None
    prev_hidden_states: torch.Tensor | None = None


@dataclass
class DecodeResult:
    """Outputs from one decode step."""

    hidden_states: torch.Tensor
    logits: torch.Tensor
    sampled_token_ids: torch.Tensor | None = None
    next_hidden_states: torch.Tensor | None = None


@dataclass
class GenerateResult:
    """Final text, generated IDs, and stop reason for one request."""

    text: str
    token_ids: list[int]
    finish_reason: str


@dataclass
class WorkerCommand:
    """Command sent from main process to worker process."""
    type: str  # "step" | "shutdown"
    scheduler_output: object | None = None
    finished_request_ids: list | None = None


@dataclass
class StepOutput:
    """Result returned from worker process after executing a batch step."""
    new_tokens: dict  # {request_id: int}
    error: str | None = None
