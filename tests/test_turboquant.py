# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Host-side unit tests for the TurboQuant (TQ) NPU path.

These cover the TQ wiring ported onto main's ``qwen3_l3_dispatch``
architecture without an NPU:
  * the quantized KV-cache allocation driven by ``kv_quant_config``;
  * the TQ kernel-argument builders (26-arg signature + ordering);
  * static upload of the TQ rotation matrices / codebook.

They mirror ``test_batching.py``'s fake style and import the real runner, so
they need ``pypto``/``simpler`` importable (same as ``test_batching.py``) but no
NPU. The kernel-argument ordering is the highest-value check: a wrong order is
silent until the fused kernel runs on hardware.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch

from python.core.model_runner import ModelRunner
from python.core.types import KvQuantConfig, ModelConfig, RuntimeConfig
from examples.model.qwen3_14b.runner.npu_runner import (
    Qwen314BModelRunner,
    _CompiledKernels,
    _DecodeKernelInputs,
    _L3Callable,
    _PrefillInputs,
    _StaticDeviceTensor,
)


def _config() -> ModelConfig:
    return ModelConfig(
        model_id="tq-test",
        architecture="qwen3",
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        bos_token_id=None,
        eos_token_id=None,
        pad_token_id=None,
        torch_dtype="float32",
    )


def _runtime(*, kv_quant_config: KvQuantConfig | None = None) -> RuntimeConfig:
    return RuntimeConfig(
        page_size=64,
        max_batch_size=1,
        max_seq_len=128,
        kv_dtype="bfloat16",
        kv_quant_config=kv_quant_config,
    )


def _decode_weights(hidden_size: int, kv_hidden: int, head_dim: int, intermediate_size: int):
    return {
        "decode_input_rms_weight": torch.ones(1, hidden_size),
        "decode_wq": torch.zeros(hidden_size, hidden_size),
        "decode_wk": torch.zeros(hidden_size, kv_hidden),
        "decode_wv": torch.zeros(hidden_size, kv_hidden),
        "decode_q_norm_weight": torch.ones(1, head_dim),
        "decode_k_norm_weight": torch.ones(1, head_dim),
        "decode_wo": torch.zeros(hidden_size, hidden_size),
        "decode_post_rms_weight": torch.ones(1, hidden_size),
        "decode_w_gate": torch.zeros(hidden_size, intermediate_size),
        "decode_w_up": torch.zeros(hidden_size, intermediate_size),
        "decode_w_down": torch.zeros(intermediate_size, hidden_size),
    }


def _compiled_kernels(config: ModelConfig, *, tq: bool = False) -> _CompiledKernels:
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    head_dim = config.head_dim
    kv_hidden = config.num_key_value_heads * head_dim
    kernel_batch = 1
    max_seq = 128
    max_blocks = math.ceil(max_seq / 64)
    callable_ = _L3Callable(compiled=object(), name="fake", block_dim=1, aicpu_thread_num=1)
    return _CompiledKernels(
        prefill=callable_,
        decode=callable_,
        final_norm_weight=torch.ones(1, hidden_size),
        rope_cos=torch.zeros(max_seq, head_dim),
        rope_sin=torch.zeros(max_seq, head_dim),
        padded_vocab=config.vocab_size,
        padded_lm_head_weight=torch.zeros(config.vocab_size, hidden_size),
        decode_weights=_decode_weights(hidden_size, kv_hidden, head_dim, intermediate_size),
        prefill_hidden_buffer=torch.empty(kernel_batch * max_seq, hidden_size, dtype=torch.bfloat16),
        prefill_seq_lens_buffer=torch.empty(kernel_batch, dtype=torch.int32),
        prefill_chunk_lens_buffer=torch.empty(kernel_batch, dtype=torch.int32),
        prefill_chunk_offsets_buffer=torch.empty(kernel_batch, dtype=torch.int32),
        prefill_block_table_buffer=torch.empty(kernel_batch * max_blocks, dtype=torch.int32),
        prefill_slot_mapping_buffer=torch.empty(kernel_batch * max_seq, dtype=torch.int32),
        prefill_logits_buffer=torch.empty(kernel_batch, config.vocab_size),
        decode_hidden_buffer=torch.zeros(kernel_batch, hidden_size, dtype=torch.bfloat16),
        decode_seq_lens_buffer=torch.zeros(kernel_batch, dtype=torch.int32),
        decode_block_table_buffer=torch.zeros(kernel_batch * max_blocks, dtype=torch.int32),
        decode_slot_mapping_buffer=torch.zeros(kernel_batch, dtype=torch.int32),
        decode_logits_buffer=torch.zeros(kernel_batch, config.vocab_size),
        tq_mode=tq,
        rot_matrices=(
            torch.zeros(config.num_hidden_layers * head_dim, head_dim, dtype=torch.bfloat16)
            if tq
            else None
        ),
        tq_codebook=torch.zeros(1, 16, dtype=torch.float32) if tq else None,
    )


class _RecordingRunner(ModelRunner):
    """Minimal concrete ModelRunner that records KV-cache allocations."""

    def __init__(self) -> None:
        super().__init__()
        self.allocs: list[tuple[tuple[int, ...], torch.dtype]] = []

    def _alloc_kv_cache_tensor(self, shape, dtype) -> Any:
        record = (tuple(shape), dtype)
        self.allocs.append(record)
        return record

    def _free_kv_cache_tensor(self, tensor) -> None:  # noqa: ARG002
        return None

    def run_prefill(self, model, batch):  # noqa: ARG002
        raise NotImplementedError

    def run_decode(self, model, batch):  # noqa: ARG002
        raise NotImplementedError


def _sentinel() -> Any:
    """A unique object usable as a stand-in for a DeviceTensor/tensor arg."""
    return object()


def test_init_kv_cache_allocates_quant_pool_when_kv_quant_config_enabled():
    config = _config()
    runtime = _runtime(kv_quant_config=KvQuantConfig(enabled=True))
    runner = _RecordingRunner()
    runner.init_kv_cache("m", config, runtime)
    pool = runner._kv_caches["m"]

    # BF16 key/value pages are NOT allocated in TQ mode.
    assert pool.key_pages is None
    assert pool.value_pages is None
    # UINT8 quantized K/V caches + FP32 per-row scales are.
    assert pool.quant_k_pages is not None
    assert pool.quant_v_pages is not None
    assert pool.k_scales_pages is not None
    assert pool.v_scales_pages is not None

    num_pages = math.ceil(runtime.max_seq_len / runtime.page_size)
    cache_rows = config.num_hidden_layers * num_pages * config.num_key_value_heads * runtime.page_size
    # Quant indices are nibble-packed: 2x4-bit per byte -> head_dim // 2 wide.
    cache_shape = (cache_rows, config.head_dim // 2)
    scales_shape = (cache_rows, 1)
    assert runner.allocs == [
        (cache_shape, torch.uint8),
        (cache_shape, torch.uint8),
        (scales_shape, torch.float32),
        (scales_shape, torch.float32),
    ]


def test_init_kv_cache_allocates_bf16_pool_by_default():
    config = _config()
    runtime = _runtime()  # kv_quant_config=None -> standard path
    runner = _RecordingRunner()
    runner.init_kv_cache("m", config, runtime)
    pool = runner._kv_caches["m"]

    assert pool.key_pages is not None
    assert pool.value_pages is not None
    assert pool.quant_k_pages is None
    assert pool.quant_v_pages is None
    assert pool.k_scales_pages is None
    assert pool.v_scales_pages is None

    num_pages = math.ceil(runtime.max_seq_len / runtime.page_size)
    cache_rows = config.num_hidden_layers * num_pages * config.num_key_value_heads * runtime.page_size
    cache_shape = (cache_rows, config.head_dim)
    assert runner.allocs == [
        (cache_shape, torch.bfloat16),
        (cache_shape, torch.bfloat16),
    ]


def _tq_runner(config: ModelConfig) -> Qwen314BModelRunner:
    return Qwen314BModelRunner(compiled=_compiled_kernels(config, tq=True), tq_mode=True)


def _prefill_inputs(hidden_size: int) -> _PrefillInputs:
    return _PrefillInputs(
        actual_batch=1,
        hidden=torch.zeros(1, hidden_size, dtype=torch.bfloat16),
        seq_lens=torch.zeros(1, dtype=torch.int32),
        chunk_lens=torch.zeros(1, dtype=torch.int32),
        chunk_offsets=torch.zeros(1, dtype=torch.int32),
        block_table=torch.zeros(2, dtype=torch.int32),
        slot_mapping=torch.zeros(1, dtype=torch.int32),
    )


def test_prefill_tq_kernel_args_match_host_wrapper_signature():
    config = _config()
    runner = _tq_runner(config)
    static = runner._require_static_args()
    inputs = _prefill_inputs(config.hidden_size)
    quant_k, quant_v = _sentinel(), _sentinel()
    k_scales, v_scales = _sentinel(), _sentinel()
    logits = _sentinel()

    args = runner._prefill_tq_kernel_args(inputs, quant_k, quant_v, k_scales, v_scales, logits)

    assert len(args) == 26
    # args 0-1: hidden, seq_lens (TQ prefill drops chunk_lens/chunk_offsets).
    assert args[0] is inputs.hidden
    assert args[1] is inputs.seq_lens
    # args 8-11: rope + addressing.
    assert args[8] is static.rope_cos
    assert args[9] is static.rope_sin
    assert args[10] is inputs.block_table
    assert args[11] is inputs.slot_mapping
    # args 12-17: TQ quant caches + scales + rotation + codebook.
    assert args[12] is quant_k
    assert args[13] is quant_v
    assert args[14] is k_scales
    assert args[15] is v_scales
    assert args[16] is static.rot_matrices
    assert args[17] is static.tq_codebook
    # args 18-22: weight order wo, post_rms, w_gate, w_up, w_down.
    assert args[18] is static.decode_weights["decode_wo"]
    assert args[19] is static.decode_weights["decode_post_rms_weight"]
    assert args[20] is static.decode_weights["decode_w_gate"]
    assert args[21] is static.decode_weights["decode_w_up"]
    assert args[22] is static.decode_weights["decode_w_down"]
    # tail: final_norm, lm_head, out.
    assert args[23] is static.final_norm_weight
    assert args[24] is static.padded_lm_head_weight
    assert args[25] is logits


def test_decode_tq_kernel_args_match_host_wrapper_signature():
    config = _config()
    runner = _tq_runner(config)
    static = runner._require_static_args()
    inputs = _DecodeKernelInputs(
        actual_batch=1,
        hidden=torch.zeros(1, config.hidden_size, dtype=torch.bfloat16),
        seq_lens=torch.zeros(1, dtype=torch.int32),
        block_table=torch.zeros(2, dtype=torch.int32),
        slot_mapping=torch.zeros(1, dtype=torch.int32),
        logits=_sentinel(),
    )
    quant_k, quant_v = _sentinel(), _sentinel()
    k_scales, v_scales = _sentinel(), _sentinel()

    args = runner._decode_tq_kernel_args(inputs, quant_k, quant_v, k_scales, v_scales)

    assert len(args) == 26
    assert args[0] is inputs.hidden
    # args 7-9: seq_lens, block_table, slot_mapping.
    assert args[7] is inputs.seq_lens
    assert args[8] is inputs.block_table
    assert args[9] is inputs.slot_mapping
    # args 12-17: TQ tensors.
    assert args[12] is quant_k
    assert args[13] is quant_v
    assert args[14] is k_scales
    assert args[15] is v_scales
    assert args[16] is static.rot_matrices
    assert args[17] is static.tq_codebook
    # args 18-22: TQ decode weight order is wo, post_rms, w_gate, w_up, w_down
    # (differs from non-TQ decode: wo, w_gate, w_up, w_down, post_rms).
    assert args[18] is static.decode_weights["decode_wo"]
    assert args[19] is static.decode_weights["decode_post_rms_weight"]
    assert args[20] is static.decode_weights["decode_w_gate"]
    assert args[21] is static.decode_weights["decode_w_up"]
    assert args[22] is static.decode_weights["decode_w_down"]
    assert args[25] is inputs.logits


def test_tq_static_tensors_include_rotation_and_codebook():
    config = _config()
    runner = _tq_runner(config)
    static = runner._require_static_args()

    assert isinstance(static.rot_matrices, _StaticDeviceTensor)
    assert isinstance(static.tq_codebook, _StaticDeviceTensor)
    # They must be shared before the worker forks. Use identity (tensors do
    # not support boolean `in`).
    shared = runner._iter_static_host_tensors()
    assert any(runner._compiled.rot_matrices is t for t in shared)
    assert any(runner._compiled.tq_codebook is t for t in shared)


def test_non_tq_compiled_kernels_have_no_tq_statics():
    config = _config()
    tq_runner = _tq_runner(config)
    fp_runner = Qwen314BModelRunner(compiled=_compiled_kernels(config, tq=False), tq_mode=False)

    fp_static = fp_runner._require_static_args()
    assert fp_static.rot_matrices is None
    assert fp_static.tq_codebook is None

    # TQ mode uploads exactly two extra static tensors (rot + codebook) on top
    # of the same shared set the standard path uploads.
    tq_shared = tq_runner._iter_static_host_tensors()
    fp_shared = fp_runner._iter_static_host_tensors()
    assert len(tq_shared) == len(fp_shared) + 2


def test_tq_kernel_args_reject_missing_rotation_artifacts():
    config = _config()
    # tq_mode runner built from a non-TQ compiled kernel set (no rot/codebook).
    runner = Qwen314BModelRunner(compiled=_compiled_kernels(config, tq=False), tq_mode=True)
    inputs = _prefill_inputs(config.hidden_size)
    with pytest.raises(RuntimeError, match="rotation matrices"):
        runner._prefill_tq_kernel_args(
            inputs, _sentinel(), _sentinel(), _sentinel(), _sentinel(), _sentinel()
        )
