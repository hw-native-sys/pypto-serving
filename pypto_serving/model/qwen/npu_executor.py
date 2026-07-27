# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import importlib.util
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from pypto_serving.config.types import RuntimeModel
from pypto_serving.model.common.executor.pypto_executor import PyptoExecutor as CorePyptoExecutor
from pypto_serving.model.common.executor.utils import rope_tables, round_up
from pypto_serving.model.common.runner.model_runner import ModelRunner
from pypto_serving.model.qwen import qwen3_l3_dispatch
from pypto_serving.model.qwen.npu_runner import (
    _CompiledKernels,
    _L3Callable,
    Qwen314BModelRunner,
)


_VOCAB_PAD_MULTIPLE = 512  # must be a multiple of lm_head.VOCAB_CHUNK (64)
_QWEN14B_PAGE_SIZE = 128
_QWEN14B_BLOCK_DIM = 24


@dataclass
class _KernelLayerWeights:
    """Kernel-ready weights for one transformer layer."""

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


def _find_pypto_lib_qwen14b_dir(pypto_root: str | None = None) -> Path:
    """Find the Qwen3-14B kernel directory from configuration or a checkout."""
    if pypto_root is None:
        pypto_root = os.environ.get("PYPTO_ROOT")
    if pypto_root:
        candidate = Path(pypto_root) / "models" / "qwen3" / "14b"
        if candidate.is_dir():
            return candidate
        raise FileNotFoundError(f"Qwen3-14B kernel directory not found under PYPTO_ROOT={pypto_root!r}")

    start_dir = Path(__file__).resolve().parent
    for directory in (start_dir, *start_dir.parents):
        candidate = directory / "pypto-lib" / "models" / "qwen3" / "14b"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Cannot locate Qwen3-14B kernels. Run from a checkout with pypto-lib available "
        "or set PYPTO_ROOT to a pypto-lib checkout."
    )


def _load_pypto_lib_qwen14b_module(module_name: str, kernel_dir: Path) -> object:
    """Load a Qwen3-14B kernel module from the pypto-lib submodule."""
    module_path = kernel_dir / f"qwen3_14b_{module_name}.py"
    if not module_path.is_file():
        module_path = kernel_dir / f"qwen3_14b_{module_name}_draft.py"
    if not module_path.is_file():
        module_path = kernel_dir / f"{module_name}.py"
    if not module_path.is_file():
        raise FileNotFoundError(
            f"Missing pypto-lib Qwen3-14B kernel module: {module_path}. "
            "Run `git submodule update --init --recursive`."
        )
    spec = importlib.util.spec_from_file_location(
        f"_pypto_lib_qwen3_14b_{module_name}",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load pypto-lib kernel module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(kernel_dir))
    try:
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(kernel_dir))
        except ValueError:
            pass
    return module


class Qwen314BPyptoExecutor(CorePyptoExecutor):
    """PyPTO executor that compiles and registers the Qwen3-14B kernels."""

    def __init__(
        self,
        kv_cache_manager=None,
        *,
        platform: str = "a2a3sim",
        device_ids: Sequence[int] = (0,),
        save_kernels_dir: str | None = None,
        pypto_root: str | None = None,
        l3_trace: bool = False,
        tq_mode: bool = False,
    ) -> None:
        super().__init__(
            kv_cache_manager,
            platform=platform,
            device_ids=device_ids,
            save_kernels_dir=save_kernels_dir,
        )
        self._pypto_root = pypto_root
        self._l3_trace = l3_trace
        self._tq_mode = tq_mode

    @property
    def profile_verbose(self) -> bool:
        """Return whether compile and L3 execution timing logs are enabled."""
        return self._l3_trace

    @property
    def supports_device_sampling(self) -> bool:
        """Qwen3 NPU runner can return greedy sampled token ids.

        TurboQuant decode returns raw logits (no fused device sampling), so the
        engine argmaxes/samples on the host instead.
        """
        return not self._tq_mode

    @property
    def supports_device_embedding(self) -> bool:
        """Qwen3 NPU decode embeds greedy token ids inside the device kernel.

        Disabled in TurboQuant mode (host-side embedding lookup + CPU sampling).
        """
        return not self._tq_mode

    def _create_runner(self, model_id: str, compiled: object) -> ModelRunner:
        """Create the Qwen3-14B runtime runner for compiled kernels."""
        if not isinstance(compiled, _CompiledKernels):
            raise TypeError("Qwen314BPyptoExecutor requires Qwen3-14B compiled kernels.")
        return Qwen314BModelRunner(
            compiled=compiled,
            device_id=self._device_ids[0],
            tq_mode=self._tq_mode,
        )

    def _pad_vocab_weights(self, model: RuntimeModel, padded_vocab: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad lm_head/embed weights to ``padded_vocab`` rows and move to shared bf16 CPU.

``lm_head`` is padded by replicating its row 0; ``embed`` is padded with zeros.
        """
        lm_head_weight = model.lm_head
        if padded_vocab != lm_head_weight.shape[0]:
            pad_rows = padded_vocab - lm_head_weight.shape[0]
            padding = lm_head_weight[:1].expand(pad_rows, -1).clone()
            lm_head_weight = torch.cat([lm_head_weight, padding], dim=0)
        padded_lm_head_weight = self._shared_tensor(lm_head_weight.to(torch.bfloat16).contiguous().cpu())
        embed_weight = model.embed_tokens
        if padded_vocab != embed_weight.shape[0]:
            pad_rows = padded_vocab - embed_weight.shape[0]
            padding = torch.zeros(
                (pad_rows, embed_weight.shape[1]),
                dtype=embed_weight.dtype,
                device=embed_weight.device,
            )
            embed_weight = torch.cat([embed_weight, padding], dim=0)
        padded_embed_weight = self._shared_tensor(embed_weight.to(torch.bfloat16).contiguous().cpu())
        return padded_lm_head_weight, padded_embed_weight

    def _compile_model(self, model: RuntimeModel) -> _CompiledKernels:
        """Compile Qwen3-14B PyPTO kernels and pack runtime artifacts."""
        if self._tq_mode:
            return self._compile_tq_kernels(model)
        kernel_dir = _find_pypto_lib_qwen14b_dir(self._pypto_root)
        qwen3_prefill_fwd = _load_pypto_lib_qwen14b_module("prefill_fwd", kernel_dir)
        # The fused all-layer decode lives in decode_fwd.decode_fwd. It is
        # PAGED: it consumes block_table + slot_mapping and reads/writes the SAME
        # device-resident paged KV pool prefill writes (self._kv_caches), so no
        # contiguous bridge / MAX_SEQ env is needed.
        qwen3_decode_fwd = _load_pypto_lib_qwen14b_module("decode_fwd", kernel_dir)
        qwen3_greedy_sample = _load_pypto_lib_qwen14b_module("greedy_sample", kernel_dir)
        qwen3_l3_dispatch.prefill_fwd = qwen3_prefill_fwd.prefill_fwd
        qwen3_l3_dispatch.decode_fwd = qwen3_decode_fwd.decode_fwd
        qwen3_l3_dispatch.greedy_sample_fwd = qwen3_greedy_sample.greedy_sample_fwd

        self._validate_supported_shape(model)
        kernel_batch = model.runtime.max_batch_size

        kernel_max_seq = int(getattr(qwen3_decode_fwd, "MAX_SEQ", 4096))
        if model.runtime.max_seq_len > kernel_max_seq:
            raise ValueError(
                f"max_model_len {model.runtime.max_seq_len} exceeds the kernel's "
                f"compile-time MAX_SEQ {kernel_max_seq} (config.py). The decode/prefill "
                "kernels precompute MAX_CTX_BLOCKS, NUM_PAGES, and rope table sizes from "
                "MAX_SEQ; a larger runtime value silently produces wrong attention and "
                "out-of-bounds rope reads. Rebuild the kernel with a larger MAX_SEQ."
            )

        if int(qwen3_decode_fwd.BATCH) != kernel_batch:
            raise ValueError(
                "decode_fwd.decode_fwd is compiled for a fixed kernel BATCH of "
                f"{int(qwen3_decode_fwd.BATCH)}, but runtime max_batch_size is "
                f"{kernel_batch}; they must match (decode statically computes and "
                "writes BATCH rows / BATCH logit rows)."
            )
        if int(model.config.num_hidden_layers) != int(qwen3_decode_fwd.NUM_LAYERS):
            raise ValueError(
                "decode_fwd.decode_fwd fuses a FIXED "
                f"NUM_LAYERS={int(qwen3_decode_fwd.NUM_LAYERS)} loop (the layer count "
                "is a kernel constant, not derived from the weight tensors), but the "
                f"model has {model.config.num_hidden_layers} layers. The fused decode "
                "does not support --num-layers-override; run the full model."
            )
        self._validate_total_kv_pages(model, kernel_batch)

        padded_vocab = round_up(model.config.vocab_size, _VOCAB_PAD_MULTIPLE)
        if padded_vocab != int(qwen3_decode_fwd.VOCAB):
            raise ValueError(
                f"decode_fwd.decode_fwd hard-codes VOCAB={int(qwen3_decode_fwd.VOCAB)} "
                f"(config.VOCAB) for its fused LM head, but the runtime padded vocab is "
                f"{padded_vocab} (round_up({model.config.vocab_size}, {_VOCAB_PAD_MULTIPLE})); "
                "they must match for the decode logits buffer / lm_head weight to line up."
            )
        if model.config.vocab_size != int(qwen3_decode_fwd.REAL_VOCAB):
            raise ValueError(
                "decode_fwd.decode_fwd hard-codes REAL_VOCAB for padded-token masking, "
                f"but the runtime model vocab_size is {model.config.vocab_size}; expected "
                f"{int(qwen3_decode_fwd.REAL_VOCAB)}."
            )
        if int(qwen3_greedy_sample.BATCH) != kernel_batch:
            raise ValueError(
                "greedy_sample_fwd is compiled for a fixed kernel BATCH of "
                f"{int(qwen3_greedy_sample.BATCH)}, but runtime max_batch_size is {kernel_batch}."
            )
        if int(qwen3_greedy_sample.VOCAB) != padded_vocab:
            raise ValueError(
                "greedy_sample_fwd VOCAB must match the padded logits vocab: "
                f"{int(qwen3_greedy_sample.VOCAB)} != {padded_vocab}."
            )
        sampled_ids_width = int(
            getattr(qwen3_decode_fwd, "SAMPLED_IDS_PAD", getattr(qwen3_greedy_sample, "SAMPLED_IDS_PAD", 1))
        )
        page_size = model.runtime.page_size
        max_blocks_per_seq = (model.runtime.max_seq_len + page_size - 1) // page_size
        prefill = self._compile_prefill_fwd_callable(
            qwen3_l3_dispatch.qwen3_prefill_host,
            batch=kernel_batch,
            max_seq=model.runtime.max_seq_len,
            hidden_size=model.config.hidden_size,
            intermediate_size=model.config.intermediate_size,
            num_heads=model.config.num_attention_heads,
            num_kv_heads=model.config.num_key_value_heads,
            head_dim=model.config.head_dim,
            num_layers=model.config.num_hidden_layers,
            vocab_size=padded_vocab,
            block_table_stride=max_blocks_per_seq,
            page_size=page_size,
            sampled_ids_width=sampled_ids_width,
        )
        decode = self._compile_decode_fwd_callable(
            qwen3_l3_dispatch.qwen3_decode_host,
            batch=kernel_batch,
            max_seq=model.runtime.max_seq_len,
            block_table_stride=max_blocks_per_seq,
            hidden_size=model.config.hidden_size,
            intermediate_size=model.config.intermediate_size,
            num_heads=model.config.num_attention_heads,
            num_kv_heads=model.config.num_key_value_heads,
            head_dim=model.config.head_dim,
            num_layers=model.config.num_hidden_layers,
            vocab_size=padded_vocab,
            page_size=page_size,
            sampled_ids_width=sampled_ids_width,
        )
        greedy_sample = self._compile_greedy_sample_callable(
            qwen3_l3_dispatch.qwen3_greedy_sample_host,
            batch=kernel_batch,
            sampled_ids_width=sampled_ids_width,
            vocab_size=padded_vocab,
        )
        rope_cos_raw, rope_sin_raw = rope_tables(
            model.runtime.max_seq_len,
            model.config.head_dim,
            model.config.rope_theta,
        )
        rope_cos = self._shared_tensor(rope_cos_raw)
        rope_sin = self._shared_tensor(rope_sin_raw)

        padded_lm_head_weight, padded_embed_weight = self._pad_vocab_weights(model, padded_vocab)
        final_norm_weight = self._shared_tensor(model.final_norm_weight.view(1, -1).float().cpu())
        decode_weights = self._stage_stacked_decode_weights(model)
        prefill_token_ids_buffer = torch.empty(
            (kernel_batch * model.runtime.max_seq_len,),
            dtype=torch.int32,
        ).share_memory_()
        prefill_seq_lens_buffer = torch.empty((kernel_batch,), dtype=torch.int32).share_memory_()
        prefill_chunk_lens_buffer = torch.empty((kernel_batch,), dtype=torch.int32).share_memory_()
        prefill_chunk_offsets_buffer = torch.empty((kernel_batch,), dtype=torch.int32).share_memory_()
        prefill_block_table_buffer = torch.empty(
            (kernel_batch * max_blocks_per_seq,),
            dtype=torch.int32,
        ).share_memory_()
        prefill_slot_mapping_buffer = torch.empty(
            (kernel_batch * model.runtime.max_seq_len,),
            dtype=torch.int32,
        ).share_memory_()
        prefill_logits_buffer = torch.empty(
            (kernel_batch, padded_vocab),
            dtype=torch.float32,
        ).share_memory_()
        prefill_sampled_ids_buffer = torch.empty(
            (kernel_batch, sampled_ids_width),
            dtype=torch.int32,
        ).share_memory_()
        prefill_next_hidden_buffer = torch.empty(
            (kernel_batch, model.config.hidden_size),
            dtype=torch.bfloat16,
        ).share_memory_()
        decode_logits_buffer = torch.empty(
            (kernel_batch, padded_vocab),
            dtype=torch.float32,
        ).share_memory_()
        decode_seq_lens_buffer = torch.empty((kernel_batch,), dtype=torch.int32).share_memory_()
        decode_block_table_buffer = torch.empty(
            (kernel_batch * max_blocks_per_seq,),
            dtype=torch.int32,
        ).share_memory_()
        decode_slot_mapping_buffer = torch.empty((kernel_batch,), dtype=torch.int32).share_memory_()
        decode_token_ids_buffer = torch.empty(
            (kernel_batch, sampled_ids_width),
            dtype=torch.int32,
        ).share_memory_()
        decode_sampled_ids_buffer = torch.empty(
            (kernel_batch, sampled_ids_width),
            dtype=torch.int32,
        ).share_memory_()
        decode_next_hidden_buffer = torch.empty(
            (kernel_batch, model.config.hidden_size),
            dtype=torch.bfloat16,
        ).share_memory_()
        return _CompiledKernels(
            prefill=prefill,
            decode=decode,
            greedy_sample=greedy_sample,
            final_norm_weight=final_norm_weight,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            padded_vocab=padded_vocab,
            padded_lm_head_weight=padded_lm_head_weight,
            padded_embed_weight=padded_embed_weight,
            decode_weights=decode_weights,
            prefill_token_ids_buffer=prefill_token_ids_buffer,
            prefill_seq_lens_buffer=prefill_seq_lens_buffer,
            prefill_chunk_lens_buffer=prefill_chunk_lens_buffer,
            prefill_chunk_offsets_buffer=prefill_chunk_offsets_buffer,
            prefill_block_table_buffer=prefill_block_table_buffer,
            prefill_slot_mapping_buffer=prefill_slot_mapping_buffer,
            prefill_logits_buffer=prefill_logits_buffer,
            prefill_sampled_ids_buffer=prefill_sampled_ids_buffer,
            prefill_next_hidden_buffer=prefill_next_hidden_buffer,
            decode_seq_lens_buffer=decode_seq_lens_buffer,
            decode_block_table_buffer=decode_block_table_buffer,
            decode_slot_mapping_buffer=decode_slot_mapping_buffer,
            decode_logits_buffer=decode_logits_buffer,
            decode_token_ids_buffer=decode_token_ids_buffer,
            decode_sampled_ids_buffer=decode_sampled_ids_buffer,
            decode_next_hidden_buffer=decode_next_hidden_buffer,
        )

    def _compile_prefill_fwd_callable(
        self,
        jit_fn: object,
        *,
        batch: int,
        max_seq: int,
        block_table_stride: int,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        num_layers: int,
        vocab_size: int,
        page_size: int,
        sampled_ids_width: int,
    ) -> _L3Callable:
        """Compile the prefill HOST wrapper into a distributed program."""
        kv_hidden = num_kv_heads * head_dim
        total_tokens = batch * max_seq
        runtime_cache_blocks = (max_seq + page_size - 1) // page_size
        cache_rows = batch * runtime_cache_blocks * num_layers * num_kv_heads * page_size
        dummy_args = [
            torch.empty((total_tokens,), dtype=torch.int32),
            torch.empty((batch,), dtype=torch.int32),
            torch.empty((batch,), dtype=torch.int32),
            torch.empty((batch,), dtype=torch.int32),
            torch.empty((num_layers, hidden_size), dtype=torch.float32),
            torch.empty((num_layers * hidden_size, hidden_size), dtype=torch.bfloat16),
            torch.empty((num_layers * hidden_size, kv_hidden), dtype=torch.bfloat16),
            torch.empty((num_layers * hidden_size, kv_hidden), dtype=torch.bfloat16),
            torch.empty((num_layers, head_dim), dtype=torch.float32),
            torch.empty((num_layers, head_dim), dtype=torch.float32),
            torch.empty((max_seq, head_dim), dtype=torch.float32),
            torch.empty((max_seq, head_dim), dtype=torch.float32),
            torch.empty((batch * block_table_stride,), dtype=torch.int32),
            torch.empty((total_tokens,), dtype=torch.int32),
            torch.empty((cache_rows, head_dim), dtype=torch.bfloat16),
            torch.empty((cache_rows, head_dim), dtype=torch.bfloat16),
            torch.empty((num_layers * hidden_size, hidden_size), dtype=torch.bfloat16),
            torch.empty((num_layers * hidden_size, intermediate_size), dtype=torch.bfloat16),
            torch.empty((num_layers * hidden_size, intermediate_size), dtype=torch.bfloat16),
            torch.empty((num_layers * intermediate_size, hidden_size), dtype=torch.bfloat16),
            torch.empty((num_layers, hidden_size), dtype=torch.float32),
            torch.empty((1, hidden_size), dtype=torch.float32),
            torch.empty((vocab_size, hidden_size), dtype=torch.bfloat16),
            torch.empty((vocab_size, hidden_size), dtype=torch.bfloat16),
            torch.empty((batch, vocab_size), dtype=torch.float32),
        ]
        return self._compile_jit_fwd_callable("prefill_fwd", jit_fn, dummy_args)

    def _compile_decode_fwd_callable(
        self,
        jit_fn: object,
        *,
        batch: int,
        max_seq: int,
        block_table_stride: int,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        num_layers: int,
        vocab_size: int,
        page_size: int,
        sampled_ids_width: int,
    ) -> _L3Callable:
        """Compile the fused all-layer PAGED decode HOST wrapper into a distributed program.

        Signature (21 args; PAGED KV via block_table + slot_mapping, same pool as
        prefill):
          input_rms_weight, wq, wk, wv, q_norm_weight,
          k_norm_weight, seq_lens, block_table, slot_mapping, rope_cos, rope_sin,
          k_cache, v_cache, wo, w_gate, w_up, w_down, post_rms_weight,
          final_norm_weight, lm_head_weight, out.

        k_cache/v_cache are the PAGED pool (rows = num_layers * batch *
        runtime_cache_blocks * num_kv_heads * page_size — identical to prefill);
        the kernel derives the per-layer stride + max_blocks_per_seq from the
        tensor dims. Projection weights are stacked ``[num_layers*HIDDEN, ...]``
        and norm gammas ``[num_layers, dim]`` — exactly what
        ``_stack_decode_weights`` produces.
        """
        kv_hidden = num_kv_heads * head_dim
        runtime_cache_blocks = (max_seq + page_size - 1) // page_size
        cache_rows = num_layers * batch * runtime_cache_blocks * num_kv_heads * page_size
        dummy_args = [
            torch.empty((num_layers, hidden_size), dtype=torch.float32),                      # input_rms_weight
            torch.empty((num_layers * hidden_size, hidden_size), dtype=torch.bfloat16),        # wq
            torch.empty((num_layers * hidden_size, kv_hidden), dtype=torch.bfloat16),          # wk
            torch.empty((num_layers * hidden_size, kv_hidden), dtype=torch.bfloat16),          # wv
            torch.empty((num_layers, head_dim), dtype=torch.float32),                          # q_norm_weight
            torch.empty((num_layers, head_dim), dtype=torch.float32),                          # k_norm_weight
            torch.empty((batch,), dtype=torch.int32),                                          # seq_lens
            torch.empty((batch * block_table_stride,), dtype=torch.int32),                     # block_table
            torch.empty((batch,), dtype=torch.int32),                                          # slot_mapping
            torch.empty((max_seq, head_dim), dtype=torch.float32),                             # rope_cos
            torch.empty((max_seq, head_dim), dtype=torch.float32),                             # rope_sin
            torch.empty((cache_rows, head_dim), dtype=torch.bfloat16),                         # k_cache (paged pool)
            torch.empty((cache_rows, head_dim), dtype=torch.bfloat16),                         # v_cache (paged pool)
            torch.empty((num_layers * hidden_size, hidden_size), dtype=torch.bfloat16),        # wo
            torch.empty((num_layers * hidden_size, intermediate_size), dtype=torch.bfloat16),  # w_gate
            torch.empty((num_layers * hidden_size, intermediate_size), dtype=torch.bfloat16),  # w_up
            torch.empty((num_layers * intermediate_size, hidden_size), dtype=torch.bfloat16),  # w_down
            torch.empty((num_layers, hidden_size), dtype=torch.float32),                       # post_rms_weight
            torch.empty((1, hidden_size), dtype=torch.float32),                                # final_norm_weight
            torch.empty((vocab_size, hidden_size), dtype=torch.bfloat16),                      # lm_head_weight
            torch.empty((batch, vocab_size), dtype=torch.float32),                             # out
            torch.empty((vocab_size, hidden_size), dtype=torch.bfloat16),                      # embed_weight
            torch.empty((batch, sampled_ids_width), dtype=torch.int32),                        # sampled_ids_in
            torch.empty((batch, sampled_ids_width), dtype=torch.int32),                        # sampled_ids_out
            torch.empty((batch, hidden_size), dtype=torch.bfloat16),                           # next_hidden
        ]
        return self._compile_jit_fwd_callable("decode_fwd", jit_fn, dummy_args)

    # ------------------------------------------------------------------
    # TurboQuant (TQ) compile path
    # ------------------------------------------------------------------

    def _compile_tq_kernels(self, model: RuntimeModel) -> _CompiledKernels:
        """Compile TurboQuant prefill/decode kernels and pack runtime artifacts.

        TQ decode is 26-param and returns raw logits (no fused device
        sampling/embedding), so the engine argmaxes/samples on the host and
        ``supports_device_*`` are False.  KV caches are nibble-packed UINT8
        (``head_dim // 2`` bytes/row) + per-row FP32 scales, allocated by the
        runner from ``kv_quant_config``.
        """
        kernel_dir = _find_pypto_lib_qwen14b_dir(self._pypto_root)
        qwen3_prefill_tq = _load_pypto_lib_qwen14b_module("prefill_tq", kernel_dir)
        qwen3_decode_tq = _load_pypto_lib_qwen14b_module("decode_tq", kernel_dir)
        tq_kv = _load_pypto_lib_qwen14b_module("turboquant_kv", kernel_dir)
        qwen3_l3_dispatch.prefill_fwd_tq = qwen3_prefill_tq.prefill_fwd_tq
        qwen3_l3_dispatch.decode_fwd_tq = qwen3_decode_tq.decode_fwd_tq

        self._validate_supported_shape(model)
        self._validate_tq_shape(model, qwen3_decode_tq, qwen3_prefill_tq)
        kernel_batch = model.runtime.max_batch_size

        page_size = model.runtime.page_size
        max_seq = model.runtime.max_seq_len
        max_blocks_per_seq = (max_seq + page_size - 1) // page_size
        padded_vocab = round_up(model.config.vocab_size, _VOCAB_PAD_MULTIPLE)
        # Prefill TQ rope is STATIC [MAX_SEQ, HEAD_DIM] (MAX_SEQ = M.max_seq = 4096).
        tq_prefill_max_seq = int(qwen3_prefill_tq.MAX_SEQ)

        prefill = self._compile_prefill_fwd_tq_callable(
            qwen3_l3_dispatch.qwen3_prefill_tq_host,
            batch=kernel_batch,
            max_seq=max_seq,
            block_table_stride=max_blocks_per_seq,
            hidden_size=model.config.hidden_size,
            intermediate_size=model.config.intermediate_size,
            num_heads=model.config.num_attention_heads,
            num_kv_heads=model.config.num_key_value_heads,
            head_dim=model.config.head_dim,
            num_layers=model.config.num_hidden_layers,
            vocab_size=padded_vocab,
            page_size=page_size,
            prefill_rope_seq=tq_prefill_max_seq,
        )
        decode = self._compile_decode_fwd_tq_callable(
            qwen3_l3_dispatch.qwen3_decode_tq_host,
            batch=kernel_batch,
            max_seq=max_seq,
            block_table_stride=max_blocks_per_seq,
            hidden_size=model.config.hidden_size,
            intermediate_size=model.config.intermediate_size,
            num_heads=model.config.num_attention_heads,
            num_kv_heads=model.config.num_key_value_heads,
            head_dim=model.config.head_dim,
            num_layers=model.config.num_hidden_layers,
            vocab_size=padded_vocab,
            page_size=page_size,
        )
        # rot_matrices: one random orthogonal [head_dim, head_dim] per layer,
        # stacked -> [num_layers * head_dim, head_dim] BF16 (seed=42, matches the
        # lib golden init so serving + reference stay aligned).
        rot_matrices = self._build_tq_rot_matrices(
            model.config.num_hidden_layers, model.config.head_dim,
        )
        # tq_codebook: Lloyd-Max 4-bit centroids -> [1, 16] FP32.
        tq_codebook = self._build_tq_codebook(tq_kv, model.config.head_dim)
        rope_cos, rope_sin = rope_tables(max_seq, model.config.head_dim, model.config.rope_theta)
        rope_cos = self._shared_tensor(rope_cos)
        rope_sin = self._shared_tensor(rope_sin)
        # Prefill TQ needs a static [tq_prefill_max_seq, head_dim] rope table.
        pf_rope_cos, pf_rope_sin = rope_tables(tq_prefill_max_seq, model.config.head_dim, model.config.rope_theta)
        tq_prefill_rope_cos = self._shared_tensor(pf_rope_cos)
        tq_prefill_rope_sin = self._shared_tensor(pf_rope_sin)
        padded_lm_head_weight, padded_embed_weight = self._pad_vocab_weights(model, padded_vocab)
        layers = []
        for layer in model.layers:
            layers.append(self._kernel_layer_weights(layer))
            self._release_layer_weights(layer)
        final_norm_weight = self._shared_tensor(model.final_norm_weight.view(1, -1).float().cpu())
        decode_weights = {
            name: self._shared_tensor(tensor)
            for name, tensor in self._stack_decode_weights(layers).items()
        }
        hidden_size = model.config.hidden_size
        prefill_hidden_buffer = torch.empty(
            (kernel_batch * max_seq, hidden_size), dtype=torch.bfloat16,
        ).share_memory_()
        prefill_seq_lens_buffer = torch.empty((kernel_batch,), dtype=torch.int32).share_memory_()
        prefill_chunk_lens_buffer = torch.empty((kernel_batch,), dtype=torch.int32).share_memory_()
        prefill_chunk_offsets_buffer = torch.empty((kernel_batch,), dtype=torch.int32).share_memory_()
        prefill_block_table_buffer = torch.empty(
            (kernel_batch * max_blocks_per_seq,), dtype=torch.int32,
        ).share_memory_()
        prefill_slot_mapping_buffer = torch.empty(
            (kernel_batch * max_seq,), dtype=torch.int32,
        ).share_memory_()
        prefill_logits_buffer = torch.empty(
            (kernel_batch, padded_vocab), dtype=torch.float32,
        ).share_memory_()
        # TQ prefill returns logits only (CPU sampling); these are unused but
        # retained so the static-share path stays uniform with FP.
        prefill_sampled_ids_buffer = torch.empty((kernel_batch, 1), dtype=torch.int32).share_memory_()
        prefill_next_hidden_buffer = torch.empty(
            (kernel_batch, hidden_size), dtype=torch.bfloat16,
        ).share_memory_()
        decode_logits_buffer = torch.empty(
            (kernel_batch, padded_vocab), dtype=torch.float32,
        ).share_memory_()
        decode_seq_lens_buffer = torch.empty((kernel_batch,), dtype=torch.int32).share_memory_()
        decode_block_table_buffer = torch.empty(
            (kernel_batch * max_blocks_per_seq,), dtype=torch.int32,
        ).share_memory_()
        decode_slot_mapping_buffer = torch.empty((kernel_batch,), dtype=torch.int32).share_memory_()
        # TQ decode takes hidden_states (the embedding) as its first input;
        # FP uses token_ids instead.  Allocate a hidden input buffer for TQ.
        decode_hidden_buffer = torch.empty(
            (kernel_batch, hidden_size), dtype=torch.bfloat16,
        ).share_memory_()
        # Unused by TQ kernels but kept for static-share uniformity.
        decode_token_ids_buffer = torch.empty((kernel_batch, 1), dtype=torch.int32).share_memory_()
        decode_sampled_ids_buffer = torch.empty((kernel_batch, 1), dtype=torch.int32).share_memory_()
        decode_next_hidden_buffer = torch.empty(
            (kernel_batch, hidden_size), dtype=torch.bfloat16,
        ).share_memory_()
        return _CompiledKernels(
            prefill=prefill,
            decode=decode,
            greedy_sample=None,
            tq_mode=True,
            rot_matrices=rot_matrices,
            tq_codebook=tq_codebook,
            final_norm_weight=final_norm_weight,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            tq_prefill_rope_cos=tq_prefill_rope_cos,
            tq_prefill_rope_sin=tq_prefill_rope_sin,
            padded_vocab=padded_vocab,
            padded_lm_head_weight=padded_lm_head_weight,
            padded_embed_weight=padded_embed_weight,
            decode_weights=decode_weights,
            prefill_hidden_buffer=prefill_hidden_buffer,
            prefill_seq_lens_buffer=prefill_seq_lens_buffer,
            prefill_chunk_lens_buffer=prefill_chunk_lens_buffer,
            prefill_chunk_offsets_buffer=prefill_chunk_offsets_buffer,
            prefill_block_table_buffer=prefill_block_table_buffer,
            prefill_slot_mapping_buffer=prefill_slot_mapping_buffer,
            prefill_logits_buffer=prefill_logits_buffer,
            prefill_sampled_ids_buffer=prefill_sampled_ids_buffer,
            prefill_next_hidden_buffer=prefill_next_hidden_buffer,
            decode_seq_lens_buffer=decode_seq_lens_buffer,
            decode_block_table_buffer=decode_block_table_buffer,
            decode_slot_mapping_buffer=decode_slot_mapping_buffer,
            decode_logits_buffer=decode_logits_buffer,
            decode_token_ids_buffer=decode_token_ids_buffer,
            decode_sampled_ids_buffer=decode_sampled_ids_buffer,
            decode_next_hidden_buffer=decode_next_hidden_buffer,
            decode_hidden_buffer=decode_hidden_buffer,
        )

    @classmethod
    def _validate_tq_shape(
        cls,
        model: RuntimeModel,
        qwen3_decode_tq: object,
        qwen3_prefill_tq: object,
    ) -> None:
        """Validate runtime shape against the fixed TQ kernel constants."""
        kernel_batch = model.runtime.max_batch_size
        if int(qwen3_decode_tq.BATCH) != kernel_batch:
            raise ValueError(
                "decode_fwd_tq is compiled for a fixed kernel BATCH of "
                f"{int(qwen3_decode_tq.BATCH)}, but runtime max_batch_size is "
                f"{kernel_batch}; they must match."
            )
        if int(model.config.num_hidden_layers) != int(qwen3_decode_tq.NUM_LAYERS):
            raise ValueError(
                f"decode_fwd_tq fuses a FIXED NUM_LAYERS={int(qwen3_decode_tq.NUM_LAYERS)}, "
                f"but the model has {model.config.num_hidden_layers} layers."
            )
        padded_vocab = round_up(model.config.vocab_size, _VOCAB_PAD_MULTIPLE)
        if padded_vocab != int(qwen3_decode_tq.VOCAB):
            raise ValueError(
                f"decode_fwd_tq hard-codes VOCAB={int(qwen3_decode_tq.VOCAB)}, "
                f"but the runtime padded vocab is {padded_vocab}."
            )
        # TQ modules do not export REAL_VOCAB; read it from QWEN3_14B.real_vocab.
        real_vocab = int(qwen3_decode_tq.M.real_vocab)
        if model.config.vocab_size != real_vocab:
            raise ValueError(
                "decode_fwd_tq expects REAL_VOCAB for padded-token masking, "
                f"but the runtime model vocab_size is {model.config.vocab_size}; "
                f"expected {real_vocab}."
            )
        cls._validate_total_kv_pages(model, kernel_batch)

    @staticmethod
    def _build_tq_rot_matrices(num_layers: int, head_dim: int) -> torch.Tensor:
        """Build stacked per-layer random orthogonal matrices for TQ.

        Returns ``[num_layers * head_dim, head_dim]`` BF16.  seed=42 matches the
        pypto-lib golden ``init_rot_matrices`` so the served rotation matches the
        reference quantization/dequantization.
        """
        torch.manual_seed(42)
        rot = []
        for _ in range(num_layers):
            q, _ = torch.linalg.qr(torch.randn(head_dim, head_dim))
            rot.append(q)
        return torch.cat(rot, dim=0).to(torch.bfloat16).contiguous()

    @staticmethod
    def _build_tq_codebook(tq_kv: object, head_dim: int) -> torch.Tensor:
        """Build the Lloyd-Max 4-bit codebook -> ``[1, 16]`` FP32."""
        centroids, _ = tq_kv.solve_lloyd_max(head_dim, 4)
        return centroids.float().unsqueeze(0).contiguous()

    def _compile_prefill_fwd_tq_callable(
        self,
        jit_fn: object,
        *,
        batch: int,
        max_seq: int,
        block_table_stride: int,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        num_layers: int,
        vocab_size: int,
        page_size: int,
        prefill_rope_seq: int,
    ) -> _L3Callable:
        """Compile the TQ prefill HOST wrapper (26 params, returns logits).

        Signature (TQ order — NO chunk_lens/chunk_offsets; ``seq_lens`` at
        position 1; ``rope_cos``/``rope_sin`` are a STATIC ``[prefill_rope_seq,
        head_dim]`` table):
          hidden_states, seq_lens, input_rms_weight, wq, wk, wv, q_norm_weight,
          k_norm_weight, rope_cos, rope_sin, block_table, slot_mapping,
          quant_k_cache, quant_v_cache, quant_k_scales, quant_v_scales,
          rot_matrices, tq_codebook, wo, post_rms_weight, w_gate, w_up, w_down,
          final_norm_weight, lm_head_weight, out.
        """
        del num_heads  # unused (kept for API symmetry with the FP compile helper)
        kv_hidden = num_kv_heads * head_dim
        total_tokens = batch * max_seq
        runtime_cache_blocks = (max_seq + page_size - 1) // page_size
        cache_rows = batch * runtime_cache_blocks * num_layers * num_kv_heads * page_size
        half_dim = head_dim // 2
        dummy_args = [
            torch.empty((total_tokens, hidden_size), dtype=torch.bfloat16),                 # hidden_states
            torch.empty((batch,), dtype=torch.int32),                                       # seq_lens
            torch.empty((num_layers, hidden_size), dtype=torch.float32),                    # input_rms_weight
            torch.empty((num_layers * hidden_size, hidden_size), dtype=torch.bfloat16),     # wq
            torch.empty((num_layers * hidden_size, kv_hidden), dtype=torch.bfloat16),       # wk
            torch.empty((num_layers * hidden_size, kv_hidden), dtype=torch.bfloat16),       # wv
            torch.empty((num_layers, head_dim), dtype=torch.float32),                       # q_norm_weight
            torch.empty((num_layers, head_dim), dtype=torch.float32),                       # k_norm_weight
            torch.empty((prefill_rope_seq, head_dim), dtype=torch.float32),                 # rope_cos (static)
            torch.empty((prefill_rope_seq, head_dim), dtype=torch.float32),                 # rope_sin (static)
            torch.empty((batch * block_table_stride,), dtype=torch.int32),                  # block_table
            torch.empty((total_tokens,), dtype=torch.int32),                                # slot_mapping
            torch.empty((cache_rows, half_dim), dtype=torch.uint8),                         # quant_k_cache
            torch.empty((cache_rows, half_dim), dtype=torch.uint8),                         # quant_v_cache
            torch.empty((cache_rows, 1), dtype=torch.float32),                              # quant_k_scales
            torch.empty((cache_rows, 1), dtype=torch.float32),                              # quant_v_scales
            torch.empty((num_layers * head_dim, head_dim), dtype=torch.bfloat16),           # rot_matrices
            torch.empty((1, 16), dtype=torch.float32),                                      # tq_codebook
            torch.empty((num_layers * hidden_size, hidden_size), dtype=torch.bfloat16),     # wo
            torch.empty((num_layers, hidden_size), dtype=torch.float32),                    # post_rms_weight
            torch.empty((num_layers * hidden_size, intermediate_size), dtype=torch.bfloat16),  # w_gate
            torch.empty((num_layers * hidden_size, intermediate_size), dtype=torch.bfloat16),  # w_up
            torch.empty((num_layers * intermediate_size, hidden_size), dtype=torch.bfloat16),  # w_down
            torch.empty((1, hidden_size), dtype=torch.float32),                             # final_norm_weight
            torch.empty((vocab_size, hidden_size), dtype=torch.bfloat16),                   # lm_head_weight
            torch.empty((batch, vocab_size), dtype=torch.float32),                          # out
        ]
        return self._compile_jit_fwd_callable("prefill_fwd_tq", jit_fn, dummy_args)

    def _compile_decode_fwd_tq_callable(
        self,
        jit_fn: object,
        *,
        batch: int,
        max_seq: int,
        block_table_stride: int,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        num_layers: int,
        vocab_size: int,
        page_size: int,
    ) -> _L3Callable:
        """Compile the TQ decode HOST wrapper (26 params, returns logits).

        Signature (TQ order — ``hidden_states`` is the input embedding; rope is
        DYNAMIC ``[ROPE_SEQ_DYN, head_dim]``):
          hidden_states, input_rms_weight, wq, wk, wv, q_norm_weight,
          k_norm_weight, seq_lens, block_table, slot_mapping, rope_cos, rope_sin,
          quant_k_cache, quant_v_cache, quant_k_scales, quant_v_scales,
          rot_matrices, tq_codebook, wo, post_rms_weight, w_gate, w_up, w_down,
          final_norm_weight, lm_head_weight, out.
        """
        del num_heads  # unused (kept for API symmetry with the FP compile helper)
        kv_hidden = num_kv_heads * head_dim
        runtime_cache_blocks = (max_seq + page_size - 1) // page_size
        cache_rows = num_layers * batch * runtime_cache_blocks * num_kv_heads * page_size
        half_dim = head_dim // 2
        dummy_args = [
            torch.empty((batch, hidden_size), dtype=torch.bfloat16),                        # hidden_states
            torch.empty((num_layers, hidden_size), dtype=torch.float32),                    # input_rms_weight
            torch.empty((num_layers * hidden_size, hidden_size), dtype=torch.bfloat16),     # wq
            torch.empty((num_layers * hidden_size, kv_hidden), dtype=torch.bfloat16),       # wk
            torch.empty((num_layers * hidden_size, kv_hidden), dtype=torch.bfloat16),       # wv
            torch.empty((num_layers, head_dim), dtype=torch.float32),                       # q_norm_weight
            torch.empty((num_layers, head_dim), dtype=torch.float32),                       # k_norm_weight
            torch.empty((batch,), dtype=torch.int32),                                       # seq_lens
            torch.empty((batch * block_table_stride,), dtype=torch.int32),                  # block_table
            torch.empty((batch,), dtype=torch.int32),                                       # slot_mapping
            torch.empty((max_seq, head_dim), dtype=torch.float32),                          # rope_cos (dynamic)
            torch.empty((max_seq, head_dim), dtype=torch.float32),                          # rope_sin (dynamic)
            torch.empty((cache_rows, half_dim), dtype=torch.uint8),                         # quant_k_cache
            torch.empty((cache_rows, half_dim), dtype=torch.uint8),                         # quant_v_cache
            torch.empty((cache_rows, 1), dtype=torch.float32),                              # quant_k_scales
            torch.empty((cache_rows, 1), dtype=torch.float32),                              # quant_v_scales
            torch.empty((num_layers * head_dim, head_dim), dtype=torch.bfloat16),           # rot_matrices
            torch.empty((1, 16), dtype=torch.float32),                                      # tq_codebook
            torch.empty((num_layers * hidden_size, hidden_size), dtype=torch.bfloat16),     # wo
            torch.empty((num_layers, hidden_size), dtype=torch.float32),                    # post_rms_weight
            torch.empty((num_layers * hidden_size, intermediate_size), dtype=torch.bfloat16),  # w_gate
            torch.empty((num_layers * hidden_size, intermediate_size), dtype=torch.bfloat16),  # w_up
            torch.empty((num_layers * intermediate_size, hidden_size), dtype=torch.bfloat16),  # w_down
            torch.empty((1, hidden_size), dtype=torch.float32),                             # final_norm_weight
            torch.empty((vocab_size, hidden_size), dtype=torch.bfloat16),                   # lm_head_weight
            torch.empty((batch, vocab_size), dtype=torch.float32),                          # out
        ]
        return self._compile_jit_fwd_callable("decode_fwd_tq", jit_fn, dummy_args)

    def _compile_greedy_sample_callable(
        self,
        jit_fn: object,
        *,
        batch: int,
        sampled_ids_width: int,
        vocab_size: int,
    ) -> _L3Callable:
        """Compile the greedy sampling HOST wrapper."""
        dummy_args = [
            torch.empty((batch, vocab_size), dtype=torch.float32),
            torch.empty((batch, sampled_ids_width), dtype=torch.int32),
        ]
        return self._compile_jit_fwd_callable("greedy_sample_fwd", jit_fn, dummy_args)

    def _compile_jit_fwd_callable(
        self,
        name: str,
        jit_fn: object,
        dummy_args: list[torch.Tensor],
    ) -> _L3Callable:
        """Compile a HOST wrapper into a PyPTO DistributedCompiledProgram."""
        from pypto.ir.distributed_compiled_program import DistributedCompiledProgram  # noqa: PLC0415
        from pypto.ir.distributed_compiled_program import DistributedConfig  # noqa: PLC0415
        from pypto.runtime import RunConfig  # noqa: PLC0415

        config = self._run_config(codegen_only=True)
        distributed_config = DistributedConfig(
            device_ids=list(self._device_ids),
            num_sub_workers=0,
            block_dim=_QWEN14B_BLOCK_DIM,
            aicpu_thread_num=4,
        )
        run_config = RunConfig(
            platform=config.platform,
            device_id=config.device_id,
            backend_type=config.backend_type,
            strategy=config.strategy,
            dump_passes=config.dump_passes,
            save_kernels=config.save_kernels,
            save_kernels_dir=config.save_kernels_dir,
            codegen_only=True,
            diagnostic_phase=config.diagnostic_phase,
            disabled_diagnostics=config.disabled_diagnostics,
            compile_profiling=config.compile_profiling,
            distributed_config=distributed_config,
        )
        compiled = jit_fn.compile(*dummy_args, config=run_config)
        if not isinstance(compiled, DistributedCompiledProgram):
            raise TypeError(
                f"{name} did not compile to DistributedCompiledProgram; got {type(compiled).__name__}"
            )
        return _L3Callable(
            compiled=compiled,
            name=name,
            block_dim=_QWEN14B_BLOCK_DIM,
            aicpu_thread_num=4,
        )

    @staticmethod
    def _load_runtime_config(output_dir: Path) -> dict[str, Any]:
        """Load ``RUNTIME_CONFIG`` from a generated ``kernel_config.py``."""
        config_path = output_dir / "kernel_config.py"
        spec = importlib.util.spec_from_file_location(f"_qwen_l2_kernel_config_{abs(hash(output_dir))}", config_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load kernel_config.py from {config_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return dict(getattr(module, "RUNTIME_CONFIG", {}))

    @classmethod
    def _stage_stacked_decode_weights(cls, model: RuntimeModel) -> dict[str, torch.Tensor]:
        """Stage per-layer weights into pre-allocated stacked shm tensors,
        copying layers in parallel across worker threads.

        Same output as building every per-layer ``_KernelLayerWeights`` and then
        ``torch.cat``-ing them (see ``_stack_decode_weights``), and the same ~1x
        peak host memory as the serial stream (only the stacked destination plus
        the transient views live at once). The per-layer copies dominate startup
        (~90s serially for a 14B), so they run on a thread pool: each layer owns a
        disjoint row-slice of every stacked tensor, and ``copy_`` releases the GIL
        for the memcpy + dtype cast, so the copies genuinely overlap.
        """
        from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

        layers = model.layers
        num_layers = len(layers)
        fields = (
            ("input_rms_weight", "decode_input_rms_weight", "norm"),
            ("wq", "decode_wq", "proj"),
            ("wk", "decode_wk", "proj"),
            ("wv", "decode_wv", "proj"),
            ("q_norm_weight", "decode_q_norm_weight", "norm"),
            ("k_norm_weight", "decode_k_norm_weight", "norm"),
            ("wo", "decode_wo", "proj"),
            ("post_rms_weight", "decode_post_rms_weight", "norm"),
            ("w_gate", "decode_w_gate", "proj"),
            ("w_up", "decode_w_up", "proj"),
            ("w_down", "decode_w_down", "proj"),
        )

        def _ready_view(layer, attr: str, kind: str):
            t = getattr(layer, attr).cpu()
            # reshape (not view): norm gammas are 1-D contiguous today, but
            # reshape also handles a non-contiguous source without raising.
            return t.transpose(0, 1) if kind == "proj" else t.reshape(1, -1)

        # Pre-allocate every stacked shm tensor once (shapes taken from layer 0,
        # uniform across a transformer) so the parallel loop only writes into
        # already-sized, disjoint slices -- no dict mutation or allocation race.
        # Sizes come straight from tensor metadata: a "proj" weight [out, in]
        # stacks its transpose to [num_layers*in, out]; a "norm" gamma [dim]
        # stacks to [num_layers, dim]. Reading only .shape/.dtype avoids a
        # redundant .cpu()/transpose of layer 0 (which _stage_layer(0) redoes).
        stacked: dict[str, torch.Tensor] = {}
        rows_by_key: dict[str, int] = {}
        first = layers[0]
        for attr, key, kind in fields:
            t = getattr(first, attr)
            if kind == "proj":
                rows = t.shape[1]
                shape = (num_layers * rows, t.shape[0])
                dtype = torch.bfloat16
            else:
                rows = 1
                shape = (num_layers, t.shape[0])
                dtype = torch.float32
            rows_by_key[key] = rows
            stacked[key] = torch.empty(shape, dtype=dtype).share_memory_()

        def _stage_layer(i: int) -> None:
            layer = layers[i]
            for attr, key, kind in fields:
                rows = rows_by_key[key]
                # Disjoint per-layer slice -> safe to write concurrently.
                stacked[key][i * rows:(i + 1) * rows].copy_(_ready_view(layer, attr, kind))
            cls._release_layer_weights(layer)

        workers = cls._staging_worker_count(num_layers)
        if workers <= 1:
            for i in range(num_layers):
                _stage_layer(i)
        else:
            # Pin torch intra-op parallelism to 1 for the duration of the pool:
            # each copy_ would otherwise fan out to its own OpenMP/MKL threads,
            # and N pool threads x that fan-out oversubscribes a many-core host
            # (the >32-thread staging regression). One intra-op thread per copy
            # keeps the coarse per-layer parallelism clean.
            orig_threads = torch.get_num_threads()
            torch.set_num_threads(1)
            try:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    list(pool.map(_stage_layer, range(num_layers)))
            finally:
                torch.set_num_threads(orig_threads)
        return stacked

    @staticmethod
    def _staging_worker_count(num_layers: int) -> int:
        """Thread count for parallel weight staging (env-tunable)."""
        raw = os.environ.get("PYPTO_STAGING_THREADS")
        if raw:
            try:
                return max(1, min(int(raw), num_layers))
            except ValueError:
                pass
        # Staging is memory-bandwidth bound: it plateaus by ~16-32 threads and
        # regresses beyond that, so cap the default even on many-core hosts.
        return max(1, min(num_layers, os.cpu_count() or 8, 32))

    @staticmethod
    def _stack_decode_weights(layers: list[_KernelLayerWeights]) -> dict[str, torch.Tensor]:
        """Stack per-layer weights into fused decode-kernel tensors."""
        # Stack from already-prepared per-layer kernel weights. Each
        # _KernelLayerWeights field is already in the kernel-ready shape/dtype
        # (transposed bf16 cpu for projections, [1, N] float cpu for norms),
        # so a plain cat along dim 0 is all that's left. Reading from the
        # original model.layers here would crash because _release_layer_weights
        # has already replaced those tensors with torch.empty(0).
        def cat(attr: str) -> torch.Tensor:
            return torch.cat([getattr(l, attr) for l in layers], dim=0)

        return {
            "decode_input_rms_weight": cat("input_rms_weight").contiguous(),
            "decode_wq":               cat("wq"),
            "decode_wk":               cat("wk"),
            "decode_wv":               cat("wv"),
            "decode_q_norm_weight":    cat("q_norm_weight").contiguous(),
            "decode_k_norm_weight":    cat("k_norm_weight").contiguous(),
            "decode_wo":               cat("wo"),
            "decode_post_rms_weight":  cat("post_rms_weight").contiguous(),
            "decode_w_gate":           cat("w_gate"),
            "decode_w_up":             cat("w_up"),
            "decode_w_down":           cat("w_down"),
        }

    @classmethod
    def _validate_total_kv_pages(cls, model: RuntimeModel, kernel_batch: int) -> None:
        """Validate that runtime KV page count covers the batch capacity."""
        if model.runtime.total_kv_pages is None:
            return
        if model.runtime.total_kv_pages < kernel_batch:
            raise ValueError(
                f"total_kv_pages must be at least kernel_batch ({kernel_batch}), "
                f"got {model.runtime.total_kv_pages}"
            )

    @staticmethod
    def _kernel_weight(weight: torch.Tensor) -> torch.Tensor:
        """Convert a 2-D model weight into kernel-ready orientation and dtype."""
        return weight.transpose(0, 1).to(torch.bfloat16).contiguous().cpu().share_memory_()

    @classmethod
    def _kernel_layer_weights(cls, layer) -> _KernelLayerWeights:
        """Convert one Hugging Face layer into kernel-ready weight tensors."""
        return _KernelLayerWeights(
            input_rms_weight=cls._shared_tensor(layer.input_rms_weight.view(1, -1).float().cpu()),
            wq=cls._kernel_weight(layer.wq),
            wk=cls._kernel_weight(layer.wk),
            wv=cls._kernel_weight(layer.wv),
            q_norm_weight=cls._shared_tensor(layer.q_norm_weight.view(1, -1).float().cpu()),
            k_norm_weight=cls._shared_tensor(layer.k_norm_weight.view(1, -1).float().cpu()),
            wo=cls._kernel_weight(layer.wo),
            post_rms_weight=cls._shared_tensor(layer.post_rms_weight.view(1, -1).float().cpu()),
            w_gate=cls._kernel_weight(layer.w_gate),
            w_up=cls._kernel_weight(layer.w_up),
            w_down=cls._kernel_weight(layer.w_down),
        )

    @staticmethod
    def _shared_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Move a CPU tensor into shared memory if needed."""
        if tensor.device.type == "cpu" and not tensor.is_shared():
            return tensor.share_memory_()
        return tensor

    @staticmethod
    def _release_layer_weights(layer) -> None:
        """Drop original layer tensors after kernel-ready copies are built."""
        empty = torch.empty(0)
        layer.input_rms_weight = empty
        layer.wq = empty
        layer.wk = empty
        layer.wv = empty
        layer.q_norm_weight = empty
        layer.k_norm_weight = empty
        layer.wo = empty
        layer.post_rms_weight = empty
        layer.w_gate = empty
        layer.w_up = empty
        layer.w_down = empty

    @staticmethod
    def _validate_supported_shape(model: RuntimeModel) -> None:
        """Ensure the loaded model matches the bundled Qwen3-14B kernels."""
        config = model.config
        expected = {
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "num_attention_heads": 40,
            "num_key_value_heads": 8,
            "head_dim": 128,
        }
        actual = {
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
        }
        if actual != expected:
            mismatch = ", ".join(f"{k}={actual[k]} (expected {v})" for k, v in expected.items() if actual[k] != v)
            raise ValueError(
                "Bundled kernels under model/ currently support Qwen3-14B layer shapes only: " + mismatch
            )
        if model.runtime.page_size != _QWEN14B_PAGE_SIZE:
            raise ValueError(
                "PyPTO Qwen3-14B kernels require runtime page_size "
                f"{_QWEN14B_PAGE_SIZE}, got {model.runtime.page_size}."
            )
