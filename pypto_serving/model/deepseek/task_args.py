# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek per-dispatch-class :class:`TaskArgs` builders.

Each builder registers, **in kernel-positional order**, every argument one L3
callable takes -- host-shared I/O slots (allocated/staged here), static weights
(uploaded once via ``StaticDeviceTensor``), worker-resident handles (kv cache,
device weights -- lazy sources), and stacked layer weights. There is no separate
kernel-order tuple and no resident-policy dict: the order is the ``add_slot`` /
``add_arg`` registration sequence, and each arg declares its kind at
registration.

The builders read layout/shapes from ``runner._compiled`` and weights/handles
from runner accessors.  The ``_X_TENSOR_ORDER`` constants below ARE each
kernel's positional contract -- this module is their single source of truth
(they replace the runner's ``_X_fwd_args`` builders +
``_mark_resident_args`` pipeline).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from pypto_serving.model.common.runner.buffer_set import (
    ClearPolicy,
    Placement,
    Slot,
    StaticDeviceTensor,
    copy_shared,
)
from pypto_serving.model.common.runner.task_args import TaskArgs
from pypto_serving.model.deepseek.npu_runner import (
    DEEPSEEK_V4_CACHE_GROUP_NAMES,
    DEEPSEEK_V4_PREFILL_MAX_LOGIT_ROWS,
    DEEPSEEK_V4_SAMPLED_IDS_PAD,
    DEEPSEEK_V4_VOCAB_SIZE,
)

if TYPE_CHECKING:
    from pypto_serving.model.deepseek.npu_runner import DeepSeekV4ModelRunner

__all__ = [
    "DeepSeekPrefillTaskArgs",
    "decode_task_args",
    "mtp_decode_task_args",
    "mtp_prefill_task_args",
    "prefill_task_args",
]


# Argument order for the packed all-43-layer ``l3_prefill_fwd`` kernel. This
# mirrors pypto-lib prefill_fwd.py ``l3_prefill_fwd`` host signature: every
# layer-stacked weight/state tensor in core-parameter order, followed by the
# ``hc_head`` collapse weights, final RMSNorm input, device LM-head weights, and
# hidden/logit outputs and owner-major execution metadata.
# The cache pools are ``pl.InOut`` tensors shared by prefill and decode; mutable
# block tables, slot mappings and token metadata remain shared host inputs.
_PREFILL_FWD_TENSOR_ORDER = (
    "x_hc",
    "hc_attn_fn",
    "hc_attn_scale",
    "hc_attn_base",
    "attn_norm_w",
    "wq_a",
    "wq_b",
    "wq_b_scale",
    "wkv",
    "gamma_cq",
    "gamma_ckv",
    "kv_cache",
    "attn_sink",
    "wo_a",
    "wo_b",
    "wo_b_scale",
    "hca_cmp_kv",
    "csa_cmp_kv",
    "hca_cmp_wkv",
    "hca_cmp_wgate",
    "hca_cmp_ape",
    "hca_cmp_norm_w",
    "hca_compress_state",
    "csa_cmp_wkv",
    "csa_cmp_wgate",
    "csa_cmp_ape",
    "csa_cmp_norm_w",
    "csa_compress_state",
    "csa_hadamard_idx",
    "csa_idx_wq_b",
    "csa_idx_wq_b_scale",
    "csa_weights_proj",
    "csa_inner_wkv",
    "csa_inner_wgate",
    "csa_inner_ape",
    "csa_inner_norm_w",
    "csa_inner_compress_state",
    "idx_kv_cache",
    "idx_kv_scale",
    "hca_compress_state_block_table",
    "csa_compress_state_block_table",
    "csa_inner_compress_state_block_table",
    "freqs_cos",
    "freqs_sin",
    "ori_block_table",
    "hca_cmp_block_table",
    "csa_cmp_block_table",
    "idx_block_table",
    "ori_slot_mapping",
    "position_ids",
    "input_ids",
    "hca_cmp_slot_mapping",
    "hca_state_slot_mapping",
    "csa_cmp_slot_mapping",
    "csa_idx_slot_mapping",
    "csa_state_slot_mapping",
    "csa_inner_state_slot_mapping",
    "hc_ffn_fn",
    "hc_ffn_scale",
    "hc_ffn_base",
    "norm_w",
    "gate_w",
    "gate_bias",
    "tid2eid",
    "routed_w1",
    "routed_w1_scale",
    "routed_w3",
    "routed_w3_scale",
    "routed_w2",
    "routed_w2_scale",
    "shared_w1",
    "shared_w1_scale",
    "shared_w3",
    "shared_w3_scale",
    "shared_w2",
    "shared_w2_scale",
    "hc_head_fn",
    "hc_head_scale",
    "hc_head_base",
    "final_norm_w",
    "pre_hc_hidden_out",
    "lm_head_weight",
    "hidden_out",
    "logits",
    "moe_token_counts",
    "num_tokens_per_owner",
    "logit_row_indices",
)

_PREFILL_DYNAMIC_INPUT_NAMES = frozenset(
    {
        "x_hc",
        "ori_slot_mapping",
        "position_ids",
        "input_ids",
        "hca_cmp_slot_mapping",
        "hca_state_slot_mapping",
        "csa_cmp_slot_mapping",
        "csa_idx_slot_mapping",
        "csa_state_slot_mapping",
        "csa_inner_state_slot_mapping",
    }
)
_PREFILL_DYNAMIC_ARG_NAMES = _PREFILL_DYNAMIC_INPUT_NAMES | {"hidden_out"}

# Argument order shared by the current standalone and fused main-decode ABIs.
# Both entries consume the raw preamble inputs and the split HCA/CSA metadata.
_DECODE_FWD_TENSOR_ORDER = (
    "embed_weight",
    "hc_attn_fn",
    "hc_attn_scale",
    "hc_attn_base",
    "attn_norm_w",
    "wq_a",
    "wq_b",
    "wq_b_scale",
    "wkv",
    "gamma_cq",
    "gamma_ckv",
    "kv_cache",
    "attn_sink",
    "wo_a",
    "wo_b",
    "wo_b_scale",
    "hca_cmp_wkv",
    "hca_cmp_wgate",
    "hca_cmp_ape",
    "hca_cmp_norm_w",
    "hca_compress_state",
    "csa_cmp_wkv",
    "csa_cmp_wgate",
    "csa_cmp_ape",
    "csa_cmp_norm_w",
    "csa_compress_state",
    "csa_idx_wq_b",
    "csa_idx_wq_b_scale",
    "csa_weights_proj",
    "csa_hadamard_idx",
    "csa_inner_wkv",
    "csa_inner_wgate",
    "csa_inner_ape",
    "csa_inner_norm_w",
    "csa_inner_compress_state",
    "hca_cmp_kv",
    "csa_cmp_kv",
    "idx_kv_cache",
    "idx_kv_scale",
    "hc_ffn_fn",
    "hc_ffn_scale",
    "hc_ffn_base",
    "norm_w",
    "gate_w",
    "gate_bias",
    "tid2eid",
    "routed_w1",
    "routed_w1_scale",
    "routed_w3",
    "routed_w3_scale",
    "routed_w2",
    "routed_w2_scale",
    "shared_w1",
    "shared_w1_scale",
    "shared_w3",
    "shared_w3_scale",
    "shared_w2",
    "shared_w2_scale",
    "freqs_cos",
    "freqs_sin",
    "block_table",
    "position_ids",
    "kv_seq_lens",
    "hca_compress_state_block_table",
    "csa_compress_state_block_table",
    "csa_inner_compress_state_block_table",
    "hca_cmp_block_table",
    "csa_cmp_block_table",
    "idx_block_table",
    "block_counts",
    "input_ids",
    "hc_head_fn",
    "hc_head_scale",
    "hc_head_base",
    "final_norm_w",
    "pre_hc_hidden_out",
    "lm_head_weight",
    "hidden_out",
    "logits",
    "sampled_ids",
    "moe_token_counts",
    "num_tokens_per_owner",
    "logit_row_indices",
)

_MTP_PREFILL_TENSOR_ORDER = (
    "hidden_states", "prev_hidden_states",
    "enorm_w", "hnorm_w", "e_proj_w", "e_proj_w_scale", "e_proj_smooth",
    "h_proj_w", "h_proj_w_scale", "h_proj_smooth",
    "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_w",
    "wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv",
    "freqs_cos", "freqs_sin", "kv_cache", "ori_block_table", "ori_slot_mapping",
    "position_ids", "attn_sink", "wo_a", "wo_b", "wo_b_scale",
    "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "norm_w",
    "gate_w", "gate_bias", "tid2eid", "input_ids",
    "routed_w1", "routed_w1_scale", "routed_w3", "routed_w3_scale",
    "routed_w2", "routed_w2_scale", "shared_w1", "shared_w1_scale",
    "shared_w3", "shared_w3_scale", "shared_w2", "shared_w2_scale",
    "mtp_hc_head_fn", "mtp_hc_head_scale", "mtp_hc_head_base", "mtp_norm_w",
    "lm_head_weight", "hidden_out", "pre_hc_hidden_out", "logits", "logit_row_indices",
    "moe_token_counts",
)

_FUSED_MTP_BASE_TENSOR_ORDER = (
    "embed_weight", "main_pre_hc_hidden", "tail_pre_hc_pool",
    "accepted_counts", "tail_slot_ids", "position_ids",
    "enorm_w", "hnorm_w", "e_proj_w", "e_proj_w_scale", "e_proj_smooth",
    "h_proj_w", "h_proj_w_scale", "h_proj_smooth",
    "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_w",
    "wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv",
    "freqs_cos", "freqs_sin", "kv_cache", "ori_block_table",
    "attn_sink", "wo_a", "wo_b", "wo_b_scale",
    "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "norm_w",
    "gate_w", "gate_bias", "tid2eid", "input_ids",
    "routed_w1", "routed_w1_scale", "routed_w3", "routed_w3_scale",
    "routed_w2", "routed_w2_scale", "shared_w1", "shared_w1_scale",
    "shared_w3", "shared_w3_scale", "shared_w2", "shared_w2_scale",
    "mtp_hc_head_fn", "mtp_hc_head_scale", "mtp_hc_head_base", "mtp_norm_w",
    "lm_head_weight", "hidden_out", "next_pre_hc_hidden", "logits", "sampled_ids",
    "logit_row_indices", "moe_token_counts",
)

# PR985 body-only standalone MTP ABI. Embedding lookup, previous-hidden packing,
# and SWA metadata lowering are performed by serving before the dispatch.
_MTP_DECODE_TENSOR_ORDER = (
    "hidden_states", "prev_pre_hc_hidden", "position_ids",
    "enorm_w", "hnorm_w", "e_proj_w", "e_proj_w_scale", "e_proj_smooth",
    "h_proj_w", "h_proj_w_scale", "h_proj_smooth",
    "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_w",
    "wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv",
    "freqs_cos", "freqs_sin", "kv_cache",
    "swa_slot_mapping", "swa_indices", "swa_lens",
    "attn_sink", "wo_a", "wo_b", "wo_b_scale",
    "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "norm_w",
    "gate_w", "gate_bias", "tid2eid", "input_ids",
    "routed_w1", "routed_w1_scale", "routed_w3", "routed_w3_scale",
    "routed_w2", "routed_w2_scale", "shared_w1", "shared_w1_scale",
    "shared_w3", "shared_w3_scale", "shared_w2", "shared_w2_scale",
    "mtp_hc_head_fn", "mtp_hc_head_scale", "mtp_hc_head_base", "mtp_norm_w",
    "lm_head_weight", "hidden_out", "next_pre_hc_hidden", "logits", "sampled_ids",
    "logit_row_indices", "moe_token_counts",
)

# The K=1 fused entry kept the old raw-preamble MTP ABI and adds persistent
# device state after the tail-slot metadata.
_FUSED_MTP_DECODE_TENSOR_ORDER = (
    *_FUSED_MTP_BASE_TENSOR_ORDER[:5],
    "state_generations", "state_tokens", "state_meta",
    *_FUSED_MTP_BASE_TENSOR_ORDER[5:],
)

_FUSED_MTP_SHARED_TENSORS = frozenset(
    {
        "embed_weight",
        "main_pre_hc_hidden",
        "freqs_cos",
        "freqs_sin",
        "ori_block_table",
        "lm_head_weight",
        "moe_token_counts",
    }
)


# Names whose value is a static weight uploaded once (the old
# ``_MAIN_STATIC_RESIDENT_POLICY`` keys, cache_state=False). At registration
# these are wrapped in ``StaticDeviceTensor`` so the resolver uploads+caches them.
_PREFILL_STATIC_WEIGHTS = (
    "freqs_cos",
    "freqs_sin",
    "hc_head_fn",
    "hc_head_scale",
    "hc_head_base",
    "final_norm_w",
    "lm_head_weight",
)

# Worker-resident cache pools (the old ``_MAIN_CACHE_RESIDENT_POLICY`` keys).
# These are already ``StackedDeviceTensor`` handles, so they pass straight through.
_PREFILL_CACHE_POOLS = (
    "kv_cache",
    "hca_cmp_kv",
    "csa_cmp_kv",
    "idx_kv_cache",
    "idx_kv_scale",
    "hca_compress_state",
    "csa_compress_state",
    "csa_inner_compress_state",
)


class DeepSeekPrefillTaskArgs(TaskArgs):
    """Task arguments whose token-shaped slots use the active prefill extent."""

    def __init__(self, ranks: int) -> None:
        super().__init__(stacked=True)
        self._ranks = int(ranks)
        if self._ranks <= 0:
            raise ValueError("DeepSeekV4 prefill ranks must be positive")

    def token_view(self, name: str, kernel_tokens: int) -> torch.Tensor:
        """Return a packed active-token view over one full-capacity shared slot."""
        storage = self.tensors[name]
        if not isinstance(storage, torch.Tensor):
            raise TypeError(f"DeepSeekV4 prefill slot {name!r} is not a Host tensor")
        if storage.ndim < 2 or int(storage.shape[0]) != self._ranks:
            raise ValueError(
                "DeepSeekV4 prefill storage must start with "
                f"[{self._ranks}, capacity], got {name} shape={tuple(storage.shape)}"
            )
        if not storage.is_contiguous():
            raise ValueError(f"DeepSeekV4 prefill storage {name!r} must be contiguous")

        kernel_tokens = int(kernel_tokens)
        capacity = int(storage.shape[1])
        if kernel_tokens <= 0 or kernel_tokens > capacity:
            raise ValueError(
                f"DeepSeekV4 prefill extent {kernel_tokens} exceeds shared capacity {capacity}"
            )
        tail_shape = tuple(int(dim) for dim in storage.shape[2:])
        used_elements = self._ranks * kernel_tokens * math.prod(tail_shape)
        return storage.reshape(-1)[:used_elements].view(
            self._ranks,
            kernel_tokens,
            *tail_shape,
        )

    def stage_for_tokens(
        self,
        inputs: dict[str, torch.Tensor],
        kernel_tokens: int,
    ) -> None:
        """Stage fixed inputs normally and pack token-shaped inputs contiguously."""
        fixed_inputs = {
            name: tensor
            for name, tensor in inputs.items()
            if name not in _PREFILL_DYNAMIC_INPUT_NAMES
        }
        self.stage(fixed_inputs)
        for name in _PREFILL_DYNAMIC_INPUT_NAMES:
            tensor = inputs.get(name)
            if tensor is None:
                continue
            copy_shared(
                self.token_view(name, kernel_tokens),
                tensor,
                name=f"prefill_fwd_{name}",
            )

    def build_for_tokens(self, kernel_tokens: int) -> tuple[object, ...]:
        """Build the kernel tuple with active views for token-shaped arguments."""
        args = self.build()
        return tuple(
            self.token_view(name, kernel_tokens)
            if name in _PREFILL_DYNAMIC_ARG_NAMES
            else arg
            for name, arg in zip(self.names, args, strict=True)
        )


def _static_weight_source(runner: DeepSeekV4ModelRunner, name: str):
    """Return a zero-arg source producing an upload-once ``StaticDeviceTensor``.

    Resolved lazily at ``build()`` time so construction does not require the
    static weights to be allocated yet.
    """
    if name in ("freqs_cos", "freqs_sin"):
        accessor = runner._static_freqs_cos_tensor if name == "freqs_cos" else runner._static_freqs_sin_tensor
    elif name in ("hc_head_fn", "hc_head_scale", "hc_head_base"):
        def accessor():
            return runner._hc_head_tensors()[name]
    elif name == "final_norm_w":
        accessor = runner._static_final_norm_weight_tensor
    elif name == "lm_head_weight":
        accessor = runner._static_lm_head_weight_tensor
    else:  # pragma: no cover - exhaustive over _PREFILL_STATIC_WEIGHTS
        raise KeyError(name)
    return lambda: StaticDeviceTensor(accessor())


def _prefill_slot_specs(
    layout,
    hidden: int,
    vocab: int,
    *,
    token_capacity: int | None = None,
) -> dict[str, tuple[torch.dtype, tuple[int, ...], ClearPolicy]]:
    """Host-shared slot name -> (dtype, full shape, clear policy) for the packed prefill dispatch."""
    ranks = layout.ranks
    seq = layout.prefill_seq if token_capacity is None else int(token_capacity)
    if seq <= 0:
        raise ValueError("DeepSeekV4 prefill token capacity must be positive")
    hc_mult = layout.hc_mult
    zero = ClearPolicy.ZERO
    return {
        "x_hc": (torch.float32, (ranks, seq, hc_mult, hidden), ClearPolicy.NONE),
        "hca_compress_state_block_table": (torch.int32, (ranks, layout.prefill_hca_state_max_blocks), ClearPolicy.NONE),
        "csa_compress_state_block_table": (torch.int32, (ranks, layout.prefill_csa_state_max_blocks), ClearPolicy.NONE),
        "csa_inner_compress_state_block_table": (
            torch.int32,
            (ranks, layout.prefill_csa_inner_state_max_blocks),
            ClearPolicy.NONE,
        ),
        "ori_block_table": (torch.int32, (ranks, layout.prefill_ori_max_blocks), ClearPolicy.NONE),
        "hca_cmp_block_table": (
            torch.int32,
            (ranks, layout.prefill_cmp_max_blocks),
            ClearPolicy.NONE,
        ),
        "csa_cmp_block_table": (
            torch.int32,
            (ranks, layout.prefill_cmp_max_blocks),
            ClearPolicy.NONE,
        ),
        "idx_block_table": (torch.int32, (ranks, layout.prefill_idx_max_blocks), ClearPolicy.NONE),
        "ori_slot_mapping": (torch.int64, (ranks, seq), ClearPolicy.NONE),
        "position_ids": (torch.int32, (ranks, seq), ClearPolicy.NONE),
        "input_ids": (torch.int64, (ranks, seq), ClearPolicy.NONE),
        "hca_cmp_slot_mapping": (torch.int64, (ranks, seq), ClearPolicy.NONE),
        "hca_state_slot_mapping": (torch.int64, (ranks, seq), ClearPolicy.NONE),
        "csa_cmp_slot_mapping": (torch.int64, (ranks, seq), ClearPolicy.NONE),
        "csa_idx_slot_mapping": (torch.int64, (ranks, seq), ClearPolicy.NONE),
        "csa_state_slot_mapping": (torch.int64, (ranks, seq), ClearPolicy.NONE),
        "csa_inner_state_slot_mapping": (torch.int64, (ranks, seq), ClearPolicy.NONE),
        "num_tokens_per_owner": (torch.int32, (ranks,), ClearPolicy.NONE),
        "logit_row_indices": (torch.int32, (ranks, DEEPSEEK_V4_PREFILL_MAX_LOGIT_ROWS), ClearPolicy.NONE),
        # Outputs read back by the host are zeroed before each dispatch. The
        # kernel overwrites the active hidden_out extent, so clearing its full
        # max-sequence backing would add a large, unnecessary host memset.
        # The main kernel exposes the final 128 valid pre-HC rows per owner.
        "pre_hc_hidden_out": (
            torch.float32,
            (ranks, layout.prefill_seq, hc_mult, hidden),
            zero,
        ),
        "hidden_out": (
            torch.bfloat16,
            (ranks, seq, hidden),
            ClearPolicy.NONE,
        ),
        "logits": (torch.float32, (ranks, DEEPSEEK_V4_PREFILL_MAX_LOGIT_ROWS, vocab), zero),
    }


def prefill_task_args(
    runner: DeepSeekV4ModelRunner,
    hidden: int,
    vocab: int,
) -> DeepSeekPrefillTaskArgs:
    """Build the ``TaskArgs`` for the single packed ``l3_prefill_fwd`` dispatch.

    Args are registered in ``_PREFILL_FWD_TENSOR_ORDER`` (the kernel's positional
    contract). Stacked layer weights come from ``runner._require_stacked_weights``
    (everything not classified as a slot / static weight / cache pool); cache
    pools and the stacked weights are lazy so they resolve to device handles at
    ``build()`` time (post-fork, post-resident-upload).
    """
    layout = runner._compiled.layout
    slot_specs = _prefill_slot_specs(
        layout,
        hidden,
        vocab,
        token_capacity=runner._prefill_buffer_tokens(),
    )
    static_weights = set(_PREFILL_STATIC_WEIGHTS)
    cache_pools = set(_PREFILL_CACHE_POOLS)

    ta = DeepSeekPrefillTaskArgs(layout.ranks)
    for name in _PREFILL_FWD_TENSOR_ORDER:
        if name in slot_specs:
            dtype, shape, clear = slot_specs[name]
            ta.add_slot(Slot(name, Placement.HOST_SHARED, dtype, lambda _, s=shape: s, clear=clear))
        elif name in static_weights:
            ta.add_arg(name, _static_weight_source(runner, name))
        elif name in cache_pools:
            ta.add_arg(name, lambda n=name: runner._device_cache_values()[n])
        elif name == "moe_token_counts":
            ta.add_arg(name, lambda: runner._prepare_moe_stats(0))
        else:
            # stacked layer weight -- resolved lazily so construction does not
            # require the stacked weights to be loaded yet.
            ta.add_arg(name, lambda n=name: runner._require_stacked_weights().tensors[n])
    return ta


# Decode reuses the same static-weight / cache-pool name sets as prefill.
_DECODE_STATIC_WEIGHTS = _PREFILL_STATIC_WEIGHTS
_DECODE_CACHE_POOLS = _PREFILL_CACHE_POOLS

_DECODE_STATIC_METADATA_FIELDS = (
    "block_table",
    "hca_cmp_block_table",
    "csa_cmp_block_table",
    "idx_block_table",
    "hca_compress_state_block_table",
    "csa_compress_state_block_table",
    "csa_inner_compress_state_block_table",
    "block_counts",
)


def _decode_slot_specs(layout, hidden: int, vocab: int) -> dict[str, tuple[torch.dtype, tuple[int, ...]]]:
    """Host-shared buffer name -> (dtype, full shape) owned by the decode TaskArgs.

    These are the per-dispatch I/O buffers (metadata, sampled ids, outputs) --
    owned and staged by the TaskArgs, not pulled from the runner.
    """
    ranks = layout.ranks
    batch = layout.decode_batch
    tokens = layout.decode_tokens
    n_groups = len(DEEPSEEK_V4_CACHE_GROUP_NAMES)
    return {
        "block_table": (torch.int32, (ranks, batch, layout.ori_table_max_blocks)),
        "position_ids": (torch.int32, (ranks, tokens)),
        "kv_seq_lens": (torch.int32, (ranks, batch)),
        "hca_compress_state_block_table": (torch.int32, (ranks, batch, layout.prefill_hca_state_max_blocks)),
        "csa_compress_state_block_table": (torch.int32, (ranks, batch, layout.prefill_csa_state_max_blocks)),
        "csa_inner_compress_state_block_table": (torch.int32, (ranks, batch, layout.prefill_csa_inner_state_max_blocks)),
        "hca_cmp_block_table": (torch.int32, (ranks, batch, layout.cmp_max_blocks)),
        "csa_cmp_block_table": (torch.int32, (ranks, batch, layout.cmp_max_blocks)),
        "idx_block_table": (torch.int32, (ranks, batch, layout.idx_max_blocks)),
        "block_counts": (torch.int32, (ranks, batch, n_groups)),
        "input_ids": (torch.int64, (ranks, tokens)),
        "num_tokens_per_owner": (torch.int32, (ranks,)),
        "logit_row_indices": (torch.int32, (ranks, tokens)),
        "sampled_ids": (torch.int32, (ranks, tokens, DEEPSEEK_V4_SAMPLED_IDS_PAD)),
        # outputs
        "hidden_out": (torch.bfloat16, (ranks, tokens, hidden)),
        "logits": (torch.float32, (ranks, tokens, vocab)),
    }


def decode_task_args(
    runner: DeepSeekV4ModelRunner,
    hidden: int,
    vocab: int,
    *,
    buffer_slot: int = 0,
) -> TaskArgs:
    """Build the ``TaskArgs`` for the single packed ``l3_decode_fwd`` dispatch.

    Owns every per-dispatch buffer as a slot. Fused K=1 MTP keeps the write-only
    hidden/logits outputs device-resident; other modes retain host outputs for
    scheduler-visible decode results. The pre-HC output is always resident.
    """
    layout = runner._compiled.layout
    slot_specs = _decode_slot_specs(layout, hidden, vocab)
    static_weights = set(_DECODE_STATIC_WEIGHTS)
    cache_pools = set(_DECODE_CACHE_POOLS)
    ta = TaskArgs(stacked=True)
    for name in _DECODE_FWD_TENSOR_ORDER:
        if runner._compiled.num_speculative_tokens == 1 and name in _DECODE_STATIC_METADATA_FIELDS:
            ta.add_arg(
                name,
                lambda n=name, slot=buffer_slot: runner._materialize_decode_device_metadata(slot)[n],
            )
        elif name in slot_specs:
            dtype, shape = slot_specs[name]
            placement = (
                Placement.DEVICE_RESIDENT
                if runner._compiled.num_speculative_tokens == 1
                and name in {"hidden_out", "logits"}
                else Placement.HOST_SHARED
            )
            ta.add_slot(Slot(name, placement, dtype, lambda _, s=shape: s))
        elif name == "pre_hc_hidden_out":
            ta.add_slot(
                Slot(
                    "pre_hc_hidden_out",
                    Placement.DEVICE_RESIDENT,
                    torch.float32,
                    lambda _: (layout.ranks, layout.decode_tokens, layout.hc_mult, hidden),
                )
            )
        elif name == "embed_weight":
            ta.add_arg(name, lambda: runner._materialize_embedding_device_weight())
        elif name == "moe_token_counts":
            ta.add_arg(name, lambda slot=buffer_slot: runner._prepare_moe_stats(slot + 1))
        elif name in static_weights:
            ta.add_arg(name, _static_weight_source(runner, name))
        elif name in cache_pools:
            ta.add_arg(name, lambda n=name: runner._device_cache_values()[n])
        else:
            ta.add_arg(name, lambda n=name: runner._require_stacked_weights().tensors[n])
    return ta


# MTP prefill reuses the main static-weight markers (freqs/lm_head). kv_cache is
# the MTP-private cache (_materialize_mtp_device_kv_cache), not the main pools.
_MTP_PREFILL_STATIC_WEIGHTS = ("freqs_cos", "freqs_sin", "lm_head_weight")


def _mtp_prefill_input_slots(layout, hidden: int) -> dict[str, tuple[torch.dtype, tuple[int, ...]]]:
    """Host-shared prefill input slots owned by the MTP prefill TaskArgs."""
    ranks = layout.ranks
    seq = layout.prefill_seq
    return {
        "hidden_states": (torch.bfloat16, (ranks, seq, hidden)),
        "prev_hidden_states": (torch.float32, (ranks, seq, layout.hc_mult, hidden)),
        "ori_block_table": (torch.int32, (ranks, layout.prefill_ori_max_blocks)),
        "ori_slot_mapping": (torch.int64, (ranks, seq)),
        "position_ids": (torch.int32, (ranks, seq)),
        "input_ids": (torch.int64, (ranks, seq)),
        "logit_row_indices": (torch.int32, (ranks, DEEPSEEK_V4_PREFILL_MAX_LOGIT_ROWS)),
    }


def _mtp_prefill_device_outputs(layout, hidden: int) -> dict[str, tuple[torch.dtype, tuple[int, ...]]]:
    """Device-resident prefill output slots owned by the MTP prefill TaskArgs."""
    ranks = layout.ranks
    seq = layout.prefill_seq
    return {
        "hidden_out": (torch.bfloat16, (ranks, seq, hidden)),
        "pre_hc_hidden_out": (torch.float32, (ranks, seq, layout.hc_mult, hidden)),
        "logits": (torch.float32, (ranks, DEEPSEEK_V4_PREFILL_MAX_LOGIT_ROWS, DEEPSEEK_V4_VOCAB_SIZE)),
    }


def mtp_prefill_task_args(runner: DeepSeekV4ModelRunner, hidden: int) -> TaskArgs:
    """Build the ``TaskArgs`` for the single packed ``l3_mtp_prefill`` dispatch.

    Owns the per-request prefill host inputs and the device-resident outputs as
    slots; static weights become upload-once markers, the MTP kv_cache and layer
    weights are lazy sources resolved at ``build()``.  The trailing actual-tokens
    scalar is appended by the caller (``_initialize_mtp_draft``) after ``build()``.
    """
    layout = runner._compiled.layout
    input_slots = _mtp_prefill_input_slots(layout, hidden)
    device_outputs = _mtp_prefill_device_outputs(layout, hidden)
    static_weights = set(_MTP_PREFILL_STATIC_WEIGHTS)

    ta = TaskArgs(stacked=True)
    for name in _MTP_PREFILL_TENSOR_ORDER:
        if name in input_slots:
            dtype, shape = input_slots[name]
            ta.add_slot(Slot(name, Placement.HOST_SHARED, dtype, lambda _, s=shape: s))
        elif name in device_outputs:
            dtype, shape = device_outputs[name]
            ta.add_slot(Slot(name, Placement.DEVICE_RESIDENT, dtype, lambda _, s=shape: s))
        elif name == "kv_cache":
            ta.add_arg(name, lambda: runner._materialize_mtp_device_kv_cache())
        elif name == "moe_token_counts":
            ta.add_arg(name, lambda: runner._prepare_moe_stats(0))
        elif name in static_weights:
            ta.add_arg(name, _static_weight_source(runner, name))
        else:
            # MTP layer weight -- resolved lazily so construction does not
            # require the weights to be uploaded yet.
            ta.add_arg(
                name,
                lambda n=name: (
                    runner._mtp_device_weights
                    if runner._mtp_device_weights is not None
                    else runner._require_mtp_buffers().weights
                )[n],
            )
    return ta


# MTP decode reuses the main static-weight markers. These, plus embed_weight,
# main_pre_hc_hidden and ori_block_table, are strip-shared with the main decode
# tuple in the fused compose (_FUSED_MTP_SHARED_TENSORS) -- so their lazy values
# never reach the fused kernel; they must still be registered in order.
_MTP_DECODE_STATIC_WEIGHTS = ("freqs_cos", "freqs_sin", "lm_head_weight")


def _mtp_decode_slots(layout, hidden: int) -> dict[str, tuple[torch.dtype, tuple[int, ...]]]:
    """Per ping-pong slot host buffers owned by the MTP decode TaskArgs.

    The reclaimed outputs (accepted_counts/sampled_ids/committed_*) are read in
    ``_reclaim_mtp_decode``; hidden_out/next_pre_hc_hidden/logits are write-only.
    """
    ranks = layout.ranks
    batch = layout.decode_batch
    tokens = layout.decode_tokens
    return {
        "hidden_states": (torch.bfloat16, (ranks, tokens, hidden)),
        "prev_pre_hc_hidden": (torch.float32, (ranks, tokens, layout.hc_mult, hidden)),
        "accepted_counts": (torch.int32, (ranks, batch)),
        "tail_slot_ids": (torch.int32, (ranks, batch)),
        "state_generations": (torch.int32, (ranks, batch)),
        "position_ids": (torch.int32, (ranks, tokens)),
        "swa_slot_mapping": (torch.int64, (ranks, tokens)),
        "swa_indices": (torch.int32, (ranks, tokens, layout.sliding_window)),
        "swa_lens": (torch.int32, (ranks, tokens)),
        "input_ids": (torch.int64, (ranks, tokens)),
        "sampled_ids": (torch.int32, (ranks, tokens, DEEPSEEK_V4_SAMPLED_IDS_PAD)),
        "logit_row_indices": (torch.int32, (ranks, tokens)),
        "hidden_out": (torch.bfloat16, (ranks, tokens, hidden)),
        "next_pre_hc_hidden": (torch.float32, (ranks, tokens, layout.hc_mult, hidden)),
        "logits": (torch.float32, (ranks, tokens, DEEPSEEK_V4_VOCAB_SIZE)),
    }


def mtp_decode_task_args(
    runner: DeepSeekV4ModelRunner,
    hidden: int,
    *,
    buffer_slot: int = 0,
) -> TaskArgs:
    """Build the ``TaskArgs`` for the single packed ``l3_mtp_decode`` dispatch.

    One per ping-pong slot (the runner keeps a list of 2). K=1 registers the
    fused raw-preamble ABI, keeps write-only outputs device-resident, and uses
    persistent device state. K>1 registers the standalone body-only ABI and
    retains host slots for recurrent readback. Static weights, cache pools, and
    recurrent state are resolved lazily at ``build()``.
    """
    layout = runner._compiled.layout
    slots = _mtp_decode_slots(layout, hidden)
    static_weights = set(_MTP_DECODE_STATIC_WEIGHTS)
    tensor_order = (
        _FUSED_MTP_DECODE_TENSOR_ORDER
        if runner._compiled.num_speculative_tokens == 1
        else _MTP_DECODE_TENSOR_ORDER
    )

    ta = TaskArgs(stacked=True)
    for name in tensor_order:
        if name in slots:
            dtype, shape = slots[name]
            resident_outputs = {
                "input_ids",
                "position_ids",
                "hidden_out",
                "next_pre_hc_hidden",
                "logits",
            }
            placement = (
                Placement.DEVICE_RESIDENT
                if runner._compiled.num_speculative_tokens == 1
                and name in resident_outputs
                else Placement.HOST_SHARED
            )
            ta.add_slot(Slot(name, placement, dtype, lambda _, s=shape: s))
        elif name in static_weights:
            ta.add_arg(name, _static_weight_source(runner, name))
        elif name == "kv_cache":
            ta.add_arg(name, lambda: runner._materialize_mtp_device_kv_cache())
        elif name == "tail_pre_hc_pool":
            ta.add_arg(name, lambda: runner._materialize_mtp_tail_pre_hc_pool(hidden))
        elif name == "state_tokens":
            ta.add_arg(name, lambda: runner._materialize_mtp_device_state_tokens())
        elif name == "state_meta":
            ta.add_arg(name, lambda: runner._materialize_mtp_device_state_meta())
        elif name == "embed_weight":
            ta.add_arg(name, lambda: runner._materialize_embedding_device_weight())
        elif name == "main_pre_hc_hidden":
            ta.add_arg(name, lambda: runner._materialize_main_pre_hc_device(hidden))
        elif name == "ori_block_table":
            if runner._compiled.num_speculative_tokens == 1:
                ta.add_arg(
                    name,
                    lambda slot=buffer_slot: runner._materialize_decode_device_metadata(slot)[
                        "block_table"
                    ],
                )
            else:
                ta.add_arg(
                    name,
                    lambda slot=buffer_slot: runner._decode_task_args[slot].tensors["block_table"],
                )
        elif name == "moe_token_counts":
            ta.add_arg(name, lambda slot=buffer_slot: runner._prepare_moe_stats(slot + 1))
        else:
            ta.add_arg(
                name,
                lambda n=name: (
                    runner._mtp_device_weights
                    if runner._mtp_device_weights is not None
                    else runner._require_mtp_buffers().weights
                )[n],
            )
    return ta
