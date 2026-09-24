# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DSpark W8A8 checkpoint-to-kernel weight mapping.

The DSpark kernels (``pypto-lib/models/deepseek_v4_flash_dspark``) read the same
W8A8 checkpoint tensors as the MTP variant, under the same BF16/INT8 dtypes
(``gamma_ckv`` included: every dspark signature declares it BF16, and a FP32
staging is bit-reinterpreted on device).  The packing differs from the MTP
variant in three ways:

* ``wo_a`` / ``wo_b`` are tensor-parallel sharded -- each of the 4 TP ranks owns
  2 of the 8 output-projection groups, and the kernel regathers the full
  projection on device.  ``wo_b`` is group-major ``[O_GROUPS, D, O_LORA]``
  (pypto-lib #1359), so both names shard whole leading groups.
* The NZ set is not the MTP one: pypto-lib #1359 streams ``wq_a`` / ``wq_b``,
  the o-projection pair, the shared and routed experts and ``lm_head`` NZ, but
  keeps ``wkv`` and the CSA indexer pair ND (NZ addressability cannot prove the
  fused indexer offsets non-negative).  Decode and prefill even disagree on two
  of them: prefill reads ``wo_a`` / ``wo_b`` ND (its TP gather assembles whole
  groups through an ND window) and ``csa_weights_proj`` NZ on a new layer axis,
  so the ``prefill_`` rules below pack a second, layout-divergent copy of those
  checkpoint tensors.
* ``hc_attn_fn`` / ``hc_ffn_fn`` are stored by the decode weight bank padded to
  32 storage rows per layer (the kernel's fixed ``HC_FN_STORAGE_ROWS``), while
  the prefill wrapper consumes the natural 24-row ``MIX_HC`` layout -- the loader
  derives the unpadded prefill slab from the padded decode slab.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from pypto_serving.model.common.weights.spec import LayerRule, LayerWeightRule, OptionalWeightRule
from pypto_serving.model.common.weights.stacker import StackGroup
from pypto_serving.model.deepseek.weight_spec import (
    DEEPSEEK_V4_CSA_RATIO,
    DEEPSEEK_V4_LAYER_RULES,
    DEEPSEEK_V4_STAGING_POLICY,
    deepseek_v4_expert_parallel,
    deepseek_v4_factories,
    deepseek_v4_stack_groups,
)

__all__ = [
    "DSPARK_DRAFT_LAYERS",
    "DSPARK_DRAFTER_LAYERS",
    "DSPARK_DRAFTER_REPLICATED_WEIGHT_NAMES",
    "DSPARK_HC_FN_STORAGE_ROWS",
    "DSPARK_LAYER_RULES",
    "DSPARK_MIX_HC",
    "DSPARK_NEW_AXIS_STACKED_WEIGHT_NAMES",
    "DSPARK_O_GROUPS",
    "DSPARK_PADDED_ROW_WEIGHT_NAMES",
    "DSPARK_PREFILL_LAYER_RULES",
    "DSPARK_TP_SIZE",
    "DSParkShardPolicy",
    "dspark_drafter_required_weight_names",
    "dspark_expert_parallel",
    "dspark_factories",
    "dspark_stack_groups",
]

# Canonical 16-card topology: TP4/DP4 => EP16.
DSPARK_TP_SIZE = 4
DSPARK_O_GROUPS = 8
DSPARK_O_LORA = 1024
# The decode weight bank pads the HC function matrices to this many rows.
DSPARK_HC_FN_STORAGE_ROWS = 32
DSPARK_MIX_HC = 24

# The decode bank stores these with zero padding past ``DSPARK_MIX_HC`` rows.
DSPARK_PADDED_ROW_WEIGHT_NAMES = frozenset({"hc_attn_fn", "hc_ffn_fn"})

DSPARK_MISMATCH_ERROR = (
    "packed DSpark destination {name} shape/dtype mismatch: expected={expected}, got={got}"
)

# The V4 rules whose pack_nz the DSpark kernels do not share: `wkv` and the CSA
# indexer pair stay ND here (pypto-lib #1359; see the module docstring).
_DSPARK_ND_DECODE_NAMES = frozenset({"wkv", "csa_idx_wq_b", "csa_weights_proj"})

# Layer rules fork the V4 tuple — same names, sources, dtypes and order — with the
# DSpark NZ marks: everything V4 packs NZ stays NZ except the three ND names above.
DSPARK_LAYER_RULES: tuple[LayerRule, ...] = tuple(
    replace(rule, pack_nz=False) if getattr(rule, "name", None) in _DSPARK_ND_DECODE_NAMES else rule
    for rule in DEEPSEEK_V4_LAYER_RULES
)

# Prefill reads the o-projection pair ND and the CSA indexer projection NZ on a new
# layer axis, while decode reads the opposite of each — so prefill gets its own
# copies packed under `prefill_` names and routed to its own slab uploads. Rule
# order stays appended-only: DSpark has no prepacked sidecar, but the slab layout
# follows the packed mapping and interleaving would churn every destination.
_DSPARK_HIDDEN = 4096
_DSPARK_INDEXER_HEADS = 64
DSPARK_PREFILL_LAYER_RULES: tuple[LayerRule, ...] = (
    LayerWeightRule("prefill_wo_a", "attn.wo_a.weight", torch.bfloat16, reshape_groups=DSPARK_O_GROUPS),
    LayerWeightRule("prefill_wo_b", "attn.wo_b.weight", torch.int8, column_reshape_groups=DSPARK_O_GROUPS),
    OptionalWeightRule(
        "prefill_csa_weights_proj",
        "attn.indexer.weights_proj.weight",
        torch.bfloat16,
        (_DSPARK_HIDDEN, _DSPARK_INDEXER_HEADS),
        (DEEPSEEK_V4_CSA_RATIO,),
        transpose=True,
        pack_nz=True,
    ),
)

# The decode bank stacks these on a new layer axis instead of widening the
# fractal-blocked row axis (NZ cannot address a row window) — mirrors pypto-lib's
# NZ_ROW_STACKED_NAMES in deepseek_v4_flash_dspark decode_fwd.py, plus the prefill
# NZ indexer copy the prefill side of the divergence adds. `wo_a` / `wo_b` and the
# routed experts are NZ too but already stack on an existing leading axis.
DSPARK_NEW_AXIS_STACKED_WEIGHT_NAMES = frozenset(
    {
        "wq_a",
        "wq_b",
        "shared_w1",
        "shared_w3",
        "shared_w2",
        "prefill_csa_weights_proj",
    }
)

DSPARK_RANK_ERROR = "packed DSpark weight {name} must have rank >= 2, got {ndim}"
DSPARK_STACK_MISMATCH_ERROR = (
    "packed DSpark weight {name} shape/dtype mismatch: source={source}, destination={destination}"
)
DSPARK_SOURCE_MISSING_ERROR = "missing raw DeepSeekV4 layer tensor: {name}"
DSPARK_EXPERT_MISSING_ERROR = "missing raw DeepSeekV4 expert tensor: {name}"


class DSParkShardPolicy:
    """Pack one layer tensor under the DSpark rank layout.

    A rule-name dispatch rather than one flat policy: most weights replicate
    across all 16 ranks, the output projection shards across the 4 TP ranks,
    and the two HC function matrices pad to the decode bank's storage rows.
    Keeping the dispatch here lets the shared generic packer stay unaware of
    per-name layouts.
    """

    def __init__(self, ranks: int, *, tp_size: int = DSPARK_TP_SIZE) -> None:
        self.ranks = int(ranks)
        self.tp_size = int(tp_size)
        # The generic packer's zero-fill path reads this attribute for its
        # destination-shape diagnostics, mirroring Replicate/ExpertParallel.
        self.mismatch_error = DSPARK_MISMATCH_ERROR
        if self.ranks <= 0 or self.ranks % self.tp_size:
            raise ValueError(f"DSpark packing needs a rank count divisible by TP={self.tp_size}, got {ranks}")

    def apply(
        self,
        name: str,
        tensor: torch.Tensor,
        *,
        dtype: torch.dtype | None,
        destination: torch.Tensor | None,
    ) -> torch.Tensor:
        """Replicate, TP-shard, or row-pad ``tensor`` across the rank axis."""
        source = tensor.cpu() if tensor.device.type != "cpu" else tensor
        output_dtype = source.dtype if dtype is None else dtype
        # The `prefill_` copies of the o-projection shard exactly like the decode ones,
        # so the dispatch strips the prefix rather than naming both spellings.
        base = name.removeprefix("prefill_")
        if base in ("wo_a", "wo_b"):
            return self._shard_groups(name, source, output_dtype, destination)
        if base in DSPARK_PADDED_ROW_WEIGHT_NAMES:
            return self._pad_rows(name, source, output_dtype, destination)
        return self._replicate(name, source, output_dtype, destination)

    def _replicate(
        self,
        name: str,
        source: torch.Tensor,
        output_dtype: torch.dtype,
        destination: torch.Tensor | None,
    ) -> torch.Tensor:
        expected = (self.ranks, *source.shape)
        if destination is not None:
            if tuple(destination.shape) != expected or destination.dtype != output_dtype:
                raise ValueError(
                    DSPARK_MISMATCH_ERROR.format(
                        name=name,
                        expected=f"{expected}/{output_dtype}",
                        got=f"{tuple(destination.shape)}/{destination.dtype}",
                    )
                )
            destination.copy_(source.unsqueeze(0))
            return destination
        if source.dtype is not output_dtype:
            source = source.to(dtype=output_dtype)
        return source.contiguous().unsqueeze(0).expand(self.ranks, *source.shape).contiguous()

    def _shard_groups(
        self,
        name: str,
        source: torch.Tensor,
        output_dtype: torch.dtype,
        destination: torch.Tensor | None,
    ) -> torch.Tensor:
        """Give each TP rank 2 of the 8 reshaped output-projection groups.

        ``wo_b`` arrives group-major ``[O_GROUPS, D, O_LORA]`` from the packer's
        ``column_reshape_groups``, so whole-group selection covers it too — a folded
        column slice would cut into the trailing axis the NZ decode bank blocks.
        """
        if source.ndim != 3 or int(source.shape[0]) != DSPARK_O_GROUPS:
            raise ValueError(
                f"{name} must arrive reshaped to [{DSPARK_O_GROUPS}, *, *], got shape={tuple(source.shape)}"
            )
        local_groups = DSPARK_O_GROUPS // self.tp_size
        shards = [
            source[(rank % self.tp_size) * local_groups : (rank % self.tp_size + 1) * local_groups]
            for rank in range(self.ranks)
        ]
        return self._stack_shards(name, shards, output_dtype, destination)

    def _stack_shards(
        self,
        name: str,
        shards: list[torch.Tensor],
        output_dtype: torch.dtype,
        destination: torch.Tensor | None,
    ) -> torch.Tensor:
        expected = (self.ranks, *shards[0].shape)
        if destination is not None:
            if tuple(destination.shape) != expected or destination.dtype != output_dtype:
                raise ValueError(
                    DSPARK_MISMATCH_ERROR.format(
                        name=name,
                        expected=f"{expected}/{output_dtype}",
                        got=f"{tuple(destination.shape)}/{destination.dtype}",
                    )
                )
            for rank, shard in enumerate(shards):
                destination[rank].copy_(shard)
            return destination
        stacked = torch.stack([shard.to(dtype=output_dtype).contiguous() for shard in shards], dim=0)
        return stacked.contiguous()

    def _pad_rows(
        self,
        name: str,
        source: torch.Tensor,
        output_dtype: torch.dtype,
        destination: torch.Tensor | None,
    ) -> torch.Tensor:
        """Zero-pad the HC function rows to the decode bank's storage height."""
        if source.ndim != 2 or int(source.shape[0]) != DSPARK_MIX_HC:
            raise ValueError(f"{name} must have {DSPARK_MIX_HC} rows, got shape={tuple(source.shape)}")
        expected = (self.ranks, DSPARK_HC_FN_STORAGE_ROWS, *source.shape[1:])
        if destination is not None:
            if tuple(destination.shape) != expected or destination.dtype != output_dtype:
                raise ValueError(
                    DSPARK_MISMATCH_ERROR.format(
                        name=name,
                        expected=f"{expected}/{output_dtype}",
                        got=f"{tuple(destination.shape)}/{destination.dtype}",
                    )
                )
            destination.zero_()
            destination[:, :DSPARK_MIX_HC].copy_(source.unsqueeze(0))
            return destination
        packed = torch.zeros(expected, dtype=output_dtype)
        packed[:, :DSPARK_MIX_HC].copy_(source.to(dtype=output_dtype).unsqueeze(0))
        return packed.contiguous()


def dspark_shard_policy(ranks: int) -> DSParkShardPolicy:
    """The DSpark rank policy, carrying the diagnostics its users recognise."""
    return DSParkShardPolicy(ranks=ranks)


def dspark_expert_parallel(ranks: int, n_routed_experts: int):
    """The expert placement policy (identical to the MTP variant)."""
    return deepseek_v4_expert_parallel(ranks, n_routed_experts)


def dspark_factories() -> dict[str, object]:
    """Synthetic-weight factories, shared with the MTP variant."""
    return deepseek_v4_factories()


def dspark_stack_groups(compress_ratios):
    """Stack-group membership: the V4 groups plus the prefill-only copies.

    The prefill o-projection copies fall through to the FWD catch-all (nothing else
    claims them), while the prefill CSA indexer copy joins the CSA group so it stacks
    on that group's layer ids.
    """
    return tuple(
        StackGroup(
            id=group.id,
            members=(*group.members, "prefill_csa_weights_proj") if group.id == "csa" else group.members,
            layer_ids=group.layer_ids,
        )
        for group in deepseek_v4_stack_groups(compress_ratios)
    )


DSPARK_STAGING_POLICY = DEEPSEEK_V4_STAGING_POLICY

# ---- DSpark drafter (milestone 2) ----
# The speculative drafter consumes the checkpoint's ``mtp.0/1/2`` modules plus
# three replicated heads, flattened into 3-layer banks along the leading
# rank-local axis exactly as ``l3_dspark_drafter`` declares them
# (``DSPARK_DRAFT_LAYERS * D`` rows etc.; no decode-bank row padding -- the
# drafter keeps the natural ``MIX_HC`` layout).
DSPARK_DRAFT_LAYERS = 3
# ``tid2eid`` lives only on the target's hash layers; the drafter's router reads
# the same per-token routing tables stacked in draft-layer order.
DSPARK_DRAFTER_HASH_LAYERS = (0, 1, 2)
DSPARK_DRAFTER_LAYERS = (0, 1, 2)

# Kernel names that replicate one checkpoint tensor to every rank unchanged.
DSPARK_DRAFTER_REPLICATED_WEIGHT_NAMES = (
    "main_proj_weight",
    "main_norm_weight",
    "hc_head_fn",
    "hc_head_scale",
    "hc_head_base",
    "final_norm_weight",
    "markov_w1",
    "markov_w2",
    "confidence_head_weight",
)


def dspark_drafter_required_weight_names(n_routed_experts: int) -> tuple[str, ...]:
    """Every checkpoint tensor the drafter + markov programs consume.

    Startup validation for K=7 rejects a checkpoint missing any of these before
    any device work; the K=0 path never loads them.
    """
    names: list[str] = [
        "mtp.0.main_proj.weight",
        "mtp.0.main_norm.weight",
        "mtp.2.norm.weight",
        "mtp.2.markov_head.markov_w1.weight",
        "mtp.2.markov_head.markov_w2.weight",
        "mtp.2.confidence_head.proj.weight",
        "mtp.2.hc_head_fn",
        "mtp.2.hc_head_scale",
        "mtp.2.hc_head_base",
    ]
    for layer in DSPARK_DRAFTER_LAYERS:
        names.extend(
            [
                f"mtp.{layer}.attn_norm.weight",
                f"mtp.{layer}.ffn_norm.weight",
                f"mtp.{layer}.attn.wq_a.weight",
                f"mtp.{layer}.attn.wq_b.weight",
                f"mtp.{layer}.attn.wq_b.scale",
                f"mtp.{layer}.attn.wkv.weight",
                f"mtp.{layer}.attn.q_norm.weight",
                f"mtp.{layer}.attn.kv_norm.weight",
                f"mtp.{layer}.attn.attn_sink",
                f"mtp.{layer}.attn.wo_a.weight",
                f"mtp.{layer}.attn.wo_b.weight",
                f"mtp.{layer}.attn.wo_b.scale",
                f"mtp.{layer}.hc_attn_fn",
                f"mtp.{layer}.hc_attn_scale",
                f"mtp.{layer}.hc_attn_base",
                f"mtp.{layer}.hc_ffn_fn",
                f"mtp.{layer}.hc_ffn_scale",
                f"mtp.{layer}.hc_ffn_base",
                f"mtp.{layer}.ffn.gate.weight",
                f"mtp.{layer}.ffn.gate.bias",
                f"mtp.{layer}.ffn.shared_experts.w1.weight",
                f"mtp.{layer}.ffn.shared_experts.w1.scale",
                f"mtp.{layer}.ffn.shared_experts.w2.weight",
                f"mtp.{layer}.ffn.shared_experts.w2.scale",
                f"mtp.{layer}.ffn.shared_experts.w3.weight",
                f"mtp.{layer}.ffn.shared_experts.w3.scale",
            ]
        )
        names.extend(
            f"mtp.{layer}.ffn.experts.{expert}.{name}"
            for expert in range(int(n_routed_experts))
            for name in ("w1.weight", "w1.scale", "w2.weight", "w2.scale", "w3.weight", "w3.scale")
        )
    names.extend(f"layers.{layer}.ffn.gate.tid2eid" for layer in DSPARK_DRAFTER_HASH_LAYERS)
    return tuple(names)
