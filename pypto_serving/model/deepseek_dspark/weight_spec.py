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
staging is bit-reinterpreted on device).  The packing of three names differs:

* ``wo_a`` / ``wo_b`` are tensor-parallel sharded -- each of the 4 TP ranks owns
  2 of the 8 output-projection groups (and the matching INT8 column slice), and
  the kernel regathers the full projection on device.
* ``hc_attn_fn`` / ``hc_ffn_fn`` are stored by the decode weight bank padded to
  32 storage rows per layer (the kernel's fixed ``HC_FN_STORAGE_ROWS``), while
  the prefill wrapper consumes the natural 24-row ``MIX_HC`` layout -- the loader
  derives the unpadded prefill slab from the padded decode slab.

Stack-group membership is identical to the DeepSeek V4 variant, so
``deepseek_v4_stack_groups`` is reused as-is: the slab shapes come from the
per-layer template this family packs, and the TP-sharded / padded shapes flow
from the same templates.
"""

from __future__ import annotations

import torch

from pypto_serving.model.deepseek.weight_spec import (
    DEEPSEEK_V4_EXPERT_LAYER_RULES,
    DEEPSEEK_V4_CORE_LAYER_RULES,
    DEEPSEEK_V4_LAYER_RULES,
    DEEPSEEK_V4_OPTIONAL_LAYER_RULES,
    DEEPSEEK_V4_ROUTER_LAYER_RULES,
    DEEPSEEK_V4_STAGING_POLICY,
    deepseek_v4_expert_parallel,
    deepseek_v4_factories,
    deepseek_v4_stack_groups,
)

__all__ = [
    "DSPARK_HC_FN_STORAGE_ROWS",
    "DSPARK_LAYER_RULES",
    "DSPARK_MIX_HC",
    "DSPARK_O_PROJ_LOCAL_COLS",
    "DSPARK_O_PROJ_LOCAL_GROUPS",
    "DSPARK_O_PROJ_LOCAL_WIDTH",
    "DSPARK_PADDED_ROW_WEIGHT_NAMES",
    "DSPARK_TP_SIZE",
    "DSParkShardPolicy",
    "dspark_expert_parallel",
    "dspark_factories",
    "dspark_stack_groups",
]

# Canonical 16-card topology: TP4/DP4 => EP16.
DSPARK_TP_SIZE = 4
DSPARK_O_GROUPS = 8
DSPARK_O_PROJ_LOCAL_GROUPS = DSPARK_O_GROUPS // DSPARK_TP_SIZE
DSPARK_O_LORA = 1024
DSPARK_O_PROJ_LOCAL_COLS = DSPARK_O_PROJ_LOCAL_GROUPS * DSPARK_O_LORA
# Full flattened o-projection output width (all groups, one rank's rows).
DSPARK_O_PROJ_LOCAL_WIDTH = DSPARK_O_PROJ_LOCAL_COLS
# The decode weight bank pads the HC function matrices to this many rows.
DSPARK_HC_FN_STORAGE_ROWS = 32
DSPARK_MIX_HC = 24

# The decode bank stores these with zero padding past ``DSPARK_MIX_HC`` rows.
DSPARK_PADDED_ROW_WEIGHT_NAMES = frozenset({"hc_attn_fn", "hc_ffn_fn"})

DSPARK_MISMATCH_ERROR = (
    "packed DSpark destination {name} shape/dtype mismatch: expected={expected}, got={got}"
)

# Layer rules are reused verbatim: the kernels declare every per-layer weight
# with the checkpoint's own dtype (BF16 gains, INT8 weights + FP32 scales), so
# any "convenience" dtype promotion here is a device-side bit reinterpretation.
DSPARK_LAYER_RULES = DEEPSEEK_V4_LAYER_RULES

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
            raise ValueError(
                f"DSpark packing needs a rank count divisible by TP={self.tp_size}, got {ranks}"
            )

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
        if name == "wo_a":
            return self._shard_groups(name, source, output_dtype, destination)
        if name == "wo_b":
            return self._shard_columns(name, source, output_dtype, destination)
        if name in DSPARK_PADDED_ROW_WEIGHT_NAMES:
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
        """Give each TP rank 2 of the 8 reshaped output-projection groups."""
        if source.ndim != 3 or int(source.shape[0]) != DSPARK_O_GROUPS:
            raise ValueError(
                f"{name} must arrive reshaped to [{DSPARK_O_GROUPS}, *, *], "
                f"got shape={tuple(source.shape)}"
            )
        local_groups = DSPARK_O_GROUPS // self.tp_size
        shards = [
            source[(rank % self.tp_size) * local_groups : (rank % self.tp_size + 1) * local_groups]
            for rank in range(self.ranks)
        ]
        return self._stack_shards(name, shards, output_dtype, destination)

    def _shard_columns(
        self,
        name: str,
        source: torch.Tensor,
        output_dtype: torch.dtype,
        destination: torch.Tensor | None,
    ) -> torch.Tensor:
        """Column-slice the INT8 output projection by TP rank."""
        if source.ndim != 2:
            raise ValueError(f"{name} must be rank-2, got shape={tuple(source.shape)}")
        columns = int(source.shape[1])
        if columns % self.tp_size:
            raise ValueError(f"{name} columns {columns} must divide by TP={self.tp_size}")
        width = columns // self.tp_size
        shards = [
            source[:, (rank % self.tp_size) * width : (rank % self.tp_size + 1) * width]
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
        stacked = torch.stack(
            [shard.to(dtype=output_dtype).contiguous() for shard in shards], dim=0
        )
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
            raise ValueError(
                f"{name} must have {DSPARK_MIX_HC} rows, got shape={tuple(source.shape)}"
            )
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
    """Stack-group membership (identical to the MTP variant)."""
    return deepseek_v4_stack_groups(compress_ratios)


DSPARK_STAGING_POLICY = DEEPSEEK_V4_STAGING_POLICY

# Re-exported for the loader's rule tuple assembly; kept explicit so the
# dspark package does not reach into the sibling's private grouping.
DSPARK_CORE_LAYER_RULES = DEEPSEEK_V4_CORE_LAYER_RULES
DSPARK_OPTIONAL_LAYER_RULES = DEEPSEEK_V4_OPTIONAL_LAYER_RULES
DSPARK_ROUTER_LAYER_RULES = DEEPSEEK_V4_ROUTER_LAYER_RULES
DSPARK_EXPERT_LAYER_RULES = DEEPSEEK_V4_EXPERT_LAYER_RULES
