# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DSpark W8A8 weight loading.

``DSparkWeightStore`` inherits the DeepSeek V4 safetensors contract unchanged
(same index handling, startup validation, global weights, LM-head packing) and
rebinds only the layer packing: the DSpark shard policy (TP-sharded o-proj,
bank-padded HC function rows) and the extra unpadded prefill HC slabs.

There is deliberately no prepacked-sidecar path here: the DSpark slabs are
packed from the shards on every start through the standard lazy store.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from pypto_serving.model.deepseek.weight_loader import (
    DeepSeekV4PackedLayerWeights,
    DeepSeekV4WeightStore,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DSparkStackedLayerWeights",
    "DSparkWeightStore",
    "dspark_prefill_hc_slab",
]


@dataclass(frozen=True)
class DSparkStackedLayerWeights:
    """All hidden-layer weights stacked on the layer axis for the DSpark kernels.

    ``tensors`` holds the decode weight-bank layout (HC function matrices padded
    to ``DSPARK_HC_FN_STORAGE_ROWS`` rows); ``prefill_tensors`` holds the two
    prefill-only unpadded HC slabs.  Every other name is shared by both
    dispatch classes and appears once, in ``tensors``.
    """

    tensors: Mapping[str, torch.Tensor]
    prefill_tensors: Mapping[str, torch.Tensor]

    def args(self, names: Sequence[str]) -> tuple[torch.Tensor, ...]:
        """Return stacked tensors in a kernel host order."""
        missing = [name for name in names if name not in self.tensors]
        if missing:
            raise KeyError(f"Stacked DSpark weights are missing tensors: {', '.join(missing)}")
        return tuple(self.tensors[name] for name in names)


def dspark_prefill_hc_slab(
    padded: torch.Tensor,
    *,
    layers: int,
    mix_hc_rows: int,
    storage_rows: int,
) -> torch.Tensor:
    """Strip the decode bank's zero padding from one HC function slab.

    ``padded`` is ``[ranks, layers * storage_rows, width]``; the prefill wrapper
    consumes ``[ranks, layers * mix_hc_rows, width]``.  Narrow-and-copy per
    layer rather than a view: the padded rows sit between consecutive layers,
    so the unpadded slab is not contiguous in the source.
    """
    if padded.ndim != 3:
        raise ValueError(f"padded HC slab must be rank-3, got shape={tuple(padded.shape)}")
    rows = int(padded.shape[1])
    if rows != layers * storage_rows:
        raise ValueError(
            f"padded HC slab has {rows} rows, expected {layers} x {storage_rows}"
        )
    unpadded = torch.empty(
        (padded.shape[0], layers * mix_hc_rows, padded.shape[2]),
        dtype=padded.dtype,
    )
    for layer in range(layers):
        source = padded[:, layer * storage_rows : layer * storage_rows + mix_hc_rows]
        unpadded[:, layer * mix_hc_rows : (layer + 1) * mix_hc_rows].copy_(source)
    return unpadded.contiguous()


class DSparkWeightStore(DeepSeekV4WeightStore):
    """Lazy W8A8 store that packs layers into the DSpark kernel layouts."""

    def load_packed_layer_weights(
        self,
        layer_id: int,
        *,
        ranks: int,
        n_routed_experts: int,
        compress_ratio: int = 0,
        include_tid2eid: bool = False,
        include_gate_bias: bool = False,
        expert_ids=None,
        destinations: Mapping[str, torch.Tensor] | None = None,
    ) -> DeepSeekV4PackedLayerWeights:
        """Pack one layer under the DSpark rank layout (TP o-proj, padded HC)."""
        from pypto_serving.model.common.weights.packer import pack_layer  # noqa: PLC0415
        from pypto_serving.model.common.weights.spec import LayerContext  # noqa: PLC0415

        from pypto_serving.model.deepseek.weight_loader import (  # noqa: PLC0415
            deepseek_v4_layer_weight_names,
        )
        from pypto_serving.model.deepseek_dspark.weight_spec import (  # noqa: PLC0415
            DSPARK_EXPERT_MISSING_ERROR,
            DSPARK_LAYER_RULES,
            DSPARK_SOURCE_MISSING_ERROR,
            dspark_expert_parallel,
            dspark_factories,
            dspark_shard_policy,
        )

        all_experts = range(n_routed_experts) if expert_ids is None else tuple(expert_ids)
        raw = self.load_many(
            deepseek_v4_layer_weight_names(
                layer_id,
                n_routed_experts=n_routed_experts,
                compress_ratio=compress_ratio,
                include_tid2eid=include_tid2eid,
                include_gate_bias=include_gate_bias,
                expert_ids=all_experts,
            )
        )
        context = LayerContext(
            layer_id=int(layer_id),
            prefix=f"layers.{int(layer_id)}",
            ranks=int(ranks),
            compress_ratio=int(compress_ratio),
            n_routed_experts=int(n_routed_experts),
            include_tid2eid=bool(include_tid2eid),
            include_gate_bias=bool(include_gate_bias),
        )
        tensors = pack_layer(
            DSPARK_LAYER_RULES,
            raw,
            context,
            policy=dspark_shard_policy(int(ranks)),
            expert_policy=dspark_expert_parallel(int(ranks), int(n_routed_experts)),
            factories=dspark_factories(),
            destinations=destinations,
            missing_source_error=DSPARK_SOURCE_MISSING_ERROR,
            missing_expert_error=DSPARK_EXPERT_MISSING_ERROR,
        )
        return DeepSeekV4PackedLayerWeights(layer_id=layer_id, tensors=tensors)

    def load_stacked_layer_weights(
        self,
        *,
        ranks: int,
        n_routed_experts: int,
        compress_ratios: Sequence[int],
        num_hash_layers: int,
        use_prepacked: bool = True,
    ) -> DSparkStackedLayerWeights:
        """Load every hidden layer once and stack the DSpark weight banks.

        The FWD bank stacks all 43 layers, the CSA bank the 21 ratio-4 layers,
        and the HCA bank the 20 ratio-128 layers, each flattened along the first
        rank-local axis.  The two HC function matrices are stacked in the
        decode-padded layout and re-derived unpadded for prefill.
        """
        from pypto_serving.model.common.weights.stacker import stack_layers  # noqa: PLC0415

        from pypto_serving.model.deepseek_dspark.weight_spec import (  # noqa: PLC0415
            DSPARK_HC_FN_STORAGE_ROWS,
            DSPARK_MIX_HC,
            DSPARK_RANK_ERROR,
            DSPARK_STACK_MISMATCH_ERROR,
            DSPARK_STAGING_POLICY,
            dspark_stack_groups,
        )

        if use_prepacked:
            # No prepack sidecar exists for the DSpark layout by design; the
            # keyword stays so the inherited call sites read unchanged.
            use_prepacked = False
        num_hidden_layers = len(compress_ratios)
        if num_hidden_layers <= 0:
            raise ValueError("compress_ratios must include at least one entry per hidden layer")

        first = self.load_packed_layer_weights(
            0,
            ranks=ranks,
            n_routed_experts=n_routed_experts,
            compress_ratio=int(compress_ratios[0]),
            include_tid2eid=num_hash_layers > 0,
            include_gate_bias=num_hash_layers <= 0,
        )

        def pack_into(layer_id: int, destinations: Mapping[str, torch.Tensor]) -> None:
            self.load_packed_layer_weights(
                layer_id,
                ranks=ranks,
                n_routed_experts=n_routed_experts,
                compress_ratio=int(compress_ratios[layer_id]),
                include_tid2eid=layer_id < num_hash_layers,
                include_gate_bias=layer_id >= num_hash_layers,
                destinations=destinations,
            )

        def log_progress(layer_id: int) -> None:
            if layer_id % 5 == 0 or layer_id == num_hidden_layers - 1:
                logger.info("DSpark weight load progress: layer %d/%d", layer_id + 1, num_hidden_layers)

        stacked = stack_layers(
            dspark_stack_groups(compress_ratios),
            first.tensors,
            layer_ids=range(num_hidden_layers),
            pack_into=pack_into,
            template_layer_id=0,
            on_layer_done=log_progress,
            policy=DSPARK_STAGING_POLICY,
            rank_error=DSPARK_RANK_ERROR,
            mismatch_error=DSPARK_STACK_MISMATCH_ERROR,
        )

        prefill_tensors = {
            name: dspark_prefill_hc_slab(
                stacked[name],
                layers=num_hidden_layers,
                mix_hc_rows=DSPARK_MIX_HC,
                storage_rows=DSPARK_HC_FN_STORAGE_ROWS,
            )
            for name in ("hc_attn_fn", "hc_ffn_fn")
        }
        return DSparkStackedLayerWeights(tensors=stacked, prefill_tensors=prefill_tensors)
