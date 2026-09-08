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

``load_drafter_weights`` additionally packs the milestone-2 speculative
drafter: the checkpoint's ``mtp.0/1/2`` modules, three replicated heads, and
the hash layers' ``tid2eid`` tables, flattened into the exact bank shapes
``l3_dspark_drafter`` / ``l3_distributed_markov_sample`` declare.

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
    deepseek_v4_local_expert_ids,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DSparkDrafterWeights",
    "DSparkStackedLayerWeights",
    "DSparkWeightStore",
    "dspark_prefill_hc_slab",
]


@dataclass(frozen=True)
class DSparkDrafterWeights:
    """Drafter + markov banks, each already ``[ranks, ...]``-stacked on host.

    Every tensor matches the per-rank annotation of ``l3_dspark_drafter`` or
    ``l3_distributed_markov_sample`` with a leading rank axis: the 3-layer
    banks flatten the draft layers along their first rank-local axis (no
    decode-bank row padding -- the drafter keeps the natural ``MIX_HC``
    layout), the o-projection is TP group/column sharded per rank, and the
    routed experts are EP sharded.  ``embedding_weight`` / ``lm_head_weight``
    are reused from the target banks and deliberately absent here.
    """

    tensors: Mapping[str, torch.Tensor]

    def args(self, names: Sequence[str]) -> tuple[torch.Tensor, ...]:
        """Return stacked tensors in a kernel host order."""
        missing = [name for name in names if name not in self.tensors]
        if missing:
            raise KeyError(f"Stacked DSpark drafter weights are missing: {', '.join(missing)}")
        return tuple(self.tensors[name] for name in names)


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

    def validate_drafter_startup_contract(self, *, n_routed_experts: int) -> None:
        """Reject a checkpoint without the full drafter module set (K=7 only)."""
        from pypto_serving.model.deepseek_dspark.weight_spec import (  # noqa: PLC0415
            dspark_drafter_required_weight_names,
        )

        self.require(dspark_drafter_required_weight_names(n_routed_experts))

    def load_drafter_weights(
        self,
        *,
        ranks: int,
        n_routed_experts: int,
    ) -> DSparkDrafterWeights:
        """Pack the ``mtp.0/1/2`` drafter into the kernel's stacked banks.

        The banks flatten the three draft layers along each tensor's first
        rank-local axis exactly as ``l3_dspark_drafter`` declares them, with
        per-name transforms the target banks do not need: the checkpoint's
        ``[out, in]`` projections transpose into the kernel's ``[in, out]``
        matmul orientation (``wq_a``/``wq_b``/``wkv``), the router gate and
        confidence head cast to FP32, ``tid2eid`` stacks from the target's
        hash layers as INT32, and the o-projection / routed experts are
        TP- / EP-sharded per rank.  Model dims are inferred from the loaded
        tensors and cross-checked so a mismatched checkpoint fails here, not
        on device.
        """
        from pypto_serving.model.deepseek_dspark.weight_spec import (  # noqa: PLC0415
            DSPARK_DRAFT_LAYERS,
            DSPARK_DRAFTER_HASH_LAYERS,
            DSPARK_DRAFTER_LAYERS,
            DSPARK_TP_SIZE,
        )

        if ranks <= 0 or ranks % DSPARK_TP_SIZE:
            raise ValueError(
                f"DSpark drafter packing needs a rank count divisible by TP={DSPARK_TP_SIZE}, "
                f"got {ranks}"
            )
        if n_routed_experts % ranks:
            raise ValueError(
                f"DSpark drafter needs experts divisible by ranks={ranks}, "
                f"got {n_routed_experts}"
            )
        self.validate_drafter_startup_contract(n_routed_experts=n_routed_experts)

        # ---- replicated heads; these also fix the reference dims ----
        heads = self.load_many(
            [
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
        )
        main_proj = heads["mtp.0.main_proj.weight"]
        main_norm = heads["mtp.0.main_norm.weight"]
        markov_w1 = heads["mtp.2.markov_head.markov_w1.weight"]
        markov_w2 = heads["mtp.2.markov_head.markov_w2.weight"]
        confidence = heads["mtp.2.confidence_head.proj.weight"]
        if main_proj.ndim != 2 or main_norm.ndim != 1:
            raise ValueError("DSpark main projection must be rank-2 and its norm rank-1")
        hidden = int(main_norm.shape[0])
        if tuple(main_proj.shape) != (hidden, DSPARK_DRAFT_LAYERS * hidden):
            raise ValueError(
                f"DSpark main_proj must be [D, 3*D] with D={hidden}, got {tuple(main_proj.shape)}"
            )
        if markov_w1.ndim != 2 or tuple(markov_w1.shape) != tuple(markov_w2.shape):
            raise ValueError(
                "DSpark markov tables must be matching rank-2, got "
                f"{tuple(markov_w1.shape)} / {tuple(markov_w2.shape)}"
            )
        vocab, markov_rank = (int(dim) for dim in markov_w1.shape)
        if tuple(confidence.shape) != (1, hidden + markov_rank):
            raise ValueError(
                f"DSpark confidence head must be [1, {hidden + markov_rank}], "
                f"got {tuple(confidence.shape)}"
            )

        # ---- per-layer tensors, one shard-grouped read per draft layer ----
        raw_layers = [
            self.load_many(
                [
                    f"mtp.{layer}.{suffix}"
                    for suffix in (
                        "attn_norm.weight",
                        "ffn_norm.weight",
                        "attn.wq_a.weight",
                        "attn.wq_b.weight",
                        "attn.wq_b.scale",
                        "attn.wkv.weight",
                        "attn.q_norm.weight",
                        "attn.kv_norm.weight",
                        "attn.attn_sink",
                        "attn.wo_a.weight",
                        "attn.wo_b.weight",
                        "attn.wo_b.scale",
                        "hc_attn_fn",
                        "hc_attn_scale",
                        "hc_attn_base",
                        "hc_ffn_fn",
                        "hc_ffn_scale",
                        "hc_ffn_base",
                        "ffn.gate.weight",
                        "ffn.gate.bias",
                        "ffn.shared_experts.w1.weight",
                        "ffn.shared_experts.w1.scale",
                        "ffn.shared_experts.w2.weight",
                        "ffn.shared_experts.w2.scale",
                        "ffn.shared_experts.w3.weight",
                        "ffn.shared_experts.w3.scale",
                    )
                ]
            )
            for layer in DSPARK_DRAFTER_LAYERS
        ]

        def tensor(layer: int, suffix: str) -> torch.Tensor:
            return raw_layers[layer][f"mtp.{DSPARK_DRAFTER_LAYERS[layer]}.{suffix}"]

        # ---- infer and cross-check the per-layer dims from layer zero ----
        wq_a0 = tensor(0, "attn.wq_a.weight")
        wq_b0 = tensor(0, "attn.wq_b.weight")
        wkv0 = tensor(0, "attn.wkv.weight")
        sink0 = tensor(0, "attn.attn_sink")
        wo_a0 = tensor(0, "attn.wo_a.weight")
        wo_b0 = tensor(0, "attn.wo_b.weight")
        gate0 = tensor(0, "ffn.gate.weight")
        shared_w1_0 = tensor(0, "ffn.shared_experts.w1.weight")
        shared_w2_0 = tensor(0, "ffn.shared_experts.w2.weight")
        hc_attn0 = tensor(0, "hc_attn_fn")
        if wq_a0.ndim != 2 or wq_b0.ndim != 2 or wkv0.ndim != 2:
            raise ValueError("DSpark drafter projections must be rank-2")
        q_lora = int(wq_a0.shape[0])
        q_width = int(wq_b0.shape[0])
        head_dim = int(wkv0.shape[0])
        n_heads = int(sink0.shape[0])
        if q_width != n_heads * head_dim:
            raise ValueError(
                f"DSpark wq_b width {q_width} != heads*head_dim {n_heads * head_dim}"
            )
        if int(tensor(0, "attn.q_norm.weight").shape[0]) != q_lora:
            raise ValueError("DSpark q_norm length disagrees with the q-lora dim")
        if int(tensor(0, "attn.kv_norm.weight").shape[0]) != head_dim:
            raise ValueError("DSpark kv_norm length disagrees with the head dim")
        if wq_a0.shape[1] != hidden or wkv0.shape[1] != hidden:
            raise ValueError("DSpark projection inputs disagree with the model dim")
        o_lora = q_lora
        o_groups = int(wo_a0.shape[0]) // o_lora
        if int(wo_a0.shape[0]) % o_lora or o_groups <= 0:
            raise ValueError(
                f"DSpark wo_a shape {tuple(wo_a0.shape)} disagrees with the o-lora dim {o_lora}"
            )
        # wo_a is block-diagonal: each of its O_GROUPS row groups reads its own
        # q_width/O_GROUPS-wide input slice, so the column count is the per-group
        # input, not the full projection width.
        o_group_in = q_width // o_groups
        if int(wo_a0.shape[1]) != o_group_in or o_group_in * o_groups != q_width:
            raise ValueError(
                f"DSpark wo_a columns {int(wo_a0.shape[1])} disagree with the per-group "
                f"input width {o_group_in}"
            )
        if tuple(wo_b0.shape) != (hidden, o_groups * o_lora):
            raise ValueError(
                f"DSpark wo_b must be [D, O_GROUPS*O_LORA]={(hidden, o_groups * o_lora)}, "
                f"got {tuple(wo_b0.shape)}"
            )
        if o_groups % DSPARK_TP_SIZE:
            raise ValueError(
                f"DSpark o-groups {o_groups} must divide by TP={DSPARK_TP_SIZE}"
            )
        local_o_groups = o_groups // DSPARK_TP_SIZE
        local_o_width = local_o_groups * o_lora
        if int(gate0.shape[0]) != n_routed_experts or gate0.shape[1] != hidden:
            raise ValueError(
                f"DSpark gate must be [{n_routed_experts}, {hidden}], got {tuple(gate0.shape)}"
            )
        moe_inter = int(shared_w1_0.shape[0])
        if tuple(shared_w2_0.shape) != (hidden, moe_inter):
            raise ValueError(
                "DSpark shared expert shapes disagree: "
                f"w1={tuple(shared_w1_0.shape)}, w2={tuple(shared_w2_0.shape)}"
            )
        if hc_attn0.ndim != 2:
            raise ValueError("DSpark HC function matrices must be rank-2")
        mix_hc = int(hc_attn0.shape[0])
        hc_dim = int(hc_attn0.shape[1])
        n_local = n_routed_experts // ranks

        tensors: dict[str, torch.Tensor] = {}

        def bank(
            name: str, per_layer_shape: tuple[int, ...], dtype: torch.dtype
        ) -> torch.Tensor:
            rows = per_layer_shape[0]
            destination = torch.empty(
                (ranks, DSPARK_DRAFT_LAYERS * rows, *per_layer_shape[1:]), dtype=dtype
            )
            tensors[name] = destination
            return destination

        def fill_flat(name: str, destination: torch.Tensor, rows: int, layer: int) -> torch.Tensor:
            return destination[:, layer * rows : (layer + 1) * rows]

        def transpose2d(source: torch.Tensor) -> torch.Tensor:
            return source.t().contiguous()

        # (name, suffix, dtype, per-layer shape, transform); transform maps a
        # checkpoint tensor into the kernel's matmul orientation, never dtype.
        flat_specs: list[tuple[str, str, torch.dtype, tuple[int, ...], object]] = [
            ("attn_norm_w", "attn_norm.weight", torch.bfloat16, (hidden,), None),
            ("ffn_norm_w", "ffn_norm.weight", torch.bfloat16, (hidden,), None),
            ("wq_a", "attn.wq_a.weight", torch.bfloat16, (hidden, q_lora), transpose2d),
            ("wq_b", "attn.wq_b.weight", torch.int8, (q_lora, q_width), transpose2d),
            ("wq_b_scale", "attn.wq_b.scale", torch.float32, (q_width,), None),
            ("wkv", "attn.wkv.weight", torch.bfloat16, (hidden, head_dim), transpose2d),
            ("gamma_cq", "attn.q_norm.weight", torch.bfloat16, (q_lora,), None),
            ("gamma_ckv", "attn.kv_norm.weight", torch.bfloat16, (head_dim,), None),
            ("attn_sink", "attn.attn_sink", torch.float32, (n_heads,), None),
            ("wo_b_scale", "attn.wo_b.scale", torch.float32, (hidden,), None),
            ("hc_attn_fn", "hc_attn_fn", torch.float32, (mix_hc, hc_dim), None),
            ("hc_attn_scale", "hc_attn_scale", torch.float32, (3,), None),
            ("hc_attn_base", "hc_attn_base", torch.float32, (mix_hc,), None),
            ("hc_ffn_fn", "hc_ffn_fn", torch.float32, (mix_hc, hc_dim), None),
            ("hc_ffn_scale", "hc_ffn_scale", torch.float32, (3,), None),
            ("hc_ffn_base", "hc_ffn_base", torch.float32, (mix_hc,), None),
            (
                "shared_w1",
                "ffn.shared_experts.w1.weight",
                torch.int8,
                (moe_inter, hidden),
                None,
            ),
            (
                "shared_w1_scale",
                "ffn.shared_experts.w1.scale",
                torch.float32,
                (moe_inter,),
                None,
            ),
            (
                "shared_w3",
                "ffn.shared_experts.w3.weight",
                torch.int8,
                (moe_inter, hidden),
                None,
            ),
            (
                "shared_w3_scale",
                "ffn.shared_experts.w3.scale",
                torch.float32,
                (moe_inter,),
                None,
            ),
            (
                "shared_w2",
                "ffn.shared_experts.w2.weight",
                torch.int8,
                (hidden, moe_inter),
                None,
            ),
            (
                "shared_w2_scale",
                "ffn.shared_experts.w2.scale",
                torch.float32,
                (hidden,),
                None,
            ),
        ]
        banks: dict[str, torch.Tensor] = {}
        for name, _, dtype, shape, _ in flat_specs:
            banks[name] = bank(name, shape, dtype)
        for layer in range(DSPARK_DRAFT_LAYERS):
            for name, suffix, dtype, shape, transform in flat_specs:
                source = tensor(layer, suffix)
                prepared = transform(source) if transform is not None else source
                if tuple(prepared.shape) != shape:
                    raise ValueError(
                        f"DSpark drafter {name} layer {layer} must be {shape}, "
                        f"got {tuple(prepared.shape)}"
                    )
                if prepared.dtype is not dtype:
                    raise ValueError(
                        f"DSpark drafter {name} layer {layer} must be {dtype}, "
                        f"got {prepared.dtype}"
                    )
                fill_flat(name, banks[name], shape[0], layer).copy_(prepared)
        del banks

        # Router gate: the checkpoint stores BF16 weights; the bank is FP32.
        gate_w = bank("gate_w", (n_routed_experts, hidden), torch.float32)
        gate_bias = bank("gate_bias", (n_routed_experts,), torch.float32)
        for layer in range(DSPARK_DRAFT_LAYERS):
            fill_flat("gate_w", gate_w, n_routed_experts, layer).copy_(
                tensor(layer, "ffn.gate.weight").to(torch.float32)
            )
            fill_flat("gate_bias", gate_bias, n_routed_experts, layer).copy_(
                tensor(layer, "ffn.gate.bias")
            )

        # tid2eid: the target hash layers' routing tables, INT32, every rank.
        tid2eid = torch.cat(
            [
                self.load_tensor(f"layers.{layer}.ffn.gate.tid2eid").to(torch.int32)
                for layer in DSPARK_DRAFTER_HASH_LAYERS
            ],
            dim=0,
        ).contiguous()
        if int(tid2eid.shape[0]) != DSPARK_DRAFT_LAYERS * vocab:
            raise ValueError(
                f"DSpark tid2eid covers {int(tid2eid.shape[0])} rows, "
                f"expected {DSPARK_DRAFT_LAYERS * vocab}"
            )
        tensors["tid2eid"] = (
            tid2eid.unsqueeze(0).expand(ranks, *tid2eid.shape).contiguous()
        )
        del tid2eid

        # O-projection: TP group shard of wo_a, column slice of wo_b.
        wo_a = bank(
            "wo_a",
            (local_o_groups, o_lora, int(wo_a0.shape[1])),
            torch.bfloat16,
        )
        wo_b = bank(
            "wo_b",
            (hidden, local_o_width),
            torch.int8,
        )
        wo_a_sources = [
            self.load_tensor(f"mtp.{layer}.attn.wo_a.weight")
            .reshape(o_groups, o_lora, int(wo_a0.shape[1]))
            .contiguous()
            for layer in DSPARK_DRAFTER_LAYERS
        ]
        wo_b_sources = [
            self.load_tensor(f"mtp.{layer}.attn.wo_b.weight")
            for layer in DSPARK_DRAFTER_LAYERS
        ]
        for rank in range(ranks):
            tp_rank = rank % DSPARK_TP_SIZE
            for layer in range(DSPARK_DRAFT_LAYERS):
                wo_a[rank, layer * local_o_groups : (layer + 1) * local_o_groups].copy_(
                    wo_a_sources[layer][
                        tp_rank * local_o_groups : (tp_rank + 1) * local_o_groups
                    ]
                )
                wo_b[rank, layer * hidden : (layer + 1) * hidden].copy_(
                    wo_b_sources[layer][
                        :, tp_rank * local_o_width : (tp_rank + 1) * local_o_width
                    ]
                )
        del wo_a_sources, wo_b_sources

        # Routed experts: EP sharded per rank, flattened along (layer, local);
        # each bank row is one expert, so the tail (not the leading axis) is
        # the expert's own shape.
        routed_specs = (
            ("routed_w1", "w1.weight", (moe_inter, hidden), torch.int8),
            ("routed_w1_scale", "w1.scale", (moe_inter,), torch.float32),
            ("routed_w3", "w3.weight", (moe_inter, hidden), torch.int8),
            ("routed_w3_scale", "w3.scale", (moe_inter,), torch.float32),
            ("routed_w2", "w2.weight", (hidden, moe_inter), torch.int8),
            ("routed_w2_scale", "w2.scale", (hidden,), torch.float32),
        )
        expert_banks: dict[str, torch.Tensor] = {}
        for name, _, tail, dtype in routed_specs:
            destination = torch.empty(
                (ranks, DSPARK_DRAFT_LAYERS * n_local, *tail), dtype=dtype
            )
            tensors[name] = destination
            expert_banks[name] = destination
        for layer in range(DSPARK_DRAFT_LAYERS):
            prefix_layer = DSPARK_DRAFTER_LAYERS[layer]
            raw = self.load_many(
                [
                    f"mtp.{prefix_layer}.ffn.experts.{expert}.{suffix}"
                    for expert in range(n_routed_experts)
                    for _, suffix, _, _ in routed_specs
                ]
            )
            for rank in range(ranks):
                local_ids = deepseek_v4_local_expert_ids(
                    rank=rank, ranks=ranks, n_routed_experts=n_routed_experts
                )
                for local_index, expert in enumerate(local_ids):
                    row = layer * n_local + local_index
                    prefix = f"mtp.{prefix_layer}.ffn.experts.{expert}"
                    for name, suffix, shape, _ in routed_specs:
                        source = raw[f"{prefix}.{suffix}"]
                        if tuple(source.shape) != shape:
                            raise ValueError(
                                f"DSpark drafter expert {expert} {name} must be {shape}, "
                                f"got {tuple(source.shape)}"
                            )
                        expert_banks[name][rank, row].copy_(source)
            del raw
        del expert_banks

        # Replicated heads.
        def replicate(name: str, source: torch.Tensor, dtype: torch.dtype) -> None:
            if source.dtype is not dtype:
                source = source.to(dtype=dtype)
            tensors[name] = (
                source.contiguous().unsqueeze(0).expand(ranks, *source.shape).contiguous()
            )

        replicate("main_proj_weight", main_proj, torch.bfloat16)
        replicate("main_norm_weight", main_norm, torch.bfloat16)
        replicate("hc_head_fn", heads["mtp.2.hc_head_fn"], torch.float32)
        replicate("hc_head_scale", heads["mtp.2.hc_head_scale"], torch.float32)
        replicate("hc_head_base", heads["mtp.2.hc_head_base"], torch.float32)
        replicate("final_norm_weight", heads["mtp.2.norm.weight"], torch.bfloat16)
        replicate("markov_w1", markov_w1, torch.bfloat16)
        replicate("markov_w2", markov_w2, torch.bfloat16)
        # The confidence head ships BF16 but the kernel consumes it as FP32.
        replicate("confidence_head_weight", confidence, torch.float32)
        del heads

        total_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
        logger.info(
            "DSpark drafter weights loaded: %d banks, %.2f GiB host across %d ranks",
            len(tensors),
            total_bytes / (1 << 30),
            ranks,
        )
        return DSparkDrafterWeights(tensors=tensors)
