# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Checkpoint contracts for the DeepSeek V4.1 DSpark drafter.

The V4.1 backbone loader intentionally excludes deferred draft modules.  This
module keeps their checkpoint contract separate from backbone loading and only
loads the small replicated heads needed by the Markov stage.  Layer weights are
exposed through per-layer source names.  The packed entry point below processes
one EP rank and one draft layer at a time; callers can retain the result in a
resident bank without ever materialising the other ranks' experts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from .weight_loader import mxfp4_pair_lut
from .weight_spec import fp8_input_major, pack_fp4_tiles, pack_grouped_wo_a


DSPARK_DRAFT_LAYERS = (0, 1, 2)

# Raw checkpoint names and the names used by l3_dspark_drafter /
# l3_markov_sample_from_hidden.  The projections are still in checkpoint
# orientation; the eventual packer owns transpose, quantization and EP/TP
# placement.  Keeping this map explicit prevents accidental use of backbone
# layers as draft weights.
DSPARK_HEAD_SOURCE_TO_ABI = {
    "mtp.0.main_proj.weight": "main_proj_weight",
    "mtp.0.main_norm.weight": "main_norm_weight",
    "mtp.2.norm.weight": "final_norm_weight",
    "mtp.2.markov_head.markov_w1.weight": "markov_w1",
    "mtp.2.markov_head.markov_w2.weight": "markov_w2",
    "mtp.2.confidence_head.proj.weight": "confidence_weight",
}

# The layer ABI uses short names while checkpoints keep the module hierarchy.
# ``load_layer_abi`` maps names only; packing, TP/EP sharding and device upload
# remain separate because this contract intentionally stages one layer at a time.
DSPARK_LAYER_SOURCE_TO_ABI = {
    "attn_norm.weight": "attn_norm_weight",
    "ffn_norm.weight": "ffn_norm_weight",
    "attn.wq_a.weight": "wq_a",
    "attn.wq_a.scale": "wq_a_scale",
    "attn.wq_b.weight": "wq_b",
    "attn.wq_b.scale": "wq_b_scale",
    "attn.wkv.weight": "wkv",
    "attn.wkv.scale": "wkv_scale",
    "attn.q_norm.weight": "q_norm_weight",
    "attn.kv_norm.weight": "kv_norm_weight",
    "attn.attn_sink": "attn_sink",
    "attn.wo_a.weight": "wo_a",
    "attn.wo_a.scale": "wo_a_scale",
    "attn.wo_b.weight": "wo_b",
    "attn.wo_b.scale": "wo_b_scale",
    "hc_attn_fn": "hc_attn_fn",
    "hc_attn_scale": "hc_attn_scale",
    "hc_attn_base": "hc_attn_base",
    "hc_ffn_fn": "hc_ffn_fn",
    "hc_ffn_scale": "hc_ffn_scale",
    "hc_ffn_base": "hc_ffn_base",
    "ffn.gate.weight": "gate_weight",
    "ffn.gate.bias": "correction_bias",
    "ffn.shared_experts.w1.weight": "shared_w1",
    "ffn.shared_experts.w1.scale": "shared_w1_scale",
    "ffn.shared_experts.w2.weight": "shared_w2",
    "ffn.shared_experts.w2.scale": "shared_w2_scale",
    "ffn.shared_experts.w3.weight": "shared_w3",
    "ffn.shared_experts.w3.scale": "shared_w3_scale",
}

DSPARK_EXPERT_SOURCE_TO_ABI = {
    "w1.weight": "routed_w1",
    "w1.scale": "routed_w1_scale",
    "w2.weight": "routed_w2",
    "w2.scale": "routed_w2_scale",
    "w3.weight": "routed_w3",
    "w3.scale": "routed_w3_scale",
}

_LAYER_SUFFIXES = (
    "attn_norm.weight",
    "ffn_norm.weight",
    "attn.wq_a.weight",
    "attn.wq_a.scale",
    "attn.wq_b.weight",
    "attn.wq_b.scale",
    "attn.wkv.weight",
    "attn.wkv.scale",
    "attn.q_norm.weight",
    "attn.kv_norm.weight",
    "attn.attn_sink",
    "attn.wo_a.weight",
    "attn.wo_a.scale",
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


def dspark_v41_required_weight_names(n_routed_experts: int) -> tuple[str, ...]:
    """Return the deferred checkpoint names required by the K=5 DSpark path."""
    if type(n_routed_experts) is not int or n_routed_experts <= 0:
        raise ValueError("n_routed_experts must be a positive integer")
    names = list(DSPARK_HEAD_SOURCE_TO_ABI)
    for layer in DSPARK_DRAFT_LAYERS:
        names.extend(f"mtp.{layer}.{suffix}" for suffix in _LAYER_SUFFIXES)
        names.extend(
            f"mtp.{layer}.ffn.experts.{expert}.{suffix}"
            for expert in range(n_routed_experts)
            for suffix in ("w1.weight", "w1.scale", "w2.weight", "w2.scale", "w3.weight", "w3.scale")
        )
    return tuple(names)


def dspark_v41_layer_weight_names(layer: int, n_routed_experts: int) -> tuple[str, ...]:
    """Return one draft layer's raw names for bounded, per-layer staging."""
    if type(layer) is not int or layer not in DSPARK_DRAFT_LAYERS:
        raise ValueError("DSpark draft layer must be one of 0, 1 or 2")
    prefix = f"mtp.{layer}."
    names = [f"{prefix}{suffix}" for suffix in _LAYER_SUFFIXES]
    names.extend(
        f"{prefix}ffn.experts.{expert}.{suffix}"
        for expert in range(n_routed_experts)
        for suffix in DSPARK_EXPERT_SOURCE_TO_ABI
    )
    return tuple(names)


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


@dataclass(frozen=True)
class V41DSparkHeadWeights:
    """Small replicated heads in the ABI names expected by the lib stages."""

    tensors: Mapping[str, torch.Tensor]

    def __getitem__(self, name: str) -> torch.Tensor:
        return self.tensors[name]


@dataclass(frozen=True)
class V41DSparkLayerWeights:
    """One draft layer under the lib ABI names, still in checkpoint layout.

    ``routed_experts`` keeps the source expert index so an EP packer can select
    only its local experts without first materializing all three draft layers.
    """

    tensors: Mapping[str, torch.Tensor]
    routed_experts: Mapping[str, Mapping[int, torch.Tensor]]

    def __getitem__(self, name: str) -> torch.Tensor:
        return self.tensors[name]


@dataclass(frozen=True)
class V41DSparkPackedWeights:
    """Packed draft weights for one EP rank.

    ``load_packed_weights`` returns tensors without an EP axis.  Use
    :meth:`stack_ranks` after every rank has been packed to create the leading
    axis required by the host composite.  Keeping these two operations separate
    lets a caller fill a resident bank incrementally instead of retaining every
    rank's temporary source tensors.
    """

    tensors: Mapping[str, torch.Tensor]

    def __getitem__(self, name: str) -> torch.Tensor:
        return self.tensors[name]

    @classmethod
    def stack_ranks(
        cls, values: Sequence["V41DSparkPackedWeights"]
    ) -> "V41DSparkPackedWeights":
        """Stack matching per-rank payloads on a new leading EP axis."""
        ranks = tuple(values)
        if not ranks:
            raise ValueError("at least one EP rank is required")
        names = tuple(ranks[0].tensors)
        if any(tuple(value.tensors) != names for value in ranks[1:]):
            raise ValueError("DSpark rank payloads must expose matching ABI names")
        stacked = {}
        for name in names:
            tensors = tuple(value.tensors[name] for value in ranks)
            if any(tensor.shape != tensors[0].shape or tensor.dtype != tensors[0].dtype
                   for tensor in tensors[1:]):
                raise ValueError(f"DSpark rank payload shape/dtype mismatch for {name}")
            stacked[name] = torch.stack(tensors, dim=0).contiguous()
        return cls(stacked)


_PACKED_LAYER_NAMES = (
    "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_weight",
    "wq_a", "wq_a_scale", "q_norm_weight", "wq_b", "wq_b_scale", "wkv",
    "wkv_scale", "kv_norm_weight", "attn_sink", "wo_a", "wo_a_scale", "wo_b",
    "wo_b_scale", "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "ffn_norm_weight",
    "gate_weight", "correction_bias", "routed_w1", "routed_w1_scale", "routed_w2",
    "routed_w2_scale", "routed_w3", "routed_w3_scale", "shared_w1", "shared_w1_scale",
    "shared_w2", "shared_w2_scale", "shared_w3", "shared_w3_scale",
)


class V41DSparkWeightContract:
    """Validate and lazily read deferred DSpark tensors from a V41 loader.

    ``loader`` is deliberately duck-typed to keep this contract usable by the
    existing V41WeightLoader and by host tests.  It must expose ``store`` and
    ``text``.  No device allocation or full-bank materialization occurs here.
    """

    def __init__(self, loader):
        if not hasattr(loader, "store") or not hasattr(loader, "text"):
            raise TypeError("V41 DSpark contract requires a V41WeightLoader-like object")
        self.loader = loader
        self.n_routed_experts = loader.text.get("n_routed_experts")
        if type(self.n_routed_experts) is not int or self.n_routed_experts <= 0:
            raise ValueError("text.n_routed_experts must be a positive integer")

    @property
    def required_names(self) -> tuple[str, ...]:
        return dspark_v41_required_weight_names(self.n_routed_experts)

    def validate_index(self) -> None:
        """Fail before payload I/O when a speculative checkpoint is incomplete."""
        self.loader.store.require(self.required_names)

    def load_heads(self) -> V41DSparkHeadWeights:
        """Load only the replicated head tensors and normalize main-proj layout.

        The checkpoint stores ``main_proj`` as ``[D, 3D]`` while the lib ABI
        consumes ``[3D, D]``.  Other head tensors already use their ABI logical
        orientation; quantized layer transforms remain the packer's concern.
        """
        values = self.loader.store.load_many(tuple(DSPARK_HEAD_SOURCE_TO_ABI))
        main_proj = values["mtp.0.main_proj.weight"]
        main_norm = values["mtp.0.main_norm.weight"]
        markov_w1 = values["mtp.2.markov_head.markov_w1.weight"]
        markov_w2 = values["mtp.2.markov_head.markov_w2.weight"]
        confidence = values["mtp.2.confidence_head.proj.weight"]
        if main_proj.ndim != 2 or main_norm.ndim != 1:
            raise ValueError("DSpark main projection must be rank-2 with a rank-1 norm")
        hidden = int(main_norm.shape[0])
        configured_hidden = self.loader.text.get("hidden_size")
        if configured_hidden is not None and hidden != configured_hidden:
            raise ValueError(
                f"DSpark main_norm length must match hidden_size={configured_hidden}, got {hidden}"
            )
        if tuple(main_proj.shape) != (hidden, 3 * hidden):
            raise ValueError(
                f"DSpark main_proj must be [D, 3*D] with D={hidden}, got {tuple(main_proj.shape)}"
            )
        if markov_w1.ndim != 2 or tuple(markov_w1.shape) != tuple(markov_w2.shape):
            raise ValueError("DSpark Markov tables must have matching rank-2 shapes")
        configured_vocab = self.loader.text.get("vocab_size")
        if configured_vocab is not None and int(markov_w1.shape[0]) != configured_vocab:
            raise ValueError(
                f"DSpark Markov tables must have vocab_size={configured_vocab}, "
                f"got {int(markov_w1.shape[0])} rows"
            )
        markov_rank = int(markov_w1.shape[1])
        if tuple(confidence.shape) != (1, hidden + markov_rank):
            raise ValueError(
                f"DSpark confidence weight must be [1, {hidden + markov_rank}], "
                f"got {tuple(confidence.shape)}"
            )
        result = {
            DSPARK_HEAD_SOURCE_TO_ABI[name]: value
            for name, value in values.items()
        }
        result["main_proj_weight"] = main_proj.transpose(0, 1).contiguous()
        return V41DSparkHeadWeights(result)

    def load_layer(self, layer: int) -> Mapping[str, torch.Tensor]:
        """Load one raw draft layer after validating its source names.

        The returned tensors retain checkpoint orientation.  This is the
        bounded input to the future V4.1 FP8/FP4 packer and is intentionally
        not uploaded to an NPU by this module.
        """
        names = dspark_v41_layer_weight_names(layer, self.n_routed_experts)
        self.loader.store.require(names)
        return self.loader.store.load_many(names)

    def _layer_source_shapes(self) -> dict[str, tuple[int, ...]]:
        """Derive the checkpoint shapes for one deferred draft layer."""
        text = self.loader.text
        required = (
            "hidden_size", "num_attention_heads", "head_dim", "q_lora_rank",
            "o_lora_rank", "o_groups", "moe_intermediate_size", "hc_mult",
        )
        missing = [name for name in required if type(text.get(name)) is not int]
        if missing:
            raise ValueError(f"text_config is missing DSpark dimensions: {', '.join(missing)}")
        hidden = text["hidden_size"]
        heads = text["num_attention_heads"]
        head_dim = text["head_dim"]
        q_lora = text["q_lora_rank"]
        o_lora = text["o_lora_rank"]
        o_groups = text["o_groups"]
        moe_inter = text["moe_intermediate_size"]
        hc_mult = text["hc_mult"]
        q_width = heads * head_dim
        o_width = q_width // o_groups
        if q_width % o_groups:
            raise ValueError("num_attention_heads * head_dim must divide evenly by o_groups")
        mix_hc = (2 + hc_mult) * hc_mult
        return {
            "attn_norm.weight": (hidden,),
            "ffn_norm.weight": (hidden,),
            "attn.wq_a.weight": (q_lora, hidden),
            "attn.wq_a.scale": (_ceil_div(q_lora, 32), _ceil_div(hidden, 32)),
            "attn.wq_b.weight": (q_width, q_lora),
            "attn.wq_b.scale": (_ceil_div(q_width, 32), _ceil_div(q_lora, 32)),
            "attn.wkv.weight": (head_dim, hidden),
            "attn.wkv.scale": (_ceil_div(head_dim, 32), _ceil_div(hidden, 32)),
            "attn.q_norm.weight": (q_lora,),
            "attn.kv_norm.weight": (head_dim,),
            "attn.attn_sink": (heads,),
            "attn.wo_a.weight": (o_groups * o_lora, o_width),
            "attn.wo_a.scale": (_ceil_div(o_groups * o_lora, 32), _ceil_div(o_width, 32)),
            "attn.wo_b.weight": (hidden, o_groups * o_lora),
            "attn.wo_b.scale": (_ceil_div(hidden, 32), _ceil_div(o_groups * o_lora, 32)),
            "hc_attn_fn": (mix_hc, hc_mult * hidden),
            "hc_attn_scale": (3,),
            "hc_attn_base": (mix_hc,),
            "hc_ffn_fn": (mix_hc, hc_mult * hidden),
            "hc_ffn_scale": (3,),
            "hc_ffn_base": (mix_hc,),
            "ffn.gate.weight": (self.n_routed_experts, hidden),
            "ffn.gate.bias": (self.n_routed_experts,),
            "ffn.shared_experts.w1.weight": (moe_inter, hidden),
            "ffn.shared_experts.w1.scale": (_ceil_div(moe_inter, 32), _ceil_div(hidden, 32)),
            "ffn.shared_experts.w2.weight": (hidden, moe_inter),
            "ffn.shared_experts.w2.scale": (_ceil_div(hidden, 32), _ceil_div(moe_inter, 32)),
            "ffn.shared_experts.w3.weight": (moe_inter, hidden),
            "ffn.shared_experts.w3.scale": (_ceil_div(moe_inter, 32), _ceil_div(hidden, 32)),
        }

    def load_layer_abi(self, layer: int) -> V41DSparkLayerWeights:
        """Load and validate one layer under the drafter ABI names.

        Returned projections and scales retain checkpoint orientation.  This is
        deliberate: TP/EP slicing and FP8/FP4 packing need the runtime topology
        and belong to the future device packer, while this method guarantees the
        source contract before any such step is attempted.
        """
        raw = self.load_layer(layer)
        expected = self._layer_source_shapes()
        tensors: dict[str, torch.Tensor] = {}
        prefix = f"mtp.{layer}."
        for suffix, abi_name in DSPARK_LAYER_SOURCE_TO_ABI.items():
            source_name = prefix + suffix
            value = raw[source_name]
            shape = expected[suffix]
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"DSpark {source_name} must be {shape}, got {tuple(value.shape)}"
                )
            tensors[abi_name] = value

        expert_shapes = {
            "w1.weight": (self.loader.text["moe_intermediate_size"], self.loader.text["hidden_size"] // 2),
            "w1.scale": (
                self.loader.text["moe_intermediate_size"],
                self.loader.text["hidden_size"] // 32,
            ),
            "w2.weight": (self.loader.text["hidden_size"], self.loader.text["moe_intermediate_size"] // 2),
            "w2.scale": (
                self.loader.text["hidden_size"],
                self.loader.text["moe_intermediate_size"] // 32,
            ),
            "w3.weight": (self.loader.text["moe_intermediate_size"], self.loader.text["hidden_size"] // 2),
            "w3.scale": (
                self.loader.text["moe_intermediate_size"],
                self.loader.text["hidden_size"] // 32,
            ),
        }
        routed: dict[str, dict[int, torch.Tensor]] = {
            abi_name: {} for abi_name in DSPARK_EXPERT_SOURCE_TO_ABI.values()
        }
        for expert in range(self.n_routed_experts):
            expert_prefix = f"{prefix}ffn.experts.{expert}."
            for suffix, abi_name in DSPARK_EXPERT_SOURCE_TO_ABI.items():
                value = raw[expert_prefix + suffix]
                shape = expert_shapes[suffix]
                if tuple(value.shape) != shape:
                    raise ValueError(
                        f"DSpark {expert_prefix + suffix} must be {shape}, "
                        f"got {tuple(value.shape)}"
                    )
                routed[abi_name][expert] = value
        return V41DSparkLayerWeights(tensors=tensors, routed_experts=routed)

    def _load_layer_for_rank(self, layer: int) -> dict[str, torch.Tensor]:
        """Read the non-expert tensors for one draft layer.

        ``load_layer`` is kept as the shape-contract/debugging API and reads
        every expert. Runtime staging keeps the public layer payload separate
        from expert reads; :meth:`_load_expert_for_rank` fetches one local
        expert at a time.
        """
        if type(layer) is not int or layer not in DSPARK_DRAFT_LAYERS:
            raise ValueError("DSpark draft layer must be one of 0, 1 or 2")
        ep_size = getattr(self.loader, "ep_size", 1)
        ep_rank = getattr(self.loader, "ep_rank", 0)
        if type(ep_size) is not int or ep_size <= 0 or type(ep_rank) is not int:
            raise ValueError("DSpark packer requires a rank-aware V41WeightLoader")
        if ep_rank < 0 or ep_rank >= ep_size or self.n_routed_experts % ep_size:
            raise ValueError("DSpark routed experts must divide the selected EP world")
        if getattr(self.loader, "tp_size", 1) != 1:
            raise ValueError("V4.1 DSpark drafter packing currently requires TP1")
        prefix = f"mtp.{layer}."
        names = [prefix + suffix for suffix in _LAYER_SUFFIXES]
        self.loader.store.require(names)
        return self.loader.store.load_many(names)

    def _load_expert_for_rank(self, layer: int, expert: int) -> dict[str, torch.Tensor]:
        """Read one EP-local expert's three projections and scales."""
        ep_size = int(getattr(self.loader, "ep_size", 1))
        ep_rank = int(getattr(self.loader, "ep_rank", 0))
        local_count = self.n_routed_experts // ep_size
        first = ep_rank * local_count
        last = first + local_count
        if type(layer) is not int or layer not in DSPARK_DRAFT_LAYERS:
            raise ValueError("DSpark draft layer must be one of 0, 1 or 2")
        if type(expert) is not int or not first <= expert < last:
            raise ValueError("DSpark expert is outside the selected EP rank")
        prefix = f"mtp.{layer}.ffn.experts.{expert}."
        names = [prefix + suffix for suffix in DSPARK_EXPERT_SOURCE_TO_ABI]
        self.loader.store.require(names)
        return self.loader.store.load_many(names)

    def load_layer_packed(self, layer: int) -> V41DSparkLayerWeights:
        """Pack one EP-local draft layer for ``l3_dspark_drafter``.

        The returned tensors have no leading EP or draft-layer axis.  The
        caller is expected to copy them into a resident ``[EP, 3, ...]`` bank.
        Native FP8 projections retain their E4M3 payload and receive the
        physical E8M0 scale order required by the lib.  Routed MXFP4 payloads
        are tile-reordered without float expansion, and their scales are
        packed independently per local expert before concatenation.
        """
        layer_raw = self._load_layer_for_rank(layer)
        expected = self._layer_source_shapes()
        prefix = f"mtp.{layer}."

        def source(
            suffix: str,
            *,
            values=layer_raw,
            shapes=expected,
            name_prefix=prefix,
        ) -> torch.Tensor:
            value = values[name_prefix + suffix]
            shape = shapes[suffix]
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"DSpark {name_prefix + suffix} must be {shape}, got {tuple(value.shape)}"
                )
            return value

        result: dict[str, torch.Tensor] = {}
        for suffix, abi_name in DSPARK_LAYER_SOURCE_TO_ABI.items():
            if suffix.endswith(".scale"):
                continue
            value = source(suffix)
            if suffix in (
                "attn.wq_a.weight", "attn.wq_b.weight", "attn.wkv.weight",
                "attn.wo_b.weight", "ffn.shared_experts.w1.weight",
                "ffn.shared_experts.w2.weight", "ffn.shared_experts.w3.weight",
            ):
                scale_suffix = suffix.removesuffix(".weight") + ".scale"
                weight, scale = fp8_input_major(value, source(scale_suffix))
                result[abi_name] = weight
                result[abi_name + "_scale"] = scale
            elif suffix == "attn.wo_a.weight":
                weight, scale = pack_grouped_wo_a(
                    value, source("attn.wo_a.scale"),
                    int(self.loader.text["o_groups"]),
                )
                result[abi_name] = weight
                result[abi_name + "_scale"] = scale
            elif suffix == "ffn.gate.weight":
                result[abi_name] = value.float().contiguous()
            else:
                result[abi_name] = value.contiguous()

        local_count = self.n_routed_experts // int(self.loader.ep_size)
        routed_values: dict[str, list[torch.Tensor]] = {
            name: [] for name in DSPARK_EXPERT_SOURCE_TO_ABI.values()
        }
        for expert in range(
            int(self.loader.ep_rank) * local_count,
            (int(self.loader.ep_rank) + 1) * local_count,
        ):
            expert_raw = self._load_expert_for_rank(layer, expert)
            for suffix, abi_name in DSPARK_EXPERT_SOURCE_TO_ABI.items():
                value = expert_raw[f"mtp.{layer}.ffn.experts.{expert}.{suffix}"]
                if suffix.endswith(".weight"):
                    packed = pack_fp4_tiles(value)
                    routed_values[abi_name].append(packed)
                else:
                    # The scale source is [N, K/32].  The lib flattens local
                    # experts before the K-group axis, so pack each expert
                    # independently and concatenate on the group axis.
                    scale = value.view(torch.uint8).T.contiguous()
                    routed_values[abi_name].append(
                        _pack_expert_scale(scale, value.dtype)
                    )
            del expert_raw
        del layer_raw
        for name, values in routed_values.items():
            result[name] = torch.cat(values, dim=0).contiguous()

        return V41DSparkLayerWeights(tensors=result, routed_experts={})

    def load_packed_weights(self) -> V41DSparkPackedWeights:
        """Pack all three draft layers for this EP rank, bounded by layer.

        This helper intentionally returns one rank only.  A serving resource
        can call it once per EP rank and fill a shared bank incrementally;
        keeping rank ownership here prevents every worker from reading all
        routed experts.  The small heads are included for convenience.
        """
        heads = self.load_heads().tensors
        layers = [self.load_layer_packed(layer).tensors for layer in DSPARK_DRAFT_LAYERS]
        packed: dict[str, torch.Tensor] = {}
        for name in _PACKED_LAYER_NAMES:
            values = [layer[name] for layer in layers]
            packed[name] = torch.stack(values, dim=0).contiguous()
        packed.update({name: value.contiguous() for name, value in heads.items()})
        packed["mxfp4_pair_lut"] = mxfp4_pair_lut()
        return V41DSparkPackedWeights(packed)


def _pack_expert_scale(scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Pack one expert's logical [K/32, N] scale matrix to MX_B_NN."""
    groups, width = scale.shape
    if groups % 2 or width % 16:
        raise ValueError("MXFP4 expert scales require K divisible by 64 and N by 16")
    from .weight_spec import pack_mx_scale

    return pack_mx_scale(scale).view(dtype).contiguous()
