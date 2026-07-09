# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ContextManager, Protocol

import torch


class _SafeTensorReader(Protocol):
    """Minimal safetensors reader protocol used by the lazy weight store."""

    def get_tensor(self, name: str) -> torch.Tensor:
        """Return one tensor by name."""
        raise NotImplementedError


class _SafeOpenFn(Protocol):
    """Callable shape for injectable safetensors openers."""

    def __call__(self, path: Path, device: str) -> ContextManager[_SafeTensorReader]:
        """Open one safetensors shard."""
        raise NotImplementedError


_GLOBAL_WEIGHT_NAMES = (
    "embed.weight",
    "norm.weight",
    "head.weight",
    "hc_head_fn",
    "hc_head_scale",
    "hc_head_base",
)
_LM_HEAD_VOCAB_CHUNK = 512
_LAYER_COMMON_SUFFIXES = (
    "attn.attn_sink",
    "attn.kv_norm.weight",
    "attn.q_norm.weight",
    "attn.wkv.weight",
    "attn.wo_a.weight",
    "attn.wo_b.weight",
    "attn.wo_b.scale",
    "attn.wq_a.weight",
    "attn.wq_b.weight",
    "attn.wq_b.scale",
    "attn_norm.weight",
    "ffn.gate.weight",
    "ffn.shared_experts.w1.weight",
    "ffn.shared_experts.w1.scale",
    "ffn.shared_experts.w2.weight",
    "ffn.shared_experts.w2.scale",
    "ffn.shared_experts.w3.weight",
    "ffn.shared_experts.w3.scale",
    "ffn_norm.weight",
    "hc_attn_base",
    "hc_attn_fn",
    "hc_attn_scale",
    "hc_ffn_base",
    "hc_ffn_fn",
    "hc_ffn_scale",
)
_LAYER_COMPRESSOR_SUFFIXES = (
    "attn.compressor.ape",
    "attn.compressor.norm.weight",
    "attn.compressor.wgate.weight",
    "attn.compressor.wkv.weight",
)
_LAYER_INDEXER_SUFFIXES = (
    "attn.indexer.compressor.ape",
    "attn.indexer.compressor.norm.weight",
    "attn.indexer.compressor.wgate.weight",
    "attn.indexer.compressor.wkv.weight",
    "attn.indexer.weights_proj.weight",
    "attn.indexer.wq_b.weight",
    "attn.indexer.wq_b.scale",
)
_EXPERT_SUFFIXES = ("w1.weight", "w1.scale", "w2.weight", "w2.scale", "w3.weight", "w3.scale")
_DEEPSEEK_V4_O_GROUPS = 8
_DEEPSEEK_V4_HADAMARD_IDX_DIM = 128
_DEEPSEEK_V4_HCA_COMPRESS_RATIO = 128
_DEEPSEEK_V4_CSA_COMPRESS_RATIO = 4
_DEEPSEEK_V4_HCA_MAIN_OUT_DIM = 512
_DEEPSEEK_V4_CSA_MAIN_OUT_DIM = 1024
_DEEPSEEK_V4_CSA_INNER_OUT_DIM = 256
_DEEPSEEK_V4_HIDDEN_SIZE = 4096
_DEEPSEEK_V4_Q_LORA = 1024
_DEEPSEEK_V4_HEAD_DIM = 512
_DEEPSEEK_V4_ATTENTION_OUT = 64 * 512
_DEEPSEEK_V4_N_ROUTED_EXPERTS = 256
_DEEPSEEK_V4_TOPK = 6
_DEEPSEEK_V4_VOCAB_SIZE = 129280


def _default_safe_open(path: Path, device: str) -> ContextManager[_SafeTensorReader]:
    """Open a safetensors shard without loading unrelated tensors."""
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("safetensors is required to read DeepSeekV4 W8A8 weights.") from exc

    return safe_open(str(path), framework="pt", device=device)


def deepseek_v4_global_weight_names() -> tuple[str, ...]:
    """Return global DeepSeekV4 tensor names needed outside the layer stack."""
    return _GLOBAL_WEIGHT_NAMES


@dataclass(frozen=True)
class DeepSeekV4LmHeadLayout:
    """8-way tensor-parallel LM-head shard layout."""

    ranks: int
    vocab_size: int
    hidden_size: int
    vocab_per_rank: int
    padded_vocab_per_rank: int


@dataclass(frozen=True)
class DeepSeekV4GlobalWeights:
    """Global DeepSeekV4 weights packed for serving kernels."""

    embed_weight: torch.Tensor
    final_norm_weight: torch.Tensor
    lm_head_weight: torch.Tensor
    lm_head_layout: DeepSeekV4LmHeadLayout
    hc_head_fn: torch.Tensor
    hc_head_scale: torch.Tensor
    hc_head_base: torch.Tensor


@dataclass(frozen=True)
class DeepSeekV4PackedLayerWeights:
    """One DeepSeekV4 layer's tensors packed in pypto-lib host argument names."""

    layer_id: int
    tensors: Mapping[str, torch.Tensor]

    def args(self, names: Sequence[str]) -> tuple[torch.Tensor, ...]:
        """Return tensors in a kernel host order."""
        missing = [name for name in names if name not in self.tensors]
        if missing:
            raise KeyError(f"Packed DeepSeekV4 layer is missing tensors: {', '.join(missing)}")
        return tuple(self.tensors[name] for name in names)


# Layer-stacking groups for the packed all-layer ``l3_decode_fwd`` kernel. These
# mirror the name groups in pypto-lib decode_fwd.py, but only cover *loaded*
# weights -- the per-layer work-cache/state tensors (kv_cache, cmp_kv,
# idx_kv_cache, *_compress_state) are owned by the runner work cache and are not
# emitted by the weight loader.
DEEPSEEK_V4_CSA_STACKED_WEIGHT_NAMES = (
    "csa_cmp_wkv",
    "csa_cmp_wgate",
    "csa_cmp_ape",
    "csa_cmp_norm_w",
    "csa_idx_wq_b",
    "csa_idx_wq_b_scale",
    "csa_weights_proj",
    "csa_hadamard_idx",
    "csa_inner_wkv",
    "csa_inner_wgate",
    "csa_inner_ape",
    "csa_inner_norm_w",
)
DEEPSEEK_V4_HCA_STACKED_WEIGHT_NAMES = (
    "hca_cmp_wkv",
    "hca_cmp_wgate",
    "hca_cmp_ape",
    "hca_cmp_norm_w",
)
_DEEPSEEK_V4_CSA_COMPRESS_RATIO_VALUE = 4
_DEEPSEEK_V4_HCA_COMPRESS_RATIO_VALUE = 128


@dataclass(frozen=True)
class DeepSeekV4StackedLayerWeights:
    """All hidden-layer weights stacked on the layer axis for ``l3_decode_fwd``.

    Each tensor fuses its layer axis into dim 1: ``[ranks, layer_count*d1, ...]``.
    FWD weights stack across all 43 hidden layers; CSA-group weights stack across
    the 21 compress_ratio==4 layers in order; HCA-group weights stack across the
    20 compress_ratio==128 layers in order.
    """

    tensors: Mapping[str, torch.Tensor]

    def args(self, names: Sequence[str]) -> tuple[torch.Tensor, ...]:
        """Return stacked tensors in a kernel host order."""
        missing = [name for name in names if name not in self.tensors]
        if missing:
            raise KeyError(f"Stacked DeepSeekV4 weights are missing tensors: {', '.join(missing)}")
        return tuple(self.tensors[name] for name in names)


def deepseek_v4_lm_head_layout(
    *,
    vocab_size: int,
    hidden_size: int,
    ranks: int,
    vocab_chunk: int = _LM_HEAD_VOCAB_CHUNK,
) -> DeepSeekV4LmHeadLayout:
    """Return the LM-head shard shape expected by ``lm_head.py``."""
    if ranks <= 0:
        raise ValueError("ranks must be positive")
    if vocab_chunk <= 0:
        raise ValueError("vocab_chunk must be positive")
    if vocab_size % ranks != 0:
        raise ValueError(f"vocab_size={vocab_size} must divide evenly across ranks={ranks}")
    vocab_per_rank = vocab_size // ranks
    padded_vocab_per_rank = ((vocab_per_rank + vocab_chunk - 1) // vocab_chunk) * vocab_chunk
    return DeepSeekV4LmHeadLayout(
        ranks=ranks,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        vocab_per_rank=vocab_per_rank,
        padded_vocab_per_rank=padded_vocab_per_rank,
    )


def pack_deepseek_v4_lm_head_weight(
    weight: torch.Tensor,
    *,
    ranks: int,
    vocab_chunk: int = _LM_HEAD_VOCAB_CHUNK,
) -> tuple[torch.Tensor, DeepSeekV4LmHeadLayout]:
    """Pack flat ``head.weight`` into contiguous TP vocab shards."""
    if weight.ndim != 2:
        raise ValueError(f"lm_head weight must be rank-2, got shape={tuple(weight.shape)}")
    vocab_size, hidden_size = (int(dim) for dim in weight.shape)
    layout = deepseek_v4_lm_head_layout(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        ranks=ranks,
        vocab_chunk=vocab_chunk,
    )
    packed = torch.zeros(
        (layout.ranks, layout.padded_vocab_per_rank, layout.hidden_size),
        dtype=weight.dtype,
        device=weight.device,
    )
    for rank in range(layout.ranks):
        start = rank * layout.vocab_per_rank
        end = start + layout.vocab_per_rank
        packed[rank, : layout.vocab_per_rank].copy_(weight[start:end])
    return packed.contiguous(), layout


def _attention_suffixes_for_compress_ratio(compress_ratio: int) -> tuple[str, ...]:
    """Return attention parameter suffixes required by one layer attention mode."""
    if compress_ratio == 0:
        return ()
    if compress_ratio == 128:
        return _LAYER_COMPRESSOR_SUFFIXES
    if compress_ratio == 4:
        return (*_LAYER_COMPRESSOR_SUFFIXES, *_LAYER_INDEXER_SUFFIXES)
    raise ValueError(f"unsupported DeepSeekV4 attention compress ratio: {compress_ratio}")


def deepseek_v4_layer_core_weight_names(
    layer_id: int,
    *,
    compress_ratio: int = 0,
    include_tid2eid: bool = False,
    include_gate_bias: bool = False,
) -> tuple[str, ...]:
    """Return non-routed-expert tensor names for one DeepSeekV4 layer."""
    prefix = f"layers.{int(layer_id)}"
    suffixes = [*_LAYER_COMMON_SUFFIXES, *_attention_suffixes_for_compress_ratio(compress_ratio)]
    if include_tid2eid:
        suffixes.append("ffn.gate.tid2eid")
    if include_gate_bias:
        suffixes.append("ffn.gate.bias")
    return tuple(f"{prefix}.{suffix}" for suffix in suffixes)


def deepseek_v4_routed_expert_weight_names(layer_id: int, expert_ids: Iterable[int]) -> tuple[str, ...]:
    """Return routed expert tensor names for one DeepSeekV4 layer."""
    names: list[str] = []
    for expert_id in expert_ids:
        prefix = f"layers.{int(layer_id)}.ffn.experts.{int(expert_id)}"
        names.extend(f"{prefix}.{suffix}" for suffix in _EXPERT_SUFFIXES)
    return tuple(names)


def deepseek_v4_local_expert_ids(*, rank: int, ranks: int, n_routed_experts: int) -> tuple[int, ...]:
    """Return the contiguous routed-expert ids owned by one EP rank."""
    if ranks <= 0:
        raise ValueError("ranks must be positive")
    if not 0 <= rank < ranks:
        raise ValueError(f"rank must be in [0, {ranks - 1}], got {rank}")
    if n_routed_experts <= 0:
        raise ValueError("n_routed_experts must be positive")
    if n_routed_experts % ranks != 0:
        raise ValueError(f"n_routed_experts={n_routed_experts} must divide evenly across ranks={ranks}")
    local_count = n_routed_experts // ranks
    start = rank * local_count
    return tuple(range(start, start + local_count))


def deepseek_v4_hadamard_idx(dim: int = _DEEPSEEK_V4_HADAMARD_IDX_DIM) -> torch.Tensor:
    """Return the normalized Hadamard matrix used by the CSA indexer."""
    if dim <= 0 or dim & (dim - 1) != 0:
        raise ValueError("Hadamard dimension must be a positive power of two")
    h = torch.ones((1, 1), dtype=torch.bfloat16)
    while h.shape[0] < dim:
        h = torch.cat(
            [torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)],
            dim=0,
        )
    return (h * (dim**-0.5)).contiguous()


def deepseek_v4_layer_weight_names(
    layer_id: int,
    *,
    n_routed_experts: int,
    compress_ratio: int = 0,
    include_tid2eid: bool = False,
    include_gate_bias: bool = False,
    expert_ids: Iterable[int] | None = None,
) -> tuple[str, ...]:
    """Return all tensor names needed to execute one DeepSeekV4 layer."""
    if n_routed_experts <= 0:
        raise ValueError("n_routed_experts must be positive")
    expert_ids = range(n_routed_experts) if expert_ids is None else tuple(expert_ids)
    return (
        *deepseek_v4_layer_core_weight_names(
            layer_id,
            compress_ratio=compress_ratio,
            include_tid2eid=include_tid2eid,
            include_gate_bias=include_gate_bias,
        ),
        *deepseek_v4_routed_expert_weight_names(layer_id, expert_ids),
    )


def deepseek_v4_startup_weight_names(
    num_hidden_layers: int,
    *,
    n_routed_experts: int,
    compress_ratios: Sequence[int] | None = None,
    num_hash_layers: int = 3,
) -> tuple[str, ...]:
    """Return tensor names used for metadata-only checkpoint contract validation.

    Startup checks every layer's core tensors plus the first and last routed
    expert in each layer. Full expert materialization remains an explicit
    per-layer load so serving startup does not read shard payloads.
    """
    if num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be positive")
    if n_routed_experts <= 0:
        raise ValueError("n_routed_experts must be positive")
    if compress_ratios is None:
        compress_ratios = (0,) * num_hidden_layers
    if len(compress_ratios) < num_hidden_layers:
        raise ValueError("compress_ratios must include at least one entry per hidden layer")

    edge_experts = tuple(dict.fromkeys((0, n_routed_experts - 1)))
    names = list(_GLOBAL_WEIGHT_NAMES)
    for layer_id in range(num_hidden_layers):
        names.extend(
            deepseek_v4_layer_core_weight_names(
                layer_id,
                compress_ratio=int(compress_ratios[layer_id]),
                include_tid2eid=layer_id < num_hash_layers,
                include_gate_bias=layer_id >= num_hash_layers,
            )
        )
        names.extend(deepseek_v4_routed_expert_weight_names(layer_id, edge_experts))
    return tuple(dict.fromkeys(names))


class DeepSeekV4WeightStore:
    """Lazy name-based safetensors access for DeepSeekV4 W8A8 checkpoints."""

    def __init__(
        self,
        *,
        model_dir: str | Path,
        weight_map: Mapping[str, str],
        device: str = "cpu",
        safe_open_fn: _SafeOpenFn | None = None,
    ) -> None:
        """Create a store from the Hugging Face safetensors index."""
        self.model_dir = Path(model_dir)
        self.weight_map = dict(weight_map)
        self.device = device
        self._safe_open_fn = _default_safe_open if safe_open_fn is None else safe_open_fn

    def __contains__(self, name: object) -> bool:
        """Return whether the checkpoint index exposes ``name``."""
        return isinstance(name, str) and name in self.weight_map

    def filename_for(self, name: str) -> str:
        """Return the safetensors shard filename for ``name``."""
        try:
            return self.weight_map[name]
        except KeyError as exc:
            raise KeyError(f"Missing DeepSeekV4 weight tensor in index: {name}") from exc

    def path_for(self, name: str) -> Path:
        """Return the shard path containing ``name``."""
        return self.model_dir / self.filename_for(name)

    def require(self, names: Iterable[str]) -> None:
        """Validate that all tensor names are present in the checkpoint index."""
        missing = [name for name in names if name not in self.weight_map]
        if missing:
            preview = ", ".join(missing[:8])
            suffix = "" if len(missing) <= 8 else f", ... ({len(missing)} total)"
            raise KeyError(f"DeepSeekV4 W8A8 checkpoint is missing required tensors: {preview}{suffix}")

    def validate_startup_contract(
        self,
        *,
        num_hidden_layers: int,
        n_routed_experts: int,
        compress_ratios: Sequence[int] | None = None,
        num_hash_layers: int = 3,
    ) -> None:
        """Validate the startup-visible checkpoint contract without opening shards."""
        self.require(
            deepseek_v4_startup_weight_names(
                num_hidden_layers,
                n_routed_experts=n_routed_experts,
                compress_ratios=compress_ratios,
                num_hash_layers=num_hash_layers,
            )
        )

    def load_tensor(self, name: str) -> torch.Tensor:
        """Load one tensor by name, leaving all unrelated shard tensors untouched."""
        return self.load_many([name])[name]

    def load_many(self, names: Sequence[str]) -> dict[str, torch.Tensor]:
        """Load a set of named tensors grouped by shard file."""
        unique_names = tuple(dict.fromkeys(names))
        self.require(unique_names)

        groups: dict[str, list[str]] = {}
        for name in unique_names:
            groups.setdefault(self.filename_for(name), []).append(name)

        loaded: dict[str, torch.Tensor] = {}
        for filename, shard_names in groups.items():
            path = self.model_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"Missing safetensors shard for DeepSeekV4 weight load: {path}")
            with self._safe_open_fn(path, self.device) as reader:
                for name in shard_names:
                    loaded[name] = reader.get_tensor(name)

        return {name: loaded[name] for name in unique_names}

    def load_global_weights(self) -> dict[str, torch.Tensor]:
        """Load embedding, final norm, and LM head tensors."""
        return self.load_many(deepseek_v4_global_weight_names())

    def load_packed_global_weights(self, *, ranks: int) -> DeepSeekV4GlobalWeights:
        """Load and pack global tensors for the DeepSeekV4 serving kernels."""
        weights = self.load_global_weights()
        packed_lm_head, layout = pack_deepseek_v4_lm_head_weight(weights["head.weight"], ranks=ranks)
        if weights["embed.weight"].ndim != 2:
            raise ValueError(f"embed.weight must be rank-2, got shape={tuple(weights['embed.weight'].shape)}")
        if weights["norm.weight"].ndim != 1:
            raise ValueError(f"norm.weight must be rank-1, got shape={tuple(weights['norm.weight'].shape)}")
        if tuple(weights["embed.weight"].shape) != (layout.vocab_size, layout.hidden_size):
            raise ValueError(
                "embed.weight shape must match head.weight shape, "
                f"got embed={tuple(weights['embed.weight'].shape)}, head={tuple(weights['head.weight'].shape)}"
            )
        if int(weights["norm.weight"].shape[0]) != layout.hidden_size:
            raise ValueError(
                f"norm.weight hidden size must be {layout.hidden_size}, "
                f"got {int(weights['norm.weight'].shape[0])}"
            )
        if tuple(weights["hc_head_fn"].shape) != (4, layout.hidden_size * 4):
            raise ValueError(f"hc_head_fn has unsupported shape {tuple(weights['hc_head_fn'].shape)}")
        if tuple(weights["hc_head_scale"].shape) != (1,):
            raise ValueError(f"hc_head_scale has unsupported shape {tuple(weights['hc_head_scale'].shape)}")
        if tuple(weights["hc_head_base"].shape) != (4,):
            raise ValueError(f"hc_head_base has unsupported shape {tuple(weights['hc_head_base'].shape)}")
        return DeepSeekV4GlobalWeights(
            embed_weight=weights["embed.weight"],
            final_norm_weight=weights["norm.weight"],
            lm_head_weight=packed_lm_head,
            lm_head_layout=layout,
            hc_head_fn=weights["hc_head_fn"].to(torch.float32).contiguous().cpu(),
            hc_head_scale=weights["hc_head_scale"].to(torch.float32).contiguous().cpu(),
            hc_head_base=weights["hc_head_base"].to(torch.float32).contiguous().cpu(),
        )

    def load_layer_weights(
        self,
        layer_id: int,
        *,
        n_routed_experts: int,
        compress_ratio: int = 0,
        include_tid2eid: bool = False,
        include_gate_bias: bool = False,
        expert_ids: Iterable[int] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Load all tensors needed for one DeepSeekV4 layer."""
        return self.load_many(
            deepseek_v4_layer_weight_names(
                layer_id,
                n_routed_experts=n_routed_experts,
                compress_ratio=compress_ratio,
                include_tid2eid=include_tid2eid,
                include_gate_bias=include_gate_bias,
                expert_ids=expert_ids,
            )
        )

    def load_rank_layer_weights(
        self,
        layer_id: int,
        *,
        rank: int,
        ranks: int,
        n_routed_experts: int,
        compress_ratio: int = 0,
        include_tid2eid: bool = False,
        include_gate_bias: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Load common layer tensors plus the routed experts owned by one rank."""
        local_experts = deepseek_v4_local_expert_ids(
            rank=rank,
            ranks=ranks,
            n_routed_experts=n_routed_experts,
        )
        return self.load_layer_weights(
            layer_id,
            n_routed_experts=n_routed_experts,
            compress_ratio=compress_ratio,
            include_tid2eid=include_tid2eid,
            include_gate_bias=include_gate_bias,
            expert_ids=local_experts,
        )

    def load_packed_layer_weights(
        self,
        layer_id: int,
        *,
        ranks: int,
        n_routed_experts: int,
        compress_ratio: int = 0,
        include_tid2eid: bool = False,
        include_gate_bias: bool = False,
    ) -> DeepSeekV4PackedLayerWeights:
        """Load and pack one layer into the tensor names expected by pypto-lib kernels."""
        all_experts = range(n_routed_experts)
        raw = self.load_layer_weights(
            layer_id,
            n_routed_experts=n_routed_experts,
            compress_ratio=compress_ratio,
            include_tid2eid=include_tid2eid,
            include_gate_bias=include_gate_bias,
            expert_ids=all_experts,
        )
        return pack_deepseek_v4_layer_weights(
            layer_id,
            raw,
            ranks=ranks,
            n_routed_experts=n_routed_experts,
            compress_ratio=compress_ratio,
            include_tid2eid=include_tid2eid,
            include_gate_bias=include_gate_bias,
        )

    def load_stacked_layer_weights(
        self,
        *,
        ranks: int,
        n_routed_experts: int,
        compress_ratios: Sequence[int],
        num_hash_layers: int,
    ) -> DeepSeekV4StackedLayerWeights:
        """Load every hidden layer once and stack weights on the layer axis.

        FWD weights are concatenated across all hidden layers in order; CSA-group
        weights across the compress_ratio==4 layers in order; HCA-group weights
        across the compress_ratio==128 layers in order. Each per-layer tensor is
        ``[ranks, d1, ...]`` and stacking concatenates on dim 1.
        """
        num_hidden_layers = len(compress_ratios)
        if num_hidden_layers <= 0:
            raise ValueError("compress_ratios must include at least one entry per hidden layer")
        per_layer: list[DeepSeekV4PackedLayerWeights] = []
        for layer_id in range(num_hidden_layers):
            per_layer.append(
                self.load_packed_layer_weights(
                    layer_id,
                    ranks=ranks,
                    n_routed_experts=n_routed_experts,
                    compress_ratio=int(compress_ratios[layer_id]),
                    include_tid2eid=layer_id < num_hash_layers,
                    include_gate_bias=layer_id >= num_hash_layers,
                )
            )
        return stack_deepseek_v4_layer_weights(per_layer, compress_ratios=compress_ratios)


def stack_deepseek_v4_layer_weights(
    per_layer: Sequence[DeepSeekV4PackedLayerWeights],
    *,
    compress_ratios: Sequence[int],
) -> DeepSeekV4StackedLayerWeights:
    """Concatenate per-layer packed weights into the layer-stacked decode_fwd groups."""
    num_hidden_layers = len(per_layer)
    if num_hidden_layers != len(compress_ratios):
        raise ValueError("per_layer count must match compress_ratios length")
    if num_hidden_layers <= 0:
        raise ValueError("per_layer must include at least one layer")

    csa_layers = [i for i in range(num_hidden_layers) if int(compress_ratios[i]) == _DEEPSEEK_V4_CSA_COMPRESS_RATIO_VALUE]
    hca_layers = [i for i in range(num_hidden_layers) if int(compress_ratios[i]) == _DEEPSEEK_V4_HCA_COMPRESS_RATIO_VALUE]

    csa_grouped = set(DEEPSEEK_V4_CSA_STACKED_WEIGHT_NAMES)
    hca_grouped = set(DEEPSEEK_V4_HCA_STACKED_WEIGHT_NAMES)
    fwd_names = [
        name
        for name in per_layer[0].tensors
        if name not in csa_grouped and name not in hca_grouped
    ]

    def cat(names: Sequence[str], layer_ids: Sequence[int]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for name in names:
            tensors = []
            for layer_id in layer_ids:
                tensor = per_layer[layer_id].tensors[name]
                tensors.append(tensor.contiguous().cpu())
            out[name] = torch.cat(tensors, dim=1).contiguous()
        return out

    stacked: dict[str, torch.Tensor] = {}
    stacked.update(cat(fwd_names, range(num_hidden_layers)))
    stacked.update(cat(DEEPSEEK_V4_CSA_STACKED_WEIGHT_NAMES, csa_layers))
    stacked.update(cat(DEEPSEEK_V4_HCA_STACKED_WEIGHT_NAMES, hca_layers))
    return DeepSeekV4StackedLayerWeights(tensors=stacked)


def pack_deepseek_v4_layer_weights(
    layer_id: int,
    raw: Mapping[str, torch.Tensor],
    *,
    ranks: int,
    n_routed_experts: int,
    compress_ratio: int,
    include_tid2eid: bool,
    include_gate_bias: bool,
) -> DeepSeekV4PackedLayerWeights:
    """Pack raw checkpoint tensors for one layer into rank-stacked kernel tensors."""
    prefix = f"layers.{int(layer_id)}"

    def get(suffix: str) -> torch.Tensor:
        name = f"{prefix}.{suffix}"
        try:
            return raw[name]
        except KeyError as exc:
            raise KeyError(f"missing raw DeepSeekV4 layer tensor: {name}") from exc

    def maybe(suffix: str) -> torch.Tensor | None:
        return raw.get(f"{prefix}.{suffix}")

    def replicated(tensor: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        tensor = tensor.contiguous().cpu()
        return tensor.unsqueeze(0).expand(ranks, *tensor.shape).contiguous()

    def transposed(tensor: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        out = tensor.transpose(0, 1).contiguous().cpu()
        return out.to(dtype=dtype) if dtype is not None else out

    def replicated_transposed(tensor: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        return replicated(transposed(tensor, dtype=dtype))

    tensors: dict[str, torch.Tensor] = {
        "hc_attn_fn": replicated(get("hc_attn_fn"), dtype=torch.float32),
        "hc_attn_scale": replicated(get("hc_attn_scale"), dtype=torch.float32),
        "hc_attn_base": replicated(get("hc_attn_base"), dtype=torch.float32),
        "attn_norm_w": replicated(get("attn_norm.weight"), dtype=torch.bfloat16),
        "wq_a": replicated_transposed(get("attn.wq_a.weight"), dtype=torch.bfloat16),
        "wq_b": replicated_transposed(get("attn.wq_b.weight"), dtype=torch.int8),
        "wq_b_scale": replicated(get("attn.wq_b.scale"), dtype=torch.float32),
        "wkv": replicated_transposed(get("attn.wkv.weight"), dtype=torch.bfloat16),
        "gamma_cq": replicated(get("attn.q_norm.weight"), dtype=torch.bfloat16),
        "gamma_ckv": replicated(get("attn.kv_norm.weight"), dtype=torch.bfloat16),
        "attn_sink": replicated(get("attn.attn_sink"), dtype=torch.float32),
        "wo_a": replicated(_pack_wo_a(get("attn.wo_a.weight")), dtype=torch.bfloat16),
        "wo_b": replicated(get("attn.wo_b.weight"), dtype=torch.int8),
        "wo_b_scale": replicated(get("attn.wo_b.scale"), dtype=torch.float32),
        "hc_ffn_fn": replicated(get("hc_ffn_fn"), dtype=torch.float32),
        "hc_ffn_scale": replicated(get("hc_ffn_scale"), dtype=torch.float32),
        "hc_ffn_base": replicated(get("hc_ffn_base"), dtype=torch.float32),
        "norm_w": replicated(get("ffn_norm.weight"), dtype=torch.bfloat16),
        "gate_w": replicated(get("ffn.gate.weight"), dtype=torch.float32),
        "shared_w1": replicated(get("ffn.shared_experts.w1.weight"), dtype=torch.int8),
        "shared_w1_scale": replicated(get("ffn.shared_experts.w1.scale"), dtype=torch.float32),
        "shared_w3": replicated(get("ffn.shared_experts.w3.weight"), dtype=torch.int8),
        "shared_w3_scale": replicated(get("ffn.shared_experts.w3.scale"), dtype=torch.float32),
        "shared_w2": replicated(get("ffn.shared_experts.w2.weight"), dtype=torch.int8),
        "shared_w2_scale": replicated(get("ffn.shared_experts.w2.scale"), dtype=torch.float32),
    }

    tensors.update(_pack_deepseek_v4_optional_attention(prefix, raw, ranks, compress_ratio=compress_ratio))
    tensors.update(
        _pack_deepseek_v4_router(
            prefix,
            raw,
            ranks=ranks,
            n_routed_experts=n_routed_experts,
            include_tid2eid=include_tid2eid,
            include_gate_bias=include_gate_bias,
        )
    )
    tensors.update(
        _pack_deepseek_v4_routed_experts(
            prefix,
            raw,
            ranks=ranks,
            n_routed_experts=n_routed_experts,
        )
    )
    return DeepSeekV4PackedLayerWeights(layer_id=layer_id, tensors=tensors)


def _pack_wo_a(weight: torch.Tensor) -> torch.Tensor:
    """Pack flattened output-LoRA A projection to ``[o_groups, o_lora, group_in]``."""
    if weight.ndim != 2:
        raise ValueError(f"wo_a weight must be rank-2, got shape={tuple(weight.shape)}")
    if int(weight.shape[0]) % _DEEPSEEK_V4_O_GROUPS != 0:
        raise ValueError(
            f"wo_a first dimension {int(weight.shape[0])} must divide by {_DEEPSEEK_V4_O_GROUPS}"
        )
    return weight.reshape(_DEEPSEEK_V4_O_GROUPS, int(weight.shape[0]) // _DEEPSEEK_V4_O_GROUPS, int(weight.shape[1]))


def _pack_deepseek_v4_optional_attention(
    prefix: str,
    raw: Mapping[str, torch.Tensor],
    ranks: int,
    *,
    compress_ratio: int,
) -> dict[str, torch.Tensor]:
    """Pack compressor/indexer tensors, filling inactive branch placeholders."""

    def raw_tensor(suffix: str) -> torch.Tensor | None:
        return raw.get(f"{prefix}.{suffix}")

    def zeros(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros((ranks, *shape), dtype=dtype)

    def replicated(tensor: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        tensor = tensor.contiguous().cpu()
        return tensor.unsqueeze(0).expand(ranks, *tensor.shape).contiguous()

    def replicated_transposed(
        suffix: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        *,
        enabled: bool,
    ) -> torch.Tensor:
        tensor = raw_tensor(suffix) if enabled else None
        if tensor is None:
            return zeros(shape, dtype)
        return replicated(tensor.transpose(0, 1).contiguous(), dtype=dtype)

    def replicated_plain(
        suffix: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        *,
        enabled: bool,
    ) -> torch.Tensor:
        tensor = raw_tensor(suffix) if enabled else None
        if tensor is None:
            return zeros(shape, dtype)
        return replicated(tensor, dtype=dtype)

    is_hca = int(compress_ratio) == _DEEPSEEK_V4_HCA_COMPRESS_RATIO
    is_csa = int(compress_ratio) == _DEEPSEEK_V4_CSA_COMPRESS_RATIO
    return {
        "hca_cmp_wkv": replicated_plain(
            "attn.compressor.wkv.weight",
            (_DEEPSEEK_V4_HCA_MAIN_OUT_DIM, _DEEPSEEK_V4_HIDDEN_SIZE),
            torch.bfloat16,
            enabled=is_hca,
        ),
        "hca_cmp_wgate": replicated_plain(
            "attn.compressor.wgate.weight",
            (_DEEPSEEK_V4_HCA_MAIN_OUT_DIM, _DEEPSEEK_V4_HIDDEN_SIZE),
            torch.bfloat16,
            enabled=is_hca,
        ),
        "hca_cmp_ape": replicated_plain(
            "attn.compressor.ape",
            (_DEEPSEEK_V4_HCA_COMPRESS_RATIO, _DEEPSEEK_V4_HCA_MAIN_OUT_DIM),
            torch.float32,
            enabled=is_hca,
        ),
        "hca_cmp_norm_w": replicated_plain(
            "attn.compressor.norm.weight",
            (_DEEPSEEK_V4_HEAD_DIM,),
            torch.bfloat16,
            enabled=is_hca,
        ),
        "csa_cmp_wkv": replicated_plain(
            "attn.compressor.wkv.weight",
            (_DEEPSEEK_V4_CSA_MAIN_OUT_DIM, _DEEPSEEK_V4_HIDDEN_SIZE),
            torch.bfloat16,
            enabled=is_csa,
        ),
        "csa_cmp_wgate": replicated_plain(
            "attn.compressor.wgate.weight",
            (_DEEPSEEK_V4_CSA_MAIN_OUT_DIM, _DEEPSEEK_V4_HIDDEN_SIZE),
            torch.bfloat16,
            enabled=is_csa,
        ),
        "csa_cmp_ape": replicated_plain(
            "attn.compressor.ape",
            (_DEEPSEEK_V4_CSA_COMPRESS_RATIO, _DEEPSEEK_V4_CSA_MAIN_OUT_DIM),
            torch.float32,
            enabled=is_csa,
        ),
        "csa_cmp_norm_w": replicated_plain(
            "attn.compressor.norm.weight",
            (_DEEPSEEK_V4_HEAD_DIM,),
            torch.bfloat16,
            enabled=is_csa,
        ),
        "csa_idx_wq_b": replicated_transposed(
            "attn.indexer.wq_b.weight",
            (_DEEPSEEK_V4_Q_LORA, _DEEPSEEK_V4_ATTENTION_OUT // 4),
            torch.int8,
            enabled=is_csa,
        ),
        "csa_idx_wq_b_scale": replicated_plain(
            "attn.indexer.wq_b.scale",
            (_DEEPSEEK_V4_ATTENTION_OUT // 4,),
            torch.float32,
            enabled=is_csa,
        ),
        "csa_weights_proj": replicated_transposed(
            "attn.indexer.weights_proj.weight",
            (_DEEPSEEK_V4_HIDDEN_SIZE, 64),
            torch.bfloat16,
            enabled=is_csa,
        ),
        "csa_hadamard_idx": replicated(deepseek_v4_hadamard_idx(), dtype=torch.bfloat16),
        "csa_inner_wkv": replicated_plain(
            "attn.indexer.compressor.wkv.weight",
            (_DEEPSEEK_V4_CSA_INNER_OUT_DIM, _DEEPSEEK_V4_HIDDEN_SIZE),
            torch.bfloat16,
            enabled=is_csa,
        ),
        "csa_inner_wgate": replicated_plain(
            "attn.indexer.compressor.wgate.weight",
            (_DEEPSEEK_V4_CSA_INNER_OUT_DIM, _DEEPSEEK_V4_HIDDEN_SIZE),
            torch.bfloat16,
            enabled=is_csa,
        ),
        "csa_inner_ape": replicated_plain(
            "attn.indexer.compressor.ape",
            (_DEEPSEEK_V4_CSA_COMPRESS_RATIO, _DEEPSEEK_V4_CSA_INNER_OUT_DIM),
            torch.float32,
            enabled=is_csa,
        ),
        "csa_inner_norm_w": replicated_plain(
            "attn.indexer.compressor.norm.weight",
            (_DEEPSEEK_V4_HADAMARD_IDX_DIM,),
            torch.bfloat16,
            enabled=is_csa,
        ),
    }


def _pack_deepseek_v4_router(
    prefix: str,
    raw: Mapping[str, torch.Tensor],
    *,
    ranks: int,
    n_routed_experts: int,
    include_tid2eid: bool,
    include_gate_bias: bool,
) -> dict[str, torch.Tensor]:
    """Pack router-only tensors and placeholders for inactive router modes."""
    gate_bias = raw.get(f"{prefix}.ffn.gate.bias")
    if gate_bias is None:
        if include_gate_bias:
            raise KeyError(f"missing raw DeepSeekV4 layer tensor: {prefix}.ffn.gate.bias")
        gate_bias = torch.zeros((n_routed_experts,), dtype=torch.float32)
    tid2eid = raw.get(f"{prefix}.ffn.gate.tid2eid")
    if tid2eid is None:
        if include_tid2eid:
            raise KeyError(f"missing raw DeepSeekV4 layer tensor: {prefix}.ffn.gate.tid2eid")
        tid2eid = torch.zeros((_DEEPSEEK_V4_VOCAB_SIZE, _DEEPSEEK_V4_TOPK), dtype=torch.int32)
    return {
        "gate_bias": gate_bias.to(torch.float32).contiguous().cpu().unsqueeze(0).expand(ranks, -1).contiguous(),
        "tid2eid": tid2eid.to(torch.int32).contiguous().cpu().unsqueeze(0).expand(ranks, *tid2eid.shape).contiguous(),
    }


def _pack_deepseek_v4_routed_experts(
    prefix: str,
    raw: Mapping[str, torch.Tensor],
    *,
    ranks: int,
    n_routed_experts: int,
) -> dict[str, torch.Tensor]:
    """Stack rank-local routed experts into EP-rank-major tensors."""

    def expert(expert_id: int, suffix: str) -> torch.Tensor:
        name = f"{prefix}.ffn.experts.{expert_id}.{suffix}"
        try:
            return raw[name].contiguous().cpu()
        except KeyError as exc:
            raise KeyError(f"missing raw DeepSeekV4 expert tensor: {name}") from exc

    def stack(suffix: str, dtype: torch.dtype) -> torch.Tensor:
        per_rank = []
        for rank in range(ranks):
            ids = deepseek_v4_local_expert_ids(
                rank=rank,
                ranks=ranks,
                n_routed_experts=n_routed_experts,
            )
            per_rank.append(torch.stack([expert(expert_id, suffix).to(dtype=dtype) for expert_id in ids], dim=0))
        return torch.stack(per_rank, dim=0).contiguous()

    return {
        "routed_w1": stack("w1.weight", torch.int8),
        "routed_w1_scale": stack("w1.scale", torch.float32),
        "routed_w3": stack("w3.weight", torch.int8),
        "routed_w3_scale": stack("w3.scale", torch.float32),
        "routed_w2": stack("w2.weight", torch.int8),
        "routed_w2_scale": stack("w2.scale", torch.float32),
    }
