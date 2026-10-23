# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek V4.1 weight loader."""

from __future__ import annotations

import ctypes
import hashlib
import importlib
import json
import logging
import math
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

from pypto_serving.model.common.weights.store import LazySafetensorsStore
from pypto_serving.model.deepseek.weight_loader import (
    _PREPACKED_MIN_CACHE_RESIDENCY,
    _map_shared_prepacked_tensors,
    _sample_file_page_cache_residency,
)

from .encoding import V41TextConfig
from .weight_spec import (
    backbone_weight_specs,
    fp8_input_major,
    pack_grouped_wo_a,
    pack_fp4_tiles,
    pack_mx_scale,
)

# weight_loader
_BYTES = {"BF16": 2, "F32": 4, "F8_E4M3": 1, "F8_E8M0": 1, "I8": 1}
_EXPERT = re.compile(r"^layers\.\d+\.ffn\.experts\.(\d+)\.")


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate checkpoint metadata key: {key}")
        result[key] = value
    return result


def _positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_shard_paths(model_dir, weight_map):
    shared_root = os.environ.get("PYPTO_V41_SHARED_SHARD_DIR")
    shared_root = Path(shared_root).resolve() if shared_root else None
    if shared_root is not None and not shared_root.is_dir():
        raise ValueError("PYPTO_V41_SHARED_SHARD_DIR must be an existing directory")
    for name, filename in weight_map.items():
        if not isinstance(name, str) or not isinstance(filename, str):
            raise ValueError("weight_map must map names to shard filenames")
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("checkpoint shard path must stay inside model directory")
        path = (model_dir / relative).resolve()
        if not path.is_relative_to(model_dir) and (shared_root is None or not path.is_relative_to(shared_root)):
            raise ValueError("checkpoint shard path must stay inside model or shared shard directory")


@dataclass(frozen=True)
class WeightBundle:
    """Owned CPU tensors, with explicit layout and checkpoint provenance."""

    weight: torch.Tensor
    scale: torch.Tensor | None
    layout: str
    source_names: tuple[str, ...]
    tp_rank: int
    ep_rank: int
    estimated_peak_bytes: int


class V41WeightLoader:
    """Read only selected text weights; reject budget overflow before payload I/O.

    max_load_bytes bounds conservative tensor buffers for one operation, not mmap
    address space, the complete process or accumulated caller-owned results.
    """

    def __init__(self, model_dir, *, tp_size=1, tp_rank=0, ep_size=1, ep_rank=0,
                 max_load_bytes=256 << 20, safe_open_fn=None):
        self.model_dir = Path(model_dir).resolve()
        raw = json.loads((self.model_dir / "config.json").read_text(encoding="utf-8"), object_pairs_hook=_unique)
        self.config = V41TextConfig.from_dict(raw)
        self.specs = backbone_weight_specs(raw)
        self.text = raw["text_config"]
        self.tp_size = _positive(tp_size, "tp_size")
        self.ep_size = _positive(ep_size, "ep_size")
        for rank, size, name in ((tp_rank, tp_size, "tp_rank"), (ep_rank, ep_size, "ep_rank")):
            if type(rank) is not int or not 0 <= rank < size:
                raise ValueError(f"invalid {name}")
        self.tp_rank, self.ep_rank = tp_rank, ep_rank
        self.max_load_bytes = _positive(max_load_bytes, "max_load_bytes")
        if self.text["n_routed_experts"] % ep_size:
            raise ValueError("routed expert count must divide EP size")
        for name in ("num_attention_heads", "o_groups", "vocab_size"):
            if self.text[name] % tp_size:
                raise ValueError(f"{name} must divide TP size")
        index = json.loads((self.model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"),
                           object_pairs_hook=_unique)
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("checkpoint index requires a nonempty weight_map")
        _validate_shard_paths(self.model_dir, weight_map)
        # Deferred-module entries are not required, validated or opened. Only
        # declared text weights are reachable through this loader.
        self.store = LazySafetensorsStore(model_dir=self.model_dir, weight_map=weight_map,
                                         safe_open_fn=safe_open_fn)
        self.store.require(self.specs)

    def names(self, layer_id=None):
        """Selected rank's names; None lists global weights, excluding deferred modules."""
        if layer_id is not None and (type(layer_id) is not int or not 0 <= layer_id < self.config.num_hidden_layers):
            raise ValueError("invalid backbone layer_id")
        prefix = None if layer_id is None else f"layers.{layer_id}."
        return tuple(name for name in self.specs
                     if (not name.startswith("layers.") if prefix is None else name.startswith(prefix))
                     and self._owns(name))

    def _owns(self, name):
        match = _EXPERT.match(name)
        if not match:
            return True
        count = self.text["n_routed_experts"] // self.ep_size
        return self.ep_rank * count <= int(match[1]) < (self.ep_rank + 1) * count

    def _ranges(self, name):
        if name not in self.specs:
            raise KeyError(f"not a supported text-backbone weight: {name}")
        if not self._owns(name):
            raise ValueError(f"expert is not owned by this EP rank: {name}")
        spec = self.specs[name]
        ranges = [slice(0, size) for size in spec.shape]
        for axis in (0, 1):
            if f"tp_shard_axis{axis}" in spec.conversion:
                if spec.shape[axis] % self.tp_size:
                    raise ValueError(f"unaligned TP shard: {name}")
                size = spec.shape[axis] // self.tp_size
                ranges[axis] = slice(self.tp_rank * size, (self.tp_rank + 1) * size)
        return tuple(ranges)

    def _estimate(self, requests):
        # Include selected source copies, output/reordering, per-block conversion
        # and byte scratch. No full-model or full FP4 float expansion is performed.
        total = sum(math.prod(p.stop-p.start for p in ranges) * _BYTES[self.specs[name].dtype]
                    for name, ranges in requests)
        estimate = total * 12 + (1 << 20)
        if estimate > self.max_load_bytes:
            raise ValueError(f"weight load requires estimated {estimate} bytes; budget={self.max_load_bytes}")
        return estimate

    def _read(self, name, ranges, session=None):
        spec = self.specs[name]
        store = self.store if session is None else session
        value = store.load_slice(name, ranges, shape=spec.shape, dtype=spec.dtype)
        if spec.dtype == "F8_E8M0" and bool((value.view(torch.uint8) == 255).any()):
            raise ValueError(f"non-finite E8M0 scale: {name}")
        return value

    def load_rows(self, name, start, stop):
        """Read local TP vocabulary rows without loading the complete table.

        start/stop are relative to this rank's embedding/head shard. Selection
        of arbitrary token IDs and device lookup belong to the input stage.
        """
        if name not in ("embed.weight", "head.weight"):
            raise ValueError("load_rows supports embedding and LM-head weights only")
        ranges = self._ranges(name)
        shard = ranges[0]
        if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= shard.stop-shard.start:
            raise ValueError("invalid local vocabulary row range")
        ranges = (slice(shard.start+start, shard.start+stop), ranges[1])
        estimate = self._estimate([(name, ranges)])
        return WeightBundle(self._read(name, ranges), None, "vocabulary_rows_bf16", (name,),
                            self.tp_rank, self.ep_rank, estimate)

    def load(self, name, *, session=None):
        """Load one dense weight or one projection with its required scales.

        Pass the checkpoint .weight name for projections; scales cannot be
        loaded independently because TP slicing and packing must stay aligned.
        """
        ranges = self._ranges(name)
        spec = self.specs[name]
        if spec.dtype == "F8_E8M0":
            raise ValueError("load the projection weight to obtain aligned scales")
        quantized = spec.dtype in ("F8_E4M3", "I8")
        requests = [(name, ranges)]
        scale_name = name.removesuffix(".weight") + ".scale"
        if quantized:
            n_part, k_part = ranges
            n, k = n_part.stop-n_part.start, k_part.stop-k_part.start
            if spec.dtype == "I8" and (n % 256 or (2*k) % 256):
                raise ValueError("routed FP4 weights require complete 256x256 tiles")
            if spec.dtype == "F8_E4M3" and "dequantize_fp8" not in spec.conversion and k % 64:
                raise ValueError("FP8 native projection requires K divisible by 64")
            divisor = 16 if spec.dtype == "I8" else 32
            if any(v % 32 for v in (n_part.start, n_part.stop)) and spec.dtype == "F8_E4M3":
                raise ValueError("FP8 TP shard must align to output block32")
            if k_part.start % divisor or k_part.stop % divisor:
                raise ValueError("quantized TP shard must align to input block32")
            scale_ranges = (n_part if spec.dtype == "I8" else slice(n_part.start//32, n_part.stop//32),
                            slice(k_part.start//divisor, k_part.stop//divisor))
            requests.append((scale_name, scale_ranges))
        estimate = self._estimate(requests)
        weight = self._read(name, ranges, session)
        scale = self._read(*requests[1], session=session) if quantized else None
        if spec.dtype == "I8":
            weight = pack_fp4_tiles(weight)
            scale = pack_mx_scale(scale.view(torch.uint8).T.contiguous()).view(torch.float8_e8m0fnu)
            layout = "routed_fp4_tiles_256;scale_mx_b_nn"
        elif name.endswith(".attn.wo_a.weight"):
            weight, scale = pack_grouped_wo_a(
                weight, scale, self.text["o_groups"] // self.tp_size)
            layout = "grouped_input_major_fp8;scale_mx_b_nn"
        elif spec.dtype == "F8_E4M3":
            weight, scale = fp8_input_major(weight, scale)
            layout = "input_major_fp8;scale_mx_b_nn"
        else:
            if "cast_" in spec.conversion:
                weight = weight.float()
            if name.endswith((".compressor.wkv.weight", ".compressor.wgate.weight",
                              ".indexer.wk.weight", ".indexer.weights_proj.weight")):
                weight = weight.T.contiguous()
                layout = "input_major_dense"
            else:
                layout = "checkpoint_dense"
        return WeightBundle(weight, scale, layout, tuple(item[0] for item in requests),
                            self.tp_rank, self.ep_rank, estimate)


# execution_plan
@dataclass(frozen=True)
class RankPlacement:
    """Contiguous TP groups inside an EP world; rank is not a physical device ID."""

    rank: int
    tp_size: int = 1
    dp_size: int = 8
    ep_size: int = 8

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in (self.tp_size, self.dp_size, self.ep_size)):
            raise ValueError("parallel sizes must be positive integers")
        if self.tp_size * self.dp_size != self.ep_size:
            raise ValueError("Attention TP * DP must equal the MoE EP world size")
        if type(self.rank) is not int or not 0 <= self.rank < self.ep_size:
            raise ValueError("logical rank must belong to the EP world")

    @property
    def tp_rank(self):
        return self.rank % self.tp_size

    @property
    def dp_rank(self):
        return self.rank // self.tp_size

    @property
    def tp_group_start(self):
        return self.dp_rank * self.tp_size


@dataclass(frozen=True)
class LayerPlan:
    """Each layer owns its SWA window; compressed pools use explicit producers.

    REINDEX consumes its KV producer's index-key cache and publishes a new
    Top-K selection. REUSE consumes the index producer's Top-K selection.
    Producers are layer IDs within one request/DP partition, never page IDs.
    """

    layer_id: int
    mode: str
    kv_source: int | None
    index_source: int | None
    candidate_source: int | None


def plan_layers(raw):
    """Follow lib config.layer_config source resolution without importing JIT modules."""
    config = V41TextConfig.from_dict(raw)
    text = raw["text_config"]
    ratios = config.compress_ratios

    def sources(name):
        values = text.get(name)
        if not isinstance(values, list) or any(
            type(i) is not int or not 0 <= i < len(ratios) or ratios[i] == 0 for i in values
        ):
            raise ValueError(f"invalid {name}")
        if len(set(values)) != len(values):
            raise ValueError(f"duplicate {name}")
        return set(values)

    kv, index = sources("kv_source_layer_ids"), sources("index_source_layer_ids")
    if not kv <= index:
        raise ValueError("KV producers must also produce index keys/selections")
    candidate = text.get("candidate_source_layer_id")
    if 1 in ratios and (type(candidate) is not int or candidate not in kv or ratios[candidate] != 1):
        raise ValueError("C1A candidate source must be a C1A Full layer")

    def active(layer, ratio, owners):
        matches = [i for i in owners if i <= layer and ratios[i] == ratio]
        if not matches:
            raise ValueError(f"layer {layer} has no preceding source with compression ratio {ratio}")
        return max(matches)

    result = []
    for layer, ratio in enumerate(ratios):
        if ratio == 0:
            result.append(LayerPlan(layer, "swa", None, None, None))
            continue
        kv_owner, index_owner = active(layer, ratio, kv), active(layer, ratio, index)
        mode = "full" if layer in kv else "reindex" if layer in index else "reuse"
        if ratio == 2 and mode == "reindex":
            raise ValueError("C2A Reindex has no supported model contract")
        if ratio == 1 and (candidate > layer or kv_owner != candidate):
            raise ValueError("C1A candidates must refer to the same preceding compressed KV producer")
        result.append(LayerPlan(layer, f"c{ratio}a_{mode}", kv_owner, index_owner,
                                candidate if ratio == 1 else None))
    return tuple(result)


class V41ExecutionPlan:
    """Metadata and lazy weight preparation to be consumed by the future Executor.

    This deliberately does not register a ModelRunner or allocate generic K/V
    pools: V4.1 needs request/DP-owned window, compressed, index and pending
    state pools whose runtime contracts have not yet been integrated.
    """

    def __init__(self, model_dir, placement: RankPlacement, *, max_load_bytes=256 << 20):
        self.placement = placement
        raw = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
        self.layers = plan_layers(raw)
        self.weights = V41WeightLoader(
            model_dir, tp_size=placement.tp_size, tp_rank=placement.tp_rank,
            ep_size=placement.ep_size, ep_rank=placement.rank, max_load_bytes=max_load_bytes,
        )

    def layer(self, layer_id):
        if type(layer_id) is not int or not 0 <= layer_id < len(self.layers):
            raise ValueError("invalid backbone layer_id")
        return self.layers[layer_id]

    def weight_names(self, layer_id):
        """One projection name per bundle, owned experts only; scales load with payloads.

        Reuse layers consume shared cache state, not their producer's weights.
        Keep checkpoint names until a specific composite binding is validated.
        """
        self.layer(layer_id)
        return tuple(name for name in self.weights.names(layer_id) if not name.endswith(".scale"))

    def load_weight(self, layer_id, name):
        """Load one owned CPU bundle; the eventual runner owns upload/residency limits."""
        if name not in self.weight_names(layer_id):
            raise ValueError("weight is not owned by the selected layer and rank")
        return self.weights.load(name)

    def for_rank(self, rank):
        """Create metadata-only owned weight access for one collective participant."""
        placement = RankPlacement(rank, self.placement.tp_size, self.placement.dp_size, self.placement.ep_size)
        if placement == self.placement:
            return self
        return V41ExecutionPlan(self.weights.model_dir, placement, max_load_bytes=self.weights.max_load_bytes)


# swa_weights
def stack_bytes(values):
    """Stack even float8 tensors on CPU, preserving every payload bit."""
    dtype = values[0].dtype
    if any(v.dtype != dtype or v.shape != values[0].shape for v in values):
        raise ValueError("rank weights must have matching shape and dtype")
    return torch.stack([v.contiguous().view(torch.uint8) for v in values]).view(dtype)


def merge_expert_scales(scales):
    """Repack per-expert MX_B_NN into the lib's flattened expert/K-group ABI.

    Concatenating the already packed buffers would put expert before output
    block, whereas the kernel expects output block before expert/K-group.
    Only scale bytes are reordered; routed weight payloads stay packed FP4.
    """
    groups, width = scales[0].shape
    if groups % 2 or width % 16 or any(s.shape != scales[0].shape for s in scales):
        raise ValueError("expert scales must share a valid MX_B_NN shape")
    logical = [s.contiguous().view(torch.uint8).reshape(width // 16, groups // 2, 16, 2)
               .permute(1, 3, 0, 2).reshape(groups, width) for s in scales]
    return pack_mx_scale(torch.cat(logical)).view(scales[0].dtype)


def mxfp4_pair_lut():
    """Decode two E2M1 nibbles into two E4M3FN bytes in the lib LUT ABI."""
    codes = torch.tensor([0x00, 0x30, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C,
                          0x80, 0xB0, 0xB8, 0xBC, 0xC0, 0xC4, 0xC8, 0xCC], dtype=torch.int64)
    packed = torch.arange(256)
    pair = codes[packed & 15] | (codes[packed >> 4] << 8)
    pair = torch.where(pair < 32768, pair, pair - 65536)
    return pair.to(torch.int16).reshape(1, 256).repeat(2, 1)


def load_swa_layer_weights(model_dir, layer_id, topology, *, max_bundle_bytes=32 << 30):
    """Return stacked Attention and MoE weight maps for one SWA layer.

    The caller owns the results and uploads them once. max_bundle_bytes is a
    conservative bound on stacking buffers for this call, not process memory
    or the accumulated weights of other layers. Individual payload reads retain
    V41WeightLoader's separate pre-read budget.
    """
    return _load_layer_weights(model_dir, layer_id, topology, max_bundle_bytes, swa_only=True)


def load_prefill_layer_weights(model_dir, layer_id, topology, *, max_bundle_bytes=32 << 30,
                               loaders=None, parallel_experts=False):
    """Load this layer's real Attention/MoE weights for the prefill SP entries.

    Full modes own compressor and index-key weights; Reindex owns only index
    query/gating weights; Reuse owns neither. Producer-owned caches and unused
    C1A ABI buffers must be supplied by the resource adapter, not fabricated by
    the loader. Index heads are replicated, matching lib's global INDEX_H.
    """
    return _load_layer_weights(model_dir, layer_id, topology, max_bundle_bytes,
                               swa_only=False, loaders=loaders,
                               parallel_experts=parallel_experts)


def load_prefill_attention_weights(model_dir, layer_id, topology, *, max_bundle_bytes=4 << 30):
    """Read only Attention weights for isolated tracing or bounded preparation.

    Uses the identical packing/placement as the complete layer bundle. In
    particular, an Attention probe must not read all routed expert payloads.
    """
    attention, _ = _load_layer_weights(model_dir, layer_id, topology, max_bundle_bytes,
                                       swa_only=False, include_moe=False)
    return attention


def _load_routed_experts(loader, layer_id, rank, ep_size):
    count = loader.text["n_routed_experts"] // ep_size
    result = {}
    held_bytes = 0
    with loader.store.slice_session() as session:
        for name in ("w1", "w2", "w3"):
            bundles = [loader.load(f"layers.{layer_id}.ffn.experts.{expert}.{name}.weight",
                                   session=session)
                       for expert in range(rank * count, (rank + 1) * count)]
            held_bytes += sum(t.numel() * t.element_size()
                              for bundle in bundles for t in (bundle.weight, bundle.scale)
                              if t is not None)
            result["routed_" + name] = stack_bytes([b.weight for b in bundles])
            result["routed_" + name + "_scale"] = merge_expert_scales([b.scale for b in bundles])
    return result, held_bytes


def _load_layer_weights(model_dir, layer_id, topology, max_bundle_bytes, *, swa_only,
                        include_moe=True, loaders=None, parallel_experts=False):
    if type(max_bundle_bytes) is not int or max_bundle_bytes <= 0:
        raise ValueError("max_bundle_bytes must be positive")
    attention, moe = [], []
    shared_bundles = {}
    held_bytes = 0
    if loaders is None:
        loaders = tuple(V41WeightLoader(
            model_dir, tp_size=topology.tp, tp_rank=rank % topology.tp,
            ep_size=topology.world, ep_rank=rank, max_load_bytes=512 << 20,
        ) for rank in range(topology.world))
    if len(loaders) != topology.world:
        raise ValueError("one V4.1 weight loader is required per EP rank")
    model_path = Path(model_dir).resolve()
    for rank, loader in enumerate(loaders):
        if (loader.model_dir != model_path or loader.tp_size != topology.tp
                or loader.tp_rank != rank % topology.tp or loader.ep_size != topology.world
                or loader.ep_rank != rank):
            raise ValueError("V4.1 weight loader placement does not match the layer topology")
        if type(layer_id) is not int or not 0 <= layer_id < loader.config.num_hidden_layers:
            raise ValueError("selected layer must be a backbone layer")
        ratio = loader.text["compress_ratios"][layer_id]
        if swa_only and ratio != 0:
            raise ValueError("selected layer must be a SWA backbone layer")
    if parallel_experts and (not include_moe or topology.world != 4):
        raise ValueError("parallel expert packing requires a four-rank MoE layer")
    routed = None
    if parallel_experts:
        with ThreadPoolExecutor(max_workers=4) as executor:
            routed = list(executor.map(
                lambda pair: _load_routed_experts(pair[1], layer_id, pair[0], topology.world),
                enumerate(loaders),
            ))
    for rank, loader in enumerate(loaders):
        prefix = f"layers.{layer_id}."
        a, m = {}, {}
        with loader.store.slice_session() as session:
            def load(name):
                nonlocal held_bytes
                key = (loader.tp_rank, name) if not name.startswith("ffn.experts.") else None
                bundle = shared_bundles.get(key) if key is not None else None
                if bundle is None:
                    bundle = loader.load(prefix + name, session=session)
                    if key is not None:
                        shared_bundles[key] = bundle
                held_bytes += sum(t.numel() * t.element_size() for t in (bundle.weight, bundle.scale)
                                  if t is not None)
                if 3 * held_bytes > max_bundle_bytes:
                    raise ValueError("stacked layer weight budget exceeded")
                return bundle

            for target, source in {
                "hc_attn_fn": "hc_attn_fn", "hc_attn_scale": "hc_attn_scale",
                "hc_attn_base": "hc_attn_base", "attn_norm_weight": "attn_norm.weight",
                "q_norm_weight": "attn.q_norm.weight", "kv_norm_weight": "attn.kv_norm.weight",
                "attn_sink": "attn.attn_sink",
            }.items():
                a[target] = load(source).weight
            for name in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
                bundle = load(f"attn.{name}.weight")
                a[name] = bundle.weight
                if bundle.scale is not None:
                    a[name + "_scale"] = bundle.scale
            if layer_id in loader.text["kv_source_layer_ids"]:
                for target, source in {
                    "compressor_wkv": "attn.compressor.wkv.weight",
                    "compressor_norm_weight": "attn.compressor.norm.weight",
                    "index_wk": "attn.indexer.wk.weight",
                    "index_norm_weight": "attn.indexer.k_norm.weight",
                }.items():
                    a[target] = load(source).weight
                if ratio == 2:
                    a["compressor_wgate"] = load("attn.compressor.wgate.weight").weight
            if layer_id in loader.text["index_source_layer_ids"]:
                bundle = load("attn.indexer.wq_b.weight")
                a["index_wq_b"], a["index_wq_b_scale"] = bundle.weight, bundle.scale
                a["index_weights_proj"] = load("attn.indexer.weights_proj.weight").weight
            attention.append(a)
            if not include_moe:
                continue
            for target, source in {
                "hc_ffn_fn": "hc_ffn_fn", "hc_ffn_scale": "hc_ffn_scale", "hc_ffn_base": "hc_ffn_base",
                "norm_weight": "ffn_norm.weight", "gate_weight": "ffn.gate.weight",
                "correction_bias": "ffn.gate.bias",
            }.items():
                m[target] = load(source).weight
            expert_fields, expert_bytes = routed[rank] if routed is not None else (None, 0)
            for name in ("w1", "w2", "w3"):
                bundle = load(f"ffn.shared_experts.{name}.weight")
                m["shared_" + name], m["shared_" + name + "_scale"] = bundle.weight, bundle.scale
                if expert_fields is not None:
                    m["routed_" + name] = expert_fields["routed_" + name]
                    m["routed_" + name + "_scale"] = expert_fields["routed_" + name + "_scale"]
                else:
                    count = loader.text["n_routed_experts"] // topology.world
                    bundles = [load(f"ffn.experts.{expert}.{name}.weight")
                               for expert in range(rank * count, (rank + 1) * count)]
                    m["routed_" + name] = stack_bytes([b.weight for b in bundles])
                    m["routed_" + name + "_scale"] = merge_expert_scales([b.scale for b in bundles])
                    del bundles
            if expert_fields is not None:
                held_bytes += expert_bytes
                if 3 * held_bytes > max_bundle_bytes:
                    raise ValueError("stacked layer weight budget exceeded")
            m["mxfp4_pair_lut"] = mxfp4_pair_lut()
            moe.append(m)
    return ({name: stack_bytes([r[name] for r in attention]) for name in attention[0]},
            {name: stack_bytes([r[name] for r in moe]) for name in moe[0]} if moe else {})


# decode_weights
_ATTENTION = {
    "hc_attn_fn": "hc_attn_fn", "hc_attn_scale": "hc_attn_scale",
    "hc_attn_base": "hc_attn_base", "attn_norm_weight": "attn_norm_weight",
    "wq_a": "wq_a", "wq_a_scale": "wq_a_scale", "q_norm_weight": "q_norm_weight",
    "wq_b": "wq_b", "wq_b_scale": "wq_b_scale", "wkv": "wkv",
    "wkv_scale": "wkv_scale", "kv_norm_weight": "kv_norm_weight",
    "attn_sink": "attn_sink", "wo_a": "wo_a", "wo_a_scale": "wo_a_scale", "wo_b": "wo_b",
    "wo_b_scale": "wo_b_scale",
}
_MOE = {
    "hc_ffn_fn": "hc_ffn_fn", "hc_ffn_scale": "hc_ffn_scale",
    "hc_ffn_base": "hc_ffn_base", "ffn_norm_weight": "norm_weight",
    "gate_weight": "gate_weight", "correction_bias": "correction_bias",
    "routed_w1": "routed_w1", "routed_w1_scale": "routed_w1_scale",
    "routed_w2": "routed_w2", "routed_w2_scale": "routed_w2_scale",
    "routed_w3": "routed_w3", "routed_w3_scale": "routed_w3_scale",
    "shared_w1": "shared_w1", "shared_w1_scale": "shared_w1_scale",
    "shared_w2": "shared_w2", "shared_w2_scale": "shared_w2_scale",
    "shared_w3": "shared_w3", "shared_w3_scale": "shared_w3_scale",
}


@dataclass(frozen=True)
class DecodeWeightSlice:
    """One EP-stacked payload and its destination slot in the decode ABI."""

    name: str
    slot: int
    value: torch.Tensor


def bind_decode_layer_weights(layer, attention, moe, *, kv_sources, index_sources, c2a_sources):
    """Map one loaded layer to the lib's layer/source axes without copying data."""
    if not attention or not moe or "mxfp4_pair_lut" not in moe:
        raise ValueError("decode requires complete Attention and packed-FP4 MoE weights")
    result = [DecodeWeightSlice(target, layer.layer_id, attention[source])
              for target, source in _ATTENTION.items()]
    result += [DecodeWeightSlice(target, layer.layer_id, moe[source])
               for target, source in _MOE.items()]
    if layer.layer_id == 0:
        result.append(DecodeWeightSlice("mxfp4_pair_lut", 0, moe["mxfp4_pair_lut"]))
    if layer.layer_id in kv_sources:
        kv_slot = kv_sources.index(layer.layer_id)
        result.append(DecodeWeightSlice("compressor_norm_weight", kv_slot,
                                        attention["compressor_norm_weight"]))
        if layer.mode == "c2a_full":
            slot = c2a_sources.index(layer.layer_id)
            result.extend((
                DecodeWeightSlice("c2a_compressor_wkv", slot, attention["compressor_wkv"]),
                DecodeWeightSlice("c2a_compressor_wgate", slot, attention["compressor_wgate"]),
            ))
        elif layer.mode == "c1a_full":
            result.append(DecodeWeightSlice("c1a_compressor_wkv", 0, attention["compressor_wkv"]))
        else:
            raise ValueError("decode KV source must be a C1A/C2A Full layer")
        # The decode ABI reserves eight slots, but only the four KV producers
        # have index-key and key-norm checkpoint weights.
        index_slot = index_sources.index(layer.layer_id)
        result.extend((
            DecodeWeightSlice("index_wk", index_slot, attention["index_wk"]),
            DecodeWeightSlice("index_norm_weight", index_slot, attention["index_norm_weight"]),
        ))
    if layer.layer_id in index_sources:
        slot = index_sources.index(layer.layer_id)
        result.extend((
            DecodeWeightSlice("index_wq_b", slot, attention["index_wq_b"]),
            DecodeWeightSlice("index_wq_b_scale", slot, attention["index_wq_b_scale"]),
            DecodeWeightSlice("index_weights_proj", slot, attention["index_weights_proj"]),
        ))
    return tuple(result)


def iter_decode_weight_slices(model_dir, topology, *, max_bundle_bytes=32 << 30,
                              parallel_experts=False):
    """Yield one layer at a time in checkpoint order for bounded staging."""
    raw = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    layers = plan_layers(raw)
    if len(layers) != 40 or (topology.tp, topology.dp, topology.world) not in ((1, 4, 4), (1, 8, 8)):
        raise ValueError("V4.1 requires a 40-layer TP1/DP4/EP4 or TP1/DP8/EP8 schedule")
    text = raw["text_config"]
    kv_sources = tuple(text["kv_source_layer_ids"])
    index_sources = tuple(text["index_source_layer_ids"])
    c2a_sources = tuple(layer.layer_id for layer in layers if layer.mode == "c2a_full")
    if len(kv_sources) != 4 or len(index_sources) != 8 or len(c2a_sources) != 3:
        raise ValueError("lib decode_fwd source axes disagree with the checkpoint schedule")
    loaders = tuple(V41WeightLoader(
        model_dir, tp_size=topology.tp, tp_rank=rank % topology.tp,
        ep_size=topology.world, ep_rank=rank, max_load_bytes=512 << 20,
    ) for rank in range(topology.world))
    for layer in layers:
        attention, moe = load_prefill_layer_weights(
            model_dir, layer.layer_id, topology, max_bundle_bytes=max_bundle_bytes,
            loaders=loaders, parallel_experts=parallel_experts,
        )
        yield from bind_decode_layer_weights(
            layer, attention, moe, kv_sources=kv_sources, index_sources=index_sources,
            c2a_sources=c2a_sources,
        )
    # decode_fwd reserves one key/norm slot per index source, although only KV
    # producers have these checkpoint tensors. The trailing four slots are not
    # read by the current graph; initialize them deterministically nonetheless.
    head_dim = text["head_dim"]
    index_dim = text["index_head_dim"]
    for slot in range(len(kv_sources), len(index_sources)):
        yield DecodeWeightSlice("index_wk", slot,
                                torch.zeros((topology.world, head_dim, index_dim), dtype=torch.bfloat16))
        yield DecodeWeightSlice("index_norm_weight", slot,
                                torch.zeros((topology.world, index_dim), dtype=torch.bfloat16))


# decode_weight_resources
_FLAT_LAYER = frozenset({
    "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_weight", "wq_a_scale",
    "q_norm_weight", "wq_b_scale", "wkv_scale", "kv_norm_weight", "attn_sink",
    "wo_a", "wo_a_scale", "wo_b_scale", "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base",
    "ffn_norm_weight", "gate_weight", "correction_bias", "routed_w1",
    "routed_w1_scale", "routed_w2", "routed_w2_scale", "routed_w3",
    "routed_w3_scale", "shared_w1_scale", "shared_w2_scale", "shared_w3_scale",
})
_AXIS_LAYER = frozenset({
    "wq_a", "wq_b", "wkv", "wo_b", "shared_w1", "shared_w2", "shared_w3",
})
_FLAT_KV = frozenset({"compressor_norm_weight"})
_FLAT_INDEX = frozenset({"index_wk", "index_norm_weight", "index_wq_b_scale"})
_AXIS_INDEX = frozenset({"index_wq_b", "index_weights_proj"})
_AXIS_C2A = frozenset({"c2a_compressor_wkv", "c2a_compressor_wgate"})
_SINGLE = frozenset({"c1a_compressor_wkv", "mxfp4_pair_lut"})
_ALL = (_FLAT_LAYER | _AXIS_LAYER | _FLAT_KV | _FLAT_INDEX |
        _AXIS_INDEX | _AXIS_C2A | _SINGLE)
_LAYER_ABI_ALIASES = {
    "c2a_compressor_wkv": "compressor_wkv",
    "c2a_compressor_wgate": "compressor_wgate",
}


def _destination_shape(layout, name, source_shape):
    if len(source_shape) < 2 or source_shape[0] != layout.world or any(dim <= 0 for dim in source_shape):
        raise ValueError("decode weight geometry requires nonempty EP-stacked slices")
    slots = layout.slots(name)
    if name in _SINGLE:
        return tuple(source_shape)
    # The executable layer ABI borrows one slot at a time. Slot-major storage
    # keeps each [world, ...] view contiguous for the resident upload.
    return (slots, *source_shape)


def estimate_decode_weight_bytes(layout, geometry):
    """Budget all 40-layer shared tensors before reading checkpoint payloads."""
    if set(geometry) != _ALL:
        raise ValueError("decode weight geometry must cover exactly the lib weight fields")
    total = 0
    for name, (shape, dtype) in geometry.items():
        destination = _destination_shape(layout, name, shape)
        total += torch.Size(destination).numel() * torch.empty((), dtype=dtype).element_size()
    return total


@dataclass(frozen=True)
class DecodeWeightLayout:
    layers: int
    kv_sources: int
    index_sources: int
    c2a_sources: int
    world: int = 8
    tp_size: int = 1

    @classmethod
    def from_lib(cls, abi):
        return cls(abi.N_LAYERS, abi.KV_SOURCE_COUNT, abi.INDEX_SOURCE_COUNT,
                   abi.C2A_SOURCE_COUNT, abi.EP_SIZE, abi.TP_SIZE)

    def __post_init__(self):
        if any(type(n) is not int or n <= 0 for n in (
            self.layers, self.kv_sources, self.index_sources, self.c2a_sources,
            self.world, self.tp_size,
        )):
            raise ValueError("decode weight dimensions must be positive integers")
        if self.world % self.tp_size:
            raise ValueError("decode weight world size must be divisible by TP size")

    def slots(self, name):
        if name in _FLAT_LAYER | _AXIS_LAYER:
            return self.layers
        if name in _FLAT_KV:
            return self.kv_sources
        if name in _FLAT_INDEX | _AXIS_INDEX:
            return self.index_sources
        if name in _AXIS_C2A:
            return self.c2a_sources
        if name in _SINGLE:
            return 1
        raise ValueError(f"unknown decode weight ABI field: {name}")


class DecodeWeightBank:
    """Shared host tensors created before the persistent worker forks.

    One slot is written exactly once. Float8 and packed FP4 tensors are copied
    through byte views so no arithmetic conversion can change payload bits.
    """

    def __init__(self, layout: DecodeWeightLayout, *, max_host_bytes=350 << 30):
        if type(max_host_bytes) is not int or max_host_bytes <= 0:
            raise ValueError("decode host weight budget must be positive")
        self.layout = layout
        self.max_host_bytes = max_host_bytes
        self.allocated_bytes = 0
        self.tensors = {}
        self._seen = set()
        self.prepacked = False
        self.rank_major_uploads = {}

    def prepare_rank_major_uploads(self, *, max_field_bytes=4 << 30):
        """Coalesce small slot-major fields into one contiguous upload per rank."""
        if self.rank_major_uploads:
            return
        for name, source in self.tensors.items():
            if name in _SINGLE or self.layout.slots(name) < 2:
                continue
            if source.numel() * source.element_size() > max_field_bytes:
                continue
            slots = self.layout.slots(name)
            if source.shape[:2] != (slots, self.layout.world):
                raise ValueError(f"{name} weight bank has the wrong slot/rank axes")
            target = torch.empty((self.layout.world, slots, *source.shape[2:]),
                                 dtype=source.dtype).share_memory_()
            target.view(torch.uint8).copy_(source.transpose(0, 1).view(torch.uint8))
            self.rank_major_uploads[name] = target

    def append(self, item: DecodeWeightSlice):
        if not isinstance(item, DecodeWeightSlice) or item.name not in _ALL:
            raise ValueError("unknown decode weight slice")
        value = item.value
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                or not value.is_contiguous()):
            raise ValueError("decode weight slice must be contiguous CPU EP-stacked data")
        slots = self.layout.slots(item.name)
        if type(item.slot) is not int or not 0 <= item.slot < slots:
            raise ValueError("decode weight slot is outside its source axis")
        key = (item.name, item.slot)
        if key in self._seen:
            raise ValueError("duplicate decode weight slot")
        shape = _destination_shape(self.layout, item.name, value.shape)
        destination = self.tensors.get(item.name)
        if destination is None:
            size = value.element_size() * torch.Size(shape).numel()
            if size > self.max_host_bytes - self.allocated_bytes:
                raise MemoryError("decode weight bank exceeds the shared host budget")
            destination = torch.empty(shape, dtype=value.dtype).share_memory_()
            self.tensors[item.name] = destination
            self.allocated_bytes += size
        if destination.dtype != value.dtype or tuple(destination.shape) != shape:
            raise ValueError("decode weight slices disagree on shape or dtype")
        target = destination if item.name in _SINGLE else destination[item.slot]
        target.view(torch.uint8).copy_(value.view(torch.uint8))
        self._seen.add(key)

    def finish(self, parameter_names):
        expected = {(name, slot) for name in _ALL for slot in range(self.layout.slots(name))}
        missing = expected - self._seen
        if missing:
            name, slot = sorted(missing)[0]
            raise ValueError(f"decode weight bank is missing {name}[{slot}]")
        self.validate_abi(parameter_names)
        return self.tensors

    @staticmethod
    def validate_abi(parameter_names):
        names = set(parameter_names)
        absent = {name for name in _ALL
                  if name not in names and _LAYER_ABI_ALIASES.get(name) not in names}
        if absent:
            raise ValueError("lib decode ABI omits checkpoint weight fields: " + ", ".join(sorted(absent)))


class ResidentDecodeWeights:
    """Upload the verified shared bank once and release every handle on close."""

    def __init__(self, worker, bank: DecodeWeightBank, parameter_names):
        if not isinstance(bank, DecodeWeightBank):
            raise TypeError("decode weight upload requires a completed host bank")
        bank.finish(parameter_names)
        if not bank.prepacked and any(not tensor.is_shared() for tensor in bank.tensors.values()):
            raise ValueError("decode weights must be shared before worker startup")
        self.worker = worker
        self.layout = bank.layout
        self.tensors = {}
        try:
            for name, tensor in bank.tensors.items():
                slots = self.layout.slots(name)
                for slot in range(slots):
                    host = tensor if name in _SINGLE else tensor[slot]
                    self.tensors[name, slot] = worker.alloc_stacked_tensor(host)
        except BaseException:
            self.close()
            raise

    def view(self, name, slot):
        """Borrow one layer/source slice without a second device allocation."""
        slots = self.layout.slots(name)
        if type(slot) is not int or not 0 <= slot < slots:
            raise ValueError("decode weight view slot is outside its source axis")
        return self.tensors[name, slot]

    def prefill_weights(self, layer, *, kv_sources, index_sources, c2a_sources):
        """Bind real producer weights to the existing half-layer ABI names."""
        if not 0 <= layer.layer_id < self.layout.layers:
            raise ValueError("prefill layer is outside the resident decode bank")
        attention = {source: self.view(target, layer.layer_id)
                     for target, source in _ATTENTION.items()}
        moe = {source: self.view(target, layer.layer_id)
               for target, source in _MOE.items()}
        moe["mxfp4_pair_lut"] = self.view("mxfp4_pair_lut", 0)
        if layer.layer_id in kv_sources:
            source = kv_sources.index(layer.layer_id)
            attention["compressor_norm_weight"] = self.view("compressor_norm_weight", source)
            attention["index_wk"] = self.view("index_wk", index_sources.index(layer.layer_id))
            attention["index_norm_weight"] = self.view("index_norm_weight", index_sources.index(layer.layer_id))
            if layer.mode == "c2a_full":
                slot = c2a_sources.index(layer.layer_id)
                attention["compressor_wkv"] = self.view("c2a_compressor_wkv", slot)
                attention["compressor_wgate"] = self.view("c2a_compressor_wgate", slot)
            elif layer.mode == "c1a_full":
                attention["compressor_wkv"] = self.view("c1a_compressor_wkv", 0)
            else:
                raise ValueError("KV producer must be a C1A/C2A Full layer")
        if layer.layer_id in index_sources:
            slot = index_sources.index(layer.layer_id)
            for name in ("index_wq_b", "index_wq_b_scale", "index_weights_proj"):
                attention[name] = self.view(name, slot)
        return attention, moe

    def close(self):
        for tensor in reversed(tuple(self.tensors.values())):
            self.worker.free_stacked_tensor(tensor)
        self.tensors.clear()


class ResidentFullDecodeWeights(ResidentDecodeWeights):
    """Upload contiguous per-rank banks and alias their layer slices for prefill."""

    def __init__(self, worker, bank: DecodeWeightBank, parameter_names):
        from pypto.runtime import StackedDeviceTensor

        bank.finish(parameter_names)
        self.worker = worker
        self.layout = bank.layout
        self.tensors = {}
        self._full = {}
        self._slot_shapes = {}
        self.layout_names = tuple(bank.tensors)
        self._stacked_tensor = StackedDeviceTensor
        try:
            for name, host in bank.tensors.items():
                if name in _SINGLE:
                    self._full[name] = worker.alloc_stacked_tensor(host)
                    continue
                slots = self.layout.slots(name)
                if host.shape[0:2] != (slots, self.layout.world):
                    raise ValueError(f"{name} weight bank has the wrong slot/rank axes")
                slot_shape = tuple(host.shape[2:])
                self._slot_shapes[name] = slot_shape
                full_shape = ((slots * slot_shape[0], *slot_shape[1:]) if name in
                              (_FLAT_LAYER | _FLAT_KV | _FLAT_INDEX) else
                              (slots, *slot_shape))
                slot_bytes = host[0, 0].numel() * host.element_size()
                shards = []
                try:
                    rank_major = getattr(bank, "rank_major_uploads", {}).get(name)
                    for rank in range(self.layout.world):
                        if rank_major is not None:
                            source = rank_major[rank].view(full_shape)
                            shards.append(worker.alloc_tensor(full_shape, host.dtype,
                                                              init=source, worker_id=rank))
                            continue
                        base = worker.alloc_tensor(full_shape, host.dtype, worker_id=rank)
                        shards.append(base)
                        for slot in range(slots):
                            source = host[slot, rank]
                            if not source.is_contiguous():
                                raise ValueError(f"{name} weight slice is not contiguous")
                            worker.copy_to(
                                base.data_ptr, source.data_ptr(), slot_bytes,
                                dst_offset=slot * slot_bytes, worker_id=rank,
                            )
                except BaseException:
                    for rank, shard in enumerate(shards):
                        worker.free_tensor(shard, worker_id=rank)
                    raise
                self._full[name] = StackedDeviceTensor(
                    shards, (self.layout.world, *full_shape), range(self.layout.world))
        except BaseException:
            self.close()
            raise

    def full(self, name):
        return self._full[name]

    def view(self, name, slot):
        if name in _SINGLE:
            if slot != 0:
                raise ValueError("single weight has only slot zero")
            return self._full[name]
        slots = self.layout.slots(name)
        if type(slot) is not int or not 0 <= slot < slots:
            raise ValueError("decode weight view slot is outside its source axis")
        base = self._full[name]
        shape = self._slot_shapes[name]
        slot_bytes = math.prod(shape) * torch.empty((), dtype=base.dtype).element_size()
        shards = tuple(self.worker.view_tensor(shard, shape, byte_offset=slot * slot_bytes,
                                               worker_id=rank)
                       for rank, shard in enumerate(base.shards))
        return self._stacked_tensor(shards, (self.layout.world, *shape), base.worker_ids)

    def close(self):
        for tensor in reversed(tuple(getattr(self, "_full", {}).values())):
            self.worker.free_stacked_tensor(tensor)
        self._full.clear()
        self.tensors.clear()


# weight_sidecar
_LOG = logging.getLogger(__name__)
_FORMAT = "pypto-deepseek-v41-decode-slot-major-v1"
_FLOAT8 = {"float8_e4m3fn": torch.float8_e4m3fn,
           "float8_e8m0fnu": torch.float8_e8m0fnu}


def sidecar_path(model_dir, world=8):
    override = os.environ.get("PYPTO_V41_PREPACKED_WEIGHTS")
    return Path(override) if override else Path(model_dir) / f"pypto-deepseek-v41-decode-r{world}.safetensors"


def source_fingerprint(model_dir, layout):
    model_dir = Path(model_dir)
    config = (model_dir / "config.json").read_bytes()
    index = (model_dir / "model.safetensors.index.json").read_bytes()
    weight_map = json.loads(index)["weight_map"]
    source_files = []
    for filename in sorted(set(weight_map.values())):
        source = (model_dir / filename).stat()
        source_files.append((filename, source.st_size, source.st_mtime_ns))
    payload = json.dumps({
        "format": _FORMAT,
        "layout": asdict(layout),
        "config_sha256": hashlib.sha256(config).hexdigest(),
        "index_sha256": hashlib.sha256(index).hexdigest(),
        "source_files": source_files,
    }, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def load_prepacked_bank(model_dir, layout, parameter_names, *, path=None):
    path = sidecar_path(model_dir, layout.world) if path is None else Path(path)
    if not path.is_file():
        return None
    from safetensors import safe_open

    fd = os.open(path, os.O_RDONLY)
    try:
        residency = _sample_file_page_cache_residency(fd, path)
        _LOG.info("V4.1 weight sidecar page-cache residency: %s",
                  "unknown" if residency is None else f"{100 * residency:.1f}%")
        force_sidecar = bool(os.environ.get("PYPTO_V41_PREPACKED_WEIGHTS"))
        use_cold_sidecar = force_sidecar
        if not use_cold_sidecar and (residency is None or residency < _PREPACKED_MIN_CACHE_RESIDENCY):
            _LOG.info("Skipping cold V4.1 weight sidecar %s (requires %.1f%% page-cache residency)",
                      path, 100 * _PREPACKED_MIN_CACHE_RESIDENCY)
            return None
        if use_cold_sidecar and (residency is None or residency < _PREPACKED_MIN_CACHE_RESIDENCY):
            _LOG.info("Using cold V4.1 weight sidecar %s", path)
        with safe_open(f"/proc/self/fd/{fd}", framework="pt", device="cpu") as reader:
            metadata = reader.metadata() or {}
            if metadata.get("format") != _FORMAT or metadata.get("source_fingerprint") != source_fingerprint(
                    model_dir, layout):
                _LOG.warning("Ignoring stale V4.1 weight sidecar %s", path)
                return None
            names = frozenset(reader.keys())
            if names != _ALL:
                _LOG.warning("Ignoring V4.1 weight sidecar with wrong fields: %s", path)
                return None
            float8_fields = json.loads(metadata.get("float8_fields", "{}"))
            if any(name not in _ALL or dtype not in _FLOAT8 for name, dtype in float8_fields.items()):
                raise ValueError("V4.1 weight sidecar has invalid float8 metadata")
            for name in names:
                source = reader.get_slice(name)
                shape = tuple(source.get_shape())
                if (shape[0] != (layout.world if name in _SINGLE else layout.slots(name))
                        or (name not in _SINGLE and (len(shape) < 3 or shape[1] != layout.world))):
                    raise ValueError(f"V4.1 sidecar weight has invalid rank/slot shape: {name}")
                if name in float8_fields and source.get_dtype() != "U8":
                    raise ValueError(f"V4.1 sidecar float8 weight is not byte-backed: {name}")
        tensors = _map_shared_prepacked_tensors(fd, names)
        for name, dtype in float8_fields.items():
            tensors[name] = tensors[name].view(_FLOAT8[dtype])
        bank = DecodeWeightBank(layout)
        bank.tensors = tensors
        bank.allocated_bytes = sum(value.numel() * value.element_size() for value in tensors.values())
        bank._seen = {(name, slot) for name in _ALL for slot in range(layout.slots(name))}
        bank.prepacked = True
        bank.finish(parameter_names)
        _LOG.info("Mapped prepacked V4.1 weight bank: %.1f GiB from %s",
                  bank.allocated_bytes / (1 << 30), path)
        return bank
    finally:
        os.close(fd)


def _prefill_bank_metadata(model_dir, layout, lib_revision, manifest):
    return {
        "format": "pypto-deepseek-v41-prefill-bank-v1",
        "source_fingerprint": source_fingerprint(model_dir, layout),
        "lib_revision": lib_revision,
        "ranks": manifest.ranks,
        "banks": {name: {"shape": list(shape), "dtype": str(dtype),
                         "nbytes": math.prod(shape) * torch.empty((), dtype=dtype).element_size()}
                  for name, (shape, dtype) in manifest.tensor_layouts.items()},
    }


class PrepackedPrefillBank:
    """Read a complete rank-major prefill bank produced from this checkpoint and lib."""

    def __init__(self, path, model_dir, layout, lib_revision, manifest):
        self.path = Path(path)
        expected = _prefill_bank_metadata(model_dir, layout, lib_revision, manifest)
        recorded = json.loads((self.path / "manifest.json").read_text(encoding="utf-8"))
        if recorded != expected:
            raise ValueError("V4.1 prefill bank sidecar is stale or has another layout")
        for rank in range(manifest.ranks):
            for name, spec in expected["banks"].items():
                if (self.path / f"rank{rank}-{name}.bin").stat().st_size != spec["nbytes"]:
                    raise ValueError(f"V4.1 prefill bank sidecar is truncated: rank {rank} {name}")
        self.ranks = manifest.ranks
        self.tensor_layouts = manifest.tensor_layouts
        self.bytes_per_rank = manifest.bytes_per_rank

    def upload(self, worker, rank, handles):
        if not 0 <= rank < self.ranks:
            raise ValueError("prefill bank rank is out of range")
        for name, (shape, dtype) in self.tensor_layouts.items():
            size = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
            mapped = torch.from_file(str(self.path / f"rank{rank}-{name}.bin"),
                                     shared=False, size=size, dtype=torch.uint8)
            for offset in range(0, size, 64 << 20):
                count = min(64 << 20, size - offset)
                worker.copy_to(handles[name].data_ptr, mapped.data_ptr() + offset,
                               count, dst_offset=offset, worker_id=rank)


def prepack_prefill_bank(manifest, model_dir, layout, lib_revision, path):
    """Create the reusable prefill bank without holding NPU devices."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    metadata = _prefill_bank_metadata(model_dir, layout, lib_revision, manifest)
    marker = path / "manifest.json"
    if marker.exists():
        PrepackedPrefillBank(path, model_dir, layout, lib_revision, manifest)
        return

    class HostCopy:
        @staticmethod
        def copy_to(dst_ptr, src_ptr, count, *, dst_offset, worker_id):
            ctypes.memmove(dst_ptr + dst_offset, src_ptr, count)

    for rank in range(manifest.ranks):
        tensors = {}
        handles = {}
        for name, spec in metadata["banks"].items():
            target = path / f"rank{rank}-{name}.bin"
            with target.open("wb") as output:
                output.truncate(spec["nbytes"])
            tensor = torch.from_file(str(target), shared=True,
                                     size=spec["nbytes"], dtype=torch.uint8)
            tensors[name] = tensor
            handles[name] = SimpleNamespace(data_ptr=tensor.data_ptr())
        manifest.upload(HostCopy(), rank, handles)
        for name in tensors:
            with (path / f"rank{rank}-{name}.bin").open("r+b") as output:
                os.fsync(output.fileno())
        tensors.clear()
        _LOG.info("Packed V4.1 prefill bank rank %d/%d", rank + 1, manifest.ranks)
    pending = path / "manifest.json.tmp"
    pending.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
    pending.replace(marker)


def _prepack_prefill_main():
    if len(sys.argv) != 5 or sys.argv[1] != "--prefill-bank":
        raise SystemExit("usage: python -m pypto_serving.model.deepseek_v41.weight_loader "
                         "--prefill-bank MODEL_DIR LIB_ROOT OUTPUT_DIR")
    model_dir, lib_root, output = map(Path, sys.argv[2:])
    from .npu_runner import SegmentTopology, load_segment_modules, _revision

    topology = SegmentTopology.for_decode()
    load_segment_modules(lib_root, topology)
    abi = importlib.import_module("models.deepseek_v4_1_flash.decode_fwd")
    utils = importlib.import_module("models.deepseek_v4_1_flash.utils")
    layout = DecodeWeightLayout.from_lib(abi)
    manifest = utils.ResidentPrefillWeights(utils.PrefillCheckpoint(model_dir),
                                            ep_size=topology.world)
    prepack_prefill_bank(manifest, model_dir, layout, _revision(lib_root), output)


if __name__ == "__main__":
    _prepack_prefill_main()
