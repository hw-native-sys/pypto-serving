# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU preparation for the V4-style Executor/Runner boundary, not a device backend.

Resolve logical ranks and cache producers before allocating weights or device
state. Kernel selection and cache allocation stay blocked until the composite
ABI is integrated; mode names here describe model semantics, not availability.
"""

from dataclasses import dataclass
import json
from pathlib import Path

from .config import V41TextConfig
from .weight_loader import V41WeightLoader


@dataclass(frozen=True)
class RankPlacement:
    """Contiguous TP groups inside an EP world; rank is not a physical device ID."""

    rank: int
    tp_size: int = 4
    dp_size: int = 2
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
