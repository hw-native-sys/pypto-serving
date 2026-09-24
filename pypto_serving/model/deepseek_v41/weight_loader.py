# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Selective checkpoint reads for V4.1 composite weight ABIs.

Follows V4's name/spec/store separation. A load returns one matrix and its
scales or one dense tensor, not a resident model or an executable backend.
TP shards attention projections; EP selects whole routed experts. The caller
owns device placement and the lifetime/budget of previously returned bundles.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

import torch

from pypto_serving.model.common.weights.store import LazySafetensorsStore
from .config import V41TextConfig
from .weight_spec import backbone_weight_specs
from .weight_packing import dequantize_output_groups, fp8_input_major, pack_fp4_tiles, pack_mx_scale


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
        for name in ("num_attention_heads", "o_groups", "index_n_heads", "vocab_size"):
            if self.text[name] % tp_size:
                raise ValueError(f"{name} must divide TP size")
        index = json.loads((self.model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"),
                           object_pairs_hook=_unique)
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("checkpoint index requires a nonempty weight_map")
        for name, filename in weight_map.items():
            if not isinstance(name, str) or not isinstance(filename, str):
                raise ValueError("weight_map must map names to shard filenames")
            path = (self.model_dir / filename).resolve()
            if Path(filename).is_absolute() or not path.is_relative_to(self.model_dir):
                raise ValueError("checkpoint shard path must stay inside model directory")
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

    def _read(self, name, ranges):
        spec = self.specs[name]
        value = self.store.load_slice(name, ranges, shape=spec.shape, dtype=spec.dtype)
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

    def load(self, name):
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
        weight = self._read(name, ranges)
        scale = self._read(*requests[1]) if quantized else None
        if spec.dtype == "I8":
            weight = pack_fp4_tiles(weight)
            scale = pack_mx_scale(scale.view(torch.uint8).T.contiguous()).view(torch.float8_e8m0fnu)
            layout = "routed_fp4_tiles_256;scale_mx_b_nn"
        elif spec.dtype == "F8_E4M3" and "dequantize_fp8" in spec.conversion:
            weight = dequantize_output_groups(weight, scale, self.text["o_groups"] // self.tp_size)
            scale, layout = None, "grouped_bf16"
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
