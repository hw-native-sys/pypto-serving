# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Stream real checkpoint layers into resident ND banks without EP/NZ packing."""

from contextlib import ExitStack
import json
import logging
from math import prod

import torch

from pypto_serving.model.common.weights.store import LazySafetensorsStore
from pypto_serving.model.deepseek.weight_loader import deepseek_v4_hadamard_idx

from .bindings import DTYPES, borrow_view, parameter_rows
from .weight_spec import (
    COMPRESSOR_RULES,
    EXPERT_NAMES,
    GLOBAL_SOURCES,
    LAYER_RULES,
    WEIGHT_NAMES,
    checkpoint_coverage,
    layer_ids,
    shard_expert,
    transform_weight,
)


logger = logging.getLogger(__name__)


class LowlatWeightStore(LazySafetensorsStore):
    def __init__(self, model, programs, arena=None):
        metadata = model.extra
        super().__init__(model_dir=metadata["model_dir"], weight_map=metadata["weight_map"])
        self.programs = programs
        self.arena = arena
        self.worker = arena.worker if arena is not None else None
        self.rows = parameter_rows(programs.single_metadata)
        self.owners = {}
        self.embedding = None
        self.coverage = checkpoint_coverage(self.weight_map)
        (programs.output_dir / "lowlat-weight-coverage.json").write_text(
            json.dumps(self.coverage, indent=2) + "\n"
        )

    def planned_bytes_per_rank(self):
        return sum(
            prod(self.rows[name]["shape"][1:])
            * torch.empty((), dtype=DTYPES[self.rows[name]["dtype"]]).element_size()
            for name in WEIGHT_NAMES
        )

    def _layer_shape(self, name):
        if name in ("wo_a", "wo_b"):
            shape = self.rows[name + "_shard"]["shape"][1:]
            if name == "wo_a":
                return (8, *shape[1:])
            return (shape[0] // 43, shape[1] * 8)
        shape = self.rows[name]["shape"][1:]
        return (shape[0] // len(layer_ids(name)), *shape[1:])

    def validate_sources(self):
        """Check real safetensors headers before preparing a device Worker."""
        checked = 0
        examples = {}
        with ExitStack() as stack:
            readers = {}

            def check(name, expected, *, integer=None, channel_scale=False):
                nonlocal checked
                path = self.path_for(name)
                if path not in readers:
                    readers[path] = stack.enter_context(self._safe_open_fn(path, "cpu"))
                source = readers[path].get_slice(name)
                shape, dtype = tuple(source.get_shape()), source.get_dtype()
                valid_shape = shape == expected or (channel_scale and shape == (*expected, 1))
                if not valid_shape or (integer is not None and dtype not in integer):
                    raise ValueError(
                        f"lowlat checkpoint source mismatch: {name}: {shape}/{dtype}, expected {expected}/{integer}"
                    )
                checked += 1
                examples.setdefault(".".join(name.split(".")[-2:]), {"shape": shape, "dtype": dtype})

            for rule in (*LAYER_RULES, *COMPRESSOR_RULES):
                ids = layer_ids(rule.name)
                shape = self._layer_shape(rule.name)
                if rule.groups:
                    shape = (shape[0] * shape[1], *shape[2:])
                if rule.transpose:
                    shape = tuple(reversed(shape))
                for layer in ids:
                    check(
                        f"layers.{layer}.{rule.source}",
                        shape,
                        integer={"I8"} if rule.dtype == torch.int8 else None,
                        channel_scale=rule.name.endswith("_scale"),
                    )
            for name, source in GLOBAL_SOURCES.items():
                shape = (
                    (129280, 4096)
                    if name in ("embed_weight", "lm_head_weight")
                    else tuple(self.rows[name]["shape"][1:])
                )
                check(source, shape)
            for layer in range(43):
                if layer < 3:
                    check(f"layers.{layer}.ffn.gate.tid2eid", (129280, 6), integer={"I32", "I64"})
                else:
                    check(f"layers.{layer}.ffn.gate.bias", (256,))
                for expert in range(257):
                    prefix = f"layers.{layer}.ffn." + (
                        f"experts.{expert}" if expert < 256 else "shared_experts"
                    )
                    for matrix in (1, 3, 2):
                        shape = (4096, 2048) if matrix == 2 else (2048, 4096)
                        check(f"{prefix}.w{matrix}.weight", shape, integer={"I8"})
                        check(f"{prefix}.w{matrix}.scale", (shape[0],), integer={"F32"}, channel_scale=True)
        report = {
            "status": "PASS",
            "checked_source_tensors": checked,
            "weights_bytes_per_rank": self.planned_bytes_per_rank(),
            "source_examples": examples,
        }
        (self.programs.output_dir / "lowlat-source-metadata.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        return report

    def _copy_slice(self, name, rank, offset, value, elements):
        owner = self.owners[name]
        value = value.to(owner.dtype).contiguous()
        if value.numel() != elements:
            raise ValueError(
                f"checkpoint layout mismatch for {name}: {tuple(value.shape)}; expected {elements} values"
            )
        size = value.element_size()
        self.worker.copy_to(
            owner.shards[rank].data_ptr,
            value.data_ptr(),
            elements * size,
            dst_offset=offset * size,
            worker_id=rank,
        )

    def load(self):
        with ExitStack() as stack:
            self._readers = {}
            self._reader_stack = stack
            try:
                return self._load()
            finally:
                self._readers = None

    def load_many(self, names):
        names = tuple(names)
        if getattr(self, "_readers", None) is None:
            return super().load_many(names)
        self.require(names)
        result = {}
        for name in names:
            path = self.path_for(name)
            if path not in self._readers:
                self._readers[path] = self._reader_stack.enter_context(self._safe_open_fn(path, self.device))
            result[name] = self._readers[path].get_tensor(name)
        return result

    def _load(self):
        for name in sorted(WEIGHT_NAMES):
            row = self.rows[name]
            self.owners[name] = self.arena.allocate(tuple(row["shape"]), DTYPES[row["dtype"]])
        for name, source in GLOBAL_SOURCES.items():
            host = self.load_tensor(source)
            if name == "embed_weight":
                if host.shape != (129280, 4096) or host.dtype != torch.bfloat16:
                    raise ValueError("lowlat requires the complete BF16 embedding table")
                self.embedding = host
            owner = self.owners[name]
            elements = prod(owner.shape[1:])
            for rank in range(8):
                value = host
                if name in ("embed_weight", "lm_head_weight"):
                    if tuple(host.shape) != (129280, 4096):
                        raise ValueError(f"invalid vocabulary weight: {source}")
                    value = host[rank * 16160 : (rank + 1) * 16160]
                self._copy_slice(name, rank, 0, value, elements)

        rules = {rule.name: rule for rule in (*LAYER_RULES, *COMPRESSOR_RULES)}
        for layer in range(43):
            names = {name: rule for name, rule in rules.items() if layer in layer_ids(name)}
            raw = self.load_many(f"layers.{layer}.{rule.source}" for rule in names.values())
            for name, rule in names.items():
                value = transform_weight(rule, raw[f"layers.{layer}.{rule.source}"])
                ids = layer_ids(name)
                expected = self._layer_shape(name)
                if tuple(value.shape) != expected:
                    raise ValueError(
                        f"{rule.source}: expected transformed shape {expected}, got {value.shape}"
                    )
                if name not in ("wo_a", "wo_b"):
                    stride = prod(self.owners[name].shape[1:]) // len(ids)
                    for rank in range(8):
                        self._copy_slice(name, rank, ids.index(layer) * stride, value, stride)
                if name in ("wo_a", "wo_b"):
                    shard_name = name + "_shard"
                    stride = prod(self.owners[shard_name].shape[1:]) // 43
                    for rank in range(8):
                        shard = value[rank] if name == "wo_a" else value[:, rank * 1024 : (rank + 1) * 1024]
                        self._copy_slice(shard_name, rank, layer * stride, shard, stride)
            for name, suffix, needed, shape, dtype in (
                ("gate_bias", "ffn.gate.bias", layer >= 3, (256,), torch.float32),
                ("tid2eid", "ffn.gate.tid2eid", layer < 3, (129280, 6), torch.int32),
            ):
                value = (
                    self.load_tensor(f"layers.{layer}.{suffix}")
                    if needed
                    else torch.zeros(shape, dtype=dtype)
                )
                if tuple(value.shape) != shape:
                    raise ValueError(f"invalid {name} source shape: {value.shape}")
                for rank in range(8):
                    self._copy_slice(name, rank, layer * prod(shape), value, prod(shape))
            if layer >= 2 and layer % 2 == 0:
                value = deepseek_v4_hadamard_idx()
                index = (layer - 2) // 2
                for rank in range(8):
                    self._copy_slice("csa_hadamard_idx", rank, index * 128 * 128, value, 128 * 128)
            self._load_experts(layer)
            logger.info("lowlat resident weights: layer %d/43 uploaded", layer + 1)
        return self

    def _load_experts(self, layer):
        # One rank/layer/bank staging allocation bounds host peak memory and
        # amortizes uploads across 257 experts. Readers retain mmap handles;
        # only the selected TP rows/columns are copied into this host bank.
        for name in EXPERT_NAMES:
            matrix = int(name[len("routed_w")])
            scale = name.endswith("_scale")
            expected = (
                ((4096,) if matrix == 2 else (2048,))
                if scale
                else ((4096, 2048) if matrix == 2 else (2048, 4096))
            )
            stride = prod(self.owners[name].shape[1:]) // (43 * 257)
            replicated_w2_scales = None
            if name == "routed_w2_scale":
                replicated_w2_scales = {
                    expert: self.load_tensor(
                        f"layers.{layer}.ffn."
                        + (f"experts.{expert}" if expert < 256 else "shared_experts")
                        + ".w2.scale"
                    )
                    for expert in range(257)
                }
            for rank in range(8):
                bank = torch.empty((257, stride), dtype=self.owners[name].dtype)
                for expert in range(257):
                    prefix = f"layers.{layer}.ffn." + (
                        f"experts.{expert}" if expert < 256 else "shared_experts"
                    )
                    source = (
                        replicated_w2_scales[expert]
                        if replicated_w2_scales is not None
                        else self.load_tensor(f"{prefix}.w{matrix}." + ("scale" if scale else "weight"))
                    )
                    if tuple(source.shape) != expected:
                        if not scale or source.numel() != prod(expected):
                            raise ValueError(
                                f"invalid checkpoint expert shape: {prefix}/{name}: {source.shape}"
                            )
                    value = shard_expert(source, matrix=matrix, scale=scale, rank=rank)
                    bank[expert].copy_(value.reshape(-1))
                self._copy_slice(name, rank, layer * 257 * stride, bank, 257 * stride)

    def prefill_bindings(self):
        result = dict(self.owners)
        for name, source in (("wq_a", "wq_a"), ("wq_b", "wq_b"), ("csa_weights_proj", "csa_weights_proj")):
            row = parameter_rows(self.programs.prefill_metadata)[name]
            result[name] = borrow_view(self.worker, self.owners[source], tuple(row["shape"]))
        for name in ("wo_a", "wo_b"):
            row = parameter_rows(self.programs.prefill_metadata)[name]
            result[name] = borrow_view(self.worker, self.owners[name + "_shard"], tuple(row["shape"]))
        return result

    def lookup_embeddings(self, ids):
        if self.embedding is None:
            raise RuntimeError("lowlat embedding table has not been loaded")
        ids = ids.to(device="cpu", dtype=torch.int64)
        return self.embedding.index_select(0, ids.reshape(-1)).reshape(*ids.shape, 4096)
