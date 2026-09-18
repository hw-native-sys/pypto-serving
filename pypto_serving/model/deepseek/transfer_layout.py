# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DSV4 resident-cache views and deterministic page transfer lowering."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from math import prod

from pypto_serving.transfer.types import (
    ProviderCapabilities, ProviderTransferTask, RegionLease, Segment, TransferAttemptRef, integer,
)

COMPONENTS = {
    "ori": ("kv_cache", "ori", False),
    # PR223 split the compressed cache into distinct HCA/CSA physical pools.
    # Keep them distinct through leasing and lowering because their layer sets
    # and per-page storage geometry differ.
    "hca_cmp": ("hca_cmp_kv", "hca_cmp", False),
    "csa_cmp": ("csa_cmp_kv", "csa_cmp", False),
    "idx_k": ("idx_kv_cache", "index", False),
    "idx_scale": ("idx_kv_scale", "index", False),
    "hca_state": ("hca_compress_state", "hca_state", True),
    "csa_state": ("csa_compress_state", "csa_state", True),
    "csa_inner_state": ("csa_inner_compress_state", "csa_inner_state", True),
}


@dataclass(frozen=True)
class ComponentLayout:
    component_id: str
    dtype: str
    item_bytes: int
    layers: tuple[int, ...]
    blocks_per_layer: int
    block_tokens: int
    token_stride_bytes: int

    def __post_init__(self):
        if self.component_id not in COMPONENTS or not self.dtype:
            raise ValueError("unknown component or dtype")
        for name in ("item_bytes", "blocks_per_layer", "block_tokens", "token_stride_bytes"):
            integer(getattr(self, name), name, 1)
        if not isinstance(self.layers, tuple) or not self.layers or len(set(self.layers)) != len(self.layers):
            raise ValueError("layer mapping must be nonempty, immutable and unique")
        for layer in self.layers:
            integer(layer, "layer")
        if self.token_stride_bytes % self.item_bytes:
            raise ValueError("token stride must contain whole elements")
        integer(self.extent, "component extent", 1)

    @property
    def block_stride_bytes(self):
        return self.block_tokens * self.token_stride_bytes

    @property
    def layer_stride_bytes(self):
        return self.blocks_per_layer * self.block_stride_bytes

    @property
    def extent(self):
        return len(self.layers) * self.layer_stride_bytes


@dataclass(frozen=True)
class BlockCopy:
    component_id: str
    layer: int
    source_block: int
    destination_block: int
    valid_tokens: int


@dataclass(frozen=True)
class DSV4Registry:
    model_revision: str
    topology: tuple[int, ...]
    components: tuple[ComponentLayout, ...]

    def __post_init__(self):
        if not self.model_revision or not isinstance(self.topology, tuple) or not self.topology:
            raise ValueError("model revision and topology are required")
        for dim in self.topology:
            integer(dim, "topology dimension", 1)
        if not isinstance(self.components, tuple):
            raise ValueError("components must be immutable")
        if len(self.components) != len(COMPONENTS) or {
            c.component_id for c in self.components
        } != set(COMPONENTS):
            raise ValueError("registry requires every DSV4 physical cache component")
        key, scale = self.entry("idx_k"), self.entry("idx_scale")
        if (key.layers, key.blocks_per_layer, key.block_tokens) != (
                scale.layers, scale.blocks_per_layer, scale.block_tokens):
            raise ValueError("index K/scale logical geometry must match")

    def entry(self, component_id: str) -> ComponentLayout:
        return next(c for c in self.components if c.component_id == component_id)

    @property
    def fingerprint(self) -> str:
        manifest = {
            "schema": 1, "model_revision": self.model_revision, "topology": self.topology,
            "components": [asdict(self.entry(name)) for name in sorted(COMPONENTS)],
            "policies": COMPONENTS, "final_partial": "exclusive_full_page",
        }
        return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    @property
    def layout_fingerprint(self) -> str:
        """Identify transfer geometry without requiring equal arena capacity.

        A K7 Decode worker reserves HBM for drafter state, so its target-cache
        arena can be smaller than the target-only Prefill arena.  The two peers
        remain transfer compatible when dtype, layer mapping and per-page
        strides match.  Capacity is validated separately against each region
        lease when a concrete segment is lowered.
        """
        manifest = {
            "schema": 1,
            "model_revision": self.model_revision,
            "topology": self.topology,
            "components": [
                {
                    "component_id": entry.component_id,
                    "dtype": entry.dtype,
                    "item_bytes": entry.item_bytes,
                    "layers": entry.layers,
                    "block_tokens": entry.block_tokens,
                    "token_stride_bytes": entry.token_stride_bytes,
                }
                for entry in (self.entry(name) for name in sorted(COMPONENTS))
            ],
            "policies": COMPONENTS,
            "final_partial": "exclusive_full_page",
        }
        return hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @classmethod
    def from_device_cache(cls, cache, *, rank: int, model_revision: str, topology: tuple[int, ...],
                          layer_mapping: dict[str, tuple[int, ...]]) -> DSV4Registry:
        """Inspect shape/dtype only; buffer descriptors stay in the owner bridge."""
        entries = []
        for name, (tensor_name, _, _) in COMPONENTS.items():
            stacked = cache[tensor_name] if isinstance(cache, Mapping) else getattr(cache, tensor_name)
            if rank < 0 or rank >= len(stacked.shards):
                raise ValueError("rank outside resident cache")
            shard = stacked.shards[rank]
            shape = tuple(shard.shape)
            layers = layer_mapping[name]
            if len(shape) < 2 or not layers or shape[0] % len(layers):
                raise ValueError("cache is not a contiguous layer/block/token layout")
            elements = prod(shape)
            item_bytes, remainder = divmod(shard.nbytes, elements)
            if remainder or not item_bytes:
                raise ValueError("invalid tensor extent")
            entries.append(ComponentLayout(name, str(shard.dtype), item_bytes, layers,
                                           shape[0] // len(layers), shape[1],
                                           prod(shape[2:]) * item_bytes))
        return cls(model_revision, topology, tuple(entries))

    def lower(self, attempt: TransferAttemptRef, copies: tuple[BlockCopy, ...], *,
              destination_layout_fingerprint: str, source_leases: dict[str, RegionLease],
              destination_leases: dict[str, RegionLease], final: bool,
              capabilities: ProviderCapabilities = ProviderCapabilities()) -> ProviderTransferTask:
        if destination_layout_fingerprint != self.layout_fingerprint:
            raise ValueError("physical layout fingerprint mismatch")
        if type(final) is not bool or not copies:
            raise ValueError("nonempty block manifest and explicit final flag required")
        if len(set(copies)) != len(copies):
            raise ValueError("duplicate block copy")
        index_groups = {}
        occupied = set()
        segments = []
        for copy in sorted(copies, key=lambda c: (c.component_id, c.layer, c.destination_block, c.source_block)):
            if copy.component_id not in COMPONENTS:
                raise ValueError("unknown component")
            entry = self.entry(copy.component_id)
            _, atomic_group, final_only = COMPONENTS[copy.component_id]
            if final_only and not final:
                raise ValueError("mutable state is final-only")
            for name in ("layer", "source_block", "destination_block", "valid_tokens"):
                integer(getattr(copy, name), name)
            src, dst = source_leases[copy.component_id], destination_leases[copy.component_id]
            destination_layer_extent = len(entry.layers) * entry.block_stride_bytes
            destination_blocks, destination_remainder = divmod(
                dst.extent,
                destination_layer_extent,
            )
            if destination_remainder or not destination_blocks:
                raise ValueError("destination arena extent differs from transfer geometry")
            if (copy.layer not in entry.layers or copy.source_block >= entry.blocks_per_layer
                    or copy.destination_block >= destination_blocks
                    or not 0 < copy.valid_tokens <= entry.block_tokens):
                raise ValueError("logical block/layer/token out of bounds")
            if not final and copy.valid_tokens != entry.block_tokens:
                raise ValueError("non-final chunks may only contain closed pages")
            key = (copy.component_id, copy.layer, copy.destination_block)
            if key in occupied:
                raise ValueError("overlapping destination pages")
            occupied.add(key)
            if atomic_group == "index":
                group = (copy.layer, copy.source_block, copy.destination_block, copy.valid_tokens)
                index_groups.setdefault(group, set()).add(copy.component_id)
            if src.extent != entry.extent:
                raise ValueError("source arena extent differs from registry")
            layer_index = entry.layers.index(copy.layer)
            source_layer_offset = layer_index * entry.layer_stride_bytes
            destination_layer_offset = (
                layer_index * destination_blocks * entry.block_stride_bytes
            )
            # Partial final pages are copied in full into exclusively reserved destination pages.
            # The logical manifest retains valid_tokens; padded bytes never grant semantic readiness.
            segment = Segment(copy.component_id, src, dst,
                              source_layer_offset + copy.source_block * entry.block_stride_bytes,
                              destination_layer_offset
                              + copy.destination_block * entry.block_stride_bytes,
                              entry.block_stride_bytes)
            if segments:
                previous = segments[-1]
                if (previous.component_id == segment.component_id and previous.source == segment.source
                        and previous.destination == segment.destination
                        and previous.source_offset + previous.length == segment.source_offset
                        and previous.destination_offset + previous.length == segment.destination_offset):
                    segments[-1] = replace(previous, length=previous.length + segment.length)
                    continue
            segments.append(segment)
        if any(group != {"idx_k", "idx_scale"} for group in index_groups.values()):
            raise ValueError("index K and scale must appear together")
        if attempt.component_manifest_hash != self.manifest_hash(copies, final):
            raise ValueError("component manifest hash mismatch")
        task = ProviderTransferTask(attempt, tuple(segments))
        task.validate(capabilities)
        return task

    def manifest_hash(self, copies: tuple[BlockCopy, ...], final: bool) -> str:
        manifest = {
            "fingerprint": self.fingerprint, "final": final,
            "copies": [asdict(c) for c in sorted(
                copies, key=lambda c: (c.component_id, c.layer, c.source_block,
                                       c.destination_block, c.valid_tokens))],
        }
        return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
