# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek V4.1 npu runner."""

from __future__ import annotations

import ctypes
import importlib
import json
import logging
import math
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import Callable

import torch

from pypto_serving.config.types import DecodeResult, KVCacheGroupSpec, PrefillResult
from pypto_serving.model.common.runner.buffer_set import alloc_device_buffer

from .task_args import (
    ForwardStep,
    RequestLedger,
    RequestSlice,
    V41CacheGroups,
    build_cache_groups,
    validate_cache_groups,
)
from .weight_loader import (
    DecodeWeightBank,
    DecodeWeightLayout,
    LayerPlan,
    RankPlacement,
    ResidentDecodeWeights,
    V41ExecutionPlan,
    V41WeightLoader,
    iter_decode_weight_slices,
    load_prepacked_bank,
    plan_layers,
)


# cache_resources
@dataclass(frozen=True)
class CacheBufferSpec:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass(frozen=True)
class DecodeCachePlan:
    primary_num_blocks: int
    family_groups: V41CacheGroups
    group_blocks: dict[str, int]
    physical_group_blocks: dict[str, int]
    group_sources: dict[str, dict[str, tuple[int, ...]]]
    buffers: tuple[CacheBufferSpec, ...]
    request_slots: int
    physical_request_slots: int


def cache_bytes_per_rank(plan: DecodeCachePlan) -> int:
    """Count the physical pool, including pages hidden from the scheduler."""
    return sum(torch.Size(spec.shape[1:]).numel() * torch.empty((), dtype=spec.dtype).element_size()
               for spec in plan.buffers)


def make_post_resident_cache_planner(layers, groups, *, max_seq_len, abi,
                                     utilization, max_request_slots,
                                     reserved_bytes_per_rank, requested_pages=None):
    """Build a V4-style post-resident cache planner for one persistent worker.

    The caller must charge every layer scratch allocation that follows cache
    allocation in ``reserved_bytes_per_rank``. Group capacities remain unset
    until the worker reports HBM; the scheduler sees only non-padding pages.
    """
    if (not isinstance(utilization, (float, int)) or not 0 < utilization <= 1
            or type(max_request_slots) is not int or max_request_slots <= 0
            or type(reserved_bytes_per_rank) is not int or reserved_bytes_per_rank <= 0
            or requested_pages is not None and (type(requested_pages) is not int or requested_pages <= 0)):
        raise ValueError("invalid post-resident cache budget")
    if any(group.num_blocks is not None for group in groups):
        raise ValueError("dynamic cache groups must not fix scheduler block counts")
    window_pages = groups[0].max_blocks_per_seq

    def plan_for_slots(slots):
        return plan_decode_cache_pools(
            layers, groups, max_seq_len=max_seq_len,
            primary_num_blocks=slots * window_pages,
            request_slots=slots, abi=abi)

    one, two = plan_for_slots(1), plan_for_slots(2)
    one_bytes, two_bytes = cache_bytes_per_rank(one), cache_bytes_per_rank(two)
    slot_bytes = two_bytes - one_bytes
    fixed_bytes = one_bytes - slot_bytes
    if slot_bytes <= 0 or fixed_bytes <= 0:
        raise ValueError("cache shape does not scale linearly with request slots")

    def choose(worker):
        budgets = []
        for rank in range(one.buffers[0].shape[0]):
            free_bytes, total_bytes = worker.device_memory_info(rank)
            if not 0 <= free_bytes <= total_bytes:
                raise ValueError("device memory report is inconsistent")
            used_bytes = total_bytes - free_bytes
            budgets.append(int(total_bytes * utilization) - used_bytes - reserved_bytes_per_rank)
        available = min(budgets)
        if available < fixed_bytes + slot_bytes:
            raise MemoryError("post-resident HBM cannot hold one V4.1 cache request slot")
        slots = min(max_request_slots, (available - fixed_bytes) // slot_bytes)
        if requested_pages is not None:
            slots = min(slots, requested_pages // window_pages)
        if slots < 1:
            raise MemoryError("configured cache pages cannot hold one V4.1 request slot")
        return slots, plan_for_slots

    return choose


def allocate_cache_with_retry(worker, requested_slots, plan_for_slots, state_reset, page_resets):
    """Release every cache family before retrying a smaller shared slot count."""
    if type(requested_slots) is not int or requested_slots <= 0:
        raise ValueError("requested cache slots must be positive")
    slots = requested_slots
    while True:
        try:
            return DecodeCachePools(worker, plan_for_slots(slots), state_reset, page_resets)
        except (MemoryError, RuntimeError) as exc:
            if slots == 1:
                raise RuntimeError("V4.1 cache allocation failed at the one-slot minimum") from exc
            slots = max(slots // 2, 1)


def plan_decode_cache_pools(layers, groups, *, max_seq_len, primary_num_blocks,
                            request_slots, abi) -> DecodeCachePlan:
    """Match scheduler page capacities to decode_fwd's source-sliced pools.

    ``abi`` is the imported decode_fwd module from the selected lib revision.
    A single primary block count is returned to the grouped scheduler, which
    scales each group by its own max_blocks_per_seq. All shapes include the EP
    rank axis; each rank has a private copy of its DP partition's page IDs.
    """
    resolved = validate_cache_groups(tuple(layers), tuple(groups), max_seq_len)
    by_name = {group.name: group for group in groups}
    window = by_name[resolved.window]
    if groups[0].name != resolved.window:
        raise ValueError("window must be the primary scheduler cache group")
    if (type(primary_num_blocks) is not int or primary_num_blocks <= 0
            or primary_num_blocks % window.max_blocks_per_seq):
        raise ValueError("primary cache blocks must contain whole request slots")
    if type(request_slots) is not int or request_slots <= 0:
        raise ValueError("request slot capacity must be positive")
    slots = primary_num_blocks // window.max_blocks_per_seq
    group_blocks = {}
    for group in groups:
        blocks = slots * group.max_blocks_per_seq
        if group.num_blocks is not None and group.num_blocks != blocks:
            raise ValueError(f"{group.name} cache capacity conflicts with the scheduler")
        group_blocks[group.name] = blocks
    physical_group_blocks = {name: blocks + 1 for name, blocks in group_blocks.items()}

    kv_sources = sum(layer.kv_source == layer.layer_id for layer in layers)
    c2a_sources = sum(layer.mode == "c2a_full" for layer in layers)
    index_sources = sum(layer.index_source == layer.layer_id for layer in layers)
    if (abi.N_LAYERS, abi.KV_SOURCE_COUNT, abi.C2A_SOURCE_COUNT,
            abi.INDEX_SOURCE_COUNT, abi.BLOCK_SIZE) != (
                len(layers), kv_sources, c2a_sources, index_sources, 128):
        raise ValueError("decode cache source axes disagree with the layer schedule")
    kv_layers = tuple(layer for layer in layers if layer.kv_source == layer.layer_id)
    index_layers = tuple(layer for layer in layers if layer.index_source == layer.layer_id)
    group_sources = {
        resolved.window: {"window": tuple(range(len(layers)))},
        resolved.c2a: {
            "compressed": tuple(i for i, layer in enumerate(kv_layers) if layer.mode.startswith("c2a_")),
            "index": tuple(i for i, layer in enumerate(index_layers) if layer.mode.startswith("c2a_")),
        },
        resolved.c1a: {
            "compressed": tuple(i for i, layer in enumerate(kv_layers) if layer.mode.startswith("c1a_")),
            "index": tuple(i for i, layer in enumerate(index_layers) if layer.mode.startswith("c1a_")),
        },
    }
    head, index, world = abi.HEAD_DIM, abi.INDEX_DIM, abi.EP_SIZE
    if (world != 8 or any(type(value) is not int or value <= 0 for value in
                       (head, index, abi.STATE_CAPACITY, abi.STATE_WIDTH))
            or head % abi.WINDOW_CACHE_GROUP or head % abi.COMPRESSED_CACHE_GROUP
            or index % abi.INDEX_CACHE_GROUP):
        raise ValueError("decode cache dimensions disagree with TP4/DP2/EP8")
    block = abi.BLOCK_SIZE
    window_pages = physical_group_blocks[resolved.window]
    # One lib source axis covers both compression families, so its stride must
    # fit the larger physical page namespace without remapping scheduler IDs.
    compressed_pages = max(physical_group_blocks[resolved.c2a],
                           physical_group_blocks[resolved.c1a])
    buffers = (
        CacheBufferSpec("window_cache_pool", (world, len(layers) * window_pages, block, 1, head),
                        torch.float8_e4m3fn),
        CacheBufferSpec("window_cache_scale_pool", (
            world, len(layers) * window_pages, block, 1, head // abi.WINDOW_CACHE_GROUP),
                        torch.float8_e8m0fnu),
        CacheBufferSpec("compressed_cache_pool", (
            world, kv_sources * compressed_pages, block, 1, head // 2), torch.uint8),
        CacheBufferSpec("compressed_cache_scale_pool", (
            world, kv_sources * compressed_pages, block, 1, head // abi.COMPRESSED_CACHE_GROUP),
                        torch.float8_e4m3fn),
        CacheBufferSpec("index_cache_pool", (
            world, index_sources * compressed_pages, block, 1, index // 2), torch.uint8),
        CacheBufferSpec("index_cache_scale_pool", (
            world, index_sources * compressed_pages, block, 1, index // abi.INDEX_CACHE_GROUP),
                        torch.float8_e8m0fnu),
        CacheBufferSpec("state_cache_pool", (
            world, c2a_sources * (request_slots + 1), abi.STATE_CAPACITY, abi.STATE_WIDTH), torch.float32),
    )
    return DecodeCachePlan(primary_num_blocks, resolved, group_blocks, physical_group_blocks,
                           group_sources, buffers, request_slots, request_slots + 1)


def make_state_reset_buffer(plan: DecodeCachePlan) -> torch.Tensor:
    """Create the worker-inherited zero row used when a request slot is freed."""
    shape = next(spec.shape for spec in plan.buffers if spec.name == "state_cache_pool")
    return torch.zeros((shape[0], *shape[2:]), dtype=torch.float32).share_memory_()


def make_cache_page_reset_buffers(plan: DecodeCachePlan) -> dict[str, torch.Tensor]:
    """Allocate one pre-fork zero page per cache payload/scale field."""
    result = {}
    for spec in plan.buffers:
        if spec.name == "state_cache_pool":
            continue
        page = torch.empty((spec.shape[0], *spec.shape[2:]), dtype=spec.dtype).share_memory_()
        page.view(torch.uint8).zero_()
        if spec.name.endswith("scale_pool"):
            page.view(torch.uint8).fill_(0x38 if spec.name == "compressed_cache_scale_pool" else 127)
        result[spec.name] = page
    return result


class DecodeCachePools:
    """Own cache handles; the caller must wait for all launches before close."""

    def __init__(self, worker, plan: DecodeCachePlan, state_reset: torch.Tensor,
                 page_resets: dict[str, torch.Tensor] | None = None):
        state_spec = next(spec for spec in plan.buffers if spec.name == "state_cache_pool")
        if (not isinstance(state_reset, torch.Tensor) or state_reset.device.type != "cpu"
                or state_reset.dtype != torch.float32 or not state_reset.is_shared()
                or not state_reset.is_contiguous()
                or tuple(state_reset.shape) != (state_spec.shape[0], *state_spec.shape[2:])
                or not bool((state_reset == 0).all())):
            raise ValueError("state reset rows must be shared zero FP32 buffers created before worker startup")
        self.worker = worker
        self.plan = plan
        self.state_reset = state_reset
        self.page_resets = page_resets
        if page_resets is not None:
            expected = {spec.name for spec in plan.buffers if spec.name != "state_cache_pool"}
            if set(page_resets) != expected:
                raise ValueError("cache page reset buffers must cover every cache field")
            for spec in plan.buffers:
                if spec.name == "state_cache_pool":
                    continue
                value = page_resets[spec.name]
                if (not isinstance(value, torch.Tensor) or not value.is_shared()
                        or not value.is_contiguous() or value.dtype != spec.dtype
                        or tuple(value.shape) != (spec.shape[0], *spec.shape[2:])):
                    raise ValueError(f"invalid pre-fork reset page for {spec.name}")
        self.tensors = {}
        self.source_counts = {}
        try:
            for spec in plan.buffers:
                if spec.name.startswith("window_cache_"):
                    group = plan.family_groups.window
                    stride = plan.physical_group_blocks[group]
                    sources = len(plan.group_sources[group]["window"])
                elif spec.name.startswith(("compressed_cache_", "index_cache_")):
                    family = "compressed" if spec.name.startswith("compressed_") else "index"
                    stride = max(plan.physical_group_blocks[group] for group, families in
                                 plan.group_sources.items() if family in families)
                    ids = {slot for families in plan.group_sources.values()
                           for slot in families.get(family, ())}
                    sources = len(ids)
                    if ids != set(range(sources)):
                        raise ValueError(f"{family} cache source IDs must be contiguous")
                else:
                    stride = plan.physical_request_slots
                    sources = spec.shape[1] // stride
                if spec.shape[1] != sources * stride:
                    raise ValueError("cache source axis disagrees with the resident pool")
                self.source_counts[spec.name] = sources
                for source in range(sources):
                    self.tensors[spec.name, source] = alloc_device_buffer(
                        worker, (spec.shape[0], stride, *spec.shape[2:]), spec.dtype,
                        stacked=True)
        except BaseException:
            self.close()
            raise

    def view(self, name: str, source: int):
        """Borrow one layer/producer pool with the same page IDs as decode."""
        if name not in self.source_counts:
            raise ValueError(f"unknown V4.1 cache pool: {name}")
        if type(source) is not int or not 0 <= source < self.source_counts[name]:
            raise ValueError("cache source is outside the resident pool")
        return self.tensors[name, source]

    def prefill_views(self, layer, *, kv_sources, index_sources, c2a_sources):
        """Resolve a layer's physical producer pools for its lib prefill entry."""
        result = {
            "window_cache": self.view("window_cache_pool", layer.layer_id),
            "window_cache_scale": self.view("window_cache_scale_pool", layer.layer_id),
        }
        if layer.mode == "swa":
            return result
        kv_slot = kv_sources.index(layer.kv_source)
        index_slot = index_sources.index(layer.index_source)
        for name in ("compressed_cache", "compressed_cache_scale", "index_cache", "index_cache_scale"):
            slot = kv_slot if name.startswith("compressed_") else index_slot
            result[name] = self.view(name + "_pool", slot)
        if layer.mode == "c2a_full":
            result["state_cache"] = self.view("state_cache_pool", c2a_sources.index(layer.layer_id))
        return result

    def reset_state_slot(self, partition: int, slot: int):
        """Clear the three C2A pending-pair rows for one DP request slot."""
        if type(partition) is not int or not 0 <= partition < 2:
            raise ValueError("invalid DP partition")
        if type(slot) is not int or not 0 <= slot < self.plan.physical_request_slots:
            raise ValueError("invalid compressor state slot")
        row = self.state_reset[0]
        row_bytes = row.numel() * row.element_size()
        for rank in range(partition * 4, (partition + 1) * 4):
            source = self.state_reset[rank]
            for index in range(self.source_counts["state_cache_pool"]):
                state = self.view("state_cache_pool", index)
                self.worker.copy_to(
                    state.shards[rank].data_ptr, source.data_ptr(), row_bytes,
                    dst_offset=slot * row_bytes,
                    worker_id=rank,
                )

    def reset_request(self, owner, *, padding=False):
        """Clear only this request's physical pages and C2A pending state."""
        if self.page_resets is None:
            raise ValueError("request reset requires pre-fork cache page buffers")
        partition = owner.partition
        if type(partition) is not int or not 0 <= partition < 2:
            raise ValueError("invalid DP partition")
        if set(owner.pages) != set(self.plan.group_sources):
            raise ValueError("request page families disagree with the cache plan")
        if (type(owner.slot) is not int or
                (padding and owner.slot != self.plan.request_slots) or
                (not padding and not 0 <= owner.slot < self.plan.request_slots)):
            raise ValueError("request state slot is outside its cache family")
        fields = {
            "window": ("window_cache_pool", "window_cache_scale_pool"),
            "compressed": ("compressed_cache_pool", "compressed_cache_scale_pool"),
            "index": ("index_cache_pool", "index_cache_scale_pool"),
        }
        specs = {spec.name: spec for spec in self.plan.buffers}
        for group, pages in owner.pages.items():
            limit = self.plan.physical_group_blocks[group] if padding else self.plan.group_blocks[group]
            if any(type(page) is not int or not 0 <= page < limit
                   for page in pages):
                raise ValueError("request page is outside its cache family")
            if padding and tuple(pages) != (self.plan.group_blocks[group],):
                raise ValueError("padding must use the dedicated cache page")
            for family, sources in self.plan.group_sources[group].items():
                for name in fields[family]:
                    spec = specs[name]
                    zero = self.page_resets[name]
                    page_bytes = zero[0].numel() * zero.element_size()
                    for rank in range(partition * 4, (partition + 1) * 4):
                        for source in sources:
                            target = self.view(name, source)
                            for page in pages:
                                self.worker.copy_to(
                                    target.shards[rank].data_ptr, zero[rank].data_ptr(), page_bytes,
                                    dst_offset=page * page_bytes, worker_id=rank,
                                )
        self.reset_state_slot(partition, owner.slot)

    def reset_padding(self, partition):
        from types import SimpleNamespace

        owner = SimpleNamespace(
            partition=partition, slot=self.plan.request_slots,
            pages={name: (blocks,) for name, blocks in self.plan.group_blocks.items()},
        )
        self.reset_request(owner, padding=True)

    def close(self):
        for tensor in reversed(tuple(self.tensors.values())):
            self.worker.free_stacked_tensor(tensor)
        self.tensors.clear()


# composite
class MissingCompositeInterface(NotImplementedError):
    """The selected lib revision cannot execute the requested serving segment."""


@dataclass(frozen=True)
class BuildOptions:
    """Worker-selected compilation settings passed unchanged to the adapter."""
    platform: str = "a5"
    pypto_build_dir: str | None = None
    use_compile_cache: bool = False


@dataclass(frozen=True)
class LayerState:
    """Opaque device state carried between composites, never converted on host."""
    residual: object
    pre_mix: object
    layout: str = "tp_replicated"


@dataclass(frozen=True)
class CompositeBindings:
    """Explicit adapter seam for a verified lib revision.

    Entries consume (LayerPlan, LayerState, step, resources, weights) and return
    LayerState. initialize consumes (embeddings, step, resources); output consumes
    (state, step, resources) and returns host logits in original request order.
    The adapter owns device uploads and lib ABI binding. Serving never expands
    packed FP4 or calls the sub-operators of a complete layer.
    allocate consumes (plan, device_ids, runtime, BuildOptions), including the
    worker's build directory and compile-cache choice, and returns (resources,
    num_pages). It also prepares global weights needed by initialize/output.

    No default implementation fabricates results. A supplied adapter must handle
    all DP partitions collectively, including empty partitions, and retain input
    objects until wait() completes. reset_request must clear every persistent
    cache and compressor slot for that request before returning.
    """
    revision: str
    entries: Mapping[tuple[str, str], Callable]
    initialize: Callable
    output: Callable
    allocate: Callable
    prepare_weights: Callable
    reset_request: Callable
    wait: Callable
    close: Callable
    cache_groups: tuple = ()
    input_layout: str = "tp_replicated"
    output_layout: str = "tp_replicated"
    decode_backbone: Callable | None = None
    prefill_backbone: Callable | None = None

    def require(self, layers: tuple[LayerPlan, ...], placement: RankPlacement) -> None:
        if not self.revision:
            raise ValueError("composite bindings must identify the validated lib revision")
        phases = tuple(phase for phase, backbone in (
            ("prefill", self.prefill_backbone), ("decode", self.decode_backbone),
        ) if not callable(backbone))
        missing = sorted({f"{phase}/{layer.mode}" for phase in phases
                          for layer in layers if not callable(self.entries.get((phase, layer.mode)))})
        if missing:
            raise MissingCompositeInterface("missing complete layer composites: " + ", ".join(missing))
        for name in ("initialize", "output", "allocate", "prepare_weights", "reset_request", "wait", "close"):
            if not callable(getattr(self, name)):
                raise MissingCompositeInterface(f"missing composite resource/output operation: {name}")
        if self.input_layout not in ("tp_replicated", "tp_local_token") or self.output_layout != self.input_layout:
            raise MissingCompositeInterface("layer entries must preserve their declared residual/pre_mix layout")
        if (placement.tp_size, placement.dp_size, placement.ep_size) != (4, 2, 8):
            raise ValueError("V4.1 serving currently targets TP4/DP2/EP8")


def load_composite_bindings(model_dir=None, max_seq_len=None) -> CompositeBindings:
    """Resolve the model-specific contract without opening a device."""
    if model_dir is None or max_seq_len is None:
        raise ValueError("V4.1 composite bindings require a model directory and context length")
    return make_production_bindings(model_dir, max_seq_len)


# decode_layer_resources
class DecodeLayerResources:
    """Own decode scratch and selections; borrow uploaded weights and caches."""

    def __init__(self, worker, topology, plans, weights, cache, metadata, param_names, *,
                 kv_sources, index_sources, c2a_sources, index_topk=512):
        self.worker, self.topology, self.plans = worker, topology, tuple(plans)
        if len(self.plans) != 40 or tuple(p.layer_id for p in self.plans) != tuple(range(40)):
            raise ValueError("decode layer binding requires the complete ordered backbone")
        if type(index_topk) is not int or index_topk <= 0:
            raise ValueError("decode index top-K must be positive")
        self._owned = []
        world, local, tokens = topology.world, topology.local_capacity, topology.capacity

        def allocate(shape, dtype):
            value = alloc_device_buffer(worker, shape, dtype, stacked=True)
            self._owned.append(value)
            return value

        try:
            states = [(allocate((world, local, 4, 5120), torch.float32),
                       allocate((world, local, 4), torch.float32)) for _ in range(2)]
            scratch = {
                "attention_output": allocate((world, local, 5120), torch.bfloat16),
                "attention_hidden": allocate((world, local, 4, 5120), torch.float32),
                "attention_pre_mix": allocate((world, local, 4), torch.float32),
                "gathered": allocate((world, tokens, 5120), torch.bfloat16),
                "x_mixed": allocate((world, local, 5120), torch.bfloat16),
            }
            topk = {source: allocate((world, tokens, index_topk), torch.int32)
                    for source in index_sources}
            unused_topk = allocate((world, tokens, index_topk), torch.int32)
            unused_indices = allocate((world, tokens, index_topk), torch.int32)
            compressed_pages = max(cache.plan.physical_group_blocks[group] for group, families in
                                   cache.plan.group_sources.items() if "compressed" in families)
            candidate = allocate((world, tokens, compressed_pages * 128), torch.uint8)
            c2a_metadata = metadata.arguments("c2a_full")
            c1a_metadata = metadata.arguments("c1a_full")
            result = {}
            for plan in self.plans:
                layer, mode = plan.layer_id, plan.mode
                a, m = weights.prefill_weights(
                    plan, kv_sources=kv_sources, index_sources=index_sources,
                    c2a_sources=c2a_sources)
                arguments = dict(a)
                arguments.update(m)
                arguments["ffn_norm_weight"] = arguments.pop("norm_weight")
                family_metadata = c1a_metadata if mode.startswith("c1a_") else c2a_metadata
                arguments.update(family_metadata)
                arguments.update(cache.prefill_views(
                    plan, kv_sources=kv_sources, index_sources=index_sources,
                    c2a_sources=c2a_sources))
                if mode == "swa":
                    kv_source, index_source = kv_sources[0], index_sources[0]
                else:
                    kv_source, index_source = plan.kv_source, plan.index_source
                kv_slot = kv_sources.index(kv_source)
                index_slot = index_sources.index(index_source)
                for name in ("compressor_norm_weight",):
                    arguments[name] = weights.view(name, kv_slot)
                for name in ("index_wk", "index_norm_weight"):
                    arguments[name] = weights.view(name, index_sources.index(kv_source))
                for name in ("index_wq_b", "index_wq_b_scale", "index_weights_proj"):
                    arguments[name] = weights.view(name, index_slot)
                arguments["compressor_wkv"] = weights.view("c2a_compressor_wkv", 0)
                arguments["compressor_wgate"] = weights.view("c2a_compressor_wgate", 0)
                arguments["c1a_compressor_wkv"] = weights.view("c1a_compressor_wkv", 0)
                if mode == "c2a_full":
                    c2a_slot = c2a_sources.index(layer)
                    arguments["compressor_wkv"] = weights.view("c2a_compressor_wkv", c2a_slot)
                    arguments["compressor_wgate"] = weights.view("c2a_compressor_wgate", c2a_slot)
                if mode == "swa":
                    for name in ("compressed_cache", "compressed_cache_scale",
                                 "index_cache", "index_cache_scale"):
                        slot = kv_slot if name.startswith("compressed_") else index_slot
                        arguments[name] = cache.view(name + "_pool", slot)
                if mode.startswith("c1a_"):
                    for name in ("index_cache", "index_cache_scale"):
                        arguments[name] = cache.view(name + "_pool", index_sources.index(kv_source))
                arguments["state_cache"] = cache.view(
                    "state_cache_pool", c2a_sources.index(layer) if mode == "c2a_full" else 0)
                arguments["token_to_req_indices"] = c2a_metadata["token_to_req_indices"]
                arguments["position_ids"] = c2a_metadata["position_ids"]
                arguments["query_start_loc"] = c2a_metadata["query_start_loc"]
                arguments["state_block_table"] = c2a_metadata["state_block_table"]
                arguments["topk_indices"] = (
                    topk[index_source] if mode == "c2a_reuse" else topk.get(layer, unused_topk)
                )
                arguments["candidate_mask"] = candidate
                arguments["compressed_indices"] = (
                    topk[index_source] if mode == "c1a_reuse" else unused_indices)
                arguments.update(scratch)
                arguments["x_next"], arguments["next_pre_mix"] = states[layer % 2]
                reserved = {"x_hc", "incoming_pre_mix", "layer_id", "attention_num_tokens",
                            "moe_num_tokens", "attention_epoch", "moe_epoch"}
                missing = set(param_names) - reserved - arguments.keys()
                if missing:
                    raise ValueError(f"decode layer {layer} has unbound ABI fields: {sorted(missing)}")
                result[layer] = arguments
            self.arguments = result
        except BaseException:
            self.close()
            raise

    def close(self):
        for value in reversed(self._owned):
            self.worker.free_stacked_tensor(value)
        self._owned.clear()


# decode_padding
def pad_single_decode(step: ForwardStep, embeddings: torch.Tensor, cache_plan):
    """Return a device-only filler; the request ledger retains the original step."""
    if not isinstance(step, ForwardStep) or step.phase != "decode":
        raise ValueError("decode padding requires a decode ForwardStep")
    if (not isinstance(embeddings, torch.Tensor) or embeddings.device.type != "cpu"
            or embeddings.dtype != torch.bfloat16 or embeddings.ndim != 2
            or embeddings.shape[0] != len(step.token_ids)):
        raise ValueError("decode padding requires one packed BF16 embedding per token")
    counts = tuple(sum(len(request.token_ids) for request in step.requests
                       if request.partition == partition) for partition in range(2))
    if counts == (1, 1):
        return step, embeddings, None
    if len(step.requests) != 1 or counts not in ((1, 0), (0, 1)):
        raise ValueError("only one-request asymmetric decode has isolated filler resources")
    if (len(cache_plan.group_blocks) != 3 or
            set(cache_plan.group_blocks) != set(cache_plan.group_sources) or
            any(cache_plan.physical_group_blocks.get(group, 0) <= blocks
                for group, blocks in cache_plan.group_blocks.items()) or
            cache_plan.physical_request_slots <= cache_plan.request_slots):
        raise ValueError("decode filler requires all three reserved cache families")
    partition = counts.index(0)
    name = "__v41_decode_padding__"
    while any(request.request_id == name for request in step.requests):
        name += "_"
    filler = RequestSlice(
        request_id=name, partition=partition, state_slot=cache_plan.request_slots,
        start=0, token_ids=(step.requests[0].token_ids[0],), prompt_length=1,
        pages=MappingProxyType({group: (blocks,) for group, blocks in cache_plan.group_blocks.items()}),
    )
    padded = ForwardStep(step.phase, (*step.requests, filler), step.epoch)
    return padded, torch.cat((embeddings, embeddings[:1]), dim=0), partition


# input_preparation
def lookup_token_embeddings(
    loaders: Sequence[V41WeightLoader],
    token_ids: torch.Tensor,
    *,
    max_prepare_bytes: int = 256 << 20,
    max_rows_per_read: int = 64,
) -> torch.Tensor:
    """Return owned CPU BF16 rows with shape ``(*token_ids.shape, hidden_size)``.

    Supply one complete set of TP vocabulary shards from the same checkpoint,
    in any order. DP groups share this checkpoint table; token order and all
    leading dimensions are preserved, without duplicating rows for TP ranks.
    IDs must be CPU int32/int64 with at least one dimension. Empty inputs are
    allowed, but padding sentinels such as -1 are not vocabulary indices.

    Repeated IDs are read once. Adjacent unique rows in the same TP shard are
    read together, with at most max_rows_per_read rows per load_rows call.
    max_prepare_bytes bounds conservative preparation tensor storage, including
    the result, unique rows, indexing scratch and a read buffer; the loader's
    separate max_load_bytes still bounds its own read/conversion operation.
    Neither budget represents total process memory or previously returned data.
    """
    for value, name in ((max_prepare_bytes, "max_prepare_bytes"),
                        (max_rows_per_read, "max_rows_per_read")):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    shards = tuple(loaders)
    if not shards or any(not isinstance(loader, V41WeightLoader) for loader in shards):
        raise ValueError("embedding lookup requires V41WeightLoader TP shards")
    first = shards[0]
    if len(shards) != first.tp_size or {loader.tp_rank for loader in shards} != set(range(first.tp_size)):
        raise ValueError("embedding lookup requires exactly one loader per TP shard")
    if any(loader.model_dir != first.model_dir or loader.config != first.config
           or loader.tp_size != first.tp_size or loader.ep_size != first.ep_size for loader in shards):
        raise ValueError("embedding TP shards must use the same checkpoint and parallel sizes")
    if not isinstance(token_ids, torch.Tensor) or token_ids.layout != torch.strided:
        raise ValueError("token_ids must be a strided CPU tensor")
    if token_ids.device.type != "cpu" or token_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("token_ids must be a CPU int32/int64 tensor")
    if token_ids.ndim == 0:
        raise ValueError("token_ids must have at least one dimension")

    hidden = first.config.hidden_size
    count = token_ids.numel()
    if count == 0:
        return torch.empty((*token_ids.shape, hidden), dtype=torch.bfloat16)
    # Bound allocations before flattening or sorting. Worst case every ID is
    # unique; count*128 allows conservative int64 indexing/sorting scratch.
    row_bytes = hidden * 2
    estimate = (2 * count + min(count, max_rows_per_read)) * row_bytes + count * 128
    if estimate > max_prepare_bytes:
        raise ValueError(
            f"embedding preparation requires estimated {estimate} bytes; budget={max_prepare_bytes}"
        )
    if bool(((token_ids < 0) | (token_ids >= first.config.vocab_size)).any()):
        raise ValueError("token_ids must be inside the checkpoint vocabulary")

    flat = token_ids.reshape(-1).to(torch.int64)
    unique, inverse = torch.unique(flat, sorted=True, return_inverse=True)
    unique_rows = torch.empty((unique.numel(), hidden), dtype=torch.bfloat16)
    by_rank = {loader.tp_rank: loader for loader in shards}
    shard_size = first.config.vocab_size // first.tp_size
    start = 0
    while start < unique.numel():
        token = int(unique[start])
        rank, local_row = divmod(token, shard_size)
        stop = start + 1
        limit = min(unique.numel(), start + max_rows_per_read, start + shard_size - local_row)
        while stop < limit and int(unique[stop]) == token + stop - start:
            stop += 1
        bundle = by_rank[rank].load_rows("embed.weight", local_row, local_row + stop - start)
        unique_rows[start:stop].copy_(bundle.weight)
        del bundle
        start = stop
    return unique_rows.index_select(0, inverse).reshape(*token_ids.shape, hidden)


# metadata
_MAX_REQUESTS_PER_DP = 32
_MAX_PREFILL_TOKENS_PER_DP = 4096


def _validate_dp_capacity(partitions, token_counts, *, phase):
    requests = [0, 0]
    tokens = [0, 0]
    for partition, count in zip(partitions, token_counts):
        if type(partition) is not int or not 0 <= partition < 2:
            raise ValueError("requests require an explicit DP cache partition in [0, 2)")
        requests[partition] += 1
        tokens[partition] += count
    if any(count > _MAX_REQUESTS_PER_DP for count in requests):
        raise ValueError("V4.1 lib supports at most 32 requests per DP partition")
    if phase == "prefill" and any(count > _MAX_PREFILL_TOKENS_PER_DP for count in tokens):
        raise ValueError("V4.1 lib supports at most 4096 prefill tokens per DP partition")


def validate_page_tables(rows, partitions, ends, groups):
    """Copy private full-history page tables and reject active request aliases.

    Physical IDs are local to each DP partition. Without an explicit num_blocks
    the device allocator must check its actual capacity before launching work.
    Rolling groups require a position-to-ring-slot ABI and are not guessed here.
    """
    if not groups or any(not isinstance(group, KVCacheGroupSpec) for group in groups):
        raise MissingCompositeInterface("V4.1 requires explicit grouped cache specifications")
    if len({group.name for group in groups}) != len(groups):
        raise ValueError("cache group names must be unique")
    if any(group.num_partitions != 2 for group in groups):
        raise ValueError("V4.1 cache groups require two DP partitions")
    if any(group.sliding_window is not None for group in groups):
        raise MissingCompositeInterface("rolling cache page lowering requires a verified composite contract")
    if len(rows) != len(partitions) or len(rows) != len(ends):
        raise ValueError("page tables, partitions and request extents must have matching rows")
    required = {group.name for group in groups}
    owners, result = set(), []
    for pages, partition, end in zip(rows, partitions, ends):
        if type(partition) is not int or not 0 <= partition < 2:
            raise ValueError("requests require an explicit DP cache partition in [0, 2)")
        if type(end) is not int or end <= 0:
            raise ValueError("cache request extent must be a positive integer")
        if not isinstance(pages, Mapping) or set(pages) != required:
            raise ValueError("request page tables must match the declared cache groups")
        copied = {}
        for group in groups:
            ids = pages[group.name]
            if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes)):
                raise ValueError("a cache page table must be a sequence of physical IDs")
            ids = tuple(ids)
            if any(type(page) is not int or page < 0 for page in ids):
                raise ValueError("physical page IDs must be nonnegative integers")
            if group.num_blocks is not None and any(page >= group.num_blocks for page in ids):
                raise ValueError("physical page ID exceeds the configured cache pool")
            required_pages = math.ceil(end / group.spec.token_capacity)
            if not required_pages <= len(ids) <= group.max_blocks_per_seq:
                raise ValueError("cache page table does not cover the request extent within its capacity")
            if len(set(ids)) != len(ids):
                raise ValueError("a full-history request must not alias its own cache pages")
            addresses = {(partition, group.name, page) for page in ids}
            if owners.intersection(addresses):
                raise ValueError("active requests must not share writable cache pages")
            owners.update(addresses)
            copied[group.name] = ids
        result.append(copied)
    return tuple(result)


def prefill_requests(batch, config, runtime, groups):
    """Build RequestLedger.begin_prefill items from the shared packed batch.

    seq_lens is the end of the current chunk; prompt_lens is the original total
    prompt length. Confusing them would mark the first chunk as terminal and
    prevent subsequent chunks from continuing their persistent state.
    """
    count = len(batch.request_ids)
    if not count or count > runtime.max_batch_size:
        raise ValueError("prefill request count exceeds the configured batch capacity")
    if any(not isinstance(key, str) or not key for key in batch.request_ids):
        raise ValueError("request IDs must be nonempty strings")
    if len(set(batch.request_ids)) != count:
        raise ValueError("prefill requests must have distinct IDs")
    for name in ("chunk_lens", "chunk_offsets", "chunk_starts", "seq_lens", "prompt_lens",
                 "block_ids_by_group", "cache_partitions"):
        if len(getattr(batch, name)) != count:
            raise ValueError(f"prefill {name} must contain one entry per request")
    tokens = batch.token_ids
    if not isinstance(tokens, torch.Tensor) or tokens.device.type != "cpu" or tokens.ndim != 1:
        raise ValueError("prefill token IDs must be a flat CPU tensor")
    if tokens.dtype not in (torch.int32, torch.int64):
        raise ValueError("prefill token IDs must use int32 or int64")
    if tokens.numel() > runtime.max_num_batched_tokens:
        raise ValueError("prefill tokens exceed the configured dispatch capacity")
    token_ids = tokens.tolist()
    if any(not 0 <= token < config.vocab_size for token in token_ids):
        raise ValueError("prefill token ID is outside the vocabulary")

    cursor, chunks = 0, []
    for size, offset, start, end, prompt in zip(
        batch.chunk_lens, batch.chunk_offsets, batch.chunk_starts, batch.seq_lens, batch.prompt_lens,
    ):
        if any(type(value) is not int for value in (size, offset, start, end, prompt)):
            raise ValueError("prefill offsets and lengths must be integers")
        if size <= 0 or offset != cursor or offset + size > len(token_ids):
            raise ValueError("prefill chunks must cover consecutive packed token spans")
        if not 0 <= start < end == start + size <= prompt <= min(
            runtime.max_seq_len, config.max_position_embeddings,
        ):
            raise ValueError("prefill chunk extent or total prompt length is invalid")
        if runtime.max_prefill_tokens_per_request is not None and size > runtime.max_prefill_tokens_per_request:
            raise ValueError("prefill chunk exceeds the per-request capacity")
        chunks.append(tuple(token_ids[offset:offset + size]))
        cursor += size
    if cursor != len(token_ids):
        raise ValueError("prefill chunk spans leave unclaimed packed tokens")
    _validate_dp_capacity(batch.cache_partitions, batch.chunk_lens, phase="prefill")
    pages = validate_page_tables(
        batch.block_ids_by_group, batch.cache_partitions, batch.seq_lens, groups,
    )
    return [
        (key, partition, start, chunk, prompt, table)
        for key, partition, start, chunk, prompt, table in zip(
            batch.request_ids, batch.cache_partitions, batch.chunk_starts, chunks, batch.prompt_lens, pages,
        )
    ]


def decode_requests(batch, config, runtime, groups):
    """Decode consumes exactly one supplied token per request; no padding rows."""
    count = len(batch.request_ids)
    if not count or count > min(runtime.max_batch_size, runtime.max_num_batched_tokens):
        raise ValueError("decode request count exceeds the configured batch capacity")
    if len(set(batch.request_ids)) != count or any(not isinstance(k, str) or not k for k in batch.request_ids):
        raise ValueError("decode requires distinct nonempty request IDs")
    tokens, lengths = batch.token_ids, batch.seq_lens
    for value in (tokens, lengths):
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype not in (
            torch.int32, torch.int64,
        ):
            raise ValueError("decode tokens and lengths must be CPU integer tensors")
    if tuple(tokens.shape) not in ((count,), (count, 1)) or tuple(lengths.shape) != (count,):
        raise ValueError("decode requires one token and one sequence length per request")
    ids, ends = tokens.reshape(-1).tolist(), lengths.tolist()
    if any(not 0 <= token < config.vocab_size for token in ids):
        raise ValueError("decode token ID is outside the vocabulary")
    if any(not 1 <= end <= min(runtime.max_seq_len, config.max_position_embeddings) for end in ends):
        raise ValueError("decode sequence length exceeds model capacity")
    if len(batch.cache_partitions) != count or len(batch.block_ids_by_group) != count:
        raise ValueError("decode requires one partition and grouped page table per request")
    _validate_dp_capacity(batch.cache_partitions, (1 for _ in range(count)), phase="decode")
    pages = validate_page_tables(batch.block_ids_by_group, batch.cache_partitions, ends, groups)
    return [(key, partition, end - 1, token, table)
            for key, partition, end, token, table in zip(
                batch.request_ids, batch.cache_partitions, ends, ids, pages)]


# swa_segment
ATTENTION_ARGS = (
    "x_hc incoming_pre_mix hc_attn_fn hc_attn_scale hc_attn_base attn_norm_weight "
    "wq_a wq_a_scale q_norm_weight wq_b wq_b_scale wkv wkv_scale kv_norm_weight "
    "attn_sink wo_a wo_b wo_b_scale rope_cos rope_sin window_slots window_indices "
    "window_cache window_cache_scale output next_pre_mix hidden attn_out num_tokens attention_epoch"
).split()
MOE_ARGS = (
    "x_hc pre_mix hc_ffn_fn hc_ffn_scale hc_ffn_base norm_weight gate_weight correction_bias "
    "routed_w1 routed_w1_scale routed_w2 routed_w2_scale routed_w3 routed_w3_scale mxfp4_pair_lut "
    "shared_w1 shared_w1_scale shared_w2 shared_w2_scale shared_w3 shared_w3_scale "
    "next_pre_mix x_mixed x_next num_tokens moe_epoch"
).split()


@dataclass(frozen=True)
class SegmentTopology:
    tp: int = 4
    dp: int = 2
    local_capacity: int = 16

    @classmethod
    def for_decode(cls, *, tp: int = 4, dp: int = 2):
        """Match lib's padded local MoE extent for a full decode batch."""
        if type(tp) is not int or tp not in (1, 2, 4, 8):
            raise ValueError("unsupported decode TP size")
        decode_tokens = 32 * (5 + 1)
        row_tile = 16
        local_rows = (decode_tokens + tp - 1) // tp
        return cls(tp=tp, dp=dp, local_capacity=(local_rows + row_tile - 1) // row_tile * row_tile)

    def __post_init__(self):
        if self.tp not in (1, 2, 4, 8) or type(self.tp) is not int:
            raise ValueError("unsupported TP size")
        if type(self.dp) is not int or self.dp <= 0 or self.world not in (2, 4, 8):
            raise ValueError("TP * DP must be a supported EP size")
        if type(self.local_capacity) is not int or self.local_capacity <= 0:
            raise ValueError("local capacity must be positive")

    @property
    def world(self):
        return self.tp * self.dp

    @property
    def capacity(self):
        return self.tp * self.local_capacity

    def counts(self, group_counts):
        """Contiguous SP slabs; padded/empty ranks still participate in collectives."""
        if len(group_counts) != self.dp or any(
            type(n) is not int or not 0 <= n <= self.capacity for n in group_counts
        ):
            raise ValueError("one bounded active-token count is required per DP group")
        attention = tuple(n for n in group_counts for _ in range(self.tp))
        local = tuple(max(0, min(self.local_capacity, n - r * self.local_capacity))
                      for n in group_counts for r in range(self.tp))
        return attention, local


def load_segment_modules(lib_root, topology):
    """Use V4's explicit lib path and import-time topology, in a model worker.

    Never reload an already imported model with another topology/revision.
    Lib currently captures dimensions from argv when modules are imported.
    """
    root = Path(lib_root).resolve()
    package = "models.deepseek_v4_1_flash"
    existing = sys.modules.get(package + ".config")
    expected = root / "models/deepseek_v4_1_flash/config.py"
    if existing is not None and Path(existing.__file__).resolve() != expected:
        raise RuntimeError("V41 lib from another checkout is already imported; use a fresh worker")
    old_argv, old_path = sys.argv[:], sys.path[:]
    try:
        sys.path.insert(0, str(root))
        sys.argv = [old_argv[0], "--tp", str(topology.tp), "--ep", str(topology.world)]
        config = importlib.import_module(package + ".config")
        if Path(config.__file__).resolve() != expected:
            raise RuntimeError("import resolved to a different lib checkout")
        if (config.TP_SIZE, config.EP_SIZE, config.MOE_TOKENS) != (
            topology.tp, topology.world, topology.local_capacity
        ):
            raise ValueError("lib topology/MOE_TOKENS disagrees with the segment capacity")
        attention = importlib.import_module(package + ".prefill_swa")
        moe = importlib.import_module(package + ".moe")
        if moe.SKIP_SHARED_TEST or moe.SKIP_TRANSPORT_TEST:
            raise ValueError("cannot run serving with lib test-only MoE bypasses")
        return attention, moe
    finally:
        sys.argv, sys.path = old_argv, old_path


def compile_segment(compiler, lib_root, topology):
    """Compile the two existing lib composites with the shared V4 compiler."""
    import pypto.language as pl

    attention, moe = load_segment_modules(lib_root, topology)
    swa = compiler.compile(
        "v41_swa_segment", attention.make_hc_program(topology.capacity, topology.world, epochs=1),
        attention_epoch=pl.RUNTIME,
    )
    ffn = compiler.compile("v41_moe_segment", moe.l3_moe, moe_epoch=pl.RUNTIME)
    return swa, ffn


def make_segment_worker(programs, run_config, inherited_host_tensors=()):
    """Retain both programs' communication windows as in the V4 runner.

    Attention waits for (epoch - 1) * 2 before publishing the next phase.
    The runtime's default reset-on-reuse policy would erase that completion
    signal and stall the second layer. Each compiled program owns its windows.
    """
    from pypto.runtime import DistributedWorker

    return DistributedWorker(
        [program.compiled for program in programs], config=run_config,
        persistent=True, reset_persistent_windows=False,
        inherited_host_tensors=inherited_host_tensors,
    )


class SwaSegment:
    """Synchronous half-layer dispatcher on one persistent DistributedWorker.

    Construct the worker with make_segment_worker(). Metadata must be shared
    CPU tensors created before worker startup:
    attention_counts [world, 1] and moe_counts [world], both int32.
    Per-layer argument maps contain all ABI tensors except input state, counts
    and epoch. They own separate output/scratch/cache buffers. No host copies
    occur between half-layers or between consecutive calls to run_layer().
    """

    def __init__(self, worker, programs, topology, attention_counts, moe_counts, run_config):
        import torch

        for value, shape in ((attention_counts, (topology.world, 1)),
                             (moe_counts, (topology.world,))):
            if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                    or tuple(value.shape) != shape or value.dtype != torch.int32 or not value.is_shared()):
                raise ValueError("counts must be shared CPU int32 metadata allocated before worker startup")
        self.worker, self.programs, self.topology = worker, programs, topology
        self.attention_counts, self.moe_counts = attention_counts, moe_counts
        self.run_config = run_config
        self._epoch = 0
        self._attention_epochs = {}
        self._failed = False

    def _check_state(self, state):
        import torch
        from pypto.runtime import StackedDeviceTensor

        if state.layout != "tp_local_token":
            raise ValueError("SWA segment requires contiguous TP-local token state")
        prefix = (self.topology.world, self.topology.local_capacity)
        for value, shape in ((state.residual, (*prefix, 4, 5120)), (state.pre_mix, (*prefix, 4))):
            if (not isinstance(value, StackedDeviceTensor) or tuple(value.shape) != shape
                    or value.dtype != torch.float32
                    or tuple(value.worker_ids) != tuple(range(self.topology.world))):
                raise ValueError("state must be FP32 device shards with canonical rank placement and capacity")

    def run_layer(self, state, attention, moe, *, group_counts):
        """Return MoE outputs directly for the following layer's Attention input.

        Request packing, RoPE, page mapping and cache initialization are owned
        by the caller and must use the same contiguous group/slab ordering.
        """
        return self._run_layer(state, attention, moe, group_counts=group_counts,
                               attention_program=self.programs[0], argument_names=ATTENTION_ARGS,
                               input_mix="incoming_pre_mix", output_residual="output")

    def _run_layer(self, state, attention, moe, *, group_counts, attention_program,
                   argument_names, input_mix, output_residual):
        """Dispatch an audited TP-local Attention ABI and the shared MoE entry."""
        if self._failed:
            raise RuntimeError("segment dispatch failed; close this worker before attempting recovery")
        self._check_state(state)
        group, local = self.topology.counts(group_counts)
        a = dict(attention, x_hc=state.residual, num_tokens=self.attention_counts)
        a[input_mix] = state.pre_mix
        attention_state = LayerState(a[output_residual], a["next_pre_mix"], "tp_local_token")
        m = dict(moe, x_hc=attention_state.residual, pre_mix=attention_state.pre_mix,
                 num_tokens=self.moe_counts)
        result = LayerState(m["x_next"], m["next_pre_mix"], "tp_local_token")
        self._check_state(attention_state)
        self._check_state(result)
        # Reject aliases before launching either kernel, including distinct
        # handles pointing at the same allocation. Reuse across layers is fine.
        buffers = (state.residual, state.pre_mix, attention_state.residual,
                   attention_state.pre_mix, result.residual, result.pre_mix)
        seen = set()
        for buffer in buffers:
            for rank, shard in zip(buffer.worker_ids, buffer.shards):
                key = (rank, shard.data_ptr)
                if key in seen:
                    raise ValueError("input and half-layer outputs must not alias")
                seen.add(key)
        moe_reset = self._epoch == 512
        epoch = 1 if moe_reset else self._epoch + 1
        program_key = id(attention_program.compiled)
        prior_attention_epoch = self._attention_epochs.get(program_key, 0)
        attention_reset = prior_attention_epoch == 512
        attention_epoch = 1 if attention_reset else prior_attention_epoch + 1
        if max(epoch, attention_epoch) > (2**31 - 1) // 2:
            raise OverflowError("communication epoch exhausted; recreate worker")
        # Each compiled Attention owns separate retained signal windows. A
        # mode's first call must use epoch 1 even after another mode ran.
        a["attention_epoch"] = ctypes.c_int32(attention_epoch)
        m["moe_epoch"] = ctypes.c_int32(epoch)
        # Resolve all arguments before the first collective: a missing MoE
        # weight must not leave an already-mutated Attention cache behind.
        a_args = [a[name] for name in argument_names]
        m_args = [m[name] for name in MOE_ARGS]
        self._check_weights(m)
        for rank, (global_n, local_n) in enumerate(zip(group, local)):
            self.attention_counts[rank, 0] = global_n
            self.moe_counts[rank] = local_n
        self._epoch = epoch
        self._attention_epochs[program_key] = attention_epoch
        if attention_reset or moe_reset:
            if not getattr(self.worker, "_persistent", False) or not hasattr(
                self.worker, "_reset_persistent_windows"
            ):
                raise RuntimeError("prefill epoch reset requires a persistent PyPTO worker")
            previous_reset = self.worker._reset_persistent_windows
        try:
            if attention_reset:
                self.worker._reset_persistent_windows = True
            try:
                self.worker.run(attention_program.compiled, *a_args, config=self.run_config)
            finally:
                if attention_reset:
                    self.worker._reset_persistent_windows = previous_reset
            if moe_reset:
                self.worker._reset_persistent_windows = True
            try:
                self.worker.run(self.programs[1].compiled, *m_args, config=self.run_config)
            finally:
                if moe_reset:
                    self.worker._reset_persistent_windows = previous_reset
        except BaseException:
            self._failed = True
            raise
        return result

    @staticmethod
    def _check_weights(moe):
        import torch
        from pypto.runtime import StackedDeviceTensor

        for name in ("routed_w1", "routed_w2", "routed_w3", "mxfp4_pair_lut"):
            value = moe[name]
            expected = torch.int16 if name == "mxfp4_pair_lut" else torch.uint8
            if not isinstance(value, StackedDeviceTensor) or value.dtype != expected:
                raise ValueError(f"{name} must be a resident packed-FP4 ABI tensor ({expected})")


# compressed_metadata
@dataclass(frozen=True)
class CompressedMetadata:
    position_ids: torch.Tensor
    request_ids: torch.Tensor
    query_start_loc: torch.Tensor
    compressed_lens: torch.Tensor
    compressed_slots: torch.Tensor
    compressed_rope_positions: torch.Tensor
    index_block_table: torch.Tensor
    state_block_table: torch.Tensor
    group_counts: tuple[int, ...]


def prepare_compressed_metadata(
    step: ForwardStep,
    topology: SegmentTopology,
    *,
    ratio: int,
    compressed_group: str,
    cache_pages: int,
    max_requests: int,
    state_blocks: int,
    max_prepare_bytes: int = 16 << 20,
) -> CompressedMetadata:
    """Lower full-history pages with joint KV/index physical page numbering.

    The resource allocator must use the same page IDs for compressed KV and
    index-key pools. Each pool stores 128 compressed rows per page. This does
    not select candidates/Top-K or materialize compressed attention indices.
    Those remain device outputs from the appropriate lib producer composite.

    Request IDs are local packed batch rows; state slots are stable ledger IDs.
    Ratio-2 publishes only completed pairs, rotating them at the pair's first
    position. An odd-length chunk leaves its tail in the request's state block.
    The formulas match lib metadata.paged_slots/compressor_metadata at fbe92bfc.
    """
    if not isinstance(step, ForwardStep) or not isinstance(topology, SegmentTopology):
        raise ValueError("compressed metadata requires a ForwardStep and SegmentTopology")
    if type(ratio) is not int or ratio not in (1, 2):
        raise ValueError("compression ratio must be 1 or 2")
    if (type(cache_pages) is not int or not 0 < cache_pages <= (2**31 - 1) // 128
            or type(max_requests) is not int or max_requests <= 0
            or type(state_blocks) is not int or not 0 < state_blocks <= 2**31 - 1
            or type(max_prepare_bytes) is not int or max_prepare_bytes <= 0):
        raise ValueError("cache, request, state and allocation capacities must be positive and fit the ABI")
    if not isinstance(compressed_group, str) or not compressed_group:
        raise ValueError("compressed_group must identify a jointly numbered KV/index pool")
    if step.phase not in ("prefill", "decode"):
        raise ValueError("unsupported forward phase")
    groups = [[] for _ in range(topology.dp)]
    counts, owners, slots, names = [0] * topology.dp, set(), set(), set()
    columns = 1
    for request in step.requests:
        group, count = request.partition, len(request.token_ids)
        if (type(group) is not int or not 0 <= group < topology.dp or count <= 0
                or type(request.start) is not int or not 0 <= request.start < request.end <= 2**31 - 1):
            raise ValueError("invalid compressed request partition or extent")
        if request.request_id in names:
            raise ValueError("duplicate request in forward step")
        names.add(request.request_id)
        if step.phase == "decode" and count != 1:
            raise ValueError("decode requires one token per request")
        if counts[group] + count > topology.capacity or len(groups[group]) >= max_requests:
            raise ValueError("compressed requests exceed token or request capacity")
        if type(request.state_slot) is not int or not 0 <= request.state_slot < state_blocks:
            raise ValueError("request state slot is outside the compressor pool")
        key = (group, request.state_slot)
        if ratio == 2 and key in slots:
            raise ValueError("requests must own distinct compressor state slots")
        slots.add(key)
        pages = request.pages.get(compressed_group)
        needed = (request.end // ratio + 127) // 128
        if not isinstance(pages, (tuple, list)) or len(pages) < needed:
            raise ValueError("compressed pages must cover all completed history")
        for page in pages:
            if type(page) is not int or not 0 <= page < cache_pages:
                raise ValueError("compressed page is outside the joint KV/index pool")
            if (group, page) in owners:
                raise ValueError("compressed pages must be private within each DP group")
            owners.add((group, page))
        columns = max(columns, len(pages))
        groups[group].append((request, counts[group], tuple(pages)))
        counts[group] += count
    # Include DP scratch plus replicated TP outputs and conservative small work arrays.
    estimate = topology.dp * (topology.capacity * 40 + max_requests * (columns + 4) * 4) * (topology.tp + 1)
    if estimate > max_prepare_bytes:
        raise ValueError("compressed metadata exceeds its allocation budget")
    shape = (topology.dp, topology.capacity)
    positions = torch.zeros(shape, dtype=torch.int32)
    ids = torch.full(shape, -1, dtype=torch.int32)
    lens = torch.zeros(shape, dtype=torch.int32)
    writes = torch.full(shape, -1, dtype=torch.int64)
    rope = torch.full(shape, -1, dtype=torch.int32)
    starts = torch.zeros(topology.dp, max_requests + 1, dtype=torch.int32)
    tables = torch.full((topology.dp, max_requests, columns), -1, dtype=torch.int32)
    states = torch.full((topology.dp, max_requests, 1), -1, dtype=torch.int32)
    for group, requests in enumerate(groups):
        for batch_row, (request, offset, pages) in enumerate(requests):
            count = len(request.token_ids)
            section = slice(offset, offset + count)
            p = torch.arange(request.start, request.end, dtype=torch.int32)
            positions[group, section] = p
            ids[group, section] = batch_row
            lens[group, section] = (p + 1) // ratio
            if ratio == 2:
                states[group, batch_row, 0] = request.state_slot
            if pages:
                tables[group, batch_row, :len(pages)] = torch.tensor(pages, dtype=torch.int32)
            starts[group, batch_row + 1:] = offset + count
            for local, position in enumerate(range(request.start, request.end)):
                if (position + 1) % ratio == 0:
                    logical = position // ratio
                    writes[group, offset + local] = pages[logical // 128] * 128 + logical % 128
                    rope[group, offset + local] = position + 1 - ratio
    def replicate(tensor):
        return tensor.repeat_interleave(topology.tp, dim=0)
    return CompressedMetadata(*(replicate(t) for t in (positions, ids, starts, lens, writes, rope, tables, states)),
                              tuple(counts))


# decode_backbone
def compile_decode_layer(compiler, lib_root, topology):
    """Compile lib's complete per-layer Block with family-specific metadata."""
    import pypto.language as pl

    load_segment_modules(lib_root, topology)
    module = importlib.import_module("models.deepseek_v4_1_flash.decode_layer")
    entry = module.l3_decode_layer
    program = compiler.compile(
        "v41_decode_layer", entry, layer_id=pl.RUNTIME,
        attention_num_tokens=pl.RUNTIME, attention_epoch=pl.RUNTIME,
        moe_epoch=pl.RUNTIME,
    )
    return program, tuple(entry.param_names)


class DecodeLayerSegment:
    """Dispatch lib's complete Block once per layer on one persistent worker."""

    def __init__(self, worker, program, param_names, run_config, topology, moe_counts):
        expected = {"x_hc", "incoming_pre_mix", "x_next", "next_pre_mix",
                    "layer_id", "attention_num_tokens", "moe_num_tokens",
                    "attention_epoch", "moe_epoch"}
        self.param_names = tuple(param_names)
        if len(set(self.param_names)) != len(self.param_names) or not expected <= set(self.param_names):
            raise ValueError("decode layer ABI is missing complete Block state or counts")
        if (not isinstance(moe_counts, torch.Tensor) or moe_counts.device.type != "cpu"
                or moe_counts.dtype != torch.int32 or moe_counts.shape != (topology.world,)
                or not moe_counts.is_shared()):
            raise ValueError("decode MoE counts must be shared before worker startup")
        self.worker, self.program, self.run_config = worker, program, run_config
        self.topology, self.moe_counts = topology, moe_counts
        self.epoch = 0
        self.failed = False

    def run_layer(self, state: LayerState, plan, arguments, *, group_counts):
        if self.failed:
            raise RuntimeError("decode Block failed; recreate the worker")
        if state.layout != "tp_local_token":
            raise ValueError("decode Block requires TP-local state")
        if type(plan.layer_id) is not int or not 0 <= plan.layer_id < 40:
            raise ValueError("decode Block requires a checkpoint layer ID")
        if (len(group_counts) != self.topology.dp or
                any(type(n) is not int or not 0 <= n <= self.topology.capacity for n in group_counts)):
            raise ValueError("decode Block requires one bounded count per DP group")
        if len(set(group_counts)) != 1 or not group_counts[0]:
            raise MissingCompositeInterface(
                "decode_layer has one scalar attention_num_tokens for both DP groups; "
                "asymmetric or empty groups require isolated filler rows and cache pages "
                "before serving can safely use that ABI")
        reserved = {"x_hc", "incoming_pre_mix", "layer_id", "attention_num_tokens",
                    "moe_num_tokens", "attention_epoch", "moe_epoch"}
        if reserved & arguments.keys():
            raise ValueError("decode Block state and runtime scalars must come from dispatch")
        missing = set(self.param_names) - reserved - arguments.keys()
        if missing:
            raise ValueError("missing decode Block arguments: " + ", ".join(sorted(missing)))
        if (arguments["x_next"] is state.residual or arguments["next_pre_mix"] is state.pre_mix
                or arguments["x_next"] is arguments["next_pre_mix"]):
            raise ValueError("decode Block input and output state must not alias")
        if self.epoch >= (2**31 - 1) // 2:
            raise OverflowError("decode Block communication epoch exhausted")
        _, local = self.topology.counts(group_counts)
        self.moe_counts.copy_(torch.tensor(local, dtype=torch.int32))
        reset_windows = self.epoch == 512
        epoch = 1 if reset_windows else self.epoch + 1
        bound = dict(arguments, x_hc=state.residual, incoming_pre_mix=state.pre_mix,
                     layer_id=ctypes.c_int32(plan.layer_id),
                     attention_num_tokens=ctypes.c_int32(group_counts[0]),
                     moe_num_tokens=self.moe_counts,
                     attention_epoch=ctypes.c_int32(epoch), moe_epoch=ctypes.c_int32(epoch))
        if reset_windows:
            if not getattr(self.worker, "_persistent", False) or not hasattr(
                self.worker, "_reset_persistent_windows"
            ):
                raise RuntimeError("decode epoch reset requires a persistent PyPTO worker")
            previous_reset = self.worker._reset_persistent_windows
            self.worker._reset_persistent_windows = True
        try:
            self.worker.run(self.program.compiled,
                            *(bound[name] for name in self.param_names), config=self.run_config)
        except BaseException:
            self.failed = True
            raise
        finally:
            if reset_windows:
                self.worker._reset_persistent_windows = previous_reset
        self.epoch = epoch
        return LayerState(arguments["x_next"], arguments["next_pre_mix"], "tp_local_token")


# npu_runner
class V41ModelRunner:
    """Own one collective session; missing entries fail before resource creation.

    This follows the shared runner lifecycle but deliberately does not inherit
    its generic K/V allocator: V4.1 pools include index and compressor state.
    """
    def __init__(self, plan: V41ExecutionPlan, bindings: CompositeBindings, *, device_ids, runtime,
                 build_options: BuildOptions = BuildOptions()):
        self.plan, self.bindings = plan, bindings
        self.device_ids = tuple(device_ids)
        if len(self.device_ids) != plan.placement.ep_size or len(set(self.device_ids)) != len(self.device_ids):
            raise ValueError("one distinct physical device is required for each logical EP rank")
        if any(type(i) is not int or i < 0 for i in self.device_ids):
            raise ValueError("device IDs must be nonnegative integers")
        bindings.require(plan.layers, plan.placement)
        if len(plan.layers) == 40:
            self.cache_groups = validate_cache_groups(
                plan.layers, bindings.cache_groups, runtime.max_seq_len)
        else:
            self.cache_groups = None
        self.runtime = runtime
        self.build_options = build_options
        self.resources = None
        self.num_pages = None
        self.closed = False
        self.failed = False
        self.ledger = None
        self._lock = RLock()
        self._plans = None

    def preflight(self):
        if self.failed:
            raise RuntimeError("runner initialization failed; close the session before retrying")
        if self.closed:
            raise RuntimeError("runner is closed")
        if self.resources is not None:
            return self.num_pages
        try:
            resources, pages = self.bindings.allocate(
                self.plan, self.device_ids, self.runtime, self.build_options)
            self.resources = resources
            if resources is None or type(pages) is not int or pages <= 0:
                raise ValueError("composite allocator must return resources and a positive page capacity")
            self.bindings.wait(resources)
            self.num_pages = pages
        except Exception:
            self.failed = True
            self.close()
            raise
        return self.num_pages

    def close(self):
        if self.closed:
            return
        if self.resources is not None:
            # Do not free or reuse storage if completion itself fails.
            self.bindings.wait(self.resources)
            self.bindings.close(self.resources)
            self.resources = None
        self.closed = True

    def _request_ledger(self):
        self.preflight()
        if self.ledger is None:
            self.ledger = RequestLedger(max_requests=self.runtime.max_batch_size,
                                        max_seq_len=self.runtime.max_seq_len)
        return self.ledger

    def _reset_request(self, key, owner):
        self.bindings.reset_request(self.resources, key, owner)
        self.bindings.wait(self.resources)

    def run_prefill(self, model, batch):
        with self._lock:
            ledger = self._request_ledger()
            requests = prefill_requests(batch, model.config, self.runtime, self.bindings.cache_groups)
            step = ledger.begin_prefill(requests)
            return self._run_transaction(step, batch.input_embeddings)

    def _run_transaction(self, step, embeddings):
        try:
            result = self._execute_step(step, embeddings)
            self.bindings.wait(self.resources)
            self.ledger.commit(step)
            return result
        except Exception:
            try:
                self.bindings.wait(self.resources)
            except Exception:
                self.failed = True
                self.ledger.poisoned = True
                # Keep pending state and buffers owned: completion is unknown.
                raise
            self.ledger.abort(step, self._reset_request)
            raise

    def _rank_plans(self):
        if self._plans is None:
            self._plans = tuple(self.plan.for_rank(rank) for rank in range(self.plan.placement.ep_size))
        return self._plans

    def lookup_embeddings(self, token_ids):
        # A complete TP vocabulary exists in each DP group; read it once on host.
        plans = self._rank_plans()[:self.plan.placement.tp_size]
        return lookup_token_embeddings([plan.weights for plan in plans], token_ids)

    def _check_state(self, state):
        if not isinstance(state, LayerState) or state.residual is None or state.pre_mix is None:
            raise ValueError("composite must return both residual and delayed pre_mix")
        if state.layout != self.bindings.output_layout:
            raise ValueError("composite output token layout differs from the next layer input")
        return state

    def _run_layers(self, step, embeddings):
        config = self.plan.weights.config
        if embeddings is None:
            embeddings = self.lookup_embeddings(torch.tensor(step.token_ids, dtype=torch.int64))
        if (not isinstance(embeddings, torch.Tensor) or embeddings.device.type != "cpu"
                or embeddings.dtype != torch.bfloat16
                or tuple(embeddings.shape) != (len(step.token_ids), config.hidden_size)):
            raise ValueError("input embeddings must be packed CPU BF16 [active_tokens, hidden_size]")
        plans = self._rank_plans()
        state = self.bindings.initialize(embeddings, step, self.resources)
        self.bindings.wait(self.resources)
        self._check_state(state)
        backbone = (self.bindings.prefill_backbone if step.phase == "prefill"
                    else self.bindings.decode_backbone)
        if backbone is not None:
            state = backbone(state, step, self.resources, plans)
            self.bindings.wait(self.resources)
            return self._check_state(state)
        for layer in self.plan.layers:
            # The adapter chooses bounded staging or residency; payloads stay packed.
            weights = self.bindings.prepare_weights(plans, layer, self.resources)
            self.bindings.wait(self.resources)
            state = self.bindings.entries[(step.phase, layer.mode)](
                layer, state, step, self.resources, weights)
            self.bindings.wait(self.resources)
            self._check_state(state)
        return state

    def _execute_step(self, step, embeddings):
        state = self._run_layers(step, embeddings)
        return self._finish_step(state, step)

    def _finish_step(self, state, step):
        logits = self.bindings.output(state, step, self.resources)
        self.bindings.wait(self.resources)
        shape = (len(step.requests), self.plan.weights.config.vocab_size)
        if (not isinstance(logits, torch.Tensor) or logits.device.type != "cpu"
                or tuple(logits.shape) != shape or logits.dtype not in (
                    torch.float32, torch.bfloat16, torch.float16)):
            raise ValueError("output composite must return CPU logits [requests, vocabulary] in request order")
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("output composite returned non-finite logits")
        # Shared workers may sample after the next dispatch reuses device output
        # scratch. Keep returned host logits independent of adapter-owned buffers.
        owned_logits = logits.detach().clone()
        if step.phase == "prefill":
            return PrefillResult(last_hidden=None, logits=owned_logits)
        return DecodeResult(hidden_states=None, logits=owned_logits)

    def run_decode(self, model, batch):
        with self._lock:
            ledger = self._request_ledger()
            requests = decode_requests(batch, model.config, self.runtime, self.bindings.cache_groups)
            step = ledger.begin_decode(requests)
            return self._run_transaction(step, batch.hidden_states)

    def release_finished_requests(self, request_ids):
        with self._lock:
            if self.ledger is not None:
                self.bindings.wait(self.resources)
                self.ledger.release(request_ids, self._reset_request)


# prefill_segment
_WEIGHTS = (
    "x_hc pre_mix hc_attn_fn hc_attn_scale hc_attn_base attn_norm_weight "
    "wq_a wq_a_scale q_norm_weight wq_b wq_b_scale wkv wkv_scale kv_norm_weight "
    "attn_sink wo_a wo_b wo_b_scale "
)
_WINDOW = "window_slots window_indices window_cache window_cache_scale "
_OUTPUTS = "attn_input attn_output next_pre_mix x_hc_out num_tokens attention_epoch"
C2A_FULL_ARGS = (_WEIGHTS + "freqs_cos freqs_sin " + _WINDOW +
    "compressed_cache compressed_cache_scale token_to_req_indices compressed_lens "
    "index_cache index_cache_scale index_block_table position_ids compressed_freqs_cos compressed_freqs_sin "
    "compressed_rope_positions compressor_wkv compressor_wgate query_start_loc state_block_table state_cache "
    "compressor_norm_weight compressed_slots index_wk index_norm_weight index_wq_b index_wq_b_scale "
    "index_weights_proj topk_indices " + _OUTPUTS).split()
C2A_REUSE_ARGS = (_WEIGHTS + "rope_cos rope_sin " + _WINDOW +
    "compressed_cache compressed_cache_scale compressed_indices " + _OUTPUTS).split()
C1A_ARGS = (_WEIGHTS + "rope_cos rope_sin " + _WINDOW +
    "compressed_cache compressed_cache_scale request_ids compressed_lens index_cache index_cache_scale "
    "index_block_table compressed_rope_cos compressed_rope_sin compressor_wkv compressor_norm_weight "
    "compressed_slots index_wk index_norm_weight index_wq_b index_wq_b_scale index_weights_proj "
    "topk_indices candidate_mask compressed_indices " + _OUTPUTS).split()
PREFILL_ARGUMENTS = {
    "swa": ATTENTION_ARGS, "c2a_full": C2A_FULL_ARGS, "c2a_reuse": C2A_REUSE_ARGS,
    "c1a_full": C1A_ARGS, "c1a_reindex": C1A_ARGS, "c1a_reuse": C1A_ARGS,
}


def bind_prefill_producers(plans, arguments):
    """Bind one ordered forward chain to its actual producer allocations.

    Like V4's worker-owned cache arguments, handles stay resident for the whole
    dispatch. V4.1 additionally shares compressed KV/index pools and physical
    Top-K rows according to LayerPlan. This performs no tensor computation.
    Every producer must execute earlier in this chain, so a previous request's
    transient Top-K/candidate rows cannot satisfy a missing dependency.

    C1A's common ABI retains unused Full weights in Reindex/Reuse. Bind those
    slots to their real producer weights; never allocate placeholder weights.
    Keep unused index slots in separate allocations. The common C1A ABI marks
    them writable even when a selected mode does not use them; aliasing those
    arguments makes the runtime reject overlapping write ranges (lib fbe92bfc).
    """
    plans = tuple(plans)
    if not plans or any(not isinstance(p, LayerPlan) for p in plans):
        raise ValueError("prefill chain requires LayerPlan entries")
    if any(b.layer_id != a.layer_id + 1 for a, b in zip(plans, plans[1:])):
        raise ValueError("prefill chain must execute consecutive layers in order")
    result, seen = {}, {}
    for plan in plans:
        if plan.mode not in PREFILL_ARGUMENTS:
            raise ValueError("unsupported prefill mode")
        current = dict(arguments[plan.layer_id])
        if plan.mode != "swa":
            family, mode = plan.mode.split("_")

            def producer(source, kind):
                if source == plan.layer_id:
                    if mode != "full" and not (kind == "index" and mode == "reindex"):
                        raise ValueError(f"{plan.mode} cannot produce its own {kind}")
                    return current
                if source not in seen or not seen[source].mode.startswith(family + "_"):
                    raise ValueError(f"missing preceding {kind} producer for layer {plan.layer_id}")
                source_mode = seen[source].mode.split("_")[1]
                if source_mode != "full" and not (kind == "index" and source_mode == "reindex"):
                    raise ValueError(f"invalid {kind} producer mode")
                return result[source]

            if mode == "full" and (plan.kv_source != plan.layer_id or plan.index_source != plan.layer_id):
                raise ValueError("Full must own its KV and index outputs")
            if mode == "reindex" and (family != "c1a" or plan.index_source != plan.layer_id):
                raise ValueError("Reindex must own its C1A index output")
            kv = producer(plan.kv_source, "KV")
            index = producer(plan.index_source, "index")
            if plan.index_source != plan.layer_id and seen[plan.index_source].kv_source != plan.kv_source:
                raise ValueError("index selection must address the same KV producer")
            for name in ("compressed_cache", "compressed_cache_scale"):
                current[name] = kv[name]
            if family == "c1a":
                candidate = producer(plan.candidate_source, "candidate")
                if plan.candidate_source != plan.kv_source:
                    raise ValueError("C1A candidates must use the same KV producer")
                for name in ("index_cache", "index_cache_scale", "compressor_wkv", "compressor_norm_weight",
                             "index_wk", "index_norm_weight"):
                    current[name] = kv[name]
                current["candidate_mask"] = candidate["candidate_mask"]
                if mode == "reuse":
                    for name in ("index_wq_b", "index_wq_b_scale", "index_weights_proj"):
                        current[name] = index[name]
                    current["compressed_indices"] = index["topk_indices"]
                if current["compressed_indices"] is current["topk_indices"]:
                    raise ValueError("C1A index ABI arguments require separate allocations")
            elif mode == "reuse":
                current["compressed_indices"] = index["topk_indices"]
        elif any(source is not None for source in (plan.kv_source, plan.index_source, plan.candidate_source)):
            raise ValueError("SWA must not declare compressed producers")
        result[plan.layer_id], seen[plan.layer_id] = current, plan
    return result


def compile_prefill_segments(compiler, lib_root, topology, modes):
    """Compile existing composite entries, sharing one packed-FP4 MoE program.

    C1A must use prefill_c1a_sp, not the older replicated-residual wrappers.
    No model operators are composed by this serving module.
    """
    import pypto.language as pl

    modes = tuple(dict.fromkeys(modes))
    if not modes or any(mode not in PREFILL_ARGUMENTS for mode in modes):
        raise ValueError("unsupported or empty prefill mode selection")
    swa, moe = load_segment_modules(lib_root, topology)
    attention = {}
    for mode in modes:
        if mode == "swa":
            entry = swa.make_hc_program(topology.capacity, topology.world, epochs=1)
        elif mode.startswith("c2a_"):
            module = importlib.import_module("models.deepseek_v4_1_flash.prefill_" + mode)
            entry = module.make_hc_program(topology.capacity, topology.world, epochs=1)
        else:
            module = importlib.import_module("models.deepseek_v4_1_flash.prefill_c1a_sp")
            entry = module.make_program(mode.removeprefix("c1a_"), topology.world, epochs=1)
        if tuple(entry.param_names) != tuple(PREFILL_ARGUMENTS[mode]):
            raise ValueError(f"{mode} prefill parameter order disagrees with the selected lib")
        attention[mode] = compiler.compile("v41_prefill_" + mode, entry, attention_epoch=pl.RUNTIME)
    if tuple(moe.l3_moe.param_names) != tuple(MOE_ARGS):
        raise ValueError("prefill MoE parameter order disagrees with the selected lib")
    ffn = compiler.compile("v41_moe_segment", moe.l3_moe, moe_epoch=pl.RUNTIME)
    return attention, ffn


class PrefillSegment(SwaSegment):
    """Carry device residual/pre_mix through a selected prefill mode and MoE.

    The worker must own all supplied programs via make_segment_worker(). Each
    Attention program advances its own epoch; the shared MoE advances on every
    layer. Cache ownership, cross-layer producer bindings and padded metadata
    remain caller responsibilities. Any runtime failure poisons the whole worker.
    """

    def __init__(self, worker, attention_programs, moe_program, topology,
                 attention_counts, moe_counts, run_config):
        if not attention_programs or any(mode not in PREFILL_ARGUMENTS for mode in attention_programs):
            raise ValueError("unsupported or empty prefill program set")
        self.attention_programs = dict(attention_programs)
        super().__init__(worker, (next(iter(attention_programs.values())), moe_program),
                         topology, attention_counts, moe_counts, run_config)

    def run_layer(self, state, attention, moe, *, group_counts, mode):
        if mode not in self.attention_programs:
            raise ValueError(f"prefill mode was not compiled: {mode}")
        return self._run_layer(
            state, attention, moe, group_counts=group_counts,
            attention_program=self.attention_programs[mode], argument_names=PREFILL_ARGUMENTS[mode],
            input_mix="incoming_pre_mix" if mode == "swa" else "pre_mix",
            output_residual="output" if mode == "swa" else "x_hc_out",
        )

    def run_chain(self, state, plans, attention, moe, *, group_counts):
        """Execute one request step through its producers and consumers in order.

        The resource owner must prepare every layer for the SAME packed step
        and retain all allocations until this synchronous call returns. This
        is still a bounded prefill segment, not a complete serving backend.
        """
        plans = tuple(plans)
        bound = bind_prefill_producers(plans, attention)
        self.topology.counts(group_counts)
        # Resolve the whole chain before any cache mutation. Runtime state,
        # active counts and per-program epochs are provided by run_layer.
        dynamic = {"x_hc", "pre_mix", "incoming_pre_mix", "num_tokens", "attention_epoch", "moe_epoch"}
        for plan in plans:
            if plan.mode not in self.attention_programs:
                raise ValueError(f"prefill mode was not compiled: {plan.mode}")
            for name in PREFILL_ARGUMENTS[plan.mode]:
                if name not in dynamic:
                    bound[plan.layer_id][name]
            for name in MOE_ARGS:
                if name not in dynamic:
                    moe[plan.layer_id][name]
            self._check_weights(moe[plan.layer_id])
        for plan in plans:
            state = self.run_layer(state, bound[plan.layer_id], moe[plan.layer_id],
                                   group_counts=group_counts, mode=plan.mode)
        return state


# segment_inputs
@dataclass(frozen=True)
class SegmentInputs:
    residual: torch.Tensor
    pre_mix: torch.Tensor
    # Source indices into the original packed request order; -1 means padding.
    row_indices: torch.Tensor
    # (world rank, local row) in the original request order, for output selection.
    request_last_rows: tuple[tuple[int, int], ...]
    group_counts: tuple[int, ...]


def prepare_segment_inputs(
    embeddings: torch.Tensor,
    step: ForwardStep,
    topology: SegmentTopology,
    *,
    max_prepare_bytes: int = 1 << 30,
) -> SegmentInputs:
    """Pack fresh embedding inputs without depending on request batch order.

    Each DP group contains contiguous local-capacity TP slabs, including empty
    slabs. Padding is zero in both residual and pre_mix. row_indices supplies the
    same mapping for positions and other token metadata. This does not lower
    cache pages or initialize a production all-mode model adapter.
    """
    if type(max_prepare_bytes) is not int or max_prepare_bytes <= 0:
        raise ValueError("max_prepare_bytes must be a positive integer")
    if not isinstance(step, ForwardStep) or not isinstance(topology, SegmentTopology):
        raise ValueError("segment input preparation requires a ForwardStep and SegmentTopology")
    if (not isinstance(embeddings, torch.Tensor) or embeddings.layout != torch.strided
            or embeddings.device.type != "cpu" or embeddings.dtype != torch.bfloat16
            or embeddings.ndim != 2 or embeddings.shape[1] <= 0):
        raise ValueError("embeddings must be packed CPU BF16 [active_tokens, hidden_size]")
    counts = [0] * topology.dp
    source_ranges, last_rows = [], []
    offset = 0
    request_ids = set()
    for request in step.requests:
        partition, count = request.partition, len(request.token_ids)
        if type(partition) is not int or not 0 <= partition < topology.dp or count <= 0:
            raise ValueError("requests require valid DP partitions and nonempty token slices")
        if request.request_id in request_ids:
            raise ValueError("duplicate request in forward step")
        request_ids.add(request.request_id)
        start = counts[partition]
        if start + count > topology.capacity:
            raise ValueError("DP token count exceeds segment capacity; split the step first")
        if step.phase == "decode" and count != 1:
            raise ValueError("decode requires one fresh token per request")
        source_ranges.append((partition, start, offset, count))
        owner, row = divmod(start + count - 1, topology.local_capacity)
        last_rows.append((partition * topology.tp + owner, row))
        counts[partition] += count
        offset += count
    if step.phase not in ("prefill", "decode") or embeddings.shape[0] != offset:
        raise ValueError("embedding rows must match a prefill/decode forward step")
    hidden = embeddings.shape[1]
    # Include output storage and the largest temporary expanded FP32 row block.
    allocated_rows = topology.world * topology.local_capacity
    estimate = allocated_rows * (4 * hidden * 4 + 4 * 4 + 8) + offset * hidden * 4
    if estimate > max_prepare_bytes:
        raise ValueError(f"segment preparation requires estimated {estimate} bytes; budget={max_prepare_bytes}")
    if not bool(torch.isfinite(embeddings).all()):
        raise ValueError("embeddings must be finite")
    residual = torch.zeros(topology.dp, topology.capacity, 4, hidden, dtype=torch.float32)
    pre_mix = torch.zeros(topology.dp, topology.capacity, 4, dtype=torch.float32)
    row_indices = torch.full((topology.dp, topology.capacity), -1, dtype=torch.int64)
    for partition, start, source, count in source_ranges:
        residual[partition, start:start + count].copy_(embeddings[source:source + count, None])
        pre_mix[partition, start:start + count, 0] = 1
        row_indices[partition, start:start + count] = torch.arange(source, source + count)
    return SegmentInputs(
        residual.reshape(topology.world, topology.local_capacity, 4, hidden),
        pre_mix.reshape(topology.world, topology.local_capacity, 4),
        row_indices.reshape(topology.world, topology.local_capacity),
        tuple(last_rows), tuple(counts),
    )


# swa_metadata
@dataclass(frozen=True)
class SwaWindowMetadata:
    positions: torch.Tensor
    window_slots: torch.Tensor
    window_indices: torch.Tensor
    group_counts: tuple[int, ...]


def gather_swa_rope_rows(
    metadata: SwaWindowMetadata,
    tables: tuple[torch.Tensor, torch.Tensor],
    *,
    max_prepare_bytes: int = 16 << 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather lib-built FP32 SWA tables in the same order as cache metadata.

    Like V4, the executor builds tables with the selected lib's host helper.
    This boundary only selects rows; it does not implement model-side RoPE.
    Padding receives identity rotation and does not index the table.
    """
    if not isinstance(metadata, SwaWindowMetadata) or len(tables) != 2:
        raise ValueError("SWA RoPE requires metadata and a cosine/sine table pair")
    positions, slots = metadata.positions, metadata.window_slots
    if (positions.device.type != "cpu" or positions.dtype != torch.int64 or positions.ndim != 2
            or slots.device.type != "cpu" or slots.dtype != torch.int64 or slots.shape != positions.shape):
        raise ValueError("SWA positions and slots must be matching CPU INT64 matrices")
    cos, sin = tables
    if any(not isinstance(t, torch.Tensor) or t.device.type != "cpu" or t.dtype != torch.float32
           or t.layout != torch.strided or t.ndim != 2 for t in tables):
        raise ValueError("SWA RoPE tables must be CPU FP32 matrices")
    if cos.shape != sin.shape or min(cos.shape) <= 0:
        raise ValueError("SWA cosine/sine tables must have matching nonempty shapes")
    estimate = positions.numel() * (cos.shape[1] * 4 * 4 + 32)
    if type(max_prepare_bytes) is not int or max_prepare_bytes <= 0 or estimate > max_prepare_bytes:
        raise ValueError("SWA RoPE preparation exceeds its allocation budget")
    active = slots >= 0
    selected = positions[active]
    if bool(((selected < 0) | (selected >= cos.shape[0])).any()):
        raise ValueError("active SWA position is outside the RoPE table")
    selected_cos, selected_sin = cos[selected], sin[selected]
    if not bool(torch.isfinite(selected_cos).all() and torch.isfinite(selected_sin).all()):
        raise ValueError("selected SWA RoPE rows must be finite")
    shape = (*positions.shape, cos.shape[1])
    rows_cos, rows_sin = torch.ones(shape, dtype=torch.float32), torch.zeros(shape, dtype=torch.float32)
    rows_cos[active], rows_sin[active] = selected_cos, selected_sin
    return rows_cos, rows_sin


def prepare_swa_window_metadata(
    step: ForwardStep,
    topology: SegmentTopology,
    *,
    cache_pages: int,
    window_group: str = "window",
    max_prepare_bytes: int = 32 << 20,
) -> SwaWindowMetadata:
    """Map each active row to a unique cache write and at most 128 causal reads.

    Page IDs address a partition-private cache of [pages, 128, 1, 512] payloads.
    Full-history tables preserve old rows while the whole new chunk is published;
    a 128-slot modulo ring would overwrite keys needed by early chunk queries.
    TP peers receive the same group-global metadata. No cache contents, RoPE,
    request ownership or completion state are changed by this host preparation.
    """
    if not isinstance(step, ForwardStep) or not isinstance(topology, SegmentTopology):
        raise ValueError("SWA metadata requires a ForwardStep and SegmentTopology")
    if type(cache_pages) is not int or not 0 < cache_pages <= (2**31 - 1) // 128:
        raise ValueError("cache_pages must fit the INT32 window-index ABI")
    if type(max_prepare_bytes) is not int or max_prepare_bytes <= 0:
        raise ValueError("max_prepare_bytes must be a positive integer")
    if step.phase not in ("prefill", "decode"):
        raise ValueError("unsupported forward phase")
    counts, ranges, owners, requests = [0] * topology.dp, [], set(), set()
    for request in step.requests:
        group, count = request.partition, len(request.token_ids)
        if (type(group) is not int or not 0 <= group < topology.dp or count <= 0
                or type(request.start) is not int or request.start < 0):
            raise ValueError("request partition, start and token count must be valid")
        if request.request_id in requests:
            raise ValueError("duplicate request in forward step")
        requests.add(request.request_id)
        if step.phase == "decode" and count != 1:
            raise ValueError("decode requires one token per request")
        if counts[group] + count > topology.capacity:
            raise ValueError("DP token count exceeds SWA capacity")
        pages = request.pages.get(window_group)
        if not isinstance(pages, (tuple, list)) or len(pages) < (request.end + 127) // 128:
            raise ValueError("SWA requires a full-history physical page table through the chunk end")
        for page in pages:
            if type(page) is not int or not 0 <= page < cache_pages:
                raise ValueError("SWA physical page is outside its partition cache pool")
            key = (group, page)
            if key in owners:
                raise ValueError("SWA pages must be private; shared or repeated physical pages are unsupported")
            owners.add(key)
        ranges.append((group, counts[group], request.start, count, tuple(pages)))
        counts[group] += count
    rows = topology.dp * topology.capacity
    estimate = rows * (128 * 4 + 2 * 8) * (topology.tp + 1)
    if estimate > max_prepare_bytes:
        raise ValueError(f"SWA metadata requires estimated {estimate} bytes; budget={max_prepare_bytes}")
    positions = torch.zeros(topology.dp, topology.capacity, dtype=torch.int64)
    slots = torch.full_like(positions, -1)
    indices = torch.full((topology.dp, topology.capacity, 128), -1, dtype=torch.int32)
    for group, row, start, count, pages in ranges:
        for local in range(count):
            position = start + local
            positions[group, row + local] = position
            slots[group, row + local] = pages[position // 128] * 128 + position % 128
            visible = range(max(0, position - 127), position + 1)
            addresses = [pages[p // 128] * 128 + p % 128 for p in visible]
            indices[group, row + local, :len(addresses)] = torch.tensor(addresses, dtype=torch.int32)
    return SwaWindowMetadata(
        positions.repeat_interleave(topology.tp, dim=0),
        slots.repeat_interleave(topology.tp, dim=0),
        indices.repeat_interleave(topology.tp, dim=0), tuple(counts),
    )


# final_output
@dataclass(frozen=True)
class FinalOutputRows:
    indices: torch.Tensor
    locations: tuple[tuple[int, int], ...]


def prepare_final_output_rows(inputs: SegmentInputs, indices: torch.Tensor, *,
                              local_capacity: int) -> FinalOutputRows:
    """Stage final-token rows into a pre-fork shared buffer in request order."""
    if not isinstance(inputs, SegmentInputs) or type(local_capacity) is not int or local_capacity <= 0:
        raise ValueError("final output requires valid packed inputs and rank capacities")
    if (not isinstance(indices, torch.Tensor) or indices.device.type != "cpu"
            or indices.dtype != torch.int32 or indices.ndim != 2
            or min(indices.shape) <= 0 or not indices.is_contiguous() or not indices.is_shared()):
        raise ValueError("LM-head row indices must be a pre-fork shared CPU INT32 matrix")
    world, max_logit_rows = indices.shape
    if tuple(inputs.row_indices.shape) != (world, local_capacity):
        raise ValueError("final output row map differs from the layer-state layout")
    counts = [0] * world
    locations = []
    selected = set()
    for rank, row in inputs.request_last_rows:
        if (type(rank) is not int or not 0 <= rank < world or type(row) is not int
                or not 0 <= row < local_capacity or inputs.row_indices[rank, row] < 0):
            raise ValueError("request final row is absent from the packed layer state")
        if (rank, row) in selected:
            raise ValueError("two requests cannot select the same final token row")
        selected.add((rank, row))
        slot = counts[rank]
        if slot >= max_logit_rows:
            raise ValueError("request final rows exceed the lib LM-head capacity on one rank")
        counts[rank] += 1
        locations.append((rank, slot))
    indices.fill_(-1)
    for (rank, row), (_, slot) in zip(inputs.request_last_rows, locations):
        indices[rank, slot] = row
    return FinalOutputRows(indices, tuple(locations))


def collect_final_logits(logits: torch.Tensor, rows: FinalOutputRows) -> torch.Tensor:
    """Return owned CPU logits in request order after output completion."""
    if (not isinstance(logits, torch.Tensor) or logits.device.type != "cpu"
            or logits.ndim != 3 or logits.shape[:2] != rows.indices.shape
            or logits.dtype != torch.float32):
        raise ValueError("lib output must be CPU FP32 [rank, logit_row, vocabulary]")
    if not rows.locations:
        raise ValueError("final output requires at least one request")
    result = torch.stack([logits[rank, slot] for rank, slot in rows.locations]).clone()
    if not bool(torch.isfinite(result).all()):
        raise ValueError("lib output contains non-finite request logits")
    return result


def make_final_output_program(lib_root, topology):
    load_segment_modules(lib_root, topology)
    module = importlib.import_module("models.deepseek_v4_1_flash.final_output")
    return module.make_final_output_program(topology.tp, topology.dp)


def compile_final_output(compiler, lib_root, topology):
    import pypto.language as pl

    entry = make_final_output_program(lib_root, topology)
    return compiler.compile("v41_final_output", entry, done_epoch=pl.RUNTIME), tuple(entry.param_names)


class FinalOutput:
    """Dispatch final logits while retaining one completion epoch per worker."""

    def __init__(self, worker, program, param_names, run_config):
        self.worker = worker
        self.program = program
        self.param_names = tuple(param_names)
        self.run_config = run_config
        self.epoch = 0
        self.failed = False
        required = {"x_hc", "pre_mix", "norm_weight", "head_weight", "logit_row_indices",
                    "normed", "logits", "sampled_ids", "done_epoch"}
        if set(self.param_names) != required or len(self.param_names) != len(required):
            raise ValueError("final output ABI differs from the lib composite")

    def run(self, state: LayerState, arguments):
        if self.failed:
            raise RuntimeError("output dispatch failed; close the worker before retrying")
        if state.layout != "tp_local_token":
            raise ValueError("final output requires TP-local residual and pre_mix")
        reserved = {"x_hc", "pre_mix", "done_epoch"}
        if reserved & arguments.keys():
            raise ValueError("final state and epoch must come from this dispatch")
        missing = set(self.param_names) - reserved - arguments.keys()
        if missing:
            raise ValueError("missing output ABI arguments: " + ", ".join(sorted(missing)))
        if self.epoch >= (2**31 - 1) // 16:
            raise OverflowError("output completion epoch exhausted; recreate worker")
        reset_windows = self.epoch == 128
        next_epoch = 1 if reset_windows else self.epoch + 1
        bound = dict(arguments, x_hc=state.residual, pre_mix=state.pre_mix,
                     done_epoch=ctypes.c_int32(next_epoch))
        if reset_windows:
            if not getattr(self.worker, "_persistent", False) or not hasattr(
                self.worker, "_reset_persistent_windows"
            ):
                raise RuntimeError("final output epoch reset requires a persistent PyPTO worker")
            previous_reset = self.worker._reset_persistent_windows
            self.worker._reset_persistent_windows = True
        try:
            self.worker.run(self.program.compiled, *(bound[name] for name in self.param_names),
                            config=self.run_config)
        except BaseException:
            self.failed = True
            raise
        finally:
            if reset_windows:
                self.worker._reset_persistent_windows = previous_reset
        self.epoch = next_epoch
        return arguments["logits"], arguments["sampled_ids"]


# input_resources
@dataclass(frozen=True)
class InputHostResources:
    topology: SegmentTopology
    residual: torch.Tensor
    pre_mix: torch.Tensor

    @property
    def inherited_host_tensors(self):
        return self.residual, self.pre_mix


def make_input_host_resources(topology: SegmentTopology, hidden_size: int) -> InputHostResources:
    """Allocate shared staging before the persistent DistributedWorker forks."""
    if not isinstance(topology, SegmentTopology) or type(hidden_size) is not int or hidden_size <= 0:
        raise ValueError("input staging requires a topology and positive hidden size")
    residual = torch.empty((topology.world, topology.local_capacity, 4, hidden_size),
                           dtype=torch.float32).share_memory_()
    pre_mix = torch.empty((topology.world, topology.local_capacity, 4),
                          dtype=torch.float32).share_memory_()
    return InputHostResources(topology, residual, pre_mix)


class InputStateRuntime:
    """Upload each packed step into fixed device handles for its phase capacity."""

    def __init__(self, worker, host: InputHostResources):
        if not isinstance(host, InputHostResources):
            raise ValueError("input state requires pre-fork host resources")
        for tensor in host.inherited_host_tensors:
            if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu"
                    or tensor.dtype != torch.float32
                    or not tensor.is_shared() or not tensor.is_contiguous()):
                raise ValueError("input staging must be contiguous shared CPU FP32")
        topology = host.topology
        if (host.residual.shape[:3] != (topology.world, topology.local_capacity, 4)
                or host.pre_mix.shape != host.residual.shape[:-1]):
            raise ValueError("input staging shapes disagree with the TP-local layout")
        self.worker, self.host = worker, host
        self.failed = False
        self._resident = []
        try:
            self.residual = alloc_device_buffer(worker, host.residual.shape, torch.float32, stacked=True)
            self._resident.append(self.residual)
            self.pre_mix = alloc_device_buffer(worker, host.pre_mix.shape, torch.float32, stacked=True)
            self._resident.append(self.pre_mix)
        except BaseException:
            self.close()
            raise

    def stage(self, embeddings: torch.Tensor, step: ForwardStep) -> tuple[LayerState, SegmentInputs]:
        if self.failed:
            raise RuntimeError("input upload failed; recreate the worker before reuse")
        packed = prepare_segment_inputs(embeddings, step, self.host.topology)
        self.host.residual.copy_(packed.residual)
        self.host.pre_mix.copy_(packed.pre_mix)
        try:
            for source, destination in ((self.host.residual, self.residual),
                                        (self.host.pre_mix, self.pre_mix)):
                for rank, shard in enumerate(destination.shards):
                    row = source[rank]
                    self.worker.copy_to(shard.data_ptr, row.data_ptr(),
                                        row.numel() * row.element_size(), worker_id=rank)
        except BaseException:
            self.failed = True
            raise
        inputs = SegmentInputs(self.host.residual, self.host.pre_mix,
                               packed.row_indices, packed.request_last_rows, packed.group_counts)
        return LayerState(self.residual, self.pre_mix, "tp_local_token"), inputs

    def close(self):
        for tensor in reversed(self._resident):
            self.worker.free_stacked_tensor(tensor)
        self._resident.clear()


# prefill_layer_resources
class PrefillLayerResources:
    """Own only transient device buffers; weight and cache handles are borrowed."""

    def __init__(self, worker, topology, plans, weights, cache, metadata, *,
                 kv_sources, index_sources, c2a_sources, max_seq_len, index_topk=512):
        self.worker = worker
        self.topology = topology
        self.plans = tuple(plans)
        self.metadata = metadata
        if (not self.plans or any(plan.mode not in PREFILL_ARGUMENTS for plan in self.plans)
                or any(type(n) is not int or n <= 0 for n in (max_seq_len, index_topk))):
            raise ValueError("prefill binding requires ordered modes and positive model capacities")
        self._owned = []
        world, local, tokens = topology.world, topology.local_capacity, topology.capacity

        def allocate(shape, dtype):
            value = alloc_device_buffer(worker, shape, dtype, stacked=True)
            self._owned.append(value)
            return value

        try:
            # A layer's Attention and MoE outputs must differ from each other
            # and from the preceding MoE output. Three slots suffice for all layers.
            states = [(allocate((world, local, 4, 5120), torch.float32),
                       allocate((world, local, 4), torch.float32)) for _ in range(3)]
            attention_hidden = allocate((world, tokens, 5120), torch.bfloat16)
            attention_out = allocate((world, local, 5120), torch.bfloat16)
            moe_mixed = allocate((world, local, 5120), torch.bfloat16)
            topk = {plan.layer_id: allocate((world, tokens, index_topk), torch.int32)
                    for plan in self.plans if plan.index_source == plan.layer_id}
            unused_topk = allocate((world, tokens, index_topk), torch.int32)
            unused_indices = allocate((world, tokens, index_topk), torch.int32)
            candidate = {plan.layer_id: allocate((world, tokens, max_seq_len), torch.uint8)
                         for plan in self.plans if plan.mode == "c1a_full"}
            attention, moe = {}, {}
            for plan in self.plans:
                layer = plan.layer_id
                a, m = weights.prefill_weights(
                    plan, kv_sources=kv_sources, index_sources=index_sources,
                    c2a_sources=c2a_sources)
                a = dict(a, **cache.prefill_views(
                    plan, kv_sources=kv_sources, index_sources=index_sources,
                    c2a_sources=c2a_sources))
                a.update(metadata.arguments(plan.mode))
                attn_residual, attn_mix = states[(2 * layer) % 3]
                moe_residual, moe_mix = states[(2 * layer + 1) % 3]
                a.update(attn_input=attention_hidden, attn_output=attention_out,
                         hidden=attention_hidden, attn_out=attention_out,
                         output=attn_residual, x_hc_out=attn_residual,
                         next_pre_mix=attn_mix)
                if plan.mode.startswith("c1a_"):
                    a.update(topk_indices=topk.get(layer, unused_topk),
                             compressed_indices=unused_indices,
                             candidate_mask=candidate[plan.candidate_source])
                elif plan.mode == "c2a_full":
                    a["topk_indices"] = topk[layer]
                elif plan.mode == "c2a_reuse":
                    a["compressed_indices"] = unused_indices
                m = dict(m, next_pre_mix=moe_mix, x_next=moe_residual, x_mixed=moe_mixed)
                attention[layer], moe[layer] = a, m
            bound = bind_prefill_producers(self.plans, attention)
            dynamic = {"x_hc", "pre_mix", "incoming_pre_mix", "num_tokens",
                       "attention_epoch", "moe_epoch"}
            for plan in self.plans:
                missing_a = set(PREFILL_ARGUMENTS[plan.mode]) - dynamic - bound[plan.layer_id].keys()
                missing_m = set(MOE_ARGS) - dynamic - moe[plan.layer_id].keys()
                if missing_a or missing_m:
                    raise ValueError(f"layer {plan.layer_id} has unbound prefill ABI fields: "
                                     f"{sorted(missing_a | missing_m)}")
            self.attention, self.moe = bound, moe
        except BaseException:
            self.close()
            raise

    def close(self):
        for value in reversed(self._owned):
            self.worker.free_stacked_tensor(value)
        self._owned.clear()


# prefill_metadata_resources
class PrefillMetadataResources:
    """Keep C2A and C1A page/length contracts separate across one layer chain."""

    def __init__(self, topology, *, max_seq_len, max_requests, window_pages,
                 c2a_pages, c1a_pages, rope_tables, compressed_rope_tables):
        if any(type(value) is not int or value <= 0 for value in
               (max_seq_len, max_requests, window_pages, c2a_pages, c1a_pages)):
            raise ValueError("prefill metadata capacities must be positive integers")
        for pair in (rope_tables, compressed_rope_tables):
            if (len(pair) != 2 or any(not isinstance(value, torch.Tensor) or
                value.device.type != "cpu" or value.dtype != torch.float32 or
                value.ndim != 2 or value.shape[0] < max_seq_len for value in pair)
                    or pair[0].shape != pair[1].shape):
                raise ValueError("prefill requires lib FP32 RoPE tables covering the model context")
        self.topology = topology
        self.max_requests = max_requests
        self.window_pages = window_pages
        self.compressed_pages = {"c2a": c2a_pages, "c1a": c1a_pages}
        self.fields = {}

        def shared(name, shape, dtype, fill=0):
            value = torch.empty(shape, dtype=dtype).share_memory_()
            value.fill_(fill)
            self.fields[name] = value
            return value

        world, tokens, width = topology.world, topology.capacity, rope_tables[0].shape[1]
        shared("attention_counts", (world, 1), torch.int32)
        shared("moe_counts", (world,), torch.int32)
        shared("window_slots", (world, tokens), torch.int64, -1)
        shared("window_indices", (world, tokens, 128), torch.int32, -1)
        for name in ("rope_cos", "rope_sin", "compressed_rope_cos", "compressed_rope_sin"):
            shared(name, (world, tokens, width), torch.float32)
        for name in ("c2a_compressed_rope_cos", "c2a_compressed_rope_sin"):
            shared(name, (world, tokens, width), torch.float32)
        for prefix, pages in self.compressed_pages.items():
            for name, shape, dtype, fill in (
                ("position_ids", (world, tokens), torch.int32, 0),
                ("request_ids", (world, tokens), torch.int32, -1),
                ("query_start_loc", (world, max_requests + 1), torch.int32, 0),
                ("compressed_lens", (world, tokens), torch.int32, 0),
                ("compressed_slots", (world, tokens), torch.int64, -1),
                ("compressed_rope_positions", (world, tokens), torch.int32, -1),
                ("index_block_table", (world, max_requests, pages), torch.int32, -1),
                ("state_block_table", (world, max_requests, 1), torch.int32, -1),
            ):
                shared(prefix + "_" + name, shape, dtype, fill)
        self.tables = {}
        for prefix, pair in (("", rope_tables), ("compressed_", compressed_rope_tables)):
            for suffix, value in zip(("cos", "sin"), pair):
                target = shared(prefix + "freqs_" + suffix,
                                (world, max_seq_len, width), torch.float32)
                target.copy_(value[:max_seq_len].unsqueeze(0).expand_as(target))
            self.tables[prefix] = pair

    @property
    def inherited_host_tensors(self):
        return tuple(self.fields.values())

    def _compressed_rope_rows(self, positions):
        cos, sin = self.tables["compressed_"]
        active = positions >= 0
        if bool((positions[active] >= cos.shape[0]).any()):
            raise ValueError("compressed RoPE position exceeds the lib table")
        result_cos = torch.ones((*positions.shape, cos.shape[1]), dtype=torch.float32)
        result_sin = torch.zeros_like(result_cos)
        result_cos[active] = cos[positions[active].long()]
        result_sin[active] = sin[positions[active].long()]
        return result_cos, result_sin

    def stage(self, step):
        swa = prepare_swa_window_metadata(
            step, self.topology, cache_pages=self.window_pages)
        self.fields["window_slots"].copy_(swa.window_slots)
        self.fields["window_indices"].copy_(swa.window_indices)
        cos, sin = gather_swa_rope_rows(
            swa, self.tables[""], max_prepare_bytes=64 << 20)
        self.fields["rope_cos"].copy_(cos)
        self.fields["rope_sin"].copy_(sin)
        attention, moe = self.topology.counts(swa.group_counts)
        self.fields["attention_counts"][:, 0].copy_(torch.tensor(attention, dtype=torch.int32))
        self.fields["moe_counts"].copy_(torch.tensor(moe, dtype=torch.int32))
        for prefix, ratio in (("c2a", 2), ("c1a", 1)):
            metadata = prepare_compressed_metadata(
                step, self.topology, ratio=ratio, compressed_group=prefix,
                cache_pages=self.compressed_pages[prefix], max_requests=self.max_requests,
                state_blocks=self.max_requests,
            )
            if metadata.group_counts != swa.group_counts:
                raise ValueError("SWA and compressed metadata token layouts disagree")
            for name in ("position_ids", "request_ids", "query_start_loc", "compressed_lens",
                         "compressed_slots", "compressed_rope_positions", "state_block_table"):
                self.fields[prefix + "_" + name].copy_(getattr(metadata, name))
            table = self.fields[prefix + "_index_block_table"]
            source = metadata.index_block_table
            if source.shape[2] > table.shape[2]:
                raise ValueError("compressed page table exceeds its pre-fork capacity")
            table.fill_(-1)
            table[:, :, :source.shape[2]].copy_(source)
            rows = self._compressed_rope_rows(metadata.compressed_rope_positions)
            if prefix == "c2a":
                self.fields["c2a_compressed_rope_cos"].copy_(rows[0])
                self.fields["c2a_compressed_rope_sin"].copy_(rows[1])
            else:
                self.fields["compressed_rope_cos"].copy_(rows[0])
                self.fields["compressed_rope_sin"].copy_(rows[1])
        return swa.group_counts

    def arguments(self, mode):
        common = {name: self.fields[name] for name in
                  ("window_slots", "window_indices", "rope_cos", "rope_sin")}
        if mode == "swa":
            return common
        if mode not in ("c2a_full", "c2a_reuse", "c1a_full", "c1a_reindex", "c1a_reuse"):
            raise ValueError("unknown V4.1 prefill metadata mode")
        prefix = mode[:3]
        names = ("request_ids", "compressed_lens", "compressed_slots", "index_block_table")
        result = dict(common, **{name: self.fields[prefix + "_" + name] for name in names})
        if mode == "c2a_full":
            result.update({
                "freqs_cos": self.fields["freqs_cos"],
                "freqs_sin": self.fields["freqs_sin"],
                "compressed_freqs_cos": self.fields["compressed_freqs_cos"],
                "compressed_freqs_sin": self.fields["compressed_freqs_sin"],
                "token_to_req_indices": result["request_ids"],
                "position_ids": self.fields["c2a_position_ids"],
                "compressed_rope_positions": self.fields["c2a_compressed_rope_positions"],
                "compressed_rope_cos": self.fields["c2a_compressed_rope_cos"],
                "compressed_rope_sin": self.fields["c2a_compressed_rope_sin"],
                "query_start_loc": self.fields["c2a_query_start_loc"],
                "state_block_table": self.fields["c2a_state_block_table"],
            })
        elif mode.startswith("c1a_"):
            result["compressed_rope_cos"] = self.fields["compressed_rope_cos"]
            result["compressed_rope_sin"] = self.fields["compressed_rope_sin"]
        return result


# final_output_resources
@dataclass(frozen=True)
class FinalOutputHostResources:
    norm_weight: torch.Tensor
    head_weight: torch.Tensor
    logit_row_indices: torch.Tensor
    logits: torch.Tensor
    sampled_ids: torch.Tensor

    @property
    def inherited_host_tensors(self):
        return self.norm_weight, self.head_weight

    @property
    def all_host_tensors(self):
        return (self.norm_weight, self.head_weight, self.logit_row_indices,
                self.logits, self.sampled_ids)


def load_final_output_host_resources(model_dir, topology: SegmentTopology, *,
                                     max_logit_rows: int = 16, chunk_rows: int = 64,
                                     max_host_bytes: int = 4 << 30) -> FinalOutputHostResources:
    """Load one TP vocabulary shard at a time before the worker forks."""
    if not isinstance(topology, SegmentTopology) or (topology.tp, topology.dp) != (4, 2):
        raise ValueError("V4.1 final output requires TP4/DP2 placement")
    if any(type(value) is not int or value <= 0 for value in
           (max_logit_rows, chunk_rows, max_host_bytes)):
        raise ValueError("final output capacities and memory budget must be positive")
    loader = V41WeightLoader(model_dir, tp_size=topology.tp, tp_rank=0,
                             ep_size=topology.world, ep_rank=0)
    vocab, hidden = loader.config.vocab_size, loader.config.hidden_size
    if vocab % topology.tp:
        raise ValueError("LM-head vocabulary must divide TP size")
    shard_rows = vocab // topology.tp
    sampled_pad = 8
    required_bytes = (topology.world * shard_rows * hidden * 2
                      + topology.world * hidden * 2
                      + topology.world * max_logit_rows * vocab * 4
                      + topology.world * max_logit_rows * (1 + sampled_pad) * 4)
    if required_bytes > max_host_bytes:
        raise ValueError(f"final output host buffers require {required_bytes} bytes")

    norm = torch.empty((topology.world, hidden), dtype=torch.bfloat16).share_memory_()
    head = torch.empty((topology.world, shard_rows, hidden), dtype=torch.bfloat16).share_memory_()
    indices = torch.full((topology.world, max_logit_rows), -1, dtype=torch.int32).share_memory_()
    logits = torch.zeros((topology.world, max_logit_rows, vocab), dtype=torch.float32).share_memory_()
    sampled = torch.full((topology.world, max_logit_rows, sampled_pad), -1,
                         dtype=torch.int32).share_memory_()

    norm_value = loader.load("norm.weight").weight
    if norm_value.dtype != torch.bfloat16 or tuple(norm_value.shape) != (hidden,):
        raise ValueError("checkpoint final Norm weight disagrees with the lib output ABI")
    norm.copy_(norm_value.expand_as(norm))
    for tp_rank in range(topology.tp):
        rank_loader = (loader if tp_rank == 0 else V41WeightLoader(
            model_dir, tp_size=topology.tp, tp_rank=tp_rank,
            ep_size=topology.world, ep_rank=tp_rank))
        for start in range(0, shard_rows, chunk_rows):
            stop = min(start + chunk_rows, shard_rows)
            value = rank_loader.load_rows("head.weight", start, stop).weight
            if value.dtype != torch.bfloat16 or tuple(value.shape) != (stop - start, hidden):
                raise ValueError("checkpoint LM-head shard disagrees with the lib output ABI")
            head[tp_rank, start:stop].copy_(value)
        head[tp_rank + topology.tp].copy_(head[tp_rank])
    return FinalOutputHostResources(norm, head, indices, logits, sampled)


class FinalOutputRuntime:
    """Keep output weights resident and stage only request rows per dispatch.

    The host resources must exist before the shared DistributedWorker forks.
    The caller owns and closes the worker after this runtime releases its tensors.
    """

    def __init__(self, worker, program, param_names, run_config, topology: SegmentTopology,
                 host: FinalOutputHostResources, *, prefill_local_capacity: int | None = None):
        if not isinstance(host, FinalOutputHostResources) or not isinstance(topology, SegmentTopology):
            raise ValueError("final output requires host resources and a segment topology")
        world, local = topology.world, topology.local_capacity
        if any(not tensor.is_shared() for tensor in host.all_host_tensors):
            raise ValueError("final output host buffers must be shared before worker creation")
        if (host.norm_weight.dtype != torch.bfloat16 or host.norm_weight.shape != (world, host.head_weight.shape[-1])
                or host.head_weight.dtype != torch.bfloat16 or host.head_weight.shape[0] != world
                or host.logit_row_indices.dtype != torch.int32 or host.logit_row_indices.shape != (world, 16)
                or host.logits.dtype != torch.float32
                or host.logits.shape != (world, 16, host.head_weight.shape[1] * topology.tp)
                or host.sampled_ids.dtype != torch.int32 or host.sampled_ids.shape != (world, 16, 8)):
            raise ValueError("final output host buffers disagree with the lib TP4/DP2 ABI")
        self.worker, self.host, self.topology = worker, host, topology
        if (prefill_local_capacity is not None and
                (type(prefill_local_capacity) is not int or not 0 < prefill_local_capacity <= 1024)):
            raise ValueError("prefill output capacity must fit the M0 boundary")
        self.dispatcher = FinalOutput(worker, program, param_names, run_config)
        self._resident = []
        try:
            self.norm_weight = worker.alloc_stacked_tensor(host.norm_weight)
            self._resident.append(self.norm_weight)
            self.head_weight = worker.alloc_stacked_tensor(host.head_weight)
            self._resident.append(self.head_weight)
            self.normed = alloc_device_buffer(
                worker, (world, local, host.head_weight.shape[-1]), torch.bfloat16, stacked=True,
            )
            self._resident.append(self.normed)
            self._normed_by_capacity = {local: self.normed}
            if prefill_local_capacity is not None and prefill_local_capacity != local:
                prefill_normed = alloc_device_buffer(
                    worker, (world, prefill_local_capacity, host.head_weight.shape[-1]),
                    torch.bfloat16, stacked=True)
                self._resident.append(prefill_normed)
                self._normed_by_capacity[prefill_local_capacity] = prefill_normed
        except BaseException:
            self.close()
            raise

    def run(self, state, inputs):
        local = inputs.row_indices.shape[1]
        normed = self._normed_by_capacity.get(local)
        if normed is None:
            raise ValueError("final output capacity was not preallocated before worker dispatch")
        rows = prepare_final_output_rows(
            inputs, self.host.logit_row_indices, local_capacity=local,
        )
        self.host.logits.zero_()
        self.host.sampled_ids.fill_(-1)
        self.dispatcher.run(state, {
            "norm_weight": self.norm_weight,
            "head_weight": self.head_weight,
            "logit_row_indices": self.host.logit_row_indices,
            "normed": normed,
            "logits": self.host.logits,
            "sampled_ids": self.host.sampled_ids,
        })
        return collect_final_logits(self.host.logits, rows)

    def close(self):
        for tensor in reversed(self._resident):
            self.worker.free_stacked_tensor(tensor)
        self._resident.clear()


# resident_resources
class V41ResidentResources:
    """Own resident buffers and their shared host mappings until worker close.

    This is the resource lifetime for a production binding, not an executable
    binding by itself: the caller still has to lower every prefill/decode ABI
    argument and dispatch the compiled layer programs.
    """

    def __init__(self, programs, run_config, *, weight_bank, weight_param_names,
                 cache_plan, state_reset, page_resets, input_host, final_host,
                 prefill_host, final_program, final_param_names, topology,
                 decode_input_host=None, decode_host=None, worker_factory=make_segment_worker):
        weight_bank.finish(weight_param_names)
        if decode_host is not None and (decode_input_host is None or
                decode_host.topology != decode_input_host.topology):
            raise ValueError("decode metadata and input must share one TP-local topology")
        inherited = (input_host.inherited_host_tensors
                     + (() if decode_input_host is None else decode_input_host.inherited_host_tensors)
                     + prefill_host.inherited_host_tensors
                     + (() if decode_host is None else decode_host.inherited_host_tensors)
                     + final_host.all_host_tensors
                     + (state_reset, *page_resets.values()))
        if any(not tensor.is_shared() for tensor in inherited):
            raise ValueError("all inherited V4.1 host buffers must be shared before worker creation")
        if not getattr(weight_bank, "prepacked", False) and any(
                not tensor.is_shared() for tensor in weight_bank.tensors.values()):
            raise ValueError("all inherited V4.1 host buffers must be shared before worker creation")
        if final_program not in programs:
            raise ValueError("final composite must be registered on the shared worker")

        self.worker = worker_factory(programs, run_config,
                                     tuple(weight_bank.tensors.values()) + inherited)
        self.host_resources = (weight_bank, input_host, decode_input_host, prefill_host, decode_host,
                               final_host, state_reset, page_resets)
        self.prefill_host = prefill_host
        self.decode_host = decode_host
        self._owned = []
        self._step = None
        self._inputs = None
        self._group_counts = None
        self._prefill = None
        self._decode_layers = None
        self.closed = False
        try:
            self.weights = self._own(ResidentDecodeWeights(
                self.worker, weight_bank, weight_param_names))
            self.input = self._own(InputStateRuntime(self.worker, input_host))
            self.decode_input = (None if decode_input_host is None else
                                 self._own(InputStateRuntime(self.worker, decode_input_host)))
            self.final = self._own(FinalOutputRuntime(
                self.worker, final_program, final_param_names, run_config, topology,
                final_host, prefill_local_capacity=prefill_host.topology.local_capacity))
            if callable(cache_plan):
                slots, plan_for_slots = cache_plan(self.worker)
                cache = allocate_cache_with_retry(
                    self.worker, slots, plan_for_slots, state_reset, page_resets)
            else:
                cache = DecodeCachePools(self.worker, cache_plan, state_reset, page_resets)
            self.cache = self._own(cache)
            if decode_host is not None:
                plan = cache.plan
                if decode_host.max_requests < plan.physical_request_slots:
                    raise ValueError("decode metadata must reserve the filler state slot")
                if decode_host.window_pages < plan.physical_group_blocks[plan.family_groups.window]:
                    raise ValueError("decode window metadata must cover the filler page")
                for family in ("c2a", "c1a"):
                    group = getattr(plan.family_groups, family)
                    if decode_host.compressed_pages[family] < plan.physical_group_blocks[group]:
                        raise ValueError(f"decode {family} metadata must cover the filler page")
        except BaseException:
            self.close()
            raise

    def _own(self, resource):
        self._owned.append(resource)
        return resource

    def begin_step(self, embeddings, step):
        if self.closed or self._step is not None:
            raise RuntimeError("V4.1 worker is closed or a step is already active")
        dispatch_step = step
        if step.phase == "decode":
            dispatch_step, embeddings, padding_partition = pad_single_decode(
                step, embeddings, self.cache.plan)
            if padding_partition is not None:
                self.cache.reset_padding(padding_partition)
        host = self.prefill_host if step.phase == "prefill" else self.decode_host
        counts = host.stage(dispatch_step) if host is not None else None
        input_runtime = self.decode_input if step.phase == "decode" else self.input
        if input_runtime is None:
            raise ValueError("decode requires its own TP-local input capacity")
        state, inputs = input_runtime.stage(embeddings, dispatch_step)
        if dispatch_step is not step:
            inputs = replace(inputs, request_last_rows=inputs.request_last_rows[:len(step.requests)])
        self._step, self._inputs = step, inputs
        self._group_counts = counts
        return state

    def prefill_arguments(self, mode, step):
        if self.closed or step is not self._step or step.phase != "prefill":
            raise ValueError("prefill arguments require the active prefill step")
        return self.prefill_host.arguments(mode), self._group_counts

    def bind_prefill(self, plans, *, kv_sources, index_sources, c2a_sources,
                     max_seq_len, index_topk=512):
        """Complete per-layer arguments using this worker's resident resources."""
        if self.closed or self._prefill is not None or self._step is not None:
            raise RuntimeError("prefill must be bound once before a request step")
        self._prefill = self._own(PrefillLayerResources(
            self.worker, self.prefill_host.topology, plans, self.weights, self.cache,
            self.prefill_host, kv_sources=kv_sources, index_sources=index_sources,
            c2a_sources=c2a_sources, max_seq_len=max_seq_len, index_topk=index_topk))
        return self._prefill

    def run_prefill(self, segment, state, step):
        if (self.closed or self._prefill is None or step is not self._step
                or step.phase != "prefill" or segment.worker is not self.worker):
            raise ValueError("prefill dispatch requires the active step and shared worker")
        return segment.run_chain(
            state, self._prefill.plans, self._prefill.attention, self._prefill.moe,
            group_counts=self._group_counts)

    def decode_backbone_metadata(self, step):
        """Reject the current lib ABI before binding unequal C2A/C1A fields."""
        if (self.closed or step is not self._step or step.phase != "decode"
                or self.decode_host is None):
            raise ValueError("decode metadata requires the active decode step")
        raise MissingCompositeInterface(
            "decode_fwd accepts one compressed_lens, compressed_slots, "
            "index_block_table and compressed_rope pair for both C2A and C1A; "
            "the two families have different values and page namespaces")

    def bind_decode_layers(self, plans, param_names, *, kv_sources, index_sources,
                           c2a_sources, index_topk=512):
        """Prepare the lib's complete per-layer Block on this worker."""
        if (self.closed or self._decode_layers is not None or self._step is not None
                or self.decode_host is None or self.decode_input is None):
            raise RuntimeError("decode Blocks require pre-fork metadata and one idle worker")
        self._decode_layers = self._own(DecodeLayerResources(
            self.worker, self.decode_host.topology, plans, self.weights, self.cache,
            self.decode_host, param_names, kv_sources=kv_sources,
            index_sources=index_sources, c2a_sources=c2a_sources,
            index_topk=index_topk))
        return self._decode_layers

    def run_decode_layers(self, segment, state, step):
        if (self.closed or self._decode_layers is None or step is not self._step
                or step.phase != "decode" or segment.worker is not self.worker):
            raise ValueError("decode Blocks require the active step and shared worker")
        for plan in self._decode_layers.plans:
            state = segment.run_layer(
                state, plan, self._decode_layers.arguments[plan.layer_id],
                group_counts=self._group_counts)
        return state

    def finish_step(self, state, step):
        if self.closed or step is not self._step:
            raise ValueError("final output must use the active request step")
        try:
            return self.final.run(state, self._inputs)
        finally:
            self._step = self._inputs = self._group_counts = None

    def reset_request(self, owner):
        self._step = self._inputs = self._group_counts = None
        self.cache.reset_request(owner)

    def close(self):
        if self.closed:
            return
        self._step = self._inputs = self._group_counts = None
        try:
            for resource in reversed(self._owned):
                resource.close()
        finally:
            self.worker.close()
            self.closed = True
            self._owned.clear()


# production_binding
_MODES = ("swa", "c2a_full", "c2a_reuse", "c1a_full", "c1a_reindex", "c1a_reuse")
_LOG = logging.getLogger(__name__)


def _lib_root():
    configured = os.environ.get("PYPTO_LIB_ROOT")
    root = Path(configured) if configured else Path.cwd() / "pypto-lib"
    root = root.resolve()
    if not (root / "models/deepseek_v4_1_flash/final_output.py").is_file():
        raise MissingCompositeInterface(
            f"V4.1 final-output composite is absent under {root}; set PYPTO_LIB_ROOT "
            "to a lib checkout containing the verified output entry")
    return root


def _revision(root):
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
    ).strip()


def _sources(layers):
    return (
        tuple(layer.layer_id for layer in layers if layer.kv_source == layer.layer_id),
        tuple(layer.layer_id for layer in layers if layer.index_source == layer.layer_id),
        tuple(layer.layer_id for layer in layers if layer.mode == "c2a_full"),
    )


def _check_runtime_binaries(platform):
    from simpler_setup.runtime_builder import RuntimeBuilder

    builder = RuntimeBuilder(platform=platform)
    for runtime in ("host_build_graph", "tensormap_and_ringbuffer"):
        builder.get_binaries(runtime)


def make_production_bindings(model_dir, max_seq_len):
    model_dir = Path(model_dir).resolve()
    raw = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    layers = plan_layers(raw)
    groups = build_cache_groups(layers, max_seq_len)
    root = _lib_root()
    revision = _revision(root)

    def allocate(plan, device_ids, runtime, options):
        from pypto_serving.model.common.compiler.compiler import KernelCompiler
        from pypto_serving.model.common.executor.utils import build_pypto_run_config

        if Path(plan.weights.model_dir).resolve() != model_dir or tuple(plan.layers) != layers:
            raise ValueError("V4.1 binding and execution plan refer to different checkpoints")
        if runtime.max_seq_len != max_seq_len or runtime.kv_cache_groups != groups:
            raise ValueError("V4.1 scheduler and resident cache contracts differ")
        if len(device_ids) != 8 or options.platform != "a5":
            raise ValueError("V4.1 production binding requires eight A5 devices")
        _check_runtime_binaries(options.platform)
        topology = SegmentTopology.for_decode()
        run_config = replace(
            build_pypto_run_config(
                platform=options.platform,
                device_ids=device_ids,
                pypto_build_dir=options.pypto_build_dir,
            ),
            ring_heap=1024 * 1024 * 1024,
        )
        compiler = KernelCompiler(run_config=run_config)
        attention, moe = compile_prefill_segments(compiler, root, topology, _MODES)
        decode, decode_names = compile_decode_layer(compiler, root, topology)
        final, final_names = compile_final_output(compiler, root, topology)
        abi = importlib.import_module("models.deepseek_v4_1_flash.decode_fwd")
        kv_sources, index_sources, c2a_sources = _sources(layers)
        layout = DecodeWeightLayout.from_lib(abi)
        if (layout.layers, layout.kv_sources, layout.index_sources,
                layout.c2a_sources, layout.world) != (
                    len(layers), len(kv_sources), len(index_sources), len(c2a_sources), 8):
            raise ValueError("V4.1 checkpoint source axes disagree with the selected lib")
        DecodeWeightBank.validate_abi(decode_names)

        bank_started = time.perf_counter()
        bank = load_prepacked_bank(model_dir, layout, decode_names)
        if bank is None:
            bank = DecodeWeightBank(layout)
            append_seconds = 0.0
            layer_started = time.perf_counter()
            for item in iter_decode_weight_slices(model_dir, topology):
                append_started = time.perf_counter()
                bank.append(item)
                append_seconds += time.perf_counter() - append_started
                if item.name == "hc_attn_fn":
                    now = time.perf_counter()
                    _LOG.info("V4.1 checkpoint layer %d/%d loaded into shared weight bank (%.2fs)",
                              item.slot + 1, layout.layers, now - layer_started)
                    layer_started = now
            bank.finish(decode_names)
            _LOG.info("V4.1 checkpoint staging %.2fs (bank copies %.2fs)",
                      time.perf_counter() - bank_started, append_seconds)
        _LOG.info("V4.1 shared weight bank ready: %.1f GiB",
                  bank.allocated_bytes / (1 << 30))

        rope = importlib.import_module("models.deepseek_v4_1_flash.rope_tables")
        swa_tables = rope.precompute_rope_tables(max_seq_len, False)
        compressed_tables = rope.precompute_rope_tables(max_seq_len, True)
        cache_plan = plan_decode_cache_pools(
            layers, groups, max_seq_len=max_seq_len,
            primary_num_blocks=groups[0].max_blocks_per_seq,
            request_slots=1, abi=abi,
        )
        if (runtime.total_kv_pages is not None and
                runtime.total_kv_pages < cache_plan.primary_num_blocks):
            raise ValueError("configured cache pages cannot hold one V4.1 request")
        resets = make_cache_page_reset_buffers(cache_plan)
        state_reset = make_state_reset_buffer(cache_plan)
        family = cache_plan.family_groups
        host_kwargs = dict(
            max_seq_len=max_seq_len,
            max_requests=cache_plan.physical_request_slots,
            window_pages=cache_plan.physical_group_blocks[family.window],
            c2a_pages=cache_plan.physical_group_blocks[family.c2a],
            c1a_pages=cache_plan.physical_group_blocks[family.c1a],
            rope_tables=swa_tables,
            compressed_rope_tables=compressed_tables,
        )
        prefill_host = PrefillMetadataResources(topology, **host_kwargs)
        decode_host = PrefillMetadataResources(topology, **host_kwargs)
        input_host = make_input_host_resources(topology, plan.weights.config.hidden_size)
        decode_input_host = make_input_host_resources(topology, plan.weights.config.hidden_size)
        final_host = load_final_output_host_resources(model_dir, topology)
        programs = (*attention.values(), moe, decode, final)
        resident_started = time.perf_counter()
        resident = V41ResidentResources(
            programs, run_config, weight_bank=bank, weight_param_names=decode_names,
            cache_plan=cache_plan, state_reset=state_reset, page_resets=resets,
            input_host=input_host, decode_input_host=decode_input_host,
            prefill_host=prefill_host, decode_host=decode_host,
            final_host=final_host, final_program=final, final_param_names=final_names,
            topology=topology,
        )
        _LOG.info("V4.1 resident upload and allocation %.2fs",
                  time.perf_counter() - resident_started)
        try:
            resident.bind_prefill(
                layers, kv_sources=kv_sources, index_sources=index_sources,
                c2a_sources=c2a_sources, max_seq_len=max_seq_len,
            )
            resident.bind_decode_layers(
                layers, decode_names, kv_sources=kv_sources,
                index_sources=index_sources, c2a_sources=c2a_sources,
            )
            resident.prefill_segment = PrefillSegment(
                resident.worker, attention, moe, topology,
                prefill_host.fields["attention_counts"], prefill_host.fields["moe_counts"],
                run_config,
            )
            resident.decode_segment = DecodeLayerSegment(
                resident.worker, decode, decode_names, run_config, topology,
                decode_host.fields["moe_counts"],
            )
        except BaseException:
            resident.close()
            raise
        return resident, cache_plan.primary_num_blocks

    return CompositeBindings(
        revision=revision,
        entries={},
        initialize=lambda embeddings, step, resources: resources.begin_step(embeddings, step),
        output=lambda state, step, resources: resources.finish_step(state, step),
        allocate=allocate,
        prepare_weights=lambda plans, layer, resources: None,
        reset_request=lambda resources, key, owner: resources.reset_request(owner),
        wait=lambda resources: None,
        close=lambda resources: resources.close(),
        cache_groups=groups,
        input_layout="tp_local_token",
        output_layout="tp_local_token",
        prefill_backbone=lambda state, step, resources, plans: resources.run_prefill(
            resources.prefill_segment, state, step),
        decode_backbone=lambda state, step, resources, plans: resources.run_decode_layers(
            resources.decode_segment, state, step),
    )
