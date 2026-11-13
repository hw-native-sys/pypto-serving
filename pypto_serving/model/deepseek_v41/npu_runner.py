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

import ast
import ctypes
import importlib
import inspect
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
from .engram import HostEngram
from .dspark_resources import DSPARK_MAX_BATCH_PER_DP, DSPARK_WINDOW
from .dspark_runtime import (
    DECODE_STAGE_ORDER,
    PREFILL_STAGE_ORDER,
    DSPARK_ROWS_PER_REQUEST,
    DSPARK_SPECULATIVE_TOKENS,
    DSparkDecodeDispatch,
    DSparkLoweredInputs,
    DSparkLoweringProvider,
    DSparkPrefillDispatch,
    V41DSparkRuntime,
)
from .weight_loader import (
    DecodeWeightBank,
    DecodeWeightLayout,
    LayerPlan,
    RankPlacement,
    ResidentFullDecodeWeights,
    V41ExecutionPlan,
    V41WeightLoader,
    iter_decode_weight_slices,
    load_prepacked_bank,
    peak_rss_bytes,
    plan_layers,
)
from .weight_spec import pack_mx_scale


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
    # V4.1 decode_fwd intentionally imports only EP_SIZE: DSpark is currently
    # compiled for TP1 and the lib module does not expose a TP_SIZE constant.
    # Use the selected serving topology as the source of truth instead of
    # reaching through an optional lib implementation detail.
    abi_tp_size = int(getattr(abi, "TP_SIZE", 1))
    if abi_tp_size <= 0:
        raise ValueError(f"invalid V4.1 decode TP size: {abi_tp_size}")
    resolved = validate_cache_groups(tuple(layers), tuple(groups), max_seq_len,
                                     dp_size=abi.EP_SIZE // abi_tp_size)
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
    if (world not in (4, 8) or any(type(value) is not int or value <= 0 for value in
                       (head, index, abi.STATE_CAPACITY, abi.STATE_WIDTH))
            or index % abi.INDEX_CACHE_GROUP):
        raise ValueError("decode cache dimensions disagree with the V4.1 EP4/EP8 topology")
    block = abi.BLOCK_SIZE
    window_pages = physical_group_blocks[resolved.window]
    # One lib source axis covers both compression families, so its stride must
    # fit the larger physical page namespace without remapping scheduler IDs.
    compressed_pages = max(physical_group_blocks[resolved.c2a],
                           physical_group_blocks[resolved.c1a])
    buffers = (
        CacheBufferSpec("window_cache_pool", (
            world, len(layers) * window_pages, block, 1, abi.WINDOW_CACHE_BYTES), torch.uint8),
        CacheBufferSpec("compressed_cache_pool", (
            world, kv_sources * compressed_pages, block, 1, abi.COMPRESSED_CACHE_BYTES), torch.uint8),
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
        if spec.name in ("window_cache_pool", "compressed_cache_pool"):
            from models.deepseek_v4_1_flash.quant import unpack_quantized_cache

            _, factors = unpack_quantized_cache(page, spec.name == "compressed_cache_pool")
            factors.fill_(1)
        elif spec.name == "index_cache_scale_pool":
            page.view(torch.uint8).fill_(127)
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
        }
        if layer.mode == "swa":
            return result
        kv_slot = kv_sources.index(layer.kv_source)
        index_slot = index_sources.index(layer.index_source)
        for name in ("compressed_cache", "index_cache", "index_cache_scale"):
            slot = kv_slot if name.startswith("compressed_") else index_slot
            result[name] = self.view(name + "_pool", slot)
        if layer.mode == "c2a_full":
            result["state_cache"] = self.view("state_cache_pool", c2a_sources.index(layer.layer_id))
        return result

    def reset_state_slot(self, partition: int, slot: int):
        """Clear the three C2A pending-pair rows for one DP request slot."""
        world = self.plan.buffers[0].shape[0]
        if type(partition) is not int or not 0 <= partition < world:
            raise ValueError("invalid DP partition")
        if type(slot) is not int or not 0 <= slot < self.plan.physical_request_slots:
            raise ValueError("invalid compressor state slot")
        row = self.state_reset[0]
        row_bytes = row.numel() * row.element_size()
        source = self.state_reset[partition]
        for index in range(self.source_counts["state_cache_pool"]):
            state = self.view("state_cache_pool", index)
            shard = state.shards[partition]
            self.worker.copy_to(
                shard.data_ptr - shard.byte_offset, source.data_ptr(), row_bytes,
                dst_offset=shard.byte_offset + slot * row_bytes,
                worker_id=partition,
            )

    def reset_request(self, owner, *, padding=False):
        """Clear only this request's physical pages and C2A pending state."""
        if self.page_resets is None:
            raise ValueError("request reset requires pre-fork cache page buffers")
        partition = owner.partition
        if type(partition) is not int or not 0 <= partition < self.plan.buffers[0].shape[0]:
            raise ValueError("invalid DP partition")
        if set(owner.pages) != set(self.plan.group_sources):
            raise ValueError("request page families disagree with the cache plan")
        if (type(owner.slot) is not int or
                (padding and owner.slot != self.plan.request_slots) or
                (not padding and not 0 <= owner.slot < self.plan.request_slots)):
            raise ValueError("request state slot is outside its cache family")
        fields = {
            "window": ("window_cache_pool",),
            "compressed": ("compressed_cache_pool",),
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
                    for source in sources:
                        target = self.view(name, source)
                        shard = target.shards[partition]
                        for page in pages:
                            self.worker.copy_to(
                                shard.data_ptr - shard.byte_offset, zero[partition].data_ptr(), page_bytes,
                                dst_offset=shard.byte_offset + page * page_bytes,
                                worker_id=partition,
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


class DecodeFullCachePools(DecodeCachePools):
    """One packed pool per field, with source views shared by prefill."""

    def __init__(self, worker, plan, state_reset, page_resets):
        from pypto.runtime import StackedDeviceTensor

        self.worker, self.plan = worker, plan
        self.state_reset, self.page_resets = state_reset, page_resets
        self.tensors, self.source_counts, self._full = {}, {}, {}
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
                    sources = len({slot for families in plan.group_sources.values()
                                   for slot in families.get(family, ())})
                else:
                    stride = plan.physical_request_slots
                    sources = spec.shape[1] // stride
                if spec.shape[1] != sources * stride:
                    raise ValueError(f"{spec.name} packed cache source axis is invalid")
                self.source_counts[spec.name] = sources
                base = alloc_device_buffer(worker, spec.shape, spec.dtype, stacked=True)
                self._full[spec.name] = base
                slice_shape = (stride, *spec.shape[2:])
                slot_bytes = math.prod(slice_shape) * torch.empty((), dtype=spec.dtype).element_size()
                for source in range(sources):
                    shards = tuple(worker.view_tensor(shard, slice_shape,
                                                      byte_offset=source * slot_bytes,
                                                      worker_id=rank)
                                   for rank, shard in enumerate(base.shards))
                    self.tensors[spec.name, source] = StackedDeviceTensor(
                        shards, (spec.shape[0], *slice_shape), base.worker_ids)
        except BaseException:
            self.close()
            raise

    def full(self, name):
        return self._full[name]

    def prefill_full_views(self):
        """Borrow lib prefill's source/page axes from the decode flat pools."""
        from pypto.runtime import StackedDeviceTensor

        plan = self.plan
        family = plan.family_groups
        window_pages = plan.physical_group_blocks[family.window]
        compressed_pages = max(plan.physical_group_blocks[family.c2a],
                               plan.physical_group_blocks[family.c1a])
        if window_pages != compressed_pages:
            raise ValueError("lib prefill requires one common cache page extent")
        names = {
            "window_cache": "window_cache_pool",
            "compressed_cache": "compressed_cache_pool",
            "index_cache": "index_cache_pool",
            "index_scales": "index_cache_scale_pool",
            "state_cache": "state_cache_pool",
        }
        compressed_sources = self.source_counts["compressed_cache_pool"]
        if compressed_sources != 4 or self.source_counts["index_cache_pool"] < compressed_sources:
            raise ValueError("lib prefill requires four shared compressed/index KV sources")
        result = {}
        for target, source in names.items():
            base = self.full(source)
            pages = plan.physical_request_slots if target == "state_cache" else window_pages
            sources = compressed_sources if target in ("index_cache", "index_scales") else self.source_counts[source]
            if base.shape[1] < sources * pages or (target not in ("index_cache", "index_scales")
                                                   and base.shape[1] != sources * pages):
                raise ValueError(f"{source} cannot be viewed with lib prefill source/page axes")
            shape = (sources, pages, *base.shape[2:])
            shards = tuple(self.worker.view_tensor(shard, shape, byte_offset=0, worker_id=rank)
                           for rank, shard in enumerate(base.shards))
            result[target] = StackedDeviceTensor(shards, (base.shape[0], *shape), base.worker_ids)
        return result

    def close(self):
        for tensor in reversed(tuple(self._full.values())):
            self.worker.free_stacked_tensor(tensor)
        self._full.clear()
        self.tensors.clear()


# composite
class MissingCompositeInterface(NotImplementedError):
    """The selected lib revision cannot execute the requested serving segment."""


@dataclass(frozen=True)
class BuildOptions:
    """Worker-selected compilation settings passed unchanged to the adapter."""
    platform: str = "a5"
    pypto_build_dir: str | None = None


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
    worker's build directory, and returns (resources,
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
    segment_topology: SegmentTopology | None = None
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
        if (placement.tp_size != 1 or placement.dp_size not in (4, 8)
                or placement.ep_size != placement.dp_size):
            raise ValueError("V4.1 serving requires TP1 with matching DP/EP size 4 or 8")


def load_composite_bindings(
    model_dir=None,
    max_seq_len=None,
    tokenizer=None,
    *,
    dp_size=None,
    dspark_lowering_provider=None,
    request_slots=1,
) -> CompositeBindings:
    """Resolve the model-specific contract without opening a device."""
    if model_dir is None or max_seq_len is None:
        raise ValueError("V4.1 composite bindings require a model directory and context length")
    return make_production_bindings(
        model_dir,
        max_seq_len,
        tokenizer=tokenizer,
        dp_size=dp_size,
        dspark_lowering_provider=dspark_lowering_provider,
        request_slots=request_slots,
    )


# decode_padding
def pad_single_decode(step: ForwardStep, embeddings: torch.Tensor, cache_plan):
    """Fill idle DP ranks for the collective; the ledger retains the original step."""
    if not isinstance(step, ForwardStep) or step.phase != "decode":
        raise ValueError("decode padding requires a decode ForwardStep")
    if (not isinstance(embeddings, torch.Tensor) or embeddings.device.type != "cpu"
            or embeddings.dtype != torch.bfloat16 or embeddings.ndim != 2
            or embeddings.shape[0] != len(step.token_ids)):
        raise ValueError("decode padding requires one packed BF16 embedding per token")
    dp_size = cache_plan.buffers[0].shape[0]
    counts = tuple(sum(len(request.token_ids) for request in step.requests
                       if request.partition == partition) for partition in range(dp_size))
    idle_partitions = tuple(partition for partition, count in enumerate(counts) if count == 0)
    if not idle_partitions:
        return step, embeddings, ()
    if (len(cache_plan.group_blocks) != 3 or
            set(cache_plan.group_blocks) != set(cache_plan.group_sources) or
            any(cache_plan.physical_group_blocks.get(group, 0) <= blocks
                for group, blocks in cache_plan.group_blocks.items()) or
            cache_plan.physical_request_slots <= cache_plan.request_slots):
        raise ValueError("decode filler requires all three reserved cache families")
    fillers = []
    existing_ids = {request.request_id for request in step.requests}
    for partition in idle_partitions:
        name = f"__v41_decode_padding_{partition}__"
        while name in existing_ids:
            name += "_"
        existing_ids.add(name)
        fillers.append(RequestSlice(
            request_id=name, partition=partition, state_slot=cache_plan.request_slots,
            start=0, token_ids=(step.requests[0].token_ids[0],), prompt_length=1,
            pages=MappingProxyType({group: (blocks,) for group, blocks in cache_plan.group_blocks.items()}),
        ))
    padded = ForwardStep(step.phase, (*step.requests, *fillers), step.epoch)
    filler_embeddings = embeddings[:1].expand(len(fillers), -1)
    return padded, torch.cat((embeddings, filler_embeddings), dim=0), idle_partitions


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
_MAX_REQUESTS_PER_DP = 12
MAX_PREFILL_TOKENS_PER_DP = 4096


def _validate_dp_capacity(partitions, token_counts, *, phase, dp_size):
    requests = [0] * dp_size
    tokens = [0] * dp_size
    for partition, count in zip(partitions, token_counts):
        if type(partition) is not int or not 0 <= partition < dp_size:
            raise ValueError(f"requests require an explicit DP cache partition in [0, {dp_size})")
        requests[partition] += 1
        tokens[partition] += count
    if any(count > _MAX_REQUESTS_PER_DP for count in requests):
        raise ValueError("V4.1 lib supports at most 12 requests per DP partition")
    if phase == "prefill" and any(count > MAX_PREFILL_TOKENS_PER_DP for count in tokens):
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
    dp_size = groups[0].num_partitions
    if any(group.num_partitions != dp_size for group in groups):
        raise ValueError("V4.1 cache groups require the same DP partition count")
    if any(group.sliding_window is not None for group in groups):
        raise MissingCompositeInterface("rolling cache page lowering requires a verified composite contract")
    if len(rows) != len(partitions) or len(rows) != len(ends):
        raise ValueError("page tables, partitions and request extents must have matching rows")
    required = {group.name for group in groups}
    owners, result = set(), []
    for pages, partition, end in zip(rows, partitions, ends):
        if type(partition) is not int or not 0 <= partition < dp_size:
            raise ValueError(f"requests require an explicit DP cache partition in [0, {dp_size})")
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
    _validate_dp_capacity(batch.cache_partitions, batch.chunk_lens, phase="prefill",
                          dp_size=groups[0].num_partitions)
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
    _validate_dp_capacity(batch.cache_partitions, (1 for _ in range(count)), phase="decode",
                          dp_size=groups[0].num_partitions)
    pages = validate_page_tables(batch.block_ids_by_group, batch.cache_partitions, ends, groups)
    return [(key, partition, end - 1, token, table)
            for key, partition, end, token, table in zip(
                batch.request_ids, batch.cache_partitions, ends, ids, pages)]


@dataclass(frozen=True)
class SegmentTopology:
    tp: int = 1
    dp: int = 8
    local_capacity: int = 16

    @classmethod
    def for_prefill(cls, *, dp: int, max_tokens: int):
        """Keep prefill rows independent of the fixed decode ABI."""
        if type(max_tokens) is not int or not 0 < max_tokens <= MAX_PREFILL_TOKENS_PER_DP:
            raise ValueError("V4.1 prefill capacity must be between 1 and 4096 tokens")
        # Shared MoE kernels retain decode's extent even for a smaller token budget.
        return cls(tp=1, dp=dp, local_capacity=max(
            cls.for_decode(dp=dp).local_capacity, (max_tokens + 15) // 16 * 16))

    @classmethod
    def for_decode(cls, *, tp: int = 1, dp: int = 8):
        """Match the current lib's 12-request, six-row decode extent."""
        if type(tp) is not int or tp not in (1, 2, 4, 8):
            raise ValueError("unsupported decode TP size")
        decode_tokens = 12 * 6
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
    expected = root / "models/deepseek_v4_1_flash/config.py"
    existing = None
    for name in ("config", package + ".config"):
        module = sys.modules.get(name)
        if module is not None:
            if Path(module.__file__).resolve() != expected or existing not in (None, module):
                raise RuntimeError("V41 lib from another checkout is already imported; use a fresh worker")
            existing = module
    old_argv = sys.argv[:]
    try:
        # The lib's model modules import sibling files by their top-level names.
        for path in (root, root / "models/deepseek_v4_1_flash"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        sys.argv = [old_argv[0], "--cp", "1", "--lm-head-tp", str(topology.tp),
                    "--ep", str(topology.world)]
        config = existing or importlib.import_module("config")
        if Path(config.__file__).resolve() != expected:
            raise RuntimeError("import resolved to a different lib checkout")
        # Lib siblings import `config`; package imports must use that same
        # object. Prefill and Decode freeze their own, different MoE extents.
        sys.modules["config"] = sys.modules[package + ".config"] = config
        if (config.CP_SIZE, config.LM_HEAD_TP_SIZE, config.EP_SIZE, config.DP_SIZE) != (
            1, topology.tp, topology.world, topology.dp
        ):
            raise ValueError("lib CP/LM-head/EP/DP topology disagrees with the serving topology")
        if not (root / "models/deepseek_v4_1_flash/moe.py").is_file():
            return None, None
        attention = importlib.import_module(package + ".prefill_swa")
        moe = importlib.import_module(package + ".moe")
        if moe.SKIP_SHARED_TEST or moe.SKIP_TRANSPORT_TEST:
            raise ValueError("cannot run serving with lib test-only MoE bypasses")
        return attention, moe
    finally:
        sys.argv = old_argv


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
def compile_decode_fwd(compiler, lib_root, topology):
    """Compile the complete 40-layer backbone from the selected lib revision."""
    import pypto.language as pl

    load_segment_modules(lib_root, topology)
    module = importlib.import_module("models.deepseek_v4_1_flash.decode_fwd")
    entry = module.l3_decode_fwd
    annotation = inspect.signature(entry._func).parameters["attention_num_tokens"].annotation
    shape = tuple(getattr(annotation, "shape", ()))
    if shape == (topology.world,):
        program = compiler.compile("v41_decode_fwd", entry)
        counts_are_tensor = True
    elif not shape:
        program = compiler.compile("v41_decode_fwd", entry, attention_num_tokens=pl.RUNTIME)
        counts_are_tensor = False
    else:
        raise ValueError(f"unsupported decode attention_num_tokens shape: {shape}")
    return program, tuple(entry.param_names), counts_are_tensor


def compile_decode_fwd_dspark(compiler, lib_root, topology):
    """Compile the recurrent decode entry and the prefill bootstrap programs."""
    import pypto.language as pl

    if topology.tp != 1:
        raise MissingCompositeInterface("V4.1 DSpark currently requires TP1")
    load_segment_modules(lib_root, topology)
    module = importlib.import_module("models.deepseek_v4_1_flash.decode_fwd_dspark")
    entries = {"decode": module.l3_decode_fwd_dspark}
    state = importlib.import_module("models.deepseek_v4_1_flash.dspark_state")
    entries.update({name.removeprefix("l3_"): getattr(state, name) for name in (
        "l3_bootstrap_inputs", "l3_publish_bootstrap",
        "l3_prepare_prefill_drafter_geometry")})
    entries["dspark_drafter"] = importlib.import_module(
        "models.deepseek_v4_1_flash.dspark_drafter").l3_dspark_drafter
    entries["markov_sample"] = importlib.import_module(
        "models.deepseek_v4_1_flash.dspark_markov").l3_markov_sample_from_hidden
    programs = {}
    param_names = {}
    for name, entry in entries.items():
        kwargs = {}
        params = set(entry.param_names)
        if "moe_epoch" in params:
            kwargs["moe_epoch"] = pl.RUNTIME
        if "logits_epoch" in params:
            kwargs["logits_epoch"] = pl.RUNTIME
        programs[name] = compiler.compile("v41_dspark_" + name, entry, **kwargs)
        param_names[name] = tuple(entry.param_names)
    return programs, param_names


class DecodeForwardSegment:
    """Bind one full-backbone dispatch to resident weights, caches and metadata."""

    _METADATA = {
        "freqs_cos": "full_freqs_cos", "freqs_sin": "full_freqs_sin",
        "token_to_req_indices": "c2a_request_ids",
        "position_ids": "c2a_position_ids",
        "c2a_index_block_table": "c2a_index_block_table",
        "c1a_index_block_table": "c1a_index_block_table",
        "query_start_loc": "c2a_query_start_loc",
        "state_block_table": "c2a_state_block_table",
        "engram_lookup": "engram_lookup",
        "dspark_target_hidden": "dspark_target_hidden",
    }

    def __init__(self, worker, program, param_names, run_config, topology,
                 counts_are_tensor,
                 weights, cache, metadata, final, engram=None):
        self.worker, self.program = worker, program
        self.param_names = tuple(param_names)
        self.run_config, self.topology = run_config, topology
        self.counts_are_tensor = counts_are_tensor
        self.metadata = metadata
        arguments = {name: weights.full(name) for name in weights.layout_names}
        arguments.update({name: cache.full(name) for name in cache._full})
        arguments.update({name: metadata.fields[source]
                          for name, source in self._METADATA.items()})
        arguments.update({name: metadata.fields[name] for name in (
            "window_slots", "window_indices", "c2a_compressed_slots", "c1a_compressed_slots",
        )})
        arguments.update(
            final_norm_weight=final.norm_weight,
            lm_head_weight=final.head_weight,
            logit_row_indices=final.host.logit_row_indices,
            logits=final.host.logits,
            sampled_ids=final.host.sampled_ids,
        )
        if engram is not None and "engram_lookup" in param_names:
            arguments.update(engram_lookup=metadata.fields["engram_lookup"],
                            engram_wkv=engram.wkv,
                            engram_wkv_scale=engram.wkv_scale,
                            engram_weight=engram.weight)
        arguments["moe_num_tokens"] = metadata.fields["moe_counts"]
        if counts_are_tensor:
            if topology.tp != 1:
                raise ValueError("decode attention counts share MoE rows only at TP1")
            arguments["attention_num_tokens"] = metadata.fields["moe_counts"]
        missing = set(param_names) - set(arguments) - {
            "x_hc", "pre_mix", "attention_num_tokens"}
        extra = set(arguments) - set(param_names)
        if missing or extra:
            raise ValueError(f"full decode ABI mismatch: missing={sorted(missing)} extra={sorted(extra)}")
        self.arguments = arguments
        self.failed = False

    def run(self, state, *, group_counts):
        if self.failed:
            raise RuntimeError("full decode failed; close the worker before retrying")
        if state.layout != "tp_local_token":
            raise ValueError("full decode requires TP-local state")
        if (len(group_counts) != self.topology.dp or
                any(count <= 0 or count > self.topology.capacity for count in group_counts)):
            raise ValueError("full decode requires nonempty token rows within capacity on all DP ranks")
        if not self.counts_are_tensor and len(set(group_counts)) != 1:
            raise ValueError("scalar-count full decode requires equal token rows on all DP ranks")
        bound = dict(self.arguments, x_hc=state.residual, pre_mix=state.pre_mix)
        if not self.counts_are_tensor:
            bound["attention_num_tokens"] = ctypes.c_int32(group_counts[0])
        if not getattr(self.worker, "_persistent", False) or not hasattr(
                self.worker, "_reset_persistent_windows"):
            raise RuntimeError("full decode requires a persistent worker")
        previous_reset = self.worker._reset_persistent_windows
        self.worker._reset_persistent_windows = False
        try:
            self.worker.run(self.program.compiled,
                            *(bound[name] for name in self.param_names), config=self.run_config)
        except BaseException:
            self.failed = True
            raise
        finally:
            self.worker._reset_persistent_windows = previous_reset
        return state


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
                plan.layers, bindings.cache_groups, runtime.max_seq_len,
                dp_size=plan.placement.dp_size)
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
            self.ledger = RequestLedger(
                max_requests=self.bindings.cache_groups[0].max_requests_per_partition,
                max_seq_len=self.runtime.max_seq_len,
                partitions=self.plan.placement.dp_size)
        if getattr(self.resources, "dspark", None) is not None:
            self.resources.ledger = self.ledger
        return self.ledger

    def _reset_request(self, key, owner):
        self.bindings.reset_request(self.resources, key, owner)
        self.bindings.wait(self.resources)
        self.resources.release_request(key)

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

    def run_dspark_prefill(self, request_id, *, slot, generation, prompt_length,
                           anchor, stages, lowered=None):
        """Publish the first five drafts after a completed V4.1 prefill.

        The scheduler/device adapter supplies already-bound stage arguments;
        this method only owns request state and invokes the lib composites in
        their documented order.
        """
        if self.runtime.num_speculative_tokens != 5:
            raise ValueError("V4.1 DSpark requires num_speculative_tokens=5")
        if not hasattr(self.resources, "run_dspark_prefill"):
            raise MissingCompositeInterface("V4.1 resources do not expose DSpark stage dispatch")
        return self.resources.run_dspark_prefill(
            request_id, slot=slot, generation=generation,
            prompt_length=prompt_length, anchor=anchor, stages=stages,
            lowered=lowered,
        )

    def lower_dspark_prefill_batch(self, model, batch, *, request_ids, sampled_token_ids):
        """Lower a terminal prefill batch through the resident provider."""
        if self.resources is None:
            raise MissingCompositeInterface("V4.1 DSpark resources are not initialized")
        return self.resources.lower_dspark_prefill(
            model=model, batch=batch, request_ids=request_ids,
            sampled_token_ids=sampled_token_ids,
        )

    def run_dspark_decode(self, request_ids, *, stages, accepted_counts=None,
                          last_sampled=None, draft_ids=None, lowered=None,
                          request_ranks=None):
        """Run one target-accept-drafter-Markov-commit verification round."""
        if self.runtime.num_speculative_tokens != 5:
            raise ValueError("V4.1 DSpark requires num_speculative_tokens=5")
        if not hasattr(self.resources, "run_dspark_decode"):
            raise MissingCompositeInterface("V4.1 resources do not expose DSpark stage dispatch")
        return self.resources.run_dspark_decode(
            request_ids, stages=stages, accepted_counts=accepted_counts,
            last_sampled=last_sampled, draft_ids=draft_ids,
            lowered=lowered, request_ranks=request_ranks,
        )

    def lower_dspark_decode_batch(self, model, batch):
        """Lower one decode batch through the resident provider."""
        if self.resources is None:
            raise MissingCompositeInterface("V4.1 DSpark resources are not initialized")
        with self._lock:
            ledger = self._request_ledger()
            requests = decode_requests(batch, model.config, self.runtime, self.bindings.cache_groups)
            drafts = {key: self.resources.dspark.states.get(key).draft_ids for key in batch.request_ids}
            validate_page_tables(
                batch.block_ids_by_group, batch.cache_partitions,
                [start + 1 + len(drafts[key]) for key, _, start, _, _ in requests],
                self.bindings.cache_groups,
            )
            step = ledger.begin_decode(requests, drafts=drafts)
            try:
                return self.resources.lower_dspark_decode(model=model, batch=batch)
            except Exception:
                ledger.abort(step, self._reset_request)
                raise

    def run_dspark_prefill_batch(self, batch):
        """Consume the typed DSpark prefill payload from a serving batch."""
        dispatch = DSparkPrefillDispatch.from_batch(batch)
        return self.run_dspark_prefill(
            dispatch.request_id,
            slot=dispatch.slot,
            generation=dispatch.generation,
            prompt_length=dispatch.prompt_length,
            anchor=dispatch.anchor,
            stages=dispatch.stages,
            lowered=dispatch.lowered,
        )

    def run_dspark_decode_batch(self, batch):
        """Consume the typed DSpark decode payload from a serving batch."""
        dispatch = DSparkDecodeDispatch.from_batch(batch)
        with self._lock:
            ledger = self._request_ledger()
            step = ledger.pending
            if step is None or tuple(request.request_id for request in step.requests) != tuple(dispatch.request_ids):
                raise ValueError("DSpark decode requires its matching request transaction")
            try:
                result = self.run_dspark_decode(
                    dispatch.request_ids,
                    stages=dispatch.stages,
                    accepted_counts=dispatch.accepted_counts,
                    last_sampled=dispatch.last_sampled,
                    draft_ids=dispatch.draft_ids,
                    lowered=dispatch.lowered,
                    request_ranks=dispatch.request_ranks,
                )
                self.bindings.wait(self.resources)
                ledger.commit(step, accepted_counts=[len(tokens) for tokens in result.accepted_token_ids])
                return result
            except Exception:
                try:
                    self.bindings.wait(self.resources)
                except Exception:
                    self.failed = ledger.poisoned = True
                    raise
                ledger.abort(step, self._reset_request)
                raise

def compile_prefill_fwd(compiler, lib_root, topology):
    """Compile the lib's complete CED prefill at the serving row capacity."""
    decode_topology = SegmentTopology.for_decode(tp=topology.tp, dp=topology.dp)
    load_segment_modules(lib_root, decode_topology)
    previous_argv = sys.argv[:]
    try:
        sys.argv = [previous_argv[0], "--tp", str(topology.tp), "--ep", str(topology.world),
                    "--dp", str(topology.dp), "--capacity", str(topology.local_capacity),
                    "--moe-tokens", str(decode_topology.local_capacity)]
        module = importlib.import_module("models.deepseek_v4_1_flash.prefill_fwd")
    finally:
        sys.argv = previous_argv
    if (module.CAPACITY, module.RANKS) != (topology.local_capacity, topology.world):
        raise ValueError("lib prefill_fwd was imported for another capacity or EP topology")
    if module.MOE_TOKENS != decode_topology.local_capacity:
        raise ValueError("lib prefill MoE extent disagrees with the shared decode kernels")
    entry = module.l3_prefill_fwd
    return compiler.compile("v41_prefill_fwd", entry), tuple(entry.param_names)


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
    max_prepare_bytes: int = 4 << 30,
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


class HostEngramResources:
    """Host hash lookup plus resident Engram projection weights."""

    def __init__(self, model_dir, topology, tokenizer, *, capacity):
        loader = V41WeightLoader(model_dir, tp_size=1, tp_rank=0,
                                 ep_size=topology.world, ep_rank=0,
                                 max_load_bytes=8 << 30)
        config = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))["text_config"]
        # The adapter follows the official model's raw tokenizer ABI. It owns
        # host hashing/history; pypto-lib contributes only device kernels.
        self.host = HostEngram(loader, tokenizer, config)
        layers = self.host.layout.layer_ids
        if len(layers) != 2:
            raise ValueError("V4.1 serving currently requires two Engram layers")
        wkv_values, scale_values, gate_values = [], [], []
        for layer in layers:
            wkv = loader.store.load_tensor(f"layers.{layer}.engram.wkv.weight")
            scale = loader.store.load_tensor(f"layers.{layer}.engram.wkv.scale")
            q = loader.store.load_tensor(f"layers.{layer}.engram.q_weight").float()
            k = loader.store.load_tensor(f"layers.{layer}.engram.k_weight").float()
            if tuple(wkv.shape) != (self.host.kv_out, self.host.layout.lookup_width) or tuple(scale.shape) != (
                    self.host.kv_out // 32, self.host.layout.lookup_width // 32):
                raise ValueError(f"unsupported Engram projection shape at layer {layer}")
            wkv_values.append(wkv.t().contiguous())
            scale_values.append(pack_mx_scale(scale.view(torch.uint8).t().repeat_interleave(32, dim=1))
                                 .view(torch.float8_e8m0fnu))
            gate_values.append((q * k).to(torch.float32))
        def shared(value):
            return value.unsqueeze(0).expand(topology.world, *value.shape).contiguous().share_memory_()
        self.lookup = torch.zeros((topology.world, 2, capacity, self.host.layout.lookup_width),
                                  dtype=torch.bfloat16).share_memory_()
        self.wkv = shared(torch.stack(wkv_values))
        # The Engram ABI flattens the two layer scale matrices along the
        # group axis: [2 * (ENGRAM_K // 32), KV_OUT].  The weight and gate
        # tensors keep an explicit layer dimension, so their layouts cannot
        # be handled by the same stack operation.
        self.wkv_scale = shared(torch.cat(scale_values, dim=0))
        self.weight = shared(torch.stack(gate_values))

    @property
    def inherited_host_tensors(self):
        return self.lookup, self.wkv, self.wkv_scale, self.weight

    def release_request(self, request_id):
        self.host.release_request(request_id)

    def stage_lookup(self, target, step, topology, *, row_stride=None):
        if (target.shape[1] != self.lookup.shape[1] or target.shape[3] != self.lookup.shape[3]
                or target.shape[2] > self.lookup.shape[2]):
            raise ValueError("Engram lookup capacity disagrees with the forward ABI")
        target.zero_()
        offsets = [0] * topology.dp
        for request in step.requests:
            start = offsets[request.partition]
            values = self.host.lookup(request.token_ids, topology.capacity,
                                      request.start, request_id=request.request_id)
            count = len(request.token_ids)
            if row_stride is not None and count > row_stride:
                raise ValueError("Engram active rows exceed the request stride")
            for tp_rank in range(topology.tp):
                rank = request.partition * topology.tp + tp_rank
                target[rank, :, start:start + count].copy_(values[:, :count])
            offsets[request.partition] += count if row_stride is None else row_stride


class ResidentEngramWeights:
    def __init__(self, worker, host):
        self.worker = worker
        self.host = host
        self._owned = [worker.alloc_stacked_tensor(value) for value in (
            host.wkv, host.wkv_scale, host.weight)]

    @property
    def wkv(self):
        return self._owned[0]

    @property
    def wkv_scale(self):
        return self._owned[1]

    @property
    def weight(self):
        return self._owned[2]

    def close(self):
        for value in reversed(self._owned):
            self.worker.free_stacked_tensor(value)
        self._owned.clear()


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
# prefill_metadata_resources
class PrefillMetadataResources:
    """Keep C2A and C1A page/length contracts separate for full forwards."""

    def __init__(self, topology, *, max_seq_len, max_requests, window_pages,
                 c2a_pages, c1a_pages, rope_tables, compressed_rope_tables,
                 full_hidden_size=None, dspark_hidden_size=None, engram_k=6144):
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
        self.full_hidden_size = full_hidden_size
        if full_hidden_size is not None and (type(full_hidden_size) is not int or full_hidden_size <= 0):
            raise ValueError("full prefill hidden size must be positive")
        if dspark_hidden_size is not None and (
                type(dspark_hidden_size) is not int or dspark_hidden_size <= 0):
            raise ValueError("DSpark hidden size must be positive")
        self.dspark_hidden_size = dspark_hidden_size

        def shared(name, shape, dtype, fill=0):
            value = torch.empty(shape, dtype=dtype).share_memory_()
            value.fill_(fill)
            self.fields[name] = value
            return value

        world, tokens, width = topology.world, topology.capacity, rope_tables[0].shape[1]
        if type(engram_k) is not int or engram_k <= 0:
            raise ValueError("Engram width must be positive")
        shared("engram_lookup", (world, 2, tokens, engram_k), torch.bfloat16)
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
        for suffix, base, compressed in zip(("cos", "sin"), rope_tables,
                                             compressed_rope_tables):
            tables = shared("full_freqs_" + suffix,
                            (world, 2, max_seq_len, width), torch.float32)
            tables[:, 0].copy_(base[:max_seq_len])
            tables[:, 1].copy_(compressed[:max_seq_len])
        if full_hidden_size is not None:
            shared("full_embedding", (world, tokens, full_hidden_size), torch.bfloat16)
            shared("full_embedding_ids", (world, tokens), torch.int64)
            shared("full_initial_pre", (world, tokens, 4), torch.float32)
            shared("full_identity_rows", (world, tokens), torch.int32, -1)
            shared("full_counts", (world, 2), torch.int32)
            shared("full_replay_rows", (world, tokens), torch.int32, -1)
            shared("full_position_ids", (world, 2, tokens), torch.int32, -1)
            shared("full_window_slots", (world, 2, tokens), torch.int64, -1)
            shared("full_window_indices", (world, 2, tokens, 128), torch.int32, -1)
            shared("full_token_requests", (world, 2, tokens), torch.int32, -1)
            shared("full_query_starts", (world, 2, max_requests + 1), torch.int32)
            shared("full_logit_rows", (world, 16), torch.int32, -1)
        if dspark_hidden_size is not None:
            # The target hidden is an output of the complete backbone and is
            # retained for the following DSpark drafter dispatch.
            shared("dspark_target_hidden", (world, tokens, dspark_hidden_size), torch.bfloat16)

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

    def stage(self, step, engram=None):
        # DP8 at the maximum 4096-row capacity needs 33 MiB of SWA scratch.
        swa = prepare_swa_window_metadata(
            step, self.topology, cache_pages=self.window_pages, max_prepare_bytes=64 << 20)
        self.fields["window_slots"].copy_(swa.window_slots)
        self.fields["window_indices"].copy_(swa.window_indices)
        cos, sin = gather_swa_rope_rows(
            swa, self.tables[""], max_prepare_bytes=64 << 20)
        self.fields["rope_cos"].copy_(cos)
        self.fields["rope_sin"].copy_(sin)
        attention, moe = self.topology.counts(swa.group_counts)
        self.fields["attention_counts"][:, 0].copy_(torch.tensor(attention, dtype=torch.int32))
        self.fields["moe_counts"].copy_(torch.tensor(moe, dtype=torch.int32))
        if engram is not None:
            engram.stage_lookup(self.fields["engram_lookup"], step, self.topology)
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

    def stage_full(self, step, embeddings, engram=None):
        """Lower one chunk into the two-stage lib prefill ABI using scheduler pages."""
        if self.full_hidden_size is None:
            raise ValueError("full prefill staging was not allocated before worker startup")
        if embeddings.shape[-1] != self.full_hidden_size:
            raise ValueError("full prefill embedding width disagrees with the checkpoint")
        counts = self.stage(step, engram)
        packed = prepare_segment_inputs(embeddings, step, self.topology)
        f = self.fields
        f["full_embedding"].zero_()
        active = packed.row_indices >= 0
        f["full_embedding"][active] = embeddings[packed.row_indices[active]]
        f["full_embedding_ids"].copy_(
            torch.arange(self.topology.capacity, dtype=torch.int64).expand(self.topology.world, -1))
        f["full_initial_pre"].zero_()
        f["full_initial_pre"][:, :, 0] = 1
        f["full_identity_rows"].fill_(-1)
        f["full_replay_rows"].fill_(-1)
        f["full_counts"].zero_()
        for rank, count in enumerate(counts):
            f["full_counts"][rank] = count
            f["full_identity_rows"][rank, :count] = torch.arange(count, dtype=torch.int32)
            f["full_replay_rows"][rank, :count] = torch.arange(count, dtype=torch.int32)
        for stage, family in enumerate(("c2a", "c1a")):
            f["full_position_ids"][:, stage].copy_(f[family + "_position_ids"])
            f["full_window_slots"][:, stage].copy_(f["window_slots"])
            f["full_window_indices"][:, stage].copy_(f["window_indices"])
            f["full_token_requests"][:, stage].copy_(f[family + "_request_ids"])
            f["full_query_starts"][:, stage].copy_(f[family + "_query_start_loc"])
        f["full_logit_rows"].fill_(-1)
        per_rank = [0] * self.topology.world
        rows_per_rank = [0] * self.topology.world
        for request in step.requests:
            rank = request.partition
            slot = per_rank[rank]
            if slot >= 16:
                raise ValueError("full prefill supports at most 16 output requests per rank")
            f["full_logit_rows"][rank, slot] = rows_per_rank[rank] + len(request.token_ids) - 1
            rows_per_rank[rank] += len(request.token_ids)
            per_rank[rank] += 1
        return counts

    def full_arguments(self):
        if self.full_hidden_size is None:
            raise ValueError("full prefill staging was not allocated")
        f = self.fields
        names = {
            "embedding": "full_embedding", "embedding_ids": "full_embedding_ids",
            "engram_lookup": "engram_lookup",
            "initial_pre": "full_initial_pre", "identity_rows": "full_identity_rows",
            "counts": "full_counts", "replay_rows": "full_replay_rows",
            "logit_rows": "full_logit_rows",
            "position_ids": "full_position_ids", "freqs_cos": "full_freqs_cos",
            "freqs_sin": "full_freqs_sin", "window_slots": "full_window_slots",
            "window_indices": "full_window_indices",
            "token_to_req_indices": "full_token_requests",
            "query_start_loc": "full_query_starts",
            "c2a_index_block_table": "c2a_index_block_table",
            "c1a_index_block_table": "c1a_index_block_table",
            "c2a_compressed_slots": "c2a_compressed_slots",
            "c1a_compressed_slots": "c1a_compressed_slots",
            "state_block_table": "c2a_state_block_table",
        }
        for name in ("engram_lookup", "dspark_target_hidden"):
            if name in f:
                names[name] = name
        return {name: f[source] for name, source in names.items()}

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
                                     max_host_bytes: int = 16 << 30) -> FinalOutputHostResources:
    """Load one TP vocabulary shard at a time before the worker forks."""
    if (not isinstance(topology, SegmentTopology) or topology.tp != 1
            or topology.dp not in (4, 8)):
        raise ValueError("V4.1 final output requires TP1/DP4 or TP1/DP8 placement")
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
    head = torch.empty((topology.world, hidden, shard_rows), dtype=torch.bfloat16).share_memory_()
    indices = torch.full((topology.world, max_logit_rows), -1, dtype=torch.int32).share_memory_()
    logits = torch.zeros((topology.world, max_logit_rows, vocab), dtype=torch.float32).share_memory_()
    sampled = torch.full((topology.world, max_logit_rows, sampled_pad), -1,
                         dtype=torch.int32).share_memory_()

    norm_value = loader.load("norm.weight").weight
    if norm_value.dtype != torch.bfloat16 or tuple(norm_value.shape) != (hidden,):
        raise ValueError("checkpoint final Norm weight disagrees with the lib output ABI")
    norm.copy_(norm_value.expand_as(norm))
    from models.deepseek_v4_1_flash.quant import pack_bf16_nz

    for tp_rank in range(topology.tp):
        rank_loader = (loader if tp_rank == 0 else V41WeightLoader(
            model_dir, tp_size=topology.tp, tp_rank=tp_rank,
            ep_size=topology.world, ep_rank=tp_rank))
        for start in range(0, shard_rows, chunk_rows):
            stop = min(start + chunk_rows, shard_rows)
            value = rank_loader.load_rows("head.weight", start, stop).weight
            if value.dtype != torch.bfloat16 or tuple(value.shape) != (stop - start, hidden):
                raise ValueError("checkpoint LM-head shard disagrees with the lib output ABI")
            head[tp_rank, :, start:stop].copy_(value.t())
        head[tp_rank].copy_(pack_bf16_nz(head[tp_rank]))
    for dp_rank in range(1, topology.dp):
        for tp_rank in range(topology.tp):
            head[dp_rank * topology.tp + tp_rank].copy_(head[tp_rank])
    return FinalOutputHostResources(norm, head, indices, logits, sampled)


class FinalOutputWeights:
    """Keep output weights resident for the full lib forwards.

    The host resources must exist before the shared DistributedWorker forks.
    The caller owns and closes the worker after this runtime releases its tensors.
    """

    def __init__(self, worker, topology: SegmentTopology, host: FinalOutputHostResources):
        if not isinstance(host, FinalOutputHostResources) or not isinstance(topology, SegmentTopology):
            raise ValueError("final output requires host resources and a segment topology")
        world = topology.world
        if any(not tensor.is_shared() for tensor in host.all_host_tensors):
            raise ValueError("final output host buffers must be shared before worker creation")
        if (host.norm_weight.dtype != torch.bfloat16 or host.norm_weight.shape != (world, host.head_weight.shape[1])
                or host.head_weight.dtype != torch.bfloat16 or host.head_weight.shape[0] != world
                or host.logit_row_indices.dtype != torch.int32 or host.logit_row_indices.shape != (world, 16)
                or host.logits.dtype != torch.float32
                or host.logits.shape != (world, 16, host.head_weight.shape[-1] * topology.tp)
                or host.sampled_ids.dtype != torch.int32 or host.sampled_ids.shape != (world, 16, 8)):
            raise ValueError("final output host buffers disagree with the lib TP1/DP ABI")
        self.worker, self.host, self.topology = worker, host, topology
        self._resident = []
        try:
            self.norm_weight = worker.alloc_stacked_tensor(host.norm_weight)
            self._resident.append(self.norm_weight)
            self.head_weight = worker.alloc_stacked_tensor(host.head_weight)
            self._resident.append(self.head_weight)
        except BaseException:
            self.close()
            raise

    def close(self):
        for tensor in reversed(self._resident):
            self.worker.free_stacked_tensor(tensor)
        self._resident.clear()


class PrefillForwardSegment:
    """Run the lib's complete 40-layer prefill using resident decode weights."""

    def __init__(self, worker, program, param_names, run_config, topology,
                 weights, metadata, cache, final, engram=None):
        self.worker, self.program = worker, program
        self.param_names = tuple(param_names)
        self.run_config, self.metadata = run_config, metadata
        self.failed = False
        self._owned = []
        try:
            self.residual = alloc_device_buffer(
                worker, (topology.world, topology.local_capacity, 4,
                         final.host.head_weight.shape[1]), torch.float32, stacked=True)
            self._owned.append(self.residual)
            self.pre_mix = alloc_device_buffer(
                worker, (topology.world, topology.local_capacity, 4),
                torch.float32, stacked=True)
            self._owned.append(self.pre_mix)
            self.arguments = dict(
                metadata.full_arguments(),
                **{name: weights.full(name) for name in weights.layout_names},
                **cache.prefill_full_views(),
                final_norm_weight=final.norm_weight, lm_head_weight=final.head_weight,
                x_hc_out=self.residual, pre_mix_out=self.pre_mix,
                logits=final.host.logits, sampled_ids=final.host.sampled_ids,
            )
            if engram is not None and "engram_lookup" in param_names:
                self.arguments.update(engram_lookup=metadata.fields["engram_lookup"],
                                      engram_wkv=engram.wkv,
                                      engram_wkv_scale=engram.wkv_scale,
                                      engram_weight=engram.weight)
            elif "engram_lookup" not in param_names:
                self.arguments.pop("engram_lookup", None)
            if len(self.param_names) != len(set(self.param_names)) or set(self.param_names) != set(self.arguments):
                raise ValueError("lib full prefill ABI differs from the serving binding")
        except BaseException:
            self.close()
            raise

    def run(self):
        if self.failed:
            raise RuntimeError("full prefill failed; recreate the worker before reuse")
        if not getattr(self.worker, "_persistent", False) or not hasattr(
            self.worker, "_reset_persistent_windows"
        ):
            raise RuntimeError("full prefill requires a persistent worker")
        previous_reset = self.worker._reset_persistent_windows
        self.worker._reset_persistent_windows = False
        try:
            self.worker.run(self.program.compiled,
                            *(self.arguments[name] for name in self.param_names),
                            config=self.run_config)
        except BaseException:
            self.failed = True
            raise
        finally:
            self.worker._reset_persistent_windows = previous_reset
        return LayerState(self.residual, self.pre_mix, "tp_local_token")

    def close(self):
        for tensor in reversed(self._owned):
            self.worker.free_stacked_tensor(tensor)
        self._owned.clear()


class ResidentDSparkWeights:
    """Upload deferred DSpark weights one EP rank and layer at a time.

    The checkpoint contract deliberately exposes bounded ``load_layer_packed``
    reads.  Keep that property here: the worker owns the final ``[EP, 3, ...]``
    tensors while only one rank/layer temporary exists on the host.
    """

    def __init__(self, worker, model_dir, world):
        from pypto.runtime import StackedDeviceTensor

        from .dspark_weights import V41DSparkWeightContract
        from .weight_loader import V41WeightLoader

        self.worker = worker
        self.tensors = {}
        self._owned = []
        contracts = [
            V41DSparkWeightContract(V41WeightLoader(
                model_dir, tp_size=1, tp_rank=0, ep_size=world, ep_rank=rank,
                max_load_bytes=512 << 20,
            )) for rank in range(world)
        ]
        try:
            # Heads are replicated over EP.  Read them once and upload one copy
            # to every rank; the three draft layers remain rank-local.
            heads = contracts[0].load_heads().tensors
            for name, value in heads.items():
                shards = []
                for rank in range(world):
                    shard = worker.alloc_tensor(tuple(value.shape), value.dtype, worker_id=rank)
                    self._owned.append((rank, shard))
                    self._copy(worker, shard, value, rank)
                    shards.append(shard)
                self.tensors[name] = StackedDeviceTensor(
                    tuple(shards), (world, *value.shape), tuple(range(world)))

            layer_shapes = {}
            layer_handles = {}
            # Stream one rank/layer at a time.  In particular, do not retain
            # the 3-layer expert payloads for all EP ranks on the host.
            for rank, contract in enumerate(contracts):
                for layer in range(3):
                    packed = contract.load_layer_packed(layer).tensors
                    for name, value in packed.items():
                        shape = tuple(value.shape)
                        previous = layer_shapes.get(name)
                        if previous is not None and previous != (shape, value.dtype):
                            raise ValueError(f"DSpark rank/layer shape mismatch for {name}")
                        layer_shapes[name] = (shape, value.dtype)
                        handles = layer_handles.setdefault(name, [None] * world)
                        if handles[rank] is None:
                            handles[rank] = worker.alloc_tensor(
                                (3, *shape), value.dtype, worker_id=rank)
                            self._owned.append((rank, handles[rank]))
                        self._copy(
                            worker, handles[rank], value, rank,
                            offset=layer * value.numel() * value.element_size(),
                        )
                    del packed
            for name, (shape, dtype) in layer_shapes.items():
                handles = layer_handles[name]
                if any(handle is None for handle in handles):
                    raise ValueError(f"DSpark layer field {name} was not loaded for every EP rank")
                self.tensors[name] = StackedDeviceTensor(
                    tuple(handles), (world, 3, *shape), tuple(range(world)))
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _copy(worker, destination, source, rank, *, offset=0):
        destination_ptr = destination.data_ptr
        if callable(destination_ptr):
            destination_ptr = destination_ptr()
        worker.copy_to(
            destination_ptr, source.data_ptr(), source.numel() * source.element_size(),
            dst_offset=offset, worker_id=rank,
        )

    def full(self, name):
        return self.tensors[name]

    def close(self):
        for rank, tensor in reversed(self._owned):
            self.worker.free_tensor(tensor, worker_id=rank)
        self._owned.clear()
        self.tensors.clear()


class ResidentDSparkEmbedding:
    """Stream the full target embedding into one device shard per EP rank."""

    def __init__(self, worker, model_dir, world, *, chunk_rows=256):
        from pypto.runtime import StackedDeviceTensor
        from .weight_loader import V41WeightLoader

        self.worker = worker
        self._owned = []
        first = V41WeightLoader(model_dir, tp_size=1, tp_rank=0, ep_size=world, ep_rank=0)
        vocab, hidden = first.config.vocab_size, first.config.hidden_size
        shards = []
        try:
            for rank in range(world):
                loader = first if rank == 0 else V41WeightLoader(
                    model_dir, tp_size=1, tp_rank=0, ep_size=world, ep_rank=rank)
                shard = worker.alloc_tensor((vocab, hidden), torch.bfloat16, worker_id=rank)
                self._owned.append((rank, shard))
                row_bytes = hidden * 2
                for start in range(0, vocab, chunk_rows):
                    stop = min(start + chunk_rows, vocab)
                    value = loader.load_rows("embed.weight", start, stop).weight
                    if value.dtype != torch.bfloat16:
                        raise ValueError("DSpark target embedding must be BF16")
                    destination_ptr = shard.data_ptr
                    if callable(destination_ptr):
                        destination_ptr = destination_ptr()
                    worker.copy_to(
                        destination_ptr, value.data_ptr(), value.numel() * value.element_size(),
                        dst_offset=start * row_bytes, worker_id=rank,
                    )
                    del value
                shards.append(shard)
        except BaseException:
            self.close()
            raise
        self.tensor = StackedDeviceTensor(tuple(shards), (world, vocab, hidden), tuple(range(world)))

    def close(self):
        for rank, tensor in reversed(self._owned):
            self.worker.free_tensor(tensor, worker_id=rank)
        self._owned.clear()
        self.tensor = None


class ResidentDSparkCache:
    """Private sliding-window KV pools for the three draft layers."""

    def __init__(self, worker, world, max_seq_len):
        from .dspark_resources import DSPARK_DRAFT_LAYERS, DSPARK_WINDOW
        from models.deepseek_v4_1_flash.config import WINDOW_CACHE_BYTES

        self.worker = worker
        blocks = DSPARK_MAX_BATCH_PER_DP * ((max_seq_len + DSPARK_WINDOW - 1) // DSPARK_WINDOW)
        self.window_cache = alloc_device_buffer(
            worker, (world, DSPARK_DRAFT_LAYERS, blocks, DSPARK_WINDOW, 1, WINDOW_CACHE_BYTES),
            torch.uint8, stacked=True)

    def close(self):
        for name in ("window_cache",):
            tensor = getattr(self, name, None)
            if tensor is not None:
                self.worker.free_stacked_tensor(tensor)
                setattr(self, name, None)


@dataclass(frozen=True)
class _DSparkDeviceState:
    slot_ids: object
    generations: object
    tokens: object
    meta: object

    def __getitem__(self, name):
        return getattr(self, {"state_tokens": "tokens", "state_meta": "meta"}.get(name, name))


class ResidentDSparkBuffers:
    """Persistent device state and scratch buffers for the DSpark composites."""

    def __init__(self, worker, host_resources):
        from .dspark_resources import DSparkHostState

        self.worker = worker
        self.host = host_resources
        self._owned = []
        try:
            self.state = DSparkHostState(*(
                self._alloc(value) for value in host_resources.state.values()
            ))
            self.decode_workspace = {
                name: self._alloc(value) for name, value in host_resources.decode_workspace.items()
            }
            self.prefill_workspace = {}
            for name, value in host_resources.prefill_workspace.items():
                # Common buffers are intentionally shared between the two
                # phases; dynamic prompt-axis buffers have distinct shapes.
                common = host_resources.decode_workspace.get(name)
                if common is value:
                    self.prefill_workspace[name] = self.decode_workspace[name]
                else:
                    self.prefill_workspace[name] = self._alloc(value)
            self.cache_tables = {name: self._alloc(value)
                                 for name, value in host_resources.cache_tables.items()}
            self.rope_tables = {name: self._alloc(value)
                                for name, value in host_resources.rope_tables.items()}
        except BaseException:
            self.close()
            raise

    def _alloc(self, host):
        # Counts have the public Tensor[EP] ABI, without a leading shard axis.
        # Their pre-fork shared storage is replicated by the runtime at dispatch.
        if host.ndim == 1:
            return host
        value = alloc_device_buffer(self.worker, tuple(host.shape), host.dtype, stacked=True)
        self._owned.append(value)
        self._copy(host, value)
        return value

    def _copy(self, host, device):
        if device is host:
            return
        for rank, shard in enumerate(device.shards):
            destination = shard.data_ptr
            if callable(destination):
                destination = destination()
            self.worker.copy_to(
                destination, host[rank].data_ptr(),
                host[rank].numel() * host.element_size(), dst_offset=0, worker_id=rank,
            )

    def sync(self, name, *, phase="decode"):
        host = self.host.for_phase(phase).get(name)
        device = (self.prefill_workspace if phase == "prefill" else self.decode_workspace).get(name)
        if host is None or device is None:
            raise ValueError(f"unknown DSpark {phase} workspace field: {name}")
        self._copy(host, device)

    def sync_state(self):
        for host, device in zip(self.host.state.values(), self.state.values(), strict=True):
            self._copy(host, device)

    def sync_cache_tables(self):
        for name, device in self.cache_tables.items():
            self._copy(self.host.cache_tables[name], device)

    def reset_slot(self, rank, slot, generation):
        host = self.host.state
        host.tokens[rank, slot].zero_()
        host.meta[rank, slot].zero_()
        host.meta[rank, slot, 1] = generation
        self._copy(host.tokens, self.state.tokens)
        self._copy(host.meta, self.state.meta)

    def close(self):
        for tensor in reversed(self._owned):
            self.worker.free_stacked_tensor(tensor)
        self._owned.clear()


# resident_resources
class _DSparkOrchestratorRunner:
    """Adapt the lib's callable stage runner to resident serving resources."""

    def __init__(self, resources, phase):
        self.resources = resources
        self.expected = PREFILL_STAGE_ORDER if phase == "prefill" else DECODE_STAGE_ORDER
        self.index = 0

    def __call__(self, _entry, arguments):
        if self.index >= len(self.expected):
            raise RuntimeError("V4.1 DSpark lib emitted more stages than its ABI")
        stage = self.expected[self.index]
        self.index += 1
        self.resources.dispatch_dspark(stage, arguments)

    def finish(self):
        if self.index != len(self.expected):
            raise RuntimeError(
                "V4.1 DSpark lib emitted an incomplete stage sequence: "
                f"{self.index}/{len(self.expected)}"
            )


class V41ResidentResources:
    """Own resident buffers and their shared host mappings until worker close.

    This is the resource lifetime for a production binding, not an executable
    binding by itself: the caller still has to lower every prefill/decode ABI
    argument and dispatch the compiled layer programs.
    """

    def __init__(self, programs, run_config, *, weight_bank, weight_param_names,
                 cache_plan, state_reset, page_resets, input_host, final_host,
                 prefill_host, topology,
                 engram_host=None,
                 decode_input_host=None, decode_host=None, worker_factory=make_segment_worker,
                 prefill_program=None, prefill_param_names=None,
                 dspark_programs=None, dspark_param_names=None, max_seq_len=None,
                 dspark_weight_contract=None, dspark_lowering_provider=None,
                 dspark_host_resources=None):
        weight_bank.finish(weight_param_names)
        if decode_host is not None and (decode_input_host is None or
                decode_host.topology != decode_input_host.topology):
            raise ValueError("decode metadata and input must share one TP-local topology")
        inherited = (input_host.inherited_host_tensors
                     + (() if decode_input_host is None else decode_input_host.inherited_host_tensors)
                     + prefill_host.inherited_host_tensors
                     + (() if decode_host is None else decode_host.inherited_host_tensors)
                     + (() if engram_host is None else engram_host.inherited_host_tensors)
                     + final_host.all_host_tensors
                     + (() if dspark_host_resources is None
                        else tuple(dspark_host_resources.state.values())
                        + tuple(dspark_host_resources.decode_workspace.values())
                        + tuple(dspark_host_resources.prefill_workspace.values())
                        + tuple(dspark_host_resources.cache_tables.values())
                        + tuple(dspark_host_resources.rope_tables.values()))
                     + (state_reset, *page_resets.values()))
        if any(not tensor.is_shared() for tensor in inherited):
            raise ValueError("all inherited V4.1 host buffers must be shared before worker creation")
        if not getattr(weight_bank, "prepacked", False) and any(
                not tensor.is_shared() for tensor in weight_bank.tensors.values()):
            raise ValueError("all inherited V4.1 host buffers must be shared before worker creation")

        self.worker = worker_factory(programs, run_config,
                                     tuple(weight_bank.tensors.values()) + inherited)
        self.run_config = run_config
        self.host_resources = (weight_bank, input_host, decode_input_host, prefill_host, decode_host,
                               final_host, dspark_host_resources, state_reset, page_resets)
        self.prefill_host = prefill_host
        self.engram_host = engram_host
        self.decode_host = decode_host
        self.dspark_host_resources = dspark_host_resources
        self.dspark_weights = None
        self.dspark_embedding = None
        self.dspark_cache = None
        self.dspark_buffers = None
        self.ledger = None
        self._dspark_generations = {}
        self._dspark_drafter_epochs = {}
        self._owned = []
        self._step = None
        self._inputs = None
        self._output_rows = None
        self._group_counts = None
        self._prefill_fwd = None
        self._decode_fwd = None
        self._decode_binding = None
        self._weight_bank = weight_bank
        self._weight_param_names = tuple(weight_param_names)
        self._prefill_binding = (prefill_program, prefill_param_names, run_config, prefill_host.topology)
        self.dspark_programs = dict(dspark_programs or {})
        self.dspark_param_names = dict(dspark_param_names or {})
        if self.dspark_programs:
            if type(max_seq_len) is not int or max_seq_len <= 0:
                raise ValueError("DSpark resources require a positive max_seq_len")
            self.dspark = V41DSparkRuntime(self, max_seq_len=max_seq_len)
        else:
            self.dspark = None
        self.weights = None
        self.dspark_weight_contract = dspark_weight_contract
        if dspark_lowering_provider is not None and not (
                hasattr(dspark_lowering_provider, "lower_prefill")
                and hasattr(dspark_lowering_provider, "lower_decode")):
            raise TypeError(
                "V4.1 DSpark lowering provider must expose lower_prefill and lower_decode"
            )
        self.dspark_lowering_provider: DSparkLoweringProvider | None = dspark_lowering_provider
        self.failed = False
        self.closed = False
        try:
            self.weights = ResidentFullDecodeWeights(
                self.worker, weight_bank, weight_param_names)
            self.input = self._own(InputStateRuntime(self.worker, input_host))
            self.decode_input = (None if decode_input_host is None else
                                 self._own(InputStateRuntime(self.worker, decode_input_host)))
            self.final = self._own(FinalOutputWeights(self.worker, topology, final_host))
            self.engram = (None if engram_host is None else self._own(
                ResidentEngramWeights(self.worker, engram_host)))
            if callable(cache_plan):
                slots, plan_for_slots = cache_plan(self.worker)
                cache = allocate_cache_with_retry(
                    self.worker, slots, plan_for_slots, state_reset, page_resets)
            else:
                cache = DecodeFullCachePools(self.worker, cache_plan, state_reset, page_resets)
            self.cache = self._own(cache)
            if self.dspark_host_resources is not None:
                if dspark_weight_contract is None:
                    raise MissingCompositeInterface(
                        "V4.1 DSpark resources require the deferred checkpoint contract"
                    )
                self.dspark_weights = self._own(
                    ResidentDSparkWeights(self.worker, dspark_weight_contract.loader.model_dir,
                                          topology.world)
                )
                self.dspark_embedding = self._own(
                    ResidentDSparkEmbedding(self.worker, dspark_weight_contract.loader.model_dir,
                                            topology.world)
                )
                self.dspark_cache = self._own(
                    ResidentDSparkCache(self.worker, topology.world, max_seq_len)
                )
                self.dspark_buffers = self._own(
                    ResidentDSparkBuffers(self.worker, self.dspark_host_resources)
                )
            if prefill_program is not None:
                if prefill_param_names is None or prefill_host.full_hidden_size is None:
                    raise ValueError("full prefill binding requires its ABI and metadata")
                self._activate_prefill()
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

    def dspark_state_generation(self, rank: int, slot: int) -> int:
        """Return the generation currently reserved for one local state slot."""
        return self._dspark_generations.get((rank, slot), 1)

    def _build_dspark_target_model(self, resources):
        from .dspark_resources import _TARGET_DYNAMIC_KEYS

        if self._decode_fwd is None:
            self._activate_decode()
        target = {name: value for name, value in self._decode_fwd.arguments.items()
                  if name not in _TARGET_DYNAMIC_KEYS}
        target["embedding_weight"] = self.dspark_embedding.tensor
        for name in ("logit_row_indices", "attention_num_tokens", "moe_num_tokens", "state_block_table"):
            target[name] = resources.decode_workspace[name]
        target["c2a_index_block_table"] = resources.cache_tables["target_c2a"]
        target["c1a_index_block_table"] = resources.cache_tables["target_c1a"]
        return target

    def _resolve_dspark_state(self, request_id):
        """Resolve the stable scheduler slot before/after bootstrap."""
        if self.dspark is not None and request_id in self.dspark.states.requests:
            state = self.dspark.states.get(request_id)
            return state.slot, state.generation
        if self.ledger is None or request_id not in self.ledger.owners:
            raise MissingCompositeInterface(
                f"V4.1 DSpark request {request_id!r} has no scheduler state"
            )
        owner = self.ledger.owners[request_id]
        generation = self._dspark_generations.setdefault((owner.partition, owner.slot), 1)
        return owner.slot, generation

    @staticmethod
    def _host_tensor(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        return torch.as_tensor(value)

    def _stage_dspark_tables(self, resources, batch, request_ranks, request_rows):
        tables = resources.cache_tables
        for value in tables.values():
            value.fill_(-1)
        if not getattr(batch, "block_ids_by_group", None):
            raise MissingCompositeInterface(
                "V4.1 DSpark requires scheduler block_ids_by_group for all cache families"
            )
        families = self.cache.plan.family_groups
        window_name, c2a_name, c1a_name = families.window, families.c2a, families.c1a
        for index, (rank, row) in enumerate(zip(request_ranks, request_rows, strict=True)):
            if not 0 <= rank < resources.layout.dp_size or not 0 <= row < DSPARK_MAX_BATCH_PER_DP:
                raise ValueError("DSpark batch row exceeds the selected DP topology")
            groups = batch.block_ids_by_group[index]
            for name, target in ((window_name, tables["target_window"]),
                                 (c1a_name, tables["target_c1a"])):
                pages = tuple(groups.get(name, ()))
                if not pages:
                    raise MissingCompositeInterface(f"DSpark request is missing {name} pages")
                if len(pages) > target.shape[-1]:
                    raise ValueError(f"DSpark {name} page table exceeds its ABI capacity")
                target[rank, row, :len(pages)] = torch.tensor(pages, dtype=torch.int32)
            pages = tuple(groups.get(c2a_name, ()))
            if not pages:
                raise MissingCompositeInterface(f"DSpark request is missing {c2a_name} pages")
            if len(pages) > tables["target_c2a"].shape[-1]:
                raise ValueError("DSpark C2A page table exceeds its ABI capacity")
            tables["target_c2a"][rank, row, :len(pages)] = torch.tensor(pages, dtype=torch.int32)
            draft = tables["draft"]
            slot, _generation = self._resolve_dspark_state(batch.request_ids[index])
            if not 0 <= slot < DSPARK_MAX_BATCH_PER_DP:
                raise ValueError("DSpark draft cache slot exceeds its capacity")
            # Target page IDs belong to another allocation. Draft pages follow
            # the stable state slot, even when the request's batch row changes.
            pages_per_slot = draft.shape[-1]
            draft[rank, :, row] = torch.arange(
                slot * pages_per_slot, (slot + 1) * pages_per_slot, dtype=torch.int32)

    def _stage_dspark_decode(self, resources, batch):
        request_ids = tuple(batch.request_ids)
        partitions = tuple(int(value) for value in batch.cache_partitions)
        if len(partitions) != len(request_ids):
            raise ValueError("DSpark decode partitions must match request order")
        rows_by_rank = {}
        request_rows = []
        for rank in partitions:
            row = rows_by_rank.get(rank, 0)
            rows_by_rank[rank] = row + 1
            request_rows.append(row)
        workspace = resources.decode_workspace
        for value in workspace.values():
            if isinstance(value, torch.Tensor):
                value.zero_()
        workspace["input_ids"].zero_()
        workspace["target_positions"].fill_(-1)
        workspace["active_widths"].zero_()
        workspace["logit_row_indices"].fill_(-1)
        workspace["state_block_table"].fill_(-1)
        tokens = self._host_tensor(batch.token_ids).reshape(-1).tolist()
        # DecodeBatch.seq_lens includes the sampled anchor token, as in
        # decode_requests; Engram resumes at the preceding committed length.
        positions = [int(end) - 1 for end in self._host_tensor(batch.seq_lens).reshape(-1).tolist()]
        if len(tokens) != len(request_ids) or len(positions) != len(request_ids):
            raise ValueError("DSpark decode token and position rows must match requests")
        state = resources.state
        engram_requests = []
        for index, (request_id, rank, row) in enumerate(
                zip(request_ids, partitions, request_rows, strict=True)):
            if not 0 <= rank < resources.layout.dp_size or row >= DSPARK_MAX_BATCH_PER_DP:
                raise ValueError("DSpark batch row exceeds the selected DP topology")
            request_state = self.dspark.states.get(request_id)
            if positions[index] != request_state.anchor:
                raise ValueError("DSpark scheduler position disagrees with the committed anchor")
            if request_state.anchor_token is not None and int(tokens[index]) != request_state.anchor_token:
                raise ValueError("DSpark scheduler token disagrees with the committed anchor")
            if len(request_state.draft_ids) not in (0, DSPARK_SPECULATIVE_TOKENS):
                raise MissingCompositeInterface(
                    f"DSpark request {request_id!r} has invalid draft state")
            active_tokens = (int(tokens[index]), *request_state.draft_ids)
            width = len(active_tokens)
            if not 0 <= positions[index] < positions[index] + width <= request_state.position_limit:
                raise ValueError("DSpark active target rows exceed the request position limit")
            slot, generation = self._resolve_dspark_state(request_id)
            state.slot_ids[rank, row] = slot
            state.generations[rank, row] = generation
            workspace["input_ids"][rank, row, :width] = torch.tensor(active_tokens, dtype=torch.int64)
            workspace["target_positions"][rank, row, :width] = torch.arange(
                int(positions[index]), int(positions[index]) + width,
                dtype=torch.int32)
            workspace["active_widths"][rank, row] = width
            workspace["zero_prefill_tokens"][rank, row] = 0
            workspace["state_block_table"][rank, row, 0] = slot
            workspace["logit_row_indices"][rank, row] = row * DSPARK_ROWS_PER_REQUEST + width - 1
            workspace["attention_num_tokens"][rank] = (row + 1) * DSPARK_ROWS_PER_REQUEST
            workspace["moe_num_tokens"][rank] = (row + 1) * DSPARK_ROWS_PER_REQUEST
            engram_requests.append(RequestSlice(
                request_id, rank, slot, int(positions[index]),
                active_tokens, request_state.prompt_length,
                batch.block_ids_by_group[index],
            ))
        self._stage_dspark_tables(resources, batch, partitions, request_rows)
        if self.engram_host is not None:
            self.engram_host.stage_lookup(
                self.decode_host.fields["engram_lookup"],
                ForwardStep("decode", tuple(engram_requests), 0), self.decode_host.topology,
                row_stride=DSPARK_ROWS_PER_REQUEST)
        return tuple(partitions), tuple(request_rows)

    def _retain_dspark_prefill(self, step):
        """Keep only the drafter's visible target taps across prefill chunks."""
        resources = self.dspark_host_resources
        if resources is None:
            return
        source = self.prefill_host.fields.get("dspark_target_hidden")
        if source is None:
            raise MissingCompositeInterface("target prefill did not expose dspark_target_hidden")
        retained = resources.prefill_workspace["prefill_target_hidden"]
        capacity = min(DSPARK_WINDOW, retained.shape[1])
        progress = getattr(self, "_dspark_prefill_progress", {})
        offsets = [0] * resources.layout.dp_size
        for request in step.requests:
            rank = request.partition
            offset = offsets[rank]
            size = len(request.token_ids)
            offsets[rank] += size
            if request.start and progress.get(rank) != (request.request_id, request.start):
                raise ValueError("DSpark prefill taps must continue the same request")
            if size >= capacity:
                retained[rank, :capacity].copy_(source[rank, offset + size - capacity:offset + size])
            else:
                previous = min(request.start, capacity - size)
                if previous:
                    old_end = min(request.start, capacity)
                    retained[rank, :previous].copy_(retained[rank, old_end - previous:old_end].clone())
                retained[rank, previous:previous + size].copy_(source[rank, offset:offset + size])
            progress[rank] = (request.request_id, request.end)
        self._dspark_prefill_progress = progress

    def _stage_dspark_prefill(self, resources, batch, request_ids, sampled_token_ids):
        if len(request_ids) != 1:
            raise ValueError("V4.1 DSpark bootstrap currently accepts one terminal request")
        request_id = str(request_ids[0])
        partitions = tuple(int(value) for value in getattr(batch, "cache_partitions", (0,)))
        rank = partitions[0] if partitions else 0
        slot, generation = self._resolve_dspark_state(request_id)
        workspace = resources.prefill_workspace
        for name, value in workspace.items():
            if name != "prefill_target_hidden" and isinstance(value, torch.Tensor):
                value.zero_()
        workspace["sampled_rows"].fill_(-1)
        workspace["prefill_positions"].fill_(-1)
        workspace["prefill_request_ids"].fill_(-1)
        prompt_length = int(batch.prompt_lens[0])
        if not 0 < prompt_length <= resources.layout.max_seq_len:
            raise ValueError("DSpark prefill prompt exceeds the configured context capacity")
        if getattr(self, "_dspark_prefill_progress", {}).get(rank) != (request_id, prompt_length):
            raise ValueError("DSpark bootstrap requires all target prefill chunks to complete")
        retained_count = min(prompt_length, resources.layout.max_prefill_tokens, DSPARK_WINDOW)
        if retained_count < min(prompt_length, DSPARK_WINDOW):
            raise ValueError("DSpark prefill buffer cannot retain the visible draft context")
        prefix_start = prompt_length - retained_count
        workspace["prompt_lengths"][rank, 0] = prompt_length
        workspace["position_limits"][rank, 0] = resources.layout.max_seq_len
        workspace["sampled_rows"][rank, 0] = 0
        workspace["next_prefill_tokens"][rank, 0] = int(sampled_token_ids[0])
        workspace["prefill_counts"][rank, 0] = retained_count
        workspace["first_query_positions"][rank, 0] = prompt_length
        workspace["prefill_positions"][rank, :retained_count] = torch.arange(
            prefix_start, prompt_length, dtype=torch.int32)
        workspace["prefill_request_ids"][rank, :retained_count] = 0
        workspace["prefill_sampled_ids"][rank, 0, 0] = int(sampled_token_ids[0])
        groups = getattr(batch, "block_ids_by_group", ())
        if len(groups) != 1:
            raise ValueError("DSpark bootstrap requires one cache allocation")
        self._stage_dspark_tables(resources, batch, (rank,), (0,))
        return request_id, slot, generation

    def _dspark_payload(self, *, model, batch, phase, resources, **extra):
        from .dspark_resources import _DRAFT_MODEL_KEYS, _MARKOV_MODEL_KEYS

        if resources is not self.dspark_host_resources:
            raise ValueError("DSpark payload resources do not belong to this resident worker")
        if self.dspark_weights is None or self.dspark_embedding is None or self.dspark_cache is None:
            raise MissingCompositeInterface("resident DSpark weights/cache are not initialized")
        draft = {name: self.dspark_weights.full(name) for name in _DRAFT_MODEL_KEYS
                  if name in self.dspark_weights.tensors}
        draft["embedding_weight"] = self.dspark_embedding.tensor
        draft["window_cache"] = self.dspark_cache.window_cache
        heads = self.dspark_weights
        markov = {name: heads.full(name) for name in _MARKOV_MODEL_KEYS
                  if name in heads.tensors}
        markov["lm_head_weight"] = self.final.head_weight
        if phase == "decode":
            ranks, rows = self._stage_dspark_decode(resources, batch)
            workspace = resources.decode_workspace
            target = self._build_dspark_target_model(resources)
            target["logits"].zero_()
            target["sampled_ids"].fill_(-1)
            outputs = {name: workspace[name] for name in (
                "accepted_counts", "accepted_ids", "last_sampled", "draft_ids", "confidences")}
            outputs["request_rows"] = rows
            return DSparkLoweredInputs(
                state=resources.state, workspace=workspace, target_model=target,
                draft_model=draft, markov_model=markov,
                cache_tables=resources.cache_tables, rope_tables=resources.rope_tables,
                outputs=outputs, request_ranks=ranks,
            )
        request_id, slot, generation = self._stage_dspark_prefill(
            resources, batch, extra["request_ids"], extra["sampled_token_ids"])
        workspace = resources.prefill_workspace
        outputs = {"draft_ids": workspace["draft_ids"], "request_rows": (0,)}
        return DSparkLoweredInputs(
            state=resources.state, workspace=workspace, draft_model=draft,
            markov_model=markov, cache_tables=resources.cache_tables,
            rope_tables=resources.rope_tables, outputs=outputs,
            request_ranks=(int(getattr(batch, "cache_partitions", [0])[0] or 0),),
        )

    def lower_dspark_prefill(self, *, model, batch, request_ids, sampled_token_ids):
        """Lower a terminal prefill batch through the serving-owned provider."""
        provider = self.dspark_lowering_provider
        if provider is None:
            raise MissingCompositeInterface(
                "V4.1 DSpark prefill lowering provider is not configured; "
                "serving must bind target hidden, state, cache, workspace and outputs"
            )
        dispatch = provider.lower_prefill(
            model=model, batch=batch, request_ids=tuple(request_ids),
            sampled_token_ids=tuple(int(token) for token in sampled_token_ids),
        )
        if not isinstance(dispatch, DSparkPrefillDispatch):
            if isinstance(dispatch, Mapping):
                dispatch = DSparkPrefillDispatch.from_batch(
                    type("DSparkBatch", (), {"dspark_stages": dispatch})()
                )
            else:
                raise TypeError("DSpark prefill lowering provider must return DSparkPrefillDispatch")
        if dispatch.lowered is None:
            raise MissingCompositeInterface(
                "V4.1 DSpark prefill lowering provider returned no complete lowered inputs"
            )
        return dispatch

    def lower_dspark_decode(self, *, model, batch):
        """Lower one decode batch through the serving-owned provider."""
        provider = self.dspark_lowering_provider
        if provider is None:
            raise MissingCompositeInterface(
                "V4.1 DSpark decode lowering provider is not configured; "
                "serving must bind target/draft/Markov state and outputs"
            )
        dispatch = provider.lower_decode(model=model, batch=batch)
        if not isinstance(dispatch, DSparkDecodeDispatch):
            if isinstance(dispatch, Mapping):
                dispatch = DSparkDecodeDispatch.from_batch(
                    type("DSparkBatch", (), {"dspark_stages": dispatch})()
                )
            else:
                raise TypeError("DSpark decode lowering provider must return DSparkDecodeDispatch")
        if dispatch.lowered is None:
            raise MissingCompositeInterface(
                "V4.1 DSpark decode lowering provider returned no complete lowered inputs"
            )
        return dispatch

    def _activate_prefill(self):
        if self._prefill_fwd is not None:
            return
        program, names, run_config, topology = self._prefill_binding
        if program is None:
            raise RuntimeError("full prefill was not registered")
        self._prefill_fwd = PrefillForwardSegment(
            self.worker, program, names, run_config, topology,
            self.weights, self.prefill_host, self.cache, self.final, self.engram)

    def _activate_decode(self):
        if self._decode_fwd is not None:
            return
        if self._decode_binding is None:
            raise RuntimeError("full decode was not registered")
        program, names, run_config, counts_are_tensor = self._decode_binding
        self._decode_fwd = DecodeForwardSegment(
            self.worker, program, names, run_config, self.decode_host.topology,
            counts_are_tensor, self.weights, self.cache, self.decode_host, self.final, self.engram)

    def begin_step(self, embeddings, step):
        if self.closed or self._step is not None:
            raise RuntimeError("V4.1 worker is closed or a step is already active")
        if step.phase == "prefill" and self._prefill_binding[0] is not None:
            self._activate_prefill()
        elif step.phase == "decode":
            self._activate_decode()
        dispatch_step = step
        if step.phase == "decode":
            dispatch_step, embeddings, padding_partitions = pad_single_decode(
                step, embeddings, self.cache.plan)
            for partition in padding_partitions:
                self.cache.reset_padding(partition)
        host = self.prefill_host if step.phase == "prefill" else self.decode_host
        counts = (host.stage_full(dispatch_step, embeddings, self.engram_host)
                  if step.phase == "prefill" and self._prefill_fwd is not None else
                  host.stage(dispatch_step, self.engram_host) if host is not None else None)
        input_runtime = self.decode_input if step.phase == "decode" else self.input
        if input_runtime is None:
            raise ValueError("decode requires its own TP-local input capacity")
        state, inputs = input_runtime.stage(embeddings, dispatch_step)
        if dispatch_step is not step:
            inputs = replace(inputs, request_last_rows=inputs.request_last_rows[:len(step.requests)])
        self._output_rows = prepare_final_output_rows(
            inputs, self.final.host.logit_row_indices,
            local_capacity=inputs.row_indices.shape[1],
        )
        self.final.host.logits.zero_()
        self.final.host.sampled_ids.fill_(-1)
        self._step, self._inputs = step, inputs
        self._group_counts = counts
        return state

    def run_prefill_fwd(self, state, step):
        if (self.closed or step is not self._step or step.phase != "prefill"
                or self._prefill_fwd is None):
            raise ValueError("full prefill requires the active prefill step")
        return self._prefill_fwd.run()

    def bind_decode_fwd(self, program, param_names, run_config, counts_are_tensor):
        if (self.closed or self._decode_binding is not None or self._step is not None
                or self.decode_host is None):
            raise RuntimeError("full decode requires an idle worker and decode metadata")
        self._decode_binding = (program, tuple(param_names), run_config, counts_are_tensor)
        if self._prefill_binding[0] is None:
            self._activate_decode()
        return self._decode_fwd

    def run_decode_fwd(self, state, step):
        if (self.closed or step is not self._step or step.phase != "decode"
                or self._decode_fwd is None):
            raise ValueError("full decode requires the active decode step")
        return self._decode_fwd.run(state, group_counts=self._group_counts)

    def dispatch_dspark(self, stage, arguments):
        """Dispatch one compiled DSpark composite with an exact ABI check.

        The request scheduler supplies the resident tensors and state for a
        stage.  Keeping this narrow boundary lets the lib's
        ``decode_fwd_dspark`` orchestrator remain the owner of stage ordering;
        serving does not rebuild the drafter or Markov graph from small ops.
        """
        if self.closed or self.failed:
            raise RuntimeError("cannot dispatch DSpark on a closed or failed worker")
        try:
            program = self.dspark_programs[stage]
            param_names = self.dspark_param_names[stage]
        except KeyError as exc:
            raise MissingCompositeInterface(f"DSpark stage is not compiled: {stage}") from exc
        supplied = set(arguments)
        required = set(param_names)
        if supplied != required:
            raise ValueError(
                f"DSpark stage {stage} ABI mismatch: "
                f"missing={sorted(required - supplied)} extra={sorted(supplied - required)}"
            )
        bound = arguments
        reset_windows = False
        if stage in ("dspark_drafter", "decode"):
            module = importlib.import_module("models.deepseek_v4_1_flash.dspark_drafter")
            hidden_name = "draft_head_hidden" if stage == "decode" else "head_hidden"
            batch = int(arguments[hidden_name].shape[1])
            chunks = (batch + module.DSPARK_CHUNK_REQUESTS - 1) // module.DSPARK_CHUNK_REQUESTS
            stride = chunks * module.DSPARK_DRAFT_LAYERS
            key = id(program.compiled)
            epoch = self._dspark_drafter_epochs.get(key, 1)
            # The pinned A5 int32 AtomicAdd signal stops advancing at 2048.
            # Reuse the prefill epoch bound before a three-layer call crosses it.
            reset_windows = epoch + stride - 1 > 512
            if reset_windows:
                epoch = 1
            if epoch + stride - 1 > (2**31 - 1) // 2:
                raise OverflowError("DSpark communication epoch exhausted; recreate worker")
            # These windows persist across both bootstrap and decode. Allocate
            # epochs at dispatch, so repeated lowering cannot consume them and
            # replaying an already lowered payload cannot reuse old signals.
            bound = dict(arguments, moe_epoch=epoch)
        if reset_windows:
            previous_reset = self.worker._reset_persistent_windows
            self.worker._reset_persistent_windows = True
        try:
            self.worker.run(program.compiled,
                            *(bound[name] for name in param_names), config=self.run_config)
        finally:
            if reset_windows:
                self.worker._reset_persistent_windows = previous_reset
        if stage in ("dspark_drafter", "decode"):
            self._dspark_drafter_epochs[key] = epoch + stride

    def _run_dspark_orchestrator(self, lowered: DSparkLoweredInputs, *, phase: str) -> None:
        """Run the lib-owned orchestration against resident compiled stages."""
        if not isinstance(lowered, DSparkLoweredInputs):
            raise TypeError("DSpark orchestration requires lowered serving inputs")
        try:
            lowered.validate(phase=phase)
        except (TypeError, ValueError) as exc:
            raise MissingCompositeInterface(str(exc)) from exc
        required_mappings = (
            ("draft_model", lowered.draft_model),
            ("markov_model", lowered.markov_model),
            ("cache_tables", lowered.cache_tables),
            ("rope_tables", lowered.rope_tables),
        )
        missing = [name for name, mapping in required_mappings if not mapping]
        if missing:
            raise MissingCompositeInterface(
                "V4.1 DSpark lowering is missing mappings: " + ", ".join(missing)
            )
        try:
            module = importlib.import_module("models.deepseek_v4_1_flash.decode_fwd_dspark")
            function = (module.prefill_bootstrap_dspark if phase == "prefill"
                        else module.decode_fwd_dspark)
        except (ImportError, AttributeError) as exc:
            raise MissingCompositeInterface(
                "V4.1 DSpark lib orchestration entry is unavailable"
            ) from exc
        runner = _DSparkOrchestratorRunner(self, phase)
        try:
            kwargs = dict(
                moe_epoch=lowered.moe_epoch,
                logits_epoch=lowered.logits_epoch,
            )
            if phase == "prefill":
                function(
                    runner, lowered.state, lowered.workspace, lowered.draft_model,
                    lowered.markov_model, lowered.cache_tables, lowered.rope_tables,
                    **kwargs,
                )
            else:
                function(
                    runner, lowered.state, lowered.workspace, lowered.target_model,
                    lowered.draft_model, lowered.markov_model, lowered.cache_tables,
                    lowered.rope_tables, **kwargs,
                )
            runner.finish()
        except MissingCompositeInterface:
            raise
        except BaseException as exc:
            self.failed = True
            raise RuntimeError(f"V4.1 DSpark {phase} orchestration failed") from exc

    def _read_dspark_output(self, lowered, name):
        """Return an owned CPU copy of one device-produced DSpark output."""
        value = lowered.outputs.get(name, lowered.workspace.get(name))
        if value is None:
            raise MissingCompositeInterface(
                f"V4.1 DSpark output lowering is missing {name!r}"
            )
        if callable(value):
            value = value()
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu":
                raise MissingCompositeInterface(
                    f"V4.1 DSpark output {name!r} needs a serving readback buffer"
                )
            return value.detach().clone()
        from pypto.runtime import StackedDeviceTensor

        shards = getattr(value, "shards", None)
        if shards is None or not isinstance(value, StackedDeviceTensor):
            raise TypeError(f"V4.1 DSpark output {name!r} is not a tensor")
        result = torch.empty(tuple(value.shape), dtype=value.dtype).share_memory_()
        for rank, shard in enumerate(shards):
            destination = result[rank]
            self.worker.copy_from(
                destination.data_ptr(), shard.data_ptr,
                destination.numel() * destination.element_size(),
                src_offset=0, worker_id=value.worker_ids[rank],
            )
        return result

    @staticmethod
    def _dspark_output_row(value, *, rank, row, kind):
        vector_output = kind in ("accepted_ids", "draft_ids")
        if value.ndim == 1:
            # Single-request lowering may expose a token vector directly.
            if vector_output:
                if row != 0:
                    raise ValueError(f"DSpark {kind} row is outside the output")
                return value
            if row >= value.shape[0]:
                raise ValueError(f"DSpark {kind} row is outside the output")
            return value[row]
        if vector_output and value.ndim == 2:
            if row >= value.shape[0]:
                raise ValueError(f"DSpark {kind} row is outside the output")
            return value[row]
        if value.ndim >= 2:
            if rank >= value.shape[0] or row >= value.shape[1]:
                raise ValueError(f"DSpark {kind} rank/row is outside the output")
            return value[rank, row]
        raise ValueError("DSpark output tensor must have at least one dimension")

    @staticmethod
    def _validate_dspark_token_row(value, *, kind, minimum):
        if getattr(value, "ndim", None) != 1 or value.shape[0] < minimum:
            raise ValueError(
                f"DSpark {kind} output must be a token vector with at least "
                f"{minimum} entries"
            )

    def _dspark_request_rows(self, lowered, count):
        ranks = lowered.request_ranks
        if ranks is None:
            ranks = tuple(range(count))
        if len(ranks) != count or any(type(rank) is not int or rank < 0 for rank in ranks):
            raise ValueError("DSpark request_ranks must match request order")
        explicit = lowered.outputs.get("request_rows")
        if explicit is None:
            explicit = lowered.workspace.get("request_rows")
        if explicit is not None:
            if len(explicit) != count:
                raise ValueError("DSpark request_rows must match request order")
            if any(type(row) is not int or row < 0 for row in explicit):
                raise ValueError("DSpark request_rows must be nonnegative integers")
            rows = tuple(explicit)
        else:
            next_row = {}
            rows = []
            for rank in ranks:
                rows.append(next_row.get(rank, 0))
                next_row[rank] = rows[-1] + 1
            rows = tuple(rows)
        return tuple(ranks), rows

    def run_dspark_prefill(self, request_id, *, slot, generation, prompt_length,
                           anchor, stages, lowered=None):
        if self.dspark is None:
            raise MissingCompositeInterface("V4.1 DSpark programs were not registered")
        if lowered is None:
            raise MissingCompositeInterface(
                "V4.1 DSpark prefill output lowering is not implemented: "
                "serving must bind target hidden/sample outputs and publish draft state"
            )
        try:
            self._run_dspark_orchestrator(lowered, phase="prefill")
            draft_ids = self._read_dspark_output(lowered, "draft_ids")
            rank, row = self._dspark_request_rows(lowered, 1)
            draft_ids = self._dspark_output_row(
                draft_ids, rank=rank[0], row=row[0], kind="draft_ids")
            drafts = []
            if anchor + DSPARK_SPECULATIVE_TOKENS < self.dspark.states.position_limit:
                self._validate_dspark_token_row(
                    draft_ids, kind="draft_ids", minimum=DSPARK_SPECULATIVE_TOKENS)
                drafts = [int(token) for token in draft_ids.tolist()]
            state = self.dspark.record_bootstrap(
                request_id, slot=slot, generation=generation,
                prompt_length=prompt_length, anchor=anchor,
                draft_ids=drafts,
                anchor_token=int(lowered.workspace["next_prefill_tokens"][rank[0], row[0]]),
            )
            return state
        except BaseException:
            self.failed = True
            raise

    def run_dspark_decode(self, request_ids, *, stages, accepted_counts=None,
                          last_sampled=None, draft_ids=None, lowered=None,
                          request_ranks=None):
        if self.dspark is None:
            raise MissingCompositeInterface("V4.1 DSpark programs were not registered")
        if lowered is None:
            raise MissingCompositeInterface(
                "V4.1 DSpark decode output lowering is not implemented: "
                "serving must bind target decode, acceptance, draft/Markov outputs "
                "and return DecodeResult"
            )
        if request_ranks is not None and lowered.request_ranks is None:
            lowered = replace(lowered, request_ranks=tuple(request_ranks))
        try:
            for request_id in request_ids:
                self.dspark.states.get(request_id)
            self._run_dspark_orchestrator(lowered, phase="decode")
            counts_tensor = self._read_dspark_output(lowered, "accepted_counts")
            accepted_tensor = self._read_dspark_output(lowered, "accepted_ids")
            sampled_tensor = self._read_dspark_output(lowered, "last_sampled")
            drafts_tensor = self._read_dspark_output(lowered, "draft_ids")
            ranks, rows = self._dspark_request_rows(lowered, len(request_ids))
            accepted, counts, sampled, drafts, proposed_counts = [], [], [], [], []
            for request_id, rank, row in zip(request_ids, ranks, rows, strict=True):
                count = int(self._dspark_output_row(
                    counts_tensor, rank=rank, row=row, kind="accepted_counts"))
                if not 1 <= count <= 6:
                    raise ValueError(f"DSpark accepted count for {request_id!r} is outside [1, 6]")
                ids = self._dspark_output_row(
                    accepted_tensor, rank=rank, row=row, kind="accepted_ids")
                self._validate_dspark_token_row(
                    ids, kind="accepted_ids", minimum=DSPARK_ROWS_PER_REQUEST)
                token_ids = [int(token) for token in ids[:count].tolist()]
                sampled_token = int(self._dspark_output_row(
                    sampled_tensor, rank=rank, row=row, kind="last_sampled"))
                state = self.dspark.states.get(request_id)
                proposed_counts.append(len(state.draft_ids))
                state.validate_accept(count, sampled_token)
                draft = []
                if state.anchor + count + DSPARK_SPECULATIVE_TOKENS < state.position_limit:
                    draft_row = self._dspark_output_row(
                        drafts_tensor, rank=rank, row=row, kind="draft_ids")
                    self._validate_dspark_token_row(
                        draft_row, kind="draft_ids", minimum=DSPARK_SPECULATIVE_TOKENS)
                    draft = [int(token) for token in draft_row[:DSPARK_SPECULATIVE_TOKENS].tolist()]
                accepted.append(token_ids)
                counts.append(count)
                sampled.append(sampled_token)
                drafts.append(draft)
            self.dspark.apply_decode_outputs(
                request_ids, accepted_counts=counts, last_sampled=sampled, draft_ids=drafts,
            )
            draft_counts = lowered.outputs.get("draft_counts", lowered.workspace.get("draft_counts"))
            if draft_counts is None:
                draft_count_values = proposed_counts
            else:
                draft_counts_tensor = self._read_dspark_output(lowered, "draft_counts")
                draft_count_values = [int(self._dspark_output_row(
                    draft_counts_tensor, rank=rank, row=row, kind="draft_counts"))
                    for rank, row in zip(ranks, rows, strict=True)]
            return DecodeResult(
                hidden_states=None, logits=None,
                accepted_token_ids=accepted,
                num_draft_tokens=draft_count_values,
            )
        except BaseException:
            self.failed = True
            raise

    def finish_step(self, state, step):
        if self.closed or step is not self._step:
            raise ValueError("final output must use the active request step")
        try:
            if step.phase == "prefill" and self.dspark_host_resources is not None:
                self._retain_dspark_prefill(step)
            return collect_final_logits(self.final.host.logits, self._output_rows)
        finally:
            self._step = self._inputs = self._group_counts = self._output_rows = None

    def reset_request(self, key, owner):
        self._step = self._inputs = self._group_counts = self._output_rows = None
        self.cache.reset_request(owner)
        if self.dspark is not None:
            self.dspark.release((key,))
            if getattr(self, "dspark_host_resources", None) is not None:
                rank, slot = owner.partition, owner.slot
                generation = self.dspark_state_generation(rank, slot) + 1
                self._dspark_generations[rank, slot] = generation
                self.dspark_buffers.reset_slot(rank, slot, generation)
                host = self.dspark_host_resources
                released = host.state.slot_ids[rank] == slot
                host.state.slot_ids[rank, released] = -1
                host.state.generations[rank, released] = 0
                host.cache_tables["draft"][rank, :, released] = -1
                getattr(self, "_dspark_prefill_progress", {}).pop(rank, None)
                # Slot-owned pages can be reused after this synchronous reset:
                # bootstrap overwrites the visible prefix before publishing
                # the new generation, so stale KV rows remain unreachable.

    def release_request(self, request_id):
        if self.engram_host is not None:
            self.engram_host.release_request(request_id)

    def close(self):
        if self.closed:
            return
        self._step = self._inputs = self._group_counts = self._output_rows = None
        try:
            if self._prefill_fwd is not None:
                self._prefill_fwd.close()
                self._prefill_fwd = None
            if self.weights is not None:
                self.weights.close()
                self.weights = None
            for resource in reversed(self._owned):
                resource.close()
        finally:
            self.worker.close()
            self.closed = True
            self._owned.clear()


# production_binding
_LOG = logging.getLogger(__name__)


def _lib_root():
    configured = os.environ.get("PYPTO_LIB_ROOT")
    root = Path(configured) if configured else Path.cwd() / "pypto-lib"
    root = root.resolve()
    if not (root / "models/deepseek_v4_1_flash/decode_fwd.py").is_file():
        raise MissingCompositeInterface(
            f"V4.1 decode_fwd is absent under {root}; set PYPTO_LIB_ROOT "
            "to a lib checkout containing the complete decode entry")
    return root


def selected_decode_topology(lib_root=None, *, dp_size=None):
    """Reject a lib whose import-time decode extent differs from this adapter."""
    root = _lib_root() if lib_root is None else Path(lib_root).resolve()
    config = root / "models/deepseek_v4_1_flash/config.py"
    try:
        module = ast.parse(config.read_text(encoding="utf-8"), filename=str(config))
    except (OSError, SyntaxError) as exc:
        raise MissingCompositeInterface(f"cannot read V4.1 lib decode config: {config}") from exc
    values = {}
    for statement in module.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not isinstance(target, ast.Name) or target.id not in (
                "DECODE_BATCH", "DSPARK_SPEC_TOKENS", "DECODE_SEQ"):
            continue
        if target.id in values:
            raise MissingCompositeInterface(f"duplicate V4.1 lib decode config: {target.id}")
        values[target.id] = statement.value
    batch = values.get("DECODE_BATCH")
    spec = values.get("DSPARK_SPEC_TOKENS")
    rows = values.get("DECODE_SEQ")
    if (not isinstance(batch, ast.Constant) or type(batch.value) is not int or batch.value != 12
            or not isinstance(spec, ast.Constant) or type(spec.value) is not int or spec.value != 5
            or not isinstance(rows, ast.BinOp) or not isinstance(rows.op, ast.Add)
            or not isinstance(rows.left, ast.Name) or rows.left.id != "DSPARK_SPEC_TOKENS"
            or not isinstance(rows.right, ast.Constant) or rows.right.value != 1):
        raise MissingCompositeInterface("V4.1 serving requires the lib's 12-request, six-row decode ABI")
    if dp_size is None:
        dp_size = int(os.environ.get("PYPTO_DSV41_DP_SIZE", "8"))
    if dp_size not in (4, 8):
        raise ValueError("V4.1 DSpark serving supports DP4/EP4 or DP8/EP8")
    return SegmentTopology.for_decode(tp=1, dp=dp_size)


def _revision(root):
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        # CI and packaged deployments may provide a source archive without a
        # .git directory.  The selected source path is still validated by the
        # ABI checks above; callers may provide a provenance value explicitly.
        return os.environ.get("PYPTO_LIB_REVISION", "archive")


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


def prefill_ring_heap(capacity):
    """Redistribute the existing 10 GiB budget for complete long prefills.

    A 4K attention scope exhausts ring 1 at 2 GiB. The worker reserves two
    resource sets, so growing their total footprint crowds out DP4 weights.
    Move space from the deepest ring while keeping the total budget fixed.
    """
    attention_heap = max(2 << 30, capacity * (1 << 20))
    return (2 << 30, attention_heap, 2 << 30, (6 << 30) - attention_heap)


def make_production_bindings(
    model_dir,
    max_seq_len,
    *,
    tokenizer=None,
    dp_size=None,
    dspark_lowering_provider=None,
    request_slots=1,
):
    model_dir = Path(model_dir).resolve()
    raw = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    layers = plan_layers(raw)
    dp_size = (int(os.environ.get("PYPTO_DSV41_DP_SIZE", "8"))
               if dp_size is None else dp_size)
    if dp_size not in (4, 8):
        raise ValueError("PYPTO_DSV41_DP_SIZE must be 4 or 8")
    groups = build_cache_groups(layers, max_seq_len, dp_size=dp_size, request_slots=request_slots)
    root = _lib_root()
    revision = _revision(root)
    topology = selected_decode_topology(root, dp_size=dp_size)

    def allocate(plan, device_ids, runtime, options):
        from pypto_serving.model.common.compiler.compiler import KernelCompiler
        from pypto_serving.model.common.executor.utils import build_pypto_run_config

        if Path(plan.weights.model_dir).resolve() != model_dir or tuple(plan.layers) != layers:
            raise ValueError("V4.1 binding and execution plan refer to different checkpoints")
        if runtime.max_seq_len != max_seq_len or runtime.kv_cache_groups != groups:
            raise ValueError("V4.1 scheduler and resident cache contracts differ")
        if len(device_ids) != topology.dp or options.platform != "a5":
            raise ValueError(f"V4.1 production binding requires {topology.dp} A5 devices")
        _check_runtime_binaries(options.platform)
        dspark_weight_contract = None
        if runtime.num_speculative_tokens:
            from .dspark_weights import V41DSparkWeightContract

            dspark_loader = V41WeightLoader(
                model_dir, tp_size=topology.tp, tp_rank=0,
                ep_size=topology.world, ep_rank=0, max_load_bytes=512 << 20,
            )
            dspark_weight_contract = V41DSparkWeightContract(dspark_loader)
            # Validate the index before compiling or allocating the worker. The
            # large three-layer draft bank is intentionally packed lazily by
            # the stage-argument provider and is never materialized here.
            dspark_weight_contract.validate_index()
        prefill_topology = SegmentTopology.for_prefill(
            dp=topology.dp, max_tokens=min(runtime.max_num_batched_tokens, MAX_PREFILL_TOKENS_PER_DP))
        run_config = replace(
            build_pypto_run_config(
                platform=options.platform,
                device_ids=device_ids,
                pypto_build_dir=options.pypto_build_dir,
            ),
            ring_heap=(runtime.ring_heap if runtime.ring_heap is not None
                       else prefill_ring_heap(prefill_topology.capacity)),
            # DP4's 8K prefill can exhaust ring 2's default dependency pool.
            ring_dep_pool=((16384, 16384, 32764, 16384)
                           if topology.dp == 4 else None),
        )
        compiler = KernelCompiler(run_config=run_config)
        decode, decode_names, decode_counts_are_tensor = compile_decode_fwd(
            compiler, root, topology)
        if runtime.num_speculative_tokens:
            dspark_programs, dspark_param_names = compile_decode_fwd_dspark(
                compiler, root, topology)
        else:
            dspark_programs, dspark_param_names = {}, {}
        prefill, prefill_names = compile_prefill_fwd(compiler, root, prefill_topology)
        abi = importlib.import_module("models.deepseek_v4_1_flash.decode_fwd")
        kv_sources, index_sources, c2a_sources = _sources(layers)
        layout = DecodeWeightLayout.from_lib(abi, tp_size=topology.tp)
        if (layout.layers, layout.kv_sources, layout.index_sources,
                layout.c2a_sources, layout.world) != (
                    len(layers), len(kv_sources), len(index_sources), len(c2a_sources), topology.world):
            raise ValueError("V4.1 checkpoint source axes disagree with the selected lib")
        DecodeWeightBank.validate_abi(decode_names)

        bank_started = time.perf_counter()
        bank = load_prepacked_bank(model_dir, layout, decode_names)
        if bank is None:
            bank = DecodeWeightBank(layout)
            append_seconds = 0.0
            layer_started = time.perf_counter()
            parallel_experts = topology.dp in (4, 8)
            previous_threads = torch.get_num_threads()
            if parallel_experts and previous_threads < 8:
                torch.set_num_threads(8)
            try:
                for item in iter_decode_weight_slices(
                        model_dir, topology, parallel_experts=parallel_experts, bank=bank):
                    append_started = time.perf_counter()
                    bank.append(item)
                    append_seconds += time.perf_counter() - append_started
                    if item.name == "hc_attn_fn":
                        now = time.perf_counter()
                        _LOG.info("V4.1 checkpoint layer %d/%d loaded into shared weight bank (%.2fs)",
                                  item.slot + 1, layout.layers, now - layer_started)
                        layer_started = now
            finally:
                if parallel_experts and previous_threads < 8:
                    torch.set_num_threads(previous_threads)
            bank.finish(decode_names)
            _LOG.info("V4.1 checkpoint staging %.2fs (bank copies %.2fs)",
                      time.perf_counter() - bank_started, append_seconds)
            peak_rss = peak_rss_bytes()
            if peak_rss is not None:
                _LOG.info("V4.1 checkpoint staging peak RSS %.1f GiB",
                          peak_rss / (1 << 30))
        _LOG.info("V4.1 shared weight bank ready: %.1f GiB",
                  bank.allocated_bytes / (1 << 30))

        rope = importlib.import_module("models.deepseek_v4_1_flash.rope_tables")
        swa_tables = rope.precompute_rope_tables(max_seq_len, False)
        compressed_tables = rope.precompute_rope_tables(max_seq_len, True)
        cache_plan = plan_decode_cache_pools(
            layers, groups, max_seq_len=max_seq_len,
            primary_num_blocks=groups[0].max_blocks_per_seq * request_slots,
            request_slots=request_slots, abi=abi,
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
            engram_k=getattr(abi, "ENGRAM_K", 6144),
            dspark_hidden_size=getattr(abi, "DSPARK_MAIN_IN", None),
        )
        decode_host = PrefillMetadataResources(topology, **host_kwargs)
        input_host = make_input_host_resources(prefill_topology, plan.weights.config.hidden_size)
        decode_input_host = make_input_host_resources(topology, plan.weights.config.hidden_size)
        final_host = load_final_output_host_resources(model_dir, topology)
        prefill_host = PrefillMetadataResources(
            prefill_topology, **host_kwargs, full_hidden_size=plan.weights.config.hidden_size)
        engram_host = None
        if "engram_lookup" in set(decode_names) | set(prefill_names):
            tokenizer_adapter = tokenizer
            if tokenizer_adapter is None:
                from pypto_serving.model.tokenizer import load_tokenizer

                tokenizer_adapter = load_tokenizer(model_dir)
            tokenizer_impl = getattr(tokenizer_adapter, "tokenizer", tokenizer_adapter)
            tokenizer_impl = getattr(tokenizer_impl, "_tokenizer", tokenizer_impl)
            engram_host = HostEngramResources(
                model_dir, topology, tokenizer_impl, capacity=max(prefill_topology.capacity, topology.capacity))
        dspark_host = None
        if runtime.num_speculative_tokens:
            from .dspark_resources import DSparkResourceLayout, allocate_dspark_host_resources

            dspark_host = allocate_dspark_host_resources(DSparkResourceLayout(
                dp_size=topology.dp, max_seq_len=max_seq_len,
                vocab_size=plan.weights.config.vocab_size,
                max_prefill_tokens=min(max_seq_len, DSPARK_WINDOW),
            ))
            # Reuse the already validated target RoPE table.  The draft ABI
            # uses the same 32-wide adjacent-pair table as the target.
            dspark_host.rope_tables["draft_cos"].copy_(decode_host.fields["freqs_cos"])
            dspark_host.rope_tables["draft_sin"].copy_(decode_host.fields["freqs_sin"])
        programs_by_id = {}
        for program in (prefill, decode, *dspark_programs.values()):
            programs_by_id.setdefault(id(program.compiled), program)
        programs = tuple(programs_by_id.values())
        resident_started = time.perf_counter()
        resident = V41ResidentResources(
            programs, run_config, weight_bank=bank, weight_param_names=decode_names,
            cache_plan=cache_plan, state_reset=state_reset, page_resets=resets,
            input_host=input_host, decode_input_host=decode_input_host,
            prefill_host=prefill_host, decode_host=decode_host,
            final_host=final_host,
            engram_host=engram_host,
            topology=topology, prefill_program=prefill, prefill_param_names=prefill_names,
            dspark_programs=dspark_programs, dspark_param_names=dspark_param_names,
            max_seq_len=max_seq_len,
            dspark_weight_contract=dspark_weight_contract,
            dspark_lowering_provider=dspark_lowering_provider,
            dspark_host_resources=dspark_host,
        )
        _LOG.info("V4.1 resident upload and allocation %.2fs",
                  time.perf_counter() - resident_started)
        try:
            resident.bind_decode_fwd(
                decode, decode_names, run_config, decode_counts_are_tensor)
            if dspark_host is not None and dspark_lowering_provider is None:
                from .dspark_resources import V41DSparkLoweringProvider

                resident.dspark_lowering_provider = V41DSparkLoweringProvider(
                    dspark_host, payload_builder=resident._dspark_payload,
                    state_resolver=resident._resolve_dspark_state,
                    target_param_names=decode_names,
                    draft_param_names=dspark_param_names["dspark_drafter"],
                    markov_param_names=dspark_param_names["markov_sample"],
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
        reset_request=lambda resources, key, owner: resources.reset_request(key, owner),
        wait=lambda resources: None,
        close=lambda resources: resources.close(),
        cache_groups=groups,
        segment_topology=topology,
        input_layout="tp_local_token",
        output_layout="tp_local_token",
        prefill_backbone=lambda state, step, resources, plans: resources.run_prefill_fwd(
            state, step),
        decode_backbone=lambda state, step, resources, plans: resources.run_decode_fwd(
            state, step),
    )
