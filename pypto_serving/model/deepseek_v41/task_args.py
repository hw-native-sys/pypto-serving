# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek V4.1 task args."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from pypto_serving.config.types import KVCacheGroupSpec, KVCacheSpec

from .weight_loader import LayerPlan


# cache_contract
@dataclass(frozen=True)
class V41CacheGroups:
    window: str
    c2a: str
    c1a: str


def build_cache_groups(layers: tuple[LayerPlan, ...], max_seq_len: int) -> tuple[KVCacheGroupSpec, ...]:
    """Declare the three scheduler page namespaces from checkpoint producers."""
    if len(layers) != 40 or tuple(layer.layer_id for layer in layers) != tuple(range(40)):
        raise ValueError("V4.1 cache groups require the complete ordered backbone")
    if type(max_seq_len) is not int or max_seq_len <= 0:
        raise ValueError("V4.1 cache groups require a positive context length")
    producers = {
        "window": tuple(range(40)),
        "c2a": tuple(layer.layer_id for layer in layers
                     if layer.mode == "c2a_full" and layer.kv_source == layer.layer_id),
        "c1a": tuple(layer.layer_id for layer in layers
                     if layer.mode == "c1a_full" and layer.kv_source == layer.layer_id),
    }
    groups = tuple(KVCacheGroupSpec(
        name, producers[name], KVCacheSpec(block_size, 16, ratio),
        (max_seq_len + block_size - 1) // block_size, num_partitions=2,
    ) for name, block_size, ratio in (("window", 128, 1), ("c2a", 256, 2), ("c1a", 128, 1)))
    validate_cache_groups(layers, groups, max_seq_len)
    return groups


def validate_cache_groups(
    layers: tuple[LayerPlan, ...], groups: tuple[KVCacheGroupSpec, ...], max_seq_len: int,
) -> V41CacheGroups:
    """Resolve the three full-history page families before device allocation."""
    if len(layers) != 40 or tuple(layer.layer_id for layer in layers) != tuple(range(40)):
        raise ValueError("V4.1 cache contract requires the complete ordered 40-layer backbone")
    if type(max_seq_len) is not int or max_seq_len <= 0:
        raise ValueError("V4.1 cache contract requires a positive sequence capacity")
    if (len(groups) != 3 or any(not isinstance(group, KVCacheGroupSpec) for group in groups)
            or len({group.name for group in groups}) != 3):
        raise ValueError("V4.1 requires separate window, C2A and C1A cache groups")

    expected = {
        "window": (tuple(range(40)), 128, 1),
        "c2a": (tuple(layer.layer_id for layer in layers
                      if layer.mode == "c2a_full" and layer.kv_source == layer.layer_id), 256, 2),
        "c1a": (tuple(layer.layer_id for layer in layers
                      if layer.mode == "c1a_full" and layer.kv_source == layer.layer_id), 128, 1),
    }
    if not expected["c2a"][0] or not expected["c1a"][0]:
        raise ValueError("V4.1 cache contract requires C2A and C1A KV producers")
    resolved = {}
    for group in groups:
        matches = [family for family, (owners, _, _) in expected.items()
                   if tuple(group.layer_indices) == owners]
        if len(matches) != 1 or matches[0] in resolved:
            raise ValueError("V4.1 cache groups must match window and KV producer layers")
        family = matches[0]
        _, block_size, ratio = expected[family]
        if (group.spec.block_size != block_size or group.spec.compress_ratio != ratio
                or group.spec.storage_block_size != 128 or group.num_partitions != 2
                or group.sliding_window is not None or group.is_eagle_group):
            raise ValueError(f"V4.1 {family} cache page layout disagrees with the lib ABI")
        required = (max_seq_len + block_size - 1) // block_size
        if (group.max_blocks_per_seq < required
                or (group.num_blocks is not None and group.num_blocks < required)):
            raise ValueError(f"V4.1 {family} cache cannot cover max_seq_len")
        resolved[family] = group.name
    return V41CacheGroups(**resolved)


# request_state
@dataclass(frozen=True)
class RequestSlice:
    request_id: str
    partition: int
    state_slot: int
    start: int
    token_ids: tuple[int, ...]
    prompt_length: int
    pages: Mapping[str, tuple[int, ...]]

    @property
    def end(self):
        return self.start + len(self.token_ids)


@dataclass(frozen=True)
class ForwardStep:
    phase: str
    requests: tuple[RequestSlice, ...]
    epoch: int

    @property
    def token_ids(self):
        return tuple(token for request in self.requests for token in request.token_ids)

    @property
    def positions(self):
        return tuple(p for request in self.requests for p in range(request.start, request.end))

    @property
    def request_rows(self):
        return tuple(i for i, request in enumerate(self.requests) for _ in request.token_ids)

    @property
    def terminal_prefill(self):
        return tuple(request.end == request.prompt_length for request in self.requests)


@dataclass
class RequestProgress:
    partition: int
    slot: int
    prompt_length: int
    length: int = 0
    pages: Mapping[str, tuple[int, ...]] = field(default_factory=dict)


class RequestLedger:
    """One synchronous in-flight batch; slots survive omission and reordering."""
    def __init__(self, *, max_requests, max_seq_len, partitions=2):
        if any(type(v) is not int or v <= 0 for v in (max_requests, max_seq_len, partitions)):
            raise ValueError("request capacities must be positive integers")
        self.max_seq_len = max_seq_len
        self.owners = {}
        self.free = [list(range(max_requests - 1, -1, -1)) for _ in range(partitions)]
        self.pending = None
        self.epoch = 1
        self.poisoned = False

    def _validate_page_ownership(self, slices):
        """Retain full-history pages even when their owner is absent this step.

        There is no cache relocation/copy contract in this adapter. Continuations
        may append pages, but cannot replace or drop their committed prefix.
        Validate before reserving slots or exposing an in-flight transaction.
        """
        occupied = {}
        for key, owner in self.owners.items():
            for group, pages in owner.pages.items():
                for page in pages:
                    occupied[owner.partition, group, page] = key
        for request in slices:
            owner = self.owners.get(request.request_id)
            if owner is not None:
                for group, previous in owner.pages.items():
                    current = request.pages.get(group, ())
                    if tuple(current[:len(previous)]) != tuple(previous):
                        raise ValueError("continuation must preserve committed full-history pages")
            for group, pages in request.pages.items():
                if len(set(pages)) != len(pages):
                    raise ValueError("request must not alias its own cache pages")
                for page in pages:
                    address = (request.partition, group, page)
                    key = occupied.get(address)
                    if key is not None and key != request.request_id:
                        raise ValueError("cache page belongs to another live request")
                    occupied[address] = request.request_id

    def begin_prefill(self, requests):
        """Validate the complete batch before taking ownership of any new slot.

        Each item is (request_id, partition, start, token_ids, prompt_length, pages).
        Input order is retained for output mapping; it never determines state slots.
        """
        if self.poisoned:
            raise RuntimeError("session requires recovery after device reset failure")
        if self.pending is not None:
            raise RuntimeError("a request batch is already in flight")
        if not requests or len({r[0] for r in requests}) != len(requests):
            raise ValueError("batch must contain distinct request IDs")
        available = [list(slots) for slots in self.free]
        additions, slices = {}, []
        for key, partition, start, tokens, prompt_length, pages in requests:
            if not isinstance(key, str) or not key:
                raise ValueError("request ID must be nonempty")
            if type(partition) is not int or not 0 <= partition < len(available):
                raise ValueError("invalid DP cache partition")
            if any(type(v) is not int for v in (start, prompt_length)):
                raise ValueError("positions and prompt length must be integers")
            if not tokens or not 0 <= start < start + len(tokens) <= prompt_length <= self.max_seq_len:
                raise ValueError("invalid prefill extent")
            owner = self.owners.get(key)
            if owner is None:
                if start != 0:
                    raise ValueError("cold requests require prefill from position zero; prefix restore is unavailable")
                if not available[partition]:
                    raise ValueError("compressor state slot capacity exhausted")
                owner = RequestProgress(partition, available[partition].pop(), prompt_length)
                additions[key] = owner
            if owner.partition != partition or owner.length != start or owner.prompt_length != prompt_length:
                raise ValueError("request partition, committed position or prompt length changed")
            if owner.length == owner.prompt_length:
                raise ValueError("prefill already completed")
            immutable_pages = MappingProxyType({name: tuple(ids) for name, ids in pages.items()})
            slices.append(RequestSlice(key, partition, owner.slot, start, tuple(tokens),
                                       prompt_length, immutable_pages))
        self._validate_page_ownership(slices)
        self.free = available
        self.owners.update(additions)
        self.pending = ForwardStep("prefill", tuple(slices), self.epoch)
        return self.pending

    def begin_decode(self, requests):
        if self.poisoned:
            raise RuntimeError("session requires recovery after device reset failure")
        if self.pending is not None:
            raise RuntimeError("a request batch is already in flight")
        if not requests or len({r[0] for r in requests}) != len(requests):
            raise ValueError("batch must contain distinct request IDs")
        slices = []
        for key, partition, start, token, pages in requests:
            owner = self.owners.get(key)
            if owner is None or owner.length < owner.prompt_length:
                raise ValueError("decode requires a completed prefill for the same request")
            if type(start) is not int or owner.length != start or owner.partition != partition:
                raise ValueError("decode must continue at the committed position and DP partition")
            if start + 1 > self.max_seq_len:
                raise ValueError("decode exceeds sequence capacity")
            slices.append(RequestSlice(key, partition, owner.slot, start, (token,), owner.prompt_length,
                                       MappingProxyType({name: tuple(ids) for name, ids in pages.items()})))
        self._validate_page_ownership(slices)
        self.pending = ForwardStep("decode", tuple(slices), self.epoch)
        return self.pending

    def commit(self, step):
        if step is not self.pending:
            raise ValueError("stale or foreign forward completion")
        for request in step.requests:
            owner = self.owners[request.request_id]
            owner.length, owner.pages = request.end, request.pages
        self.pending = None
        self.epoch += 1

    def release(self, request_ids, reset):
        """reset is synchronous; failed reset retains ownership and poisons reuse."""
        if self.pending is not None:
            raise RuntimeError("cannot release an in-flight batch")
        for key in dict.fromkeys(request_ids):
            owner = self.owners.get(key)
            if owner is None:
                continue
            try:
                reset(key, owner)
            except Exception:
                self.poisoned = True
                raise
            del self.owners[key]
            self.free[owner.partition].append(owner.slot)

    def abort(self, step, reset):
        if step is not self.pending:
            raise ValueError("stale or foreign failed forward")
        for request in step.requests:
            owner = self.owners[request.request_id]
            names = owner.pages.keys() | request.pages.keys()
            owner.pages = MappingProxyType({name: tuple(dict.fromkeys(
                (*owner.pages.get(name, ()), *request.pages.get(name, ())))) for name in names})
        self.pending = None
        # All layer/cache writes are potentially partial. Invalidate the whole
        # affected request, including its previously committed prefix.
        self.release([r.request_id for r in step.requests], reset)
        self.epoch += 1
