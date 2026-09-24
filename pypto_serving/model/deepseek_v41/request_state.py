# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Request-owned positions and stable compressor slots, independent of batch rows.

Physical pages remain owned by the shared scheduler. A failed in-place kernel
cannot be rolled back by restoring a length: its request must be reset and
recomputed. Prefix-cache attachment is intentionally unsupported here.
"""
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping


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
