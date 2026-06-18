# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field


ExtraKey = tuple[str, ...]


@dataclass(frozen=True)
class RadixKey:
    """Token prefix lookup key with an explicit cache namespace."""

    token_ids: tuple[int, ...]
    extra_key: ExtraKey = ()

    @classmethod
    def from_tokens(cls, token_ids: list[int] | tuple[int, ...], *, extra_key: ExtraKey = ()) -> "RadixKey":
        return cls(tuple(int(token_id) for token_id in token_ids), tuple(extra_key))


@dataclass
class RadixNode:
    """Compressed radix-tree node whose edge stores page-aligned token and page IDs."""

    parent: "RadixNode | None" = None
    extra_key: ExtraKey = ()
    tokens: tuple[int, ...] = ()
    page_ids: list[int] = field(default_factory=list)
    children: dict[object, "RadixNode"] = field(default_factory=dict)
    lock_ref: int = 0
    last_access_time: float = field(default_factory=time.monotonic)
    priority: int = 0


@dataclass(frozen=True)
class RadixMatch:
    """Longest page-aligned prefix match result."""

    prefix_len: int
    page_ids: list[int]
    last_node: RadixNode


@dataclass(frozen=True)
class RadixInsertResult:
    """Insert accounting for already-present and newly-inserted pages."""

    existing_prefix_len: int
    inserted_len: int


class RadixPrefixCache:
    """Page-aligned radix-tree prefix cache for physical KV page IDs."""

    def __init__(
        self,
        page_size: int,
        *,
        retain_pages: Callable[[list[int]], None] | None = None,
        release_pages: Callable[[list[int]], None] | None = None,
    ) -> None:
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        self.page_size = int(page_size)
        self._retain_pages = retain_pages
        self._release_pages = release_pages
        self.root = RadixNode(priority=-(2**63))

    def reset(self) -> None:
        """Drop all radix metadata without releasing page references."""
        self.root = RadixNode(priority=-(2**63))

    def match(self, key: RadixKey) -> RadixMatch:
        """Return the longest cached prefix and lock its path against eviction."""
        tokens = self._page_aligned_tokens(key.token_ids)
        if not tokens:
            return RadixMatch(prefix_len=0, page_ids=[], last_node=self.root)

        node = self.root
        matched_pages: list[int] = []
        offset = 0
        access_time = time.monotonic()
        self.root.last_access_time = access_time

        while offset < len(tokens):
            child_key = self._child_key(key.extra_key, tokens[offset:])
            child = node.children.get(child_key)
            if child is None:
                break

            child.last_access_time = access_time
            prefix_len = self._page_aligned_len(self._common_prefix_len(child.tokens, tokens[offset:]))
            if prefix_len == 0:
                break
            if prefix_len < len(child.tokens):
                child = self._split_node(child, prefix_len)
                child.last_access_time = access_time

            matched_pages.extend(child.page_ids)
            offset += len(child.tokens)
            node = child

        if node is not self.root:
            self.inc_lock_ref(node)
        return RadixMatch(prefix_len=offset, page_ids=list(matched_pages), last_node=node)

    def insert(self, key: RadixKey, page_ids: list[int], *, priority: int = 0) -> RadixInsertResult:
        """Insert a page-aligned token prefix and retain cache refs for new pages."""
        tokens, page_ids = self._align_tokens_and_pages(key.token_ids, page_ids)
        if not tokens:
            return RadixInsertResult(existing_prefix_len=0, inserted_len=0)

        node = self.root
        offset = 0
        page_offset = 0
        existing_prefix_len = 0
        access_time = time.monotonic()
        node.last_access_time = access_time
        node.priority = max(node.priority, priority)

        while offset < len(tokens):
            child_key = self._child_key(key.extra_key, tokens[offset:])
            child = node.children.get(child_key)
            if child is None:
                break

            child.last_access_time = access_time
            child.priority = max(child.priority, priority)
            prefix_len = self._page_aligned_len(self._common_prefix_len(child.tokens, tokens[offset:]))
            if prefix_len == 0:
                break
            if prefix_len < len(child.tokens):
                child = self._split_node(child, prefix_len)
                child.priority = max(child.priority, priority)
                child.last_access_time = access_time

            offset += len(child.tokens)
            page_offset += len(child.page_ids)
            existing_prefix_len += len(child.tokens)
            node = child

        inserted_len = len(tokens) - offset
        if inserted_len > 0:
            suffix_tokens = tokens[offset:]
            suffix_pages = list(page_ids[page_offset:])
            new_node = RadixNode(
                parent=node,
                extra_key=key.extra_key,
                tokens=suffix_tokens,
                page_ids=suffix_pages,
                priority=priority,
            )
            node.children[self._child_key(key.extra_key, suffix_tokens)] = new_node
            if suffix_pages and self._retain_pages is not None:
                self._retain_pages(suffix_pages)

        return RadixInsertResult(existing_prefix_len=existing_prefix_len, inserted_len=inserted_len)

    def inc_lock_ref(self, node: RadixNode) -> None:
        """Protect a matched path from radix eviction."""
        while node is not self.root:
            node.lock_ref += 1
            node = node.parent
            if node is None:
                raise RuntimeError("radix node is detached from its root")

    def dec_lock_ref(self, node: RadixNode) -> None:
        """Release one eviction lock from a matched path."""
        while node is not self.root:
            if node.lock_ref <= 0:
                raise RuntimeError("radix node lock_ref is already zero")
            node.lock_ref -= 1
            node = node.parent
            if node is None:
                raise RuntimeError("radix node is detached from its root")

    def evict_pages(self, min_pages: int) -> int:
        """Evict at least ``min_pages`` radix-owned pages from unlocked LRU leaves."""
        if min_pages <= 0:
            return 0

        evicted = 0
        while evicted < min_pages:
            leaves = self._evictable_leaves_by_lru()
            if not leaves:
                break
            for leaf in leaves:
                if evicted >= min_pages:
                    break
                evicted += self._evict_leaf(leaf)
        return evicted

    def total_pages(self) -> int:
        return sum(len(node.page_ids) for node in self._iter_nodes() if node is not self.root)

    def protected_pages(self) -> int:
        return sum(
            len(node.page_ids)
            for node in self._iter_nodes()
            if node is not self.root and node.lock_ref > 0
        )

    def evictable_pages(self) -> int:
        return sum(len(node.page_ids) for node in self._iter_nodes() if self._is_evictable_leaf(node))

    def _split_node(self, child: RadixNode, split_len: int) -> RadixNode:
        if split_len <= 0 or split_len >= len(child.tokens):
            raise ValueError("split_len must split the child edge")
        if split_len % self.page_size != 0:
            raise ValueError("split_len must be page-aligned")

        parent = child.parent
        if parent is None:
            raise RuntimeError("cannot split detached radix node")

        split_pages = split_len // self.page_size
        prefix_tokens = child.tokens[:split_len]
        prefix_pages = child.page_ids[:split_pages]
        suffix_tokens = child.tokens[split_len:]
        suffix_pages = child.page_ids[split_pages:]

        new_node = RadixNode(
            parent=parent,
            extra_key=child.extra_key,
            tokens=prefix_tokens,
            page_ids=list(prefix_pages),
            lock_ref=child.lock_ref,
            last_access_time=child.last_access_time,
            priority=child.priority,
        )

        old_child_key = self._child_key(child.extra_key, child.tokens)
        parent.children[old_child_key] = new_node

        child.parent = new_node
        child.tokens = suffix_tokens
        child.page_ids = list(suffix_pages)
        new_node.children[self._child_key(child.extra_key, suffix_tokens)] = child
        return new_node

    def _evictable_leaves_by_lru(self) -> list[RadixNode]:
        leaves = [node for node in self._iter_nodes() if self._is_evictable_leaf(node)]
        leaves.sort(key=lambda node: node.last_access_time)
        return leaves

    def _evict_leaf(self, leaf: RadixNode) -> int:
        if not self._is_evictable_leaf(leaf):
            return 0
        parent = leaf.parent
        if parent is None:
            return 0
        evicted = len(leaf.page_ids)
        if leaf.page_ids and self._release_pages is not None:
            self._release_pages(list(leaf.page_ids))
        parent.children.pop(self._child_key(leaf.extra_key, leaf.tokens), None)
        leaf.parent = None
        leaf.children.clear()
        leaf.page_ids = []
        leaf.tokens = ()
        return evicted

    def _is_evictable_leaf(self, node: RadixNode) -> bool:
        return node is not self.root and node.lock_ref == 0 and not node.children and bool(node.page_ids)

    def _iter_nodes(self):
        stack = [self.root]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(node.children.values())

    def _align_tokens_and_pages(
        self,
        token_ids: tuple[int, ...],
        page_ids: list[int],
    ) -> tuple[tuple[int, ...], list[int]]:
        tokens = self._page_aligned_tokens(token_ids)
        num_pages = len(tokens) // self.page_size
        if len(page_ids) < num_pages:
            raise ValueError(f"page_ids has {len(page_ids)} pages, need at least {num_pages}")
        return tokens, list(page_ids[:num_pages])

    def _page_aligned_tokens(self, token_ids: tuple[int, ...]) -> tuple[int, ...]:
        aligned_len = self._page_aligned_len(len(token_ids))
        return tuple(token_ids[:aligned_len])

    def _page_aligned_len(self, length: int) -> int:
        return (int(length) // self.page_size) * self.page_size

    def _child_key(self, extra_key: ExtraKey, tokens: tuple[int, ...]) -> object:
        if len(tokens) < self.page_size:
            raise ValueError("child key requires at least one full page")
        page_key = tuple(tokens[: self.page_size])
        return (tuple(extra_key), page_key)

    @staticmethod
    def _common_prefix_len(left: tuple[int, ...], right: tuple[int, ...]) -> int:
        limit = min(len(left), len(right))
        idx = 0
        while idx < limit and left[idx] == right[idx]:
            idx += 1
        return idx
