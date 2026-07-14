# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import heapq
import itertools
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field

from .base import BasePrefixCache, EvictResult, InsertResult, MatchResult


ExtraKey = tuple[str, ...]


@dataclass(frozen=True)
class RadixKey:
    """An exact token prefix with an explicit cache namespace and optional cap."""

    token_ids: tuple[int, ...]
    extra_key: ExtraKey = ()
    limit: int | None = None

    @classmethod
    def from_tokens(
        cls,
        token_ids: Sequence[int] | Iterable[int],
        *,
        extra_key: ExtraKey = (),
        limit: int | None = None,
    ) -> "RadixKey":
        return cls(tuple(int(token_id) for token_id in token_ids), tuple(extra_key), limit)

    def __len__(self) -> int:
        if self.limit is None:
            return len(self.token_ids)
        return max(0, min(len(self.token_ids), int(self.limit)))

    def __iter__(self) -> Iterator[int]:
        return iter(self.token_ids[: len(self)])

    def __getitem__(self, item: int | slice) -> "RadixKey":
        visible = self.token_ids[: len(self)]
        if isinstance(item, int):
            index = item if item >= 0 else len(visible) + item
            if index < 0 or index >= len(visible):
                raise IndexError("RadixKey index out of range")
            return RadixKey((visible[index],), self.extra_key)
        start, stop, step = item.indices(len(visible))
        if step != 1:
            raise ValueError("RadixKey slice step must be 1")
        return RadixKey(visible[start:stop], self.extra_key)

    def visible_tokens(self) -> tuple[int, ...]:
        return self.token_ids[: len(self)]

    def page_aligned(self, page_size: int) -> "RadixKey":
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        aligned_len = len(self) // page_size * page_size
        return self[:aligned_len]

    def match(self, other: "RadixKey", page_size: int = 1) -> int:
        """Return an exact common-prefix length rounded down to a page boundary."""
        if self.extra_key != other.extra_key:
            raise ValueError("RadixKey operations require matching extra_key")
        left = self.visible_tokens()
        right = other.visible_tokens()
        limit = min(len(left), len(right))

        # Compare expanding slices so long equal prefixes stay in C tuple comparison.
        matched = limit
        low = 0
        step = 1
        while low < limit:
            high = min(limit, low + step)
            if left[low:high] != right[low:high]:
                while high - low > 1:
                    middle = (low + high) // 2
                    if left[low:middle] == right[low:middle]:
                        low = middle
                    else:
                        high = middle
                matched = low
                break
            low = high
            step *= 2

        if page_size <= 1:
            return matched
        return matched // page_size * page_size

    def child_key(self, page_size: int = 1) -> object:
        if len(self) < page_size or page_size <= 0:
            raise ValueError("child key requires at least one full page")
        visible = self.visible_tokens()
        first = visible[0] if page_size == 1 else visible[:page_size]
        return first if not self.extra_key else (self.extra_key, first)


@dataclass(eq=False)
class TreeNode:
    """A compressed radix edge and its canonical logical KV locations."""

    id: int
    parent: "TreeNode | None" = None
    key: RadixKey = field(default_factory=lambda: RadixKey(()))
    value: list[int] = field(default_factory=list)
    children: dict[object, "TreeNode"] = field(default_factory=dict)
    lock_ref: int = 0
    last_access_time: float = field(default_factory=time.monotonic)
    creation_time: float = field(default_factory=time.monotonic)
    hit_count: int = 0
    priority: int = 0

    def __lt__(self, other: "TreeNode") -> bool:
        return self.last_access_time < other.last_access_time


class RadixCache(BasePrefixCache):
    """Page-aligned compressed radix cache for canonical KV slot indices."""

    def __init__(
        self,
        page_size: int,
        *,
        disable: bool = False,
        retain_pages: Callable[[list[int]], None] | None = None,
        release_pages: Callable[[list[int]], None] | None = None,
    ) -> None:
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        self.page_size = int(page_size)
        self.disable = bool(disable)
        self._retain_pages = retain_pages
        self._release_pages = release_pages
        self._ids = itertools.count()
        self.evictable_leaves: set[TreeNode] = set()
        self._evictable_size = 0
        self._protected_size = 0
        self.root_node = self._new_root()

    @classmethod
    def create_simulated(cls, page_size: int = 1) -> "RadixCache":
        return cls(page_size=page_size)

    def _new_root(self) -> TreeNode:
        root = TreeNode(id=next(self._ids), key=RadixKey(()), lock_ref=1, priority=-(2**63))
        return root

    def reset(self) -> None:
        if self._release_pages is not None:
            for node in self._iter_nodes():
                if node is not self.root_node and node.value:
                    self._release_pages(self._page_ids(node.value))
        self.evictable_leaves.clear()
        self._evictable_size = 0
        self._protected_size = 0
        self.root_node = self._new_root()

    def match_prefix(self, key: RadixKey) -> MatchResult:
        if self.disable:
            return self._empty_match()
        key = key.page_aligned(self.page_size)
        if len(key) == 0:
            return self._empty_match()

        values, last_node = self._match_prefix_helper(self.root_node, key)
        flattened = tuple(slot for value in values for slot in value)
        return MatchResult(
            device_indices=flattened,
            last_device_node=last_node,
            last_host_node=last_node,
            best_match_node=last_node,
        )

    def _empty_match(self) -> MatchResult:
        return MatchResult((), self.root_node, self.root_node, self.root_node)

    def insert(self, key: RadixKey, value: list[int], *, priority: int = 0) -> InsertResult:
        if self.disable:
            return InsertResult(prefix_len=0)
        key = key.page_aligned(self.page_size)
        value = [int(slot) for slot in value[: len(key)]]
        if len(value) != len(key):
            raise ValueError(f"value has {len(value)} slots, expected {len(key)}")
        self._validate_page_values(value)
        prefix_len, last_node, inserted_len = self._insert_helper(
            self.root_node,
            key,
            value,
            int(priority or 0),
        )
        return InsertResult(
            prefix_len=prefix_len,
            total_len=len(key),
            last_device_node=last_node,
            inserted_len=inserted_len,
        )

    def inc_lock_ref(self, node: TreeNode) -> int:
        if self.disable:
            return 0
        delta = 0
        while node is not self.root_node:
            if node.lock_ref == 0:
                self._evictable_size -= len(node.key)
                self._protected_size += len(node.key)
                delta -= len(node.key)
            node.lock_ref += 1
            self._update_leaf_status(node)
            if node.parent is None:
                raise RuntimeError("radix node is detached from its root")
            node = node.parent
        return delta

    def dec_lock_ref(self, node: TreeNode) -> int:
        if self.disable:
            return 0
        delta = 0
        while node is not self.root_node:
            if node.lock_ref <= 0:
                raise RuntimeError("radix node lock_ref is already zero")
            if node.lock_ref == 1:
                self._evictable_size += len(node.key)
                self._protected_size -= len(node.key)
                delta += len(node.key)
            node.lock_ref -= 1
            self._update_leaf_status(node)
            if node.parent is None:
                raise RuntimeError("radix node is detached from its root")
            node = node.parent
        return delta

    def evict(self, num_tokens: int) -> EvictResult:
        if self.disable or num_tokens <= 0:
            return EvictResult()
        heap = [(node.last_access_time, node.id, node) for node in self.evictable_leaves]
        heapq.heapify(heap)
        evicted = 0
        while heap and evicted < num_tokens:
            _, _, node = heapq.heappop(heap)
            if not self._is_evictable_leaf(node):
                continue
            parent = node.parent
            if parent is None:
                continue
            if self._release_pages is not None:
                self._release_pages(self._page_ids(node.value))
            evicted += len(node.value)
            self._delete_leaf(node)
            if self._is_evictable_leaf(parent):
                heapq.heappush(heap, (parent.last_access_time, parent.id, parent))
        return EvictResult(num_tokens_evicted=evicted)

    def evictable_size(self) -> int:
        return self._evictable_size

    def protected_size(self) -> int:
        return self._protected_size

    def total_size(self) -> int:
        return self._evictable_size + self._protected_size

    def all_values_flatten(self) -> list[int]:
        return [slot for node in self._iter_nodes() if node is not self.root_node for slot in node.value]

    def _match_prefix_helper(self, node: TreeNode, key: RadixKey) -> tuple[list[list[int]], TreeNode]:
        access_time = time.monotonic()
        node.last_access_time = access_time
        values: list[list[int]] = []
        while len(key) > 0:
            child = node.children.get(key.child_key(self.page_size))
            if child is None:
                break
            child.last_access_time = access_time
            prefix_len = child.key.match(key, self.page_size)
            if prefix_len == 0:
                break
            if prefix_len < len(child.key):
                node = self._split_node(child, prefix_len)
                values.append(node.value)
                break
            values.append(child.value)
            child.hit_count += 1
            node = child
            key = key[prefix_len:]
        return values, node

    def _insert_helper(
        self,
        node: TreeNode,
        key: RadixKey,
        value: list[int],
        priority: int,
    ) -> tuple[int, TreeNode, int]:
        access_time = time.monotonic()
        node.last_access_time = access_time
        node.priority = max(node.priority, priority)
        total_prefix = 0

        while len(key) > 0:
            child_key = key.child_key(self.page_size)
            child = node.children.get(child_key)
            if child is None:
                break
            child.last_access_time = access_time
            prefix_len = child.key.match(key, self.page_size)
            if prefix_len == 0:
                break
            total_prefix += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]
            if prefix_len < len(child.key):
                node = self._split_node(child, prefix_len)
            else:
                node = child
            node.priority = max(node.priority, priority)
            node.hit_count += 1

        inserted_len = len(key)
        if inserted_len:
            new_node = TreeNode(
                id=next(self._ids),
                parent=node,
                key=key,
                value=list(value),
                priority=priority,
            )
            node.children[key.child_key(self.page_size)] = new_node
            self._evictable_size += inserted_len
            self._update_leaf_status(node)
            self._update_leaf_status(new_node)
            if self._retain_pages is not None:
                self._retain_pages(self._page_ids(new_node.value))
            node = new_node
        return total_prefix, node, inserted_len

    def _split_node(self, child: TreeNode, split_len: int) -> TreeNode:
        if split_len <= 0 or split_len >= len(child.key):
            raise ValueError("split_len must split the child edge")
        if split_len % self.page_size:
            raise ValueError("split_len must be page aligned")
        parent = child.parent
        if parent is None:
            raise RuntimeError("cannot split a detached radix node")

        old_key = child.key
        new_node = TreeNode(
            id=next(self._ids),
            parent=parent,
            key=old_key[:split_len],
            value=list(child.value[:split_len]),
            lock_ref=child.lock_ref,
            last_access_time=child.last_access_time,
            creation_time=child.creation_time,
            hit_count=child.hit_count,
            priority=child.priority,
        )
        parent.children[old_key.child_key(self.page_size)] = new_node
        child.parent = new_node
        child.key = old_key[split_len:]
        child.value = list(child.value[split_len:])
        new_node.children[child.key.child_key(self.page_size)] = child
        self._update_leaf_status(new_node)
        self._update_leaf_status(child)
        return new_node

    def _delete_leaf(self, node: TreeNode) -> None:
        if not self._is_evictable_leaf(node):
            raise RuntimeError("only unlocked radix leaves can be evicted")
        parent = node.parent
        if parent is None:
            raise RuntimeError("cannot delete radix root")
        removed = parent.children.pop(node.key.child_key(self.page_size), None)
        if removed is not node:
            raise RuntimeError("radix parent does not own the expected child")
        self.evictable_leaves.discard(node)
        self._evictable_size -= len(node.key)
        node.parent = None
        self._update_leaf_status(parent)

    def _update_leaf_status(self, node: TreeNode) -> None:
        if node is self.root_node or node.lock_ref > 0 or node.children or not node.value:
            self.evictable_leaves.discard(node)
            return
        self.evictable_leaves.add(node)

    def _is_evictable_leaf(self, node: TreeNode) -> bool:
        return (
            node is not self.root_node
            and node.parent is not None
            and node.lock_ref == 0
            and not node.children
            and bool(node.value)
        )

    def _iter_nodes(self) -> Iterator[TreeNode]:
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(node.children.values())

    def _page_ids(self, values: Sequence[int]) -> list[int]:
        if not values:
            return []
        self._validate_page_values(values)
        return [int(values[offset]) // self.page_size for offset in range(0, len(values), self.page_size)]

    def _validate_page_values(self, values: Sequence[int]) -> None:
        if len(values) % self.page_size:
            raise ValueError("radix values must contain complete pages")
        for start in range(0, len(values), self.page_size):
            first = int(values[start])
            page_id = first // self.page_size
            expected = range(page_id * self.page_size, (page_id + 1) * self.page_size)
            actual = [int(slot) for slot in values[start : start + self.page_size]]
            if actual != list(expected):
                raise ValueError("each radix value page must contain contiguous physical slots")
