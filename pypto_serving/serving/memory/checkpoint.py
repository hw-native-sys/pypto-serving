# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Joint eviction preference for immutable rolling cache/state checkpoints.

These are allocator metadata, not extra allocations or ownership pins. Normal
prefix lookup still validates every required group, including full-history KV.
"""


class GroupPrefixCheckpoints:
    def __init__(self):
        self.checkpoints = {}
        self.pages = {}
        self.owners = {}
        self.histories = {}
        self.history_changed = False

    def publish(self, owner, pools, hashes, tokens, partition, boundary, valid_from):
        if not 0 < boundary <= tokens:
            return
        members, identity, history = [], [], {}
        for name, values in hashes.items():
            if not values:
                continue  # Request-local state is not a shareable Prefix group.
            pool = pools[name]
            capacity = pool.spec.spec.token_capacity
            end = boundary // capacity
            if end > len(values) or (pool.spec.is_eagle_group and boundary >= tokens):
                return
            identity.append((name, values[end - 1]))
            if pool.spec.sliding_window is None:
                if any((partition, values[index]) not in pool.hash_to_block for index in range(end)):
                    return
                history[name] = values, end
                continue
            start = max(0, end - pool.spec.sliding_window // capacity)
            if start * capacity < (valid_from or {}).get(name, 0):
                return
            for index in range(start, end):
                block = pool.hash_to_block.get((partition, values[index]))
                if block is None:
                    return  # Never protect only the KV half of a state checkpoint.
                members.append((name, block.block_id))
        if not members:
            return
        key = partition, boundary, tuple(sorted(identity))
        if owner is not None:
            previous = self.owners.pop(owner, None)
            if previous is not None and previous != key and previous not in self.owners.values():
                self._discard(previous)
            self.owners[owner] = key
        if key not in self.checkpoints:
            self.checkpoints[key] = tuple(members)
            self.histories[key] = history
            for page in members:
                self.pages.setdefault(page, set()).add(key)

    def release(self, owner):
        # The last complete checkpoint remains cached after request retirement;
        # intermediate checkpoints do not monopolize a small rolling pool.
        self.owners.pop(owner, None)

    def victim(self, pool, queue):
        block = queue.head
        while block is not None:
            if (pool.spec.name, block.block_id) not in self.pages:
                return block
            block = block.next_free
        return queue.head  # Bounded cache: checkpoints never prevent admission.

    def discard_page(self, pool, block):
        if pool.spec.sliding_window is None and block.block_hash is not None:
            self.history_changed = True
        for key in tuple(self.pages.get((pool.spec.name, block.block_id), ())):
            self._discard(key)  # One missing state/page retires the joint preference.

    def prune(self, pools):
        # Batch full-history eviction checks at the allocator operation boundary,
        # not once per page taken. In-flight table/snapshot refs still own data.
        if self.history_changed:
            for key, history in tuple(self.histories.items()):
                if any((key[0], values[index]) not in pools[name].hash_to_block
                       for name, (values, end) in history.items() for index in range(end)):
                    self._discard(key)
            self.history_changed = False

    def _discard(self, key):
        self.histories.pop(key, None)
        for owner in tuple(self.owners):
            if self.owners[owner] == key:
                del self.owners[owner]
        for page in self.checkpoints.pop(key, ()):
            owners = self.pages[page]
            owners.remove(key)
            if not owners:
                del self.pages[page]
