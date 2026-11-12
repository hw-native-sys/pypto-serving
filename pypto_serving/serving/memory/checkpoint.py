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

from dataclasses import dataclass


@dataclass
class CheckpointWritePlan:
    partition: int
    boundaries: tuple[int, ...]
    preferred: tuple[int, ...]
    write_blocks: dict[str, list[int]]
    publication_blocks: dict[str, dict[int, int]]
    records: dict[int, dict[str, dict[int, int]]]
    snapshot: object


class GroupCheckpointWriter:
    """Stage sparse publications in the existing grouped page pools."""

    def __init__(self, manager):
        self.manager = manager

    def prepare(self, request, start, end, prompt, replay_boundary, shared_boundary, period,
                tables, read_tables):
        manager = self.manager
        pools = manager._group_pools
        states = {name: pool for name, pool in pools.items() if pool.spec.is_checkpoint_state}
        if not manager.enable_prefix_cache or not states:
            return None
        alignment = manager.group_prefix_cache_alignment
        if type(period) is not int or period < 0 or period % alignment:
            raise ValueError(f"checkpoint period must be zero or a positive multiple of {alignment}")
        targets = {replay_boundary, shared_boundary}
        if period:
            targets.update(range((start // period + 1) * period, end + 1, period))
        targets = sorted((b for b in targets if b is not None and start < b <= min(end, prompt)),
                         key=lambda b: (b not in (replay_boundary, shared_boundary), b))
        if not targets:
            return None
        partition = manager._group_request_partitions[request]
        original = {name: list(tables[name]) for name in states}
        publications = {name: {} for name in states}
        records, fresh = {}, []
        try:
            for boundary in targets:
                members, needs = {}, {}
                for name, pool in states.items():
                    page = pool.spec.spec.token_capacity
                    begin = max(0, boundary - pool.spec.sliding_window) // page
                    stop = boundary // page
                    resident = max(0, (end + page - 1) // page - len(original[name]))
                    members[name] = {}
                    needs[name] = []
                    for logical in range(begin, stop):
                        if logical * page < start:
                            block = read_tables.get(name, {}).get(logical)
                            if block is None and logical >= resident:
                                block = original[name][logical % len(original[name])]
                            if block is None:
                                raise RuntimeError(f"Missing checkpoint entry history for {name}:{logical}")
                            members[name][logical] = block
                        elif logical >= resident:
                            members[name][logical] = original[name][logical % len(original[name])]
                        else:
                            needs[name].append(logical)
                # Keep one replacement generation available to the existing pipeline.
                if any(pool.num_free_blocks_in(partition) < len(needs[name]) + len(original[name])
                       for name, pool in states.items() if needs[name]):
                    continue
                for name, pool in states.items():
                    for logical in needs[name]:
                        block = manager._take_group_block(pool, partition)
                        fresh.append((pool, block))
                        members[name][logical] = pool.local_block_id(block)
                    publications[name].update(members[name])
                for name, pool in pools.items():
                    if pool.spec.sliding_window is None or name in states:
                        continue
                    page = pool.spec.spec.token_capacity
                    begin = max(0, boundary - pool.spec.sliding_window) // page
                    stop = boundary // page
                    if begin < max(0, (end + page - 1) // page - len(tables[name])):
                        raise RuntimeError("Checkpoint rolling window is outside the execution snapshot")
                    members[name] = {q: tables[name][q % len(tables[name])] for q in range(begin, stop)}
                records[boundary] = members
            if not records:
                return None
            pinned = {name: [] for name in pools}
            for name in states:
                pinned[name] = list(dict.fromkeys(original[name] + list(publications[name].values())))
            for members in records.values():
                for name, pages in members.items():
                    pinned[name] = list(dict.fromkeys(pinned[name] + list(pages.values())))
            snapshot = manager.retain_group_block_snapshot(pinned, partition)
            for name, pages in publications.items():
                pool = pools[name]
                page = pool.spec.spec.token_capacity
                owned = pool.request_blocks[request]
                stop = (end + page - 1) // page
                for logical, local in pages.items():
                    if not max(0, stop - len(owned)) <= logical < stop:
                        continue
                    slot = logical % len(owned)
                    previous = owned[slot]
                    block = pool.block_from_local_id(partition, local)
                    if previous is block:
                        continue
                    block.ref_cnt += 1
                    owned[slot] = block
                    previous.ref_cnt -= 1
                    if previous.ref_cnt == 0:
                        pool.free_queues[partition].append(previous)
                tables[name] = [pool.local_block_id(block) for block in owned]
            preferred = tuple(sorted(b for b in records if b in (replay_boundary, shared_boundary)))
            return CheckpointWritePlan(partition, tuple(sorted(records)), preferred,
                                       original, publications, records, snapshot)
        finally:
            for pool, block in fresh:
                block.ref_cnt -= 1
                if block.ref_cnt == 0:
                    pool.free_queues[partition].append(block)

    def complete(self, request, hashes, tokens, plan):
        if plan is None:
            return
        manager = self.manager
        partition = plan.partition
        for boundary, members in plan.records.items():
            if boundary > tokens:
                raise RuntimeError("Checkpoint publication precedes compute completion")
            for name, pages in members.items():
                pool = manager._group_pools[name]
                values = hashes[name]
                for logical, local in pages.items():
                    manager._cache_group_block(pool, partition, pool.block_from_local_id(partition, local),
                                               values[logical])
            manager._protect_group_checkpoint(request, hashes, tokens, partition, {}, boundary,
                                               preferred=boundary in plan.preferred)


class GroupPrefixCheckpoints:
    def __init__(self):
        self.checkpoints = {}
        self.pages = {}
        self.owners = {}
        self.histories = {}
        self.preferred = set()
        self.history_changed = False

    def publish(self, owner, pools, hashes, tokens, partition, boundary, valid_from, *, preferred=True):
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
        if preferred:
            self.preferred.add(key)
        if owner is not None:
            self.owners.setdefault(owner, set()).add(key)
        if key not in self.checkpoints:
            self.checkpoints[key] = tuple(members)
            self.histories[key] = history
            for page in members:
                self.pages.setdefault(page, set()).add(key)

    def release(self, owner):
        # Healthy sparse boundaries remain evictable after request retirement.
        self.owners.pop(owner, None)

    def victim(self, pool, queue):
        block = queue.head
        periodic = None
        while block is not None:
            keys = self.pages.get((pool.spec.name, block.block_id))
            if not keys:
                return block
            if periodic is None and not self.preferred.intersection(keys):
                periodic = block
            block = block.next_free
        return periodic or queue.head  # Bounded cache: checkpoints never prevent admission.

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
        self.preferred.discard(key)
        self.histories.pop(key, None)
        for owner in tuple(self.owners):
            self.owners[owner].discard(key)
            if not self.owners[owner]:
                del self.owners[owner]
        for page in self.checkpoints.pop(key, ()):
            owners = self.pages[page]
            owners.remove(key)
            if not owners:
                del self.pages[page]
