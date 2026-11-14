# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Logical checkpoint membership and borrowed replicas; physical ownership stays in each store."""

from dataclasses import dataclass
import time


@dataclass(frozen=True)
class Replica:
    store: object
    page: object

    @property
    def key(self):
        return self.page.key

    @property
    def size(self):
        return self.page.size


class CheckpointIndex:
    """Candidates only: the existing matcher must still verify every member."""

    def __init__(self):
        self.units, self.members = {}, {}

    def publish(self, identity, keys):
        if identity in self.units or not keys:
            return
        self.units[identity] = tuple(dict.fromkeys(keys))
        for key in self.units[identity]:
            self.members.setdefault(key, set()).add(identity)

    def remove(self, key):
        for identity in tuple(self.members.get(key, ())):
            for member in self.units.pop(identity):
                owners = self.members[member]
                owners.remove(identity)
                if not owners:
                    del self.members[member]


class DDRSource:
    """Adapt the existing DDR lifetime manager to the common restore-source contract."""

    name = "ddr"

    def __init__(self, tier):
        self.tier = tier

    def contains(self, key):
        return key in self.tier.pages

    def has_pages(self):
        return bool(self.tier.pages)

    def lookup(self, key):
        return self.tier.lookup(key)

    def touch(self, key):
        self.tier.touch(key)

    def acquire(self, keys):
        return {page.key: page for page in self.tier.acquire(keys)}

    def release(self, reader):
        self.tier.release((reader,))

    def load(self, page, destination, reader, *, request=None):
        if reader is not page:
            raise ValueError("DDR replica does not match its reader")
        return self.tier.load(page, destination, request=request)

    def pop_loaded(self):
        for handle in self.tier.backend.pop_completed("load"):
            yield self.tier.by_handle[handle].key

    def snapshot(self):
        tier = self.tier
        return dict(capacity_bytes=tier.capacity, used_bytes=tier.used, pages=len(tier.pages),
                    queued_pages=len(tier.queued), store_inflight=len(tier.inflight),
                    retiring_pages=len(tier.retiring), joint_checkpoints=len(tier.units),
                    pending_checkpoints=len(tier.pending_units), counters=dict(tier.counters),
                    oldest_store_age_ns=time.monotonic_ns()-tier.queued[0].queued_ns if tier.queued else 0)
