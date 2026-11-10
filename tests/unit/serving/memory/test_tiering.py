# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from concurrent.futures import Future

import pytest

from pypto_serving.serving.memory.tiering.manager import CacheTier
from pypto_serving.serving.memory.tiering.policy import LRU


def finished(value):
    result = Future()
    result.set_result(value)
    return result


class Backend:
    def __init__(self):
        self.live, self.released = {}, []

    def reserve(self, handle, source):
        return True

    def store(self, handle, size):
        self.live[handle] = size
        return finished(dict(bytes=size, stopped=True, native_completed=True))

    def release(self, handle):
        self.released.append(handle)
        del self.live[handle]
        return finished(None)

    def load(self, handle, destination):
        return finished(dict(bytes=self.live[handle], stopped=True, native_completed=True))

    def close(self):
        self.live.clear()


def setup(capacity=4):
    released = []
    tier = CacheTier(capacity, Backend(), released.append)
    return tier, released


def capture(tier, key, size=2):
    tier.capture(key, size, size, key)


def settle(tier):
    for _ in range(4):
        tier.pump()


def test_lru_touch_miss_remove_and_policy_is_separate():
    policy = LRU()
    policy.on_publish("a", 0)
    policy.on_publish("b", 1)
    policy.on_access("a", 2)
    policy.on_access("missing", 3)
    assert policy.candidates(4) == ("b", "a")
    policy.on_remove("b")
    assert policy.candidates(5) == ("a",)
    assert not hasattr(policy, "backend")


def test_store_dedup_completion_and_reader_lifetime():
    tier, released = setup()
    capture(tier, "a")
    tier.capture("a", 2, 2, "duplicate")
    assert released == ["duplicate"] and tier.lookup("a") is None
    settle(tier)
    assert released == ["duplicate", "a"]
    readers = tier.acquire(["a", "a"])
    assert len(readers) == 1
    tier.completed(tier.load(readers[0], "target"), 2)
    tier.release(readers)
    with pytest.raises(RuntimeError, match="stale"):
        tier.release(readers)


def test_lru_skips_active_reader_and_reclaims_only_required_space():
    tier, _ = setup()
    capture(tier, "a")
    capture(tier, "b")
    settle(tier)
    reader = tier.acquire(["a"])
    capture(tier, "c")
    settle(tier)
    assert tier.lookup("a") and tier.lookup("c") and tier.lookup("b") is None
    assert len(tier.backend.released) == 1 and tier.used == 4
    tier.release(reader)


@pytest.mark.parametrize("incoming_size", (2, 4))
def test_pending_native_retirement_prevents_excess_eviction(incoming_size):
    tier, _ = setup(6)
    for key in ("a", "b", "c"):
        capture(tier, key)
    settle(tier)
    pending = {}

    def release(handle):
        pending[handle] = Future()
        return pending[handle]

    tier.backend.release = release
    capture(tier, "next", incoming_size)
    for _ in range(10):
        tier.pump()
    assert len(pending) == incoming_size // 2
    assert tier.lookup("c") and tier.used == 6
    assert len(tier.queued) == 1 and tier.lookup("next") is None
    for handle, future in pending.items():
        del tier.backend.live[handle]
        future.set_result(None)
    settle(tier)
    assert tier.lookup("next") and tier.lookup("c") and tier.used == 6


def test_checkpoint_members_evict_together_and_busy_load_can_progress():
    tier, _ = setup()
    capture(tier, "kv")
    capture(tier, "state")
    tier.publish("checkpoint", ("kv", "state"))
    settle(tier)
    reader = tier.acquire(["state"])
    capture(tier, "next")
    settle(tier)
    assert len(tier.queued) == 1 and tier.used == 4
    tier.completed(tier.load(reader[0], "target"), 2)
    tier.release(reader)
    settle(tier)
    assert "checkpoint" not in tier.units
    assert tier.lookup("kv") is None and tier.lookup("state") is None
    assert tier.lookup("next") and tier.used == 2


def test_shared_history_bytes_are_not_reclaimed_twice():
    tier, _ = setup(6)
    for key in ("history", "state1", "state2"):
        capture(tier, key)
    tier.publish("c1", ("history", "state1"))
    tier.publish("c2", ("history", "state2"))
    settle(tier)
    reader = tier.acquire(["state2"])
    capture(tier, "next")
    settle(tier)
    assert tier.lookup("history") and tier.lookup("state2") and tier.lookup("next")
    assert tier.lookup("state1") is None and tier.used == 6
    tier.release(reader)


def test_incomplete_checkpoint_does_not_deadlock_a_smaller_ddr_pool():
    tier, released = setup(2)
    for key in ("history", "state"):
        capture(tier, key)
    tier.publish("larger-than-pool", ("history", "state"))
    for _ in range(8):
        tier.pump()
    assert tier.counters["stores"] == 2 and not tier.queued
    assert sorted(released) == ["history", "state"]
    assert "larger-than-pool" not in tier.units
    assert not tier.pending_units and not tier.pending_by_page


def test_unlimited_queue_does_not_drop_or_duplicate_pins():
    tier, released = setup(4)
    for key in range(300):
        capture(tier, key, 1)
    assert len(tier.queued) == 300 and not released
    for _ in range(1000):
        tier.pump()
        if not tier.queued and not tier.inflight:
            break
    assert not tier.queued and not tier.inflight
    assert sorted(released) == list(range(300))
    assert tier.counters["stores"] == 300


@pytest.mark.parametrize("proof", [dict(bytes=2, stopped=False, native_completed=True),
                                   dict(bytes=1, stopped=True, native_completed=True),
                                   dict(bytes=2, stopped=True, native_completed=False)])
def test_unknown_or_partial_native_result_retains_sources(proof):
    tier, released = setup()
    tier.backend.store = lambda handle, source: finished(proof)
    capture(tier, "a")
    tier.pump()
    with pytest.raises(RuntimeError, match="proof"):
        tier.pump()
    assert not released and tier.used == 2 and tier.lookup("a") is None
    with pytest.raises(RuntimeError, match="quarantined"):
        tier.pump()


def test_normal_close_releases_queued_and_completed_entries_once():
    tier, released = setup(2)
    capture(tier, "a")
    capture(tier, "b")
    tier.pump()
    tier.begin_close()
    for _ in range(5):
        if tier.close_ready():
            break
    assert tier.pages and tier.used == 2  # Residency stays until the backend ACK.
    tier.backend.close()
    tier.finish_close()
    assert not tier.pages and not tier.retiring and tier.used == 0
    assert sorted(released) == ["a", "b"] and not tier.backend.live


def test_close_waits_for_current_readers_before_retiring_storage():
    tier, _ = setup()
    capture(tier, "a")
    settle(tier)
    reader = tier.acquire(["a"])
    tier.begin_close()
    assert not tier.close_ready()
    with pytest.raises(RuntimeError, match="must drain"):
        tier.finish_close()
    tier.release(reader)
    assert tier.close_ready() and tier.lookup("a") is not None
    tier.backend.close()
    tier.finish_close()
