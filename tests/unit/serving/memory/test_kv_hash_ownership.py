# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Protect shared prefix lookup when concurrent prefills publish duplicate hashes."""

import pytest

from pypto_serving.serving.memory.kv_cache import KvCacheManager


@pytest.mark.parametrize("owner_is_running", [False, True])
def test_evicting_duplicate_preserves_other_block_hash_owner(owner_is_running):
    manager = KvCacheManager(num_blocks=3, block_size=2)
    tokens = [1, 2, 3]
    old, owner = manager.allocate_blocks(2)
    block_hash = manager.compute_block_hashes(tokens)[0]
    manager.cache_block(old, block_hash)
    manager.cache_block(owner, block_hash)
    manager.release(old)
    if not owner_is_running:
        manager.release(owner)
    # The unused block comes first, followed by the old duplicate; the mapped
    # owner stays physically valid whether it is running or merely reusable.
    allocated = manager.allocate_blocks(2)
    assert old in allocated
    assert owner not in allocated
    assert manager.observe_prefix(tokens).cached_tokens == 2
    assert manager.hash_to_block[block_hash] is owner


def test_republishing_duplicate_preserves_other_block_hash_owner():
    manager = KvCacheManager(num_blocks=3, block_size=2)
    old_tokens, new_tokens = [1, 2, 3], [8, 9, 10]
    old, owner = manager.allocate_blocks(2)
    old_hash = manager.compute_block_hashes(old_tokens)[0]
    new_hash = manager.compute_block_hashes(new_tokens)[0]
    manager.cache_block(old, old_hash)
    manager.cache_block(owner, old_hash)
    manager.cache_block(old, new_hash)
    assert manager.observe_prefix(old_tokens).cached_tokens == 2
    assert manager.observe_prefix(new_tokens).cached_tokens == 2
    assert manager.hash_to_block[old_hash] is owner
