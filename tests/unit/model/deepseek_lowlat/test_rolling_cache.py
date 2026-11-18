# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from types import SimpleNamespace

import pytest
import torch

from pypto_serving.model.deepseek_lowlat.cache import LowlatCachePool, TABLE_GROUPS
from pypto_serving.model.deepseek_lowlat.contract import build_lowlat_cache_group_specs
from pypto_serving.serving.memory.kv_cache import KvCacheManager


def fixture(chunk=512):
    specs = build_lowlat_cache_group_specs(max_seq_len=16384, max_prefill_tokens=chunk, layout="rolling")
    metadata = {"params": [{"name": name, "shape": [8, 1, 16384//next(s.spec.token_capacity for s in specs
                           if s.name == group)], "dtype": "int32", "direction": "In"}
                          for name, group in TABLE_GROUPS.items()]}
    cache = LowlatCachePool(SimpleNamespace(single_metadata=metadata), None, specs)
    cache.begin_generation(1)
    initialized = []
    cache._initialize_pages = lambda worker, name, root, pages: initialized.append((name, tuple(pages[0])))
    cache.owners = {name: object() for name in cache.initialized}
    cache.arena = SimpleNamespace(worker=object())
    manager = KvCacheManager(enable_prefix_cache=False)
    manager.init_groups(specs, max_batch_size=1)
    return cache, manager, initialized


def test_scheduler_and_renderer_reuse_physical_slots_without_reinitializing_live_pages():
    cache, manager, initialized = fixture()
    assert cache.counts == dict(ori=6, cmp_c128=128, cmp_c4=128, idx=128,
                               hca_state=18, csa_state=4, csa_inner_state=4)
    assert cache.specs["csa_state"].sliding_window == 8
    identities = None
    for end in range(512, 16385, 512):
        groups = manager.ensure_group_blocks("r", end)
        tables, counts = cache.tables_for(groups, input_end=end)
        for index, (name, group) in enumerate(TABLE_GROUPS.items()):
            pages = groups[group]
            assert len(pages) == min((end+cache.specs[group].spec.token_capacity-1)
                                    //cache.specs[group].spec.token_capacity, cache.counts[group])
            assert counts[0, 0, index] == len(pages)
            if cache.specs[group].sliding_window is not None:
                expected = torch.tensor(pages)[torch.arange(tables[name].shape[-1]) % len(pages)]
                assert torch.equal(tables[name][0, 0], expected)
                assert tables[name].ge(0).all()
        if end >= 1024:
            current = {g: tuple(groups[g]) for g in ("ori", "hca_state", "csa_state", "csa_inner_state")}
            if identities is not None:
                assert current == identities
            identities = current
    for tensor in cache.initialized:
        pages = [page for name, row in initialized if name == tensor for page in row]
        assert len(pages) == len(set(pages))
    with pytest.raises(ValueError, match="rewind"):
        cache.tables_for(groups, input_end=512)
    manager.release_all_group_requests("r")
    cache.release()
    cache.begin_generation(2)
    previous = len(initialized)
    groups = manager.ensure_group_blocks("next", 512)
    cache.tables_for(groups, input_end=512)
    assert len(initialized) > previous


@pytest.mark.parametrize("chunk", [128, 512, 8192])
def test_raw_and_carry_windows_do_not_alias_the_current_publication_write_set(chunk):
    cache, manager, _ = fixture(chunk)
    groups = manager.ensure_group_blocks("r", 16384)
    tables, _ = cache.tables_for(groups, input_end=16384)
    # Private prefill has DECODE_SEQ8. K5's six rows fit the same transaction bound.
    windows = {"ori": (127, chunk), "hca_state": (128, 8),
               "csa_state": (8, 8), "csa_inner_state": (8, 8)}
    for table_name, group in TABLE_GROUPS.items():
        if group not in windows:
            continue
        history, writes = windows[group]
        page_rows = cache.specs[group].spec.token_capacity
        table = tables[table_name][0, 0]
        for anchor in range(128, 1024):
            positions = range(max(0, anchor-history+1), anchor+writes+1)
            physical = [int(table[position//page_rows])*page_rows+position%page_rows for position in positions]
            assert len(physical) == len(set(physical)), (group, chunk, anchor)


def test_compressed_history_capacity_is_retained_and_storage_is_not_a_fake_shape():
    cache, _, _ = fixture(8192)
    assert cache.counts["ori"] == 66
    assert all(cache.counts[g] == 128 for g in ("cmp_c128", "cmp_c4", "idx"))
    # The original draft allows output beyond its16K prefix and uses129compressed pages.
    # This serving profile caps the entire request at16K and needs128.
    assert cache.bytes_per_rank() == 489275392
    full = build_lowlat_cache_group_specs(max_seq_len=16384)
    assert cache.bytes_per_rank() < sum(s.num_blocks*s.spec.page_size_bytes for s in full)


def test_actual_permuted_physical_page_ids_drive_the_entire_rolling_table():
    cache, manager, _ = fixture()
    groups = manager.ensure_group_blocks("r", 16384)
    groups = {name: list(reversed(ids)) for name, ids in groups.items()}
    tables, counts = cache.tables_for(groups, input_end=16384)
    for index, (name, group) in enumerate(TABLE_GROUPS.items()):
        assert counts[0, 0, index] == len(groups[group])
        if cache.specs[group].sliding_window is not None:
            expected = torch.tensor(groups[group])[torch.arange(tables[name].shape[-1]) % len(groups[group])]
            assert torch.equal(tables[name][0, 0], expected)
