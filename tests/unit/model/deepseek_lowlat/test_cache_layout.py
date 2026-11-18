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

from pypto_serving.model.deepseek_lowlat.cache import LowlatCachePool, TABLE_GROUPS
from pypto_serving.model.deepseek_lowlat.contract import build_lowlat_cache_group_specs


def pool():
    specs = build_lowlat_cache_group_specs(max_seq_len=512)
    metadata = {
        "params": [
            {"name": name, "shape": [8, 1, 512], "dtype": "int32", "direction": "In"} for name in TABLE_GROUPS
        ]
    }
    cache = LowlatCachePool(SimpleNamespace(single_metadata=metadata), None, specs)
    cache._initialize_new_pages = lambda groups: None
    return cache


def test_capacity_reserves_a_complete_request_in_every_group():
    cache = pool()
    assert cache.counts == {
        "ori": 4,
        "cmp_c128": 4,
        "cmp_c4": 4,
        "idx": 4,
        "hca_state": 64,
        "csa_state": 128,
        "csa_inner_state": 128,
    }
    assert cache.specs["ori"].spec.page_size_bytes == 43 * 128 * 512 * 2
    assert cache.specs["idx"].spec.page_size_bytes == 21 * 32 * (128 + 4)
    assert all(s.sliding_window is None for s in cache.specs.values())


def test_tables_preserve_noncontiguous_scheduler_ids_and_invalid_tail():
    cache = pool()
    groups = {name: [cache.counts[name] - 1] for name in cache.counts}
    tables, counts = cache.tables_for(groups, input_end=1)
    assert tables["block_table"][:, 0, 0].tolist() == [3] * 8
    assert tables["block_table"][:, 0, 1].tolist() == [-1] * 8
    assert counts[0, 0].tolist() == [1] * 7


def test_missing_history_and_foreign_physical_pages_fail():
    cache = pool()
    groups = {name: [0] for name in cache.counts}
    with pytest.raises(ValueError, match="cover"):
        cache.tables_for(groups, input_end=129)
    groups["ori"] = [cache.counts["ori"]]
    with pytest.raises(ValueError, match="cover"):
        cache.tables_for(groups, input_end=1)


def test_table_reuse_tracks_mutated_page_lists_and_clears_removed_entries():
    cache = pool()
    groups = {name: [0, 1] for name in cache.counts}
    tables, counts = cache.tables_for(groups, input_end=1)
    identity = id(tables["block_table"])
    groups["ori"][:] = [3]
    again, current_counts = cache.tables_for(groups, input_end=1)
    assert id(again["block_table"]) == identity
    assert again["block_table"][0, 0, :3].tolist() == [3, -1, -1]
    assert current_counts[0, 0, 0].item() == 1
    cache.release()
    assert not cache._page_ids


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_capacity_ledger_matches_selected_lib_before_allocating_host_templates(monkeypatch):
    from dataclasses import replace
    from unittest.mock import Mock
    from pypto_serving.model.deepseek_lowlat.kernel_loader import isolated_profile
    from pypto_serving.model.deepseek_lowlat import initialization

    cache = pool()
    cache.programs.context = isolated_profile
    templates = Mock(return_value=SimpleNamespace(inherited=[]))
    monkeypatch.setattr(initialization, "PageInitTemplates", templates)
    assert cache.prepare_host_templates(512) == []
    templates.assert_called_once()
    templates.reset_mock()
    group = cache.specs["ori"]
    cache.specs["ori"] = replace(group, spec=replace(group.spec, page_size_bytes=group.spec.page_size_bytes+1))
    with pytest.raises(ValueError, match="selected lib cache layouts"):
        cache.prepare_host_templates(512)
    templates.assert_not_called()
