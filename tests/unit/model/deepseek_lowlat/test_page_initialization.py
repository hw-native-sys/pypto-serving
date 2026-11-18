# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compare initialized byte ranges against the unmodified library helper."""

import ctypes
import hashlib
from math import prod
from types import SimpleNamespace

import pytest
import torch

from pypto_serving.model.deepseek_lowlat.cache import LowlatCachePool, TABLE_GROUPS, TENSOR_GROUPS
from pypto_serving.model.deepseek_lowlat.contract import build_lowlat_cache_group_specs
from pypto_serving.model.deepseek_lowlat.initialization import PageInitTemplates
from pypto_serving.model.deepseek_lowlat.kernel_loader import isolated_profile


@pytest.fixture(scope="module")
def library():
    with isolated_profile():
        from models.deepseek_v4_flash_lowlat import prefill_cache
        return prefill_cache


class Copies:
    def __init__(self):
        self.calls = []
        self.sources = []

    def copy_to(self, destination, source, size, *, dst_offset, worker_id):
        digest = hashlib.sha256(ctypes.string_at(source, size)).hexdigest()
        self.calls.append((destination, size, dst_offset, worker_id, digest))
        self.sources.append((source, size))


def root_for(name, layouts, specs):
    layers, rows, _, tail, dtype = layouts[name]
    blocks = next(s.num_blocks for s in specs if s.name == TENSOR_GROUPS[name])
    shape = (8, layers*blocks, rows, *tail)
    size = prod(shape[1:])*torch.empty((), dtype=dtype).element_size()
    shards = [SimpleNamespace(data_ptr=0x100000000+rank*0x10000000, byte_offset=0, nbytes=size)
              for rank in range(8)]
    return SimpleNamespace(shape=shape, dtype=dtype, worker_ids=tuple(range(8)), shards=shards)


@pytest.mark.parametrize("name", list(TENSOR_GROUPS))
def test_templates_match_original_payload_offsets_and_coalesced_calls(library, name):
    specs = build_lowlat_cache_group_specs(max_seq_len=512)
    template = PageInitTemplates(library.CACHE_LAYOUTS, TENSOR_GROUPS, specs, 512)
    root = root_for(name, library.CACHE_LAYOUTS, specs)
    pages = [[0, 2, 3] if rank%2==0 else [1, 2] for rank in range(8)]
    expected, actual = Copies(), Copies()
    library.initialize_cache_pages(expected, name, root, pages)
    template.initialize(actual, name, root, pages)
    assert actual.calls == expected.calls
    host = template.hosts[name]
    first, last = host.data_ptr(), host.data_ptr()+host.numel()*host.element_size()
    assert host.is_shared()
    assert all(first <= pointer and pointer+size <= last for pointer, size in actual.sources)


def test_full_pool_grants_do_not_split_more_than_original_64mib_limit(library):
    specs = build_lowlat_cache_group_specs(max_seq_len=16384)
    template = PageInitTemplates(library.CACHE_LAYOUTS, TENSOR_GROUPS, specs, 128)
    name = "csa_compress_state"
    host = template.hosts[name]
    assert host.numel()*host.element_size() == 64*1024**2
    assert host.shape[0] == 2048  # Half of4096 pages; two calls per layer/rank, like the original.
    assert all(v.numel()*v.element_size() <= template.MAX_COPY_BYTES for v in template.hosts.values())


def test_invalid_rank_page_or_owner_fails_before_any_copy(library):
    specs = build_lowlat_cache_group_specs(max_seq_len=512)
    template = PageInitTemplates(library.CACHE_LAYOUTS, TENSOR_GROUPS, specs, 512)
    name = "hca_compress_state"
    root = root_for(name, library.CACHE_LAYOUTS, specs)
    worker = Copies()
    pages = [[0] for _ in range(8)]
    pages[-1] = [True]
    with pytest.raises(ValueError, match="unique in-range"):
        template.initialize(worker, name, root, pages)
    assert not worker.calls
    with pytest.raises(ValueError, match="owner/rank"):
        template.initialize(worker, name, root, [[]])
    assert not worker.calls


def test_failed_family_cannot_acknowledge_group_ownership_and_release_invalidates(library):
    specs = build_lowlat_cache_group_specs(max_seq_len=512)
    metadata = {"params": [dict(name=name, shape=[8, 1, 512], dtype="int32", direction="In")
                           for name in TABLE_GROUPS]}
    cache = LowlatCachePool(SimpleNamespace(single_metadata=metadata), SimpleNamespace(worker=None), specs)
    cache.owners = {name: object() for name in TENSOR_GROUPS}
    cache.begin_generation(1)
    completed = []

    def initialize(worker, name, root, pages):
        if name == "idx_kv_scale":
            raise RuntimeError("scale copy failed")
        completed.append(name)

    cache._initialize_pages = initialize
    groups = {name: [0] for name in cache.specs}
    with pytest.raises(RuntimeError, match="scale copy failed"):
        cache._initialize_new_pages(groups)
    assert not any(cache.owned_pages.values())
    assert all(pages == {0} for pages in cache.initialized["idx_kv_cache"])
    assert not any(cache.initialized["idx_kv_scale"])
    cache.release()
    assert cache.generation is None and not any(any(ranks) for ranks in cache.initialized.values())
    cache.begin_generation(2)
    assert cache.generation == 2


def test_private_priming_initializes_future_storage_without_claiming_logical_pages():
    specs = build_lowlat_cache_group_specs(max_seq_len=512)
    metadata = {"params": [dict(name=name, shape=[8, 1, 512], dtype="int32", direction="In")
                           for name in TABLE_GROUPS]}
    cache = LowlatCachePool(SimpleNamespace(single_metadata=metadata), SimpleNamespace(worker=None), specs)
    cache.owners = {name: object() for name in TENSOR_GROUPS}
    calls = []
    cache._initialize_pages = lambda worker, name, root, pages: calls.append((name, pages))
    cache.begin_generation(1)
    cache.initialize_private_pool(1)
    assert len(calls) == 8
    assert not any(cache.owned_pages.values())
    assert all(all(len(pages) == cache.counts[TENSOR_GROUPS[name]] for pages in ranks)
               for name, ranks in cache.initialized.items())
    with pytest.raises(RuntimeError, match="idle new generation"):
        cache.initialize_private_pool(1)
    calls.clear()
    cache.tables_for({name: [0] for name in cache.specs}, input_end=1)
    assert not calls
    assert all(pages == {0} for pages in cache.owned_pages.values())
    cache.release()
    cache.begin_generation(2)
    cache.tables_for({name: [0] for name in cache.specs}, input_end=1)
    assert len(calls) == 8  # A new request may not inherit previous initialization.
