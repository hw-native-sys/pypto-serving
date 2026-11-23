# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Scheduler-owned logical pages backed by one set of TP8 cache allocations."""

import torch
from math import prod

from .bindings import parameter_rows


TENSOR_GROUPS = {
    "kv_cache": "ori",
    "hca_cmp_kv": "cmp_c128",
    "csa_cmp_kv": "cmp_c4",
    "idx_kv_cache": "idx",
    "idx_kv_scale": "idx",
    "hca_compress_state": "hca_state",
    "csa_compress_state": "csa_state",
    "csa_inner_compress_state": "csa_inner_state",
}
TABLE_GROUPS = {
    "block_table": "ori",
    "hca_cmp_block_table": "cmp_c128",
    "csa_cmp_block_table": "cmp_c4",
    "idx_block_table": "idx",
    "hca_compress_state_block_table": "hca_state",
    "csa_compress_state_block_table": "csa_state",
    "csa_inner_compress_state_block_table": "csa_inner_state",
}


class LowlatCachePool:
    def __init__(self, programs, arena, specs):
        self.programs, self.arena = programs, arena
        self.specs = {spec.name: spec for spec in specs}
        self.counts = {spec.name: spec.num_blocks for spec in specs}
        self.owners = {}
        self.owned_pages = {name: set() for name in self.specs}
        self.rows = parameter_rows(programs.single_metadata)
        self._tables = {
            name: torch.full(tuple(self.rows[name]["shape"]), -1, dtype=torch.int32) for name in TABLE_GROUPS
        }
        self._counts = torch.zeros((8, 1, 7), dtype=torch.int32)
        self._page_ids = {}
        self._initialize_pages = None
        self.templates = None
        self.generation = None
        self.prepared_input_end = 0
        self.preserve_contents = False
        self.page_epochs = {}
        self.initialized = {name: [set() for _ in range(8)] for name in TENSOR_GROUPS}

    def checkpoint_ranges(self, boundary):
        from pypto_serving.serving.memory.protocol import GroupPageRange

        return tuple(
            GroupPageRange(
                name,
                max(0, boundary // s.spec.token_capacity - s.sliding_window // s.spec.token_capacity)
                if s.sliding_window
                else 0,
                boundary // s.spec.token_capacity,
            )
            for name, s in self.specs.items()
        )

    def adopt_page_identities(self, identities, *, retained=False):
        if not self.preserve_contents:
            raise RuntimeError("Page epochs require the confirmed prefix protocol")
        seen = set()
        for identity in identities:
            identity.__post_init__()
            key = identity.group, identity.page_id
            if (
                key in seen
                or identity.group not in self.specs
                or identity.partition != 0
                or identity.page_id >= self.counts[identity.group]
            ):
                raise ValueError("Invalid or duplicate lowlat page identity")
            seen.add(key)
            old = self.page_epochs.get(key)
            if retained:
                if old != identity.allocation_epoch or any(
                    identity.page_id not in pages
                    for name, ranks in self.initialized.items()
                    if TENSOR_GROUPS[name] == identity.group
                    for pages in ranks
                ):
                    raise ValueError("Retained prefix page content/epoch is not initialized")
            elif old is not None and identity.allocation_epoch < old:
                raise ValueError("Lowlat page allocation epoch cannot rewind")
        for identity in identities:
            key = identity.group, identity.page_id
            if self.page_epochs.get(key) != identity.allocation_epoch:
                for name, ranks in self.initialized.items():
                    if TENSOR_GROUPS[name] == identity.group:
                        for pages in ranks:
                            pages.discard(identity.page_id)
                self.page_epochs[key] = identity.allocation_epoch

    def prepare_host_templates(self, chunk_capacity):
        from .initialization import PageInitTemplates

        with self.programs.context():
            from models.deepseek_v4_flash_lowlat.prefill_cache import CACHE_LAYOUTS

            actual = {group: 0 for group in self.specs}
            for name, (layers, rows, _, tail, dtype) in CACHE_LAYOUTS.items():
                actual[TENSOR_GROUPS[name]] += layers * rows * prod(tail) * dtype.itemsize
            if any(actual[name] != spec.spec.page_size_bytes for name, spec in self.specs.items()):
                raise ValueError("Lowlat capacity bytes differ from the selected lib cache layouts")
            self.templates = PageInitTemplates(CACHE_LAYOUTS, TENSOR_GROUPS,
                                              self.specs.values(), chunk_capacity)
        return self.templates.inherited

    def begin_generation(self, generation):
        if self.generation is not None or any(self.owned_pages.values()):
            raise RuntimeError("cache still belongs to another request generation")
        if type(generation) is not int or generation <= 0:
            raise ValueError("a positive cache generation is required")
        self.generation = generation
        self.prepared_input_end = 0

    def initialize_private_pool(self, generation):
        """Initialize private B1 storage without claiming future scheduler pages."""
        if (generation != self.generation or any(self.owned_pages.values())
                or any(any(ranks) for ranks in self.initialized.values())
                or any(spec.num_partitions != 1 or spec.max_requests_per_partition != 1
                       for spec in self.specs.values())):
            raise RuntimeError("private pool initialization requires an idle new generation")
        for tensor, group in TENSOR_GROUPS.items():
            pages = list(range(self.counts[group]))
            self._initialize_pages(self.arena.worker, tensor, self.owners[tensor], [pages] * 8)
            for initialized in self.initialized[tensor]:
                initialized.update(pages)

    def bytes_per_rank(self):
        return sum(spec.num_blocks * spec.spec.page_size_bytes for spec in self.specs.values())

    def allocate(self):
        with self.programs.context():
            from models.deepseek_v4_flash_lowlat.prefill_cache import CACHE_LAYOUTS

            self._initialize_pages = self.templates.initialize

            for name, (layers, rows, _, tail, dtype) in CACHE_LAYOUTS.items():
                blocks = self.counts[TENSOR_GROUPS[name]]
                self.owners[name] = self.arena.allocate((8, layers * blocks, rows, *tail), dtype)

    def tables_for(self, groups, *, input_end):
        # Validate every group before mutating tables or initializing any page.
        rolling = any(spec.sliding_window is not None for spec in self.specs.values())
        if rolling and input_end < self.prepared_input_end:
            raise ValueError("rolling input extent cannot rewind within a generation")
        selected = {}
        for index, (name, group) in enumerate(TABLE_GROUPS.items()):
            if group not in groups:
                raise ValueError(f"missing scheduler cache group: {group}")
            pages = list(groups[group])
            spec = self.specs[group]
            required = (input_end + spec.spec.token_capacity - 1) // spec.spec.token_capacity
            if spec.sliding_window is not None:
                required = min(required, spec.max_blocks_per_seq)
            if (
                len(pages) < required
                or len(pages) > self.rows[name]["shape"][-1]
                or len(set(pages)) != len(pages)
                or any(type(p) is not int or not 0 <= p < self.counts[group] for p in pages)
            ):
                raise ValueError(f"lowlat scheduler pages do not cover {group} input_end={input_end}")
            selected[group] = tuple(pages)
        for index, (name, group) in enumerate(TABLE_GROUPS.items()):
            pages = selected[group]
            if self._page_ids.get(group) != pages:
                table = self._tables[name]
                table.fill_(-1)
                if pages:
                    values = torch.tensor(pages, dtype=torch.int32)
                    if self.specs[group].sliding_window is not None:
                        # History readers index absolute logical pages; writers use modulo count.
                        logical = torch.arange(table.shape[-1], dtype=torch.int64)
                        table[:, 0] = values[logical.remainder(len(pages))]
                    else:
                        table[:, 0, :len(pages)] = values
                self._counts[:, 0, index] = len(pages)
                self._page_ids[group] = pages
        self._initialize_new_pages(groups)
        self.prepared_input_end = max(self.prepared_input_end, input_end)
        return self._tables, self._counts

    def _initialize_new_pages(self, groups):
        if self.generation is None:
            raise RuntimeError("cache initialization requires an active generation")
        new = {name: sorted(set(groups[name]) - self.owned_pages[name]) for name in self.specs}
        for tensor, group in TENSOR_GROUPS.items():
            pending = [sorted(set(new[group]) - initialized) for initialized in self.initialized[tensor]]
            if any(pending):
                self._initialize_pages(self.arena.worker, tensor, self.owners[tensor], pending)
                for initialized, pages in zip(self.initialized[tensor], pending, strict=True):
                    initialized.update(pages)
        for group in self.specs:
            self.owned_pages[group].update(new[group])

    def release(self):
        # The next generation must initialize each page it newly owns, even
        # when the scheduler reuses exactly the same physical IDs.
        for pages in self.owned_pages.values():
            pages.clear()
        self._page_ids.clear()
        self.generation = None
        self.prepared_input_end = 0
        if not self.preserve_contents:
            for ranks in self.initialized.values():
                for pages in ranks:
                    pages.clear()
