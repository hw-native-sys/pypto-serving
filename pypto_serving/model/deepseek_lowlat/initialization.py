# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded pre-fork cache initializers with the original layer/page semantics."""

from math import prod

import torch

from pypto.runtime import ReadOnlyHostTensor


class PageInitTemplates:
    MAX_COPY_BYTES = 64 * 1024**2

    def __init__(self, layouts, tensor_groups, specs, chunk_capacity):
        self.layouts = dict(layouts)
        self.groups = dict(tensor_groups)
        self.specs = {spec.name: spec for spec in specs}
        if type(chunk_capacity) is not int or not 1 <= chunk_capacity <= 8192:
            raise ValueError("a supported prefill chunk capacity is required")
        self.hosts = {}
        self.page_bytes = {}
        for name, (_, rows, _, tail, dtype) in self.layouts.items():
            spec = self.specs[self.groups[name]]
            nbytes = prod((rows, *tail)) * torch.empty((), dtype=dtype).element_size()
            if nbytes > self.MAX_COPY_BYTES:
                raise ValueError(f"one cache page exceeds the maximum copy payload: {name}")
            # A scheduler may grant pages ahead of the current chunk. Cover
            # every possible run without adding copies beyond the vendor cap.
            pages = min(spec.num_blocks, self.MAX_COPY_BYTES // nbytes)
            if pages <= 0:
                raise ValueError(f"cache template needs positive capacity: {name}")
            host = torch.zeros((pages, rows, *tail), dtype=dtype).share_memory_()
            if name.endswith("compress_state"):
                host[..., host.shape[-1] // 2:] = torch.finfo(torch.float32).min
            self.hosts[name] = host
            self.page_bytes[name] = nbytes

    @property
    def inherited(self):
        return [ReadOnlyHostTensor(host) for host in self.hosts.values()]

    def initialize(self, worker, name, root, pages_by_rank):
        if name not in self.layouts:
            raise ValueError(f"unknown cache family: {name}")
        layers, rows, _, tail, dtype = self.layouts[name]
        blocks = self.specs[self.groups[name]].num_blocks
        if (tuple(root.shape) != (8, layers * blocks, rows, *tail) or root.dtype != dtype
                or tuple(root.worker_ids) != tuple(range(8)) or len(pages_by_rank) != 8):
            raise ValueError(f"cache initializer owner/rank geometry mismatch: {name}")
        selected = []
        for pages in pages_by_rank:
            values = list(pages)
            if (any(type(p) is not int or not 0 <= p < blocks for p in values)
                    or len(set(values)) != len(values)):
                raise ValueError("initialization requires unique in-range physical pages")
            selected.append(sorted(values))
        template, page_bytes = self.hosts[name], self.page_bytes[name]
        for rank, pages in enumerate(selected):
            if not pages:
                continue
            runs = []
            first = previous = pages[0]
            for page in pages[1:]:
                if page != previous + 1:
                    runs.append((first, previous + 1))
                    first = page
                previous = page
            runs.append((first, previous + 1))
            shard = root.shards[rank]
            for first, end in runs:
                for start in range(first, end, template.shape[0]):
                    count = min(template.shape[0], end - start)
                    nbytes = count * page_bytes
                    for layer in range(layers):
                        offset = (layer * blocks + start) * page_bytes
                        worker.copy_to(
                            shard.data_ptr - shard.byte_offset, template.data_ptr(), nbytes,
                            dst_offset=shard.byte_offset + offset, worker_id=rank,
                        )
