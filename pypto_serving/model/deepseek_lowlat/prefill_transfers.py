# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pack prefill metadata into registered typed roots owned by one Worker."""

import importlib

from pypto.runtime import StackedDeviceTensor

from .bindings import DTYPES, borrow_view, frame_shape, parameter_rows


class PrefillTransferSlots:
    """Reuse lib layouts; upload stable tables only when their source revision changes."""

    def __init__(self, programs, chunk_capacity):
        with programs.context():
            arena = importlib.import_module(
                "models.deepseek_v4_flash_lowlat.prefill_input_plan"
            ).TypedMetadataArena
        fields = {"query_start_loc", "state_checkpoint_ends", "position_ids_full",
                  "position_ids_local", "input_ids", "logit_row_indices"}
        tables, dynamic = [], []
        for name, row in parameter_rows(programs.prefill_metadata).items():
            if name not in fields and not name.endswith(
                    ("_block_table", "_slot_mapping_full", "_freqs_cos", "_freqs_sin")):
                continue
            if row["direction"] != "In" or row["shape"] is None:
                raise ValueError("Prefill upload fields require explicit input metadata")
            shape = frame_shape(name, row["shape"], chunk_capacity)
            spec = dict(name=name, per_rank_shape=list(shape[1:]),
                        dtype=str(DTYPES[row["dtype"]]).removeprefix("torch."))
            (tables if name.endswith("_block_table") else dynamic).append(spec)
        self.layouts = (arena(tables), arena(dynamic))
        self.names = frozenset(name for layout in self.layouts for name in layout.fields)
        self.roots, self.fields = {}, {}
        self.arena = None
        self.table_revision = None
        self.poisoned = False
        self.busy = False

    @property
    def inherited(self):
        return [root for layout in self.layouts for root in layout.roots.values()]

    @property
    def bytes_per_rank(self):
        return sum(layout.bytes_per_rank for layout in self.layouts)

    def bind(self, arena):
        if self.arena is not None:
            raise RuntimeError("Prefill metadata is already bound")
        self.arena = arena
        for index, layout in enumerate(self.layouts):
            for dtype, n in layout.elements.items():
                self.roots[index, dtype] = arena.allocate((8, n), dtype)
            for name, field in layout.fields.items():
                root = self.roots[index, field.dtype]
                shards = [arena.worker.view_tensor(shard, field.shape,
                    byte_offset=field.byte_offset, worker_id=rank) for rank, shard in enumerate(root.shards)]
                self.fields[name] = StackedDeviceTensor(shards, (8, *field.shape), root.worker_ids)
        return self.fields

    def stage(self, inputs, *, revision):
        if self.arena is None or self.busy or self.poisoned or revision is None:
            raise RuntimeError("Prefill metadata is unbound, busy, poisoned or unversioned")
        shapes = {}
        for name, owner in self.fields.items():
            host = inputs[name]
            if (host.device.type != "cpu" or host.dtype != owner.dtype
                    or len(host.shape) != len(owner.shape) or host.shape[0] != 8
                    or any(actual <= 0 or actual > maximum
                           for actual, maximum in zip(host.shape, owner.shape, strict=True))):
                raise ValueError("Prefill metadata exceeds its registered field: " + name)
            shapes[name] = tuple(host.shape)
        self.busy = True
        try:
            for index, layout in enumerate(self.layouts):
                if index == 0 and revision == self.table_revision:
                    continue
                for name, target in layout.views.items():
                    target[tuple(slice(0, size) for size in shapes[name])].copy_(inputs[name])
                for dtype, host in layout.roots.items():
                    self.arena.copy(self.roots[index, dtype], host)
            self.table_revision = revision
            return {name: owner if shapes[name] == tuple(owner.shape)
                    else borrow_view(self.arena.worker, owner, shapes[name])
                    for name, owner in self.fields.items()}
        except BaseException:
            self.poisoned = True
            raise
        finally:
            self.busy = False
