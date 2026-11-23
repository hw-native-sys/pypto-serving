# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pre-fork shared sources, incremental metadata uploads and bounded leases."""

import torch

from .bindings import DTYPES, ordered_arguments, parameter_rows
from .cache import TABLE_GROUPS


DYNAMIC_FIELDS = ("input_ids", "position_ids", "kv_seq_lens")
TABLE_FIELDS = (*TABLE_GROUPS, "block_counts")
DECODE_FIELDS = (*TABLE_FIELDS, *DYNAMIC_FIELDS, "logit_row_indices")


class DecodeTransferSlots:
    """Keep registered storage and its device contents distinct from request state."""

    def __init__(self, metadata):
        rows = parameter_rows(metadata)
        self.context_capacity = rows["freqs_cos"]["shape"][-2] if "freqs_cos" in rows else 16384
        self.hosts = {}
        for name in DECODE_FIELDS:
            row = rows[name]
            shape = row["shape"]
            if shape is None or shape[0] != 8 or any(d <= 0 for d in shape):
                raise ValueError(f"decode transfer slot needs a fixed TP8 shape: {name}")
            self.hosts[name] = torch.zeros(tuple(shape), dtype=DTYPES[row["dtype"]]).share_memory_()
        self.readback = torch.empty((1,), dtype=torch.int32).share_memory_()
        self.generation = None
        self.busy = False
        self.poisoned = False
        self.uploaded_tables = set()
        self.arguments = None
        self.program = None
        self.worker = self.arena = self.destination = None
        self.last_uploads = ()
        self.names = ()

    @property
    def inherited(self):
        return [*self.hosts.values(), self.readback]

    def bind(self, worker, arena, destination, bindings, metadata, program):
        if self.arguments is not None:
            raise RuntimeError("decode slots are already bound")
        self.worker, self.arena, self.destination = worker, arena, destination
        self.arguments = ordered_arguments(metadata, bindings)
        self.names = tuple(parameter_rows(metadata))
        self.program = program
        # This constant has independent storage from the prefill profile.
        arena.copy(destination["logit_row_indices"], self.hosts["logit_row_indices"])

    def refresh(self, bindings, metadata):
        """Revalidate changed owners/views; unchanged B1 arguments stay stable."""
        if self.busy or self.poisoned or self.arguments is None:
            raise RuntimeError("cannot replace bindings while leased or poisoned")
        values = tuple(bindings[name] for name in self.names)
        if any(current is not previous for current, previous in zip(values, self.arguments, strict=True)):
            arguments = ordered_arguments(metadata, bindings)
            destination = {name: bindings[name] for name in DECODE_FIELDS}
            if destination["logit_row_indices"] is not self.destination["logit_row_indices"]:
                try:
                    self.arena.copy(destination["logit_row_indices"], self.hosts["logit_row_indices"])
                except Exception:
                    self.poisoned = True
                    raise
            self.arguments = arguments
            self.destination = destination
            self.uploaded_tables.clear()

    def acquire(self, generation, program):
        if self.poisoned or self.busy or self.program is not program or self.arguments is None:
            raise RuntimeError("decode slots are poisoned, busy, unbound or belong to another program")
        if type(generation) is not int or generation <= 0:
            raise ValueError("a positive request generation is required")
        if self.generation != generation:
            self.uploaded_tables.clear()
            self.generation = generation
        self.busy = True
        self.last_uploads = ()

    def stage_tables(self, tables, counts):
        """Update shared table owners also consumed through native target aliases."""
        if not self.busy or self.poisoned:
            raise RuntimeError("decode metadata requires a live transfer lease")
        sources = {**tables, "block_counts": counts}
        if set(sources) != set(TABLE_FIELDS):
            raise ValueError("decode requires all grouped block tables and counts")
        uploads = []
        try:
            for name in TABLE_FIELDS:
                source, host = sources[name], self.hosts[name]
                if source.dtype != host.dtype or source.shape != host.shape or source.device.type != "cpu":
                    raise ValueError(f"invalid decode metadata source: {name}")
                if name not in self.uploaded_tables or not torch.equal(host, source):
                    host.copy_(source)
                    self.arena.copy(self.destination[name], host)
                    self.uploaded_tables.add(name)
                    uploads.append(name)
        except Exception:
            self.poisoned = True
            raise
        self.last_uploads = tuple(uploads)

    def stage(self, tables, counts, *, token, end):
        if type(token) is not int or not 0 <= token < 129280 or type(end) is not int or not 1 <= end <= self.context_capacity:
            raise ValueError("invalid decode token or input extent")
        self.stage_tables(tables, counts)
        uploads = list(self.last_uploads)
        try:
            for name, value in (("input_ids", token), ("position_ids", end-1), ("kv_seq_lens", end)):
                self.hosts[name].fill_(value)
                self.arena.copy(self.destination[name], self.hosts[name])
                uploads.append(name)
        except Exception:
            self.poisoned = True
            raise
        self.last_uploads = tuple(uploads)
        return self.arguments

    def complete(self):
        if not self.busy or self.poisoned:
            raise RuntimeError("cannot release an incomplete or poisoned transfer lease")
        self.busy = False

    def poison(self):
        self.poisoned = True

    def invalidate_request(self):
        if self.busy:
            raise RuntimeError("request metadata remains leased")
        self.generation = None
        self.uploaded_tables.clear()

    def recover_after_prefill_failure(self):
        """Clear a prefill-only poison after all transfers are known idle."""
        if self.busy:
            raise RuntimeError("cannot recover decode transfers while a lease is active")
        self.poisoned = False
        self.generation = None
        self.uploaded_tables.clear()
