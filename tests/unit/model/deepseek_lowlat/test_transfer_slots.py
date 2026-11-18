# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check transfer backing, incremental copies and failure ownership on CPU."""

import pytest
import torch

from pypto.runtime import DeviceTensor, StackedDeviceTensor
from pypto_serving.model.deepseek_lowlat.bindings import TensorArena
from pypto_serving.model.deepseek_lowlat.transfers import (
    DECODE_FIELDS, DYNAMIC_FIELDS, TABLE_FIELDS, DecodeTransferSlots,
)


class CopyRecorder:
    def __init__(self):
        self.calls = []
        self.fail_after = None

    def copy_to(self, destination, source, size, *, worker_id):
        if self.fail_after is not None and len(self.calls) == self.fail_after:
            raise RuntimeError("copy failed")
        self.calls.append((destination, source, size, worker_id))


def setup_slots():
    metadata = {"params": []}
    bindings = {}
    for index, name in enumerate(DECODE_FIELDS):
        shape = (8, 1, 7) if name == "block_counts" else (8, 1, 4) if name in TABLE_FIELDS else (8, 1)
        dtype = torch.int64 if name == "input_ids" else torch.int32
        metadata["params"].append(dict(name=name, shape=list(shape), dtype="int64" if dtype == torch.int64
                                        else "int32", direction="In", layout="ND"))
        shards = [DeviceTensor((index + 1)*0x10000+rank*0x1000, shape[1:], dtype) for rank in range(8)]
        bindings[name] = StackedDeviceTensor(shards, shape, tuple(range(8)))
    slots = DecodeTransferSlots(metadata)
    worker = CopyRecorder()
    program = object()
    slots.bind(worker, TensorArena(worker), bindings, bindings, metadata, program)
    worker.calls.clear()
    tables = {name: torch.full_like(slots.hosts[name], -1) for name in TABLE_FIELDS if name != "block_counts"}
    counts = torch.ones_like(slots.hosts["block_counts"])
    for value in tables.values():
        value[:, :, 0] = 0
    return slots, worker, program, tables, counts


def stage(slots, program, tables, counts, generation=1, end=2):
    slots.acquire(generation, program)
    return slots.stage(tables, counts, token=42, end=end)


def test_only_shared_registered_backing_supplies_all_rank_copies():
    slots, worker, program, tables, counts = setup_slots()
    addresses = {name: value.data_ptr() for name, value in slots.hosts.items()}
    args = stage(slots, program, tables, counts)
    assert args is slots.arguments
    assert len(worker.calls) == 88  # Eight tables/count fields and three dynamic fields, eight ranks.
    assert all(value.is_shared() for value in slots.inherited)
    ranges = [(h.data_ptr(), h.data_ptr()+h.numel()*h.element_size()) for h in slots.inherited]
    assert all(any(first <= ptr and ptr+size <= end for first, end in ranges)
               for _, ptr, size, _ in worker.calls)
    assert addresses == {name: value.data_ptr() for name, value in slots.hosts.items()}


def test_steady_step_uploads_three_dynamic_fields_only():
    slots, worker, program, tables, counts = setup_slots()
    stage(slots, program, tables, counts)
    slots.complete()
    worker.calls.clear()
    stage(slots, program, tables, counts, end=3)
    assert slots.last_uploads == DYNAMIC_FIELDS
    assert len(worker.calls) == 24
    assert slots.hosts["position_ids"].flatten().tolist() == [2]*8
    assert slots.hosts["kv_seq_lens"].flatten().tolist() == [3]*8


def test_same_object_content_change_and_count_shrink_are_uploaded():
    slots, worker, program, tables, counts = setup_slots()
    stage(slots, program, tables, counts)
    slots.complete()
    tables["block_table"][:, :, 0] = 3
    tables["block_table"][:, :, 1:] = -1
    counts[:, :, 0] = 0
    worker.calls.clear()
    stage(slots, program, tables, counts)
    assert set(slots.last_uploads) == {*DYNAMIC_FIELDS, "block_table", "block_counts"}
    assert len(worker.calls) == 40
    assert torch.equal(slots.hosts["block_table"], tables["block_table"])


def test_new_generation_invalidates_identical_table_contents():
    slots, worker, program, tables, counts = setup_slots()
    stage(slots, program, tables, counts)
    slots.complete()
    worker.calls.clear()
    stage(slots, program, tables, counts, generation=2)
    assert len(worker.calls) == 88
    assert slots.generation == 2


def test_busy_slot_and_program_change_cannot_be_reused():
    slots, worker, program, tables, counts = setup_slots()
    stage(slots, program, tables, counts)
    with pytest.raises(RuntimeError, match="busy"):
        slots.acquire(1, program)
    with pytest.raises(RuntimeError, match="leased"):
        slots.invalidate_request()
    slots.complete()
    with pytest.raises(RuntimeError, match="another program"):
        slots.acquire(1, object())


def test_partial_copy_failure_never_acknowledges_field_or_releases_lease():
    slots, worker, program, tables, counts = setup_slots()
    worker.fail_after = 3
    with pytest.raises(RuntimeError, match="copy failed"):
        stage(slots, program, tables, counts)
    assert slots.poisoned and slots.busy
    assert not slots.uploaded_tables
    with pytest.raises(RuntimeError, match="poisoned"):
        slots.complete()
    with pytest.raises(RuntimeError, match="poisoned"):
        slots.acquire(2, program)


def test_changed_owner_is_revalidated_and_receives_subsequent_uploads():
    slots, worker, program, tables, counts = setup_slots()
    previous = slots.arguments
    bindings = dict(zip(slots.names, previous, strict=True))
    metadata = {"params": [dict(name=name, shape=list(value.shape), direction="In", layout="ND",
                               dtype="int64" if value.dtype == torch.int64 else "int32")
                           for name, value in bindings.items()]}
    replacement = StackedDeviceTensor(
        [DeviceTensor(0xA00000+rank*0x1000, (1,), torch.int64) for rank in range(8)],
        (8, 1), tuple(range(8)),
    )
    bindings["input_ids"] = replacement
    slots.refresh(bindings, metadata)
    assert slots.arguments is not previous
    assert slots.destination["input_ids"] is replacement
    stage(slots, program, tables, counts)
    assert all(any(destination == shard.data_ptr for destination, _, _, _ in worker.calls)
               for shard in replacement.shards)
