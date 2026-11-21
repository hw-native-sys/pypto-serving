# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate typed root uploads, ragged views and table revision invalidation."""

import ctypes
from math import prod
from types import SimpleNamespace

import pytest
import torch
from pypto.runtime import DeviceTensor

from pypto_serving.model.deepseek_lowlat.bindings import TensorArena
from pypto_serving.model.deepseek_lowlat.kernel_loader import isolated_profile
from pypto_serving.model.deepseek_lowlat.prefill_transfers import PrefillTransferSlots


def slots():
    rows = [("ori_block_table", [8, 1, 16], "int32"), ("position_ids_full", [8, -1], "int32"),
            ("query_start_loc", [8, 2], "int32"),
            ("input_ids", [8, -1], "int64"), ("swa_freqs_cos", [8, -1, 64], "bf16")]
    programs = SimpleNamespace(context=isolated_profile, prefill_metadata={"params": [
        dict(name=n,shape=s,dtype=d,direction="In",layout="ND") for n,s,d in rows]})
    return PrefillTransferSlots(programs, 16)


class Worker:
    def __init__(self, inherited):
        self.inherited=inherited
        self.storage={}
        self.copies=[]
        self.failure=False
        self.fail_after=None

    def alloc_tensor(self, shape, dtype, *, worker_id):
        host=torch.empty(shape,dtype=dtype)
        self.storage[host.data_ptr()]=host
        return DeviceTensor(host.data_ptr(),shape,dtype)

    def view_tensor(self, source, shape, *, byte_offset=0, worker_id):
        assert byte_offset+prod(shape)*source.dtype.itemsize <= source.nbytes
        return DeviceTensor(source.data_ptr+byte_offset,shape,source.dtype,
                            byte_offset=source.byte_offset+byte_offset)

    def copy_to(self, dst, src, size, *, worker_id):
        assert dst in self.storage
        assert any(root.data_ptr() <= src and src+size <= root.data_ptr()+root.numel()*root.element_size()
                   for root in self.inherited)
        if self.failure or (self.fail_after is not None and len(self.copies) >= self.fail_after):
            raise RuntimeError("copy failed")
        ctypes.memmove(dst,src,size)
        self.copies.append((dst,size,worker_id))

    def free_stacked_tensor(self, owner):
        for shard in owner.shards:
            assert shard.byte_offset == 0
            del self.storage[shard.data_ptr]


def inputs(length):
    return dict(ori_block_table=torch.arange(8*16,dtype=torch.int32).reshape(8,1,16),
                query_start_loc=torch.arange(16,dtype=torch.int32).reshape(8,2),
                position_ids_full=torch.arange(8*length,dtype=torch.int32).reshape(8,length),
                input_ids=torch.arange(8*length,dtype=torch.int64).reshape(8,length)+100,
                swa_freqs_cos=torch.arange(8*length*64,dtype=torch.float32).reshape(8,length,64).bfloat16())


def check_device(views, sources):
    for name, owner in views.items():
        assert tuple(owner.shape)==tuple(sources[name].shape)
        for rank, shard in enumerate(owner.shards):
            expected=sources[name][rank].contiguous().view(torch.uint8).numpy().tobytes()
            assert ctypes.string_at(shard.data_ptr,len(expected))==expected


def test_typed_roots_preserve_rank_values_and_reuse_stable_tables_across_ragged_chunks():
    transfers=slots();worker=Worker(transfers.inherited);arena=TensorArena(worker)
    assert all(root.is_shared() and root.is_contiguous() for root in transfers.inherited)
    transfers.bind(arena)
    assert sum(t.numel()*t.element_size() for t in worker.storage.values())==8*transfers.bytes_per_rank
    first=inputs(16);views=transfers.stage(first,revision=(1,0));check_device(views,first)
    assert len(worker.copies)==32
    assert any(field.shards[0].byte_offset for field in views.values())
    second=inputs(8);views=transfers.stage(second,revision=(1,0));check_device(views,second)
    assert len(worker.copies)==56
    second['ori_block_table'].add_(300)
    views=transfers.stage(second,revision=(1,1));check_device(views,second)
    assert len(worker.copies)==88
    arena.close()
    assert not worker.storage


def test_failed_copy_poison_prevents_reusing_partial_metadata():
    transfers=slots();worker=Worker(transfers.inherited);arena=TensorArena(worker);transfers.bind(arena)
    worker.failure=True
    with pytest.raises(RuntimeError,match='copy failed'): transfers.stage(inputs(8),revision=1)
    worker.failure=False
    with pytest.raises(RuntimeError,match='poisoned'): transfers.stage(inputs(8),revision=2)
    assert not transfers.busy and transfers.table_revision is None
    transfers.recover_after_stage_failure()
    check_device(transfers.stage(inputs(8), revision=2), inputs(8))
    arena.close()


def test_later_copy_failure_poison_prevents_reusing_partial_metadata():
    transfers=slots();worker=Worker(transfers.inherited);arena=TensorArena(worker);transfers.bind(arena)
    worker.fail_after=1
    with pytest.raises(RuntimeError,match='copy failed'): transfers.stage(inputs(8),revision=1)
    assert transfers.poisoned and not transfers.busy and transfers.table_revision is None
    worker.fail_after=None
    with pytest.raises(RuntimeError,match='poisoned'): transfers.stage(inputs(8),revision=2)
    transfers.recover_after_stage_failure()
    check_device(transfers.stage(inputs(8), revision=2), inputs(8))
    arena.close()
