# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import ctypes
import weakref
from types import SimpleNamespace

import pytest
import torch

from pypto_serving.model.common.moe_offload import ExpertBacking, _mapped_expert_backing


def test_mapped_rows_preserve_rank_layer_expert_and_field_bytes():
    layers, ranks, experts = 3, 2, 4
    weights = torch.arange(layers * ranks * experts * 32).byte().view(layers, ranks, experts, 32)
    scales = torch.arange(layers * ranks * experts * 16).short().view(layers, ranks, experts, 16)
    fields = tuple({"w": weights[:, rank], "scale": scales[:, rank]} for rank in range(ranks))
    backing = ExpertBacking(fields, budget_bytes=191)
    assert backing.capacity == 2

    windows = [bytearray(layers * experts * backing.expert_bytes) for _ in range(ranks)]
    bases = [ctypes.addressof(ctypes.c_char.from_buffer(window)) for window in windows]
    descriptors = backing.descriptors(bases)
    for rank, window in enumerate(windows):
        backing.initialize(rank, memoryview(window))
        for layer in range(layers):
            for field, tensor in enumerate(fields[rank].values()):
                start, stride, size, slot_stride = descriptors[rank, layer, field].tolist()
                for expert in range(experts):
                    assert ctypes.string_at(start + expert * stride, size) == tensor[layer, expert].numpy().tobytes()
                assert slot_stride == tensor[layer, 0].nbytes


def test_budget_must_fit_one_packed_expert():
    fields = ({"w": torch.empty(2, 2, 32, dtype=torch.uint8)},)
    with pytest.raises(ValueError, match="at least 32 bytes"):
        ExpertBacking(fields, 31)
    assert ExpertBacking(fields, 65536).capacity == 4


def test_mapped_backing_retains_windows_without_retaining_checkpoint():
    pytest.importorskip("_task_interface", reason="mapped windows require the native Simpler bindings")
    source = ExpertBacking(({"w": torch.arange(64).byte().reshape(1, 2, 32)},), 64)
    reference = weakref.ref(source)
    windows, released, initializers = [], [], []

    def allocate_domain(**options):
        assert options["members"] == (0,)
        assert options["retain_after_run"]
        window = bytearray(options["window_size"])
        windows.append(window)
        base = ctypes.addressof(ctypes.c_char.from_buffer(window))
        buffer = SimpleNamespace(base=base, shm=SimpleNamespace(buf=memoryview(window)))
        initializers.append(options["initializer"])
        options["initializer"]([SimpleNamespace(buffers={"data": buffer})])
        return SimpleNamespace(release=lambda: released.append(True))

    def run(allocate):
        allocate(SimpleNamespace(allocate_domain=allocate_domain), None, None)

    worker = SimpleNamespace(_w=SimpleNamespace(run=run))
    with _mapped_expert_backing(worker, source) as bases:
        del source
        assert reference() is None
        assert ctypes.string_at(bases[0], 64) == bytes(range(64))
        assert not released
    assert released == [True]
