# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import ctypes

import pytest
import torch
from pypto.runtime import DeviceTensor, StackedDeviceTensor

from pypto_serving.model.deepseek_lowlat.history import NativeHistorySeed


@pytest.mark.parametrize("prompt,chunk", [(1, 1), (7, 7), (129, 128), (513, 512)])
def test_cp_tap_seed_preserves_absolute_rows_across_chunks_and_padding(prompt, chunk):
    seed = NativeHistorySeed()
    seed.begin(1)
    copies = []
    for start in range(0, prompt, chunk):
        count = min(chunk, prompt-start)
        padded = (count+7)//8*8
        positions = torch.arange(start, start+padded, dtype=torch.int64)
        positions[count:] = -1
        positions = positions.reshape(8, -1)
        backing = []
        shards = []
        bases = {}
        for rank, pos in enumerate(positions):
            # A borrowed source retains a nonzero offset from its allocation base.
            host = torch.zeros((16+pos.numel()*12288,), dtype=torch.bfloat16)
            values = pos.to(torch.bfloat16).unsqueeze(1).expand(-1, 12288).contiguous()
            host[16:].copy_(values.reshape(-1))
            backing.append(host)
            bases[rank] = host.data_ptr()
            shards.append(DeviceTensor(host.data_ptr()+32, tuple(values.shape), torch.bfloat16,
                                       byte_offset=32))
        owner = StackedDeviceTensor(shards, (8, positions.shape[1], 12288), tuple(range(8)))
        class Worker:
            def copy_from(self, destination, source, nbytes, *, src_offset, worker_id):
                assert source == bases[worker_id]
                assert seed.host.data_ptr() <= destination
                assert destination+nbytes <= seed.host.data_ptr()+seed.host.numel()*2
                copies.append((worker_id, src_offset, nbytes))
                ctypes.memmove(destination, source+src_offset, nbytes)
        seed.capture(Worker(), owner, positions, prompt_length=prompt, generation=1)
    result = seed.ready(prompt_length=prompt, generation=1)
    for position in range(max(0, prompt-128), prompt):
        expected = torch.full((12288,), position, dtype=torch.bfloat16)
        assert torch.equal(result[position%128], expected)
    assert not result[128:].any()
    assert copies and max(n for _, _, n in copies) <= 128*12288*2
    seed.release()
    seed.begin(2)
    with pytest.raises(RuntimeError, match="cover"):
        seed.ready(prompt_length=prompt, generation=2)
    with pytest.raises(RuntimeError, match="generation"):
        seed.capture(None, owner, positions, prompt_length=prompt, generation=1)


def test_seed_rejects_missing_real_history_before_upload():
    seed = NativeHistorySeed()
    seed.begin(1)
    with pytest.raises(RuntimeError, match="cover"):
        seed.ready(prompt_length=129, generation=1)
