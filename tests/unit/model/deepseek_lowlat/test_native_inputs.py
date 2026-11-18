# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from pypto.runtime import DeviceTensor, StackedDeviceTensor

from pypto_serving.model.deepseek_lowlat.native_inputs import NativeInputs


@pytest.fixture
def inputs(monkeypatch):
    monkeypatch.setattr(NativeInputs, "MAX_ANCHORS", 64)
    metadata = json.loads(Path(__file__).with_name("native_153_parameters.json").read_text())
    return NativeInputs(metadata)


@pytest.mark.parametrize("prompt,output", [(1, 2), (7, 7), (129, 64), (16314, 64)])
def test_compact_banks_keep_absolute_context_and_query_positions(inputs, prompt, output):
    cos = torch.arange(16384, dtype=torch.bfloat16).unsqueeze(1).expand(-1, 64).contiguous()
    sin = -cos
    count = inputs.prepare_banks(prompt_tokens=prompt, max_output_tokens=output,
                                  rope_cos=cos, rope_sin=sin)
    assert count == output-1
    for index in (0, count-1):
        anchor = prompt-1+index
        expected = list(range(max(0, anchor-127), anchor+1))
        for rank in range(4):
            position = inputs.banks["d_context_group_position_ids_bank"][rank, index].reshape(4, 128)
            assert position[0, :len(expected)].tolist() == expected
            assert position[0, len(expected):].eq(-1).all()
            assert position[1:, 0].eq(0).all() and position[1:, 1:].eq(-1).all()
            query = inputs.banks["d_query_group_position_ids_bank"][rank, index].reshape(4, 20)
            assert query[0, :5].tolist() == list(range(anchor+1, anchor+6))
            assert query[1:, :5].tolist() == [list(range(1, 6))]*3
            slots = inputs.banks["d_context_group_slot_mapping_bank"][rank, index].reshape(3, 4, 128)
            for layer in range(3):
                for row, absolute in enumerate(expected):
                    page = int(inputs.tables[0, layer, 0, absolute//32])
                    assert int(slots[layer, 0, row]) == page*32+absolute%32
            actual_cos = inputs.banks["d_context_group_freqs_cos_bank"][rank, index, :len(expected)]
            assert torch.equal(actual_cos, cos[expected])
            actual_sin = inputs.banks["d_query_group_freqs_sin_bank"][rank, index, :5]
            assert torch.equal(actual_sin, sin[anchor+1:anchor+6])


def test_native_bank_uploads_use_prefork_backings_and_request_extent(inputs):
    calls = []
    class Worker:
        def view_tensor(self, owner, shape, worker_id):
            return DeviceTensor(owner.data_ptr, shape, owner.dtype)
        def copy_to(self, destination, source, nbytes, worker_id):
            calls.append((source, nbytes))
    worker = Worker()
    destinations = {}
    for index, (name, host) in enumerate(inputs.banks.items()):
        shards = [DeviceTensor(0x1000000+index*0x100000+r*0x10000, host.shape[1:], host.dtype)
                  for r in range(4)]
        destinations[name] = StackedDeviceTensor(shards, host.shape, tuple(range(4)))
    views = inputs.upload_banks(SimpleNamespace(worker=worker), destinations, 3)
    assert len(calls) == 40
    spans = [(h.data_ptr(), h.data_ptr()+h.numel()*h.element_size()) for h in inputs.banks.values()]
    assert all(any(lo <= p and p+n <= hi for lo, hi in spans) for p, n in calls)
    assert all(value.shape[1] == 3 and value.worker_ids == tuple(range(4)) for value in views.values())
