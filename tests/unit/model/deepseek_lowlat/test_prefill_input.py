# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check actual registered HC storage across ragged prompt chunks."""

import pytest
import torch
from pypto_serving.model.deepseek_lowlat.kernel_loader import isolated_profile


def test_shared_input_preserves_bf16_bits_and_clears_ragged_padding():
    with isolated_profile():
        from models.deepseek_v4_flash_lowlat.prefill_host_inputs import SharedInput

        source = SharedInput(32)
    bits = torch.tensor([0, 1, -32768, 16256, -16512, 32639, -129], dtype=torch.int16)
    row = bits.repeat((4096+len(bits)-1)//len(bits))[:4096].view(torch.bfloat16)
    first = row.repeat(17, 1)
    pointer = source.storage.data_ptr()
    expanded = source.fill(first)
    assert expanded.shape == (8, 24, 4, 4096)
    assert expanded.stride(0) == 0
    assert all(expanded[rank].data_ptr() == pointer for rank in range(8))
    assert source.storage.is_shared()
    for lane in range(4):
        assert torch.equal(expanded[7, :17, lane].view(torch.int32), first.float().view(torch.int32))
    assert torch.count_nonzero(expanded[:, 17:]) == 0
    source.begin_copy()
    with pytest.raises(RuntimeError): source.fill(first)
    source.end_copy()
    short = source.fill(first[:3])
    assert short.shape == (8, 8, 4, 4096)
    assert short[0].data_ptr() == pointer
    assert torch.count_nonzero(short[:, 3:]) == 0
    assert source.generation == 2


def test_real_chunk_adapter_retains_host_local_ids_without_dispatching_them(monkeypatch):
    from pypto.runtime import DeviceTensor, StackedDeviceTensor
    from unittest.mock import Mock
    with isolated_profile():
        from models.deepseek_v4_flash_lowlat import prefill_fwd as adapter

        def device(shape, offset):
            return StackedDeviceTensor([DeviceTensor(offset+rank*0x10000000, shape[1:], torch.float32)
                for rank in range(8)],shape,tuple(range(8)))
        bindings=dict(x_hc=device((8,8,4,4096),0x100000),attn_stage=device((8,8,4,4096),0x200000),
            input_ids_full=torch.zeros((8,8),dtype=torch.int64),position_ids_full=torch.zeros((8,8)),
            position_ids_local=torch.zeros((8,1)),local_request_ids=torch.zeros((8,1),dtype=torch.int32),
            target_hidden=device((8,1,12288),0x300000))
        for name in ('ori','hca_cmp','hca_state','csa_cmp','csa_idx','csa_state','csa_inner_state'):
            bindings[name+'_slot_mapping_full']=torch.zeros((8,8),dtype=torch.int32)
        submitted=Mock(return_value=object());monkeypatch.setattr(adapter,'_submit_chunk',submitted)
        result=adapter.submit_prefill(object(),{'chunk':object()},{'chunk':{}},bindings,
                                      group_tokens=8,terminal=True)
        assert result is submitted.return_value
        assert bindings['local_request_ids'].device.type=='cpu'
        submitted.assert_called_once()
        assert "local_request_ids" not in adapter.load_prefill_programs()["chunk"].param_names
        del bindings['local_request_ids']
        with pytest.raises(KeyError,match='local_request_ids'):
            adapter.submit_prefill(object(),{'chunk':object()},{'chunk':{}},bindings,
                                   group_tokens=8,terminal=True)
        submitted.assert_called_once()
