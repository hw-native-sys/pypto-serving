# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Admission counts real scratch and skips borrowed prefill weight views."""

from types import SimpleNamespace
from contextlib import nullcontext
from unittest.mock import Mock

import pytest
from pypto_serving.model.deepseek_lowlat import npu_runner
from pypto_serving.model.common.executor.utils import build_pypto_run_config


class BudgetObserved(Exception):
    pass


def test_admission_skips_borrowed_weights_but_retains_real_projection_workspace(monkeypatch):
    def row(name, shape, dtype, direction):
        return dict(name=name,shape=shape,dtype=dtype,direction=direction)
    single = {"params": [row("state", [8, 1], "int32", "In")]}
    prefill = {"params": [row("wo_a", [8, 43, 1024, 4096], "bfloat16", "In"),
        row("wo_b", [8, 176128, 1024], "int8", "In"),
        row("o_proj_wo_a_full", [8, 16, 4], "float32", "Out")]}
    config = build_pypto_run_config(platform="a2a3",device_ids=tuple(range(8)))
    program = SimpleNamespace(compiled=SimpleNamespace(prepare=Mock(return_value=object())))
    programs = SimpleNamespace(single=program,prefill=program,single_metadata=single,prefill_metadata=prefill,
                               run_config=config,model=object(),context=nullcontext)
    pool = SimpleNamespace(prepare_host_templates=lambda size:[],bytes_per_rank=lambda:0)
    monkeypatch.setattr(npu_runner,"LowlatCachePool",lambda *args:pool)
    monkeypatch.setattr(npu_runner,"DecodeTransferSlots",lambda *args:SimpleNamespace(inherited=[]))
    monkeypatch.setattr(npu_runner,"LowlatWeightStore",lambda *args:SimpleNamespace(planned_bytes_per_rank=lambda:500))
    instance=npu_runner.LowlatModelRunner(programs,None)
    instance._check_budget=Mock(side_effect=BudgetObserved)
    runtime=SimpleNamespace(total_kv_pages=None,max_prefill_tokens_per_request=64,max_num_batched_tokens=64,
        kv_cache_groups=(),ring_heap=(268435456,268435456,1073741824,1073741824),
        ring_task_window=131072,ring_dep_pool=131072)
    with pytest.raises(BudgetObserved):
        instance.init_kv_cache("m",None,runtime)
    # 500 resident bytes +4 decode state +256 real scratch.
    # The541065216byte view extent is already paid by resident weights.
    instance._check_budget.assert_called_once_with(runtime,760)
