# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for V4.1 weight staging limits and head layout contracts."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import sys
import threading
import time
from types import ModuleType, SimpleNamespace

import pytest
import torch

from pypto_serving.model.common.weights.nz import unpack_nz
from pypto_serving.model.deepseek_v41 import npu_runner
from pypto_serving.model.deepseek_v41.weight_loader import _ByteBudget


def test_routed_projection_budget_bounds_in_flight_bytes():
    budget = _ByteBudget(10)
    active = 0
    peak = 0
    lock = threading.Lock()

    def task():
        nonlocal active, peak
        budget.acquire(6)
        try:
            with lock:
                active += 6
                peak = max(peak, active)
            time.sleep(0.01)
            with lock:
                active -= 6
        finally:
            budget.release(6)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda _: task(), range(2)))
    assert peak == 6

    with pytest.raises(ValueError, match="requires 11 bytes"):
        budget.acquire(11)


@pytest.mark.parametrize("layout", ("nd_nk", "bf16_nz_kn"))
def test_final_head_chunked_layout_and_resident_contract(monkeypatch, layout):
    hidden, vocab = 32, 48
    checkpoint = torch.arange(vocab * hidden).reshape(vocab, hidden).bfloat16()
    requests = []

    class Loader:
        def __init__(self, _model_dir, **_kwargs):
            self.config = SimpleNamespace(hidden_size=hidden, vocab_size=vocab)

        def load(self, name):
            assert name == "norm.weight"
            return SimpleNamespace(weight=torch.ones(hidden, dtype=torch.bfloat16))

        def load_rows(self, name, start, stop):
            assert name == "head.weight"
            requests.append((start, stop))
            return SimpleNamespace(weight=checkpoint[start:stop].clone())

    monkeypatch.setattr(npu_runner, "V41WeightLoader", Loader)
    topology = npu_runner.SegmentTopology.for_decode(dp=4)
    host = npu_runner.load_final_output_host_resources("unused", topology, chunk_rows=23, head_layout=layout)
    expected_shape = (4, hidden, vocab) if layout == "bf16_nz_kn" else (4, vocab, hidden)
    assert tuple(host.head_weight.shape) == expected_shape
    assert host.hidden_size == hidden and host.head_layout == layout
    assert all(t.is_shared() for t in host.all_host_tensors)
    for rank in range(topology.world):
        restored = unpack_nz(host.head_weight[rank]) if layout == "bf16_nz_kn" else host.head_weight[rank].t()
        assert torch.equal(restored, checkpoint.t())
        assert torch.equal(host.head_weight[rank], host.head_weight[0])
    assert max(stop - start for start, stop in requests) <= 23
    assert requests == ([(0, 16), (16, 32), (32, 48)] if layout == "bf16_nz_kn"
                        else [(0, 23), (23, 46), (46, 48)])

    class Worker:
        def __init__(self):
            self.values = []

        def alloc_stacked_tensor(self, tensor):
            self.values.append(tensor)
            return SimpleNamespace(shape=tuple(tensor.shape))

        def free_stacked_tensor(self, _tensor):
            pass

    worker = Worker()
    final = npu_runner.FinalOutputWeights(worker, topology, host)
    assert worker.values[1] is host.head_weight
    shapes = []
    def allocate(_worker, shape, _dtype, *, stacked):
        assert stacked
        shapes.append(tuple(shape))
        return SimpleNamespace(shape=tuple(shape))
    monkeypatch.setattr(npu_runner, "alloc_device_buffer", allocate)
    names = ("final_norm_weight", "lm_head_weight", "x_hc_out", "pre_mix_out", "logits", "sampled_ids")
    empty_weights = SimpleNamespace(layout_names=())
    empty_cache = SimpleNamespace(prefill_full_views=lambda: {})
    empty_metadata = SimpleNamespace(full_arguments=lambda: {})
    segment = npu_runner.PrefillForwardSegment(worker, None, names, None, topology,
                                             empty_weights, empty_metadata, empty_cache, final)
    assert shapes[0] == (4, topology.local_capacity, 4, hidden)
    segment.close()
    final.close()
    opposite = "nd_nk" if layout == "bf16_nz_kn" else "bf16_nz_kn"
    with pytest.raises(ValueError, match="disagree"):
        npu_runner.FinalOutputWeights(worker, topology, replace(host, head_layout=opposite))
    for field, dtype in (("norm_weight", torch.bfloat16), ("head_weight", torch.bfloat16),
                         ("logits", torch.float32), ("logit_row_indices", torch.int32),
                         ("sampled_ids", torch.int32)):
        malformed = torch.zeros((), dtype=dtype).share_memory_()
        with pytest.raises(ValueError, match="tensor ranks"):
            npu_runner.FinalOutputWeights(worker, topology, replace(host, **{field: malformed}))
    with pytest.raises(ValueError, match="require"):
        npu_runner.load_final_output_host_resources("unused", topology, max_host_bytes=1, head_layout=layout)


@pytest.mark.parametrize("marker,expected", ((None, "nd_nk"), ("bf16_nz_kn", "bf16_nz_kn")))
def test_head_layout_metadata_is_v41_and_revision_specific(monkeypatch, tmp_path, marker, expected):
    module = ModuleType("lm_head")
    module.__file__ = str(tmp_path / "models/deepseek_v4_1_flash/lm_head.py")
    if marker is not None:
        module.LM_HEAD_WEIGHT_LAYOUT = marker
    monkeypatch.setitem(sys.modules, "lm_head", module)
    assert npu_runner._lib_head_weight_layout(tmp_path) == expected
    module.LM_HEAD_WEIGHT_LAYOUT = "unknown"
    with pytest.raises(ValueError, match="unsupported"):
        npu_runner._lib_head_weight_layout(tmp_path)
    module.__file__ = str(tmp_path / "models/another_model/lm_head.py")
    with pytest.raises(RuntimeError, match="different lib checkout or model"):
        npu_runner._lib_head_weight_layout(tmp_path)
