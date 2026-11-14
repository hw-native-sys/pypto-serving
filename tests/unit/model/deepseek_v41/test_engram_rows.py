# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Byte and ownership regressions for mapped Engram row gathering."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import gc
from threading import Lock

import pytest
import torch
from safetensors.torch import save_file

from pypto_serving.model.common.weights.store import LazySafetensorsStore, default_safe_open
from pypto_serving.model.deepseek_v41.weight_loader import V41WeightLoader


@pytest.fixture
def row_loader(tmp_path):
    def make(tensor):
        save_file({"table": tensor}, str(tmp_path / "table.safetensors"))
        opens = []

        @contextmanager
        def counted_open(path, device):
            opens.append(path)
            with default_safe_open(path, device) as reader:
                yield reader

        loader = object.__new__(V41WeightLoader)
        loader.store = LazySafetensorsStore(
            model_dir=tmp_path, weight_map={"table": "table.safetensors"}, safe_open_fn=counted_open,
        )
        loader._row_tensors = {}
        loader._row_tensors_lock = Lock()
        return loader, opens

    return make


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e8m0fnu, torch.bfloat16, torch.float32])
def test_mapped_rows_keep_bits_order_and_owned_results(row_loader, dtype):
    source = torch.arange(256, dtype=torch.int64).to(torch.uint8).view(dtype).reshape(8, -1)
    loader, opens = row_loader(source)
    indices = torch.tensor([[7, 1, 7], [0, 6, 2]])
    expected = source.view(torch.uint8)[indices]
    first = loader.tensor_rows("table", indices)
    gc.collect()  # The safe_open context has already closed.
    assert first.dtype == dtype
    assert first.shape == (*indices.shape, source.shape[1])
    assert torch.equal(first.view(torch.uint8), expected)
    first.view(torch.uint8).zero_()
    second = loader.tensor_rows("table", indices)
    assert torch.equal(second.view(torch.uint8), expected)
    assert len(opens) == 1
    del loader
    gc.collect()
    assert torch.equal(second.view(torch.uint8), expected)


def test_empty_noncontiguous_and_invalid_indices(row_loader):
    source = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    loader, _ = row_loader(source)
    assert loader.tensor_rows("table", torch.empty((2, 0), dtype=torch.int64)).shape == (2, 0, 4)
    indices = torch.tensor([[7, 0], [1, 6], [2, 5]]).t()
    assert torch.equal(loader.tensor_rows("table", indices), source[indices])
    for indices in ([-1], [8], torch.tensor(0)):
        with pytest.raises(ValueError):
            loader.tensor_rows("table", indices)
    with pytest.raises(KeyError):
        loader.tensor_rows("missing", [0])


def test_concurrent_first_lookup_and_failed_mapping_retry(row_loader):
    source = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    loader, opens = row_loader(source)
    original_open = loader.store._safe_open_fn

    @contextmanager
    def failing_open(path, device):
        raise OSError("transient open failure")
        yield

    loader.store._safe_open_fn = failing_open
    with pytest.raises(OSError, match="transient"):
        loader.tensor_rows("table", [7, 0, 7])
    loader.store._safe_open_fn = original_open
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: loader.tensor_rows("table", [7, 0, 7]), range(8)))
    assert len(opens) == 1
    assert all(torch.equal(result, source[[7, 0, 7]]) for result in results)


def test_reject_nonmatrix_table(row_loader):
    loader, _ = row_loader(torch.zeros(2, 3, 4))
    with pytest.raises(ValueError, match="shape"):
        loader.tensor_rows("table", [0])
    assert not loader._row_tensors
