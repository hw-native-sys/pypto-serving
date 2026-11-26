# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host-only coverage of routed weight residency across worker creation."""

import ctypes
import mmap
import tempfile
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch.multiprocessing.reductions import StorageWeakRef

from pypto_serving.model.deepseek.weight_loader import _map_shared_prepacked_tensors
from pypto_serving.model.deepseek_v41 import routed_weight_staging as staging
from pypto_serving.model.deepseek_v41.offload_runner import OffloadDecodeWeights, V41OffloadResources
from pypto_serving.model.common import moe_offload
from pypto_serving.model.common.moe_offload import ROUTED_FIELDS
from pypto_serving.model.deepseek_v41.weight_loader import DecodeWeightBank, DecodeWeightLayout


def test_full_page_ranges_keep_edges_shared_with_retained_weights():
    # Retained weights occupy bytes [0, 8) and [232, 300). Adjacent routed
    # fields jointly cover the page containing their boundary at byte 135.
    assert list(staging._full_page_ranges([(135, 232), (300, 320), (8, 135)], 64)) == [(64, 128)]


def test_sidecar_discards_only_full_routed_pages_before_and_after_copy(tmp_path, monkeypatch):
    page = mmap.PAGESIZE
    payload = {name: torch.full((2 * page,), i + 1, dtype=torch.uint8) for i, name in enumerate(ROUTED_FIELDS)}
    payload["attention"] = torch.full((page,), 87, dtype=torch.uint8)
    payload["shared_expert"] = torch.full((page,), 89, dtype=torch.uint8)
    path = tmp_path / "sidecar.safetensors"
    save_file(payload, path)
    bank = DecodeWeightBank(DecodeWeightLayout(1, 1, 1, 1, world=1))
    bank.prepacked = True
    with path.open("rb") as sidecar:
        bank.tensors = _map_shared_prepacked_tensors(sidecar.fileno(), payload)

    native_madvise = ctypes.CDLL(None, use_errno=True).madvise
    native_madvise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
    advice = []

    def record_madvise(address, length, operation):
        advice.append((address, length, operation))
        return native_madvise(address, length, operation)

    monkeypatch.setattr(staging, "ctypes", SimpleNamespace(
        CDLL=lambda *args, **kwargs: SimpleNamespace(madvise=record_madvise),
        c_void_p=ctypes.c_void_p, c_size_t=ctypes.c_size_t, c_int=ctypes.c_int,
    ))
    start = bank.tensors["attention"].data_ptr() + page
    end = bank.tensors["shared_expert"].data_ptr()
    assert start % page == end % page != 0
    first_page = start + page - start % page
    last_page = end - end % page
    expected_advice = (first_page, last_page - first_page, mmap.MADV_DONTNEED)
    # Fault in the source and retained pages before exercising DONTNEED.
    assert all(torch.equal(bank.tensors[name], data) for name, data in payload.items())
    with staging.stage_routed_weights(bank) as restore:
        assert advice == [expected_advice]
        restore()
        assert all(torch.equal(bank.tensors[name], data) for name, data in payload.items())
    assert advice == [expected_advice, expected_advice]
    assert set(bank.tensors) == {"attention", "shared_expert"}
    assert all(torch.equal(tensor, payload[name]) for name, tensor in bank.tensors.items())


@pytest.fixture
def eager_bank(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    bank = DecodeWeightBank(DecodeWeightLayout(1, 1, 1, 1, world=1))
    for index, name in enumerate(ROUTED_FIELDS):
        data = (torch.arange(32, dtype=torch.uint8) + index).reshape(1, 1, 2, 16)
        if name.endswith("_scale"):
            data = data.view(torch.float8_e8m0fnu)
        else:
            data = data.view(1, 1, 1, 32)
        bank.tensors[name] = data.share_memory_()
    bank.tensors["shared_expert"] = torch.ones(4).share_memory_()
    return bank


def test_eager_routed_storage_is_released_before_restore(eager_bank, tmp_path, monkeypatch):
    def unexpected_advice(_ranges):
        pytest.fail("eager shared-memory weights must use spill")

    monkeypatch.setattr(staging, "_discard_sidecar_pages", unexpected_advice)
    originals = {name: StorageWeakRef(eager_bank.tensors[name].untyped_storage()) for name in ROUTED_FIELDS}
    expected = {name: eager_bank.tensors[name].view(torch.uint8).clone() for name in ROUTED_FIELDS}
    dtypes = {name: eager_bank.tensors[name].dtype for name in ROUTED_FIELDS}
    retained = eager_bank.tensors["shared_expert"]
    with staging.stage_routed_weights(eager_bank) as restore:
        assert set(eager_bank.tensors) == {"shared_expert"}
        assert all(storage.expired() for storage in originals.values())
        assert len(list(tmp_path.glob("pypto-v41-routed-*/*"))) == len(ROUTED_FIELDS)
        restore()
        assert all(eager_bank.tensors[name].dtype == dtypes[name] for name in ROUTED_FIELDS)
        assert all(torch.equal(eager_bank.tensors[name].view(torch.uint8), expected[name]) for name in ROUTED_FIELDS)
        restored = {name: StorageWeakRef(eager_bank.tensors[name].untyped_storage()) for name in ROUTED_FIELDS}
    assert all(storage.expired() for storage in restored.values())
    assert set(eager_bank.tensors) == {"shared_expert"}
    assert eager_bank.tensors["shared_expert"] is retained
    assert not list(tmp_path.iterdir())


def test_spill_with_a_live_storage_alias_cleans_up_before_worker_creation(eager_bank, tmp_path):
    alias = eager_bank.tensors["routed_w1"].view(-1)
    with pytest.raises(RuntimeError, match="still own storage before worker creation: routed_w1"):
        with staging.stage_routed_weights(eager_bank):
            pytest.fail("worker creation must wait for the original storage to be released")
    assert alias.tolist() == list(range(32))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("fail_in_factory", [False, True])
def test_resident_spill_cleanup_when_worker_setup_fails(eager_bank, tmp_path, monkeypatch, fail_in_factory):
    monkeypatch.setattr(eager_bank, "finish", lambda _names: eager_bank.tensors)
    retained = eager_bank.tensors["shared_expert"]
    host = SimpleNamespace(inherited_host_tensors=(), topology=None, all_host_tensors=())
    closed = []

    def mapped_expert_backing(_worker, backing):
        assert set(ROUTED_FIELDS) <= eager_bank.tensors.keys()
        window = bytearray(backing.expert_bytes * backing.layers * backing.local_experts)
        backing.initialize(0, memoryview(window))
        assert window[:32] == bytes(range(32))
        raise RuntimeError("mapped copy failed")

    monkeypatch.setattr(moe_offload, "_mapped_expert_backing", mapped_expert_backing)

    def worker_factory(_programs, _run_config, inherited):
        assert set(eager_bank.tensors) == {"shared_expert"}
        assert inherited[0] is retained
        assert len(inherited) == 2
        if fail_in_factory:
            raise RuntimeError("worker creation failed")
        return SimpleNamespace(close=lambda: closed.append(True))

    with pytest.raises(RuntimeError, match="worker creation failed" if fail_in_factory else "mapped copy failed"):
        V41OffloadResources(
            (), None, weight_bank=eager_bank, weight_param_names=(), cache_plan=None,
            state_reset=torch.empty(1).share_memory_(), page_resets={}, input_host=host,
            final_host=host, prefill_host=host, topology=None, worker_factory=worker_factory,
            expert_slots=1, expert_cache_bytes=6 * 32, decode_head_program=None, decode_head_param_names=(),
        )
    assert set(eager_bank.tensors) == {"shared_expert"}
    assert not list(tmp_path.iterdir())
    assert closed == ([] if fail_in_factory else [True])


def test_offload_weights_borrow_expert_bindings_without_owning_them():
    borrowed = {"routed_w1": object(), "expert_state": object()}
    bank = SimpleNamespace(tensors={}, layout=object(), finish=lambda names: None)
    weights = OffloadDecodeWeights(None, bank, (), borrowed=borrowed)
    assert weights.layout_names == tuple(borrowed)
    for name, tensor in borrowed.items():
        assert weights.full(name) is tensor
    weights.close()
    assert weights.full("routed_w1") is borrowed["routed_w1"]
