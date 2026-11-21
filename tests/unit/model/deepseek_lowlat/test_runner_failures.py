# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise completion and release ordering through the actual runner methods."""

from types import SimpleNamespace
from dataclasses import dataclass
from unittest.mock import Mock

import pytest
import torch

from pypto_serving.config.types import DecodeBatch
from pypto_serving.model.deepseek_lowlat import npu_runner
from pypto_serving.model.deepseek_lowlat.npu_runner import LowlatModelRunner
from pypto_serving.model.deepseek_lowlat.state import LowlatRequestState
from pypto_serving.model.deepseek_lowlat.kernel_loader import isolated_profile


def runner():
    instance = LowlatModelRunner(
        SimpleNamespace(single=SimpleNamespace(compiled=object()), single_metadata={}), None
    )
    instance.state = LowlatRequestState("old", 1, 8, 8, 8, 42, "decode")
    instance.generation = 1
    instance.cache = SimpleNamespace(tables_for=Mock(return_value=({}, object())), release=Mock(), begin_generation=Mock())
    instance.weights = SimpleNamespace(owners={})
    instance.cache.owners = {}
    instance.decode_bindings = {}
    instance._stage = Mock(return_value={"sampled_ids": object()})
    instance.worker = Mock()
    instance.transfers = Mock(poisoned=False)
    instance.transfers.stage.return_value = ()
    instance.decode_bindings["sampled_ids"] = object()
    return instance


@pytest.mark.parametrize("fail_at", ["submit", "completion", "capture"])
def test_decode_failure_does_not_commit_cursor(monkeypatch, fail_at):
    instance = runner()
    handle = Mock()
    instance.worker.submit.return_value = handle
    instance._read_sample = Mock(return_value=43)
    target = {
        "submit": instance.worker.submit,
        "completion": handle.result,
        "capture": instance._read_sample,
    }[fail_at]
    target.side_effect = RuntimeError(fail_at)
    batch = DecodeBatch(["old"], torch.tensor([42]), None, torch.tensor([9]), block_ids_by_group=[{}])
    with pytest.raises(RuntimeError, match=fail_at):
        instance.run_decode(None, batch)
    assert instance.state.kv_committed_tokens == 8
    assert instance.state.last_sampled_token == 42
    instance.transfers.poison.assert_called_once()
    instance.cache.release.assert_not_called()
    if fail_at == "submit":
        assert not instance.state.pending_handles
    else:
        assert instance.state.pending_handles == [handle]


def test_release_drains_handles_before_pages_are_reusable():
    instance = runner()
    operations = []
    handle = SimpleNamespace(result=lambda: operations.append("completed"))
    instance.state.pending_handles.append(handle)
    instance.cache.release.side_effect = lambda: operations.append("released")
    instance.release_finished_requests(["unrelated"])
    assert operations == []
    instance.release_finished_requests(["old"])
    assert operations == ["completed", "released"]
    assert instance.state is None
    instance.release_finished_requests(["old"])
    assert operations == ["completed", "released"]
    next_state = instance._request(["next"], 17)
    assert next_state.generation == 2
    assert next_state.kv_committed_tokens == next_state.prefilled_tokens == 0


def test_failed_drain_keeps_pages_owned():
    instance = runner()
    handle = Mock()
    handle.result.side_effect = RuntimeError("completion failed")
    instance.state.pending_handles.append(handle)
    with pytest.raises(RuntimeError, match="completion failed"):
        instance.release_finished_requests(["old"])
    assert instance.state is not None
    assert instance.state.phase == "draining"
    instance.cache.release.assert_not_called()
    with pytest.raises(ValueError, match="interleave"):
        instance._request(["next"], 17)


def test_startup_reserves_cache_before_checkpoint_upload(monkeypatch):
    events = []

    @dataclass
    class RunConfig:
        codegen_only: bool = True
        ring_heap: object = None
        ring_task_window: object = None
        ring_dep_pool: object = None

    class Cache:
        counts = {"ori": 4}
        owners = {}

        def __init__(self, *args):
            pass

        def bytes_per_rank(self):
            return 200

        def prepare_host_templates(self, chunk_capacity):
            return []

        def allocate(self):
            events.append("cache")

    class Weights:
        def __init__(self, *args):
            self.owners = {}

        def planned_bytes_per_rank(self):
            return 300

        def load(self):
            assert "cache" in events
            events.append("weights")

        def prefill_bindings(self):
            return {}

    monkeypatch.setattr(npu_runner, "TensorArena", lambda worker: object())
    monkeypatch.setattr(npu_runner, "LowlatCachePool", Cache)
    monkeypatch.setattr(npu_runner, "LowlatWeightStore", Weights)
    monkeypatch.setattr(
        npu_runner, "DecodeTransferSlots", lambda metadata: SimpleNamespace(inherited=[], bind=lambda *a: None)
    )
    monkeypatch.setattr(npu_runner, "PrefillTransferSlots", lambda *a: SimpleNamespace(
        inherited=[], names=frozenset(), bytes_per_rank=0))
    monkeypatch.setattr(npu_runner, "parameter_rows", lambda metadata: {})
    programs = SimpleNamespace(
        run_config=RunConfig(),
        model=None,
        single=SimpleNamespace(compiled=SimpleNamespace(prepare=lambda *a, **kw: object())),
        prefill=SimpleNamespace(compiled=object()),
        single_metadata={},
        prefill_metadata={},
        context=isolated_profile,
    )
    instance = LowlatModelRunner(programs, None)
    instance._check_budget = lambda runtime, additional: events.append(("budget", additional))
    instance._allocate_frames = lambda runtime: events.append("frames")
    runtime = SimpleNamespace(
        total_kv_pages=None,
        ring_heap=None,
        ring_task_window=None,
        ring_dep_pool=None,
        kv_cache_groups=(),
        max_prefill_tokens_per_request=128,
        max_num_batched_tokens=128,
    )
    assert instance.init_kv_cache("model", None, runtime) == 4
    assert events.index("cache") < events.index("weights") < events.index("frames")
    assert events[events.index("cache") + 1][0] == "budget"
    assert events[-1] == ("budget", 0)


def test_request_initialization_policy_is_opt_in_and_failure_poisoned_before_publication():
    programs = SimpleNamespace()
    instance = LowlatModelRunner(programs, None, cache_initialization="request")
    instance.transfers = Mock(poisoned=False)
    instance.cache = Mock()
    instance.cache.initialize_private_pool.side_effect = RuntimeError("prime failed")
    with pytest.raises(RuntimeError, match="prime failed"):
        instance._request(["request"], 1)
    assert instance.state is None and instance.generation == 0
    instance.transfers.poison.assert_called_once()
    default = LowlatModelRunner(programs, None)
    default.cache = Mock()
    default._request(["default"], 1)
    default.cache.initialize_private_pool.assert_not_called()
