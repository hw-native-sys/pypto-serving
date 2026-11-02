# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Final metadata is published after capture and retired with its request lease."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import threading

import pytest
import torch

from pypto_serving.model.deepseek_dspark.npu_runner import DSparkStepDescriptor
from pypto_serving.pd.adapter import DSparkExtension


@pytest.fixture
def source(monkeypatch):
    monkeypatch.setattr("pypto_serving.pd.adapter.WorkerRuntime",
        lambda settings, *args: SimpleNamespace(settings=settings))
    states = {}
    def release(requests):
        for request in requests:
            states.pop(request, None)
    runner = SimpleNamespace(_compiled=SimpleNamespace(prefill_capture=object(), bootstrap_metadata=None,
        layout=SimpleNamespace(ranks=16, tp_size=4, sliding_window=128)), speculative=False,
        _shared_l3_worker=lambda: None, _materialize_decode_device_cache=lambda: None,
        _drafter_states=states, release_finished_requests=release)
    return DSparkExtension(settings=SimpleNamespace(capacity=2), runner=runner)


def state(source, request, length, generation=1):
    value = SimpleNamespace(prompt_len=length, lease=3, generation=generation, prefill_tail_positions=None)
    source.draft_states[request] = value
    return value


def inputs(request, start, end, length):
    row = DSparkStepDescriptor(0, 1, 0, start, end - start, length, 0, end - start, 0, 0)
    return SimpleNamespace(request_ids=(request,), descriptors=(row,))


def capture(source, batch, prepared):
    for request, row in zip(prepared.request_ids, prepared.descriptors):
        value = source.draft_states[request]
        positions = torch.arange(row.chunk_start, row.chunk_end, dtype=torch.int64)
        if value.prefill_tail_positions is not None:
            positions = torch.cat((value.prefill_tail_positions, positions))
        value.prefill_tail_positions = positions[-128:]
    return "captured"


@pytest.mark.parametrize("length", [1, 127, 128, 129, 8192, 8205, 8319, 8320])
def test_replayed_and_unaligned_tail_publishes_only_the_confirmed_final_chunk(source, length):
    value = state(source, "request", length)
    wrapped = source.wrap_capture(lambda *args: capture(source, *args))
    replay_start = max(0, (length - 128) // 128 * 128)
    for start in range(replay_start, length, 128):
        with pytest.raises(ValueError, match="not ready"):
            source.source_state("request")
        end = min(start + 128, length)
        assert wrapped(None, inputs("request", start, end, length)) == "captured"
    assert source.source_state("request") == {"bootstrap": value.lease}
    saved = source.source_states["request"]
    assert saved.generation == value.generation
    source.draft_states = None  # The reader must not touch live runner state.
    for _ in range(10):
        result = source.source_state("request")
        result["bootstrap"] = -1
        assert source.source_state("request") == {"bootstrap": saved.lease}
    source.runner.release_finished_requests(iter(("request",)))
    assert not source.source_states and not source.runner._drafter_states


def test_capture_must_finish_before_the_observer_sees_metadata(source):
    state(source, "request", 128)
    entered, completed = threading.Event(), threading.Event()
    def blocked(*args):
        entered.set()
        assert completed.wait(5)
        return capture(source, *args)
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(source.wrap_capture(blocked), None, inputs("request", 0, 128, 128))
        try:
            assert entered.wait(2)
            with pytest.raises(ValueError, match="not ready"):
                source.source_state("request")
        finally:
            completed.set()
        assert future.result(timeout=2) == "captured"
    assert source.source_state("request") == {"bootstrap": 3}


def test_incomplete_tail_never_becomes_ready_and_reused_lease_gets_a_new_generation(source):
    value = state(source, "request", 129)
    source.wrap_capture(lambda *args: None)(None, inputs("request", 128, 129, 129))
    with pytest.raises(ValueError, match="complete prompt tail"):
        source.source_state("request")
    value.prefill_tail_positions = torch.arange(1, 129, dtype=torch.int64)
    source.publish_source_state("request")
    old = source.source_states["request"]
    source.runner.release_finished_requests(("request",))
    value = state(source, "request", 128, generation=old.generation + 1)
    with pytest.raises(ValueError, match="not ready"):
        source.source_state("request")
    source.wrap_capture(lambda *args: capture(source, *args))(None, inputs("request", 0, 128, 128))
    assert source.source_states["request"].generation == value.generation != old.generation
    assert source.source_states["request"].lease == old.lease


def test_packed_requests_capacity_and_retirement_do_not_accumulate_history(source):
    a = state(source, "a", 256)
    state(source, "b", 127)
    prepared = SimpleNamespace(request_ids=("a", "b"), descriptors=(
        inputs("a", 0, 128, 256).descriptors[0], inputs("b", 0, 127, 127).descriptors[0]))
    source.wrap_capture(lambda *args: capture(source, *args))(None, prepared)
    assert set(source.source_states) == {"b"}
    a.prefill_tail_positions = torch.arange(128, 256, dtype=torch.int64)
    source.publish_source_state("a")
    state(source, "extra", 1).prefill_tail_positions = torch.arange(1, dtype=torch.int64)
    source.publish_source_state("extra")
    with pytest.raises(ValueError, match="not ready"):
        source.source_state("extra")
    assert len(source.source_states) == source.transfer.settings.capacity
    source.runner.release_finished_requests(("a", "b", "extra"))
    for generation in range(500):
        value = state(source, "request", 1, generation)
        value.prefill_tail_positions = torch.arange(1, dtype=torch.int64)
        source.publish_source_state("request")
        source.runner.release_finished_requests(("request",))
    assert not source.source_states and not source.draft_states


@pytest.mark.parametrize("order", [("bad", "good"), ("good", "bad")])
def test_bad_final_metadata_does_not_fail_a_healthy_packed_request(source, order):
    state(source, "bad", 128).prefill_tail_positions = torch.arange(1, 128, dtype=torch.int64)
    state(source, "good", 128).prefill_tail_positions = torch.arange(128, dtype=torch.int64)
    prepared = SimpleNamespace(request_ids=order, descriptors=tuple(
        inputs(request, 0, 128, 128).descriptors[0] for request in order))
    # Both model forwards completed; only one request's metadata is invalid.
    assert source.wrap_capture(lambda *a: "computed")(None, prepared) == "computed"
    assert source.source_state("good") == {"bootstrap": 3}
    with pytest.raises(ValueError, match="complete prompt tail"):
        source.source_state("bad")
    source.runner.release_finished_requests(order)
    assert not source.source_states and not source.draft_states


def test_metadata_capacity_and_changed_generation_are_per_request_rejections(source):
    value = state(source, "good", 128)
    value.prefill_tail_positions = torch.arange(128, dtype=torch.int64)
    source.publish_source_state("good")
    source.transfer.settings.capacity = 1
    state(source, "full", 128).prefill_tail_positions = torch.arange(128, dtype=torch.int64)
    source.publish_source_state("full")
    assert source.source_state("good") == {"bootstrap": 3}
    with pytest.raises(ValueError, match="not ready"):
        source.source_state("full")
    value.generation += 1
    source.publish_source_state("good")
    with pytest.raises(ValueError, match="lease changed"):
        source.source_state("good")
    assert len(source.source_states) == 1
    source.runner.release_finished_requests(("good", "full"))
    assert not source.source_states


def test_actual_device_capture_failure_still_fails_the_compute_step(source):
    state(source, "request", 128)
    def fail(*args):
        raise RuntimeError("device capture failed")
    with pytest.raises(RuntimeError, match="device capture failed"):
        source.wrap_capture(fail)(None, inputs("request", 0, 128, 128))
    assert not source.source_states
