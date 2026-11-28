# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for the full text of the License.
"""Host-side tests for the V4.1 bounded decoder replay plan."""

from types import SimpleNamespace

import pytest

from pypto_serving.model.deepseek_v41.npu_executor import DeepSeekV41PyptoExecutor
from pypto_serving.model.deepseek_v41.npu_runner import SegmentTopology
from pypto_serving.model.deepseek_v41.task_args import (
    RequestSlice,
    plan_decoder_replay_batch,
    plan_decoder_tail,
)
from pypto_serving.serving.server.serving_worker import WorkerProcess


def request(*, partition=0, start=0, tokens=128, prompt=128):
    return RequestSlice(
        "request", partition, 0, start, tuple(range(tokens)), prompt,
        {"window": (0,), "c2a": (0,), "c1a": (0,)},
    )


@pytest.mark.parametrize("tokens", [1, 8, 127, 128, 129])
def test_terminal_tail_selects_last_window_rows(tokens):
    req = request(start=0, tokens=tokens, prompt=tokens)
    plan = plan_decoder_tail(req)
    assert plan.positions == tuple(range(max(0, tokens - 128), tokens))
    assert plan.replay_rows == plan.positions
    assert not plan.has_cross_chunk_prefix


def test_intermediate_chunk_has_no_decoder_rows():
    req = request(start=0, tokens=2048, prompt=4096)
    plan = plan_decoder_tail(req)
    assert plan.positions == ()
    assert plan.replay_rows == ()
    assert not plan.terminal


def test_short_terminal_chunk_marks_cross_chunk_tail():
    req = request(start=2048, tokens=8, prompt=2056)
    plan = plan_decoder_tail(req)
    assert plan.positions == tuple(range(2048, 2056))
    assert plan.replay_rows == tuple(range(0, 8))
    assert plan.cross_chunk_positions == tuple(range(1928, 2048))
    assert plan.has_cross_chunk_prefix


def test_bounded_replay_splits_tail_across_chunks():
    first = plan_decoder_tail(request(start=0, tokens=2048, prompt=2056))
    last = plan_decoder_tail(request(start=2048, tokens=8, prompt=2056))
    assert first.positions == tuple(range(1928, 2048))
    assert last.positions == tuple(range(2048, 2056))
    assert first.replay_start == last.replay_start == 1928


def test_batch_plan_keeps_dp_offsets_and_counts():
    requests = (
        request(partition=0, start=0, tokens=256, prompt=256),
        request(partition=0, start=256, tokens=128, prompt=512),
        request(partition=1, start=2048, tokens=8, prompt=2056),
    )
    plan = plan_decoder_replay_batch(requests, partitions=2)
    assert plan.encoder_counts == (384, 8)
    assert plan.decoder_counts == (128, 8)
    assert plan.replay_rows[0] == tuple(range(128, 256))
    assert plan.replay_rows[1] == tuple(range(8))
    assert plan.has_cross_chunk_prefix


def test_single_terminal_batch_stays_within_one_replay_interval():
    plan = plan_decoder_replay_batch((request(tokens=256, prompt=256),), partitions=1)
    assert not plan.has_cross_chunk_prefix


def test_intermediate_batch_uses_empty_decoder_stage():
    plan = plan_decoder_replay_batch((request(tokens=2048, prompt=4096),), partitions=1)
    assert plan.decoder_counts == (0,)


def test_k0_prefill_packing_respects_resident_row_capacity():
    executor = DeepSeekV41PyptoExecutor()
    runtime = SimpleNamespace(
        num_speculative_tokens=0,
        max_batch_size=8,
        kv_cache_groups=(SimpleNamespace(max_requests_per_partition=2),),
    )
    host = SimpleNamespace(topology=SegmentTopology.for_prefill(dp=4, max_tokens=1024))
    executor.runners["model"] = SimpleNamespace(
        runtime=runtime, resources=SimpleNamespace(prefill_host=host),
    )
    first = SimpleNamespace(cache_partition=0, chunk_tokens=(1,) * 1024)
    second = SimpleNamespace(cache_partition=0, chunk_tokens=(2,) * 1024)
    other_dp = SimpleNamespace(cache_partition=1, chunk_tokens=(3,) * 1024)
    tail = SimpleNamespace(cache_partition=0, chunk_tokens=(4,) * 8)
    other_tail = SimpleNamespace(cache_partition=0, chunk_tokens=(5,) * 127)
    chunks = WorkerProcess._partitioned_prefill_chunks(
        [first, second, other_dp, tail], executor.max_prefill_batch_size,
        max_requests_per_partition=executor.max_prefill_requests_per_partition,
        max_tokens_per_partition=executor.max_prefill_tokens_per_partition,
    )
    assert chunks == [[first, other_dp], [second], [tail]]
    assert WorkerProcess._partitioned_prefill_chunks(
        [tail, other_tail], executor.max_prefill_batch_size,
        max_requests_per_partition=executor.max_prefill_requests_per_partition,
        max_tokens_per_partition=executor.max_prefill_tokens_per_partition,
    ) == [[tail, other_tail]]
