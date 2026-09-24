# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded scheduler-to-request metadata tests, with no model/device execution."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from pypto_serving.config.types import KVCacheGroupSpec, KVCacheSpec, RuntimeConfig
from pypto_serving.model.deepseek_v41.composite import MissingCompositeInterface
from pypto_serving.model.deepseek_v41.metadata import prefill_requests
from pypto_serving.model.deepseek_v41.request_state import RequestLedger
from pypto_serving.serving.utils.prefill import pack_prefill_batch


@pytest.fixture
def inputs():
    config = SimpleNamespace(vocab_size=32, max_position_embeddings=16)
    runtime = RuntimeConfig(max_batch_size=2, max_seq_len=16, max_num_batched_tokens=8)
    groups = (
        KVCacheGroupSpec("window", (0,), KVCacheSpec(2, 16), 8, num_blocks=16, num_partitions=2),
        KVCacheGroupSpec("compressed", (2,), KVCacheSpec(4, 16, 2), 4, num_blocks=16, num_partitions=2),
    )
    batch = pack_prefill_batch(
        request_ids=["A", "B"], token_chunks=[[4, 5, 6], [7, 8]], seq_lens=[3, 2],
        chunk_starts=[0, 0], prompt_lens=[5, 2], device="cpu", cache_partitions=[1, 0],
        block_ids_by_group=[{"window": [0, 1], "compressed": [3]}, {"window": [0], "compressed": [3]}],
    )
    return batch, config, runtime, groups


def test_chunk_end_is_not_total_prompt_length(inputs):
    batch, config, runtime, groups = inputs
    ledger = RequestLedger(max_requests=2, max_seq_len=16)
    step = ledger.begin_prefill(prefill_requests(batch, config, runtime, groups))
    assert step.positions == (0, 1, 2, 0, 1)
    assert step.request_rows == (0, 0, 0, 1, 1)
    assert step.terminal_prefill == (False, True)
    assert step.requests[0].partition == 1
    ledger.commit(step)
    next_batch = pack_prefill_batch(
        request_ids=["A"], token_chunks=[[9, 10]], seq_lens=[5], chunk_starts=[3], prompt_lens=[5],
        device="cpu", cache_partitions=[1],
        block_ids_by_group=[{"window": [0, 1, 2], "compressed": [3, 4]}],
    )
    tail = ledger.begin_prefill(prefill_requests(next_batch, config, runtime, groups))
    assert tail.positions == (3, 4)
    assert tail.terminal_prefill == (True,)
    assert tail.requests[0].state_slot == step.requests[0].state_slot


def test_tables_are_copied_and_page_ids_are_partition_local(inputs):
    batch, config, runtime, groups = inputs
    requests = prefill_requests(batch, config, runtime, groups)
    batch.block_ids_by_group[0]["window"][0] = 15
    assert requests[0][-1]["window"] == (0, 1)
    assert requests[1][-1]["window"] == (0,)


@pytest.mark.parametrize("field,value,message", [
    ("request_ids", ["A", "A"], "distinct"),
    ("chunk_offsets", [0, 2], "consecutive"),
    ("chunk_offsets", [1, 4], "consecutive"),
    ("chunk_lens", [0, 2], "consecutive"),
    ("chunk_starts", [True, 0], "must be integers"),
    ("seq_lens", [4, 2], "extent"),
    ("prompt_lens", [], "prompt_lens"),
    ("prompt_lens", [2, 2], "extent"),
    ("prompt_lens", [17, 2], "extent"),
    ("cache_partitions", [None, 0], "explicit DP"),
    ("cache_partitions", [2, 0], "explicit DP"),
    ("cache_partitions", [0, 0], "share writable"),
])
def test_invalid_batch_metadata_is_rejected(inputs, field, value, message):
    batch, config, runtime, groups = inputs
    setattr(batch, field, value)
    with pytest.raises(ValueError, match=message):
        prefill_requests(batch, config, runtime, groups)


@pytest.mark.parametrize("tokens,message", [
    (torch.tensor([4., 5., 6., 7., 8.]), "int32 or int64"),
    (torch.tensor([[4, 5, 6, 7, 8]]), "flat CPU"),
    (torch.tensor([-1, 5, 6, 7, 8]), "vocabulary"),
    (torch.tensor([32, 5, 6, 7, 8]), "vocabulary"),
    (torch.tensor([4, 5, 6, 7, 8, 9]), "unclaimed"),
])
def test_invalid_token_storage_or_range_is_rejected(inputs, tokens, message):
    batch, config, runtime, groups = inputs
    batch.token_ids = tokens
    with pytest.raises(ValueError, match=message):
        prefill_requests(batch, config, runtime, groups)


@pytest.mark.parametrize("pages,message", [
    ({"window": [0, 1]}, "declared cache groups"),
    ({"window": [0, 1], "compressed": [3], "unknown": [0]}, "declared cache groups"),
    ({"window": [0], "compressed": [3]}, "cover the request extent"),
    ({"window": [0, 0], "compressed": [3]}, "alias its own"),
    ({"window": [-1, 1], "compressed": [3]}, "nonnegative integers"),
    ({"window": [True, 1], "compressed": [3]}, "nonnegative integers"),
    ({"window": [16, 1], "compressed": [3]}, "exceeds the configured cache pool"),
])
def test_invalid_page_tables_are_rejected(inputs, pages, message):
    batch, config, runtime, groups = inputs
    batch.block_ids_by_group[0] = pages
    with pytest.raises(ValueError, match=message):
        prefill_requests(batch, config, runtime, groups)


def test_rolling_lowering_fails_without_guessing_the_ring_contract(inputs):
    batch, config, runtime, groups = inputs
    groups = (replace(groups[0], sliding_window=2), groups[1])
    with pytest.raises(MissingCompositeInterface, match="rolling cache"):
        prefill_requests(batch, config, runtime, groups)


def test_missing_group_contract_cannot_use_generic_pages(inputs):
    batch, config, runtime, _ = inputs
    batch.block_ids = [[0, 1], [0]]
    with pytest.raises(MissingCompositeInterface, match="explicit grouped"):
        prefill_requests(batch, config, runtime, ())


def test_dynamic_physical_capacity_is_left_for_allocator_validation(inputs):
    batch, config, runtime, groups = inputs
    groups = (replace(groups[0], num_blocks=None), groups[1])
    batch.block_ids_by_group[0]["window"] = [300, 301]
    result = prefill_requests(batch, config, runtime, groups)
    assert result[0][-1]["window"] == (300, 301)


def test_dispatch_and_per_request_token_limits(inputs):
    batch, config, runtime, groups = inputs
    with pytest.raises(ValueError, match="dispatch capacity"):
        prefill_requests(batch, config, replace(runtime, max_num_batched_tokens=4), groups)
    with pytest.raises(ValueError, match="per-request capacity"):
        prefill_requests(batch, config, replace(runtime, max_prefill_tokens_per_request=2), groups)


def test_decode_worker_column_tokens_preserve_request_order(inputs):
    from pypto_serving.config.types import DecodeBatch
    from pypto_serving.model.deepseek_v41.metadata import decode_requests
    _, config, runtime, groups = inputs
    batch = DecodeBatch(request_ids=["B", "A"], token_ids=torch.tensor([[7], [9]]),
                        hidden_states=None, seq_lens=torch.tensor([3, 6]), cache_partitions=[0, 1],
                        block_ids_by_group=[{"window": [0, 1], "compressed": [3]},
                                            {"window": [0, 1, 2], "compressed": [3, 4]}])
    rows = decode_requests(batch, config, runtime, groups)
    assert [r[:4] for r in rows] == [("B", 0, 2, 7), ("A", 1, 5, 9)]
    batch.token_ids = torch.tensor([[7, 8], [9, 10]])
    with pytest.raises(ValueError, match="one token"):
        decode_requests(batch, config, runtime, groups)
