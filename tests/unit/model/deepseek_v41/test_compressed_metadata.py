# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compressed history and stable compressor slots across chunk boundaries."""
import pytest
import torch

from pypto_serving.model.deepseek_v41.compressed_metadata import prepare_compressed_metadata
from pypto_serving.model.deepseek_v41.request_state import ForwardStep, RequestSlice
from pypto_serving.model.deepseek_v41.swa_segment import SegmentTopology

TOPOLOGY = SegmentTopology(tp=2, dp=2, local_capacity=2)


def req(name="a", group=0, slot=5, start=255, count=3, pages=(7, 1)):
    return RequestSlice(name, group, slot, start, tuple(range(count)), start + count, {"cmp": pages})


def prepare(requests, ratio=2, **kwargs):
    return prepare_compressed_metadata(ForwardStep("prefill", tuple(requests), 1), TOPOLOGY,
        ratio=ratio, compressed_group="cmp", cache_pages=10, max_requests=3, state_blocks=10, **kwargs)


def test_pair_completion_page_crossing_and_interleaved_request_order():
    metadata = prepare([req("b", 1, 8, 0, 1, ()), req(), req("c", 0, 8, 0, 1, ())])
    assert metadata.group_counts == (4, 1)
    assert metadata.position_ids[0].tolist() == [255, 256, 257, 0]
    assert metadata.request_ids[0].tolist() == [0, 0, 0, 1]
    assert metadata.query_start_loc[0].tolist() == [0, 3, 4, 4]
    assert metadata.compressed_lens[0].tolist() == [128, 128, 129, 0]
    assert metadata.compressed_slots[0].tolist() == [1023, -1, 128, -1]
    assert metadata.compressed_rope_positions[0].tolist() == [254, -1, 256, -1]
    assert metadata.state_block_table[0, :, 0].tolist() == [5, 8, -1]
    assert metadata.index_block_table[0].tolist() == [[7, 1], [-1, -1], [-1, -1]]
    assert metadata.state_block_table[2, :, 0].tolist() == [8, -1, -1]
    assert metadata.request_ids[2].tolist() == [0, -1, -1, -1]
    for tensor in (metadata.position_ids, metadata.query_start_loc, metadata.compressed_lens,
                   metadata.compressed_slots, metadata.index_block_table, metadata.state_block_table):
        assert torch.equal(tensor[0], tensor[1]) and torch.equal(tensor[2], tensor[3])


def test_decode_continuation_keeps_state_owner_after_batch_reordering():
    requests = [req("c", 0, 8, 1, 1, (2,)), req("a", 0, 5, 258, 1)]
    step = ForwardStep("decode", tuple(requests), 2)
    metadata = prepare_compressed_metadata(step, TOPOLOGY, ratio=2, compressed_group="cmp",
        cache_pages=10, max_requests=3, state_blocks=10)
    assert metadata.state_block_table[0, :, 0].tolist() == [8, 5, -1]
    assert metadata.compressed_slots[0].tolist() == [256, -1, -1, -1]
    assert metadata.compressed_rope_positions[0].tolist() == [0, -1, -1, -1]
    assert metadata.compressed_lens[0].tolist() == [1, 129, 0, 0]
    assert (metadata.state_block_table[2:] == -1).all()
    assert (metadata.compressed_slots[2:] == -1).all()


def test_ratio_one_publishes_each_token_and_has_no_pending_state():
    metadata = prepare([req(start=127, count=3)], ratio=1)
    assert metadata.compressed_slots[0].tolist() == [1023, 128, 129, -1]
    assert metadata.compressed_lens[0].tolist() == [128, 129, 130, 0]
    assert metadata.compressed_rope_positions[0].tolist() == [127, 128, 129, -1]
    assert (metadata.state_block_table == -1).all()


@pytest.mark.parametrize("requests,match", [
    ([req(pages=(7,))], "completed history"),
    ([req(pages=(7, 7))], "private"),
    ([req(pages=(7, 10))], "outside"),
    ([req(), req("b", slot=5, start=0, count=1, pages=())], "state slots"),
    ([req(slot=10)], "state slot"),
    ([req(start=-1)], "extent"),
])
def test_invalid_compressed_ownership_fails_before_dispatch(requests, match):
    with pytest.raises(ValueError, match=match):
        prepare(requests)


def test_compressed_allocation_budget():
    with pytest.raises(ValueError, match="budget"):
        prepare([req()], max_prepare_bytes=1)
