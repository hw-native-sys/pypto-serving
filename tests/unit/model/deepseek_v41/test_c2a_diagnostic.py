# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Ragged diagnostics retain real causal prefixes and explicit inactive padding."""
import pytest
import torch

from pypto_serving.model.deepseek_v41.swa_segment import SegmentTopology
from tools.validate_v41_c2a_chain import select_active_state


def checkpoint_config():
    import json
    from pathlib import Path

    return json.loads((Path(__file__).resolve().parents[4] /
                       "tests/fixtures/deepseek_v41/config.json").read_text())


def test_c1a_chain_includes_full_before_reindex_and_reuse():
    from tools.validate_v41_c2a_chain import select_plans

    plans = select_plans(checkpoint_config(), "c1a", 25)
    assert [p.layer_id for p in plans] == list(range(20, 26))
    assert [p.mode for p in plans] == ["c1a_full", *["c1a_reuse"] * 3,
                                     "c1a_reindex", "c1a_reuse"]
    assert all(p.kv_source == 20 and p.candidate_source == 20 for p in plans)
    assert plans[-1].index_source == 24


def test_default_c2a_diagnostic_preserves_original_two_layers():
    from tools.validate_v41_c2a_chain import select_plans

    plans = select_plans(checkpoint_config(), "c2a")
    assert [(p.layer_id, p.mode) for p in plans] == [(2, "c2a_full"), (3, "c2a_reuse")]


@pytest.mark.parametrize("family,last", [("c1a", 19), ("c1a", 40), ("c2a", 20),
                                         ("c2a", True), ("swa", None)])
def test_diagnostic_cannot_skip_its_full_producer_or_cross_families(family, last):
    from tools.validate_v41_c2a_chain import select_plans

    with pytest.raises(ValueError):
        select_plans(checkpoint_config(), family, last)


def test_c1a_reuse_captures_readonly_producer_state_for_integrity_checks():
    from types import SimpleNamespace
    from tools.validate_v41_c2a_chain import state_names

    module = SimpleNamespace(STATE_NAMES={"reuse": ("window_cache", "window_cache_scale")})
    assert set(state_names(module, "reuse")) == {
        "window_cache", "window_cache_scale", "compressed_cache", "compressed_cache_scale",
        "index_cache", "index_cache_scale", "topk_indices", "candidate_mask"}


def test_select_ragged_prefix_without_mutating_saved_state():
    topology = SegmentTopology(tp=2, dp=2)
    residual = torch.arange(4 * 16 * 4 * 5120, dtype=torch.float32).reshape(4, 16, 4, 5120)
    mix = torch.ones(4, 16, 4)
    saved = dict(actual_residual=residual, actual_pre_mix=mix)
    selected, selected_mix = select_active_state(saved, topology, [31, 0])
    assert torch.equal(selected[0], residual[0])
    assert torch.equal(selected[1, :15], residual[1, :15])
    assert not selected[1, 15:].count_nonzero() and not selected[2:].count_nonzero()
    assert not selected_mix[1, 15:].count_nonzero() and not selected_mix[2:].count_nonzero()
    assert residual[2:].count_nonzero() and mix[2:].eq(1).all()


@pytest.mark.parametrize("counts", [[33, 0], [1], [-1, 32]])
def test_select_prefix_rejects_invalid_counts_before_reading_state(counts):
    with pytest.raises(ValueError, match="active-token count"):
        select_active_state({}, SegmentTopology(tp=2, dp=2), counts)


def test_continuation_repacks_the_next_token_across_tp_slabs():
    topology = SegmentTopology(tp=2, dp=2)
    residual = torch.arange(4 * 16 * 4 * 5120, dtype=torch.float32).reshape(4, 16, 4, 5120)
    mix = torch.arange(4 * 16 * 4, dtype=torch.float32).reshape(4, 16, 4)
    selected, selected_mix = select_active_state(
        dict(actual_residual=residual, actual_pre_mix=mix), topology, [1, 0], [31, 0])
    assert torch.equal(selected[0, 0], residual[1, 15])
    assert torch.equal(selected_mix[0, 0], mix[1, 15])
    assert not selected[0, 1:].count_nonzero() and not selected[1:].count_nonzero()
    assert not selected_mix[0, 1:].count_nonzero() and not selected_mix[1:].count_nonzero()


@pytest.mark.parametrize("starts", [[32, 0], [-1, 0], [0], [True, 0]])
def test_continuation_cannot_exceed_saved_source(starts):
    with pytest.raises(ValueError, match="saved causal source"):
        select_active_state({}, SegmentTopology(tp=2, dp=2), [1, 0], starts)


@pytest.mark.parametrize("family,repeats,ratio", [("c1a", 5, 1), ("c2a", 9, 2)])
def test_repeated_diagnostic_crosses_pages_without_slicing_past_source(family, repeats, ratio):
    from tools.validate_v41_c2a_chain import diagnostic_chunks, diagnostic_step
    from pypto_serving.model.deepseek_v41.compressed_metadata import prepare_compressed_metadata
    from pypto_serving.model.deepseek_v41.swa_metadata import prepare_swa_window_metadata

    topology = SegmentTopology(tp=2, dp=2)
    ids = torch.arange(64).reshape(2, 32)
    chunks = diagnostic_chunks(topology, [32, 32], repeat_chunks=repeats)
    context = repeats * 32
    prior_writes = set()
    for counts, starts, source in chunks:
        step = diagnostic_step(ids, topology, counts, starts, source, context, family)
        assert step.requests[0].token_ids == tuple(range(32))
        assert step.requests[1].token_ids == tuple(range(32, 64))
        window = prepare_swa_window_metadata(step, topology, cache_pages=(context + 127) // 128)
        writes = set(window.window_slots[0].tolist())
        assert not prior_writes.intersection(writes)
        prior_writes.update(writes)
        cm = prepare_compressed_metadata(step, topology, ratio=ratio, compressed_group="cmp",
            cache_pages=(context // ratio + 127) // 128, max_requests=1, state_blocks=1)
    assert prior_writes == set(range(context))
    assert int(cm.compressed_slots.max()) == context // ratio - 1
    assert int(cm.compressed_slots.max()) >= 128
    assert int(cm.compressed_lens[0, -1]) == context // ratio
    assert window.window_indices[0, -1].tolist() == list(range(context - 128, context))
    assert torch.equal(cm.index_block_table[0], cm.index_block_table[1])


@pytest.mark.parametrize("repeat,counts,continuation", [
    (0, [32, 32], False), (17, [32, 32], False), (True, [32, 32], False),
    (5, [31, 32], False), (5, [32, 32], True), (5, [0, 0], False),
])
def test_repeat_diagnostic_rejects_unbounded_or_ambiguous_inputs(repeat, counts, continuation):
    from tools.validate_v41_c2a_chain import diagnostic_chunks

    with pytest.raises(ValueError):
        diagnostic_chunks(SegmentTopology(tp=2, dp=2), counts, continuation, repeat)


def test_cache_extension_preserves_payload_layout_and_independent_index_allocation():
    from tools.validate_v41_c2a_chain import size_diagnostic_caches

    values = {
        "window_cache": torch.empty(4, 1, 128, 1, 512, dtype=torch.uint8),
        "compressed_cache": torch.empty(4, 1, 128, 1, 256, dtype=torch.uint8),
        "index_cache": torch.empty(4, 1, 128, 1, 64, dtype=torch.uint8),
        "candidate_mask": torch.empty(4, 32, 128, dtype=torch.uint8),
        "topk_indices": torch.full((4, 32, 512), -1, dtype=torch.int32),
    }
    topk = values["topk_indices"]
    size_diagnostic_caches(values, 160, "c1a")
    assert values["window_cache"].shape == (4, 2, 128, 1, 512)
    assert values["compressed_cache"].shape == (4, 2, 128, 1, 256)
    assert values["index_cache"].shape == (4, 2, 128, 1, 64)
    assert values["candidate_mask"].shape == (4, 32, 256)
    assert values["topk_indices"] is topk
    assert all(t.dtype == torch.uint8 for n, t in values.items() if n != "topk_indices")


def test_repeated_single_request_does_not_activate_empty_dp_group():
    from tools.validate_v41_c2a_chain import diagnostic_chunks, diagnostic_step
    from pypto_serving.model.deepseek_v41.compressed_metadata import prepare_compressed_metadata

    topology = SegmentTopology(tp=2, dp=2)
    counts, starts, source = diagnostic_chunks(topology, [32, 0], repeat_chunks=5)[-1]
    assert starts == [128, 0]
    step = diagnostic_step(torch.arange(64).reshape(2, 32), topology, counts, starts, source, 160, "c1a")
    assert len(step.requests) == 1 and step.requests[0].start == 128
    cm = prepare_compressed_metadata(step, topology, ratio=1, compressed_group="cmp",
        cache_pages=2, max_requests=1, state_blocks=1)
    assert cm.group_counts == (32, 0)
    assert cm.compressed_slots[2:].eq(-1).all()
    assert cm.index_block_table[2:].eq(-1).all()
    assert cm.compressed_lens[2:].eq(0).all()
