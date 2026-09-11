# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Main host-side functional guard for the DSpark serving adaptation."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from pypto_serving.config.types import DecodeBatch, PrefillBatch
from pypto_serving.model.deepseek_dspark import task_args as task_args_module
from pypto_serving.model.deepseek_dspark.npu_runner import (
    DSPARK_CACHE_GROUP_NAMES,
    DSparkCacheLayout,
    DSparkCompiledKernels,
    DSparkDrafterRequestRow,
    DSparkModelRunner,
    DSparkRopeTables,
)


def _runner(*, speculative: bool = False) -> DSparkModelRunner:
    max_position = 512
    rows = torch.arange(max_position * 64, dtype=torch.float32).reshape(max_position, 64)
    rope = DSparkRopeTables(
        max_position=max_position,
        swa_cos=rows.to(torch.bfloat16),
        swa_sin=(rows + 1).to(torch.bfloat16),
        ratio4_cos=(rows + 2).to(torch.bfloat16),
        ratio4_sin=(rows + 3).to(torch.bfloat16),
        ratio128_cos=(rows + 4).to(torch.bfloat16),
        ratio128_sin=(rows + 5).to(torch.bfloat16),
        ratio128_half_cos=rows[:, :32] + 6,
        ratio128_half_sin=rows[:, :32] + 7,
    )
    # Keep the production topology but shrink prefill's token and hidden axes.
    layout = DSparkCacheLayout(
        prefill_tokens=128,
        prefill_local_tokens=32,
        hidden_size=4,
    )
    runner = DSparkModelRunner(
        compiled=DSparkCompiledKernels(
            layout=layout,
            model_dir="unused",
            weight_map={},
            weight_store=None,
            compress_ratios=(0,) * 43,
            layer_plan=(),
            kernel_dir="unused",
            rope=rope,
            num_speculative_tokens=7 if speculative else 0,
        )
    )
    runner._cache_group_num_blocks = {name: 8 for name in DSPARK_CACHE_GROUP_NAMES}
    runner._prefill_task_args = task_args_module.prefill_task_args(runner)
    runner._prefill_task_args.allocate_host_shared(None)
    runner._decode_task_args = [task_args_module.decode_task_args(runner)]
    runner._decode_task_args[0].allocate_host_shared(None)
    if speculative:
        runner._drafter_task_args = task_args_module.drafter_task_args(runner)
        runner._drafter_task_args.allocate_host_shared(None)
        runner._markov_task_args = task_args_module.markov_task_args(runner)
        runner._markov_task_args.allocate_host_shared(None)
        from pypto_serving.model.deepseek_dspark.npu_runner import (
            DSPARK_DRAFTER_BATCHES,
            DSPARK_DRAFTER_CONTEXT_BUCKETS,
            DSPARK_DRAFT_LAYERS,
            DSPARK_DRAFTER_TABLE_BLOCKS,
        )

        ranks = layout.ranks
        for extent in DSPARK_DRAFTER_CONTEXT_BUCKETS:
            group_extent = 4 * extent
            runner._drafter_context_staging[extent] = {
                "context_group_position_ids": torch.zeros(
                    (ranks, group_extent), dtype=torch.int32
                ),
                "context_group_slot_mapping": torch.full(
                    (ranks, DSPARK_DRAFT_LAYERS, group_extent), -1, dtype=torch.int64
                ),
                "context_group_freqs_cos": torch.zeros(
                    (ranks, group_extent, 64), dtype=torch.bfloat16
                ),
                "context_group_freqs_sin": torch.zeros(
                    (ranks, group_extent, 64), dtype=torch.bfloat16
                ),
            }
        for padded_batch in DSPARK_DRAFTER_BATCHES:
            runner._drafter_block_table_staging[padded_batch] = torch.zeros(
                (ranks, DSPARK_DRAFT_LAYERS, padded_batch, DSPARK_DRAFTER_TABLE_BLOCKS),
                dtype=torch.int32,
            )
    return runner


def _block_rows(count: int) -> list[dict[str, list[int]]]:
    rows = []
    for request in range(count):
        rows.append(
            {
                "ori": [(request + offset) % 8 for offset in range(6)],
                "cmp_c128": [request % 4],
                "cmp_c4": [request % 8, (request + 1) % 8],
                "idx": [request % 8, (request + 1) % 8],
                "hca_state": [request % 8],
                "csa_state": [(request + offset) % 8 for offset in range(4)],
                "csa_inner_state": [(request + offset) % 8 for offset in range(4)],
            }
        )
    return rows


def test_prefill_to_decode_staging_contract() -> None:
    runner = _runner()
    layout = runner._compiled.layout
    tokens = 95
    embeddings = torch.arange(tokens * 2 * 4, dtype=torch.float32).reshape(tokens * 2, 4)
    prefill = PrefillBatch(
        request_ids=["group-0", "group-2"],
        token_ids=torch.arange(tokens * 2, dtype=torch.long),
        input_embeddings=embeddings,
        seq_lens=[tokens, 128 + tokens],
        chunk_lens=[tokens, tokens],
        chunk_offsets=[0, tokens],
        chunk_starts=[0, 128],
        block_ids_by_group=_block_rows(2),
        cache_partitions=[0, 2],
    )

    prepared_prefill = runner.prepare_prefill_inputs(
        SimpleNamespace(runtime=SimpleNamespace(max_seq_len=512)), prefill
    )
    runner._stage_prefill_inputs(prepared_prefill)
    staged_prefill = runner._prefill_task_args.tensors
    assert prepared_prefill.physical_tokens == 96

    x_hc = runner._packed_host_prefix(staged_prefill["x_hc"], 96)
    expected_0 = embeddings[:tokens].unsqueeze(1).expand(-1, layout.hc_mult, -1)
    expected_2 = embeddings[tokens:].unsqueeze(1).expand(-1, layout.hc_mult, -1)
    torch.testing.assert_close(x_hc[0, :tokens], expected_0)
    torch.testing.assert_close(x_hc[8, :tokens], expected_2)
    assert bool(torch.count_nonzero(x_hc[:, tokens:]) == 0)

    # Groups 1 and 3 are idle: the kernel skips their attention and sampling
    # tails natively (pypto-lib#1161), so their staging stays zero-initialized
    # (query_start_loc terminal 0) instead of mirroring an active group.
    assert bool(torch.count_nonzero(x_hc[4]) == 0)
    terminals = staged_prefill["query_start_loc"][:, -1].tolist()
    assert terminals == [tokens] * 4 + [0] * 4 + [tokens] * 4 + [0] * 4
    assert staged_prefill["logit_row_indices"][0, 0].item() == tokens - 1
    assert staged_prefill["logit_row_indices"][8, 0].item() == tokens - 1
    assert bool((staged_prefill["logit_row_indices"][4] == -1).all())
    prefill_cos = runner._packed_host_prefix(staged_prefill["swa_freqs_cos"], 96)
    assert bool(torch.count_nonzero(prefill_cos[4]) == 0)
    assert not torch.equal(prefill_cos[0], prefill_cos[8])
    for name in ("ori_slot_mapping_full", "csa_cmp_slot_mapping_full"):
        mapping = runner._packed_host_prefix(staged_prefill[name], 96)
        assert bool((mapping[0, tokens:] == -1).all())
        assert bool((mapping[8, tokens:] == -1).all())
        assert bool((mapping[4] == -1).all())

    decode = DecodeBatch(
        request_ids=["group-0", "group-2", "group-0-second"],
        token_ids=torch.tensor([[10], [20], [30]], dtype=torch.long),
        hidden_states=None,
        seq_lens=torch.tensor([96, 224, 97], dtype=torch.int32),
        block_ids_by_group=_block_rows(3),
        cache_partitions=[0, 2, 0],
        allow_device_greedy_sampling=True,
    )
    prepared_decode = runner.prepare_decode_inputs(SimpleNamespace(), decode)
    staged_decode = runner._decode_task_args[0].tensors

    # Uneven requests are spread across their TP owners. Every inactive owner
    # has an explicit zero-token contract rather than a fake padding token.
    assert prepared_decode.sampled_slots == ((0, 0), (8, 0), (1, 0))
    expected_owner_tokens = [0] * layout.ranks
    for rank in (0, 1, 8):
        expected_owner_tokens[rank] = layout.decode_seq
    assert staged_decode["num_tokens_per_owner"].tolist() == expected_owner_tokens
    assert staged_decode["input_ids"].shape == (layout.ranks, layout.decode_local_tokens)
    assert staged_decode["position_ids"].shape == (layout.ranks, layout.decode_tokens)
    assert bool((staged_decode["input_ids"][4] == 0).all())
    assert bool((staged_decode["swa_indices"][4] == -1).all())
    assert bool((staged_decode["swa_lens"][4] == 0).all())

    # The fixed S=8 tile commits only row zero. Noise rows and inactive groups
    # cannot write raw KV or recurrent state, and compressed RoPE remains
    # distinct from the ordinary SWA profile used by the query path.
    for rank, active_requests in ((0, 2), (8, 1), (4, 0)):
        for name in (
            "swa_slot_mapping",
            "hca_ori_slot_mapping",
            "csa_ori_slot_mapping",
            "hca_state_slot_mapping",
            "csa_state_slot_mapping",
            "csa_inner_state_slot_mapping",
        ):
            assert int((staged_decode[name][rank] >= 0).sum()) == active_requests
    # Since pypto-lib#1182 the decode RoPE tables ride the owner-token
    # T_DYN axis: each rank carries RoPE for its own local rows only.
    local_positions = staged_decode["position_ids_local"][0].to(torch.long)
    assert staged_decode["freqs_cos"][0].shape == (layout.decode_local_tokens, 64)
    torch.testing.assert_close(
        staged_decode["freqs_cos"][0],
        runner._compiled.rope.swa_cos[local_positions].to(torch.bfloat16),
    )
    torch.testing.assert_close(
        staged_decode["compressed_freqs_cos"][0],
        runner._compiled.rope.ratio128_cos[local_positions].to(torch.bfloat16),
    )
    assert not torch.equal(
        staged_decode["freqs_cos"][0], staged_decode["compressed_freqs_cos"][0]
    )


def test_prefill_context_bound_uses_each_requests_own_length() -> None:
    """A short request must not inherit a longer group's context rejection.

    The packed physical extent is the batch maximum; a request near the
    context ceiling sharing a dispatch with a longer chunk stays valid as
    long as its own effective end fits, while a genuine overflow of its own
    length is still rejected.
    """
    runner = _runner()
    tokens = 120
    short_tokens = 8
    embeddings = torch.arange((tokens + short_tokens) * 4, dtype=torch.float32).reshape(
        tokens + short_tokens, 4
    )

    def _batch(chunk_start: int) -> PrefillBatch:
        return PrefillBatch(
            request_ids=["group-0", "group-2"],
            token_ids=torch.arange(tokens + short_tokens, dtype=torch.long),
            input_embeddings=embeddings,
            seq_lens=[tokens, chunk_start + short_tokens],
            chunk_lens=[tokens, short_tokens],
            chunk_offsets=[0, tokens],
            chunk_starts=[0, chunk_start],
            block_ids_by_group=_block_rows(2),
            cache_partitions=[0, 2],
        )

    model = SimpleNamespace(runtime=SimpleNamespace(max_seq_len=512))
    prepared = runner.prepare_prefill_inputs(model, _batch(500))
    assert prepared.physical_tokens == tokens

    with pytest.raises(ValueError, match="exceed max_seq_len=512"):
        runner.prepare_prefill_inputs(model, _batch(505))


def _pypto_lib_function(module_name: str, function_name: str) -> ast.FunctionDef:
    """Parse one l3 entry point from the pinned pypto-lib dspark kernels."""
    kernel_file = (
        Path(__file__).resolve().parents[4]
        / "pypto-lib"
        / "models"
        / "deepseek_v4_flash_dspark"
        / f"{module_name}.py"
    )
    module = ast.parse(kernel_file.read_text(encoding="utf-8"))
    return next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )


def test_dspark_task_arg_orders_match_pypto_lib_abis() -> None:
    """The pinned tuples are the exact positional l3_prefill/decode contracts."""
    prefill = _pypto_lib_function("prefill_fwd", "l3_prefill_fwd")
    decode = _pypto_lib_function("decode_fwd", "l3_decode_fwd")
    assert tuple(arg.arg for arg in prefill.args.args) == (
        task_args_module._PREFILL_TENSOR_ORDER
    )
    assert tuple(arg.arg for arg in decode.args.args) == (
        task_args_module._DECODE_TENSOR_ORDER
    )
    assert len(task_args_module._PREFILL_TENSOR_ORDER) == 101
    assert len(task_args_module._DECODE_TENSOR_ORDER) == 109


def test_drafter_and_markov_task_arg_orders_match_pypto_lib_abis() -> None:
    """The pinned drafter/markov tuples are the exact l3 positional contracts."""
    drafter = _pypto_lib_function("dspark_drafter", "l3_dspark_drafter")
    markov = _pypto_lib_function("dspark_markov", "l3_distributed_markov_sample")
    assert tuple(arg.arg for arg in drafter.args.args) == (
        task_args_module._DRAFTER_TENSOR_ORDER
    )
    assert tuple(arg.arg for arg in markov.args.args) == (
        task_args_module._MARKOV_TENSOR_ORDER
    )
    assert len(task_args_module._DRAFTER_TENSOR_ORDER) == 59
    assert len(task_args_module._MARKOV_TENSOR_ORDER) == 12


def test_drafter_staging_contract() -> None:
    """Dense rows stage the fixture's decode contract over stable leases."""
    from pypto_serving.model.deepseek_dspark.npu_runner import (
        DSPARK_DRAFTER_FILLER_BLOCK_BASE,
        DSPARK_DRAFTER_RING_BLOCKS,
    )

    runner = _runner(speculative=True)
    layout = runner._compiled.layout
    assert runner.speculative
    rows_by_rank: list[list] = [[] for _ in range(layout.ranks)]
    for request_id, lease, anchor, valid, token, hidden_row in (
        ("a", 63, 200, 5, 11, 0),
        ("b", 62, 300, 1, 22, 8),
    ):
        rows_by_rank[0].append(
            DSparkDrafterRequestRow(
                request_id=request_id,
                group=0,
                lease=lease,
                anchor=anchor,
                valid_count=valid,
                token_source=token,
                hidden_row=hidden_row,
                decode_mode=True,
            )
        )
    rows_by_rank[1].append(
        DSparkDrafterRequestRow(
            request_id="c",
            group=0,
            lease=0,
            anchor=100,
            valid_count=8,
            token_source=33,
            hidden_row=0,
            decode_mode=True,
        )
    )
    rows_by_rank[8].append(
        DSparkDrafterRequestRow(
            request_id="d",
            group=2,
            lease=5,
            anchor=64,
            valid_count=3,
            token_source=44,
            hidden_row=0,
            decode_mode=True,
        )
    )
    context_rows = 4 * layout.decode_seq
    hidden = torch.zeros(
        (layout.ranks, context_rows, task_args_module.DSPARK_MAIN_HIDDEN_DIM),
        dtype=torch.bfloat16,
    )
    batch, staged_rows = runner._prepare_drafter_inputs(
        rows_by_rank, hidden=hidden, context_rows=context_rows
    )
    assert batch == 4
    assert staged_rows == context_rows

    tensors = runner._drafter_task_args.tensors
    context = runner._drafter_context_staging[context_rows]
    # The selectors dispatch as packed prefixes over the max backing, so
    # read them through the same packed view.
    selectors = {
        name: runner._packed_host_prefix(tensors[name], batch)
        for name in ("num_sampled", "last_sampled", "anchor_positions")
    }
    # num_sampled is sign-only mode selection: 1 for real decode rows, 0 for
    # the dense fillers that keep the uniform padded batch.
    assert selectors["num_sampled"][0, :4].tolist() == [1, 1, 0, 0]
    assert selectors["num_sampled"][8, :4].tolist() == [1, 0, 0, 0]
    assert selectors["last_sampled"][0, 0].item() == 11
    assert selectors["anchor_positions"][0, :2].tolist() == [200, 300]

    # Group context assembly is rank-major and ends at each anchor; padding
    # rows keep position zero with -1 slots.
    positions = context["context_group_position_ids"][0].tolist()
    assert positions[0:5] == [196, 197, 198, 199, 200]
    assert positions[layout.decode_seq] == 300
    assert positions[context_rows : 2 * context_rows] == [
        93, 94, 95, 96, 97, 98, 99, 100
    ] + [0] * (context_rows - 8)
    slots = context["context_group_slot_mapping"][0]
    base = 63 * DSPARK_DRAFTER_RING_BLOCKS
    for layer in range(3):
        block = base + (6 + 7 * layer) % DSPARK_DRAFTER_RING_BLOCKS
        assert slots[layer, 0:5].tolist() == [
            block * 32 + offset for offset in (4, 5, 6, 7, 8)
        ]
        assert bool((slots[layer, 5 : layout.decode_seq] == -1).all())
    assert bool((slots[:, 3 * context_rows :] == -1).all())

    # Query rows: seven fresh positions per real request, -1 past them.
    query_positions = tensors["query_group_position_ids"][0].tolist()
    assert query_positions[0:7] == [201, 202, 203, 204, 205, 206, 207]
    assert query_positions[7:14] == [301, 302, 303, 304, 305, 306, 307]
    query_slots = tensors["query_group_slot_mapping"][0]
    assert bool((query_slots[:, :14] >= 0).all())
    assert bool((query_slots[:, 14:112] == -1).all())
    assert bool((query_slots[:, 112:119] >= 0).all())
    assert bool((query_slots[:, 119:] == -1).all())

    # Block tables: real rows use their lease ring, dense fillers share the
    # read-only filler range, and the two namespaces stay disjoint.
    tables = runner._drafter_block_table_staging[batch]
    assert tables[0, 0, 0, 0].item() == base  # lease 63, layer 0, logical 0
    assert tables[0, 1, 0, 0].item() == base + 7 % DSPARK_DRAFTER_RING_BLOCKS
    assert tables[0, 0, 1, 0].item() == 62 * DSPARK_DRAFTER_RING_BLOCKS
    assert tables[0, 0, 2, 0].item() == DSPARK_DRAFTER_FILLER_BLOCK_BASE

    # Markov: one logit row per (real request, step), -1 elsewhere.
    markov = runner._markov_task_args.tensors
    assert markov["logit_row_indices"][0, :14].tolist() == list(range(14))
    assert bool((markov["logit_row_indices"][0, 14:] == -1).all())
    assert bool((markov["logit_row_indices"][4] == -1).all())


def test_accept_dspark_tokens_semantics() -> None:
    """Longest matching prefix plus the bonus, for every m in 0..7."""
    from pypto_serving.model.deepseek_dspark.npu_runner import _accept_dspark_tokens

    draft = [10, 11, 12, 13, 14, 15, 16]
    # All seven match: the eighth prediction is the bonus.
    main = [10, 11, 12, 13, 14, 15, 16, 99]
    assert _accept_dspark_tokens(main, draft) == ([10, 11, 12, 13, 14, 15, 16, 99], 7)
    # First-row mismatch: only the bonus (the anchor's own prediction).
    assert _accept_dspark_tokens([50, 0, 0, 0, 0, 0, 0, 0], draft) == ([50], 0)
    # Mid-chain rejection at m=3.
    main = [10, 11, 12, 40, 0, 0, 0, 0]
    assert _accept_dspark_tokens(main, draft) == ([10, 11, 12, 40], 3)
    # A draft row matching the bonus position cannot over-run the window.
    with pytest.raises(ValueError, match="ran past"):
        _accept_dspark_tokens([10, 11, 12, 13, 14, 15, 16, 17], draft + [17])


def test_run_decode_accepts_and_redrafts(monkeypatch) -> None:
    """Acceptance drives state updates and the next drafter context rows."""
    runner = _runner(speculative=True)
    layout = runner._compiled.layout
    state = runner._reserve_drafter_state("spec", group=0, prompt_len=32)
    state.pending_draft_tokens = [501, 502, 999, 504, 505, 506, 507]
    state.prompt_len = 64
    # The committed count is authoritative: prompt (64) minus one, i.e. the
    # verify step runs with its input token at position 63.
    state.committed_count = 63
    runner._compiled.decode = object()
    runner._compiled.drafter = object()
    runner._compiled.markov = object()
    runner._l3_shared_buffers_ready = True

    decode = DecodeBatch(
        request_ids=["spec"],
        token_ids=torch.tensor([[10]], dtype=torch.long),
        hidden_states=None,
        seq_lens=torch.tensor([64], dtype=torch.int32),
        block_ids_by_group=_block_rows(1),
        cache_partitions=[0],
        allow_device_greedy_sampling=True,
    )

    dispatches: list[object] = []
    monkeypatch.setattr(
        runner,
        "_run_l3",
        lambda program, *args, config=None: dispatches.append(program),
    )
    # Stub the lazy device sources the dispatch-args build would resolve.
    monkeypatch.setattr(
        runner,
        "_alloc_zeroed_stacked_tensor",
        lambda name, shape, dtype, scope=None: torch.zeros(shape, dtype=dtype),
    )
    monkeypatch.setattr(runner, "_static_lm_head_weight_tensor", lambda: torch.zeros(1))
    monkeypatch.setattr(
        runner, "_materialize_embedding_device_weight", lambda: torch.zeros(1)
    )
    runner._drafter_host_weights = {
        name: torch.zeros(1)
        for name in (
            *task_args_module._DRAFTER_TENSOR_ORDER,
            *task_args_module._MARKOV_TENSOR_ORDER,
        )
    }
    runner._stacked_host_weights = {
        name: torch.zeros(1)
        for name in task_args_module._DECODE_TENSOR_ORDER
    }
    monkeypatch.setattr(runner, "_static_weight", lambda name: torch.zeros(1))
    monkeypatch.setattr(
        runner,
        "_device_cache_values",
        lambda: __import__("collections").defaultdict(lambda: torch.zeros(1)),
    )

    # The mocked kernel never writes its outputs: pre-stage the greedy
    # predictions the verify rows would have produced.
    main = [501, 502, 8, 9, 10, 11, 12, 13]
    sampled_slot = runner._decode_task_args[0].tensors["sampled_ids"]
    for offset, token in enumerate(main):
        sampled_slot[0, offset, 0] = token

    def fake_read(mirror, *, rows):
        assert mirror is runner._drafter_hidden_mirror
        return (
            torch.arange(rows * task_args_module.DSPARK_MAIN_HIDDEN_DIM, dtype=torch.float32)
            .reshape(1, rows, task_args_module.DSPARK_MAIN_HIDDEN_DIM)
            .to(torch.bfloat16)
            .expand(layout.ranks, -1, -1)
            .contiguous()
        )

    monkeypatch.setattr(runner, "_read_drafter_hidden", fake_read)
    fake_drafts = runner._markov_task_args.tensors["draft_token_ids"]
    fake_drafts[0, 0] = torch.arange(7, dtype=torch.int32) + 700

    result = runner.run_decode(SimpleNamespace(), decode)

    # Drafts [501, 502, 999, ...] match the first two predictions.
    assert result.accepted_token_ids == [[501, 502, 8]]
    assert state.matched_drafts == 2
    assert state.committed_count == 63 + 3
    assert state.pending_draft_tokens == [700, 701, 702, 703, 704, 705, 706]

    # The drafter row stages the committed window ending at the new anchor:
    # the verify step ran at position 63 (seq_len 64), accepted three tokens
    # (positions 63..65), so the next anchor is 63 + 3 - 1 = 65.
    tensors = runner._drafter_task_args.tensors
    context = runner._drafter_context_staging[32]
    selectors = {
        name: runner._packed_host_prefix(tensors[name], 4)
        for name in ("num_sampled", "anchor_positions")
    }
    assert selectors["num_sampled"][0, 0].item() == 1
    assert selectors["anchor_positions"][0, 0].item() == 63 + 3 - 1
    positions = context["context_group_position_ids"][0].tolist()
    assert positions[:3] == [63, 64, 65]
    summary = runner.dspark_speculation_summary()
    assert summary["verify_steps"] == 1.0
    assert summary["matched_drafts"] == 2.0
