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
    _PREFILL_GROUP_DYNAMIC_NAMES,
    _PREFILL_LOCAL_DYNAMIC_NAMES,
    DSPARK_CACHE_GROUP_NAMES,
    DSparkCacheLayout,
    DSparkCompiledKernels,
    DSparkModelRunner,
    DSparkRopeTables,
)


def _runner() -> DSparkModelRunner:
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
        )
    )
    runner._cache_group_num_blocks = {name: 8 for name in DSPARK_CACHE_GROUP_NAMES}
    runner._prefill_task_args = task_args_module.prefill_task_args(runner)
    runner._prefill_task_args.allocate_host_shared(None)
    runner._decode_task_args = [task_args_module.decode_task_args(runner)]
    runner._decode_task_args[0].allocate_host_shared(None)
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


def test_dspark_target_hidden_axis_binding_matches_kernel() -> None:
    """dspark_target_hidden is a rank-local BF16 output on both programs.

    Prefill binds it to FWD_TOKENS_DYN (each rank's OWNED prompt rows, the same
    axis as input_ids) while x_out keeps the gathered FWD_GROUP_TOKENS_DYN
    axis; decode binds it to the fixed local decode tile. Serving must slice
    the prefill slot at the local extent and keep the decode scratch at the
    full 16x8 tile, which the dynamic-name sets encode.
    """
    prefill_args = {
        arg.arg: ast.unparse(arg.annotation)
        for arg in _pypto_lib_function("prefill_fwd", "l3_prefill_fwd").args.args
    }
    decode_args = {
        arg.arg: ast.unparse(arg.annotation)
        for arg in _pypto_lib_function("decode_fwd", "l3_decode_fwd").args.args
    }

    prefill_hidden = prefill_args["dspark_target_hidden"]
    assert "pl.Out" in prefill_hidden
    assert "FWD_TOKENS_DYN" in prefill_hidden
    assert "MAIN_HIDDEN_DIM" in prefill_hidden
    assert "pl.BF16" in prefill_hidden
    assert "FWD_GROUP_TOKENS_DYN" in prefill_args["x_out"]

    decode_hidden = decode_args["dspark_target_hidden"]
    assert "pl.Out" in decode_hidden
    assert "T_DYN" in decode_hidden
    assert "MAIN_HIDDEN_DIM" in decode_hidden
    assert "pl.BF16" in decode_hidden

    assert "dspark_target_hidden" in _PREFILL_LOCAL_DYNAMIC_NAMES
    assert "dspark_target_hidden" not in _PREFILL_GROUP_DYNAMIC_NAMES
    assert "hidden_workspace" in task_args_module._DECODE_TENSOR_ORDER
