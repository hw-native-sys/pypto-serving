# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Main host-side functional guard for the DSpark serving adaptation."""

from types import SimpleNamespace

import torch

from pypto_serving.config.types import DecodeBatch, PrefillBatch
from pypto_serving.model.deepseek_dspark import task_args as task_args_module
from pypto_serving.model.deepseek_dspark.npu_runner import (
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

    # Group 1 is idle and mirrors group 0, while group 2 keeps its own request
    # positions and RoPE. Only active group leaders publish logits.
    torch.testing.assert_close(x_hc[4], x_hc[0])
    assert staged_prefill["query_start_loc"][:, -1].tolist() == [tokens] * layout.ranks
    assert staged_prefill["logit_row_indices"][0, 0].item() == tokens - 1
    assert staged_prefill["logit_row_indices"][8, 0].item() == tokens - 1
    assert bool((staged_prefill["logit_row_indices"][4] == -1).all())
    prefill_cos = runner._packed_host_prefix(staged_prefill["swa_freqs_cos"], 96)
    torch.testing.assert_close(prefill_cos[4], prefill_cos[0])
    assert not torch.equal(prefill_cos[0], prefill_cos[8])
    for name in ("ori_slot_mapping_full", "csa_cmp_slot_mapping_full"):
        mapping = runner._packed_host_prefix(staged_prefill[name], 96)
        assert bool((mapping[0, tokens:] == -1).all())
        assert bool((mapping[8, tokens:] == -1).all())

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
    positions = staged_decode["position_ids"][0].to(torch.long)
    torch.testing.assert_close(
        staged_decode["freqs_cos"][0],
        runner._compiled.rope.swa_cos[positions].to(torch.bfloat16),
    )
    torch.testing.assert_close(
        staged_decode["compressed_freqs_cos"][0],
        runner._compiled.rope.ratio128_cos[positions].to(torch.bfloat16),
    )
    assert not torch.equal(
        staged_decode["freqs_cos"][0], staged_decode["compressed_freqs_cos"][0]
    )
