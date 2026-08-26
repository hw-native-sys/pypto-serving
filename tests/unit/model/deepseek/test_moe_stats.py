# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import json

import torch

from pypto_serving.model.deepseek.moe_stats import MoeStatsWriter


def test_moe_stats_writer_flattens_rank_local_experts_and_marks_mtp(tmp_path):
    output = tmp_path / "stats" / "moe.jsonl"
    counts = torch.tensor(
        [
            [[1, 0], [0, 2]],
            [[3, 4], [5, 0]],
        ],
        dtype=torch.int32,
    )

    MoeStatsWriter(str(output)).write("decode", counts)

    record = json.loads(output.read_text())
    assert record["phase"] == "decode"
    assert record["layers"][0] == {
        "layer_id": 0,
        "kind": "main",
        "active_experts": 3,
        "routed_tokens": 8,
        "expert_token_counts": [1, 0, 3, 4],
    }
    assert record["layers"][1]["kind"] == "mtp"
    assert record["layers"][1]["expert_token_counts"] == [0, 2, 5, 0]
