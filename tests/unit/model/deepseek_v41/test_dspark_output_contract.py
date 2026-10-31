# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU-only shape contracts for the V4.1 DSpark output boundary."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from pypto_serving.model.deepseek_v41.npu_runner import V41ResidentResources


def test_single_request_token_vectors_are_not_reduced_to_scalars():
    accepted = torch.arange(6)
    drafts = torch.arange(5)

    accepted_row = V41ResidentResources._dspark_output_row(
        accepted, rank=3, row=0, kind="accepted_ids")
    draft_row = V41ResidentResources._dspark_output_row(
        drafts, rank=3, row=0, kind="draft_ids")

    assert tuple(accepted_row.shape) == (6,)
    assert tuple(draft_row.shape) == (5,)


def test_dspark_token_vector_width_is_checked_before_slicing():
    with pytest.raises(ValueError, match="accepted_ids.*at least 6"):
        V41ResidentResources._validate_dspark_token_row(
            torch.arange(5), kind="accepted_ids", minimum=6)
    with pytest.raises(ValueError, match="draft_ids.*at least 5"):
        V41ResidentResources._validate_dspark_token_row(
            torch.arange(4), kind="draft_ids", minimum=5)


def test_dspark_request_rows_reject_negative_values():
    resources = V41ResidentResources.__new__(V41ResidentResources)
    lowered = SimpleNamespace(
        request_ranks=(0,), outputs={"request_rows": (-1,)}, workspace={},
    )
    with pytest.raises(ValueError, match="request_rows must be nonnegative"):
        resources._dspark_request_rows(lowered, 1)
