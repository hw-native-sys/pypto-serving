# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import pytest
import torch

from pypto.runtime import DeviceTensor, StackedDeviceTensor
from pypto_serving.model.deepseek_lowlat.bindings import frame_shape, ordered_arguments, parameter_rows


def row(name, shape, dtype="int32"):
    return {"name": name, "shape": shape, "dtype": dtype, "direction": "In", "layout": "ND"}


def stacked(shape=(8, 2), dtype=torch.int32, worker_ids=tuple(range(8))):
    shards = [DeviceTensor(0x10000 * (rank + 1), shape[1:], dtype) for rank in range(8)]
    return StackedDeviceTensor(shards, shape, worker_ids)


def test_generated_order_and_ssa_normalization():
    metadata = {"params": [row("b__ssa_v0", [8, 2]), row("a__ssa_v0", None)]}
    value = stacked()
    result = ordered_arguments(metadata, {"a": 3, "b": value})
    assert result == (value, 3)


def test_scalar_has_no_tensor_memory_layout():
    scalar = row("terminal", None)
    scalar["layout"] = None
    assert ordered_arguments({"params": [scalar]}, {"terminal": 1}) == (1,)


def test_dynamic_request_axes_do_not_scale_with_prompt_tokens():
    assert frame_shape("csa_compress_state_block_table", (8, -1, 524288), 8192) == (8, 1, 524288)
    assert frame_shape("state_checkpoint_ends", (8, -1, -1, 6), 8192) == (8, 1, 16, 6)
    assert frame_shape("dspark_target_hidden", (8, -1, 12288), 128) == (8, 16, 12288)
    assert frame_shape("swa_freqs_cos", (8, -1, 64), 128) == (8, 128, 64)
    with pytest.raises(ValueError, match="unknown dynamic"):
        frame_shape("new_unclassified_param", (8, -1), 128)


@pytest.mark.parametrize("value", [True, 0.0, 2**31, -(2**31) - 1])
def test_scalar_rejects_bool_float_and_overflow(value):
    with pytest.raises(ValueError, match="scalar INT32"):
        ordered_arguments({"params": [row("terminal", None)]}, {"terminal": value})


def test_rank_order_and_shape_are_checked():
    metadata = {"params": [row("ids", [8, 2])]}
    with pytest.raises(ValueError, match="ABI mismatch"):
        ordered_arguments(metadata, {"ids": stacked(worker_ids=tuple(reversed(range(8))))})
    with pytest.raises(ValueError, match="ABI mismatch"):
        ordered_arguments(metadata, {"ids": stacked((8, 3))})


def test_only_declared_shared_host_control_can_cross_directly():
    metadata = {"params": [row("num_tokens_per_owner", [8])]}
    with pytest.raises(ValueError, match="pre-fork"):
        ordered_arguments(metadata, {"num_tokens_per_owner": torch.ones(8, dtype=torch.int32)})
    host = torch.ones(8, dtype=torch.int32).share_memory_()
    assert ordered_arguments(metadata, {"num_tokens_per_owner": host})[0] is host
    with pytest.raises(TypeError, match="Worker-owned"):
        ordered_arguments({"params": [row("ids", [8, 2])]}, {"ids": torch.ones((8, 2), dtype=torch.int32)})


def test_ambiguous_or_missing_generated_parameters_fail():
    with pytest.raises(ValueError, match="duplicate"):
        parameter_rows({"params": [row("x__ssa_v0", [8, 2]), row("x__ssa_v1", [8, 2])]})
    with pytest.raises(ValueError, match="missing"):
        ordered_arguments({"params": [row("x", [8, 2])]}, {})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
