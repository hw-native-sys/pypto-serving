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

from pypto_serving.model.deepseek_lowlat.weight_spec import shard_expert, transform_weight, WeightRule


@pytest.mark.parametrize("matrix", [1, 2, 3])
def test_tp8_reconstructs_all_intermediate_values(matrix):
    shape = (6, 16) if matrix == 2 else (16, 6)
    original = torch.arange(96, dtype=torch.int8).reshape(shape)
    shards = [shard_expert(original, matrix=matrix, scale=False, rank=r) for r in range(8)]
    restored = torch.cat(shards, dim=1 if matrix == 2 else 0)
    assert torch.equal(restored, original)
    assert all(s.is_contiguous() for s in shards)


def test_w2_scales_replicate_output_channels_while_w1_scales_partition():
    output_scales = torch.arange(6, dtype=torch.float32)
    hidden_scales = torch.arange(16, dtype=torch.float32)
    for rank in range(8):
        assert torch.equal(shard_expert(output_scales, matrix=2, scale=True, rank=rank), output_scales)
    restored = torch.cat([shard_expert(hidden_scales, matrix=1, scale=True, rank=r) for r in range(8)])
    assert torch.equal(restored, hidden_scales)


def test_nd_projection_transpose_is_not_nz_repacking():
    source = torch.arange(24, dtype=torch.int8).reshape(4, 6)
    rule = WeightRule("wq_b", "attn.wq_b.weight", torch.int8, transpose=True)
    actual = transform_weight(rule, source)
    assert actual.shape == (6, 4)
    assert actual[2, 1] == source[1, 2]
    assert actual.is_contiguous()


def test_float_experts_cannot_be_silently_requantized():
    with pytest.raises(ValueError, match="INT8"):
        shard_expert(torch.zeros((16, 6)), matrix=1, scale=False, rank=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_compact_projection_sources_reconstruct_without_full_device_banks():
    from types import SimpleNamespace
    from pypto_serving.model.deepseek_lowlat.weight_loader import LowlatWeightStore
    from pypto_serving.model.deepseek_lowlat.weight_spec import WEIGHT_NAMES

    assert {"wo_a_shard", "wo_b_shard"} <= WEIGHT_NAMES
    assert not {"wo_a", "wo_b"} & WEIGHT_NAMES
    store = LowlatWeightStore.__new__(LowlatWeightStore)
    store.rows = {
        "wo_a_shard": {"shape": [8, 43, 2, 3]},
        "wo_b_shard": {"shape": [8, 43 * 4, 2]},
    }
    assert store._layer_shape("wo_a") == (8, 2, 3)
    assert store._layer_shape("wo_b") == (4, 16)
    # These are the same rank group and input-axis slices used by the loader.
    a = torch.arange(48, dtype=torch.int16).reshape(8, 2, 3)
    b = torch.arange(64, dtype=torch.int8).reshape(4, 16)
    assert torch.equal(torch.stack([a[r] for r in range(8)]), a)
    assert torch.equal(torch.cat([b[:, r*2:(r+1)*2] for r in range(8)], dim=1), b)
    class Worker:
        def view_tensor(self, owner, shape, worker_id):
            return DeviceTensor(owner.data_ptr, shape, owner.dtype)
    from pypto.runtime import DeviceTensor, StackedDeviceTensor
    def owner(shape, dtype):
        return StackedDeviceTensor([DeviceTensor(0x10000*(r+1), shape[1:], dtype)
                                    for r in range(8)], shape, tuple(range(8)))
    store.worker = Worker()
    store.owners = {"wo_a_shard": owner((8, 43, 2, 3), torch.bfloat16),
                    "wo_b_shard": owner((8, 172, 2), torch.int8)}
    # Existing prefill view adapters retain their regular required weight rows.
    for name in ("wq_a", "wq_b", "csa_weights_proj"):
        store.owners[name] = owner((8, 2, 3), torch.bfloat16)
    store.programs = SimpleNamespace(prefill_metadata={"params": [
        {"name": name, "shape": shape, "dtype": "bf16", "direction": "In"}
        for name, shape in (("wo_a", [8, 43, 2, 3]), ("wo_b", [8, 43, 4, 2]),
                            ("wq_a", [8, 2, 3]), ("wq_b", [8, 2, 3]),
                            ("csa_weights_proj", [8, 2, 3]))]})
    aliases = store.prefill_bindings()
    assert aliases["wo_a"].shape == (8, 43, 2, 3)
    assert aliases["wo_b"].shape == (8, 43, 4, 2)
    assert set(store.owners) == {"wo_a_shard", "wo_b_shard", "wq_a", "wq_b", "csa_weights_proj"}
