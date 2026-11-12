# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Reject mixed kernel revisions and stale shape specializations before device execution."""

from types import SimpleNamespace

import pytest

from pypto_serving.model.deepseek_dspark.eplb.abi import EPLB_ABI_VERSION, validate_kernel_contract
from pypto_serving.model.deepseek_dspark.eplb.state import Placement


def modules_for(placement, *, speculative=True):
    shape = dict(
        EPLB_ABI_VERSION=EPLB_ABI_VERSION, EPLB_ENABLED=True, N_RANKS=placement.ranks,
        N_EXPERTS_GLOBAL=placement.logical_experts, N_LOCAL=placement.local_slots,
        EPLB_LAYERS=placement.layers, EPLB_BANK_ROWS=placement.layers * placement.logical_experts,
        META_SLOTS=(placement.local_slots + 15) // 16 * 16,
    )
    modules = {
        "prefill_fwd": SimpleNamespace(**shape, l3_prefill_fwd_eplb=lambda: None),
        "decode_fwd": SimpleNamespace(**shape, l3_decode_fwd_eplb=lambda: None),
        "eplb_transfer": SimpleNamespace(EPLB_ABI_VERSION=EPLB_ABI_VERSION, make_eplb_transfer=lambda: None),
    }
    if speculative:
        modules["decode_fwd_dspark"] = SimpleNamespace(**shape, l3_decode_fwd_dspark_eplb=lambda: None)
    return modules


@pytest.mark.parametrize("ranks,redundant", [(8, 0), (8, 8), (16, 0), (16, 16)])
@pytest.mark.parametrize("speculative", [False, True])
def test_supported_kernel_exports_and_target_capacity(ranks, redundant, speculative):
    placement = Placement.initial(layers=43, logical_experts=256, ranks=ranks, redundant_experts=redundant)
    validate_kernel_contract(modules_for(placement, speculative=speculative), placement)


@pytest.mark.parametrize("name", ["prefill_fwd", "decode_fwd", "decode_fwd_dspark", "eplb_transfer"])
@pytest.mark.parametrize("version", [None, 0, 2, "1", True])
def test_mixed_kernel_revision_is_rejected(name, version):
    placement = Placement.initial(layers=43, logical_experts=256, ranks=16, redundant_experts=16)
    modules = modules_for(placement)
    modules[name].EPLB_ABI_VERSION = version
    with pytest.raises(RuntimeError, match=rf"ABI mismatch in {name}"):
        validate_kernel_contract(modules, placement)


@pytest.mark.parametrize("dimension,wrong", [
    ("N_RANKS", 8), ("N_EXPERTS_GLOBAL", 272), ("N_LOCAL", 16),
    ("EPLB_LAYERS", 1), ("EPLB_BANK_ROWS", 256), ("META_SLOTS", 17),
])
@pytest.mark.parametrize("name", ["prefill_fwd", "decode_fwd", "decode_fwd_dspark"])
def test_stale_or_logical_physical_conflated_geometry_is_rejected(name, dimension, wrong):
    placement = Placement.initial(layers=43, logical_experts=256, ranks=16, redundant_experts=16)
    modules = modules_for(placement)
    setattr(modules[name], dimension, wrong)
    with pytest.raises(RuntimeError, match=rf"geometry mismatch in {name}.{dimension}"):
        validate_kernel_contract(modules, placement)


def test_advertised_version_without_an_entry_is_rejected():
    placement = Placement.initial(layers=43, logical_experts=256, ranks=16, redundant_experts=16)
    modules = modules_for(placement)
    modules["decode_fwd_dspark"].l3_decode_fwd_dspark_eplb = None
    with pytest.raises(RuntimeError, match="kernel export decode_fwd_dspark.l3_decode_fwd_dspark_eplb"):
        validate_kernel_contract(modules, placement)


def test_identity_capacity_cannot_hide_a_disabled_routing_specialization():
    placement = Placement.initial(layers=43, logical_experts=256, ranks=16, redundant_experts=0)
    modules = modules_for(placement)
    modules["decode_fwd"].EPLB_ENABLED = False
    with pytest.raises(RuntimeError, match="decode_fwd was imported without EPLB enabled"):
        validate_kernel_contract(modules, placement)


def test_executor_rejects_incompatible_exports_before_compiling_any_kernel(monkeypatch):
    from pypto_serving.config.types import EplbConfig
    from pypto_serving.model.deepseek_dspark import npu_executor as executor_module

    placement = Placement.initial(layers=43, logical_experts=256, ranks=16, redundant_experts=16)
    modules = modules_for(placement, speculative=False)
    modules["eplb_transfer"].EPLB_ABI_VERSION = 0
    monkeypatch.setattr(executor_module, "DSparkWeightStore", lambda **_kwargs: SimpleNamespace(
        validate_startup_contract=lambda **_kwargs: None,
    ))
    executor = executor_module.DeepSeekV4DSparkPyptoExecutor(
        device_ids=range(16), compile_kernels=True,
        eplb_config=EplbConfig(enabled=True, num_redundant_experts=16, max_migration_bytes=1 << 34),
    )
    monkeypatch.setattr(executor, "_load_kernel_modules", lambda *_args, **_kwargs: modules)

    def unexpected_compile(*_args, **_kwargs):
        pytest.fail("Incompatible EPLB exports reached the kernel compiler")

    monkeypatch.setattr(executor, "_compile_l3_callable", unexpected_compile)
    model = SimpleNamespace(
        config=SimpleNamespace(hidden_size=4096, num_hidden_layers=43, num_attention_heads=64,
                               num_key_value_heads=1, head_dim=512, vocab_size=129280,
                               max_position_embeddings=1_048_576),
        runtime=SimpleNamespace(max_batch_size=1, page_size=32, max_seq_len=1024),
        extra={"family": "deepseek_v4", "checkpoint_format": "w8a8-compressed-tensors",
               "compress_ratios": (0,) * 44, "weight_map": {}, "model_dir": "unused",
               "config_data": {"n_routed_experts": 256, "moe_intermediate_size": 2048}},
    )
    with pytest.raises(RuntimeError, match="ABI mismatch in eplb_transfer"):
        executor._compile_model(model)
