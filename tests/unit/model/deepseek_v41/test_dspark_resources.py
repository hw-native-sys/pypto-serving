# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host-only contracts for the V4.1 DSpark state/workspace binder."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from pypto_serving.model.deepseek_v41.dspark_resources import (  # noqa: E402
    DSPARK_MAX_BATCH_PER_DP,
    DSPARK_PADDED_ROWS,
    DSPARK_TARGET_ROWS,
    DSparkResourceLayout,
    V41DSparkLoweringProvider,
    allocate_dspark_host_resources,
    validate_resource_mapping,
)
from pypto_serving.model.deepseek_v41.dspark_runtime import (  # noqa: E402
    DSparkLoweredInputs,
)
from pypto_serving.model.deepseek_v41.npu_runner import MissingCompositeInterface  # noqa: E402


def _resources():
    layout = DSparkResourceLayout(dp_size=4, max_seq_len=8192, vocab_size=256)
    return allocate_dspark_host_resources(layout)


def test_allocator_matches_v41_decode_and_prefill_abi():
    resources = _resources()
    validate_resource_mapping(resources, "decode")
    validate_resource_mapping(resources, "prefill")
    assert resources.state["slot_ids"].shape == (4, DSPARK_MAX_BATCH_PER_DP)
    assert resources.state["tokens"].dtype == torch.int64
    assert resources.state["meta"].dtype == torch.int32
    assert torch.equal(resources.state["meta"], torch.zeros_like(resources.state["meta"]))
    assert resources.decode_workspace["x_hc"].shape == (4, DSPARK_PADDED_ROWS, 4, 5120)
    assert resources.decode_workspace["target_hidden"].shape == (4, DSPARK_TARGET_ROWS, 15360)
    assert resources.prefill_workspace["prefill_target_hidden"].shape == (4, 8192, 15360)
    assert resources.prefill_workspace["input_ids"] is resources.decode_workspace["input_ids"]
    assert all(value.is_shared() for value in resources.state.values())


def test_provider_rejects_missing_checkpoint_payload_before_dispatch():
    resources = _resources()
    provider = V41DSparkLoweringProvider(resources, state_resolver=lambda _request: (0, 1))
    batch = SimpleNamespace(request_ids=["request-0"], cache_partitions=[0])
    with pytest.raises(MissingCompositeInterface, match="payload builder"):
        provider.lower_decode(model=object(), batch=batch)


def test_provider_rejects_partial_payload_and_does_not_fabricate_weights():
    resources = _resources()

    def builder(**_kwargs):
        return DSparkLoweredInputs(
            state=resources.state,
            workspace=resources.decode_workspace,
            draft_model={"only_one_weight": object()},
            markov_model={"only_one_weight": object()},
            cache_tables={"draft": object()},
            rope_tables={"draft_cos": object()},
            target_model={"embedding_weight": object()},
        )

    provider = V41DSparkLoweringProvider(
        resources, payload_builder=builder, state_resolver=lambda _request: (0, 1),
    )
    batch = SimpleNamespace(request_ids=["request-0"], cache_partitions=[0])
    with pytest.raises(MissingCompositeInterface, match="missing ABI fields"):
        provider.lower_decode(model=object(), batch=batch)


def test_prefill_provider_requires_state_resolver_after_payload_binding():
    resources = _resources()

    def builder(**_kwargs):
        return DSparkLoweredInputs(
            state=resources.state,
            workspace=resources.prefill_workspace,
            draft_model={"bound": object()},
            markov_model={"bound": object()},
            cache_tables={"draft": object()},
            rope_tables={"draft_cos": object()},
        )

    provider = V41DSparkLoweringProvider(resources, payload_builder=builder)
    batch = SimpleNamespace(prompt_lens=[8])
    with pytest.raises(MissingCompositeInterface, match="state resolver"):
        provider.lower_prefill(
            model=object(), batch=batch, request_ids=("request-0",), sampled_token_ids=(3,)
        )
