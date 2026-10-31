# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host contracts for the serving-owned V4.1 DSpark lowering seam."""

from types import SimpleNamespace

import pytest

from pypto_serving.model.deepseek_v41.dspark_runtime import (
    DSparkDecodeDispatch,
    DSparkLoweredInputs,
    DSparkPrefillDispatch,
)
from pypto_serving.model.deepseek_v41.npu_runner import (
    MissingCompositeInterface,
    V41ResidentResources,
)


def _lowered():
    return DSparkLoweredInputs(
        state=object(),
        workspace={"request_rows": (0,)},
        target_model={"target": object()},
        draft_model={"draft": object()},
        markov_model={"markov": object()},
        cache_tables={"cache": object()},
        rope_tables={"rope": object()},
        outputs={"accepted_counts": object()},
    )


class _Provider:
    def lower_prefill(self, **kwargs):
        return DSparkPrefillDispatch(
            request_id=kwargs["request_ids"][0],
            slot=0,
            generation=1,
            prompt_length=8,
            anchor=8,
            stages=(),
            lowered=_lowered(),
        )

    def lower_decode(self, **kwargs):
        return DSparkDecodeDispatch(
            request_ids=tuple(kwargs["batch"].request_ids),
            stages=(),
            lowered=_lowered(),
        )


def _resources(provider=None):
    resources = V41ResidentResources.__new__(V41ResidentResources)
    resources.dspark_lowering_provider = provider
    return resources


def test_missing_provider_is_explicit():
    resources = _resources()
    with pytest.raises(MissingCompositeInterface, match="lowering provider"):
        resources.lower_dspark_decode(model=object(), batch=object())


def test_provider_returns_typed_complete_prefill_dispatch():
    resources = _resources(_Provider())
    batch = SimpleNamespace(request_ids=["request-0"])
    dispatch = resources.lower_dspark_prefill(
        model=object(), batch=batch, request_ids=("request-0",), sampled_token_ids=(11,)
    )
    assert isinstance(dispatch, DSparkPrefillDispatch)
    assert dispatch.lowered is not None
    assert dispatch.request_id == "request-0"


def test_provider_returns_typed_complete_decode_dispatch():
    resources = _resources(_Provider())
    batch = SimpleNamespace(request_ids=["request-0"])
    dispatch = resources.lower_dspark_decode(model=object(), batch=batch)
    assert isinstance(dispatch, DSparkDecodeDispatch)
    assert dispatch.lowered is not None
    assert tuple(dispatch.request_ids) == ("request-0",)


class _IncompleteProvider(_Provider):
    def lower_decode(self, **kwargs):
        return DSparkDecodeDispatch(request_ids=("request-0",), stages=(), lowered=None)


def test_provider_cannot_return_stage_only_payload():
    resources = _resources(_IncompleteProvider())
    with pytest.raises(MissingCompositeInterface, match="complete lowered inputs"):
        resources.lower_dspark_decode(
            model=object(), batch=SimpleNamespace(request_ids=["request-0"])
        )
