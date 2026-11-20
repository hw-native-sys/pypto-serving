# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from pypto_serving.model.common.executor.capabilities import resolve_executor_capabilities
from pypto_serving.serving.model_extension import (
    ModelExecutionCapabilities,
    ModelStepExtension,
    RequestExecutionContext,
)


def test_model_extension_keeps_payload_opaque():
    payload = object()
    extension = ModelStepExtension("model-a", 1, payload)
    assert extension.payload is payload
    assert extension.kind == "model-a"


def test_request_context_is_shared_admission_state_without_model_fields():
    context = RequestExecutionContext("request", 128, 64, execution_generation=3)
    assert context.prompt_tokens == 128
    assert not hasattr(context, "prefix_resume")


def test_capabilities_are_resolved_once_at_executor_boundary():
    class Executor:
        supports_device_sampling = True
        supports_async_decode_prepare = True

    capabilities = resolve_executor_capabilities(Executor())
    assert capabilities == ModelExecutionCapabilities(device_sampling=True, async_decode_prepare=True)

