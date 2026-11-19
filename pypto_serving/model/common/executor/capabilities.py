# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. See the License for the specific language governing permissions and limitations under the License.
# -----------------------------------------------------------------------------------------------------------
"""Compatibility-free capability resolution for model executors."""

from pypto_serving.serving.model_extension import ModelExecutionCapabilities


def resolve_executor_capabilities(executor) -> ModelExecutionCapabilities:
    """Read the stable executor interface once during worker initialization."""

    def read(name: str, default=False):
        # Lightweight test doubles and third-party executors may predate the
        # capability contract. Keep this compatibility handling in one place,
        # outside the worker hot path.
        return getattr(executor, name, default)

    return ModelExecutionCapabilities(
        device_sampling=bool(read("supports_device_sampling")),
        device_stochastic_sampling=bool(read("supports_device_stochastic_sampling")),
        device_embedding=bool(read("supports_device_embedding")),
        device_decode_embedding=bool(read("supports_device_decode_embedding")),
        async_decode_prepare=bool(read("supports_async_decode_prepare")),
        async_decode_reclaim=bool(read("supports_async_decode_reclaim")),
        async_prefill_reclaim=bool(read("supports_async_prefill_reclaim")),
        prefill_tail_requires_reclaim=bool(read("prefill_tail_requires_reclaim_before_dispatch")),
    )
