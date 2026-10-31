# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Model executor interfaces and helpers."""

from importlib import import_module


def resolve_executor_cls(name):
    """Load an executor shared by engine configuration and worker startup."""
    executors = {
        "PyptoQwen14BExecutor": ("qwen", "Qwen314BPyptoExecutor"),
        "PyptoDeepSeekV4Executor": ("deepseek", "DeepSeekV4PyptoExecutor"),
        "PyptoDeepSeekV4DSparkExecutor": ("deepseek_dspark", "DeepSeekV4DSparkPyptoExecutor"),
        "PyptoDeepSeekV41Executor": ("deepseek_v41", "DeepSeekV41PyptoExecutor"),
        "PyptoGlm53Executor": ("glm53", "Glm53PyptoExecutor"),
    }
    if name not in executors:
        from pypto_serving.model.common.executor.executor import ModelExecutor

        return ModelExecutor
    family, class_name = executors[name]
    return getattr(import_module(f"pypto_serving.model.{family}.npu_executor"), class_name)
