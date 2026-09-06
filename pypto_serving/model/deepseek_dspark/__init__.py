# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DSpark (deepseek_v4_flash_dspark) target-model serving package."""

from pypto_serving.model.deepseek_dspark.npu_executor import DeepSeekV4DSparkPyptoExecutor
from pypto_serving.model.deepseek_dspark.npu_runner import (
    DSparkCacheLayout,
    DSparkModelRunner,
    build_dspark_cache_group_specs,
)
from pypto_serving.model.deepseek_dspark.weight_loader import DSparkWeightStore

__all__ = [
    "DSparkCacheLayout",
    "DSparkModelRunner",
    "DSparkWeightStore",
    "DeepSeekV4DSparkPyptoExecutor",
    "build_dspark_cache_group_specs",
]
