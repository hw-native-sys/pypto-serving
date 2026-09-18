# Copyright (c) PyPTO Contributors.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""Serving-owned registration point for built-in LLM PD adapters."""

from pypto_serving.model.deepseek_dspark.pd_adapter import BUILTIN_PD_ADAPTERS
from pypto_serving.serving.pd.adapter import ModelPDAdapterRegistry


def builtin_pd_adapter_registry() -> ModelPDAdapterRegistry:
    return ModelPDAdapterRegistry(BUILTIN_PD_ADAPTERS)
