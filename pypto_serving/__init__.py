# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Public API for PyPTO Serving."""

from importlib import import_module

# Transfer contracts are intentionally importable on control hosts that do not
# carry Torch or an NPU runtime. Preserve the package's public API lazily.
_EXPORT_MODULES = {
    "ParallelConfig": "config.parallel",
    "GenerateConfig": "config.types",
    "KVCacheGroupSpec": "config.types",
    "KVCacheSpec": "config.types",
    "RuntimeConfig": "config.types",
    "ModelLoader": "model.model_loader",
    "AsyncLLMEngine": "serving.engine.async_engine",
    "EngineConfig": "serving.engine.async_engine",
    "ReplicaEngineCore": "serving.engine.async_engine",
}


def __getattr__(name):
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{_EXPORT_MODULES[name]}"), name)
    globals()[name] = value
    return value

__all__ = [
    "AsyncLLMEngine",
    "EngineConfig",
    "GenerateConfig",
    "KVCacheGroupSpec",
    "KVCacheSpec",
    "ModelLoader",
    "ParallelConfig",
    "ReplicaEngineCore",
    "RuntimeConfig",
]
