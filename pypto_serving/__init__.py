# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Public API for PyPTO Serving.

The exports below are resolved lazily (PEP 562). Every one of them reaches torch
through its module, so importing them eagerly would make *any* submodule import
-- including ``pypto_serving.router``, which runs on hosts with no NPU stack --
require the full runtime. Attribute access and ``from pypto_serving import X``
behave exactly as before.
"""

from importlib import import_module
from typing import TYPE_CHECKING

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

_LAZY_EXPORTS = {
    "AsyncLLMEngine": "pypto_serving.serving.engine.async_engine",
    "EngineConfig": "pypto_serving.serving.engine.async_engine",
    "ReplicaEngineCore": "pypto_serving.serving.engine.async_engine",
    "GenerateConfig": "pypto_serving.config.types",
    "KVCacheGroupSpec": "pypto_serving.config.types",
    "KVCacheSpec": "pypto_serving.config.types",
    "RuntimeConfig": "pypto_serving.config.types",
    "ModelLoader": "pypto_serving.model.model_loader",
    "ParallelConfig": "pypto_serving.config.parallel",
}

assert set(__all__) == set(_LAZY_EXPORTS), "__all__ and the lazy export map must agree"

if TYPE_CHECKING:  # pragma: no cover - import-time cost is the whole point
    from pypto_serving.config.parallel import ParallelConfig
    from pypto_serving.config.types import (
        GenerateConfig,
        KVCacheGroupSpec,
        KVCacheSpec,
        RuntimeConfig,
    )
    from pypto_serving.model.model_loader import ModelLoader
    from pypto_serving.serving.engine.async_engine import (
        AsyncLLMEngine,
        EngineConfig,
        ReplicaEngineCore,
    )


def __getattr__(name: str) -> object:
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_path), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
