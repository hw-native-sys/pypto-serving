# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""External prefix-cache contracts shared by the scheduler and workers."""

from pypto_serving.serving.external_cache.config import (
    ExternalPrefixCacheConfig,
    MooncakeClientConfig,
)
from pypto_serving.serving.external_cache.manifest import (
    DEEPSEEK_V4_CACHE_GROUPS,
    ExternalCacheNamespace,
    ExternalKVCheckpointManifest,
    ExternalKVObjectSpec,
    build_deepseek_checkpoint_manifest,
    checkpoint_alignment,
    checkpoint_manifest_key,
    checkpoint_prefix_digest,
    latest_checkpoint_token_count,
    stable_token_digest,
)
from pypto_serving.serving.external_cache.connector import (
    ExternalKVLoadRequest,
    ExternalKVLookupResult,
    ExternalKVPageAssignment,
    ExternalKVSaveRequest,
    ExternalKVTransferCompletion,
    ExternalKVWorkerConnector,
    ExternalPrefixCacheIndex,
)
from pypto_serving.serving.external_cache.protocol import (
    ExternalKVBackend,
    ExternalKVBuffer,
    ExternalKVTransfer,
)

__all__ = [
    "DEEPSEEK_V4_CACHE_GROUPS",
    "ExternalCacheNamespace",
    "ExternalKVBackend",
    "ExternalKVBuffer",
    "ExternalKVCheckpointManifest",
    "ExternalKVLoadRequest",
    "ExternalKVLookupResult",
    "ExternalKVObjectSpec",
    "ExternalKVPageAssignment",
    "ExternalKVSaveRequest",
    "ExternalKVTransfer",
    "ExternalKVTransferCompletion",
    "ExternalKVWorkerConnector",
    "ExternalPrefixCacheConfig",
    "ExternalPrefixCacheIndex",
    "MooncakeClientConfig",
    "build_deepseek_checkpoint_manifest",
    "checkpoint_alignment",
    "checkpoint_manifest_key",
    "checkpoint_prefix_digest",
    "latest_checkpoint_token_count",
    "stable_token_digest",
]
