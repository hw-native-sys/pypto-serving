# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

from .env import ProfileConfig, create_profile_config
from .recorder import (
    ProfileRecorder,
    configure_profiler,
    get_profiler,
    invocation_profile_args,
    is_enabled,
    merge_profile,
    profile_duration,
    profile_instant,
    profile_invocation,
    profile_span,
    start_profile,
    stop_profile,
)

__all__ = [
    "ProfileConfig",
    "ProfileRecorder",
    "configure_profiler",
    "create_profile_config",
    "get_profiler",
    "invocation_profile_args",
    "is_enabled",
    "merge_profile",
    "profile_duration",
    "profile_instant",
    "profile_invocation",
    "profile_span",
    "start_profile",
    "stop_profile",
]
