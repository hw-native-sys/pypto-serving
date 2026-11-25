# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

from .env import ProfileConfig, create_profile_config
from .recorder import (
    ProfileRecorder,
    configure_profiler,
    get_profiler,
    is_enabled,
    merge_profile,
    profile_duration,
    profile_instant,
    profile_span,
    start_profile,
    stop_profile,
)

if TYPE_CHECKING:
    from pypto_serving.config.types import DispatchInvocation

_dispatch_invocation: ContextVar[DispatchInvocation | None] = ContextVar("dispatch_invocation", default=None)


@contextmanager
def dispatch_invocation_scope(invocation: DispatchInvocation | None) -> Iterator[None]:
    """Bind the serving step to profile calls in this execution context."""
    token = _dispatch_invocation.set(invocation)
    try:
        yield
    finally:
        _dispatch_invocation.reset(token)


def get_dispatch_invocation() -> DispatchInvocation | None:
    return _dispatch_invocation.get()


def invocation_profile_args() -> dict[str, object]:
    invocation = get_dispatch_invocation()
    return invocation.as_profile_args() if invocation is not None else {}


__all__ = [
    "ProfileConfig",
    "ProfileRecorder",
    "configure_profiler",
    "create_profile_config",
    "dispatch_invocation_scope",
    "get_dispatch_invocation",
    "get_profiler",
    "invocation_profile_args",
    "is_enabled",
    "merge_profile",
    "profile_duration",
    "profile_instant",
    "profile_span",
    "start_profile",
    "stop_profile",
]
