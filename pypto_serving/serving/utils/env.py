# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Environment-driven serving logging and worker timeouts."""

from __future__ import annotations

import logging
import os

_DEFAULT_WORKER_INIT_TIMEOUT_SECONDS = 1800.0
_DEFAULT_WORKER_STEP_TIMEOUT_SECONDS = 1200.0
# How long a shutting-down server waits for the requests it has already accepted
# before cancelling them. Matched to the router's drain timeout, which is the
# same bound one layer up: that long for in-flight work, then stop regardless.
_DEFAULT_GRACEFUL_SHUTDOWN_SECONDS = 300.0


def configure_runtime_logging() -> None:
    """Apply runtime log thresholds in both the CLI and spawned worker."""
    for name in ("simpler_setup", "pypto", "simpler"):
        logging.getLogger(name).setLevel(logging.WARNING)
    # Worker.init() snapshots this threshold for device-side diagnostics.
    level = os.environ.get("PYPTO_SIMPLER_LOG_LEVEL", "").strip().upper()
    if level:
        try:
            logging.getLogger("simpler").setLevel(level)
        except ValueError:
            logging.getLogger(__name__).warning(
                "Invalid PYPTO_SIMPLER_LOG_LEVEL=%r; using WARNING", level
            )


def _positive_env_timeout_seconds(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        timeout = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive number of seconds") from exc
    if timeout <= 0:
        raise ValueError(f"{name} must be a positive number of seconds")
    return timeout


def worker_init_timeout_seconds() -> float:
    return _positive_env_timeout_seconds("PYPTO_WORKER_INIT_TIMEOUT", _DEFAULT_WORKER_INIT_TIMEOUT_SECONDS)


def worker_step_timeout_seconds() -> float:
    return _positive_env_timeout_seconds("SERVING_WORKER_STEP_TIMEOUT", _DEFAULT_WORKER_STEP_TIMEOUT_SECONDS)


def graceful_shutdown_timeout_seconds() -> float:
    """How long uvicorn waits for in-flight requests before it cancels them.

    Unbounded by default, and the wait sits in front of everything that stops
    the worker: a request that never finishes means the worker is never asked
    to stop, never signalled, and keeps its devices for as long as the process
    lives.
    """
    return _positive_env_timeout_seconds(
        "PYPTO_GRACEFUL_SHUTDOWN_TIMEOUT", _DEFAULT_GRACEFUL_SHUTDOWN_SECONDS
    )
