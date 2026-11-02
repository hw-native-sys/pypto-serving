# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Best-effort fields for the existing logger and profiler; no execution state."""

from contextlib import contextmanager
import time

from .profile import profile_duration


def record(logger, name, **fields):
    try:
        logger.info("%s %s", name, " ".join(f"{key}={value}" for key, value in fields.items()))
    except Exception:
        pass  # A diagnostic sink cannot change execution or retirement.


@contextmanager
def span(logger, name, *, cat="pd", **fields):
    started = time.perf_counter_ns()
    outcome = "completed"
    try:
        yield
    except BaseException:
        outcome = "error"
        raise
    finally:
        duration = (time.perf_counter_ns() - started) / 1000
        record(logger, name, **fields, outcome=outcome, duration_ms=duration / 1000)
        try:
            profile_duration(name, cat=cat, dur_us=duration, ts_us=started / 1000,
                             args={**fields, "outcome": outcome})
        except Exception:
            pass
