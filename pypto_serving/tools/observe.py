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
def span(logger, name, *, cat="pd", thread_cpu=False, log=True, **fields):
    """Return per-call timing; thread CPU is only meaningful for synchronous phases.

    With log=False, detail goes only to the existing on-demand profiler.
    No samples or counters are retained by this context manager.
    """
    started = time.perf_counter_ns()
    cpu_started = time.thread_time_ns() if thread_cpu else None
    timing = {}
    outcome = "completed"
    try:
        yield timing
    except BaseException:
        outcome = "error"
        raise
    finally:
        duration = (time.perf_counter_ns() - started) / 1000
        timing["duration_ms"] = duration / 1000
        cpu = {} if cpu_started is None else {"thread_cpu_ms": (time.thread_time_ns() - cpu_started) / 1e6}
        timing.update(cpu)
        if log:
            record(logger, name, **fields, **timing, outcome=outcome)
        try:
            profile_duration(name, cat=cat, dur_us=duration, ts_us=started / 1000,
                             args={**fields, **cpu, "outcome": outcome})
        except Exception:
            pass
