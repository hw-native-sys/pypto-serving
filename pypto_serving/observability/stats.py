# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SchedulerStats:
    """Scheduler state and counter deltas produced by one scheduling pass."""

    num_running_reqs: int = 0
    num_waiting_reqs: int = 0
    kv_cache_usage: float = 0.0
    prefix_cache_queries: int = 0
    prefix_cache_hits: int = 0
    num_preemptions: int = 0


@dataclass(frozen=True)
class FinishedRequestStats:
    """Request-level values emitted exactly once when a request terminates."""

    finish_reason: str
    e2e_latency: float
    num_prompt_tokens: int
    num_generation_tokens: int
    time_to_first_token: float | None = None
    mean_time_per_output_token: float | None = None


@dataclass
class IterationStats:
    """Counter and latency deltas confirmed by one worker result."""

    num_prefill_tokens: int = 0
    num_generation_tokens: int = 0
    time_to_first_tokens: list[float] = field(default_factory=list)
    inter_token_latencies: list[float] = field(default_factory=list)
    finished_requests: list[FinishedRequestStats] = field(default_factory=list)
