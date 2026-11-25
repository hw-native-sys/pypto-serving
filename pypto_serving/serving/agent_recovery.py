# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Plan recovery against a single replica's KV budget."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass


def recovery_order(
    *,
    forced: bool,
    privileged: bool,
    reasoning: bool,
    last_finished: float | None,
    wait_started: float | None,
    first_come_first_served: bool,
) -> tuple[int, float]:
    """Return the class and time key for a session waiting to run."""
    if forced:
        priority = 0
    elif privileged:
        priority = 1
    elif not reasoning:
        priority = 4
    else:
        priority = 2 if last_finished is not None else 3
    if first_come_first_served:
        return priority, wait_started if wait_started is not None else math.inf
    return priority, -last_finished if last_finished is not None else 0.0


@dataclass(frozen=True)
class RecoveryBudget:
    reserved_tokens: int
    relaxed_after: float | None
    relaxed_growth_fraction: float
    relaxed_headroom: int
    keep_cached_blocks: bool
    uncached_tokens: int | None

    def allows_relaxed(
        self,
        *,
        waited: float | None,
        needed: int,
        growth: int,
        remaining: int,
        acting_tokens: Callable[[], int],
    ) -> bool:
        if self.relaxed_after is None or waited is None or waited < self.relaxed_after:
            return False
        physical_headroom = remaining + self.reserved_tokens
        minimum_growth = math.ceil(growth * self.relaxed_growth_fraction)
        if physical_headroom < needed + minimum_growth + self.relaxed_headroom:
            return False
        if self.keep_cached_blocks:
            return self.uncached_tokens is not None and self.uncached_tokens - acting_tokens() >= needed
        return True
