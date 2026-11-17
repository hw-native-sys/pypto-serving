# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for V4.1 routed-weight staging limits."""

from concurrent.futures import ThreadPoolExecutor
import threading
import time

import pytest

from pypto_serving.model.deepseek_v41.weight_loader import _ByteBudget


def test_routed_projection_budget_bounds_in_flight_bytes():
    budget = _ByteBudget(10)
    active = 0
    peak = 0
    lock = threading.Lock()

    def task():
        nonlocal active, peak
        budget.acquire(6)
        try:
            with lock:
                active += 6
                peak = max(peak, active)
            time.sleep(0.01)
            with lock:
                active -= 6
        finally:
            budget.release(6)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda _: task(), range(2)))
    assert peak == 6

    with pytest.raises(ValueError, match="requires 11 bytes"):
        budget.acquire(11)
