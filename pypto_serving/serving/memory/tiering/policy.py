# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Retention order only; allocation and transfer references belong to the manager."""

from collections import OrderedDict
from collections.abc import Iterable
from typing import Hashable, Protocol


class RetentionPolicy(Protocol):
    def on_publish(self, key: Hashable, now: float) -> None: ...
    def on_access(self, key: Hashable, now: float) -> None: ...
    def on_remove(self, key: Hashable) -> None: ...
    def candidates(self, now: float) -> Iterable[Hashable]: ...


class LRU:
    def __init__(self):
        self._order = OrderedDict()

    def on_publish(self, key, now):
        self._order[key] = None
        self._order.move_to_end(key)

    def on_access(self, key, now):
        if key in self._order:
            self._order.move_to_end(key)

    def on_remove(self, key):
        self._order.pop(key, None)

    def candidates(self, now):
        return iter(self._order)
