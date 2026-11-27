# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Transfer permits and bounded exclusive maintenance on one local EP group."""

from threading import Condition
from time import monotonic

from .transfer import TransferError


class TransferGate:
    """Permits survive timeouts; only confirmed completion or stop retires them."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.condition = Condition()
        self.permits = set()
        self.maintenance = False
        self.failure = None

    def reserve(self, key):
        with self.condition:
            self.check_healthy()
            if key in self.permits:
                raise ValueError("duplicate transfer permit")
            if self.maintenance or len(self.permits) >= self.capacity:
                return False
            self.permits.add(key)
            return True

    def require(self, key):
        with self.condition:
            self.check_healthy()
            if key not in self.permits:
                raise ValueError("missing transfer permit")

    def release(self, key):
        with self.condition:
            self.permits.discard(key)
            self.condition.notify_all()

    def check_healthy(self):
        if self.failure is not None:
            raise TransferError("PD maintenance or native completion failed", uncertain=True) from self.failure

    def enter(self, timeout):
        deadline = monotonic() + timeout
        with self.condition:
            self.check_healthy()
            if self.maintenance:
                raise RuntimeError("concurrent maintenance on one executor")
            self.maintenance = True
            try:
                while self.permits:
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        self.maintenance = False
                        self.condition.notify_all()
                        return False
                    self.condition.wait(remaining)
                    self.check_healthy()
                return True
            except BaseException:
                self.maintenance = False
                self.condition.notify_all()
                raise

    def leave(self, error=None):
        with self.condition:
            if error is not None:
                self.failure = error
            self.maintenance = False
            self.condition.notify_all()
