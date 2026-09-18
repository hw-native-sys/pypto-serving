# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Backend protocol and a deterministic CPU-only provider."""

from __future__ import annotations

import threading
from typing import Callable, Protocol

from .errors import ErrorCode, TransferError, TransferFailure
from .types import CompletionCertainty, ProviderCapabilities, ProviderTransferTask


class TransferProvider(Protocol):
    capabilities: ProviderCapabilities

    def write(self, task: ProviderTransferTask, on_submitted: Callable[[], None] = lambda: None) -> None:
        """Return on successful completion, otherwise raise TransferFailure.

        NOT_SUBMITTED proves no native work was admitted. FAILED_DEFINITE additionally
        promises no future access to the task's memory (drained, not just an error code).
        Without that guarantee report UNKNOWN; callers must not reuse its allocations.
        """
        ...


class FakeTransferProvider:
    """An explicit event gate models blocked native work without sleeps."""

    capabilities = ProviderCapabilities()

    def __init__(self, outcome: str = "success"):
        if outcome not in ("success", "failure", "block", "lost_completion"):
            raise ValueError("unsupported fake outcome")
        self.outcome = outcome
        self.started = threading.Event()
        self.unblock = threading.Event()
        self.calls = 0

    def write(self, task: ProviderTransferTask, on_submitted: Callable[[], None] = lambda: None) -> None:
        task.validate(self.capabilities)
        on_submitted()
        self.calls += 1
        self.started.set()
        if self.outcome in ("block", "lost_completion"):
            self.unblock.wait()
        if self.outcome in ("failure", "lost_completion"):
            raise TransferFailure(TransferError(ErrorCode.BACKEND_FAILURE, CompletionCertainty.UNKNOWN))
