# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Backend-neutral transfer completion contract."""

from __future__ import annotations

from typing import Callable, Protocol

from .types import ProviderCapabilities, ProviderTransferTask


class TransferProvider(Protocol):
    capabilities: ProviderCapabilities

    def write(self, task: ProviderTransferTask, on_submitted: Callable[[], None] = lambda: None) -> None:
        """Return on successful completion, otherwise raise TransferFailure.

        NOT_SUBMITTED proves no native work was admitted. FAILED_DEFINITE additionally
        promises no future access to the task's memory (drained, not just an error code).
        Without that guarantee report UNKNOWN; callers must not reuse its allocations.
        """
        ...
