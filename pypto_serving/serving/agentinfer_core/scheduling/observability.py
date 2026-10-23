# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
# Modified by PyPTO Serving: internal imports use the bundled scheduling package.
"""Validated controls for opt-in Scheduler event and periodic diagnostic logging."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SchedulerObservabilityConfig:
    """Control Scheduler logging without changing policy decisions.

    Args:
        enabled: Whether event-driven and periodic Scheduler logs are emitted.
        log_interval_seconds: Minimum interval between periodic strategy diagnostics.
    """

    enabled: bool = False
    log_interval_seconds: float = 5.0

    def __post_init__(self) -> None:
        """Reject ambiguous booleans and unusable periodic intervals."""
        if not isinstance(self.enabled, bool):
            raise ValueError("observability.enabled must be a boolean")
        if not math.isfinite(self.log_interval_seconds) or self.log_interval_seconds <= 0:
            raise ValueError("observability.log_interval_seconds must be finite and positive")
