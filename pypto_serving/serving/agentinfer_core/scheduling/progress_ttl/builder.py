# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
# Modified by PyPTO Serving: internal imports use the bundled scheduling package.
"""Explicit Progress-TTL construction without a multi-policy registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pypto_serving.serving.agentinfer_core.scheduling.factors import StrategyFactors
from pypto_serving.serving.agentinfer_core.scheduling.progress_ttl.config import ProgressTTLConfig
from pypto_serving.serving.agentinfer_core.scheduling.progress_ttl.factors import ProgressTTLProgramFactors
from pypto_serving.serving.agentinfer_core.scheduling.progress_ttl.rolling_stats import ProgressTTLGlobalFactors
from pypto_serving.serving.agentinfer_core.scheduling.progress_ttl.strategy import ProgressTTLStrategy


@dataclass(frozen=True)
class ProgressTTLComponents:
    """Concrete Progress-TTL strategy and its empty initial decision factors."""

    strategy: ProgressTTLStrategy
    initial_factors: StrategyFactors[ProgressTTLGlobalFactors, ProgressTTLProgramFactors]


def build_progress_ttl_strategy(
    config: ProgressTTLConfig | None = None, *, capacity_cost: Callable[[int], int] | None = None,
) -> ProgressTTLComponents:
    """Construct Progress-TTL and its initial decision factors from immutable configuration."""
    resolved = config or ProgressTTLConfig()
    return ProgressTTLComponents(
        strategy=ProgressTTLStrategy(resolved, capacity_cost=capacity_cost),
        initial_factors=StrategyFactors(global_factors=ProgressTTLGlobalFactors()),
    )
