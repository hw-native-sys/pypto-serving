# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
# Modified by PyPTO Serving: internal imports use the bundled scheduling package.
"""Mutable policy-owned values that provide the basis for scheduling decisions.

This module does not define the runtime's ``ProgramState`` lifecycle. ``StrategyFactors`` stores global measurements
and per-Program values used by a policy to compare admission, resume, pause, and release choices without adding them to
the shared Program model. Runtime facts remain in Program records and immutable snapshots; factors contain only the
policy's rolling statistics, counters, timestamps, weights, or flags derived from those facts.

The host runtime owns one factor collection and passes it directly to strategy Hooks while holding the same lock that
serializes core Program transitions. A strategy updates factors only after the corresponding transition is accepted, so
a separate copy-on-write view, commit protocol, version, or rollback wrapper would duplicate synchronization semantics
and could incorrectly discard earlier factor updates from a partially accepted fixed plan.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Generic, TypeVar

from pypto_serving.serving.agentinfer_core.scheduling.domain import ProgramRef

StrategyGlobalFactorsT = TypeVar("StrategyGlobalFactorsT")
StrategyProgramFactorsT = TypeVar("StrategyProgramFactorsT")


class StrategyFactors(Generic[StrategyGlobalFactorsT, StrategyProgramFactorsT]):
    """Policy-owned global and per-Program values used as scheduling decision factors.

    Args:
        global_factors: Initial policy-wide measurements and derived decision values.
        program_factors: Initial per-Program decision values keyed by exact Program generations.

    Notes:
        The class intentionally has no lock or version. Its owner must serialize every read and write with the Router
        scheduling lock, which also protects the immutable runtime snapshot and controlled core transitions supplied
        to the same Hook.
    """

    def __init__(
        self,
        global_factors: StrategyGlobalFactorsT,
        program_factors: Iterable[tuple[ProgramRef, StrategyProgramFactorsT]] = (),
    ) -> None:
        items = tuple(program_factors)
        refs = [ref for ref, _ in items]
        if len(refs) != len(set(refs)):
            raise ValueError("strategy factors contain duplicate program references")
        self._global_factors = global_factors
        self._program_factors = dict(items)

    @property
    def global_factors(self) -> StrategyGlobalFactorsT:
        """Return the current policy-wide decision factors."""
        return self._global_factors

    @property
    def program_factors(self) -> tuple[tuple[ProgramRef, StrategyProgramFactorsT], ...]:
        """Return per-Program decision factors as an immutable sequence for inspection."""
        return tuple(self._program_factors.items())

    def for_program(self, ref: ProgramRef) -> StrategyProgramFactorsT | None:
        """Return decision factors for an exact Program generation, if present.

        Args:
            ref: Generation-safe Program reference to look up.
        """
        return self._program_factors.get(ref)

    def set_global_factors(self, factors: StrategyGlobalFactorsT) -> None:
        """Replace policy-wide factors after an accepted scheduling effect.

        Args:
            factors: New policy-wide decision factors.
        """
        self._global_factors = factors

    def set_program_factors(self, ref: ProgramRef, factors: StrategyProgramFactorsT) -> None:
        """Replace one Program's factors after an accepted scheduling effect.

        Args:
            ref: Exact Program generation owning the factors.
            factors: New strategy-defined decision factors for the Program.
        """
        self._program_factors[ref] = factors

    def remove_program_factors(self, ref: ProgramRef) -> None:
        """Remove one Program's factors after its Program is released.

        Args:
            ref: Exact Program generation whose decision factors should be removed.
        """
        self._program_factors.pop(ref, None)
