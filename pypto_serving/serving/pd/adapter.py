# Copyright (c) PyPTO Contributors.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""Automatic model-adapter selection from Serving runtime facts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .contracts import ModelPDContract


@dataclass(frozen=True)
class ModelRuntimeFacts:
    model_family: str
    model_variant: str
    num_speculative_tokens: int


class ModelPDAdapter(Protocol):
    contract: ModelPDContract

    def matches(self, facts: ModelRuntimeFacts) -> bool: ...

    def build_registry(self, bundle): ...

    def make_planner(self, registry, group_specs): ...

    def make_decode_connector(
        self, cache_manager, capabilities, registry, destination_ranks
    ): ...


class ModelPDAdapterRegistry:
    def __init__(self, adapters: tuple[ModelPDAdapter, ...]) -> None:
        if not adapters:
            raise ValueError("PD adapter registry must not be empty")
        ids = tuple(adapter.contract.adapter_id for adapter in adapters)
        if len(ids) != len(set(ids)):
            raise ValueError("PD adapter ids must be unique")
        self._adapters = adapters

    def select(self, facts: ModelRuntimeFacts) -> ModelPDAdapter:
        matches = tuple(adapter for adapter in self._adapters if adapter.matches(facts))
        if len(matches) != 1:
            raise ValueError(
                "Serving runtime must match exactly one PD model adapter; "
                f"matched={tuple(adapter.contract.adapter_id for adapter in matches)}"
            )
        return matches[0]

    def get(self, adapter_id: str) -> ModelPDAdapter:
        matches = tuple(
            adapter for adapter in self._adapters if adapter.contract.adapter_id == adapter_id
        )
        if len(matches) != 1:
            raise ValueError(f"unknown PD model adapter {adapter_id!r}")
        return matches[0]
