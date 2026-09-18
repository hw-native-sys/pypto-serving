# Copyright (c) PyPTO Contributors.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""Model-neutral PD contracts derived from the live Serving runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json


@dataclass(frozen=True)
class TransferComponent:
    component_id: str
    cache_group: str
    atomic_group: str
    final_only: bool = False


@dataclass(frozen=True)
class ModelPDContract:
    adapter_id: str
    version: int
    model_family: str
    model_variant: str
    transfer_granularity: str
    continuation_schema: str
    components: tuple[TransferComponent, ...]

    @property
    def digest(self) -> str:
        wire = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(wire).hexdigest()

    @property
    def logical_groups(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(component.cache_group for component in self.components))

    @property
    def physical_regions(self) -> tuple[str, ...]:
        return tuple(component.component_id for component in self.components)

    @property
    def group_components(self) -> dict[str, tuple[str, ...]]:
        return {
            group: tuple(
                component.component_id
                for component in self.components
                if component.cache_group == group
            )
            for group in self.logical_groups
        }

    @property
    def final_only_groups(self) -> frozenset[str]:
        return frozenset(
            component.cache_group for component in self.components if component.final_only
        )


@dataclass(frozen=True)
class RuntimeLayoutDescriptor:
    adapter_id: str
    contract_version: int
    contract_digest: str
    model_revision: str
    registry_fingerprint: str
    layout_fingerprint: str
    topology: tuple[int, ...]
    provider: str
    logical_groups: tuple[str, ...]
    physical_regions: tuple[str, ...]
    continuation_schema: str

    @classmethod
    def from_runtime_bundle(
        cls,
        contract: ModelPDContract,
        bundle,
        *,
        provider: str,
    ) -> "RuntimeLayoutDescriptor":
        return cls(
            adapter_id=contract.adapter_id,
            contract_version=contract.version,
            contract_digest=contract.digest,
            model_revision=bundle.model_revision,
            registry_fingerprint=bundle.registry_fingerprint,
            layout_fingerprint=bundle.layout_fingerprint,
            topology=bundle.topology,
            provider=provider,
            logical_groups=contract.logical_groups,
            physical_regions=contract.physical_regions,
            continuation_schema=contract.continuation_schema,
        )

