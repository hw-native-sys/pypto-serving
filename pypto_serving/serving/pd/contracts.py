# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
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
    executor_cls: str
    prefill_speculative_tokens: int
    decode_speculative_tokens: int
    supported_prefix_cache_modes: tuple[str, ...] = ("disabled",)
    prefill_async_scheduling: bool = False
    decode_async_scheduling: bool = False

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

    @property
    def prefix_cache_groups(self) -> tuple[str, ...]:
        return tuple(
            group for group in self.logical_groups if group not in self.final_only_groups
        )

    def local_speculative_tokens(self, role: str) -> int:
        if role == "prefill":
            return self.prefill_speculative_tokens
        if role == "decode":
            return self.decode_speculative_tokens
        raise ValueError(f"unknown PD role {role!r}")

    def async_scheduling_for_role(self, role: str) -> bool:
        """Return the scheduling mode required by one side of the PD contract."""
        if role == "prefill":
            return self.prefill_async_scheduling
        if role == "decode":
            return self.decode_async_scheduling
        raise ValueError(f"unknown PD role {role!r}")

    def supports_prefix_cache_mode(self, mode: str) -> bool:
        return mode in self.supported_prefix_cache_modes


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
    prefix_cache_mode: str

    @classmethod
    def from_runtime_bundle(
        cls,
        contract: ModelPDContract,
        bundle,
        *,
        provider: str,
        prefix_cache_mode: str,
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
            prefix_cache_mode=prefix_cache_mode,
        )
