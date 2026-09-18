# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from pypto_serving.model.deepseek.transfer_layout import ComponentLayout, DSV4Registry
from pypto_serving.model.deepseek_dspark.pd_adapter import DSV4_DSPARK_K7_CONTRACT
from pypto_serving.model.deepseek_dspark.npu_runner import build_dspark_cache_group_specs
from pypto_serving.serving.memory.kv_cache import KvCacheManager
from pypto_serving.serving.pd.config import PDCapabilities
from pypto_serving.serving.pd.protocol import (
    RankRegistration,
    RegionRegistration,
)


def make_cache_manager(*, capacity_slots: int = 2) -> KvCacheManager:
    ratios = tuple(4 if layer % 2 == 0 else 128 for layer in range(43))
    specs = build_dspark_cache_group_specs(43, ratios)
    manager = KvCacheManager(enable_prefix_cache=False)
    manager.init_groups(
        specs,
        max_batch_size=capacity_slots,
        primary_num_blocks=specs[0].max_blocks_per_seq * capacity_slots,
    )
    return manager


def make_registry(manager: KvCacheManager) -> DSV4Registry:
    group_by_component = {
        component: group
        for group, components in DSV4_DSPARK_K7_CONTRACT.group_components.items()
        for component in components
    }
    specs = {spec.name: spec for spec in manager.group_specs}
    components = []
    for component_id in (
        "ori",
        "hca_cmp",
        "csa_cmp",
        "idx_k",
        "idx_scale",
        "hca_state",
        "csa_state",
        "csa_inner_state",
    ):
        group = group_by_component[component_id]
        spec = specs[group]
        components.append(
            ComponentLayout(
                component_id=component_id,
                dtype="torch.uint8",
                item_bytes=1,
                layers=spec.layer_indices,
                blocks_per_layer=manager.group_num_blocks(group) + 64,
                block_tokens=spec.spec.storage_block_size,
                token_stride_bytes=64,
            )
        )
    return DSV4Registry("ds-v4-test", (16, 4), tuple(components))


def make_capabilities(registry: DSV4Registry) -> PDCapabilities:
    return PDCapabilities(
        adapter_id=DSV4_DSPARK_K7_CONTRACT.adapter_id,
        contract_version=DSV4_DSPARK_K7_CONTRACT.version,
        contract_digest=DSV4_DSPARK_K7_CONTRACT.digest,
        continuation_schema=DSV4_DSPARK_K7_CONTRACT.continuation_schema,
        model_revision=registry.model_revision,
        registry_fingerprint=registry.fingerprint,
        layout_fingerprint=registry.layout_fingerprint,
        topology=registry.topology,
        logical_groups=DSV4_DSPARK_K7_CONTRACT.logical_groups,
        physical_regions=DSV4_DSPARK_K7_CONTRACT.physical_regions,
    )


def make_rank_registrations(registry: DSV4Registry) -> tuple[RankRegistration, ...]:
    return tuple(
        RankRegistration(
            rank_id=rank_id,
            owner_generation=1,
            endpoint_generation=1,
            worker_id=f"worker-{rank_id}",
            regions=tuple(
                RegionRegistration(
                    component_id=component.component_id,
                    lease=1,
                    extent=component.extent,
                    provider_envelope=b"{}",
                )
                for component in registry.components
            ),
        )
        for rank_id in range(registry.topology[0])
    )
