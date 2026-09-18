# Copyright (c) PyPTO Contributors.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""DeepSeek V4 DSpark K7 PD adapter."""

from pypto_serving.model.deepseek.transfer_layout import ComponentLayout, DSV4Registry
from pypto_serving.serving.pd.adapter import ModelRuntimeFacts
from pypto_serving.serving.pd.connector import DecodeConnector
from pypto_serving.serving.pd.contracts import ModelPDContract, TransferComponent
from pypto_serving.serving.pd.planner import ChunkTransferPlanner


DSV4_DSPARK_K7_CONTRACT = ModelPDContract(
    adapter_id="deepseek-v4-dspark-k7",
    version=1,
    model_family="deepseek_v4",
    model_variant="dspark",
    transfer_granularity="chunk-after-prefill",
    continuation_schema="deepseek-v4-dspark-k7/v1",
    components=(
        TransferComponent("ori", "ori", "ori"),
        TransferComponent("hca_cmp", "cmp_c128", "hca_cmp"),
        TransferComponent("csa_cmp", "cmp_c4", "csa_cmp"),
        TransferComponent("idx_k", "idx", "index"),
        TransferComponent("idx_scale", "idx", "index"),
        TransferComponent("hca_state", "hca_state", "hca_state", True),
        TransferComponent("csa_state", "csa_state", "csa_state", True),
        TransferComponent(
            "csa_inner_state", "csa_inner_state", "csa_inner_state", True
        ),
    ),
    executor_cls="PyptoDeepSeekV4DSparkExecutor",
    prefill_speculative_tokens=0,
    decode_speculative_tokens=7,
)


class DeepSeekV4DSparkK7Adapter:
    contract = DSV4_DSPARK_K7_CONTRACT

    def matches(self, facts: ModelRuntimeFacts) -> bool:
        return (
            facts.model_family == self.contract.model_family
            and facts.model_variant == self.contract.model_variant
            and facts.num_speculative_tokens == 7
        )

    def build_registry(self, bundle) -> DSV4Registry:
        return DSV4Registry(
            model_revision=bundle.model_revision,
            topology=bundle.topology,
            components=tuple(
                ComponentLayout(
                    component_id=component.component_id,
                    dtype=component.dtype,
                    item_bytes=component.item_bytes,
                    layers=component.layers,
                    blocks_per_layer=component.blocks_per_layer,
                    block_tokens=component.block_tokens,
                    token_stride_bytes=component.token_stride_bytes,
                )
                for component in bundle.components
            ),
        )

    def make_planner(self, registry, group_specs) -> ChunkTransferPlanner:
        return ChunkTransferPlanner(registry, group_specs, self.contract)

    def make_decode_connector(
        self, cache_manager, capabilities, registry, destination_ranks
    ) -> DecodeConnector:
        return DecodeConnector(
            cache_manager,
            capabilities,
            registry,
            destination_ranks,
            contract=self.contract,
        )


DSV4_DSPARK_K7_ADAPTER = DeepSeekV4DSparkK7Adapter()
BUILTIN_PD_ADAPTERS = (DSV4_DSPARK_K7_ADAPTER,)
