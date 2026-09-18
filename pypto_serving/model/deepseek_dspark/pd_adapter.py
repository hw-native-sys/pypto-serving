# Copyright (c) PyPTO Contributors.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""DeepSeek V4 DSpark K7 semantic PD contract."""

from pypto_serving.serving.pd.adapter import ModelRuntimeFacts
from pypto_serving.serving.pd.contracts import ModelPDContract, TransferComponent


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
)


class DeepSeekV4DSparkK7Adapter:
    contract = DSV4_DSPARK_K7_CONTRACT

    def matches(self, facts: ModelRuntimeFacts) -> bool:
        return (
            facts.model_family == self.contract.model_family
            and facts.model_variant == self.contract.model_variant
            and facts.num_speculative_tokens == 7
        )


BUILTIN_PD_ADAPTERS = (DeepSeekV4DSparkK7Adapter(),)

