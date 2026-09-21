from dataclasses import replace

import pytest

from pypto_serving.model.deepseek_dspark.pd_adapter import (
    DSV4_DSPARK_K7_CONTRACT,
    DeepSeekV4DSparkK7Adapter,
)
from pypto_serving.serving.pd.adapter import (
    ModelPDAdapterRegistry,
    ModelRuntimeFacts,
)


def test_contract_digest_is_canonical_and_semantic() -> None:
    contract = DSV4_DSPARK_K7_CONTRACT
    assert contract.digest == contract.digest
    assert len(contract.digest) == 64
    assert replace(contract, version=contract.version + 1).digest != contract.digest
    assert contract.logical_groups == (
        "ori",
        "cmp_c128",
        "cmp_c4",
        "idx",
        "hca_state",
        "csa_state",
        "csa_inner_state",
    )
    assert len(contract.physical_regions) == 8


def test_registry_selects_exactly_one_adapter_from_runtime() -> None:
    adapter = DeepSeekV4DSparkK7Adapter()
    registry = ModelPDAdapterRegistry((adapter,))
    selected = registry.select(ModelRuntimeFacts("deepseek_v4", "dspark", 7))
    assert selected.contract is DSV4_DSPARK_K7_CONTRACT

    with pytest.raises(ValueError, match=r"matched=\(\)"):
        registry.select(ModelRuntimeFacts("deepseek_v4", "dspark", 0))
    with pytest.raises(ValueError, match="ids must be unique"):
        ModelPDAdapterRegistry((adapter, adapter))
