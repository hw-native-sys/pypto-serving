# Copyright (c) PyPTO Contributors.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""DSV4 DSpark K7 model-adapter contract tests."""

import asyncio

import pytest

from pypto_serving.config.types import GenerateConfig
from pypto_serving.model.deepseek_dspark.pd_adapter import (
    DSV4_DSPARK_K7_ADAPTER,
    DSV4_DSPARK_K7_CONTRACT,
)
from pypto_serving.serving.engine.async_engine import PrefillChunkReady, TokenOutput
from pypto_serving.serving.pd.protocol import (
    ContinuationMetadata,
    HandoffKey,
    continuation_metadata_hash,
)
from pypto_serving.serving.pd.worker_api import ComponentGeometry, WorkerRegistryBundle

from .helpers import make_cache_manager, make_rank_registrations, make_registry


KEY = HandoffKey("request", "handoff", 1, 1, 1)


def _tables(manager, request_id: str, token_count: int):
    return {
        name: tuple(ids)
        for name, ids in manager.ensure_group_blocks(request_id, token_count).items()
    }


def _bundle(registry) -> WorkerRegistryBundle:
    return WorkerRegistryBundle(
        model_revision=registry.model_revision,
        topology=registry.topology,
        registry_fingerprint=registry.fingerprint,
        layout_fingerprint=registry.layout_fingerprint,
        components=tuple(
            ComponentGeometry(
                component_id=component.component_id,
                dtype=component.dtype,
                item_bytes=component.item_bytes,
                layers=component.layers,
                blocks_per_layer=component.blocks_per_layer,
                block_tokens=component.block_tokens,
                token_stride_bytes=component.token_stride_bytes,
                extent=component.extent,
            )
            for component in registry.components
        ),
        ranks=make_rank_registrations(registry),
    )


def test_adapter_owns_eight_region_registry_and_transfer_policy() -> None:
    contract = DSV4_DSPARK_K7_CONTRACT
    assert contract.physical_regions == (
        "ori",
        "hca_cmp",
        "csa_cmp",
        "idx_k",
        "idx_scale",
        "hca_state",
        "csa_state",
        "csa_inner_state",
    )
    assert contract.final_only_groups == {
        "hca_state",
        "csa_state",
        "csa_inner_state",
    }

    manager = make_cache_manager()
    original = make_registry(manager)
    rebuilt = DSV4_DSPARK_K7_ADAPTER.build_registry(_bundle(original))
    assert rebuilt == original
    assert rebuilt.fingerprint == original.fingerprint
    assert rebuilt.layout_fingerprint == original.layout_fingerprint


def test_adapter_plans_closed_then_partial_final_pages_and_builds_manifest() -> None:
    manager = make_cache_manager()
    registry = make_registry(manager)
    planner = DSV4_DSPARK_K7_ADAPTER.make_planner(registry, manager.group_specs)
    source = _tables(manager, "source", 33)
    destination = _tables(manager, "destination", 64)

    first = planner.plan_chunk(
        KEY,
        chunk_id=0,
        start_token=0,
        end_token=32,
        final=False,
        rank_ids=(0,),
        source_blocks_by_rank={0: source},
        destination_blocks_by_rank={0: destination},
    )
    assert {copy.component_id for copy in first.ranks[0].copies} == {"ori"}
    assert {
        unit.component_id for unit in first.ranks[0].expected_units
    } == {"ori", "hca_cmp", "csa_cmp", "idx_k", "idx_scale"}
    assert any(unit.nbytes == 0 for unit in first.ranks[0].expected_units)

    final = planner.plan_chunk(
        KEY,
        chunk_id=1,
        start_token=32,
        end_token=33,
        final=True,
        rank_ids=(0,),
        source_blocks_by_rank={0: source},
        destination_blocks_by_rank={0: destination},
    )
    final_components = {copy.component_id for copy in final.ranks[0].copies}
    assert {"ori", "hca_state", "csa_state", "csa_inner_state"} <= final_components
    assert {
        unit.component_id for unit in final.ranks[0].expected_units
    } == set(DSV4_DSPARK_K7_CONTRACT.physical_regions)
    partial_ori = [
        copy for copy in final.ranks[0].copies if copy.component_id == "ori"
    ]
    assert partial_ori and {copy.valid_tokens for copy in partial_ori} == {1}

    continuation = DSV4_DSPARK_K7_ADAPTER.build_continuation(
        config=GenerateConfig(max_new_tokens=128, stream=True),
        prompt_token_ids=(1, 2, 3),
        eos_token_id=2,
    )
    chunk = PrefillChunkReady(
        request_id="request",
        chunk_id=1,
        start_token=32,
        end_token=33,
        final=True,
        first_token=101,
        block_ids_by_group=source,
        cache_partition=0,
    )
    manifest = DSV4_DSPARK_K7_ADAPTER.build_manifest(
        key=KEY,
        plan=final,
        chunk=chunk,
        continuation=continuation,
        prepared_digest="a" * 64,
    )
    assert manifest.manifest_hash == final.manifest_hash
    assert manifest.expected_units == final.expected_units
    assert manifest.copies_by_rank == final.copies_by_rank
    assert manifest.metadata_hash == continuation_metadata_hash(continuation)
    assert manifest.continuation == continuation


def test_adapter_validates_continuation_and_owns_k7_adoption() -> None:
    with pytest.raises(ValueError, match="greedy"):
        DSV4_DSPARK_K7_ADAPTER.validate_generate_config(
            GenerateConfig(temperature=0.5)
        )
    continuation = ContinuationMetadata(
        prompt_token_ids=(1, 2, 3),
        max_new_tokens=128,
        temperature=0.0,
        top_p=1.0,
        top_k=None,
        seed=7,
        stop_strings=("stop",),
        eos_token_id=2,
        stream=True,
    )
    DSV4_DSPARK_K7_ADAPTER.validate_continuation(continuation)

    captured = {}

    class Core:
        async def add_adopted_handoff(self, **kwargs):
            captured.update(kwargs)
            yield TokenOutput(token_id=101, text="ok")

    async def collect_outputs():
        return [
            output
            async for output in DSV4_DSPARK_K7_ADAPTER.adopt_decode(
                Core(),
                reservation_id="reservation",
                request_id="request",
                first_token=101,
                continuation=continuation,
            )
        ]

    outputs = asyncio.run(collect_outputs())
    assert [output.token_id for output in outputs] == [101]
    assert captured == {
        "reservation_id": "reservation",
        "request_id": "request",
        "prompt_token_ids": (1, 2, 3),
        "first_token": 101,
        "max_new_tokens": 128,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": None,
        "seed": 7,
        "stop_strings": ("stop",),
        "eos_token_id": 2,
        "stream": True,
    }
