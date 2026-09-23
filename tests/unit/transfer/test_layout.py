# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Layout golden vectors use the real DSV4 token widths without device allocation."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from pypto_serving.model.deepseek.transfer_layout import (
    COMPONENTS, BlockCopy, ComponentLayout, DSV4Registry,
)
from pypto_serving.transfer.types import OwnerRef, RegionLease, TransferAttemptRef


@pytest.fixture
def layout_case():
    entries = tuple(ComponentLayout(name, "float32", 4, (0, 2), 3, 64,
                                    4 if name == "idx_scale" else 512)
                    for name in COMPONENTS)
    registry = DSV4Registry("model", (8,), entries)
    src, dst = OwnerRef("run", 0, 1, 1, "P"), OwnerRef("run", 0, 2, 2, "D")
    attempt = TransferAttemptRef("request", "plan", "handoff", "a", 1, 1, 0, src, dst, "manifest")
    kwargs = dict(destination_layout_fingerprint=registry.layout_fingerprint,
                  source_leases={c.component_id: RegionLease(src, c.component_id, 1, c.extent) for c in entries},
                  destination_leases={c.component_id: RegionLease(dst, c.component_id, 1, c.extent) for c in entries})
    return registry, attempt, kwargs


def test_all_physical_components_and_partial_scale_page(layout_case):
    registry, attempt, kwargs = layout_case
    copies = tuple(BlockCopy(name, 2, 1, 2, 1) for name in COMPONENTS)
    attempt = replace(attempt, component_manifest_hash=registry.manifest_hash(copies, True))
    task = registry.lower(attempt, copies, final=True, **kwargs)
    assert len(task.segments) == len(COMPONENTS)
    scale = next(s for s in task.segments if s.component_id == "idx_scale")
    assert (scale.source_offset, scale.destination_offset, scale.length) == (1024, 1280, 256)
    assert registry.lower(attempt, tuple(reversed(copies)), final=True, **kwargs) == task


def test_consecutive_pages_coalesce(layout_case):
    registry, attempt, kwargs = layout_case
    copies = tuple(BlockCopy("ori", 0, block, block, 64) for block in range(3))
    attempt = replace(attempt, component_manifest_hash=registry.manifest_hash(copies, False))
    task = registry.lower(attempt, copies, final=False, **kwargs)
    assert len(task.segments) == 1
    assert task.segments[0].length == 3 * 64 * 512


@pytest.mark.parametrize("copies,final", [
    ((BlockCopy("idx_k", 0, 0, 0, 64),), True),
    ((BlockCopy("ori", 0, 0, 0, 1),), False),
    ((BlockCopy("hca_state", 0, 0, 0, 64),), False),
    ((BlockCopy("ori", 1, 0, 0, 64),), True),
    ((BlockCopy("ori", 0, 3, 0, 64),), True),
    ((BlockCopy("ori", 0, 0, 0, 65),), True),
    ((BlockCopy("ori", 0, 0, 0, 64), BlockCopy("ori", 0, 1, 0, 64)), True),
])
def test_invalid_manifest(layout_case, copies, final):
    registry, attempt, kwargs = layout_case
    with pytest.raises(ValueError):
        registry.lower(attempt, copies, final=final, **kwargs)


def test_layout_fingerprint_and_source_extent_must_match(layout_case):
    registry, attempt, kwargs = layout_case
    copies = (BlockCopy("ori", 0, 0, 0, 64),)
    with pytest.raises(ValueError, match="fingerprint"):
        registry.lower(attempt, copies, final=True,
                       **dict(kwargs, destination_layout_fingerprint="wrong"))
    kwargs["source_leases"]["ori"] = replace(kwargs["source_leases"]["ori"], extent=64)
    with pytest.raises(ValueError, match="extent"):
        registry.lower(attempt, copies, final=True, **kwargs)
    assert replace(registry, model_revision="other").fingerprint != registry.fingerprint
    assert replace(registry, components=tuple(reversed(registry.components))).fingerprint == registry.fingerprint


def test_compatible_destination_can_have_a_smaller_arena(layout_case):
    registry, attempt, kwargs = layout_case
    destination_components = tuple(
        replace(component, blocks_per_layer=2)
        for component in registry.components
    )
    destination_registry = replace(registry, components=destination_components)
    assert destination_registry.fingerprint != registry.fingerprint
    assert destination_registry.layout_fingerprint == registry.layout_fingerprint
    kwargs["destination_leases"] = {
        component.component_id: replace(
            kwargs["destination_leases"][component.component_id],
            extent=component.extent,
        )
        for component in destination_components
    }
    # Use the second mapped layer so the test proves that source and
    # destination layer bases are derived from their own arena capacities.
    copies = (BlockCopy("ori", 2, 2, 1, 64),)
    attempt = replace(
        attempt,
        component_manifest_hash=registry.manifest_hash(copies, True),
    )
    task = registry.lower(attempt, copies, final=True, **kwargs)
    assert task.segments[0].source_offset == (
        registry.entry("ori").layer_stride_bytes
        + 2 * registry.entry("ori").block_stride_bytes
    )
    assert task.segments[0].destination_offset == (
        2 * destination_registry.entry("ori").block_stride_bytes
        + destination_registry.entry("ori").block_stride_bytes
    )

    out_of_bounds = (replace(copies[0], destination_block=2),)
    attempt = replace(
        attempt,
        component_manifest_hash=registry.manifest_hash(out_of_bounds, True),
    )
    with pytest.raises(ValueError, match="out of bounds"):
        registry.lower(attempt, out_of_bounds, final=True, **kwargs)


def test_device_cache_view_does_not_require_pointer_access():
    shard = SimpleNamespace(shape=(6, 64, 1, 128), dtype="float32", nbytes=6 * 64 * 128 * 4)
    cache = SimpleNamespace(**{v[0]: SimpleNamespace(shards=(shard,)) for v in COMPONENTS.values()})
    registry = DSV4Registry.from_device_cache(cache, rank=0, model_revision="model", topology=(1,),
                                             layer_mapping={name: (0, 2) for name in COMPONENTS})
    assert registry.entry("ori").blocks_per_layer == 3


def test_model_adapter_exports_split_compressed_cache_pools():
    pytest.importorskip("torch")
    from pypto_serving.model.deepseek_dspark.pd_adapter import DSparkPDWorker

    plans = (
        SimpleNamespace(layer_id=0, compress_ratio=0),
        SimpleNamespace(layer_id=1, compress_ratio=128),
        SimpleNamespace(layer_id=2, compress_ratio=4),
    )
    layer_counts = {
        "ori": 3,
        "hca_cmp": 1,
        "csa_cmp": 1,
        "idx_k": 1,
        "idx_scale": 1,
        "hca_state": 1,
        "csa_state": 1,
        "csa_inner_state": 1,
    }
    fields = {}
    for component, (field, _, _) in COMPONENTS.items():
        shape = (layer_counts[component] * 3, 64, 1, 128)
        shard = SimpleNamespace(
            shape=shape,
            dtype="float32",
            nbytes=shape[0] * shape[1] * shape[2] * shape[3] * 4,
        )
        fields[field] = SimpleNamespace(shards=(shard,))
    adapter = DSparkPDWorker(
        config=object(),
        compiled=SimpleNamespace(
            layer_plan=plans, layout=SimpleNamespace(ranks=1, tp_size=1),
            num_speculative_tokens=0,
        ),
        worker=lambda: None, cache=lambda: SimpleNamespace(**fields),
        draft_states={}, reserve_state=None, initialize_device_state=None,
        metrics=lambda: {},
    )
    registry = adapter.registry(0, model_revision="current-serving")

    assert set(component.component_id for component in registry.components) == set(COMPONENTS)
    assert registry.entry("hca_cmp").layers == (1,)
    assert registry.entry("csa_cmp").layers == (2,)
    assert registry.entry("ori").layers == (0, 1, 2)
