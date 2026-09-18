# Copyright (c) PyPTO Contributors.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""F5 multi-node directory eligibility tests."""

import asyncio
import json

import pytest

from pypto_serving.model.deepseek_dspark.pd_adapter import DSV4_DSPARK_K7_CONTRACT
from pypto_serving.router.config import RouterConfig
from pypto_serving.router.directory import WorkerDirectory
from pypto_serving.serving.pd.config import PDCapabilities, PDRole, load_pd_document
from pypto_serving.serving.pd.http_api import CapacitySnapshot, NodeDescriptor
from pypto_serving.serving.pd.protocol import CapabilityWire


def _descriptor(
    node_id: str,
    role: PDRole,
    *,
    layout: str = "c",
    health: str = "READY",
) -> NodeDescriptor:
    contract = DSV4_DSPARK_K7_CONTRACT
    capabilities = PDCapabilities(
        adapter_id=contract.adapter_id,
        contract_version=contract.version,
        contract_digest=contract.digest,
        continuation_schema=contract.continuation_schema,
        model_revision="model",
        registry_fingerprint=node_id[0] * 64,
        layout_fingerprint=layout * 64,
        topology=(16, 4),
        logical_groups=contract.logical_groups,
        physical_regions=contract.physical_regions,
    )
    return NodeDescriptor(
        node_id=node_id,
        role=role.value,
        run_id="run",
        control_host=f"{node_id}.control",
        control_port=29831,
        owner_generation=1,
        endpoint_generation=1,
        control_incarnation=1,
        capabilities=CapabilityWire.from_capabilities(capabilities),
        health=health,
    )


def _capacity(
    node_id: str,
    role: PDRole,
    *,
    active: int = 0,
    limit: int = 4,
) -> CapacitySnapshot:
    return CapacitySnapshot(
        node_id=node_id,
        role=role.value,
        active_handoffs=active,
        prepared_requests=0,
        reservations=active,
        quarantined_reservations=0,
        snapshot_sequence=1,
        active_limit=limit,
    )


class _Client:
    def __init__(self, descriptor, capacity) -> None:
        self.descriptor = descriptor
        self.capacity = capacity

    async def get(self, path, _type):
        if path == "/internal/pd/descriptor":
            return self.descriptor
        if path == "/internal/pd/capacity":
            return self.capacity
        raise AssertionError(path)


def test_directory_filters_full_and_incompatible_nodes_without_losing_pool() -> None:
    async def exercise() -> None:
        p1 = _Client(
            _descriptor("p1", PDRole.PREFILL),
            _capacity("p1", PDRole.PREFILL),
        )
        p2 = _Client(
            _descriptor("p2", PDRole.PREFILL),
            _capacity("p2", PDRole.PREFILL),
        )
        d_full = _Client(
            _descriptor("d1", PDRole.DECODE),
            _capacity("d1", PDRole.DECODE, active=4),
        )
        d_wrong_layout = _Client(
            _descriptor("d2", PDRole.DECODE, layout="e"),
            _capacity("d2", PDRole.DECODE),
        )
        d_ready = _Client(
            _descriptor("d3", PDRole.DECODE),
            _capacity("d3", PDRole.DECODE),
        )
        directory = WorkerDirectory(
            (p1, p2),
            (d_full, d_wrong_layout, d_ready),
            "run",
            control_incarnation=1,
        )
        snapshot = await directory.refresh()
        assert [item.descriptor.node_id for item in snapshot.prefill] == ["p1", "p2"]
        assert [item.descriptor.node_id for item in snapshot.decode] == ["d2", "d3"]
        assert snapshot.rejected == ("decode[0]:RuntimeError",)
        assert {
            (p.descriptor.node_id, d.descriptor.node_id)
            for p, d in directory.compatible_pairs(snapshot)
        } == {("p1", "d3"), ("p2", "d3")}
        selected = await directory.select()
        assert (selected.prefill.node_id, selected.decode.node_id) == ("p1", "d3")

    asyncio.run(exercise())


def test_router_config_accepts_multiple_unique_endpoints(tmp_path) -> None:
    path = tmp_path / "pd.json"
    path.write_text(
        json.dumps(
            {
                "runtime": {
                    "run_id": "run",
                    "prefill": [
                        {"host": "10.0.0.1", "port": 8001, "node_id": "p1"},
                        {"host": "10.0.0.2", "port": 8001, "node_id": "p2"},
                    ],
                    "decode": [
                        {"host": "10.0.1.1", "port": 8002, "node_id": "d1"},
                        {"host": "10.0.1.2", "port": 8002, "node_id": "d2"},
                    ],
                },
                "observability": {"root": str(tmp_path / "logs")},
            }
        ),
        encoding="utf-8",
    )
    config = RouterConfig.from_document(load_pd_document(path))
    assert config.prefill_urls == ("http://10.0.0.1:8001", "http://10.0.0.2:8001")
    assert config.decode_urls == ("http://10.0.1.1:8002", "http://10.0.1.2:8002")


def test_runtime_rejects_duplicate_endpoints(tmp_path) -> None:
    path = tmp_path / "pd.json"
    endpoint = {"host": "10.0.0.1", "port": 8001}
    path.write_text(
        json.dumps(
            {
                "runtime": {
                    "prefill": [endpoint, endpoint],
                    "decode": [{"host": "10.0.1.1", "port": 8002}],
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate endpoints"):
        load_pd_document(path)
