# Copyright (c) PyPTO Contributors.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""Multi-node Router directory and compatibility eligibility gate."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from pypto_serving.serving.pd.config import PDRole
from pypto_serving.serving.pd.http_api import CapacitySnapshot, NodeDescriptor

from .client import NodeClient


@dataclass(frozen=True)
class WorkerCandidate:
    descriptor: NodeDescriptor
    capacity: CapacitySnapshot
    client: NodeClient


@dataclass(frozen=True)
class DirectorySnapshot:
    prefill: tuple[WorkerCandidate, ...]
    decode: tuple[WorkerCandidate, ...]
    rejected: tuple[str, ...]


@dataclass(frozen=True)
class SelectedPair:
    prefill: NodeDescriptor
    decode: NodeDescriptor
    prefill_client: NodeClient
    decode_client: NodeClient


class WorkerDirectory:
    """Refresh all configured nodes and expose only eligible compatible pairs."""

    def __init__(
        self,
        prefill_clients: tuple[NodeClient, ...],
        decode_clients: tuple[NodeClient, ...],
        run_id: str,
        control_incarnation: int | None = None,
    ) -> None:
        if not prefill_clients or not decode_clients:
            raise ValueError("WorkerDirectory requires nonempty P and D client pools")
        self.prefill_clients = prefill_clients
        self.decode_clients = decode_clients
        self.run_id = run_id
        self.control_incarnation = control_incarnation
        self._snapshot: DirectorySnapshot | None = None

    async def refresh(self) -> DirectorySnapshot:
        results = await asyncio.gather(
            *(
                self._refresh_client(client, PDRole.PREFILL)
                for client in self.prefill_clients
            ),
            *(
                self._refresh_client(client, PDRole.DECODE)
                for client in self.decode_clients
            ),
            return_exceptions=True,
        )
        prefill_count = len(self.prefill_clients)
        prefill, decode, rejected = [], [], []
        for index, result in enumerate(results):
            role = PDRole.PREFILL if index < prefill_count else PDRole.DECODE
            client_index = index if role is PDRole.PREFILL else index - prefill_count
            if isinstance(result, BaseException):
                rejected.append(
                    f"{role.value}[{client_index}]:{type(result).__name__}"
                )
            elif role is PDRole.PREFILL:
                prefill.append(result)
            else:
                decode.append(result)
        snapshot = DirectorySnapshot(tuple(prefill), tuple(decode), tuple(rejected))
        if not snapshot.prefill or not snapshot.decode:
            raise RuntimeError(
                "PD directory has no eligible P/D pool; "
                f"rejected={snapshot.rejected}"
            )
        if not self.compatible_pairs(snapshot):
            raise RuntimeError("PD directory has no compatible P/D pair")
        self._snapshot = snapshot
        return snapshot

    async def select(self) -> SelectedPair:
        snapshot = await self.refresh()
        prefill, decode = self.compatible_pairs(snapshot)[0]
        return SelectedPair(
            prefill=prefill.descriptor,
            decode=decode.descriptor,
            prefill_client=prefill.client,
            decode_client=decode.client,
        )

    @classmethod
    def compatible_pairs(
        cls,
        snapshot: DirectorySnapshot,
    ) -> tuple[tuple[WorkerCandidate, WorkerCandidate], ...]:
        return tuple(
            (prefill, decode)
            for prefill in snapshot.prefill
            for decode in snapshot.decode
            if cls.compatible(prefill.descriptor, decode.descriptor)
        )

    @staticmethod
    def compatible(prefill: NodeDescriptor, decode: NodeDescriptor) -> bool:
        if prefill.node_id == decode.node_id:
            return False
        if prefill.control_incarnation != decode.control_incarnation:
            return False
        p_caps = prefill.capabilities
        d_caps = decode.capabilities
        comparable = (
            "schema_version",
            "adapter_id",
            "contract_version",
            "contract_digest",
            "continuation_schema",
            "model_revision",
            "layout_fingerprint",
            "topology",
            "provider",
            "logical_groups",
            "physical_regions",
        )
        return all(getattr(p_caps, name) == getattr(d_caps, name) for name in comparable)

    async def _refresh_client(
        self,
        client: NodeClient,
        role: PDRole,
    ) -> WorkerCandidate:
        descriptor, capacity = await asyncio.gather(
            client.get("/internal/pd/descriptor", NodeDescriptor),
            client.get("/internal/pd/capacity", CapacitySnapshot),
        )
        self._validate_node(descriptor, capacity, role)
        return WorkerCandidate(descriptor, capacity, client)

    def _validate_node(
        self,
        descriptor: NodeDescriptor,
        capacity: CapacitySnapshot,
        role: PDRole,
    ) -> None:
        if descriptor.role != role.value:
            raise ValueError("Router endpoint role differs from its configured pool")
        if descriptor.run_id != self.run_id:
            raise ValueError("Router and PD node run_id differ")
        if descriptor.health != "READY":
            raise RuntimeError("PD node is not healthy")
        if descriptor.owner_generation < 1 or descriptor.endpoint_generation < 1:
            raise ValueError("PD node generation must be positive")
        if (
            self.control_incarnation is not None
            and descriptor.control_incarnation != self.control_incarnation
        ):
            raise ValueError("Router and PD node control incarnations differ")
        if capacity.node_id != descriptor.node_id or capacity.role != role.value:
            raise ValueError("PD node descriptor and capacity identity differ")
        if capacity.snapshot_sequence < 1:
            raise ValueError("PD node capacity snapshot sequence must be positive")
        if capacity.quarantined_reservations:
            raise RuntimeError("PD node has quarantined reservations")
        if capacity.active_limit < 1 or capacity.active_handoffs >= capacity.active_limit:
            raise RuntimeError("PD node has no active handoff capacity")
