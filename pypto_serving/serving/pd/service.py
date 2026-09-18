# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Serving integration for external-Router PD handoffs."""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from collections import OrderedDict
from collections.abc import AsyncGenerator, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pypto_serving.config.types import GenerateConfig
from pypto_serving.serving.engine.async_engine import TokenOutput
from pypto_serving.tools.profile import profile_instant
from pypto_serving.transfer.types import CompletionCertainty

from .admission import FairByteBudget, FairHandoffAdmission
from .config import PDCapabilities, PDConfig, PDRole
from .contracts import RuntimeLayoutDescriptor
from .connector import DecodeConnector
from .coordinator import HandoffCoordinator
from .journal import DurablePDJournal
from .metrics import PDMetrics
from .observability import write_startup_record
from .http_api import (
    AuthorizeRouteHTTP,
    CapacitySnapshot,
    DecodeStreamFrame,
    ExecutePrefillHTTP,
    NodeDescriptor,
    PlacementReservation,
    PrefillHandoffResult,
    PreparedRequest,
    ReservePlacementHTTP,
    capability_compatibility_digest,
)
from .planner import ChunkTransferPlanner, PChunkLifecycle
from .protocol import (
    AbortHandoff,
    ChunkManifest,
    CommitRequest,
    ContinuationMetadata,
    ControlAck,
    ControlError,
    DecodeOutputWire,
    HandoffKey,
    HandoffStatus,
    OpenRoute,
    QueryHandoff,
    ReadyAck,
    RegistryAdvertisement,
    ReleaseHandoff,
    ReserveAccepted,
    ReserveRejected,
    ReserveRequest,
    RouteOpened,
    CapabilityWire,
    TransferResult,
    continuation_metadata_hash,
)
from .session import (
    PDControlAcceptor,
    PDControlSession,
    MultiplexedPDControlSession,
    PeerSessionHandle,
    PeerSessionKey,
    PeerSessionPool,
    connect_control_session,
)
from .worker_api import (
    OP_INSTALL_PEER,
    OP_INSPECT_RUNTIME_METRICS,
    OP_POLL_TRANSFER_CHUNK,
    OP_PREPARE_REGISTRY,
    OP_SUBMIT_TRANSFER_CHUNK,
    OP_TRANSFER_CHUNK,
    InstallPeerRequest,
    TransferChunkRequest,
    TransferChunkResponse,
    TransferPollRequest,
    TransferPollResponse,
    TransferSubmission,
    WorkerRegistryBundle,
    WorkerRuntimeMetrics,
    decode_worker_payload,
    encode_worker_payload,
)


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pypto_serving.serving.engine.async_engine import AsyncLLMEngine, ReplicaEngineCore


@dataclass(frozen=True)
class _PreparedLocal:
    public: PreparedRequest
    prompt: str
    config: GenerateConfig
    prompt_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class _AuthorizedRoute:
    request: AuthorizeRouteHTTP
    prefill_node_id: str
    prefill_endpoint_generation: int


class PDServingService:
    """Bind request-aware Host handoffs to one Serving engine/worker group."""

    def __init__(self, engine: "AsyncLLMEngine", config: PDConfig) -> None:
        self.engine = engine
        self.config = config
        self.core: ReplicaEngineCore = engine._single_core()
        self.session: PDControlSession | None = None
        self.local_bundle: WorkerRegistryBundle | None = None
        self.peer_registry: RegistryAdvertisement | None = None
        self.capabilities: PDCapabilities | None = None
        self.coordinator = HandoffCoordinator(config)
        self.decode_connector: DecodeConnector | None = None
        self.planner: ChunkTransferPlanner | None = None
        self.source_lifecycle: PChunkLifecycle | None = None
        self._decode_loop_task: asyncio.Task | None = None
        self._control_acceptor: PDControlAcceptor | None = None
        self._admission = FairHandoffAdmission(
            config.max_active_handoffs,
            config.max_pending_handoffs,
        )
        self._transfer_budget = FairByteBudget(config.max_inflight_transfer_bytes)
        self.journal = (
            DurablePDJournal(config.journal_path, config)
            if config.journal_path
            else None
        )
        self._active_keys = set()
        self._prepared_by_id: dict[str, _PreparedLocal] = {}
        self._prepared_by_request: dict[str, str] = {}
        self._prefill_results: OrderedDict[HandoffKey, PrefillHandoffResult] = OrderedDict()
        self._prefill_result_limit = 1024
        self._reservation_capabilities: dict[HandoffKey, str] = {}
        self._reservation_bindings: dict[HandoffKey, tuple[object, ...]] = {}
        self._authorized_routes: dict[HandoffKey, _AuthorizedRoute] = {}
        self._route_authorized = asyncio.Event()
        self._decode_queues: dict[HandoffKey, asyncio.Queue[DecodeStreamFrame]] = {}
        self._decode_waiter_claimed: set[HandoffKey] = set()
        self._session_peer_node_id = ""
        self._session_peer_generation = 0
        self._session_pool = PeerSessionPool()
        self._multiplex_session: MultiplexedPDControlSession | None = None
        self._session_connect_lock = asyncio.Lock()
        self._external_handoff_tasks: set[asyncio.Task] = set()
        self._snapshot_sequence = 0
        self._health_error = ""
        self.metrics = PDMetrics(f"pd-{config.role.value}", config.node_id)
        self._closed = False

    async def start(self) -> None:
        write_startup_record(
            self.config.log_dir,
            enabled=self.config.observability_enabled,
            values={
                "process": self.config.role.value,
                "run_id": self.config.run_id,
                "node_id": self.config.node_id,
                "provider": self.config.provider,
            },
        )
        self._record("SERVICE_STARTING")
        raw_bundle = await self.core.call_pd_worker(OP_PREPARE_REGISTRY)
        bundle = decode_worker_payload(raw_bundle, WorkerRegistryBundle)
        registry = self.config.model_adapter.build_registry(bundle)
        if bundle.model_revision != self.config.model_revision:
            raise RuntimeError(
                "worker registry model revision differs from the PD configuration"
            )
        if registry.fingerprint != bundle.registry_fingerprint:
            raise RuntimeError("worker registry fingerprint is not self-consistent")
        if registry.layout_fingerprint != bundle.layout_fingerprint:
            raise RuntimeError("worker layout fingerprint is not self-consistent")
        capabilities = PDCapabilities.from_layout(
            RuntimeLayoutDescriptor.from_runtime_bundle(
                self.config.model_contract,
                bundle,
                provider=self.config.provider,
            )
        )
        local_advertisement = RegistryAdvertisement(
            model_revision=bundle.model_revision,
            topology=bundle.topology,
            registry_fingerprint=bundle.registry_fingerprint,
            layout_fingerprint=bundle.layout_fingerprint,
            ranks=bundle.ranks,
        )
        self.local_bundle = bundle
        self.capabilities = capabilities
        if self.config.role is PDRole.PREFILL:
            self.planner = self.config.model_adapter.make_planner(
                registry,
                self.core.kv_cache_manager.group_specs,
            )
            self.source_lifecycle = PChunkLifecycle(self.core.kv_cache_manager)
        else:
            self.decode_connector = self.config.model_adapter.make_decode_connector(
                self.core.kv_cache_manager,
                capabilities,
                registry,
                bundle.ranks,
            )

        if self.config.role is PDRole.DECODE:
            self._control_acceptor = PDControlAcceptor(self.config)
            self._decode_loop_task = asyncio.create_task(
                self._serve_external_connections(local_advertisement)
            )

        if self._decode_loop_task is not None:
            self._decode_loop_task.add_done_callback(self._decode_loop_done)
        self._record("SERVICE_READY")

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.session is not None:
            with suppress(Exception):
                await self.session.close()
        with suppress(Exception):
            await self._session_pool.close()
        self._multiplex_session = None
        if self._control_acceptor is not None:
            self._control_acceptor.close()
            self._control_acceptor = None
        if self._decode_loop_task is not None:
            self._decode_loop_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await self._decode_loop_task
            self._decode_loop_task = None
        for task in tuple(self._external_handoff_tasks):
            task.cancel()
        if self._external_handoff_tasks:
            await asyncio.gather(
                *self._external_handoff_tasks,
                return_exceptions=True,
            )
        self._external_handoff_tasks.clear()
        for key in tuple(self._active_keys):
            self._record(
                "RECOVERY_REQUIRED",
                key=key,
                error_code="SERVICE_STOPPED_WITH_ACTIVE_HANDOFF",
            )
            self._active_keys.discard(key)
        self._record("SERVICE_STOPPED", error_code=self._health_error)
        if self.journal is not None:
            self.journal.close()

    @property
    def health_error(self) -> str:
        return self._health_error

    def descriptor(self) -> NodeDescriptor:
        """Return the Router-visible, address-free node contract."""
        if self.capabilities is None:
            raise RuntimeError("PD service has not prepared its local registry")
        return NodeDescriptor(
            node_id=self.config.node_id,
            role=self.config.role.value,
            run_id=self.config.run_id,
            control_host=self.config.control_advertise_host or self.config.control_host,
            control_port=self.config.control_port,
            owner_generation=self.config.generation,
            endpoint_generation=self.config.generation,
            control_incarnation=self.config.control_incarnation,
            capabilities=CapabilityWire.from_capabilities(self.capabilities),
            health="RECOVERY_REQUIRED" if self._health_error else "READY",
        )

    def capacity_snapshot(self) -> CapacitySnapshot:
        reservations = tuple(self.core.kv_cache_manager.group_cache_reservations)
        self._snapshot_sequence += 1
        self.metrics.set_gauge("active_handoffs", len(self._active_keys))
        self.metrics.set_gauge("queued_handoffs", self._admission.queued)
        self.metrics.set_gauge("prepared_requests", len(self._prepared_by_id))
        self.metrics.set_gauge("reservations", len(reservations))
        self.metrics.set_gauge("inflight_transfer_bytes", self._transfer_budget.used)
        return CapacitySnapshot(
            node_id=self.config.node_id,
            role=self.config.role.value,
            active_handoffs=len(self._active_keys),
            prepared_requests=len(self._prepared_by_id),
            reservations=len(reservations),
            quarantined_reservations=sum(
                reservation.state.value == "QUARANTINED"
                for reservation in reservations
            ),
            snapshot_sequence=self._snapshot_sequence,
            active_limit=self._admission.active_limit,
            queued_handoffs=self._admission.queued,
            inflight_transfer_bytes=self._transfer_budget.used,
            inflight_transfer_byte_limit=self._transfer_budget.limit,
        )

    async def metrics_snapshot(self) -> dict[str, object]:
        """Return Host metrics plus worker-resident model/transfer counters."""
        snapshot = self.metrics.snapshot()
        try:
            raw = await self.core.call_pd_worker(OP_INSPECT_RUNTIME_METRICS)
            worker = decode_worker_payload(raw, WorkerRuntimeMetrics)
            snapshot["worker"] = worker.values
        except BaseException as exc:
            # Metrics must never take Serving down.  Preserve an address-free
            # error class so operators can distinguish unavailable telemetry.
            snapshot["worker_error"] = type(exc).__name__
        return snapshot

    def prepare_request(
        self,
        request_id: str,
        prompt: str,
        config: GenerateConfig,
        prompt_token_ids: Sequence[int],
    ) -> PreparedRequest:
        """Normalize once on P and retain the immutable execution input."""
        self._require_external_role(PDRole.PREFILL)
        self._validate_sampling(config)
        now_ns = time.time_ns()
        self._expire_prepared(now_ns)
        continuation = self._continuation(config, prompt_token_ids)
        digest = continuation_metadata_hash(continuation)
        existing_id = self._prepared_by_request.get(request_id)
        if existing_id is not None:
            existing = self._prepared_by_id[existing_id]
            if existing.public.prepared_digest != digest or existing.prompt != prompt:
                raise ValueError("prepared request id was replayed with different input")
            return existing.public
        prepared_request_id = f"prepared-{secrets.token_hex(16)}"
        public = PreparedRequest(
            request_id=request_id,
            prepared_request_id=prepared_request_id,
            prepared_digest=digest,
            continuation=continuation,
            expires_at_ns=now_ns
            + int(self.config.prepared_request_ttl_seconds * 1_000_000_000),
        )
        self._prepared_by_id[prepared_request_id] = _PreparedLocal(
            public=public,
            prompt=prompt,
            config=config,
            prompt_token_ids=tuple(int(token) for token in prompt_token_ids),
        )
        self._prepared_by_request[request_id] = prepared_request_id
        return public

    def reserve_placement(self, request: ReservePlacementHTTP) -> PlacementReservation:
        """Reserve D cache before Prefill and return only logical placement data."""
        self._require_external_role(PDRole.DECODE)
        assert self.decode_connector is not None
        if (
            request.key.route_epoch != self.config.route_epoch
            or request.key.control_incarnation != self.config.control_incarnation
        ):
            raise ValueError("D reservation route epoch or control incarnation is stale")
        if (
            not request.prepared_request_id
            or len(request.prepared_request_id) > 256
            or not request.prefill_node_id
            or len(request.prefill_node_id) > 256
            or request.prefill_endpoint_generation < 1
        ):
            raise ValueError("D reservation contains an invalid P/prepared identity")
        result = self.decode_connector.reserve(
            ReserveRequest(
                key=request.key,
                prompt_token_count=request.prompt_token_count,
                max_new_tokens=request.max_new_tokens,
                layout_fingerprint=request.layout_fingerprint,
                prepared_digest=request.prepared_digest,
            )
        )
        if isinstance(result, ReserveRejected):
            raise RuntimeError(f"D rejected the PD reservation: {result.reason}")
        binding = (
            request.prepared_request_id,
            request.prepared_digest,
            request.prefill_node_id,
            request.prefill_endpoint_generation,
            result.reservation_id,
        )
        existing_binding = self._reservation_bindings.get(request.key)
        if existing_binding is not None and existing_binding != binding:
            raise ValueError("D reservation was replayed with a different P binding")
        capability = self._reservation_capabilities.get(request.key)
        if capability is None:
            capability = secrets.token_urlsafe(32)
            self._reservation_capabilities[request.key] = capability
            self._reservation_bindings[request.key] = binding
            self._active_keys.add(request.key)
            self._record(
                "HANDOFF_CREATED",
                key=request.key,
                state="CREATED",
            )
            self._record(
                "HANDOFF_RESERVED",
                key=request.key,
                reservation_id=result.reservation_id,
                state="RESERVED",
            )
        descriptor = self.descriptor()
        return PlacementReservation(
            key=request.key,
            prepared_request_id=request.prepared_request_id,
            reservation_id=result.reservation_id,
            partition=result.partition,
            block_ids_by_group=result.block_ids_by_group,
            prepared_digest=request.prepared_digest,
            reservation_capability=capability,
            decode_node_id=self.config.node_id,
            decode_control_host=descriptor.control_host,
            decode_control_port=descriptor.control_port,
            decode_endpoint_generation=descriptor.endpoint_generation,
        )

    def authorize_route(self, request: AuthorizeRouteHTTP) -> None:
        """Bind a Router assignment to the existing destination reservation."""
        self._require_external_role(PDRole.DECODE)
        assert self.decode_connector is not None
        status = self.decode_connector.query(request.key)
        capability = self._reservation_capabilities.get(request.key)
        binding = self._reservation_bindings.get(request.key)
        if (
            request.compatibility_digest
            != capability_compatibility_digest(self.descriptor().capabilities)
            or request.decode_node_id != self.config.node_id
            or request.decode_endpoint_generation != self.config.generation
            or status.reservation_id != request.reservation_id
            or binding
            != (
                request.prepared_request_id,
                request.prepared_digest,
                request.prefill_node_id,
                request.prefill_endpoint_generation,
                request.reservation_id,
            )
            or capability is None
            or not secrets.compare_digest(capability, request.reservation_capability)
        ):
            raise ValueError("route authorization does not match the D reservation")
        existing = self._authorized_routes.get(request.key)
        authorized = _AuthorizedRoute(
            request=request,
            prefill_node_id=request.prefill_node_id,
            prefill_endpoint_generation=request.prefill_endpoint_generation,
        )
        if existing is not None and existing != authorized:
            raise ValueError("route authorization was replayed with different claims")
        self._authorized_routes[request.key] = authorized
        self._route_authorized.set()
        self._record(
            "ROUTE_AUTHORIZED",
            key=request.key,
            reservation_id=request.reservation_id,
            state=status.state,
        )

    def open_decode_stream(
        self,
        key: HandoffKey,
    ) -> AsyncGenerator[DecodeStreamFrame, None]:
        """Claim the single Router output stream before P starts Prefill."""
        self._require_external_role(PDRole.DECODE)
        if key not in self._authorized_routes:
            raise ValueError("decode stream requires an authorized route")
        if key in self._decode_waiter_claimed:
            raise ValueError("decode stream was already claimed")
        queue: asyncio.Queue[DecodeStreamFrame] = asyncio.Queue(maxsize=64)
        self._decode_queues[key] = queue
        self._decode_waiter_claimed.add(key)

        async def stream() -> AsyncGenerator[DecodeStreamFrame, None]:
            try:
                yield DecodeStreamFrame(event="waiting", state="AUTHORIZED")
                while True:
                    frame = await queue.get()
                    yield frame
                    if frame.event in ("finished", "error"):
                        return
            finally:
                if self._decode_queues.get(key) is queue:
                    self._decode_queues.pop(key, None)

        return stream()

    def query_handoff(self, key: HandoffKey) -> HandoffStatus:
        if self.config.role is PDRole.DECODE:
            assert self.decode_connector is not None
            return self.decode_connector.query(key)
        record = self.coordinator.query(key)
        if record is None:
            raise KeyError("unknown P handoff identity")
        return HandoffStatus(
            key=key,
            state=record.state.value,
            reservation_id=record.reservation_id,
            manifest_hash=record.manifest_hash,
            error_code=record.error_code,
        )

    async def abort_handoff(self, key: HandoffKey, reason: str) -> HandoffStatus:
        """Best-effort deterministic cancellation; committed D work stays owner-led."""
        if self.config.role is PDRole.DECODE:
            assert self.decode_connector is not None
            status = self.decode_connector.query(key)
            if status.state == "IN_USE":
                await self.core.abort_request(key.request_id)
                aborted = self.decode_connector.mark_decode_cancelled(
                    key,
                    error_code=reason,
                )
                self._clear_decode_route(key)
                return aborted
            aborted = self.decode_connector.abort(
                AbortHandoff(key=key, reason=reason, deterministic=True),
                deterministic=True,
            )
            if aborted.state == "ABORTED":
                self._clear_decode_route(key)
            return aborted
        if key in self._active_keys:
            await self.core.abort_request(key.request_id)
        return self.query_handoff(key)

    def _require_external_role(self, role: PDRole) -> None:
        if self.config.role is not role:
            raise ValueError(f"operation requires a {role.value} node")
        if self._health_error:
            raise RuntimeError(
                f"PD serving is fail-closed and requires restart: {self._health_error}"
            )

    def _expire_prepared(self, now_ns: int) -> None:
        expired = [
            prepared_id
            for prepared_id, prepared in self._prepared_by_id.items()
            if prepared.public.expires_at_ns < now_ns
        ]
        for prepared_id in expired:
            request_id = self._prepared_by_id.pop(prepared_id).public.request_id
            self._prepared_by_request.pop(request_id, None)

    async def execute_prefill(
        self,
        request: ExecutePrefillHTTP,
    ) -> PrefillHandoffResult:
        """Execute one Router-selected, D-first handoff on P."""
        self._require_external_role(PDRole.PREFILL)
        self._expire_prepared(time.time_ns())
        completed = self._prefill_results.get(request.key)
        if completed is not None:
            if completed.reservation_id != request.reservation_id:
                raise ValueError("completed Prefill was replayed with another reservation")
            self._prefill_results.move_to_end(request.key)
            return completed
        try:
            prepared = self._prepared_by_id[request.prepared_request_id]
        except KeyError as exc:
            raise ValueError("unknown or expired prepared request") from exc
        if (
            prepared.public.request_id != request.key.request_id
            or prepared.public.prepared_digest != request.prepared_digest
            or prepared.public.expires_at_ns < time.time_ns()
        ):
            raise ValueError("Prefill execution does not match the prepared request")
        if request.decode_endpoint_generation < 1:
            raise ValueError("Decode endpoint generation must be positive")
        await self.run_prefill_handoff(
            prepared.public.request_id,
            prepared.prompt,
            prepared.config,
            prepared.prompt_token_ids,
            external_request=request,
        )
        return self._prefill_results[request.key]

    async def _ensure_external_session(self, request: ExecutePrefillHTTP) -> None:
        assert self.capabilities is not None
        async with self._session_connect_lock:
            cached = self._session_pool.lookup(
                request.decode_node_id,
                request.decode_endpoint_generation,
            )
            if cached is not None:
                self._multiplex_session = cached.session
                self.peer_registry = cached.registry
                return
            if self._multiplex_session is not None:
                with suppress(Exception):
                    await self._multiplex_session.close()
                self._multiplex_session = None
            self.session = None
            self.peer_registry = None
            session = await connect_control_session(
                self.config,
                self.capabilities,
                peer_node_id=request.decode_node_id,
                peer_host=request.decode_control_host,
                peer_port=request.decode_control_port,
            )
            try:
                if session.peer_hello.endpoint_generation != request.decode_endpoint_generation:
                    raise ValueError("Router selected a stale Decode endpoint generation")
                peer = await session.exchange_registry(self._local_advertisement())
                await self._install_peer(peer)
                multiplex = MultiplexedPDControlSession(session)
                multiplex.start()
            except BaseException:
                await session.close()
                raise
            self.peer_registry = peer
            self._multiplex_session = multiplex
            self._session_peer_node_id = request.decode_node_id
            self._session_peer_generation = request.decode_endpoint_generation
            await self._session_pool.replace(
                PeerSessionHandle(
                    key=PeerSessionKey(
                        node_id=request.decode_node_id,
                        endpoint_generation=request.decode_endpoint_generation,
                        registry_fingerprint=peer.registry_fingerprint,
                    ),
                    session=multiplex,
                    registry=peer,
                )
            )

    async def _send_control(self, message) -> None:
        if self._multiplex_session is None:
            raise RuntimeError("PD Prefill has no multiplexed Decode session")
        await self._multiplex_session.send(message)

    async def _receive_control(self, key: HandoffKey):
        if self._multiplex_session is None:
            raise RuntimeError("PD Prefill has no multiplexed Decode session")
        return await self._multiplex_session.receive(key)

    def _local_advertisement(self) -> RegistryAdvertisement:
        if self.local_bundle is None:
            raise RuntimeError("PD local registry has not been prepared")
        bundle = self.local_bundle
        return RegistryAdvertisement(
            model_revision=bundle.model_revision,
            topology=bundle.topology,
            registry_fingerprint=bundle.registry_fingerprint,
            layout_fingerprint=bundle.layout_fingerprint,
            ranks=bundle.ranks,
        )

    async def _install_peer(self, peer: RegistryAdvertisement) -> None:
        await self.core.call_pd_worker(
            OP_INSTALL_PEER,
            encode_worker_payload(
                InstallPeerRequest(
                    registry_fingerprint=peer.registry_fingerprint,
                    layout_fingerprint=peer.layout_fingerprint,
                    ranks=peer.ranks,
                )
            ),
        )

    def _peer_partition_ranks(self, partition: int):
        if self.peer_registry is None:
            raise RuntimeError("Decode owner registry is unavailable")
        topology = self.peer_registry.topology
        group_size = topology[1] if len(topology) > 1 else 1
        rank_count = topology[0]
        if rank_count % group_size:
            raise ValueError("Decode topology has an invalid cache-replica group")
        partition_count = rank_count // group_size
        if not 0 <= partition < partition_count:
            raise ValueError("reservation partition is outside the Decode topology")
        start = partition * group_size
        return tuple(self.peer_registry.ranks[start : start + group_size])

    def _record(
        self,
        event: str,
        *,
        key=None,
        reservation_id: str = "",
        manifest_hash: str = "",
        chunk_id: int = -1,
        state: str = "",
        certainty: str = "",
        error_code: str = "",
    ) -> None:
        self.metrics.increment(f"event.{event}")
        if self.journal is None:
            return
        self.journal.append(
            event,
            key=key,
            reservation_id=reservation_id,
            manifest_hash=manifest_hash,
            chunk_id=chunk_id,
            state=state,
            certainty=certainty,
            error_code=error_code,
        )

    def _decode_loop_done(self, task: asyncio.Task) -> None:
        if self._closed or task.cancelled():
            return
        error = task.exception()
        if error is None:
            self._health_error = "DECODE_CONTROL_LOOP_STOPPED"
        else:
            self._health_error = type(error).__name__
        self._record("CONTROL_LOOP_FAILED", error_code=self._health_error)

    async def run_prefill_handoff(
        self,
        request_id: str,
        prompt: str,
        config,
        prompt_token_ids: Sequence[int],
        *,
        external_request: ExecutePrefillHTTP | None = None,
    ) -> None:
        """Run P Prefill, push each chunk, then relay D Decode output."""
        if self.config.role is not PDRole.PREFILL:
            raise ValueError("external generation is accepted only by the PD Prefill role")
        if self._health_error:
            raise RuntimeError(
                f"PD serving is fail-closed and requires restart: {self._health_error}"
            )
        self._validate_sampling(config)
        if self.planner is None or self.source_lifecycle is None:
            raise RuntimeError("PD Prefill service has not started")

        if external_request is None:
            raise ValueError("send generation requests through the external PD Router")
        await self._ensure_external_session(external_request)
        if self._multiplex_session is None:
            raise RuntimeError("PD Prefill has no Decode session")

        async with self._admission.admit():
            record = self.coordinator.register_handoff(
                external_request.key,
                prefill_node_id=self.config.node_id,
                decode_node_id=external_request.decode_node_id,
            )
            self._multiplex_session.open_route(record.key)
            self._active_keys.add(record.key)
            self._record("HANDOFF_CREATED", key=record.key, state=record.state.value)
            try:
                await self._send_control(
                    OpenRoute(
                        key=record.key,
                        prepared_request_id=external_request.prepared_request_id,
                        reservation_id=external_request.reservation_id,
                        prepared_digest=external_request.prepared_digest,
                        reservation_capability=external_request.reservation_capability,
                        compatibility_digest=external_request.compatibility_digest,
                        prefill_node_id=external_request.prefill_node_id,
                        prefill_endpoint_generation=(
                            external_request.prefill_endpoint_generation
                        ),
                    )
                )
                opened = await self._receive_control(record.key)
                if (
                    not isinstance(opened, RouteOpened)
                    or opened.key != record.key
                    or opened.reservation_id != external_request.reservation_id
                ):
                    raise RuntimeError("D did not open the Router-authorized route")
                reservation = ReserveAccepted(
                    key=record.key,
                    reservation_id=external_request.reservation_id,
                    partition=external_request.partition,
                    block_ids_by_group=external_request.block_ids_by_group,
                    ranks=self._peer_partition_ranks(external_request.partition),
                )
            except BaseException as exc:
                self.coordinator.fail(record.key, type(exc).__name__)
                self._record(
                    "HANDOFF_FAILED",
                    key=record.key,
                    state="FAILED",
                    error_code=type(exc).__name__,
                )
                self._active_keys.discard(record.key)
                if self._multiplex_session is not None:
                    self._multiplex_session.close_route(record.key)
                raise
            self.coordinator.mark_reserved(record.key, reservation.reservation_id)
            self._record(
                "HANDOFF_RESERVED",
                key=record.key,
                reservation_id=reservation.reservation_id,
                state="RESERVED",
            )
            prefill_task = asyncio.create_task(
                self._drain_prefill_request(
                    request_id,
                    prompt,
                    config,
                    prompt_token_ids,
                    reservation.partition,
                )
            )
            guard = None
            transfer_started = False
            d_ready = False
            remote_aborted = False
            try:
                while True:
                    chunk = await self._next_chunk_or_failure(request_id, prefill_task)
                    rank_ids = tuple(rank.rank_id for rank in reservation.ranks)
                    source_tables = {
                        rank_id: chunk.block_ids_by_group for rank_id in rank_ids
                    }
                    destination_tables = {
                        rank_id: reservation.block_ids_by_group for rank_id in rank_ids
                    }
                    plan = self.planner.plan_chunk(
                        record.key,
                        chunk_id=chunk.chunk_id,
                        start_token=chunk.start_token,
                        end_token=chunk.end_token,
                        final=chunk.final,
                        rank_ids=rank_ids,
                        source_blocks_by_rank=source_tables,
                        destination_blocks_by_rank=destination_tables,
                    )
                    guard = self.source_lifecycle.begin(
                        request_id,
                        chunk.chunk_id,
                        chunk.block_ids_by_group,
                        chunk.cache_partition,
                    )
                    continuation = self._continuation(
                        config,
                        prompt_token_ids,
                    ) if chunk.final else None
                    metadata_hash = (
                        continuation_metadata_hash(continuation)
                        if continuation is not None
                        else ""
                    )
                    manifest = ChunkManifest(
                        key=record.key,
                        chunk_id=chunk.chunk_id,
                        start_token=chunk.start_token,
                        end_token=chunk.end_token,
                        final=chunk.final,
                        manifest_hash=plan.manifest_hash,
                        expected_units=plan.expected_units,
                        copies_by_rank=plan.copies_by_rank,
                        first_token=chunk.first_token,
                        metadata_hash=metadata_hash,
                        continuation=continuation,
                        prepared_digest=external_request.prepared_digest,
                    )
                    self._record(
                        "CHUNK_PLANNED",
                        key=record.key,
                        reservation_id=reservation.reservation_id,
                        manifest_hash=manifest.manifest_hash,
                        chunk_id=chunk.chunk_id,
                        state="TRANSFERRING",
                    )
                    await self._send_expect_ack(manifest, "chunk_manifest", chunk.chunk_id)
                    certainty = CompletionCertainty.NOT_SUBMITTED
                    overlap_released = False
                    for attempt_ordinal in range(
                        1,
                        self.config.max_transfer_attempts + 1,
                    ):
                        transfer_started = True
                        response, released = await self._transfer_chunk(
                            request_id,
                            TransferChunkRequest(
                                manifest=manifest,
                                destination_fingerprint=(
                                    self.peer_registry.registry_fingerprint
                                ),
                                timeout_seconds=self.config.request_timeout_seconds,
                                attempt_ordinal=attempt_ordinal,
                            ),
                            release_next_chunk=(
                                self.config.enable_chunk_overlap
                                and not chunk.final
                                and not overlap_released
                            ),
                        )
                        overlap_released = overlap_released or released
                        transfer_started = False
                        certainty = await self._publish_transfer_results(response.results)
                        self._record(
                            "TRANSFER_ATTEMPT_FINISHED",
                            key=record.key,
                            reservation_id=reservation.reservation_id,
                            manifest_hash=manifest.manifest_hash,
                            chunk_id=chunk.chunk_id,
                            state="TRANSFERRING",
                            certainty=certainty.value,
                        )
                        if certainty is CompletionCertainty.COMPLETED:
                            break
                        if certainty is CompletionCertainty.UNKNOWN:
                            break
                    if certainty is not CompletionCertainty.COMPLETED:
                        self.source_lifecycle.settle(guard, certainty)
                        guard = None
                        await self._abort_remote(
                            record.key,
                            reason=f"TRANSFER_{certainty.value}",
                            deterministic=certainty is not CompletionCertainty.UNKNOWN,
                        )
                        remote_aborted = True
                        self.coordinator.fail(record.key, certainty.value)
                        terminal_event = (
                            "RECOVERY_REQUIRED"
                            if certainty is CompletionCertainty.UNKNOWN
                            else "HANDOFF_FAILED"
                        )
                        self._record(
                            terminal_event,
                            key=record.key,
                            reservation_id=reservation.reservation_id,
                            state="FAILED",
                            certainty=certainty.value,
                            error_code=f"TRANSFER_{certainty.value}",
                        )
                        self._active_keys.discard(record.key)
                        if certainty is CompletionCertainty.UNKNOWN:
                            self._health_error = "UNKNOWN_TRANSFER"
                        raise RuntimeError(f"PD chunk transfer failed: {certainty.value}")

                    if not chunk.final:
                        self.source_lifecycle.settle(guard, certainty)
                        guard = None
                        if not overlap_released:
                            self.core.complete_prefill_chunk_transfer(request_id)
                        continue

                    commit = CommitRequest(
                        key=record.key,
                        manifest_hash=manifest.manifest_hash,
                        first_token=chunk.first_token,
                        metadata_hash=metadata_hash,
                    )
                    await self._send_control(commit)
                    ready = await self._receive_control(record.key)
                    if isinstance(ready, ControlError):
                        raise RuntimeError(f"D rejected commit: {ready.error_code}")
                    if not isinstance(ready, ReadyAck) or ready.key != record.key:
                        raise RuntimeError("D did not return the matching READY acknowledgement")
                    if ready.manifest_hash != manifest.manifest_hash:
                        raise RuntimeError("D READY acknowledgement changed the manifest")
                    d_ready = True
                    self.coordinator.mark_ready(record.key, ready.manifest_hash)
                    self._record(
                        "HANDOFF_COMMITTED",
                        key=record.key,
                        reservation_id=ready.reservation_id,
                        manifest_hash=ready.manifest_hash,
                        state="READY",
                    )
                    self.source_lifecycle.settle(guard, certainty)
                    guard = None
                    self.core.acknowledge_prefill_handoff(request_id)
                    self.coordinator.release(record.key)
                    await prefill_task
                    result = PrefillHandoffResult(
                        key=record.key,
                        reservation_id=ready.reservation_id,
                        manifest_hash=ready.manifest_hash,
                        state="READY",
                    )
                    self._remember_prefill_result(result)
                    self._drop_prepared(external_request.prepared_request_id)
                    self._active_keys.discard(record.key)
                    if self._multiplex_session is not None:
                        self._multiplex_session.close_route(record.key)
                    return
            except BaseException as exc:
                logger.exception(
                    "PD handoff execution failed: request=%s handoff=%s error=%s",
                    request_id,
                    record.key.handoff_id,
                    type(exc).__name__,
                )
                if guard is not None:
                    if not transfer_started:
                        self.source_lifecycle.settle(
                            guard,
                            CompletionCertainty.NOT_SUBMITTED,
                        )
                    # A worker-IPC failure after dispatch cannot prove whether
                    # Mooncake submitted. Keep the extra source pin and make D
                    # quarantine its destination reservation.
                    else:
                        with suppress(Exception):
                            await self._abort_remote(
                                record.key,
                                reason="WORKER_TRANSFER_STATUS_UNKNOWN",
                                deterministic=False,
                            )
                            remote_aborted = True
                if not d_ready and not transfer_started and not remote_aborted:
                    with suppress(Exception):
                        await self._abort_remote(
                            record.key,
                            reason="PREFILL_HANDOFF_ABORTED",
                            deterministic=True,
                        )
                        remote_aborted = True
                if not prefill_task.done():
                    await self.core.abort_request(request_id)
                    prefill_task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await prefill_task
                if record.key in self._active_keys:
                    self._record(
                        "RECOVERY_REQUIRED" if d_ready or transfer_started else "HANDOFF_FAILED",
                        key=record.key,
                        reservation_id=reservation.reservation_id,
                        state="FAILED",
                        error_code=type(exc).__name__,
                    )
                    self._active_keys.discard(record.key)
                    if d_ready or transfer_started:
                        self._health_error = type(exc).__name__
                if self._health_error:
                    with suppress(Exception):
                        if self._multiplex_session is not None:
                            await self._multiplex_session.close()
                        elif self.session is not None:
                            await self.session.close()
                if self._multiplex_session is not None:
                    self._multiplex_session.close_route(record.key)
                raise

    async def _transfer_chunk(
        self,
        request_id: str,
        request: TransferChunkRequest,
        *,
        release_next_chunk: bool,
    ) -> tuple[TransferChunkResponse, bool]:
        """Run one bounded transfer, optionally releasing the next P chunk.

        The worker submit command returns only after all older device work is
        fenced and the owner-local progress threads own the native tasks.  A
        non-final manifest contains closed pages only, so the scheduler may
        then compute the next chunk while Host polls completion.
        """
        transfer_bytes = sum(unit.nbytes for unit in request.manifest.expected_units)
        started_ns = time.monotonic_ns()
        self.metrics.increment("transfer.chunks")
        self.metrics.increment("transfer.bytes", transfer_bytes)
        profile_instant(
            "pd.transfer.begin",
            cat="pd",
            args={
                "request_id": request_id,
                "chunk_id": request.manifest.chunk_id,
                "bytes": transfer_bytes,
                "overlap": release_next_chunk,
            },
        )
        async with self._transfer_budget.reserve(transfer_bytes):
            try:
                if not release_next_chunk:
                    raw_response = await self.core.call_pd_worker(
                        OP_TRANSFER_CHUNK,
                        encode_worker_payload(request),
                    )
                    return (
                        decode_worker_payload(raw_response, TransferChunkResponse),
                        False,
                    )

                self.metrics.increment("overlap.submitted")
                raw_submission = await self.core.call_pd_worker(
                    OP_SUBMIT_TRANSFER_CHUNK,
                    encode_worker_payload(request),
                )
                submission = decode_worker_payload(raw_submission, TransferSubmission)
                self.core.complete_prefill_chunk_transfer(request_id)
                deadline = time.monotonic() + self.config.request_timeout_seconds + 10
                while True:
                    # Yield to the engine loop before polling so the next Prefill
                    # StepCommand can reach the worker/device lane.
                    await asyncio.sleep(self.config.transfer_poll_interval_seconds)
                    raw_poll = await self.core.call_pd_worker(
                        OP_POLL_TRANSFER_CHUNK,
                        encode_worker_payload(TransferPollRequest(submission.job_id)),
                    )
                    poll = decode_worker_payload(raw_poll, TransferPollResponse)
                    if poll.job_id != submission.job_id:
                        raise RuntimeError("worker returned another PD transfer job")
                    if poll.complete:
                        self.metrics.increment("overlap.completed")
                        return TransferChunkResponse(poll.results), True
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "PD asynchronous transfer polling exceeded its bound"
                        )
            finally:
                self.metrics.observe_ns(
                    "transfer.duration",
                    time.monotonic_ns() - started_ns,
                )
                profile_instant(
                    "pd.transfer.end",
                    cat="pd",
                    args={
                        "request_id": request_id,
                        "chunk_id": request.manifest.chunk_id,
                    },
                )

    async def _drain_prefill_request(
        self,
        request_id: str,
        prompt: str,
        config,
        prompt_token_ids: Sequence[int],
        partition: int,
    ) -> None:
        async for output in self.core.add_request(
            request_id,
            prompt,
            config,
            prompt_token_ids=prompt_token_ids,
            cache_partition=partition,
        ):
            if output.finish_reason not in ("", "FINISHED_HANDOFF"):
                raise RuntimeError(f"P Prefill ended unexpectedly: {output.finish_reason}")

    async def _next_chunk_or_failure(
        self,
        request_id: str,
        prefill_task: asyncio.Task,
    ):
        chunk_task = asyncio.create_task(self.core.next_prefill_chunk(request_id))
        done, _ = await asyncio.wait(
            (chunk_task, prefill_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if chunk_task in done:
            return chunk_task.result()
        chunk_task.cancel()
        with suppress(asyncio.CancelledError):
            await chunk_task
        error = prefill_task.exception()
        if error is not None:
            raise error
        raise RuntimeError("P Prefill ended before producing a handoff chunk")

    async def _send_expect_ack(
        self,
        message,
        operation: str,
        chunk_id: int,
    ) -> None:
        await self._send_control(message)
        reply = await self._receive_control(message.key)
        if isinstance(reply, ControlError):
            raise RuntimeError(f"D rejected {operation}: {reply.error_code}")
        if (
            not isinstance(reply, ControlAck)
            or reply.key != message.key
            or reply.operation != operation
            or reply.chunk_id != chunk_id
        ):
            raise RuntimeError(f"D did not acknowledge {operation}")

    async def _publish_transfer_results(
        self,
        results: tuple[TransferResult, ...],
    ) -> CompletionCertainty:
        if not results:
            raise RuntimeError("worker returned no physical-region completion facts")
        certainties = []
        for result in results:
            await self._send_expect_ack(result, "transfer_result", result.chunk_id)
            certainties.append(CompletionCertainty(result.certainty))
        if CompletionCertainty.UNKNOWN in certainties:
            return CompletionCertainty.UNKNOWN
        if CompletionCertainty.FAILED_DEFINITE in certainties:
            return CompletionCertainty.FAILED_DEFINITE
        if CompletionCertainty.NOT_SUBMITTED in certainties:
            return CompletionCertainty.NOT_SUBMITTED
        return CompletionCertainty.COMPLETED

    async def _abort_remote(self, key, *, reason: str, deterministic: bool) -> None:
        await self._send_control(AbortHandoff(key, reason, deterministic))
        reply = await self._receive_control(key)
        if not isinstance(reply, HandoffStatus) or reply.key != key:
            raise RuntimeError("D did not return the aborted handoff status")

    async def _receive_decode_outputs(self, key) -> AsyncGenerator[TokenOutput, None]:
        assert self.session is not None
        while True:
            message = await self.session.receive()
            if isinstance(message, ControlError):
                raise RuntimeError(f"D control failure: {message.error_code}")
            if not isinstance(message, DecodeOutputWire) or message.key != key:
                raise RuntimeError("P received an unexpected Decode output message")
            output = TokenOutput(
                token_id=message.token_id,
                text=message.text,
                finished=message.finished,
                finish_reason=message.finish_reason,
                prompt_tokens=message.prompt_tokens,
                completion_tokens=message.completion_tokens,
                token_ids=message.token_ids,
            )
            yield output
            if output.finished:
                return

    async def _serve_external_connections(
        self,
        local_advertisement: RegistryAdvertisement,
    ) -> None:
        assert self._control_acceptor is not None
        assert self.capabilities is not None
        while True:
            while not self._authorized_routes:
                self._route_authorized.clear()
                await self._route_authorized.wait()
            expected_peer = next(iter(self._authorized_routes.values())).prefill_node_id
            session = await self._control_acceptor.accept(
                self.capabilities,
                expected_peer_node_id=expected_peer,
            )
            try:
                peer = await session.exchange_registry(local_advertisement)
                self.session = session
                self.peer_registry = peer
                self._session_peer_node_id = peer.ranks[0].worker_id if peer.ranks else ""
                await self._serve_external_session(session)
            except (EOFError, ConnectionError, OSError):
                pass
            finally:
                await session.close()
                if self.session is session:
                    self.session = None
                    self.peer_registry = None

    async def _serve_external_session(self, session: PDControlSession) -> None:
        """Dispatch one P connection by complete handoff key."""
        assert self.decode_connector is not None
        queues: dict[HandoffKey, asyncio.Queue] = {}
        tasks: dict[HandoffKey, asyncio.Task] = {}

        def finished(key: HandoffKey, task: asyncio.Task) -> None:
            queues.pop(key, None)
            tasks.pop(key, None)
            self._external_handoff_tasks.discard(task)
            if not task.cancelled():
                error = task.exception()
                if error is not None and not self._health_error:
                    self._health_error = type(error).__name__

        try:
            while True:
                message = await session.receive()
                key = getattr(message, "key", None)
                if not isinstance(key, HandoffKey):
                    raise ValueError("external PD message has no handoff correlation")
                if isinstance(message, OpenRoute):
                    if key in tasks:
                        await session.send(
                            ControlError(
                                key,
                                "OpenRoute",
                                "ROUTE_ALREADY_OPEN",
                                True,
                            )
                        )
                        continue
                    if len(tasks) >= self.config.max_active_handoffs:
                        await session.send(
                            ControlError(
                                key,
                                "OpenRoute",
                                "HANDOFF_BACKPRESSURE",
                                True,
                            )
                        )
                        continue
                    queue: asyncio.Queue = asyncio.Queue(maxsize=128)
                    task = asyncio.create_task(
                        self._serve_external_handoff(session, queue, key)
                    )
                    queues[key] = queue
                    tasks[key] = task
                    self._external_handoff_tasks.add(task)
                    task.add_done_callback(lambda done, route=key: finished(route, done))
                    queue.put_nowait(message)
                    continue
                queue = queues.get(key)
                if queue is not None:
                    try:
                        queue.put_nowait(message)
                    except asyncio.QueueFull as exc:
                        raise RuntimeError(
                            "external PD handoff command queue exceeded its bound"
                        ) from exc
                    continue
                if isinstance(message, QueryHandoff):
                    await session.send(self.decode_connector.query(key))
                elif isinstance(message, AbortHandoff):
                    status = self.decode_connector.abort(
                        message,
                        deterministic=message.deterministic,
                    )
                    await session.send(status)
                else:
                    await session.send(
                        ControlError(
                            key,
                            type(message).__name__,
                            "UNKNOWN_HANDOFF_CORRELATION",
                            True,
                        )
                    )
        finally:
            # EOF affects only pre-commit routes still dependent on this P
            # connection. READY/IN_USE Decode uses the Router HTTP stream and
            # is allowed to finish independently.
            cancelled = []
            for key, task in tuple(tasks.items()):
                try:
                    status = self.decode_connector.query(key)
                except KeyError:
                    task.cancel()
                    continue
                if status.state in ("READY", "IN_USE", "COMPLETED"):
                    continue
                deterministic = status.state == "RESERVED"
                with suppress(Exception):
                    self.decode_connector.abort(
                        AbortHandoff(
                            key,
                            "CONTROL_SESSION_EOF",
                            deterministic,
                        ),
                        deterministic=deterministic,
                    )
                if not deterministic:
                    self._health_error = "CONTROL_SESSION_EOF_DURING_TRANSFER"
                task.cancel()
                cancelled.append(task)
            if cancelled:
                await asyncio.gather(*cancelled, return_exceptions=True)

    async def _serve_external_handoff(
        self,
        session: PDControlSession,
        queue: asyncio.Queue,
        key: HandoffKey,
    ) -> None:
        """Preserve FIFO within one route while other routes make progress."""
        assert self.decode_connector is not None
        opened = False
        while True:
            message = await queue.get()
            try:
                if isinstance(message, OpenRoute):
                    if opened:
                        raise ValueError("route opening was replayed on an active route")
                    self._validate_open_route(message, session)
                    opened = True
                    await session.send(RouteOpened(message.key, message.reservation_id))
                elif not opened:
                    raise ValueError("handoff command arrived before OpenRoute")
                elif isinstance(message, ChunkManifest):
                    self.decode_connector.register_chunk(message)
                    status = self.decode_connector.query(key)
                    self._record(
                        "CHUNK_REGISTERED",
                        key=key,
                        reservation_id=status.reservation_id,
                        manifest_hash=message.manifest_hash,
                        chunk_id=message.chunk_id,
                        state=status.state,
                    )
                    await session.send(ControlAck(key, "chunk_manifest", message.chunk_id))
                elif isinstance(message, TransferResult):
                    self.decode_connector.record_transfer(message)
                    status = self.decode_connector.query(key)
                    self._record(
                        "TRANSFER_RESULT_RECORDED",
                        key=key,
                        reservation_id=status.reservation_id,
                        chunk_id=message.chunk_id,
                        state=status.state,
                        certainty=message.certainty,
                        error_code=message.error_code,
                    )
                    await session.send(ControlAck(key, "transfer_result", message.chunk_id))
                elif isinstance(message, CommitRequest):
                    ready = self.decode_connector.commit(message)
                    self._record(
                        "HANDOFF_COMMITTED",
                        key=key,
                        reservation_id=ready.reservation_id,
                        manifest_hash=ready.manifest_hash,
                        state="READY",
                    )
                    await session.send(ready)
                    await self._stream_decode(key)
                    return
                elif isinstance(message, QueryHandoff):
                    await session.send(self.decode_connector.query(key))
                elif isinstance(message, AbortHandoff):
                    status = self.decode_connector.abort(
                        message,
                        deterministic=message.deterministic,
                    )
                    await session.send(status)
                    self._active_keys.discard(key)
                    return
                elif isinstance(message, ReleaseHandoff):
                    await session.send(self.decode_connector.query(key))
                else:
                    raise ValueError(
                        f"unexpected P control message {type(message).__name__}"
                    )
            except (ValueError, KeyError, RuntimeError) as exc:
                status = None
                with suppress(Exception):
                    status = self.decode_connector.query(key)
                deterministic = status is None or status.state == "RESERVED"
                await session.send(
                    ControlError(
                        key=key,
                        operation=type(message).__name__,
                        error_code=type(exc).__name__,
                        deterministic=deterministic,
                    )
                )
                if status is not None and status.state not in (
                    "READY",
                    "IN_USE",
                    "COMPLETED",
                ):
                    with suppress(Exception):
                        self.decode_connector.abort(
                            AbortHandoff(key, type(exc).__name__, deterministic),
                            deterministic=deterministic,
                        )
                self._record(
                    "CONTROL_MESSAGE_REJECTED",
                    key=key,
                    error_code=type(exc).__name__,
                )
                self._active_keys.discard(key)
                if not deterministic:
                    self._health_error = type(exc).__name__
                return

    def _validate_open_route(
        self,
        message: OpenRoute,
        session: PDControlSession,
    ) -> None:
        authorized = self._authorized_routes.get(message.key)
        if authorized is None:
            raise ValueError("P opened a route that Router did not authorize")
        request = authorized.request
        if (
            message.prepared_request_id != request.prepared_request_id
            or message.reservation_id != request.reservation_id
            or message.prepared_digest != request.prepared_digest
            or message.compatibility_digest != request.compatibility_digest
            or message.prefill_node_id != request.prefill_node_id
            or message.prefill_endpoint_generation
            != request.prefill_endpoint_generation
            or not secrets.compare_digest(
                message.reservation_capability,
                request.reservation_capability,
            )
            or session.peer_hello.node_id != authorized.prefill_node_id
            or session.peer_hello.endpoint_generation
            != authorized.prefill_endpoint_generation
        ):
            raise ValueError("P route opening differs from Router authorization")
        if message.key not in self._decode_queues:
            raise RuntimeError("Router must await Decode output before P starts Prefill")
        assert self.decode_connector is not None
        status = self.decode_connector.query(message.key)
        if status.reservation_id != message.reservation_id or status.state != "RESERVED":
            raise ValueError("P route opening references an unusable D reservation")

    async def _stream_decode(self, key) -> None:
        assert self.decode_connector is not None
        manifest = self.decode_connector.committed_manifest(key)
        continuation = manifest.continuation
        if continuation is None or manifest.first_token is None:
            raise RuntimeError("committed handoff has no continuation metadata")
        if not self.decode_connector.claim_decode_admission(key):
            raise RuntimeError("committed handoff was admitted more than once")
        status = self.decode_connector.query(key)
        self._record(
            "DECODE_ADMITTED",
            key=key,
            reservation_id=status.reservation_id,
            manifest_hash=status.manifest_hash,
            state="IN_USE",
        )
        admitted = True
        output_sequence = 0
        try:
            async for output in self.core.add_adopted_handoff(
                reservation_id=self.decode_connector.query(key).reservation_id,
                request_id=key.request_id,
                prompt_token_ids=continuation.prompt_token_ids,
                first_token=manifest.first_token,
                max_new_tokens=continuation.max_new_tokens,
                temperature=continuation.temperature,
                top_p=continuation.top_p,
                top_k=continuation.top_k,
                seed=continuation.seed,
                stop_strings=continuation.stop_strings,
                eos_token_id=continuation.eos_token_id,
                stream=continuation.stream,
            ):
                output_sequence += 1
                wire = DecodeOutputWire(
                    key=key,
                    token_id=output.token_id,
                    text=output.text,
                    finished=output.finished,
                    finish_reason=output.finish_reason,
                    prompt_tokens=output.prompt_tokens,
                    completion_tokens=output.completion_tokens,
                    token_ids=output.token_ids,
                    output_sequence=output_sequence,
                )
                queue = self._decode_queues.get(key)
                if queue is None:
                    raise RuntimeError("Router Decode stream disappeared after admission")
                await queue.put(
                    DecodeStreamFrame(
                        event="finished" if output.finished else "output",
                        output=wire,
                        state="COMPLETED" if output.finished else "IN_USE",
                    )
                )
                if output.finished:
                    self.decode_connector.mark_decode_completed(key)
                    completed = self.decode_connector.query(key)
                    self._record(
                        "HANDOFF_COMPLETED",
                        key=key,
                        reservation_id=completed.reservation_id,
                        manifest_hash=completed.manifest_hash,
                        state="COMPLETED",
                    )
                    self._clear_decode_route(key)
                    self.metrics.record_terminal(
                        request_id=key.request_id,
                        state="COMPLETED",
                        prompt_tokens=output.prompt_tokens,
                        completion_tokens=output.completion_tokens,
                        token_ids=output.token_ids,
                    )
                    return
        except BaseException as exc:
            if admitted:
                await self.core.abort_request(key.request_id)
            self._record(
                "RECOVERY_REQUIRED",
                key=key,
                reservation_id=status.reservation_id,
                manifest_hash=status.manifest_hash,
                state="FAILED",
                error_code=type(exc).__name__,
            )
            self._active_keys.discard(key)
            self._health_error = type(exc).__name__
            queue = self._decode_queues.get(key)
            if queue is not None:
                with suppress(asyncio.QueueFull):
                    queue.put_nowait(
                        DecodeStreamFrame(
                            event="error",
                            state="RECOVERY_REQUIRED",
                            error_code=type(exc).__name__,
                        )
                    )
            raise

    def _remember_prefill_result(self, result: PrefillHandoffResult) -> None:
        self._prefill_results[result.key] = result
        self._prefill_results.move_to_end(result.key)
        while len(self._prefill_results) > self._prefill_result_limit:
            self._prefill_results.popitem(last=False)

    def _drop_prepared(self, prepared_request_id: str) -> None:
        prepared = self._prepared_by_id.pop(prepared_request_id, None)
        if prepared is not None:
            self._prepared_by_request.pop(prepared.public.request_id, None)

    def _clear_decode_route(self, key: HandoffKey) -> None:
        self._active_keys.discard(key)
        self._authorized_routes.pop(key, None)
        self._reservation_capabilities.pop(key, None)
        self._reservation_bindings.pop(key, None)
        self._decode_waiter_claimed.discard(key)
        self._decode_queues.pop(key, None)

    def _continuation(self, config, prompt_token_ids: Sequence[int]) -> ContinuationMetadata:
        return ContinuationMetadata(
            prompt_token_ids=tuple(int(token) for token in prompt_token_ids),
            max_new_tokens=int(config.max_new_tokens),
            temperature=float(config.temperature),
            top_p=float(config.top_p),
            top_k=config.top_k,
            seed=config.seed,
            stop_strings=tuple(config.stop) if config.stop else (),
            eos_token_id=None if config.ignore_eos else self.engine.eos_token_id,
            stream=bool(getattr(config, "stream", True)),
        )

    @staticmethod
    def _validate_sampling(config) -> None:
        if config.temperature != 0.0 or config.top_p != 1.0 or config.top_k is not None:
            raise ValueError("the first PD version supports greedy sampling only")
