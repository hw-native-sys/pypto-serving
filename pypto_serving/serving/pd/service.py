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
import importlib.util
import logging
import time
import msgspec
from collections import OrderedDict
from collections.abc import AsyncGenerator, Sequence
from contextlib import aclosing, suppress
from dataclasses import dataclass

from pypto_serving.config.types import GenerateConfig
from pypto_serving.serving.constraints import ConstraintSpec
from pypto_serving.serving.engine.async_engine import TokenOutput
from pypto_serving.serving.reasoning import OutputParserSpec
from pypto_serving.tools.profile import profile_instant
from pypto_serving.transfer.types import CompletionCertainty

from .admission import FairHandoffAdmission, PDBackpressureError
from .config import PDCapabilities, PDConfig, PDRole
from .contracts import RuntimeLayoutDescriptor
from .connector import DecodeConnector
from .handoff import CoordinatorState, HandoffCoordinator
from .journal import DurablePDJournal
from .lifecycle import TransferLifecycle
from .progress import TransferProgress
from .observability import PDMetrics, write_startup_record
from .planner import ChunkTransferPlanner, PChunkLifecycle
from .protocol import (
    AbortHandoff,
    ChunkManifest,
    CommitRequest,
    ControlAck,
    ControlError,
    HandoffKey,
    HandoffStatus,
    BeginHandoff,
    ContinuationMetadata,
    GenerationContext,
    PrefixMatchSpec,
    PrefillResult,
    CapacitySnapshot,
    NodeDescriptor,
    QueryHandoff,
    ReadyAck,
    RegistryAdvertisement,
    ReserveAccepted,
    ReserveRejected,
    ReserveRequest,
    CapabilityWire,
    TransferResult,
    TransferResultBundle,
    DrainPeer,
    SenderReleased,
    SenderReleaseAck,
    constraint_spec_hash,
    prepared_request_hash,
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
    OP_RELEASE_TRANSFER,
    OP_FINALIZE_TRANSFER,
    OP_INSPECT_RUNTIME_METRICS,
    OP_PREPARE_REGISTRY,
    OP_SUBMIT_TRANSFER_CHUNK,
    InstallPeerRequest,
    TransferChunkRequest,
    TransferChunkResponse,
    TransferSubmission,
    WorkerRegistryBundle,
    WorkerRuntimeMetrics,
    decode_worker_payload,
    encode_worker_payload,
)


logger = logging.getLogger(__name__)


class PrefillRejected(RuntimeError):
    """Positive P-local proof of no work or a safely settled failure."""

    def __init__(self, result: PrefillResult):
        self.result = result
        super().__init__(result.error_code)


@dataclass(frozen=True)
class PreparedInput:
    prompt: str
    config: GenerateConfig
    prompt_token_ids: tuple[int, ...]
    continuation: ContinuationMetadata
    prefix_match_spec: PrefixMatchSpec | None
    digest: str


@dataclass
class NodeRequest:
    context: GenerationContext
    accepted: asyncio.Future
    outputs: asyncio.Queue
    task: asyncio.Task | None = None


class PDServingService:
    """Bind request-aware Host handoffs to one Serving engine/worker group."""

    def __init__(self, core, config: PDConfig, *, eos_token_id: int | None = None) -> None:
        self.eos_token_id = eos_token_id
        self.preflight_constraint = None
        self.config = config
        self.core = core
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
        # Transfer size is model/chunk dependent and is not a resource
        # allocation: the source and destination cache pages already exist.
        # Bound request concurrency, but keep bytes as telemetry rather than a
        # model-independent rejection threshold.
        self._inflight_transfer_bytes = 0
        self.journal = (
            DurablePDJournal(config.journal_path, config)
            if config.journal_path
            else None
        )
        self._active_keys = set()
        self._requests: dict[HandoffKey, NodeRequest] = {}
        self._request_available = asyncio.Event()
        self._retired: OrderedDict[HandoffKey, HandoffStatus] = OrderedDict()
        self._history_limit = 1024
        self._router_cancelled_keys: set[HandoffKey] = set()
        self._session_peer_node_id = ""
        self._session_peer_generation = 0
        self._session_pool = PeerSessionPool()
        self._multiplex_session: MultiplexedPDControlSession | None = None
        self._session_connect_lock = asyncio.Lock()
        self._external_handoff_tasks: dict[HandoffKey, asyncio.Task] = {}
        self._snapshot_sequence = 0
        self._health_error = ""
        self.metrics = PDMetrics(f"pd-{config.role.value}", config.node_id)
        self.progress = TransferProgress(
            core.call_pd_progress, interval=config.transfer_poll_interval_seconds,
            limit=config.max_active_handoffs, metrics=self.metrics,
        )
        self._closed = False
        self._close_task: asyncio.Task | None = None
        self.lifecycle = TransferLifecycle(
            config.role,
            idle=lambda: not self._requests and not self._active_keys and self._admission.count == 0,
            release=lambda: self.core.call_pd_worker(OP_RELEASE_TRANSFER),
            timeout=min(config.request_timeout_seconds, 60),
        )

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
                enable_prefix_cache=self.config.enable_prefix_cache,
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
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        await asyncio.shield(self._close_task)

    async def _close(self) -> None:
        try:
            self.lifecycle.begin()
            # Pending HTTP/control rendezvous has no write grant. Retire it
            # explicitly; in-flight/adopted work still owns the drain barrier.
            for key in tuple(self._requests):
                if self.query_handoff(key).state == "REGISTERED":
                    await self.abort_handoff(key, "NODE_DRAINING", deterministic=True)
            await self.lifecycle.close()
            await self.progress.close()
            await self.core.call_pd_worker(OP_FINALIZE_TRANSFER)
        except BaseException as exc:
            self._health_error = f"OWNER_RETIREMENT_{type(exc).__name__}"
            self._record("RECOVERY_REQUIRED", error_code=self._health_error)
            # Do not falsely report SERVICE_STOPPED or discard owner identity.
            raise
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
        for task in tuple(self._external_handoff_tasks.values()):
            task.cancel()
        if self._external_handoff_tasks:
            await asyncio.gather(
                *tuple(self._external_handoff_tasks.values()),
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
            health=("RECOVERY_REQUIRED" if self._health_error else
                    "DRAINING" if self.lifecycle.draining else "READY"),
        )

    def capacity_snapshot(self) -> CapacitySnapshot:
        reservations = tuple(self.core.kv_cache_manager.group_cache_reservations)
        self._snapshot_sequence += 1
        self._refresh_metrics_gauges(reservations)
        return CapacitySnapshot(
            node_id=self.config.node_id,
            role=self.config.role.value,
            active_handoffs=len(self._active_keys),
            pending_requests=len(self._requests),
            reservations=len(reservations),
            quarantined_reservations=sum(
                reservation.state.value == "QUARANTINED"
                for reservation in reservations
            ),
            snapshot_sequence=self._snapshot_sequence,
            active_limit=(self.config.max_pending_handoffs if self.config.role is PDRole.DECODE
                          else self._admission.active_limit),
            queued_handoffs=self._admission.queued,
            inflight_transfer_bytes=self._inflight_transfer_bytes,
            inflight_transfer_byte_limit=0,
        )

    def _refresh_metrics_gauges(self, reservations) -> None:
        self.metrics.set_gauge("active_handoffs", len(self._active_keys))
        self.metrics.set_gauge("queued_handoffs", self._admission.queued)
        self.metrics.set_gauge("pending_requests", len(self._requests))
        self.metrics.set_gauge("reservations", len(reservations))
        self.metrics.set_gauge(
            "inflight_transfer_bytes",
            self._inflight_transfer_bytes,
        )

    async def metrics_snapshot(self) -> dict[str, object]:
        """Return Host metrics plus worker-resident model/transfer counters."""
        self._refresh_metrics_gauges(self.core.kv_cache_manager.group_cache_reservations)
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

    def register_generation(self, context: GenerationContext) -> NodeRequest:
        """Bind the original HTTP request before accepting P control messages."""
        self._assert_accepting()
        self._require_external_role(self.config.role)
        context.validate_node(self.config)
        key = context.key
        if key in self._requests or key in self._retired:
            raise ValueError("generation identity was already used")
        if len(self._requests) >= self.config.max_pending_handoffs:
            raise PDBackpressureError("node generation request limit reached")
        request = NodeRequest(
            context, asyncio.get_running_loop().create_future(), asyncio.Queue(maxsize=64),
            asyncio.current_task() if self.config.role is PDRole.PREFILL else None,
        )
        self._requests[key] = request
        self._request_available.set()
        return request

    def prepare_input(
        self, prompt: str, config: GenerateConfig, prompt_token_ids: Sequence[int], *,
        output_parser_spec: OutputParserSpec | None = None,
        constraint_spec: ConstraintSpec | None = None,
    ) -> PreparedInput:
        """Normalize once on P; no prepared-ID store or expiration protocol."""
        self._require_external_role(PDRole.PREFILL)
        self.config.model_adapter.validate_generate_config(config)
        if constraint_spec is not None:
            self.config.model_adapter.validate_constraint_spec(constraint_spec)
        continuation = self.config.model_adapter.build_continuation(
            config=config, prompt_token_ids=prompt_token_ids, eos_token_id=self.eos_token_id,
            output_parser_spec=output_parser_spec,
            **({"constraint_spec": constraint_spec} if constraint_spec else {}),
        )
        prefix = None
        if self.config.model_contract.supports_prefix_cache(PDRole.DECODE.value):
            prefix = self.config.model_adapter.build_prefix_match_spec(
                prompt_token_ids, self.core.kv_cache_manager,
            )
        return PreparedInput(
            prompt, config, tuple(int(token) for token in prompt_token_ids),
            continuation, prefix, prepared_request_hash(continuation, prefix),
        )

    def _reserve_handoff(self, message: BeginHandoff, session) -> ReserveAccepted | ReserveRejected:
        """Only the matched P session can reserve and learn writable D pages."""
        self._require_external_role(PDRole.DECODE)
        assert self.decode_connector is not None
        request = self._requests.get(message.key)
        if request is None or request.context != message.context:
            raise ValueError("P normalization has no matching Decode HTTP request")
        context = request.context
        context.validate_node(self.config)
        if (session.peer_hello.node_id != context.prefill_node_id
                or session.peer_hello.endpoint_generation != context.prefill_endpoint_generation):
            raise ValueError("P session differs from the assigned source")
        if request.accepted.done():
            raise ValueError("handoff was already accepted")
        continuation = message.continuation
        self.config.model_adapter.validate_continuation(continuation)
        constraint_digest = ""
        if continuation.constraint_spec is not None:
            try:
                constraint = ConstraintSpec.from_wire(continuation.constraint_spec)
                self.config.model_adapter.validate_constraint_spec(constraint)
                constraint_digest = constraint_spec_hash(continuation.constraint_spec)
            except (KeyError, TypeError, ValueError) as exc:
                return ReserveRejected(message.key, f"INVALID_CONSTRAINT_{type(exc).__name__}", False)
            if importlib.util.find_spec("xgrammar") is None:
                return ReserveRejected(message.key, "CONSTRAINT_PROVIDER_UNAVAILABLE", True)
        result = self.decode_connector.reserve(ReserveRequest(
            key=message.key,
            prompt_token_count=len(continuation.prompt_token_ids),
            max_new_tokens=continuation.max_new_tokens,
            layout_fingerprint=self.capabilities.layout_fingerprint,
            prepared_digest=message.prepared_digest,
            prefix_match_spec=message.prefix_match_spec,
            constraint_digest=constraint_digest,
        ))
        if isinstance(result, ReserveAccepted):
            self._active_keys.add(message.key)
            self.metrics.increment("prefix.d_requests")
            self.metrics.increment("prefix.d_hit_tokens", result.prefix_hit_tokens)
            self.metrics.set_gauge("prefix.d_last_hit_tokens", result.prefix_hit_tokens)
            self.metrics.increment("prefix.d_hit_requests" if result.prefix_hit_tokens else "prefix.d_cold_requests")
            self._record("HANDOFF_CREATED", key=message.key, state="CREATED")
            self._record("HANDOFF_RESERVED", key=message.key,
                         reservation_id=result.reservation_id, state="RESERVED")
            request.accepted.set_result(continuation)
        return result

    async def decode_outputs(self, request: NodeRequest) -> AsyncGenerator[TokenOutput, None]:
        """Expose ordinary engine outputs; public formatting stays in Serving."""
        key = request.context.key
        try:
            while True:
                output = await request.outputs.get()
                if isinstance(output, BaseException):
                    raise output
                yield output
                if output.finished:
                    return
        finally:
            if key in self._requests:
                status = self.query_handoff(key)
                await self.abort_handoff(
                    key, "DECODE_HTTP_CLOSED",
                    deterministic=status.state in ("REGISTERED", "RESERVED", "READY", "IN_USE", "COMPLETED", "ABORTED"),
                )

    def query_handoff(self, key: HandoffKey) -> HandoffStatus:
        retired = self._retired.get(key)
        if retired is not None:
            return retired
        if self.config.role is PDRole.DECODE:
            assert self.decode_connector is not None
            try:
                return self.decode_connector.query(key)
            except KeyError:
                if key in self._requests:
                    return HandoffStatus(key, "REGISTERED")
                raise
        record = self.coordinator.query(key)
        if record is None:
            if key in self._requests:
                return HandoffStatus(key, "REGISTERED")
            raise KeyError("unknown P handoff identity")
        return HandoffStatus(key, record.state.value, record.reservation_id,
                             record.manifest_hash, error_code=record.error_code)

    async def abort_handoff(self, key: HandoffKey, reason: str, *, deterministic: bool) -> HandoffStatus:
        """Retire only proven-safe work; HTTP cancellation is not native proof."""
        try:
            status = self.query_handoff(key)
        except KeyError:
            # Block a late HTTP request or BeginHandoff after abort won the race.
            status = HandoffStatus(key, "REGISTERED")
        if status.state in ("ABORTED", "COMPLETED", "RELEASED", "FAILED"):
            return status
        request = self._requests.get(key)
        if not deterministic:
            self._health_error = reason or "ROUTER_ABORT_UNCERTAIN"
            if self.config.role is PDRole.DECODE and status.state not in ("REGISTERED", "READY", "IN_USE"):
                self.decode_connector.abort(AbortHandoff(key, reason, False), deterministic=False)
            if self.config.role is PDRole.PREFILL:
                await self._cancel_prefill_task(request)
            self._finish_request(key, HandoffStatus(
                key, "RECOVERY_REQUIRED", status.reservation_id, status.manifest_hash, error_code=reason,
            ))
            return HandoffStatus(key, "RECOVERY_REQUIRED", status.reservation_id,
                                 status.manifest_hash, error_code=reason)
        if self.config.role is PDRole.DECODE:
            assert self.decode_connector is not None
            self._router_cancelled_keys.add(key)
            try:
                if status.state == "IN_USE":
                    await self.core.abort_request(key.request_id)
                if status.state in ("READY", "IN_USE"):
                    status = self.decode_connector.mark_decode_cancelled(key, error_code=reason)
                elif status.state != "REGISTERED":
                    status = self.decode_connector.abort(AbortHandoff(key, reason, True), deterministic=True)
                else:
                    status = HandoffStatus(key, "ABORTED", error_code=reason)
                if status.state == "ABORTED":
                    self._record("HANDOFF_ABORTED", key=key, reservation_id=status.reservation_id,
                                 state=status.state, error_code=reason)
                    self._finish_request(key, status)
                    await self._stop_decode_output_task(key)
                return status
            finally:
                self._router_cancelled_keys.discard(key)
        await self._cancel_prefill_task(request)
        if self._health_error:
            return HandoffStatus(key, "RECOVERY_REQUIRED", status.reservation_id,
                                 status.manifest_hash, error_code=self._health_error)
        record = self.coordinator.query(key)
        if record is not None and record.state is not CoordinatorState.RELEASED:
            self.coordinator.fail(key, reason)
        status = HandoffStatus(key, "RELEASED" if record is not None and
                               record.state is CoordinatorState.RELEASED else "ABORTED",
                               status.reservation_id, status.manifest_hash, error_code=reason)
        self._finish_request(key, status)
        return status

    @staticmethod
    async def _cancel_prefill_task(request: NodeRequest | None) -> None:
        if request is not None and request.task is not None and request.task is not asyncio.current_task():
            request.task.cancel()
            await asyncio.gather(request.task, return_exceptions=True)

    def _finish_request(self, key: HandoffKey, status: HandoffStatus) -> None:
        request = self._requests.pop(key, None)
        self._request_available.set()
        if status.state in ("QUARANTINED", "RECOVERY_REQUIRED"):
            self._active_keys.add(key)
            self._health_error = status.error_code or status.state
        else:
            self._active_keys.discard(key)
        self._retired[key] = status
        self._retired.move_to_end(key)
        while len(self._retired) > self._history_limit:
            self._retired.popitem(last=False)
        if request is not None and status.state != "COMPLETED":
            error = RuntimeError(status.error_code or status.state)
            if not request.accepted.done():
                request.accepted.set_result(error)
            with suppress(asyncio.QueueFull):
                request.outputs.put_nowait(error)

    def _assert_accepting(self):
        if self._closed or self.lifecycle.draining:
            raise RuntimeError("PD node is draining and no longer accepts new work")

    def _require_external_role(self, role: PDRole) -> None:
        if self.config.role is not role:
            raise ValueError(f"operation requires a {role.value} node")
        if self._health_error:
            raise RuntimeError(
                f"PD serving is fail-closed and requires restart: {self._health_error}"
            )

    async def prefill(self, context: GenerationContext, prepared: PreparedInput) -> PrefillResult:
        try:
            return await self.run_prefill_handoff(
                context.key.request_id, prepared.prompt, prepared.config, prepared.prompt_token_ids,
                context=context, prepared=prepared,
                output_parser_spec=prepared.continuation.output_parser_spec,
                constraint_spec=(ConstraintSpec.from_wire(prepared.continuation.constraint_spec)
                                 if prepared.continuation.constraint_spec is not None else None),
            )
        except PrefillRejected as exc:
            self._finish_request(context.key, HandoffStatus(context.key, "FAILED", error_code=exc.result.error_code))
            return exc.result
        except PDBackpressureError as exc:
            result = PrefillResult(context.key, context.request_fingerprint, "NOT_STARTED", type(exc).__name__)
            self._finish_request(context.key, HandoffStatus(context.key, "FAILED", error_code=result.error_code))
            return result
        except BaseException as exc:
            if not self._health_error:
                self._finish_request(context.key, HandoffStatus(
                    context.key, "FAILED", error_code=type(exc).__name__,
                ))
            raise

    async def _ensure_external_session(self, request: GenerationContext) -> None:
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
            if self._active_keys or self._admission.count:
                raise RuntimeError("a live handoff owns the current peer registration")
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
                self.lifecycle.bind(session)
                multiplex = MultiplexedPDControlSession(session, lifecycle=self.lifecycle)
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
        completion_facts: tuple[dict, ...] = (),
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
            completion_facts=completion_facts,
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
        context: GenerationContext,
        prepared: PreparedInput,
        output_parser_spec: OutputParserSpec | None = None,
        constraint_spec: ConstraintSpec | None = None,
    ) -> PrefillResult:
        """Run P Prefill and push chunks; D owns the public Decode response."""
        if self.config.role is not PDRole.PREFILL:
            raise ValueError("external generation is accepted only by the PD Prefill role")
        if self._health_error:
            raise RuntimeError(
                f"PD serving is fail-closed and requires restart: {self._health_error}"
            )
        self.config.model_adapter.validate_generate_config(config)
        if self.planner is None or self.source_lifecycle is None:
            raise RuntimeError("PD Prefill service has not started")

        try:
            await self._ensure_external_session(context)
        except Exception as exc:
            # No request compute or native write has been submitted here.
            raise PrefillRejected(PrefillResult(
                context.key, context.request_fingerprint, "NOT_STARTED", type(exc).__name__,
            )) from exc
        if self._multiplex_session is None:
            raise RuntimeError("PD Prefill has no Decode session")

        async with self._admission.admit():
            record = self.coordinator.register_handoff(
                context.key,
                prefill_node_id=self.config.node_id,
                decode_node_id=context.decode_node_id,
            )
            self._multiplex_session.open_route(record.key)
            self._active_keys.add(record.key)
            self._record("HANDOFF_CREATED", key=record.key, state=record.state.value)
            try:
                await self._send_control(BeginHandoff(
                    key=record.key, context=context, continuation=prepared.continuation,
                    prefix_match_spec=prepared.prefix_match_spec, prepared_digest=prepared.digest,
                ))
                reservation = await self._receive_control(record.key)
                if isinstance(reservation, ReserveRejected) and reservation.key == record.key:
                    raise PrefillRejected(PrefillResult(
                        record.key, context.request_fingerprint, "NOT_STARTED",
                        reservation.reason, reservation.retryable,
                    ))
                if (isinstance(reservation, ControlError) and reservation.key == record.key
                        and reservation.operation == "BeginHandoff" and reservation.deterministic):
                    raise PrefillRejected(PrefillResult(
                        record.key, context.request_fingerprint, "NOT_STARTED", reservation.error_code,
                    ))
                if not isinstance(reservation, ReserveAccepted) or reservation.key != record.key:
                    raise RuntimeError("D did not accept the matching handoff")
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
                if isinstance(exc, Exception) and not isinstance(exc, PrefillRejected):
                    raise PrefillRejected(PrefillResult(
                        record.key, context.request_fingerprint, "NOT_STARTED", type(exc).__name__,
                    )) from exc
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
                    constraint_spec,
                )
            )
            guard = None
            transfer_started = False
            d_ready = False
            remote_aborted = False
            source_prefix_hit_tokens: int | None = None
            source_partition: int | None = None
            rank_mapping = ()
            try:
                while True:
                    chunk = await self._next_chunk_or_failure(request_id, prefill_task)
                    if chunk.chunk_id == 0:
                        if source_prefix_hit_tokens is not None:
                            raise RuntimeError("P produced more than one first chunk")
                        source_prefix_hit_tokens = chunk.start_token
                        source_partition = chunk.cache_partition
                        rank_mapping = self.config.model_adapter.rank_mapping(
                            self.local_bundle.topology, source_partition, reservation.partition,
                        )
                        logger.info(
                            "pd_rank_mapping request_id=%s source_partition=%s destination_partition=%s pairs=%s",
                            request_id, source_partition, reservation.partition,
                            tuple((pair.source_rank_id, pair.destination_rank_id) for pair in rank_mapping),
                        )
                        if (
                            not self.config.enable_prefix_cache
                            and source_prefix_hit_tokens
                        ):
                            raise RuntimeError(
                                "P reported a hit while local prefix caching is disabled"
                            )
                        self.metrics.increment("prefix.p_requests")
                        self.metrics.increment(
                            "prefix.p_hit_tokens", source_prefix_hit_tokens
                        )
                        self.metrics.set_gauge(
                            "prefix.p_last_hit_tokens", source_prefix_hit_tokens
                        )
                        self.metrics.increment(
                            "prefix.p_hit_requests"
                            if source_prefix_hit_tokens
                            else "prefix.p_cold_requests"
                        )
                        transfer_start_token = 0
                    else:
                        if source_prefix_hit_tokens is None:
                            raise RuntimeError("P produced a non-initial chunk first")
                        transfer_start_token = chunk.start_token
                    assert source_prefix_hit_tokens is not None
                    if chunk.cache_partition != source_partition:
                        raise RuntimeError("P cache partition changed during the handoff")
                    source_tables = {
                        pair.source_rank_id: chunk.block_ids_by_group for pair in rank_mapping
                    }
                    destination_tables = {
                        pair.destination_rank_id: reservation.block_ids_by_group for pair in rank_mapping
                    }
                    plan = self.planner.plan_chunk(
                        record.key,
                        chunk_id=chunk.chunk_id,
                        start_token=transfer_start_token,
                        end_token=chunk.end_token,
                        final=chunk.final,
                        rank_mapping=rank_mapping,
                        source_blocks_by_rank=source_tables,
                        destination_blocks_by_rank=destination_tables,
                        destination_prefix_hit_tokens=(
                            reservation.prefix_hit_tokens
                        ),
                        source_prefix_hit_tokens=source_prefix_hit_tokens,
                    )
                    guard = self.source_lifecycle.begin(
                        request_id,
                        chunk.chunk_id,
                        chunk.block_ids_by_group,
                        chunk.cache_partition,
                    )
                    continuation = prepared.continuation if chunk.final else None
                    manifest = self.config.model_adapter.build_manifest(
                        key=record.key,
                        plan=plan,
                        chunk=chunk,
                        continuation=continuation,
                        prepared_digest=prepared.digest,
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
                        metadata_hash=manifest.metadata_hash,
                    )
                    await self._send_control(commit)
                    ready = await self._receive_control(record.key)
                    if isinstance(ready, ControlError):
                        if ready.deterministic and ready.error_code == "INVALID_DECODE_SEED":
                            # D checked this seed only after every write completed.
                            # Retire the source pin with that known certainty and
                            # expose a safe failure, never a no-compute assertion.
                            self.source_lifecycle.settle(guard, certainty)
                            guard = None
                            raise PrefillRejected(PrefillResult(
                                record.key, context.request_fingerprint, "FAILED", ready.error_code,
                            ))
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
                    result = PrefillResult(record.key, context.request_fingerprint, "READY")
                    self._finish_request(record.key, HandoffStatus(
                        record.key, "RELEASED", ready.reservation_id, ready.manifest_hash,
                    ))
                    self._active_keys.discard(record.key)
                    if self._multiplex_session is not None:
                        self._multiplex_session.close_route(record.key)
                    return result
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
        self._inflight_transfer_bytes += transfer_bytes
        try:
            if release_next_chunk:
                self.metrics.increment("overlap.submitted")
            submit_started = time.monotonic_ns()
            raw_submission = await self.core.call_pd_worker(
                OP_SUBMIT_TRANSFER_CHUNK,
                encode_worker_payload(request),
            )
            submission = decode_worker_payload(raw_submission, TransferSubmission)
            self.metrics.observe_ns("transfer.submit_rpc", time.monotonic_ns() - submit_started)
            if release_next_chunk:
                self.core.complete_prefill_chunk_transfer(request_id)
            poll = await self.progress.wait(
                submission.job_id, timeout=self.config.request_timeout_seconds + 10,
            )
            if release_next_chunk:
                self.metrics.increment("overlap.completed")
            return TransferChunkResponse(poll.results), release_next_chunk
        finally:
            self._inflight_transfer_bytes -= transfer_bytes
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

    def _drain_prefill_request(
        self,
        request_id: str,
        prompt: str,
        config,
        prompt_token_ids: Sequence[int],
        constraint_spec: ConstraintSpec | None = None,
    ):
        stream = self.core.add_request(
            request_id,
            prompt,
            config,
            prompt_token_ids=prompt_token_ids,
            **({"constraint_spec": constraint_spec} if constraint_spec else {}),
        )

        async def drain():
            async with aclosing(stream):
                async for output in stream:
                    if output.finish_reason not in ("", "FINISHED_PREFILL"):
                        raise RuntimeError(f"P Prefill ended unexpectedly: {output.finish_reason}")

        return drain()

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
        bundle = TransferResultBundle(results[0].key, results[0].chunk_id, results)
        certainties = [CompletionCertainty(result.certainty) for result in results]
        started = time.perf_counter_ns()
        await self._send_expect_ack(bundle, "transfer_results", bundle.chunk_id)
        self.metrics.observe_ns("completion_ack", time.perf_counter_ns() - started)
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

    async def _serve_external_connections(
        self,
        local_advertisement: RegistryAdvertisement,
    ) -> None:
        assert self._control_acceptor is not None
        assert self.capabilities is not None
        while True:
            while not self._requests:
                self._request_available.clear()
                await self._request_available.wait()
            expected_peer = next(iter(self._requests.values())).context.prefill_node_id
            session = await self._control_acceptor.accept(
                self.capabilities,
                expected_peer_node_id=expected_peer,
            )
            try:
                peer = await session.exchange_registry(local_advertisement)
                self.lifecycle.bind(session)
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
            if self._external_handoff_tasks.get(key) is task:
                self._external_handoff_tasks.pop(key)
            if not task.cancelled():
                error = task.exception()
                if error is not None and not self._health_error:
                    self._health_error = type(error).__name__
                    self._finish_request(key, HandoffStatus(key, "RECOVERY_REQUIRED", error_code=self._health_error))

        try:
            while True:
                message = await session.receive()
                if isinstance(message, (DrainPeer, SenderReleased, SenderReleaseAck)):
                    await self.lifecycle.receive(message)
                    continue
                key = getattr(message, "key", None)
                if not isinstance(key, HandoffKey):
                    raise ValueError("external PD message has no handoff correlation")
                if isinstance(message, BeginHandoff):
                    if key in tasks:
                        await session.send(
                            ControlError(
                                key,
                                "BeginHandoff",
                                "ROUTE_ALREADY_OPEN",
                                True,
                            )
                        )
                        continue
                    states = []
                    for active_key in tasks:
                        try:
                            states.append(self.query_handoff(active_key).state)
                        except KeyError:
                            states.append("WAITING_HTTP")
                    transferring = sum(state in ("WAITING_HTTP", "REGISTERED", "RESERVED", "TRANSFERRING")
                                       for state in states)
                    # Completed routes may still be finishing their coroutine.
                    # The reservation state, not callback scheduling, owns quota.
                    live = sum(state not in ("COMPLETED", "ABORTED", "FAILED", "RELEASED") for state in states)
                    if (transferring >= self.config.max_active_handoffs
                            or live >= self.config.max_pending_handoffs):
                        await session.send(
                            ControlError(
                                key,
                                "BeginHandoff",
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
                    self._external_handoff_tasks[key] = task
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
                    self._finish_request(key, HandoffStatus(key, "ABORTED", error_code="CONTROL_SESSION_EOF"))
                    task.cancel()
                    cancelled.append(task)
                    continue
                if status.state in ("READY", "IN_USE", "COMPLETED", "ABORTED"):
                    continue
                deterministic = status.state == "RESERVED"
                with suppress(Exception):
                    aborted = self.decode_connector.abort(
                        AbortHandoff(
                            key,
                            "CONTROL_SESSION_EOF",
                            deterministic,
                        ),
                        deterministic=deterministic,
                    )
                    self._finish_request(key, aborted)
                if not deterministic:
                    self._health_error = "CONTROL_SESSION_EOF_DURING_TRANSFER"
                task.cancel()
                cancelled.append(task)
            if cancelled:
                await asyncio.gather(*cancelled, return_exceptions=True)

    async def _wait_generation(self, context: GenerationContext) -> None:
        """P control and D HTTP may arrive in either order, including on reuse."""
        async def wait():
            while context.key not in self._requests:
                self._require_external_role(PDRole.DECODE)
                if context.key in self._retired:
                    raise ValueError("generation identity was retired")
                self._request_available.clear()
                await self._request_available.wait()
        await asyncio.wait_for(wait(), self.config.request_timeout_seconds)

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
                if isinstance(message, BeginHandoff):
                    if opened:
                        raise ValueError("route opening was replayed on an active route")
                    await self._wait_generation(message.context)
                    outcome = self._reserve_handoff(message, session)
                    await session.send(outcome)
                    if isinstance(outcome, ReserveRejected):
                        self._finish_request(key, HandoffStatus(key, "FAILED", error_code=outcome.reason))
                        return
                    opened = True
                elif not opened:
                    raise ValueError("handoff command arrived before BeginHandoff")
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
                elif isinstance(message, TransferResultBundle):
                    self.decode_connector.validate_transfer_results(message)
                    status = self.decode_connector.query(key)
                    self._record(
                        "TRANSFER_RESULTS_RECORDED",
                        key=key,
                        reservation_id=status.reservation_id,
                        chunk_id=message.chunk_id,
                        state=status.state,
                        completion_facts=tuple(msgspec.to_builtins(result) for result in message.results),
                    )
                    self.decode_connector.record_transfer_results(message)
                    await session.send(ControlAck(key, "transfer_results", message.chunk_id))
                elif isinstance(message, CommitRequest):
                    ready = self.decode_connector.commit(message)
                    try:
                        await self._validate_decode_seed(key)
                    except ValueError as exc:
                        logger.warning("PD Decode seed rejected: request=%s token=%s reason=%s",
                                       key.request_id, message.first_token, str(exc)[:512])
                        # Commit proves all native writes completed. No Decode
                        # was admitted yet, so a bad seed is safely recyclable.
                        status = self.decode_connector.mark_decode_cancelled(
                            key, error_code="INVALID_DECODE_SEED",
                        )
                        self._record(
                            "DECODE_SEED_REJECTED", key=key,
                            reservation_id=status.reservation_id,
                            state=status.state, error_code=type(exc).__name__,
                        )
                        self._finish_request(key, status)
                        await session.send(ControlError(
                            key, "CommitRequest", "INVALID_DECODE_SEED", True,
                        ))
                        return
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
                    self._finish_request(key, status)
                    return
                else:
                    raise ValueError(
                        f"unexpected P control message {type(message).__name__}"
                    )
            except (ValueError, KeyError, RuntimeError, TimeoutError) as exc:
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
                self._finish_request(key, HandoffStatus(
                    key, "FAILED" if deterministic else "RECOVERY_REQUIRED",
                    error_code=type(exc).__name__,
                ))
                if not deterministic:
                    self._health_error = type(exc).__name__
                return

    async def _validate_decode_seed(self, key) -> None:
        manifest = self.decode_connector.committed_manifest(key)
        continuation = manifest.continuation
        if continuation is None or manifest.first_token is None:
            raise ValueError("committed handoff has no continuation metadata")
        self.config.model_adapter.validate_continuation(continuation)
        if continuation.constraint_spec is not None:
            if self.preflight_constraint is None:
                raise ValueError("Decode constraint preflight is not installed")
            await self.preflight_constraint(
                ConstraintSpec.from_wire(continuation.constraint_spec), (manifest.first_token,),
            )

    async def _stream_decode(self, key) -> None:
        assert self.decode_connector is not None
        manifest = self.decode_connector.committed_manifest(key)
        continuation = manifest.continuation
        if continuation is None or manifest.first_token is None:
            raise RuntimeError("committed handoff has no continuation metadata")
        self.config.model_adapter.validate_continuation(continuation)
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
        request = self._requests[key]
        try:
            outputs = self.config.model_adapter.adopt_decode(
                self.core,
                reservation_id=self.decode_connector.query(key).reservation_id,
                request_id=key.request_id,
                first_token=manifest.first_token,
                continuation=continuation,
            )
            async with aclosing(outputs):
                async for output in outputs:
                    if output.finished:
                        await outputs.aclose()
                        self.decode_connector.mark_decode_completed(key)
                        completed = self.decode_connector.query(key)
                        self._record(
                            "HANDOFF_COMPLETED",
                            key=key,
                            reservation_id=completed.reservation_id,
                            manifest_hash=completed.manifest_hash,
                            state="COMPLETED",
                        )
                        self._finish_request(key, completed)
                        self.metrics.record_terminal(
                            request_id=key.request_id,
                            state="COMPLETED",
                            prompt_tokens=output.prompt_tokens,
                            completion_tokens=output.completion_tokens,
                            token_ids=output.token_ids,
                        )
                        await request.outputs.put(output)
                        return
                    await request.outputs.put(output)
        except BaseException as exc:
            router_cancelled = key in self._router_cancelled_keys
            if not router_cancelled:
                # The HTTP abort path can publish the durable ABORTED tombstone
                # immediately before this generator observes the scheduler
                # cancellation.  Accept either side of that race as the same
                # deterministic terminal fact.
                with suppress(KeyError, RuntimeError):
                    current_state = self.decode_connector.query(key).state
                    router_cancelled = current_state == "ABORTED" or (
                        current_state == "COMPLETED" and isinstance(exc, (asyncio.CancelledError, GeneratorExit))
                    )
            if router_cancelled:
                # The Router's deterministic abort endpoint owns the terminal
                # fact and releases the reservation after the scheduler has
                # confirmed this request stopped.  The resulting Decode-loop
                # exception is expected convergence, not an UNKNOWN failure.
                self._active_keys.discard(key)
                return
            logger.exception(
                "PD Decode continuation failed: request=%s handoff=%s error=%s",
                key.request_id,
                key.handoff_id,
                type(exc).__name__,
            )
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
            with suppress(asyncio.QueueFull):
                request.outputs.put_nowait(exc)
            raise

    async def _stop_decode_output_task(self, key: HandoffKey) -> None:
        # Called only after scheduler cancellation and a deterministic terminal
        # fact. A slow/disconnected Router must not leave a producer in put().
        task = self._external_handoff_tasks.get(key)
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
