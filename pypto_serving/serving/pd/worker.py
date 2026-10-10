# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Worker-owned transfer lifecycle, independent of model tensor layouts."""

from __future__ import annotations

from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


class PDWorkerRuntime:
    def __init__(self, config, model):
        self.model = model
        self._pd_worker_config = config
        self._pd_owner_bridges = ()
        self._pd_owner_leases = ()
        self._pd_owner_envelopes = ()
        self._pd_peer_leases = ()
        self._pd_transfer_agents = ()
        self._pd_transfer_jobs = OrderedDict()
        self._pd_attempt_sequences = [0] * model.ranks
        self._pd_peer_registration_request = None
        self._pd_rank_pairs = frozenset()

    def worker_options(self):
        from pypto_serving.transfer.owner import OwnerBridge, OwnerBridgeGroup
        from pypto_serving.transfer.types import OwnerRef

        owners = tuple(
            OwnerRef(
                self._pd_worker_config.run_id,
                rank,
                self._pd_worker_config.generation,
                self._pd_worker_config.endpoint_generation,
                self._pd_worker_config.worker_id,
            )
            for rank in range(self.model.ranks)
        )
        bridges = tuple(OwnerBridge(owner, self._pd_worker_config.transfer_hostname) for owner in owners)
        factories = OwnerBridgeGroup(bridges).factories
        self._pd_owner_bridges = bridges
        return {"chip_service_factories": factories}

    def worker_ready(self):
        for bridge in self._pd_owner_bridges:
            bridge.ready()

    def handle_pd_command(self, operation: str, payload: bytes) -> bytes:
        """Execute owner-local registry/install/transfer operations in FIFO order."""
        from pypto_serving.serving.pd.worker_api import (  # noqa: PLC0415
            OP_INSPECT_REGISTRY,
            OP_INSPECT_RUNTIME_METRICS,
            OP_INSTALL_PEER,
            OP_POLL_TRANSFER_CHUNK,
            OP_PREPARE_REGISTRY,
            OP_SUBMIT_TRANSFER_CHUNK,
            InstallPeerRequest,
            TransferPollRequest,
            TransferChunkRequest,
            WorkerRuntimeMetrics,
            decode_worker_payload,
            encode_worker_payload,
        )

        if operation == OP_PREPARE_REGISTRY:
            if payload:
                raise ValueError("prepare_registry does not accept a payload")
            return encode_worker_payload(self._prepare_pd_registry())
        if operation == OP_INSPECT_REGISTRY:
            if payload:
                raise ValueError("inspect_registry does not accept a payload")
            if not self._pd_owner_envelopes:
                raise RuntimeError("PD registry has not been prepared")
            return encode_worker_payload(self._pd_registry_bundle())
        if operation == OP_INSPECT_RUNTIME_METRICS:
            if payload:
                raise ValueError("inspect_runtime_metrics does not accept a payload")
            values = self.model.metrics()
            values["pd_transfer_jobs"] = float(len(self._pd_transfer_jobs))
            values["pd_transfer_jobs_active"] = float(
                sum(job["response"] is None for job in self._pd_transfer_jobs.values())
            )
            return encode_worker_payload(WorkerRuntimeMetrics(values))
        if operation == OP_INSTALL_PEER:
            request = decode_worker_payload(payload, InstallPeerRequest)
            self._install_pd_peer(request)
            return b""
        if operation == OP_SUBMIT_TRANSFER_CHUNK:
            request = decode_worker_payload(payload, TransferChunkRequest)
            return encode_worker_payload(self._submit_pd_chunk(request))
        if operation == OP_POLL_TRANSFER_CHUNK:
            request = decode_worker_payload(payload, TransferPollRequest)
            return encode_worker_payload(self._poll_pd_chunk(request.job_id))
        raise ValueError(f"unknown PD worker operation {operation!r}")

    def set_transfer_profile_active(self, active: bool) -> None:
        """Synchronize SA profiling with all owner-local transfer threads."""
        if type(active) is not bool:
            raise ValueError("transfer profile state must be boolean")
        if not self._pd_owner_bridges:
            return
        failures = []
        for bridge in self._pd_owner_bridges:
            try:
                bridge.set_profile_active(active)
            except Exception as exc:
                failures.append(f"rank {bridge.owner.rank_id}: {exc}")
        if failures and active:
            for bridge in self._pd_owner_bridges:
                try:
                    bridge.set_profile_active(False)
                except Exception:
                    pass
        if failures:
            raise RuntimeError("owner transfer profiler control failed: " + "; ".join(failures))

    def _prepare_pd_registry(self):
        """Register resident regions in their owning chip children."""
        import json  # noqa: PLC0415

        from pypto_serving.serving.pd.protocol import (  # noqa: PLC0415
            RankRegistration,
            RegionRegistration,
        )
        from pypto_serving.transfer.types import RegionLease  # noqa: PLC0415

        if self._pd_worker_config is None or not self._pd_owner_bridges:
            raise RuntimeError("PD owner services were not configured before worker creation")
        if self._pd_owner_envelopes:
            return self._pd_registry_bundle()
        worker = self.model.worker()
        cache = self.model.regions()
        leases_by_rank = []
        registrations = []
        try:
            for rank, bridge in enumerate(self._pd_owner_bridges):
                registry = self.model.registry(
                    rank,
                    model_revision=self._pd_worker_config.model_revision,
                )
                leases = {}
                regions = []
                for component_id, tensor in cache.items():
                    entry = registry.entry(component_id)
                    lease = RegionLease(
                        bridge.owner,
                        component_id,
                        self._pd_worker_config.generation,
                        entry.extent,
                    )
                    envelope = bridge.register_tensor(
                        worker,
                        tensor.shards[rank],
                        lease,
                    )
                    opaque = json.dumps(
                        envelope,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode()
                    leases[component_id] = lease
                    regions.append(
                        RegionRegistration(
                            component_id=component_id,
                            lease=lease.lease,
                            extent=lease.extent,
                            provider_envelope=opaque,
                        )
                    )
                leases_by_rank.append(leases)
                registrations.append(
                    RankRegistration(
                        rank_id=rank,
                        owner_generation=bridge.owner.generation,
                        endpoint_generation=bridge.owner.endpoint_generation,
                        worker_id=bridge.owner.worker_id,
                        regions=tuple(regions),
                    )
                )
        except BaseException:
            # No peer has received these envelopes yet, so confirmed local
            # unregister is sufficient and leaves this generation retryable.
            for bridge in self._pd_owner_bridges:
                bridge.release()
            raise
        self._pd_owner_leases = tuple(leases_by_rank)
        self._pd_owner_envelopes = tuple(registrations)
        return self._pd_registry_bundle()

    def _pd_registry_bundle(self):
        from pypto_serving.serving.pd.worker_api import (  # noqa: PLC0415
            ComponentGeometry,
            WorkerRegistryBundle,
        )

        if self._pd_worker_config is None or not self._pd_owner_envelopes:
            raise RuntimeError("PD registry has not been prepared")
        registry = self.model.registry(
            0,
            model_revision=self._pd_worker_config.model_revision,
        )
        components = tuple(
            ComponentGeometry(
                component_id=entry.component_id,
                dtype=entry.dtype,
                item_bytes=entry.item_bytes,
                layers=entry.layers,
                blocks_per_layer=entry.blocks_per_layer,
                block_tokens=entry.block_tokens,
                token_stride_bytes=entry.token_stride_bytes,
                extent=entry.extent,
            )
            for entry in registry.components
        )
        return WorkerRegistryBundle(
            model_revision=registry.model_revision,
            topology=registry.topology,
            registry_fingerprint=registry.fingerprint,
            layout_fingerprint=registry.layout_fingerprint,
            components=components,
            ranks=self._pd_owner_envelopes,
        )

    def _install_pd_peer(self, request) -> None:
        from pypto_serving.transfer.agent import TransferAgent  # noqa: PLC0415
        from pypto_serving.transfer.supervisor import OwnerSupervisor  # noqa: PLC0415
        from pypto_serving.transfer.types import OwnerRef, RegionLease  # noqa: PLC0415
        import json  # noqa: PLC0415

        local = self._prepare_pd_registry()
        if request.layout_fingerprint != local.layout_fingerprint:
            raise ValueError("peer registry layout fingerprint mismatch")
        if len(request.ranks) != self.model.ranks:
            raise ValueError("peer registry rank count mismatch")
        if self._pd_peer_leases:
            if request != self._pd_peer_registration_request:
                raise ValueError("peer registry changed without a new worker generation")
            return

        peer_leases = []
        peer_envelopes = []
        expected_components = set(self.model.component_ids)
        for rank, registration in enumerate(request.ranks):
            if registration.rank_id != rank:
                raise ValueError("peer registrations must be rank ordered")
            owner = OwnerRef(
                self._pd_worker_config.run_id,
                rank,
                registration.owner_generation,
                registration.endpoint_generation,
                registration.worker_id,
            )
            regions = {region.component_id: region for region in registration.regions}
            if set(regions) != expected_components:
                raise ValueError("peer registration does not contain all model regions")
            leases = {}
            envelopes = {}
            registry = self.model.registry(
                rank,
                model_revision=self._pd_worker_config.model_revision,
            )
            for component_id in self.model.component_ids:
                region = regions[component_id]
                entry = registry.entry(component_id)
                per_layer_extent = len(entry.layers) * entry.block_stride_bytes
                peer_blocks, remainder = divmod(region.extent, per_layer_extent)
                if remainder or not peer_blocks:
                    raise ValueError("peer region extent differs from transfer geometry")
                lease = RegionLease(owner, component_id, region.lease, region.extent)
                envelope = json.loads(region.provider_envelope)
                leases[component_id] = lease
                envelopes[component_id] = envelope
            peer_leases.append(leases)
            peer_envelopes.append(envelopes)

        partition_count = local.topology[0] // local.topology[1]
        rank_pairs = frozenset(
            pair
            for source_partition in range(partition_count)
            for destination_partition in range(partition_count)
            for pair in self.model.rank_mapping(local.topology, source_partition, destination_partition)
        )
        for source_rank, bridge in enumerate(self._pd_owner_bridges):
            destinations = sorted(pair.destination_rank_id for pair in rank_pairs if pair.source_rank_id == source_rank)
            bridge.install_destinations(tuple(
                (peer_leases[rank][component], peer_envelopes[rank][component])
                for rank in destinations for component in self.model.component_ids
            ))

        local_owners = tuple(bridge.owner for bridge in self._pd_owner_bridges)
        peer_owners = tuple(next(iter(leases.values())).owner for leases in peer_leases)
        processes = tuple(bridge.process_handle for bridge in self._pd_owner_bridges)
        agents = []
        for rank, bridge in enumerate(self._pd_owner_bridges):
            supervisor = OwnerSupervisor(
                bridge.owner,
                processes,
                emit=lambda event: logger.error(
                    "PD transfer recovery event: stage=%s owner=%s parent_event_id=%s recovery_set=%s",
                    event.stage,
                    event.owner,
                    event.parent_event_id,
                    event.recovery_set,
                ),
                local_owners=local_owners,
                peer_owners=peer_owners,
            )
            agents.append(
                TransferAgent(
                    bridge.owner,
                    bridge,
                    poison=supervisor.poison,
                    max_tasks=self._pd_worker_config.max_active_handoffs,
                )
            )
        self._pd_peer_leases = tuple(peer_leases)
        self._pd_transfer_agents = tuple(agents)
        self._pd_peer_registration_request = request
        self._pd_rank_pairs = rank_pairs

    def _next_pd_attempt_sequence(self, rank: int) -> int:
        sequence = self._pd_attempt_sequences[rank] + 1
        self._pd_attempt_sequences[rank] = sequence
        return sequence

    def _submit_pd_chunk(self, request):
        from pypto_serving.serving.pd.protocol import validate_rank_mapping  # noqa: PLC0415
        from pypto_serving.serving.pd.worker_api import TransferSubmission  # noqa: PLC0415
        from pypto_serving.transfer.agent import AlreadyCompletedFence  # noqa: PLC0415
        from pypto_serving.transfer.types import TransferAttemptRef  # noqa: PLC0415

        if not self._pd_peer_leases or not self._pd_transfer_agents:
            raise RuntimeError("peer owner registry must be installed before transfer")
        manifest = request.manifest
        if request.destination_fingerprint != self._pd_peer_registration_request.registry_fingerprint:
            raise ValueError("destination registry fingerprint mismatch")
        validate_rank_mapping(manifest.rank_mapping)
        if not set(manifest.rank_mapping) <= self._pd_rank_pairs:
            raise ValueError("source/destination rank mapping is not installed")
        source_by_destination = {
            pair.destination_rank_id: pair.source_rank_id for pair in manifest.rank_mapping
        }
        if set(source_by_destination) != set(manifest.copies_by_destination_rank):
            raise ValueError("rank mapping differs from destination page copies")
        job_id = f"{manifest.key.handoff_id}-c{manifest.chunk_id}-a{request.attempt_ordinal}"
        existing = self._pd_transfer_jobs.get(job_id)
        if existing is not None:
            if existing["request"] != request:
                raise ValueError("PD transfer job id was replayed with different input")
            self._pd_transfer_jobs.move_to_end(job_id)
            return TransferSubmission(job_id)
        job_limit = max(32, self._pd_worker_config.max_active_handoffs * 16)
        while len(self._pd_transfer_jobs) >= job_limit:
            completed_id = next(
                (
                    known_id
                    for known_id, known in self._pd_transfer_jobs.items()
                    if known["response"] is not None
                ),
                None,
            )
            if completed_id is None:
                break
            self._pd_transfer_jobs.pop(completed_id)
        if len(self._pd_transfer_jobs) >= job_limit:
            raise RuntimeError("PD transfer job index reached its configured bound")
        units_by_rank = {}
        for unit in manifest.expected_units:
            units_by_rank.setdefault(unit.destination_rank_id, {})[unit.component_id] = unit
        if set(units_by_rank) != set(manifest.copies_by_destination_rank):
            raise ValueError("manifest rank completion set differs from page copies")

        ranks = []
        for destination_rank in sorted(manifest.copies_by_destination_rank):
            source_rank = source_by_destination[destination_rank]
            copies = self.model.copies(manifest.copies_by_destination_rank[destination_rank])
            registry = self.model.registry(
                source_rank,
                model_revision=self._pd_worker_config.model_revision,
            )
            actual_bytes = {
                component_id: sum(
                    registry.entry(component_id).block_stride_bytes
                    for copy in copies
                    if copy.component_id == component_id
                )
                for component_id in units_by_rank[destination_rank]
            }
            expected_bytes = {
                component_id: unit.nbytes for component_id, unit in units_by_rank[destination_rank].items()
            }
            if actual_bytes != expected_bytes:
                raise ValueError("manifest byte accounting differs from registry lowering")
            plan_id = f"{manifest.key.handoff_id}-c{manifest.chunk_id}-s{source_rank}-d{destination_rank}"
            attempt_id = f"{plan_id}-a{request.attempt_ordinal}"
            task = None
            if copies:
                attempt = TransferAttemptRef(
                    manifest.key.request_id,
                    plan_id,
                    manifest.key.handoff_id,
                    attempt_id,
                    manifest.key.data_generation,
                    manifest.key.route_epoch,
                    manifest.chunk_id,
                    self._pd_owner_bridges[source_rank].owner,
                    next(iter(self._pd_peer_leases[destination_rank].values())).owner,
                    registry.manifest_hash(copies, manifest.final),
                    attempt_sequence=self._next_pd_attempt_sequence(source_rank),
                )
                task = registry.lower(
                    attempt,
                    copies,
                    destination_layout_fingerprint=(self._pd_peer_registration_request.layout_fingerprint),
                    source_leases=self._pd_owner_leases[source_rank],
                    destination_leases=self._pd_peer_leases[destination_rank],
                    final=manifest.final,
                )
            ranks.append(
                {
                    "source_rank": source_rank,
                    "destination_rank": destination_rank,
                    "components": tuple(units_by_rank[destination_rank]),
                    "attempt_id": attempt_id,
                    "task": task,
                    "future": None,
                    "submit_error": None,
                }
            )
        # Validate/lower every rank before starting any native writes. Then
        # enqueue all ranks without waiting for individual completions.
        stop_error = None
        for rank in ranks:
            task = rank.pop("task")
            if task is None:
                continue
            if stop_error is None:
                try:
                    rank["future"] = self._pd_transfer_agents[rank["source_rank"]].submit(
                        task, AlreadyCompletedFence(), timeout=request.timeout_seconds,
                    )
                except BaseException as exc:
                    stop_error = exc
            rank["submit_error"] = stop_error if rank["future"] is None else None
        self._pd_transfer_jobs[job_id] = {
            "request": request,
            "ranks": tuple(ranks),
            "response": None,
        }
        return TransferSubmission(job_id)

    def _poll_pd_chunk(self, job_id):
        from pypto_serving.serving.pd.protocol import TransferResult  # noqa: PLC0415
        from pypto_serving.serving.pd.worker_api import TransferPollResponse  # noqa: PLC0415
        from pypto_serving.transfer.errors import TransferFailure  # noqa: PLC0415
        from pypto_serving.transfer.types import CompletionCertainty  # noqa: PLC0415

        job = self._pd_transfer_jobs.get(job_id)
        if job is None:
            raise ValueError("unknown PD transfer job")
        response = job["response"]
        if response is not None:
            self._pd_transfer_jobs.move_to_end(job_id)
            return response
        if any(rank["future"] is not None and not rank["future"].done() for rank in job["ranks"]):
            return TransferPollResponse(job_id, False)
        manifest = job["request"].manifest
        results = []
        for rank in job["ranks"]:
            certainty = CompletionCertainty.COMPLETED
            error_code = ""
            future = rank["future"]
            submit_error = rank["submit_error"]
            if submit_error is not None:
                if isinstance(submit_error, TransferFailure):
                    certainty = submit_error.error.certainty
                    error_code = submit_error.error.code.value
                else:
                    certainty = CompletionCertainty.NOT_SUBMITTED
                    error_code = type(submit_error).__name__
            elif future is not None:
                try:
                    event = future.result()
                    certainty = event.certainty
                    error_code = event.error.code.value if event.error is not None else ""
                except BaseException as exc:
                    # A progress/supervision exception after a task was accepted
                    # cannot prove that the native writer never started.
                    certainty = CompletionCertainty.UNKNOWN
                    error_code = type(exc).__name__
            for component_id in rank["components"]:
                results.append(
                    TransferResult(
                        key=manifest.key,
                        chunk_id=manifest.chunk_id,
                        source_rank_id=rank["source_rank"],
                        destination_rank_id=rank["destination_rank"],
                        component_id=component_id,
                        attempt_id=rank["attempt_id"],
                        certainty=certainty.value,
                        error_code=error_code,
                    )
                )
        response = TransferPollResponse(job_id, True, tuple(results))
        job["response"] = response
        self._pd_transfer_jobs.move_to_end(job_id)
        return response

    def close(self) -> None:
        """Stop transfer threads and unregister arenas before allocator teardown."""
        if not self._pd_owner_bridges:
            return
        close_error: BaseException | None = None
        try:
            for agent in self._pd_transfer_agents:
                try:
                    agent.close()
                except BaseException as exc:
                    if close_error is None:
                        close_error = exc
            for bridge in self._pd_owner_bridges:
                try:
                    bridge.release()
                except BaseException as exc:
                    if close_error is None:
                        close_error = exc
                try:
                    bridge.close()
                except BaseException as exc:
                    if close_error is None:
                        close_error = exc
        finally:
            self._pd_transfer_agents = ()
            self._pd_transfer_jobs = OrderedDict()
            self._pd_peer_leases = ()
            self._pd_peer_registration_request = None
            self._pd_owner_envelopes = ()
            self._pd_owner_leases = ()
            self._pd_owner_bridges = ()
        if close_error is not None:
            raise close_error


class PDWorkerServices:
    """Install PD handlers only in explicitly configured worker processes."""

    def __init__(self, extensions, request_info):
        if len(extensions) != 1:
            raise ValueError("PD control requires one model runtime per worker")
        self.model = extensions[0]
        self.request_info = request_info

    def handle_command(self, operation, payload):
        return self.model.handle_command(operation, payload)

    def set_profile_active(self, active):
        self.model.set_profile_active(active)

    def wrap_batch(self, build):
        from dataclasses import replace

        def prepare(scheduled, runtime_model, **kwargs):
            batch = build(scheduled, runtime_model, **kwargs)
            return replace(
                batch,
                initial_request_ids=tuple(
                    request.request_id
                    for request in scheduled
                    if self.request_info(request.request_id).initialize_from_cache
                ),
            )

        return prepare
