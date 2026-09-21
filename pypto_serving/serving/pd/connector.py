# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host-side D reservation and completion orchestration."""

from __future__ import annotations

from collections import OrderedDict
import uuid

from pypto_serving.serving.memory.kv_cache import (
    GroupReservationState,
    KvCacheManager,
)

from .completion import CompletionState, CompletionTracker
from .config import PDCapabilities
from .contracts import ModelPDContract
from .planner import ChunkTransferPlanner
from .protocol import (
    AbortHandoff,
    ChunkManifest,
    CommitRequest,
    HandoffKey,
    HandoffStatus,
    RankRegistration,
    ReadyAck,
    ReserveAccepted,
    ReserveRejected,
    ReserveRequest,
    TransferResult,
    validate_prefix_match_spec,
)


class DecodeConnector:
    """D Host authority over reservations, manifests and admission facts."""

    def __init__(
        self,
        cache_manager: KvCacheManager,
        capabilities: PDCapabilities,
        registry,
        destination_ranks: tuple[RankRegistration, ...],
        *,
        contract: ModelPDContract,
        terminal_history_limit: int = 1024,
    ) -> None:
        self.cache_manager = cache_manager
        self.capabilities = capabilities
        self.registry = registry
        self.contract = contract
        self.destination_ranks = destination_ranks
        self._trackers: dict[HandoffKey, CompletionTracker] = {}
        self._reservation_by_key: dict[HandoffKey, str] = {}
        self._reservation_specs: dict[HandoffKey, tuple[object, ...]] = {}
        if terminal_history_limit <= 0:
            raise ValueError("terminal_history_limit must be positive")
        self._terminal_history_limit = terminal_history_limit
        self._terminal_status: OrderedDict[HandoffKey, HandoffStatus] = OrderedDict()
        if registry.fingerprint != capabilities.registry_fingerprint:
            raise ValueError("D registry differs from the advertised capability")
        if registry.layout_fingerprint != capabilities.layout_fingerprint:
            raise ValueError("D layout differs from the advertised capability")
        if registry.topology != capabilities.topology:
            raise ValueError("D registry topology differs from the advertised capability")
        self._planner = ChunkTransferPlanner(registry, cache_manager.group_specs, contract)
        self._validate_destination_ranks()

    def reserve(self, request: ReserveRequest) -> ReserveAccepted | ReserveRejected:
        if request.layout_fingerprint != self.capabilities.layout_fingerprint:
            return ReserveRejected(request.key, "REGISTRY_MISMATCH", False)
        if request.prompt_token_count <= 0 or request.max_new_tokens <= 0:
            return ReserveRejected(request.key, "INVALID_LENGTH", False)
        if request.prepared_digest and len(request.prepared_digest) != 64:
            return ReserveRejected(request.key, "INVALID_PREPARED_DIGEST", False)
        prefix_spec = request.prefix_match_spec
        if self.capabilities.prefix_cache_mode == "disabled":
            if prefix_spec is not None:
                return ReserveRejected(request.key, "UNEXPECTED_PREFIX_MATCH", False)
            prefix_hashes: dict[str, list[bytes]] = {}
        else:
            if prefix_spec is None:
                return ReserveRejected(request.key, "MISSING_PREFIX_MATCH", False)
            try:
                validate_prefix_match_spec(prefix_spec)
            except ValueError:
                return ReserveRejected(request.key, "INVALID_PREFIX_MATCH", False)
            if (
                prefix_spec.token_count != request.prompt_token_count
                or prefix_spec.alignment
                != self.cache_manager.group_prefix_cache_alignment
                or prefix_spec.contract_digest != self.contract.digest
                or tuple(group.group_name for group in prefix_spec.groups)
                != tuple(sorted(self.contract.prefix_cache_groups))
            ):
                return ReserveRejected(request.key, "PREFIX_MATCH_CONTRACT_MISMATCH", False)
            prefix_hashes = {
                group.group_name: list(group.block_hashes)
                for group in prefix_spec.groups
            }
            specs = {spec.name: spec for spec in self.cache_manager.group_specs}
            for group_name, hashes in prefix_hashes.items():
                spec = specs[group_name]
                expected = (
                    max(0, request.prompt_token_count - 1)
                    if spec.is_eagle_group
                    else request.prompt_token_count
                ) // spec.spec.token_capacity
                if len(hashes) != expected:
                    return ReserveRejected(
                        request.key, "PREFIX_MATCH_BLOCK_COUNT", False
                    )
        if request.key in self._terminal_status:
            return ReserveRejected(request.key, "HANDOFF_TERMINAL", False)
        spec = (
            request.prompt_token_count,
            request.max_new_tokens,
            request.layout_fingerprint,
            request.requested_partition,
            request.prepared_digest,
            "" if prefix_spec is None else prefix_spec.identity_digest,
        )
        existing_id = self._reservation_by_key.get(request.key)
        if existing_id is not None:
            if self._reservation_specs.get(request.key) != spec:
                return ReserveRejected(request.key, "RESERVATION_REPLAY_MISMATCH", False)
            reservation = self.cache_manager.group_cache_reservation(existing_id)
            if reservation is None:
                raise RuntimeError("D reservation index differs from cache manager")
            return self._accepted(request.key, reservation)
        if any(key.request_id == request.key.request_id for key in self._reservation_by_key):
            return ReserveRejected(request.key, "REQUEST_ALREADY_RESERVED", False)

        reservation_id = f"pd-{request.key.handoff_id}-{uuid.uuid4().hex[:12]}"
        try:
            reservation = self.cache_manager.reserve_group_cache(
                reservation_id,
                request.key.request_id,
                request.prompt_token_count + request.max_new_tokens,
                partition=request.requested_partition,
                prompt_token_count=request.prompt_token_count,
                prefix_block_hashes=prefix_hashes,
                shareable_group_names=self.contract.prefix_cache_groups,
            )
            # Returning destination envelopes is the authorization boundary:
            # after this message is visible, timeout alone cannot free the pages.
            reservation = self.cache_manager.authorize_group_cache_write(reservation_id)
        except (RuntimeError, ValueError) as exc:
            return ReserveRejected(request.key, type(exc).__name__, True)
        self._reservation_by_key[request.key] = reservation_id
        self._reservation_specs[request.key] = spec
        self._trackers[request.key] = CompletionTracker(request.key, reservation_id)
        return self._accepted(request.key, reservation)

    def register_chunk(self, manifest: ChunkManifest) -> None:
        self._validate_manifest(manifest)
        self._require_tracker(manifest.key).register_chunk(manifest)

    def record_transfer(self, result: TransferResult) -> None:
        tracker = self._require_tracker(result.key)
        tracker.record_transfer(result)
        if tracker.state is CompletionState.QUARANTINED:
            self.cache_manager.quarantine_group_cache(tracker.reservation_id)

    def commit(self, request: CommitRequest) -> ReadyAck:
        tracker = self._require_tracker(request.key)
        ack = tracker.commit(request)
        reservation = self.cache_manager.commit_group_cache(
            tracker.reservation_id,
            request.manifest_hash,
        )
        if reservation.state is not GroupReservationState.READY:
            raise RuntimeError("completion and cache reservation commit diverged")
        return ack

    def claim_decode_admission(self, key: HandoffKey) -> bool:
        """Return true once; the caller must immediately adopt in the scheduler."""
        return self._require_tracker(key).admit_decode()

    def mark_decode_completed(self, key: HandoffKey) -> None:
        tracker = self._require_tracker(key)
        tracker.mark_completed()
        self._retire_terminal(key, tracker)

    def mark_decode_cancelled(self, key: HandoffKey, *, error_code: str) -> HandoffStatus:
        tracker = self._require_tracker(key)
        tracker.mark_cancelled(error_code=error_code)
        reservation = self.cache_manager.group_cache_reservation(tracker.reservation_id)
        if reservation is not None and reservation.state is not GroupReservationState.RELEASED:
            self.cache_manager.release_group_cache(
                tracker.reservation_id,
                confirmed_stopped=True,
            )
        return self._retire_terminal(key, tracker)

    def abort(self, request: AbortHandoff, *, deterministic: bool) -> HandoffStatus:
        terminal = self._terminal_status.get(request.key)
        if terminal is not None:
            # Router and P may independently converge on the same abort after
            # an HTTP/control-plane race.  Once D has safely released the
            # reservation, replay the immutable terminal fact instead of
            # turning the second abort into an unrelated 500/KeyError.
            self._terminal_status.move_to_end(request.key)
            return terminal
        tracker = self._require_tracker(request.key)
        tracker.abort(deterministic=deterministic, error_code=request.reason)
        if deterministic and tracker.state is CompletionState.ABORTED:
            self.cache_manager.release_group_cache(
                tracker.reservation_id,
                confirmed_stopped=True,
            )
            return self._retire_terminal(request.key, tracker)
        else:
            self.cache_manager.quarantine_group_cache(tracker.reservation_id)
        return tracker.query()

    def query(self, key: HandoffKey) -> HandoffStatus:
        terminal = self._terminal_status.get(key)
        if terminal is not None:
            self._terminal_status.move_to_end(key)
            return terminal
        return self._require_tracker(key).query()

    def replay_ready_ack(self, key: HandoffKey) -> ReadyAck:
        return self._require_tracker(key).replay_ready_ack()

    def committed_manifest(self, key: HandoffKey) -> ChunkManifest:
        return self._require_tracker(key).final_manifest()

    def _accepted(self, key: HandoffKey, reservation) -> ReserveAccepted:
        return ReserveAccepted(
            key=key,
            reservation_id=reservation.reservation_id,
            partition=reservation.partition,
            block_ids_by_group=reservation.block_ids_by_group,
            ranks=tuple(
                self.destination_ranks[rank_id]
                for rank_id in self._active_rank_ids(reservation.partition)
            ),
            prefix_hit_tokens=reservation.prefix_hit_tokens,
        )

    def _require_tracker(self, key: HandoffKey) -> CompletionTracker:
        try:
            return self._trackers[key]
        except KeyError as exc:
            raise KeyError("unknown D handoff identity") from exc

    def _retire_terminal(
        self,
        key: HandoffKey,
        tracker: CompletionTracker,
    ) -> HandoffStatus:
        """Move a released terminal handoff into bounded replay history."""
        status = tracker.query()
        if status.state not in (CompletionState.COMPLETED.value, CompletionState.ABORTED.value):
            raise RuntimeError("only a safe terminal handoff can be retired")
        reservation = self.cache_manager.group_cache_reservation(tracker.reservation_id)
        if reservation is None or reservation.state is not GroupReservationState.RELEASED:
            raise RuntimeError("terminal handoff cache reservation is not released")
        self.cache_manager.retire_group_cache_reservation(tracker.reservation_id)
        self._trackers.pop(key, None)
        self._reservation_by_key.pop(key, None)
        self._reservation_specs.pop(key, None)
        self._terminal_status[key] = status
        self._terminal_status.move_to_end(key)
        while len(self._terminal_status) > self._terminal_history_limit:
            self._terminal_status.popitem(last=False)
        return status

    def _validate_destination_ranks(self) -> None:
        expected_rank_count = self.capabilities.topology[0]
        if len(self.destination_ranks) != expected_rank_count:
            raise ValueError("destination owner count does not match the PD topology")
        for rank_id, rank in enumerate(self.destination_ranks):
            if rank.rank_id != rank_id:
                raise ValueError("destination owners must be in rank order")
            regions = {region.component_id for region in rank.regions}
            if regions != set(self.contract.physical_regions):
                raise ValueError("destination owner regions differ from the model contract")
            for region in rank.regions:
                if region.extent != self.registry.entry(region.component_id).extent:
                    raise ValueError("destination owner extent differs from the D registry")

    def _active_rank_ids(self, partition: int) -> tuple[int, ...]:
        rank_count = self.capabilities.topology[0]
        group_size = (
            self.capabilities.topology[1]
            if len(self.capabilities.topology) > 1
            else 1
        )
        if rank_count % group_size:
            raise ValueError("PD rank count is not divisible by its cache-replica group")
        partition_count = rank_count // group_size
        if not 0 <= partition < partition_count:
            raise ValueError("D reservation partition is outside the PD topology")
        start = partition * group_size
        return tuple(range(start, start + group_size))

    def _validate_manifest(self, manifest: ChunkManifest) -> None:
        tracker = self._require_tracker(manifest.key)
        expected_digest = self._reservation_specs[manifest.key][4]
        if manifest.prepared_digest != expected_digest:
            raise ValueError("chunk manifest prepared-request digest mismatch")
        reservation = self.cache_manager.group_cache_reservation(tracker.reservation_id)
        if reservation is None:
            raise RuntimeError("D completion tracker lost its cache reservation")
        if manifest.end_token > reservation.token_capacity:
            raise ValueError("chunk exceeds the D-first reservation capacity")

        rank_ids = self._active_rank_ids(reservation.partition)
        if set(manifest.copies_by_rank) != set(rank_ids):
            raise ValueError("chunk copies do not cover exactly the reserved TP group")
        required_components = {
            component
            for group, components in self.contract.group_components.items()
            if manifest.final or group not in self.contract.final_only_groups
            for component in components
        }
        expected_unit_keys = {
            (rank_id, component)
            for rank_id in rank_ids
            for component in required_components
        }
        actual_unit_keys = {
            (unit.rank_id, unit.component_id)
            for unit in manifest.expected_units
        }
        if actual_unit_keys != expected_unit_keys:
            raise ValueError("chunk completion set omits a rank or physical region")

        destination_tables = {
            rank_id: reservation.block_ids_by_group for rank_id in rank_ids
        }
        # Source block IDs are owner-local and may differ.  Replanning with the
        # D tables in both positions yields the authoritative destination page,
        # layer and valid-row set for this token interval.
        expected = self._planner.plan_chunk(
            manifest.key,
            chunk_id=manifest.chunk_id,
            start_token=manifest.start_token,
            end_token=manifest.end_token,
            final=manifest.final,
            rank_ids=rank_ids,
            source_blocks_by_rank=destination_tables,
            destination_blocks_by_rank=destination_tables,
            destination_prefix_hit_tokens=reservation.prefix_hit_tokens,
        )
        expected_writes = {
            rank.rank_id: {
                (
                    copy.component_id,
                    copy.layer,
                    copy.destination_block,
                    copy.valid_tokens,
                )
                for copy in rank.copies
            }
            for rank in expected.ranks
        }
        actual_writes = {
            rank_id: {
                (
                    copy.component_id,
                    copy.layer,
                    copy.destination_block,
                    copy.valid_tokens,
                )
                for copy in copies
            }
            for rank_id, copies in manifest.copies_by_rank.items()
        }
        if any(
            len(copies) != len(actual_writes[rank_id])
            for rank_id, copies in manifest.copies_by_rank.items()
        ):
            raise ValueError("chunk manifest contains duplicate physical writes")
        if actual_writes != expected_writes:
            raise ValueError("chunk physical writes differ from the D reservation plan")

        expected_bytes = {
            (rank.rank_id, unit.component_id): unit.nbytes
            for rank in expected.ranks
            for unit in rank.expected_units
        }
        actual_bytes = {
            (unit.rank_id, unit.component_id): unit.nbytes
            for unit in manifest.expected_units
        }
        if actual_bytes != expected_bytes:
            raise ValueError("chunk byte accounting differs from the D registry")
        for copies in manifest.copies_by_rank.values():
            for copy in copies:
                if type(copy.source_block) is not int or copy.source_block < 0:
                    raise ValueError("source block is not a non-negative integer")
