# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Allocator-owned leases for externally populated grouped cache pages."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum


class GroupReservationState(str, Enum):
    """Lifecycle of an externally filled grouped-cache allocation."""

    PREPARING = "PREPARING"
    CONSTRUCTING = "CONSTRUCTING"
    READY = "READY"
    IN_USE = "IN_USE"
    RELEASED = "RELEASED"
    ABORTING = "ABORTING"
    QUARANTINED = "QUARANTINED"


@dataclass(frozen=True)
class GroupCacheReservation:
    """One externally populated reservation backed by the normal grouped block pools."""

    reservation_id: str
    request_id: str
    token_capacity: int
    prompt_token_count: int
    partition: int
    block_ids_by_group: dict[str, tuple[int, ...]]
    shared_block_ids_by_group: dict[str, tuple[int, ...]]
    writable_block_ids_by_group: dict[str, tuple[int, ...]]
    prefix_block_hashes: dict[str, tuple[bytes, ...]]
    prefix_hit_tokens: int
    published_block_counts: dict[str, int]
    state: GroupReservationState
    write_authorized: bool = False
    publication_valid_from: dict[str, int] = field(default_factory=dict)
    quarantined_block_ids_by_group: dict[str, tuple[int, ...]] = field(default_factory=dict)


class GroupedCacheReservations:
    """One lease ledger owned by the existing cache manager, never another allocator."""

    def __init__(self, manager):
        self.manager = manager
        self._group_reservations: dict[str, GroupCacheReservation] = {}
        self._group_request_reservations: dict[str, str] = {}

    def reserve_group_cache(
        self,
        reservation_id: str,
        request_id: str,
        token_capacity: int,
        *,
        partition: int | None = None,
        prompt_token_count: int | None = None,
        prefix_block_hashes: dict[str, list[bytes]] | None = None,
        shareable_group_names: tuple[str, ...] = (),
    ) -> GroupCacheReservation:
        """Atomically reserve every configured group through the existing allocator.

        A reservation is deliberately not a second page allocator. It is a
        lifecycle record around ``ensure_group_blocks`` and the same request maps
        used by local scheduling. Repeating the identical request is idempotent;
        changing any field under a reused identity is rejected.
        """
        if not self.manager._group_pools:
            raise RuntimeError("grouped cache must be initialized before reservation")
        if not isinstance(reservation_id, str) or not reservation_id:
            raise ValueError("reservation_id must be a nonempty string")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a nonempty string")
        if type(token_capacity) is not int or token_capacity <= 0:
            raise ValueError("token_capacity must be a positive integer")
        prompt_tokens = token_capacity if prompt_token_count is None else prompt_token_count
        if type(prompt_tokens) is not int or not 0 < prompt_tokens <= token_capacity:
            raise ValueError("prompt_token_count must be within the reservation capacity")
        prefix_hashes = prefix_block_hashes or {}
        if prefix_hashes and not shareable_group_names:
            raise ValueError("prefix hashes require explicit shareable cache groups")

        existing = self._group_reservations.get(reservation_id)
        if existing is not None:
            expected_partition = existing.partition if partition is None else partition
            if (
                existing.request_id != request_id
                or existing.token_capacity != token_capacity
                or existing.prompt_token_count != prompt_tokens
                or existing.partition != expected_partition
                or existing.prefix_block_hashes
                != {name: tuple(values) for name, values in prefix_hashes.items()}
            ):
                raise ValueError("reservation identity was reused with different parameters")
            return existing
        existing_id = self._group_request_reservations.get(request_id)
        if existing_id is not None:
            raise ValueError(f"request {request_id!r} already belongs to reservation {existing_id!r}")
        if request_id in self.manager._group_request_partitions:
            raise ValueError(f"request {request_id!r} already owns grouped cache blocks")

        # The normal allocator preflights every group before mutation. The catch
        # still rolls back an unexpected mid-allocation exception so a failed
        # reservation cannot strand a partially populated request.
        try:
            shared_ids: dict[str, list[int]] = {name: [] for name in self.manager._group_pools}
            prefix_hit_tokens = 0
            selected_prefix_partition = None
            if prefix_hashes and self.manager.enable_prefix_cache:
                (
                    shared_ids,
                    prefix_hit_tokens,
                    selected_prefix_partition,
                ) = self.manager.acquire_group_prefix_blocks(
                    request_id,
                    prefix_hashes,
                    max_cache_hit_tokens=prompt_tokens,
                    shareable_group_names=shareable_group_names,
                    partition=partition,
                )
            block_ids = self.manager.ensure_group_blocks(
                request_id,
                prompt_tokens,
                partition=(selected_prefix_partition if selected_prefix_partition is not None else partition),
            )
            selected = self.manager.group_request_partition(request_id)
            if selected is None:
                raise RuntimeError("grouped reservation has no selected partition")
            self.manager._reserve_group_decode_capacity(
                request_id,
                token_capacity,
                selected,
            )
            block_ids = {
                name: [
                    pool.local_block_id(block)
                    for block in pool.request_blocks[request_id]
                    if block is not None
                ]
                for name, pool in self.manager._group_pools.items()
            }
        except BaseException:
            if request_id in self.manager._group_request_partitions:
                self.manager.release_all_group_requests(request_id)
            raise

        shared = {name: tuple(shared_ids.get(name, ())) for name in self.manager._group_pools}
        writable = {
            name: tuple(block_id for block_id in block_ids[name] if block_id not in set(shared[name]))
            for name in self.manager._group_pools
        }
        cached_counts = {
            name: min(
                prefix_hit_tokens // self.manager._group_pools[name].spec.spec.token_capacity,
                len(prefix_hashes.get(name, ())),
            )
            for name in prefix_hashes
        }
        reservation = GroupCacheReservation(
            reservation_id=reservation_id,
            request_id=request_id,
            token_capacity=token_capacity,
            prompt_token_count=prompt_tokens,
            partition=selected,
            block_ids_by_group={name: tuple(ids) for name, ids in block_ids.items()},
            shared_block_ids_by_group=shared,
            writable_block_ids_by_group=writable,
            prefix_block_hashes={name: tuple(values) for name, values in prefix_hashes.items()},
            prefix_hit_tokens=prefix_hit_tokens,
            published_block_counts=cached_counts,
            state=GroupReservationState.CONSTRUCTING,
        )
        self._group_reservations[reservation_id] = reservation
        self._group_request_reservations[request_id] = reservation_id
        return reservation

    def group_cache_reservation(self, reservation_id: str) -> GroupCacheReservation | None:
        """Return the stable reservation fact for status/query handling."""
        return self._group_reservations.get(reservation_id)

    @property
    def group_cache_reservations(self) -> tuple[GroupCacheReservation, ...]:
        """Return lifecycle facts for diagnostics without exposing mutable indexes."""
        return tuple(self._group_reservations.values())

    def authorize_group_cache_write(self, reservation_id: str) -> GroupCacheReservation:
        """Mark the destination pages as exposed to a remote writer."""
        current = self._require_group_reservation(reservation_id)
        if current.state is GroupReservationState.CONSTRUCTING:
            current = replace(current, write_authorized=True)
            self._group_reservations[reservation_id] = current
            return current
        if current.write_authorized and current.state in (
            GroupReservationState.READY,
            GroupReservationState.IN_USE,
            GroupReservationState.QUARANTINED,
        ):
            return current
        raise ValueError(f"cannot authorize a reservation in state {current.state.value}")

    def commit_group_cache(
        self,
        reservation_id: str,
        *,
        valid_from: dict[str, int] | None = None,
    ) -> GroupCacheReservation:
        """Atomically publish a completely received reservation as READY."""
        publication_floor = dict(valid_from or {})
        unknown = set(publication_floor) - set(self.manager._group_pools)
        if unknown:
            raise ValueError("unknown cache groups in publication floor: " + ", ".join(sorted(unknown)))
        if any(type(value) is not int or value < 0 for value in publication_floor.values()):
            raise ValueError("cache publication floors must be non-negative integers")
        current = self._require_group_reservation(reservation_id)
        if current.state in (
            GroupReservationState.READY,
            GroupReservationState.IN_USE,
        ):
            if current.publication_valid_from != publication_floor:
                raise ValueError("ready reservation publication floor mismatch")
            return current
        if current.state is not GroupReservationState.CONSTRUCTING:
            raise ValueError(f"cannot commit a reservation in state {current.state.value}")
        if not current.write_authorized:
            raise ValueError("cannot commit a reservation before remote-write authorization")
        published = self.manager.cache_group_blocks(
            current.request_id,
            {name: list(values) for name, values in current.prefix_block_hashes.items()},
            current.prompt_token_count,
            current.published_block_counts,
            valid_from=publication_floor,
        )
        current = replace(
            current,
            state=GroupReservationState.READY,
            published_block_counts=published,
            publication_valid_from=publication_floor,
        )
        self._group_reservations[reservation_id] = current
        return current

    def adopt_group_cache(self, reservation_id: str) -> GroupCacheReservation:
        """Transfer a READY reservation into the normal Decode request lifecycle."""
        current = self._require_group_reservation(reservation_id)
        if current.state is GroupReservationState.IN_USE:
            return current
        if current.state is not GroupReservationState.READY:
            raise ValueError(f"cannot adopt a reservation in state {current.state.value}")
        current = replace(current, state=GroupReservationState.IN_USE)
        self._group_reservations[reservation_id] = current
        return current

    def quarantine_group_cache(self, reservation_id: str) -> GroupCacheReservation:
        """Quarantine only remote-writable pages after an uncertain write.

        Shared prefix references remain pinned until owner recovery so the
        request view stays intact, but they are never classified as polluted.
        """
        current = self._require_group_reservation(reservation_id)
        if current.state is GroupReservationState.RELEASED:
            raise ValueError("released reservations cannot be quarantined")
        if current.state is GroupReservationState.QUARANTINED:
            return current
        current = replace(
            current,
            state=GroupReservationState.QUARANTINED,
            quarantined_block_ids_by_group=current.writable_block_ids_by_group,
        )
        self._group_reservations[reservation_id] = current
        return current

    def release_group_cache(
        self,
        reservation_id: str,
        *,
        confirmed_stopped: bool = False,
    ) -> GroupCacheReservation:
        """Release through the normal group pools only when no writer can remain.

        Before authorization, abort is always deterministic. After authorization,
        CONSTRUCTING/QUARANTINED pages require an external stop/death proof.
        READY and IN_USE have a completed transfer fact and use normal lifecycle
        release.
        """
        current = self._require_group_reservation(reservation_id)
        if current.state is GroupReservationState.RELEASED:
            return current
        uncertain = (
            current.state
            in (
                GroupReservationState.CONSTRUCTING,
                GroupReservationState.ABORTING,
                GroupReservationState.QUARANTINED,
            )
            and current.write_authorized
        )
        if uncertain and not confirmed_stopped:
            raise RuntimeError("reservation may still have a native writer; keep it quarantined")
        self.manager.release_all_group_requests(current.request_id)
        self._group_request_reservations.pop(current.request_id, None)
        current = replace(current, state=GroupReservationState.RELEASED)
        self._group_reservations[reservation_id] = current
        return current

    def retire_group_cache_reservation(self, reservation_id: str) -> None:
        """Forget a terminal reservation after its stable fact was persisted.

        The cache manager owns only live allocator state. Callers keep a bounded
        terminal tombstone separately for idempotent status replies; retaining
        every RELEASED allocator record here would otherwise grow for the
        lifetime of the serving process.
        """
        current = self._require_group_reservation(reservation_id)
        if current.state is not GroupReservationState.RELEASED:
            raise RuntimeError("only a released grouped reservation can be retired")
        if current.request_id in self._group_request_reservations:
            raise RuntimeError("released reservation still has a request index")
        self._group_reservations.pop(reservation_id)

    def _require_group_reservation(self, reservation_id: str) -> GroupCacheReservation:
        try:
            return self._group_reservations[reservation_id]
        except KeyError as exc:
            raise KeyError(f"unknown grouped cache reservation {reservation_id!r}") from exc
