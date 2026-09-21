# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Deterministic chunk-to-page planning and P-side source retention."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from pypto_serving.config.types import KVCacheGroupSpec
from pypto_serving.serving.memory.kv_cache import KvCacheManager
from pypto_serving.transfer.types import CompletionCertainty

from .contracts import ModelPDContract
from .protocol import HandoffKey, PageCopy, TransferUnit, chunk_payload_hash


@dataclass(frozen=True)
class RankChunkPlan:
    rank_id: int
    copies: tuple[PageCopy, ...]
    expected_units: tuple[TransferUnit, ...]


@dataclass(frozen=True)
class ChunkPlan:
    key: HandoffKey
    chunk_id: int
    start_token: int
    end_token: int
    source_prefix_hit_tokens: int
    final: bool
    ranks: tuple[RankChunkPlan, ...]
    manifest_hash: str

    @property
    def copies_by_rank(self) -> dict[int, tuple[PageCopy, ...]]:
        return {rank.rank_id: rank.copies for rank in self.ranks}

    @property
    def expected_units(self) -> tuple[TransferUnit, ...]:
        return tuple(unit for rank in self.ranks for unit in rank.expected_units)


class ChunkTransferPlanner:
    """Plan complete physical pages from scheduler-owned logical block tables."""

    def __init__(
        self,
        registry: Any,
        group_specs: tuple[KVCacheGroupSpec, ...],
        contract: ModelPDContract,
    ) -> None:
        self.registry = registry
        self.contract = contract
        self.group_components = contract.group_components
        self.final_only_groups = contract.final_only_groups
        self.group_specs = {spec.name: spec for spec in group_specs}
        if tuple(self.group_specs) != contract.logical_groups:
            raise ValueError("cache group order does not match the PD transfer contract")
        for group_name, component_names in self.group_components.items():
            spec = self.group_specs[group_name].spec
            for component_name in component_names:
                entry = registry.entry(component_name)
                if entry.block_tokens != spec.storage_block_size:
                    raise ValueError(
                        f"physical page rows differ for {group_name}/{component_name}: "
                        f"registry={entry.block_tokens}, scheduler={spec.storage_block_size}"
                    )
        self.prefix_alignment = math.lcm(
            *(spec.spec.token_capacity for spec in self.group_specs.values())
        )

    def plan_chunk(
        self,
        key: HandoffKey,
        *,
        chunk_id: int,
        start_token: int,
        end_token: int,
        final: bool,
        rank_ids: tuple[int, ...],
        source_blocks_by_rank: dict[int, dict[str, tuple[int, ...]]],
        destination_blocks_by_rank: dict[int, dict[str, tuple[int, ...]]],
        destination_prefix_hit_tokens: int = 0,
        source_prefix_hit_tokens: int = 0,
    ) -> ChunkPlan:
        if type(chunk_id) is not int or chunk_id < 0:
            raise ValueError("chunk_id must be a non-negative integer")
        if not 0 <= start_token < end_token:
            raise ValueError("chunk token range must be nonempty and increasing")
        if (
            type(destination_prefix_hit_tokens) is not int
            or destination_prefix_hit_tokens < 0
        ):
            raise ValueError("destination prefix hit must be a non-negative integer")
        if (
            type(source_prefix_hit_tokens) is not int
            or not 0 <= source_prefix_hit_tokens <= end_token
        ):
            raise ValueError("source prefix hit must be within the chunk history")
        if source_prefix_hit_tokens % self.prefix_alignment:
            raise ValueError("source prefix hit must use the common cache alignment")
        if not rank_ids or len(rank_ids) != len(set(rank_ids)):
            raise ValueError("rank_ids must be a nonempty unique tuple")
        if set(rank_ids) != set(source_blocks_by_rank) or set(rank_ids) != set(
            destination_blocks_by_rank
        ):
            raise ValueError("source/destination block tables must cover the active ranks")

        rank_plans = []
        for rank_id in rank_ids:
            source = source_blocks_by_rank[rank_id]
            destination = destination_blocks_by_rank[rank_id]
            self._validate_group_tables(source, destination)
            copies: list[PageCopy] = []
            expected_units: list[TransferUnit] = []
            for group_name, component_names in self.group_components.items():
                if group_name in self.final_only_groups and not final:
                    continue
                group_copies = self._group_copies(
                    group_name,
                    start_token=start_token,
                    end_token=end_token,
                    final=final,
                    source_blocks=source[group_name],
                    destination_blocks=destination[group_name],
                    destination_prefix_hit_tokens=destination_prefix_hit_tokens,
                    source_prefix_hit_tokens=source_prefix_hit_tokens,
                )
                for component_name in component_names:
                    entry = self.registry.entry(component_name)
                    component_copies = tuple(
                        PageCopy(
                            component_id=component_name,
                            layer=layer,
                            source_block=source_block,
                            destination_block=destination_block,
                            valid_tokens=valid_rows,
                        )
                        for source_block, destination_block, valid_rows in group_copies
                        for layer in entry.layers
                    )
                    copies.extend(component_copies)
                    nbytes = len(component_copies) * entry.block_stride_bytes
                    # Zero-byte units remain explicit. They prove the planner
                    # considered a region instead of silently dropping it.
                    expected_units.append(
                        TransferUnit(rank_id, component_name, nbytes)
                    )
            rank_plans.append(
                RankChunkPlan(rank_id, tuple(copies), tuple(expected_units))
            )

        expected_units = tuple(
            unit for rank in rank_plans for unit in rank.expected_units
        )
        copies_by_rank = {rank.rank_id: rank.copies for rank in rank_plans}
        manifest_hash = chunk_payload_hash(
            key,
            chunk_id=chunk_id,
            start_token=start_token,
            end_token=end_token,
            final=final,
            expected_units=expected_units,
            copies_by_rank=copies_by_rank,
            source_prefix_hit_tokens=source_prefix_hit_tokens,
        )
        return ChunkPlan(
            key=key,
            chunk_id=chunk_id,
            start_token=start_token,
            end_token=end_token,
            source_prefix_hit_tokens=source_prefix_hit_tokens,
            final=final,
            ranks=tuple(rank_plans),
            manifest_hash=manifest_hash,
        )

    def publication_valid_from(
        self,
        *,
        destination_prefix_hit_tokens: int,
        source_prefix_hit_tokens: int,
    ) -> dict[str, int]:
        """Return the first newly valid token for each rolling cache group.

        When P hits farther than D, P cannot backfill rolling rows older than
        its live history window.  D may decode from the received tail, but it
        must not publish the untouched gap as reusable prefix data.
        """
        for name, value in (
            ("destination", destination_prefix_hit_tokens),
            ("source", source_prefix_hit_tokens),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} prefix hit must be a non-negative integer")
            if value % self.prefix_alignment:
                raise ValueError(
                    f"{name} prefix hit must use the common cache alignment"
                )
        return {
            group_name: max(
                destination_prefix_hit_tokens,
                source_prefix_hit_tokens - group.sliding_window,
            )
            for group_name, group in self.group_specs.items()
            if group_name not in self.final_only_groups
            and group.sliding_window is not None
        }

    def _group_copies(
        self,
        group_name: str,
        *,
        start_token: int,
        end_token: int,
        final: bool,
        source_blocks: tuple[int, ...],
        destination_blocks: tuple[int, ...],
        destination_prefix_hit_tokens: int,
        source_prefix_hit_tokens: int,
    ) -> tuple[tuple[int, int, int], ...]:
        group = self.group_specs[group_name]
        capacity = group.spec.token_capacity
        physical_rows = group.spec.storage_block_size
        ratio = group.spec.compress_ratio
        logical_end = (end_token + capacity - 1) // capacity

        if group_name in self.final_only_groups:
            first = max(0, logical_end - len(source_blocks))
            logical_indices = range(first, logical_end)
        else:
            transfer_start = max(start_token, destination_prefix_hit_tokens)
            if group.sliding_window is not None:
                # A P-local rolling hit owns only the live history window at
                # P_hit, followed by rows computed for this request. Older ring
                # slots are merely allocated table storage and must never be
                # treated as valid backfill data when P_hit > D_hit.
                transfer_start = max(
                    transfer_start,
                    source_prefix_hit_tokens - group.sliding_window,
                )
            first_closed = transfer_start // capacity
            closed_end = end_token // capacity
            logical_indices = range(first_closed, closed_end)

        rows_by_index: dict[int, int] = {
            logical: physical_rows for logical in logical_indices
        }
        remainder = end_token % capacity
        if (
            final
            and remainder
            and group_name not in self.final_only_groups
            and end_token > destination_prefix_hit_tokens
        ):
            valid_rows = remainder // ratio
            if valid_rows:
                rows_by_index[end_token // capacity] = valid_rows
        if final and group_name in self.final_only_groups and logical_end:
            last = logical_end - 1
            if remainder:
                rows_by_index[last] = remainder // ratio

        result = []
        for logical_index in sorted(rows_by_index):
            valid_rows = rows_by_index[logical_index]
            if not 0 < valid_rows <= physical_rows:
                continue
            result.append(
                (
                    self._physical_block(group, source_blocks, logical_index),
                    self._physical_block(group, destination_blocks, logical_index),
                    valid_rows,
                )
            )
        return tuple(result)

    @staticmethod
    def _physical_block(
        group: KVCacheGroupSpec,
        table: tuple[int, ...],
        logical_index: int,
    ) -> int:
        if not table:
            raise ValueError(f"cache group {group.name!r} has an empty block table")
        slot = logical_index if group.sliding_window is None else logical_index % len(table)
        if not 0 <= slot < len(table):
            raise ValueError(
                f"cache group {group.name!r} cannot address logical block {logical_index}"
            )
        return table[slot]

    def _validate_group_tables(
        self,
        source: dict[str, tuple[int, ...]],
        destination: dict[str, tuple[int, ...]],
    ) -> None:
        expected = set(self.group_specs)
        if set(source) != expected or set(destination) != expected:
            raise ValueError("rank block tables do not match the model contract")
        for group_name in expected:
            for table in (source[group_name], destination[group_name]):
                if len(table) != len(set(table)) or any(
                    type(block_id) is not int or block_id < 0 for block_id in table
                ):
                    raise ValueError(
                        f"cache group {group_name!r} has invalid physical block IDs"
                    )


@dataclass(frozen=True)
class SourceGuard:
    request_id: str
    chunk_id: int
    block_ids_by_group: dict[str, tuple[int, ...]]
    partition: int


class PChunkLifecycle:
    """One-transfer-at-a-time source pinning for the first PD version."""

    def __init__(self, cache_manager: KvCacheManager) -> None:
        self.cache_manager = cache_manager
        self._in_flight: dict[str, SourceGuard] = {}

    def begin(
        self,
        request_id: str,
        chunk_id: int,
        block_ids_by_group: dict[str, tuple[int, ...]],
        partition: int,
    ) -> SourceGuard:
        if request_id in self._in_flight:
            raise RuntimeError("a request already has an in-flight chunk transfer")
        mutable_tables = {name: list(ids) for name, ids in block_ids_by_group.items()}
        self.cache_manager.retain_group_block_snapshot(mutable_tables, partition)
        guard = SourceGuard(request_id, chunk_id, block_ids_by_group, partition)
        self._in_flight[request_id] = guard
        return guard

    def settle(self, guard: SourceGuard, certainty: CompletionCertainty) -> None:
        current = self._in_flight.get(guard.request_id)
        if current != guard:
            raise ValueError("source guard is stale or already settled")
        if certainty is CompletionCertainty.UNKNOWN:
            # UNKNOWN moves to owner/process recovery. Releasing this extra pin
            # would falsely assert that all native reads have stopped.
            return
        self.cache_manager.release_group_block_snapshot(
            {name: list(ids) for name, ids in guard.block_ids_by_group.items()},
            guard.partition,
        )
        del self._in_flight[guard.request_id]

    def abandon_after_owner_death(self, guard: SourceGuard) -> None:
        """Forget bookkeeping only after the old allocator process is dead."""
        current = self._in_flight.get(guard.request_id)
        if current != guard:
            raise ValueError("source guard is stale or already settled")
        del self._in_flight[guard.request_id]
