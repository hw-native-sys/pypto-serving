# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Address-free identities and completion proofs for grouped cache ownership."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CachePageIdentity:
    group: str
    partition: int
    page_id: int
    allocation_epoch: int

    def __post_init__(self):
        if (
            not self.group
            or any(type(v) is not int for v in (self.partition, self.page_id, self.allocation_epoch))
            or min(self.partition, self.page_id) < 0
            or self.allocation_epoch <= 0
        ):
            raise ValueError("invalid cache page identity")


@dataclass(frozen=True)
class GroupPageRange:
    group: str
    start: int
    end: int

    def __post_init__(self):
        if (
            not self.group
            or type(self.start) is not int
            or type(self.end) is not int
            or not 0 <= self.start < self.end
        ):
            raise ValueError("invalid complete logical page range")


@dataclass(frozen=True)
class CachePublicationProof:
    request_id: str
    generation: int
    mapping_version: int
    confirmed_input_end: int
    checkpoint_boundary: int
    complete_logical_pages: tuple[GroupPageRange, ...]
    page_identities: tuple[CachePageIdentity, ...]

    def __post_init__(self):
        if (
            not self.request_id
            or any(
                type(v) is not int
                for v in (
                    self.generation,
                    self.mapping_version,
                    self.confirmed_input_end,
                    self.checkpoint_boundary,
                )
            )
            or min(self.generation, self.mapping_version) <= 0
            or not 0 < self.checkpoint_boundary <= self.confirmed_input_end
            or len(set(self.page_identities)) != len(self.page_identities)
        ):
            raise ValueError("invalid cache publication proof")


@dataclass(frozen=True)
class GroupCacheCopyPlan:
    request_id: str
    generation: int
    mapping_version: int
    boundary: int
    source_tables: dict[str, tuple[int, ...]]
    destination_tables: dict[str, tuple[int, ...]]
    source_pages: tuple[CachePageIdentity, ...]
    destination_pages: tuple[CachePageIdentity, ...]
    complete_logical_pages: tuple[GroupPageRange, ...]

    def __post_init__(self):
        if (
            not self.request_id
            or any(
                type(v) is not int or v <= 0 for v in (self.generation, self.mapping_version, self.boundary)
            )
            or set(self.source_tables) != set(self.destination_tables)
        ):
            raise ValueError("Invalid checkpoint copy identity/boundary")
        for tables, pages in (
            (self.source_tables, self.source_pages),
            (self.destination_tables, self.destination_pages),
        ):
            actual = set()
            for page in pages:
                page.__post_init__()
                actual.add((page.group, page.partition, page.page_id))
            expected = set()
            partitions = {page.partition for page in pages}
            if len(partitions) != 1:
                raise ValueError("Group copy pages require a single partition")
            partition = next(iter(partitions))
            for group, ids in tables.items():
                if not ids or len(set(ids)) != len(ids) or any(type(p) is not int or p < 0 for p in ids):
                    raise ValueError("Invalid checkpoint table page IDs")
                expected.update((group, partition, page) for page in ids)
            if actual != expected or len(pages) != len(expected):
                raise ValueError("Checkpoint tables and page epochs disagree")
