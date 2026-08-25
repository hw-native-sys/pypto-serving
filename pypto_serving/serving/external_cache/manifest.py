# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from pypto_serving.config.types import KVCacheGroupSpec


EXTERNAL_CACHE_SCHEMA_VERSION = 1
DEEPSEEK_V4_CACHE_GROUPS = (
    "ori",
    "cmp_c128",
    "cmp_c4",
    "idx",
    "hca_state",
    "csa_state",
    "csa_inner_state",
)
_KEY_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")
_HEX_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_TOKEN_DIGEST_DOMAIN = b"pypto-external-prefix-tokens-v1\0"


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _digest_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _validate_digest(name: str, value: str) -> None:
    if not _HEX_DIGEST.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


def stable_token_digest(token_ids: Sequence[int]) -> str:
    """Hash one token prefix using a stable architecture-independent encoding."""
    digest = hashlib.sha256()
    digest.update(_TOKEN_DIGEST_DOMAIN)
    digest.update(struct.pack(">Q", len(token_ids)))
    for token_id in token_ids:
        token_id = int(token_id)
        if token_id < 0 or token_id > 0xFFFFFFFFFFFFFFFF:
            raise ValueError("token IDs must be unsigned 64-bit integers")
        digest.update(struct.pack(">Q", token_id))
    return digest.hexdigest()


def checkpoint_prefix_digest(
    token_ids: Sequence[int],
    token_count: int,
    *,
    mtp_enabled: bool,
) -> str:
    """Hash every token that determines the checkpoint's stored bytes."""
    dependency_count = int(token_count) + int(bool(mtp_enabled))
    if token_count <= 0 or dependency_count > len(token_ids):
        raise ValueError("checkpoint token dependency range is unavailable")
    return stable_token_digest(token_ids[:dependency_count])


def _group_layout(group: KVCacheGroupSpec) -> dict[str, Any]:
    return {
        "block_size": group.spec.block_size,
        "compress_ratio": group.spec.compress_ratio,
        "is_eagle_group": group.is_eagle_group,
        "layer_indices": list(group.layer_indices),
        "max_blocks_per_seq": group.max_blocks_per_seq,
        "name": group.name,
        "num_partitions": group.num_partitions,
        "page_size_bytes": group.spec.page_size_bytes,
        "sliding_window": group.sliding_window,
    }


@dataclass(frozen=True, slots=True)
class ExternalCacheNamespace:
    """Version all inputs that affect the bytes stored in an external cache."""

    model_id: str
    model_revision: str
    tokenizer_revision: str
    kv_dtype: str
    tensor_parallel_size: int
    world_size: int
    parallel_config_digest: str
    mtp_enabled: bool
    group_layout_digest: str
    schema_version: int = EXTERNAL_CACHE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("model_id", "model_revision", "tokenizer_revision", "kv_dtype"):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")
        if self.tensor_parallel_size <= 0 or self.world_size <= 0:
            raise ValueError("parallel sizes must be positive")
        _validate_digest("parallel_config_digest", self.parallel_config_digest)
        _validate_digest("group_layout_digest", self.group_layout_digest)
        if self.schema_version != EXTERNAL_CACHE_SCHEMA_VERSION:
            raise ValueError(f"unsupported external cache schema version {self.schema_version}")

    @classmethod
    def for_deepseek_v4(
        cls,
        *,
        model_id: str,
        model_revision: str,
        tokenizer_revision: str,
        kv_dtype: str,
        tensor_parallel_size: int,
        world_size: int,
        parallel_config: Mapping[str, Any],
        mtp_enabled: bool,
        group_specs: Sequence[KVCacheGroupSpec],
    ) -> "ExternalCacheNamespace":
        groups = _validate_deepseek_groups(group_specs)
        layout = {"cache_family": "deepseek_v4", "groups": [_group_layout(group) for group in groups]}
        return cls(
            model_id=model_id,
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision,
            kv_dtype=kv_dtype,
            tensor_parallel_size=tensor_parallel_size,
            world_size=world_size,
            parallel_config_digest=_digest_json(parallel_config),
            mtp_enabled=mtp_enabled,
            group_layout_digest=_digest_json(layout),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_layout_digest": self.group_layout_digest,
            "kv_dtype": self.kv_dtype,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "mtp_enabled": self.mtp_enabled,
            "parallel_config_digest": self.parallel_config_digest,
            "schema_version": self.schema_version,
            "tensor_parallel_size": self.tensor_parallel_size,
            "tokenizer_revision": self.tokenizer_revision,
            "world_size": self.world_size,
        }

    @property
    def digest(self) -> str:
        return _digest_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class ExternalKVObjectSpec:
    """One immutable cache page referenced by a committed checkpoint."""

    key: str
    group_name: str
    logical_block_index: int
    block_digest: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not self.key or len(self.key.encode("utf-8")) > 512:
            raise ValueError("external cache object key must contain 1 to 512 bytes")
        if not _KEY_COMPONENT.fullmatch(self.group_name):
            raise ValueError(f"invalid external cache group name {self.group_name!r}")
        if self.logical_block_index < 0:
            raise ValueError("logical_block_index must be non-negative")
        _validate_digest("block_digest", self.block_digest)
        if self.size_bytes <= 0:
            raise ValueError("external cache object size must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_digest": self.block_digest,
            "group_name": self.group_name,
            "key": self.key,
            "logical_block_index": self.logical_block_index,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class ExternalKVCheckpointManifest:
    """Atomic commit record for one recoverable DeepSeek prefix boundary."""

    namespace_digest: str
    prefix_digest: str
    token_count: int
    source_partition: int
    objects: tuple[ExternalKVObjectSpec, ...]
    schema_version: int = EXTERNAL_CACHE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_digest("namespace_digest", self.namespace_digest)
        _validate_digest("prefix_digest", self.prefix_digest)
        if self.token_count <= 0:
            raise ValueError("checkpoint token_count must be positive")
        if self.source_partition < 0:
            raise ValueError("source_partition must be non-negative")
        if not self.objects:
            raise ValueError("checkpoint must reference at least one cache object")
        keys = [item.key for item in self.objects]
        if len(keys) != len(set(keys)):
            raise ValueError("checkpoint contains duplicate cache object keys")
        positions = [(item.group_name, item.logical_block_index) for item in self.objects]
        if len(positions) != len(set(positions)):
            raise ValueError("checkpoint contains duplicate group block positions")
        if self.schema_version != EXTERNAL_CACHE_SCHEMA_VERSION:
            raise ValueError(f"unsupported external cache schema version {self.schema_version}")

    @property
    def manifest_key(self) -> str:
        return checkpoint_manifest_key(
            namespace_digest=self.namespace_digest,
            prefix_digest=self.prefix_digest,
            source_partition=self.source_partition,
            token_count=self.token_count,
            schema_version=self.schema_version,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "namespace_digest": self.namespace_digest,
            "objects": [item.to_dict() for item in self.objects],
            "prefix_digest": self.prefix_digest,
            "schema_version": self.schema_version,
            "source_partition": self.source_partition,
            "token_count": self.token_count,
        }

    def to_bytes(self) -> bytes:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_bytes(cls, payload: bytes) -> "ExternalKVCheckpointManifest":
        try:
            data = json.loads(payload)
        except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid external cache checkpoint manifest") from exc
        if not isinstance(data, dict):
            raise ValueError("external cache checkpoint manifest must be a JSON object")
        expected = {
            "namespace_digest",
            "objects",
            "prefix_digest",
            "schema_version",
            "source_partition",
            "token_count",
        }
        if set(data) != expected or not isinstance(data["objects"], list):
            raise ValueError("external cache checkpoint manifest has an invalid schema")
        try:
            objects = tuple(ExternalKVObjectSpec(**item) for item in data["objects"])
            return cls(
                namespace_digest=data["namespace_digest"],
                prefix_digest=data["prefix_digest"],
                token_count=data["token_count"],
                source_partition=data["source_partition"],
                objects=objects,
                schema_version=data["schema_version"],
            )
        except (TypeError, KeyError, ValueError) as exc:
            raise ValueError("external cache checkpoint manifest has invalid fields") from exc


def _validate_deepseek_groups(group_specs: Sequence[KVCacheGroupSpec]) -> tuple[KVCacheGroupSpec, ...]:
    by_name = {group.name: group for group in group_specs}
    if len(by_name) != len(group_specs):
        raise ValueError("DeepSeek cache group names must be unique")
    expected = set(DEEPSEEK_V4_CACHE_GROUPS)
    if set(by_name) != expected:
        missing = sorted(expected - set(by_name))
        extra = sorted(set(by_name) - expected)
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if extra:
            details.append("extra=" + ",".join(extra))
        raise ValueError("DeepSeek external cache groups do not match (" + "; ".join(details) + ")")
    return tuple(by_name[name] for name in DEEPSEEK_V4_CACHE_GROUPS)


def checkpoint_manifest_key(
    *,
    namespace_digest: str,
    prefix_digest: str,
    source_partition: int,
    token_count: int,
    schema_version: int = EXTERNAL_CACHE_SCHEMA_VERSION,
) -> str:
    """Build the commit-marker key queried by scheduler-side clients."""
    _validate_digest("namespace_digest", namespace_digest)
    _validate_digest("prefix_digest", prefix_digest)
    if source_partition < 0:
        raise ValueError("source_partition must be non-negative")
    if token_count <= 0:
        raise ValueError("checkpoint token_count must be positive")
    if schema_version != EXTERNAL_CACHE_SCHEMA_VERSION:
        raise ValueError(f"unsupported external cache schema version {schema_version}")
    return (
        f"pypto/dsv4/v{schema_version}/{namespace_digest}/"
        f"checkpoint/{prefix_digest}/p{source_partition}/t{token_count}"
    )


def checkpoint_alignment(group_specs: Sequence[KVCacheGroupSpec]) -> int:
    """Return the shared token boundary on which all groups are restorable."""
    result = 1
    for group in group_specs:
        result = math.lcm(result, group.spec.token_capacity)
    return result


def latest_checkpoint_token_count(
    group_specs: Sequence[KVCacheGroupSpec],
    published_block_counts: Mapping[str, int],
) -> int:
    """Return the largest common prefix boundary safe to publish externally."""
    groups = _validate_deepseek_groups(group_specs)
    if set(published_block_counts) != set(DEEPSEEK_V4_CACHE_GROUPS):
        raise ValueError("published block counts do not cover every DeepSeek cache group")
    safe_tokens = min(
        int(published_block_counts[group.name]) * group.spec.token_capacity for group in groups
    )
    if safe_tokens < 0:
        raise ValueError("published block counts must be non-negative")
    alignment = checkpoint_alignment(groups)
    return safe_tokens - safe_tokens % alignment


def build_deepseek_checkpoint_manifest(
    namespace: ExternalCacheNamespace,
    *,
    prefix_digest: str,
    token_count: int,
    source_partition: int,
    group_specs: Sequence[KVCacheGroupSpec],
    group_block_hashes: Mapping[str, Sequence[int]],
) -> ExternalKVCheckpointManifest:
    """Describe every page needed to restore one strict grouped-cache hit."""
    groups = _validate_deepseek_groups(group_specs)
    _validate_digest("prefix_digest", prefix_digest)
    if token_count <= 0 or token_count % checkpoint_alignment(groups):
        raise ValueError("checkpoint token_count must be a positive shared group boundary")
    if source_partition < 0 or source_partition >= groups[0].num_partitions:
        raise ValueError("source_partition is outside the DeepSeek cache partition range")
    if any(group.num_partitions != groups[0].num_partitions for group in groups):
        raise ValueError("DeepSeek cache groups must use the same partition count")
    if set(group_block_hashes) != set(DEEPSEEK_V4_CACHE_GROUPS):
        raise ValueError("block hashes do not cover every DeepSeek cache group")

    objects = []
    for group in groups:
        token_capacity = group.spec.token_capacity
        end_block = token_count // token_capacity
        hashes = group_block_hashes[group.name]
        if end_block > len(hashes):
            raise ValueError(
                f"cache group {group.name!r} has {len(hashes)} hashes but checkpoint "
                f"requires {end_block}"
            )
        if group.sliding_window is None:
            start_block = 0
        else:
            tail_blocks = min(end_block, group.sliding_window // token_capacity)
            start_block = end_block - tail_blocks
        for block_index in range(start_block, end_block):
            block_hash = int(hashes[block_index])
            if block_hash < 0 or block_hash.bit_length() > 256:
                raise ValueError("cache block hashes must be unsigned 256-bit integers")
            block_digest = f"{block_hash:064x}"
            key = (
                f"pypto/dsv4/v{namespace.schema_version}/{namespace.digest}/data/"
                f"{group.name}/p{source_partition}/b{block_index}/{block_digest}"
            )
            objects.append(
                ExternalKVObjectSpec(
                    key=key,
                    group_name=group.name,
                    logical_block_index=block_index,
                    block_digest=block_digest,
                    size_bytes=group.spec.page_size_bytes,
                )
            )

    return ExternalKVCheckpointManifest(
        namespace_digest=namespace.digest,
        prefix_digest=prefix_digest,
        token_count=token_count,
        source_partition=source_partition,
        objects=tuple(objects),
    )
