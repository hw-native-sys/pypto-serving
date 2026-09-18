# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Sequence

from pypto_serving.serving.external_cache.manifest import (
    ExternalCacheNamespace,
    checkpoint_prefix_digest,
    checkpoint_manifest_key,
)
from pypto_serving.serving.external_cache.protocol import (
    ExternalKVBackend,
    ExternalKVBuffer,
    ExternalKVTransfer,
)


@dataclass(frozen=True, slots=True)
class ExternalKVLookupResult:
    """Longest committed external prefix selected by the scheduler."""

    token_count: int
    source_partition: int
    prefix_digest: str
    manifest_key: str


@dataclass(frozen=True, slots=True)
class ExternalKVPageAssignment:
    """Map one immutable external object to a physical HBM page."""

    key: str
    group_name: str
    logical_block_index: int
    physical_block_id: int
    size_bytes: int


@dataclass(frozen=True, slots=True)
class ExternalKVLoadRequest:
    """Scheduler-to-worker request for one atomic DeepSeek checkpoint load."""

    job_id: str
    request_id: str
    manifest_key: str
    checkpoint_token_count: int
    source_partition: int
    destination_partition: int
    pages: tuple[ExternalKVPageAssignment, ...]

    def __post_init__(self) -> None:
        if not self.job_id or not self.request_id:
            raise ValueError("external cache load identifiers must not be empty")
        if self.checkpoint_token_count <= 0:
            raise ValueError("external cache load token count must be positive")
        if self.source_partition < 0 or self.destination_partition < 0:
            raise ValueError("external cache load partitions must be non-negative")
        if not self.pages:
            raise ValueError("external cache load must contain at least one destination page")


@dataclass(frozen=True, slots=True)
class ExternalKVSaveRequest:
    """Worker request to publish data pages followed by one commit marker."""

    job_id: str
    request_id: str
    manifest_key: str
    manifest_payload: bytes
    checkpoint_token_count: int
    source_partition: int
    pages: tuple[ExternalKVPageAssignment, ...]

    def __post_init__(self) -> None:
        if not self.job_id or not self.request_id or not self.manifest_key:
            raise ValueError("external cache save identifiers must not be empty")
        if not self.manifest_payload:
            raise ValueError("external cache save manifest must not be empty")
        if self.checkpoint_token_count <= 0 or self.source_partition < 0:
            raise ValueError("external cache save checkpoint metadata is invalid")
        if not self.pages:
            raise ValueError("external cache save must contain at least one source page")


class ExternalPrefixCacheIndex:
    """Scheduler-side manifest lookup without access to device buffers."""

    def __init__(
        self,
        backend: ExternalKVBackend,
        namespace: ExternalCacheNamespace,
        *,
        alignment: int,
        num_partitions: int,
        min_tokens: int,
        load_timeout_ms: int = 30_000,
        save_timeout_ms: int = 30_000,
        max_pending_saves: int = 2,
        max_pending_save_blocks: int = 256,
        enable_save: bool = True,
    ) -> None:
        if alignment <= 0:
            raise ValueError("external cache alignment must be positive")
        if num_partitions <= 0:
            raise ValueError("external cache partition count must be positive")
        if min_tokens <= 0:
            raise ValueError("external cache minimum token count must be positive")
        if load_timeout_ms <= 0:
            raise ValueError("external cache load timeout must be positive")
        if save_timeout_ms <= 0:
            raise ValueError("external cache save timeout must be positive")
        if max_pending_saves <= 0 or max_pending_save_blocks <= 0:
            raise ValueError("external cache save limits must be positive")
        self._backend = backend
        self._namespace = namespace
        self._alignment = alignment
        self._num_partitions = num_partitions
        self._min_tokens = min_tokens
        self._load_timeout_seconds = load_timeout_ms / 1000
        self._save_timeout_seconds = save_timeout_ms / 1000
        self._max_pending_saves = int(max_pending_saves)
        self._max_pending_save_blocks = int(max_pending_save_blocks)
        self._enable_save = bool(enable_save)

    @property
    def namespace(self) -> ExternalCacheNamespace:
        return self._namespace

    @property
    def load_timeout_seconds(self) -> float:
        return self._load_timeout_seconds

    @property
    def save_timeout_seconds(self) -> float:
        return self._save_timeout_seconds

    @property
    def max_pending_saves(self) -> int:
        return self._max_pending_saves

    @property
    def max_pending_save_blocks(self) -> int:
        return self._max_pending_save_blocks

    @property
    def min_tokens(self) -> int:
        return self._min_tokens

    @property
    def enable_save(self) -> bool:
        return self._enable_save

    def lookup(
        self,
        token_ids: Sequence[int],
        *,
        local_hit_tokens: int,
        max_hit_tokens: int,
    ) -> ExternalKVLookupResult | None:
        """Find the longest manifest beyond the current local HBM hit."""
        upper = min(len(token_ids), int(max_hit_tokens))
        upper -= upper % self._alignment
        lower = max(int(local_hit_tokens), self._min_tokens - 1)
        candidates = list(range(upper, lower, -self._alignment))
        if not candidates:
            return None

        descriptions = []
        keys = []
        for token_count in candidates:
            prefix_digest = checkpoint_prefix_digest(
                token_ids,
                token_count,
                mtp_enabled=self._namespace.mtp_enabled,
            )
            for partition in range(self._num_partitions):
                key = checkpoint_manifest_key(
                    namespace_digest=self._namespace.digest,
                    prefix_digest=prefix_digest,
                    source_partition=partition,
                    token_count=token_count,
                )
                descriptions.append((token_count, partition, prefix_digest, key))
                keys.append(key)

        exists = self._backend.exists(keys)
        if len(exists) != len(keys):
            raise RuntimeError(
                f"external cache lookup returned {len(exists)} results for {len(keys)} keys"
            )
        for present, description in zip(exists, descriptions, strict=True):
            if present:
                token_count, partition, prefix_digest, key = description
                return ExternalKVLookupResult(token_count, partition, prefix_digest, key)
        return None


@dataclass(frozen=True, slots=True)
class ExternalKVTransferCompletion:
    """Terminal result for one asynchronous worker-side transfer job."""

    job_id: str
    operation: str
    succeeded: bool
    status_codes: tuple[int, ...]
    error: str | None = None
    cancelled: bool = False


@dataclass(slots=True)
class _PendingTransfer:
    operation: str
    future: Future
    cancelled: bool = False


class ExternalKVWorkerConnector:
    """Run blocking backend operations off the worker's device-control lane."""

    def __init__(self, backend: ExternalKVBackend, *, max_workers: int = 2) -> None:
        if max_workers <= 0:
            raise ValueError("external cache transfer concurrency must be positive")
        self._backend = backend
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="external-kv",
        )
        self._pending: dict[str, _PendingTransfer] = {}
        self._lock = threading.Lock()
        self._closed = False

    def register_buffers(self, buffers: Sequence[ExternalKVBuffer]) -> None:
        self._require_open()
        self._backend.register_buffers(buffers)

    def start_load(
        self,
        job_id: str,
        transfers: Sequence[ExternalKVTransfer],
        *,
        partition: int | None = None,
    ) -> None:
        del partition
        self._start(job_id, "load", self._backend.get, tuple(transfers), None)

    def start_save(
        self,
        job_id: str,
        transfers: Sequence[ExternalKVTransfer],
        *,
        manifest_key: str,
        manifest_payload: bytes,
        partition: int | None = None,
    ) -> None:
        del partition
        if not manifest_key or not manifest_payload:
            raise ValueError("external cache save requires a manifest key and payload")
        self._start(
            job_id,
            "save",
            self._backend.put,
            tuple(transfers),
            (manifest_key, bytes(manifest_payload)),
        )

    def _start(self, job_id, operation, method, transfers, manifest) -> None:
        if not job_id:
            raise ValueError("external cache job_id must not be empty")
        if not transfers:
            raise ValueError("external cache job must contain data objects")
        with self._lock:
            self._require_open()
            if job_id in self._pending:
                raise ValueError(f"external cache job {job_id!r} already exists")
            future = self._executor.submit(
                self._run_transfer,
                method,
                transfers,
                manifest,
            )
            self._pending[job_id] = _PendingTransfer(operation=operation, future=future)

    def _run_transfer(self, method, transfers, manifest) -> tuple[int, ...]:
        status_codes = (
            self._put_immutable_objects(method, transfers)
            if manifest is not None
            else tuple(int(code) for code in method(transfers))
        )
        self._validate_result_count(len(transfers), status_codes)
        if any(code < 0 for code in status_codes) or manifest is None:
            return status_codes
        manifest_key, manifest_payload = manifest
        if self._single_exists(manifest_key):
            manifest_status = 0
        else:
            manifest_status = int(self._backend.put_bytes(manifest_key, manifest_payload))
            if manifest_status < 0 and self._single_exists(manifest_key):
                manifest_status = 0
        return status_codes + (manifest_status,)

    def _put_immutable_objects(self, method, transfers) -> tuple[int, ...]:
        keys = tuple(transfer.key for transfer in transfers)
        present = tuple(self._backend.exists(keys))
        self._validate_result_count(len(transfers), present)
        missing_indices = [index for index, exists in enumerate(present) if not exists]
        if not missing_indices:
            return (0,) * len(transfers)
        missing = tuple(transfers[index] for index in missing_indices)
        written = tuple(int(code) for code in method(missing))
        self._validate_result_count(len(missing), written)
        status_codes = [0] * len(transfers)
        failed_indices = []
        for index, status in zip(missing_indices, written, strict=True):
            status_codes[index] = status
            if status < 0:
                failed_indices.append(index)
        if failed_indices:
            raced_keys = tuple(keys[index] for index in failed_indices)
            raced_present = tuple(self._backend.exists(raced_keys))
            self._validate_result_count(len(raced_keys), raced_present)
            for index, exists in zip(failed_indices, raced_present, strict=True):
                if exists:
                    status_codes[index] = 0
        return tuple(status_codes)

    def _single_exists(self, key: str) -> bool:
        result = tuple(self._backend.exists((key,)))
        self._validate_result_count(1, result)
        return bool(result[0])

    @staticmethod
    def _validate_result_count(expected: int, result: Sequence[object]) -> None:
        if len(result) != expected:
            raise RuntimeError(
                f"external cache backend returned {len(result)} results for {expected} objects"
            )

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            pending = self._pending.get(job_id)
            if pending is None:
                return False
            pending.cancelled = True
            pending.future.cancel()
            return True

    def poll_completed(self) -> tuple[ExternalKVTransferCompletion, ...]:
        completed = []
        with self._lock:
            done_ids = [job_id for job_id, pending in self._pending.items() if pending.future.done()]
            for job_id in done_ids:
                pending = self._pending.pop(job_id)
                try:
                    status_codes = tuple(pending.future.result())
                    error = None
                except Exception as exc:
                    status_codes = ()
                    error = str(exc)
                succeeded = (
                    not pending.cancelled
                    and error is None
                    and bool(status_codes)
                    and all(code >= 0 for code in status_codes)
                )
                completed.append(
                    ExternalKVTransferCompletion(
                        job_id=job_id,
                        operation=pending.operation,
                        succeeded=succeeded,
                        status_codes=status_codes,
                        error=error,
                        cancelled=pending.cancelled,
                    )
                )
        return tuple(completed)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)
        close = getattr(self._backend, "close", None)
        if callable(close):
            close()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("external cache worker connector is closed")
