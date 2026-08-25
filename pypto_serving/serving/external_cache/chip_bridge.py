# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import base64
import json
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any, Callable, Sequence

from pypto_serving.serving.external_cache.config import ExternalPrefixCacheConfig, MooncakeClientConfig
from pypto_serving.serving.external_cache.connector import ExternalKVTransferCompletion
from pypto_serving.serving.external_cache.mooncake import create_mooncake_backend
from pypto_serving.serving.external_cache.protocol import ExternalKVBuffer, ExternalKVTransfer


_EXTENSION_NAME = "pypto-serving.mooncake-v1"
_PENDING_MARKER = "PYPTO_EXTERNAL_PENDING:"
_FAILED_MARKER = "PYPTO_EXTERNAL_FAILED:"
_CHILD_STATES: dict[int, "_ChildState"] = {}


@dataclass(slots=True)
class _ChildState:
    connector: Any
    terminal: dict[str, ExternalKVTransferCompletion]


def install_mooncake_chip_extension() -> None:
    """Install the handler before Simpler forks the chip processes."""
    from simpler import register_chip_control_extension

    register_chip_control_extension(_EXTENSION_NAME, _mooncake_chip_handler)


def _decode_transfers(items: Sequence[dict[str, Any]]) -> tuple[ExternalKVTransfer, ...]:
    return tuple(
        ExternalKVTransfer(
            str(item["key"]),
            tuple(
                ExternalKVBuffer(int(buffer["address"]), int(buffer["size_bytes"]))
                for buffer in item["buffers"]
            ),
        )
        for item in items
    )


def _mooncake_chip_handler(_chip_worker, payload: bytes, device_id: int) -> str | None:
    command = json.loads(payload)
    operation = command["operation"]
    if operation == "init":
        if device_id in _CHILD_STATES:
            return None
        backend = create_mooncake_backend(
            MooncakeClientConfig.from_mapping(command["mooncake"]),
            contribute_memory=True,
            storage_rank=device_id,
        )
        from pypto_serving.serving.external_cache.connector import ExternalKVWorkerConnector

        _CHILD_STATES[device_id] = _ChildState(
            connector=ExternalKVWorkerConnector(
                backend,
                max_workers=int(command["transfer_concurrency"]),
            ),
            terminal={},
        )
        return None
    if operation == "close":
        state = _CHILD_STATES.pop(device_id, None)
        if state is not None:
            state.connector.close()
        return None

    target_device_id = int(command["device_id"])
    if device_id != target_device_id:
        return None
    state = _CHILD_STATES.get(device_id)
    if state is None:
        return f"{_FAILED_MARKER}Mooncake chip client is not initialized"
    job_id = str(command["job_id"])
    if operation == "register":
        state.connector.register_buffers(
            tuple(
                ExternalKVBuffer(int(buffer["address"]), int(buffer["size_bytes"]))
                for buffer in command["buffers"]
            )
        )
        return None
    if operation in ("load", "save"):
        transfers = _decode_transfers(command["transfers"])
        if operation == "load":
            state.connector.start_load(job_id, transfers)
        else:
            manifest_payload = base64.b64decode(command["manifest_payload"], validate=True)
            state.connector.start_save(
                job_id,
                transfers,
                manifest_key=str(command["manifest_key"]),
                manifest_payload=manifest_payload,
            )
        return None
    if operation == "cancel":
        state.connector.cancel(job_id)
        return None
    if operation != "poll":
        return f"{_FAILED_MARKER}unknown Mooncake chip operation {operation!r}"

    for completion in state.connector.poll_completed():
        state.terminal[completion.job_id] = completion
    completion = state.terminal.pop(job_id, None)
    if completion is None:
        return f"{_PENDING_MARKER}{job_id}"
    if not completion.succeeded:
        return f"{_FAILED_MARKER}{completion.error or 'Mooncake transfer failed'}"
    return None


@dataclass(slots=True)
class _ParentJob:
    operation: str
    partition: int
    future: Future
    cancelled: bool = False


class SimplerMooncakeConnector:
    """Drive child-local Mooncake clients through short Simpler controls."""

    def __init__(
        self,
        control: Callable[[str, bytes], None],
        config: ExternalPrefixCacheConfig,
        device_ids: Sequence[int],
    ) -> None:
        self._control = control
        self._executor = ThreadPoolExecutor(
            max_workers=config.transfer_concurrency,
            thread_name_prefix="external-kv-chip",
        )
        self._jobs: dict[str, _ParentJob] = {}
        self._lock = threading.Lock()
        self._closed = False
        self._device_ids = tuple(int(device_id) for device_id in device_ids)
        if not self._device_ids:
            raise ValueError("external cache chip connector requires at least one device")
        try:
            self._invoke(
                {
                    "operation": "init",
                    "mooncake": asdict(config.mooncake),
                    "transfer_concurrency": config.transfer_concurrency,
                }
            )
        except Exception:
            try:
                self._invoke({"operation": "close"})
            except Exception:
                pass
            finally:
                self._closed = True
                self._executor.shutdown(wait=True, cancel_futures=True)
            raise

    def register_buffers(
        self,
        buffers: Sequence[ExternalKVBuffer],
        *,
        partition: int,
    ) -> None:
        """Register full allocations in the address-owning chip child."""
        if not buffers:
            raise ValueError("external cache registration requires device buffers")
        self._invoke(
            {
                "operation": "register",
                "job_id": "register",
                "partition": partition,
                "device_id": self._device_id(partition),
                "buffers": [
                    {"address": buffer.address, "size_bytes": buffer.size_bytes}
                    for buffer in buffers
                ],
            }
        )

    def start_load(
        self,
        job_id: str,
        transfers: Sequence[ExternalKVTransfer],
        *,
        partition: int,
    ) -> None:
        self._start(job_id, "load", partition, transfers)

    def start_save(
        self,
        job_id: str,
        transfers: Sequence[ExternalKVTransfer],
        *,
        manifest_key: str,
        manifest_payload: bytes,
        partition: int,
    ) -> None:
        self._start(
            job_id,
            "save",
            partition,
            transfers,
            manifest_key=manifest_key,
            manifest_payload=manifest_payload,
        )

    def _start(
        self,
        job_id: str,
        operation: str,
        partition: int,
        transfers: Sequence[ExternalKVTransfer],
        *,
        manifest_key: str | None = None,
        manifest_payload: bytes | None = None,
    ) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("external cache chip connector is closed")
            if job_id in self._jobs:
                raise ValueError(f"external cache job {job_id!r} already exists")
            future = self._executor.submit(
                self._run_job,
                job_id,
                operation,
                partition,
                tuple(transfers),
                manifest_key,
                manifest_payload,
            )
            self._jobs[job_id] = _ParentJob(operation, partition, future)

    def _run_job(
        self,
        job_id: str,
        operation: str,
        partition: int,
        transfers: tuple[ExternalKVTransfer, ...],
        manifest_key: str | None,
        manifest_payload: bytes | None,
    ) -> tuple[int, ...]:
        command: dict[str, Any] = {
            "operation": operation,
            "job_id": job_id,
            "partition": partition,
            "device_id": self._device_id(partition),
            "transfers": [
                {
                    "key": transfer.key,
                    "buffers": [
                        {"address": buffer.address, "size_bytes": buffer.size_bytes}
                        for buffer in transfer.buffers
                    ],
                }
                for transfer in transfers
            ],
        }
        if operation == "save":
            command["manifest_key"] = manifest_key
            command["manifest_payload"] = base64.b64encode(manifest_payload or b"").decode("ascii")
        self._invoke_retry(command)
        cancel_sent = False
        while True:
            with self._lock:
                job = self._jobs.get(job_id)
                cancelled = bool(job is not None and job.cancelled)
            if cancelled and not cancel_sent:
                self._invoke_retry(
                    {
                        "operation": "cancel",
                        "job_id": job_id,
                        "partition": partition,
                        "device_id": self._device_id(partition),
                    }
                )
                cancel_sent = True
            try:
                self._invoke_retry(
                    {
                        "operation": "poll",
                        "job_id": job_id,
                        "partition": partition,
                        "device_id": self._device_id(partition),
                    }
                )
            except RuntimeError as exc:
                if _PENDING_MARKER not in str(exc):
                    raise
                time.sleep(0.005)
                continue
            if cancelled:
                raise RuntimeError("external cache transfer cancelled")
            result_count = len(transfers) + (1 if operation == "save" else 0)
            return (0,) * result_count

    def _invoke_retry(self, command: dict[str, Any]) -> None:
        while True:
            try:
                self._invoke(command)
                return
            except RuntimeError as exc:
                if "run(s) still in flight" not in str(exc):
                    raise
                time.sleep(0.001)

    def _invoke(self, command: dict[str, Any]) -> None:
        self._control(_EXTENSION_NAME, json.dumps(command, separators=(",", ":")).encode("utf-8"))

    def _device_id(self, partition: int) -> int:
        if not 0 <= partition < len(self._device_ids):
            raise ValueError(f"external cache partition {partition} is out of range")
        return self._device_ids[partition]

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            job.cancelled = True
            return True

    def poll_completed(self) -> tuple[ExternalKVTransferCompletion, ...]:
        completed = []
        with self._lock:
            done_ids = [job_id for job_id, job in self._jobs.items() if job.future.done()]
            for job_id in done_ids:
                job = self._jobs.pop(job_id)
                try:
                    status_codes = tuple(job.future.result())
                    error = None
                except Exception as exc:
                    status_codes = ()
                    error = str(exc)
                completed.append(
                    ExternalKVTransferCompletion(
                        job_id=job_id,
                        operation=job.operation,
                        succeeded=not job.cancelled and error is None,
                        status_codes=status_codes,
                        error=error,
                        cancelled=job.cancelled,
                    )
                )
        return tuple(completed)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for job in self._jobs.values():
                job.cancelled = True
        self._executor.shutdown(wait=True, cancel_futures=False)
        self._invoke({"operation": "close"})
