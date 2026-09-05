# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import inspect
import os
import socket
from collections.abc import Callable, Sequence
from typing import Any

from pypto_serving.serving.external_cache.config import MooncakeClientConfig
from pypto_serving.serving.external_cache.protocol import ExternalKVBuffer, ExternalKVTransfer


class MooncakeStoreBackend:
    """Thin adapter over Mooncake Store's multi-buffer object API.

    Store construction belongs in the chip child after NPU initialization.
    Accepting the initialized store and registration callback here keeps this
    module importable in scheduler processes where Mooncake is not installed.
    """

    def __init__(
        self,
        store: Any,
        *,
        register_buffer: Callable[[list[int], list[int]], None],
        unregister_buffer: Callable[[list[int]], None] | None = None,
        replicate_config: Any | None = None,
    ) -> None:
        required = ("batch_is_exist", "batch_put_from_multi_buffers", "batch_get_into_multi_buffers")
        missing = [name for name in required if not callable(getattr(store, name, None))]
        if missing:
            raise TypeError("Mooncake store is missing methods: " + ", ".join(missing))
        self._store = store
        self._register_buffer = register_buffer
        self._unregister_buffer = unregister_buffer
        self._replicate_config = replicate_config
        self._registered_buffers: dict[int, int] = {}

    def register_buffers(self, buffers: Sequence[ExternalKVBuffer]) -> None:
        for buffer in buffers:
            registered_size = self._registered_buffers.get(buffer.address)
            if registered_size is not None and registered_size != buffer.size_bytes:
                raise ValueError(
                    f"Mooncake buffer 0x{buffer.address:x} was already registered with "
                    f"{registered_size} bytes, not {buffer.size_bytes}"
                )
        new_buffers = [buffer for buffer in buffers if buffer.address not in self._registered_buffers]
        if not new_buffers:
            return
        self._register_buffer(
            [buffer.address for buffer in new_buffers],
            [buffer.size_bytes for buffer in new_buffers],
        )
        self._registered_buffers.update(
            (buffer.address, buffer.size_bytes) for buffer in new_buffers
        )

    def exists(self, keys: Sequence[str]) -> tuple[bool, ...]:
        if not keys:
            return ()
        result = tuple(
            value is True or value == 1 for value in self._store.batch_is_exist(list(keys))
        )
        self._validate_result_count("batch_is_exist", len(keys), result)
        return result

    def put(self, transfers: Sequence[ExternalKVTransfer]) -> tuple[int, ...]:
        if not transfers:
            return ()
        keys, addresses, sizes = self._unpack(transfers)
        args = (keys, addresses, sizes)
        if self._replicate_config is not None:
            args += (self._replicate_config,)
        result = tuple(int(value) for value in self._store.batch_put_from_multi_buffers(*args))
        self._validate_result_count("batch_put_from_multi_buffers", len(transfers), result)
        return result

    def put_bytes(self, key: str, payload: bytes) -> int:
        if not key or not payload:
            raise ValueError("Mooncake byte object key and payload must not be empty")
        return int(self._store.put(key, payload))

    def get(self, transfers: Sequence[ExternalKVTransfer]) -> tuple[int, ...]:
        if not transfers:
            return ()
        keys, addresses, sizes = self._unpack(transfers)
        result = tuple(
            int(value)
            for value in self._store.batch_get_into_multi_buffers(keys, addresses, sizes)
        )
        self._validate_result_count("batch_get_into_multi_buffers", len(transfers), result)
        return result

    def close(self) -> None:
        try:
            if self._unregister_buffer is not None and self._registered_buffers:
                self._unregister_buffer(sorted(self._registered_buffers))
                self._registered_buffers.clear()
        finally:
            close = getattr(self._store, "close", None)
            if callable(close):
                close()

    @staticmethod
    def _unpack(
        transfers: Sequence[ExternalKVTransfer],
    ) -> tuple[list[str], list[list[int]], list[list[int]]]:
        return (
            [transfer.key for transfer in transfers],
            [[buffer.address for buffer in transfer.buffers] for transfer in transfers],
            [[buffer.size_bytes for buffer in transfer.buffers] for transfer in transfers],
        )

    @staticmethod
    def _validate_result_count(operation: str, expected: int, result: Sequence[object]) -> None:
        if len(result) != expected:
            raise RuntimeError(
                f"Mooncake {operation} returned {len(result)} results for {expected} objects"
            )


def create_mooncake_backend(
    config: MooncakeClientConfig,
    *,
    contribute_memory: bool,
    storage_rank: int | None = None,
) -> MooncakeStoreBackend:
    """Create one Mooncake requester without importing Mooncake at module load."""
    try:
        import mooncake.store as mooncake_store
    except ImportError as exc:
        raise RuntimeError(
            "Mooncake Store Python bindings are required for external prefix caching"
        ) from exc

    replicate_config = None
    if contribute_memory:
        replicate_config_cls = getattr(mooncake_store, "ReplicateConfig", None)
        if replicate_config_cls is None:
            raise RuntimeError(
                "Mooncake Store binding does not expose ReplicateConfig required for "
                "rank-local Ascend storage placement"
            )
        replicate_config = replicate_config_cls()

    store = mooncake_store.MooncakeDistributedStore()
    hostname = config.local_hostname or _local_ip()
    global_segment_size = config.global_segment_size if contribute_memory else 0
    local_buffer_size = config.local_buffer_size if contribute_memory else 0
    protocol = config.protocol
    device_name = config.device_name
    setup_kwargs: dict[str, object] = {}
    transfer_engine = None
    if config.protocol == "ascend":
        if config.ascend_buffer_pool:
            # Mooncake reads this process-wide ADXL option during engine initialization.
            os.environ["ASCEND_BUFFER_POOL"] = config.ascend_buffer_pool
        try:
            from mooncake.engine import TransferEngine
        except ImportError as exc:
            raise RuntimeError(
                "Mooncake TransferEngine Python bindings with Ascend support are required "
                "for external prefix caching"
            ) from exc
        transfer_engine = TransferEngine()
        status = int(
            transfer_engine.initialize(
                hostname,
                "P2PHANDSHAKE",
                config.protocol,
                config.device_name,
            )
        )
        if status != 0:
            raise RuntimeError(f"Mooncake Ascend TransferEngine setup failed with status {status}")
        if not _setup_supports_parameter(store.setup, "engine"):
            raise RuntimeError(
                "installed Mooncake Store binding cannot accept an external TransferEngine"
            )
        hostname = f"{hostname}:{int(transfer_engine.get_rpc_port())}"
        setup_kwargs["engine"] = transfer_engine.get_engine()
    if replicate_config is not None:
        replicate_config.preferred_segment = hostname
        replicate_config.prefer_alloc_in_same_node = True
    if config.tenant_id != "default":
        if not _setup_supports_parameter(store.setup, "tenant_id"):
            raise RuntimeError(
                "external cache config uses a non-default tenant, but the installed "
                "Mooncake Store binding does not expose tenant_id"
            )
        setup_kwargs["tenant_id"] = config.tenant_id
    if config.enable_ssd_offload and contribute_memory:
        if not _setup_supports_parameter(store.setup, "enable_ssd_offload"):
            raise RuntimeError(
                "external cache enables SSD offload, but the installed Mooncake Store binding "
                "does not expose enable_ssd_offload; upgrade Mooncake or disable SSD offload"
            )
        ssd_path = config.ssd_offload_path
        if contribute_memory and storage_rank is not None:
            ssd_path = os.path.join(ssd_path, f"rank_{storage_rank}")
            try:
                os.makedirs(ssd_path, exist_ok=True)
            except OSError as exc:
                raise RuntimeError(
                    f"cannot create Mooncake SSD offload directory {ssd_path!r}: {exc}"
                ) from exc
        setup_kwargs = {
            **setup_kwargs,
            "enable_ssd_offload": True,
            "ssd_offload_path": ssd_path,
        }
    setup_args = (
        hostname,
        config.metadata_server,
        global_segment_size,
        local_buffer_size,
        protocol,
        device_name,
        config.master_server_address,
    )
    result = store.setup(*setup_args, **setup_kwargs)
    if result != 0:
        raise RuntimeError(f"Mooncake Store setup failed with status {result}")

    def register_buffers(addresses: list[int], sizes: list[int]) -> None:
        if transfer_engine is None or not contribute_memory:
            raise RuntimeError("Mooncake control client cannot register device buffers")
        registered = []
        try:
            for address, size in zip(addresses, sizes, strict=True):
                status = int(transfer_engine.register_memory(address, size))
                if status != 0:
                    raise RuntimeError(
                        f"Mooncake failed to register buffer 0x{address:x} ({size} bytes): {status}"
                    )
                registered.append(address)
        except Exception:
            for address in reversed(registered):
                transfer_engine.unregister_memory(address)
            raise

    def unregister_buffers(addresses: list[int]) -> None:
        if transfer_engine is None:
            return
        failures = []
        for address in addresses:
            status = int(transfer_engine.unregister_memory(address))
            if status != 0:
                failures.append((address, status))
        if failures:
            details = ", ".join(f"0x{address:x}:{status}" for address, status in failures)
            raise RuntimeError(f"Mooncake failed to unregister buffers: {details}")

    return MooncakeStoreBackend(
        store,
        register_buffer=register_buffers,
        unregister_buffer=unregister_buffers if transfer_engine is not None else None,
        replicate_config=replicate_config,
    )


def _setup_supports_parameter(setup: Callable[..., object], name: str) -> bool:
    try:
        return name in inspect.signature(setup).parameters
    except (TypeError, ValueError):
        return name in (setup.__doc__ or "")


def _local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return str(sock.getsockname()[0])
    except OSError:
        return "127.0.0.1"
