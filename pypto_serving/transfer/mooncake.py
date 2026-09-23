# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Mooncake's owner-local implementation. Addresses never enter public events."""

from __future__ import annotations

from dataclasses import asdict
import importlib
import os
import threading
from typing import Callable

from pypto_serving.tools.profile import profile_span

from .errors import ErrorCode, TransferError, TransferFailure
from .types import CompletionCertainty as Certainty, ProviderCapabilities, ProviderTransferTask, RegionLease


class MooncakeTransferProvider:
    capabilities = ProviderCapabilities()

    def __init__(self, hostname: str, *, engine_factory: Callable | None = None):
        self._thread_id = threading.get_ident()
        self._pid = os.getpid()
        self._poisoned = False
        self._regions: dict[str, tuple[RegionLease, int]] = {}
        self._destinations: dict[tuple[str, str, int, str], dict] = {}
        self._last_leases: dict[str, int] = {}
        if engine_factory is None:
            engine_factory = importlib.import_module("mooncake.engine").TransferEngine
        self._engine = engine_factory()
        try:
            self._check_code(self._engine.initialize(hostname, "P2PHANDSHAKE", "ascend", ""), Certainty.NOT_SUBMITTED)
            port = int(self._engine.get_rpc_port())
            if not 0 < port < 65536:
                raise ValueError("invalid engine endpoint")
            self._endpoint = f"{hostname}:{port}"
        except BaseException:
            self._poisoned = True
            raise

    def _check(self):
        if self._pid != os.getpid() or self._thread_id != threading.get_ident() or self._poisoned:
            raise TransferFailure(TransferError(ErrorCode.POISONED, Certainty.NOT_SUBMITTED))

    def _check_code(self, code, certainty):
        if type(code) is not int or code != 0:
            self._poisoned = True
            raise TransferFailure(TransferError(ErrorCode.BACKEND_FAILURE, certainty,
                                                backend_code=code if type(code) is int else None))

    def register(self, lease: RegionLease, address: int) -> dict:
        self._check()
        if (type(address) is not int or address <= 0 or address % 64 or lease.extent % 64
                or address + lease.extent > 1 << 64 or lease.region_id in self._regions
                or lease.lease <= self._last_leases.get(lease.region_id, -1)):
            raise ValueError("invalid or duplicate registration")
        try:
            self._check_code(self._engine.register_memory(address, lease.extent), Certainty.NOT_SUBMITTED)
        except BaseException:
            self._poisoned = True
            raise
        self._regions[lease.region_id] = (lease, address)
        self._last_leases[lease.region_id] = lease.lease
        return {"lease": asdict(lease), "endpoint": self._endpoint, "address": address}

    def install_destination(self, lease: RegionLease, envelope: dict):
        self._check()
        if envelope.get("lease") != asdict(lease):
            raise ValueError("endpoint envelope lease mismatch")
        address = envelope.get("address")
        if type(address) is not int or address <= 0 or address % 64 or address + lease.extent > 1 << 64:
            raise ValueError("invalid private destination address")
        if not isinstance(envelope.get("endpoint"), str) or not envelope["endpoint"]:
            raise ValueError("invalid private destination endpoint")
        previous = self._destinations.get(lease.registration_key)
        if previous is not None:
            old = previous["lease"]
            if (lease.owner.generation < old["owner"]["generation"]
                    or lease.owner.endpoint_generation < old["owner"]["endpoint_generation"]
                    or (lease.owner.generation == old["owner"]["generation"] and lease.lease < old["lease"])
                    or (old == asdict(lease) and previous != envelope)):
                raise ValueError("stale or inconsistent destination registration")
        self._destinations[lease.registration_key] = dict(envelope)

    def write(self, task: ProviderTransferTask, on_submitted: Callable[[], None] = lambda: None) -> None:
        self._check()
        task.validate(self.capabilities)
        local, remote, lengths, endpoints = [], [], [], set()
        for segment in task.segments:
            lease, address = self._regions[segment.source.region_id]
            envelope = self._destinations[segment.destination.registration_key]
            if lease != segment.source or envelope["lease"] != asdict(segment.destination):
                raise ValueError("stale region lease")
            local.append(address + segment.source_offset)
            remote.append(envelope["address"] + segment.destination_offset)
            endpoints.add(envelope["endpoint"])
            lengths.append(segment.length)
        if len(endpoints) != 1:
            raise ValueError("one batch must target one endpoint")
        on_submitted()
        try:
            # This span brackets the synchronous native Mooncake call in the
            # owner-local thread.  It carries sizes only: endpoints and native
            # addresses must never enter the merged application trace.
            with profile_span(
                "MooncakeTransferEngine.batch_transfer_sync_write",
                cat="native-transfer",
                args={"bytes": sum(lengths), "segments": len(lengths)},
            ):
                self._check_code(
                    self._engine.batch_transfer_sync_write(
                        next(iter(endpoints)), local, remote, lengths
                    ),
                    Certainty.UNKNOWN,
                )
        except BaseException as exc:
            self._poisoned = True
            if isinstance(exc, TransferFailure):
                raise
            raise TransferFailure(TransferError(ErrorCode.BACKEND_FAILURE, Certainty.UNKNOWN)) from None

    def release(self):
        """Caller orders sender release before receiver release after all writes complete."""
        self._check()
        for region_id, (_, address) in tuple(self._regions.items()):
            try:
                self._check_code(self._engine.unregister_memory(address), Certainty.UNKNOWN)
            except BaseException:
                self._poisoned = True
                raise
            del self._regions[region_id]
        self._destinations.clear()
