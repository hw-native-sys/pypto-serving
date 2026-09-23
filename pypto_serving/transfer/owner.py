# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Explicit parent/chip-child service channel; no allocator interception."""

from __future__ import annotations

import ctypes
from dataclasses import asdict
import json
import os
import socket
import struct
import threading
import select
from collections import deque
from functools import partial, wraps
from typing import Callable

from pypto_serving.tools.profile import configure_profiler, get_profiler

from .errors import ErrorCode, TransferError, TransferFailure
from .mooncake import MooncakeTransferProvider
from .types import (CompletionCertainty as Certainty, OwnerRef, ProviderCapabilities,
                    ProviderTransferTask, RegionLease, Segment, TransferAttemptRef)

MAX_MESSAGE_BYTES = 4 << 20


def _serialized(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return call


def _send(channel: socket.socket, message: dict) -> None:
    wire = json.dumps(message, separators=(",", ":"), allow_nan=False).encode()
    if len(wire) > MAX_MESSAGE_BYTES:
        raise ValueError("owner command exceeds channel limit")
    channel.sendall(struct.pack("!I", len(wire)) + wire)


def _receive(channel: socket.socket, owner_fd=None) -> dict:
    def exact(length):
        data = bytearray()
        while len(data) < length:
            if owner_fd is not None:
                readable = select.select([channel, owner_fd], [], [], channel.gettimeout())[0]
                if owner_fd in readable and channel not in readable:
                    raise EOFError("owner process exited")
                if not readable:
                    raise TimeoutError("owner response deadline")
            part = channel.recv(length - len(data))
            if not part:
                raise EOFError("owner channel closed")
            data.extend(part)
        return bytes(data)
    length = struct.unpack("!I", exact(4))[0]
    if not 0 < length <= MAX_MESSAGE_BYTES:
        raise ValueError("invalid owner message length")
    value = json.loads(exact(length))
    if not isinstance(value, dict):
        raise ValueError("owner message must be an object")
    return value


def _lease(wire: dict) -> RegionLease:
    return RegionLease(**dict(wire, owner=OwnerRef(**wire["owner"])))


def _task(wire: dict) -> ProviderTransferTask:
    ref = wire["attempt"]
    attempt = TransferAttemptRef(**dict(ref, source=OwnerRef(**ref["source"]),
                                       destination=OwnerRef(**ref["destination"])))
    return ProviderTransferTask(attempt, tuple(Segment(**dict(
        s, source=_lease(s["source"]), destination=_lease(s["destination"]))) for s in wire["segments"]))


def _descriptor_wire(descriptor) -> dict:
    return {
        "owner": bytes(descriptor.identity.owner_instance_id).hex(),
        "buffer_id": int(descriptor.identity.buffer_id), "generation": int(descriptor.identity.generation),
        "address_space": int(descriptor.address_space), "access": int(descriptor.access),
        "backend_kind": int(descriptor.backend_kind), "nbytes": int(descriptor.nbytes),
        "body": bytes(descriptor.body).hex(), "owner_worker_path_id": int(descriptor.owner_worker_path_id),
    }


def _descriptor(wire: dict):
    from simpler.buffer import AccessMode, AddressSpace, BackendKind, BufferDescriptor, CanonicalIdentity
    if AddressSpace(wire["address_space"]) != AddressSpace.DEVICE:
        raise ValueError("owner service only accepts device memory")
    if BackendKind(wire["backend_kind"]) != BackendKind.DEVICE_MALLOC:
        raise ValueError("owner service only accepts malloc-backed resident arenas")
    return BufferDescriptor(
        CanonicalIdentity(bytes.fromhex(wire["owner"]), wire["buffer_id"], wire["generation"]),
        AddressSpace(wire["address_space"]), AccessMode(wire["access"]), BackendKind(wire["backend_kind"]),
        wire["nbytes"], bytes.fromhex(wire["body"]), wire["owner_worker_path_id"],
    )


class OwnerBridge:
    capabilities = ProviderCapabilities()

    def __init__(self, owner: OwnerRef, hostname: str, *, command_timeout: float = 40,
                 native_timeout: int = 30, on_owner_lost: Callable[[OwnerRef], None] | None = None):
        if not 5 <= native_timeout < command_timeout:
            raise ValueError("require 5 <= native timeout < command timeout")
        self.owner, self.hostname = owner, hostname
        self.command_timeout, self.native_timeout = command_timeout, native_timeout
        self._parent, self._child = socket.socketpair()
        self._parent.settimeout(command_timeout)
        self._lock = threading.RLock()
        self._sequence = 0
        self._poisoned = False
        self._closed = False
        self._retentions = []
        self._parent_pid = os.getpid()
        self.owner_pid = None
        self.process_handle = None
        self._monitor = None
        self._on_owner_lost = on_owner_lost
        self._profile_config = get_profiler(initially_active=False).config
        self.health_events = deque(maxlen=16)
        self.commands = deque(maxlen=64)
        self._sources = {}
        self._destinations = {}

    def factory(self, context):
        """Pass this bound method as the rank's chip_service_factory before prepare()."""
        if os.getpid() == self._parent_pid:
            raise RuntimeError("owner service must run inside the chip child")
        self._parent.close()
        return _OwnerService(
            context,
            self._child,
            self.owner,
            self.hostname,
            self.native_timeout,
            self._profile_config,
        )

    @_serialized
    def ready(self):
        self._child.close()
        reply = self._request("probe")
        self.owner_pid = reply["pid"]
        if self.owner_pid == self._parent_pid:
            raise RuntimeError("service is not in an owner child")
        from .supervisor import OwnerHealthMonitor, OwnerProcessHandle
        self.process_handle = OwnerProcessHandle(self.owner_pid)
        self._monitor = OwnerHealthMonitor(self.process_handle, self._owner_exited)
        return reply

    def _owner_exited(self):
        # Never wait for the command lock: its holder may be waiting for this dead owner.
        self._poisoned = True
        self.health_events.append({"stage": "OwnerExited", "owner": asdict(self.owner)})
        if self._on_owner_lost is not None:
            try:
                self._on_owner_lost(self.owner)
            except BaseException as exc:
                self.health_events.append({"stage": "HealthDeliveryFailed", "error_type": type(exc).__name__})

    def _request(self, operation, **payload):
        with self._lock:
            if self._poisoned or self._closed:
                raise TransferFailure(TransferError(ErrorCode.POISONED, Certainty.NOT_SUBMITTED))
            self._sequence += 1
            self.commands.append({"sequence": self._sequence, "operation": operation})
            try:
                _send(self._parent, dict(operation=operation, sequence=self._sequence,
                                         owner=asdict(self.owner), **payload))
                reply = _receive(self._parent, self.process_handle.fd if self.process_handle else None)
                if reply.get("sequence") != self._sequence or reply.get("owner") != asdict(self.owner):
                    raise ValueError("stale owner response")
                if not reply.get("ok"):
                    backend_code = reply.get("backend_code")
                    if backend_code is not None and type(backend_code) is not int:
                        raise ValueError("invalid backend error code")
                    error = TransferError(ErrorCode(reply["code"]), Certainty(reply["certainty"]),
                                          backend_code=backend_code)
                    if error.certainty == Certainty.UNKNOWN:
                        self._poisoned = True
                    raise TransferFailure(error)
                return reply
            except TransferFailure:
                raise
            except BaseException:
                self._poisoned = True
                raise TransferFailure(TransferError(ErrorCode.OWNER_LOST, Certainty.UNKNOWN)) from None

    @_serialized
    def register_tensor(self, runtime, tensor, lease: RegionLease) -> dict:
        if lease.owner != self.owner or tensor.nbytes != lease.extent:
            raise ValueError("tensor extent/owner mismatch")
        token, descriptor = runtime.retain_service_tensor(tensor, self.owner.rank_id)
        # Retain on any channel uncertainty; only confirmed unregistration permits release.
        self._retentions.append((runtime, token))
        envelope = self._request("register", token=token, descriptor=_descriptor_wire(descriptor),
                                 lease=asdict(lease))["envelope"]
        self._sources[lease.region_id] = lease
        return envelope

    def install_destination(self, lease: RegionLease, envelope: dict):
        self.install_destinations(((lease, envelope),))

    @_serialized
    def install_destinations(self, entries: tuple[tuple[RegionLease, dict], ...]):
        self._request("destinations", entries=[
            {"lease": asdict(lease), "envelope": envelope} for lease, envelope in entries
        ])
        for lease, _ in entries:
            self._destinations[lease.registration_key] = lease

    @_serialized
    def write(self, task: ProviderTransferTask, on_submitted: Callable[[], None] = lambda: None):
        if self._closed or self._poisoned:
            raise TransferFailure(TransferError(ErrorCode.POISONED, Certainty.NOT_SUBMITTED))
        if task.attempt.source != self.owner:
            raise TransferFailure(TransferError(ErrorCode.STALE_GENERATION, Certainty.NOT_SUBMITTED))
        task.validate(self.capabilities)
        for segment in task.segments:
            if (self._sources.get(segment.source.region_id) != segment.source
                    or self._destinations.get(segment.destination.registration_key) != segment.destination):
                raise TransferFailure(TransferError(ErrorCode.STALE_GENERATION, Certainty.NOT_SUBMITTED))
        # After publication, lost replies cannot prove whether native submit occurred.
        on_submitted()
        self._request("write", task=asdict(task))

    @_serialized
    def set_profile_active(self, active: bool) -> None:
        if type(active) is not bool:
            raise ValueError("owner profile state must be boolean")
        reply = self._request("profile", active=active)
        if reply.get("profile_error"):
            raise RuntimeError(str(reply["profile_error"]))
        if reply.get("profile_active") is not active:
            raise RuntimeError("owner profiler did not enter the requested state")

    @_serialized
    def release(self):
        self._request("release")
        for runtime, token in self._retentions:
            runtime.release_service_tensor(token)
        self._retentions.clear()
        self._sources.clear()
        self._destinations.clear()

    @_serialized
    def close(self):
        if self._closed:
            return
        self._request("stop")
        self._closed = True
        self._parent.close()
        if self._monitor is not None:
            self._monitor.close()
        if self.process_handle is not None:
            self.process_handle.close()

    @_serialized
    def close_after_owner_death(self):
        """Dispose IPC only after pidfd exit proof; never unregister or release allocation pins."""
        if self._closed:
            return
        if self.process_handle is None:
            raise RuntimeError("owner death must be observed before abandoning its channel")
        self.process_handle.wait(0)
        self._poisoned = True
        if self._monitor is not None:
            self._monitor.close()
        self._parent.close()
        self._child.close()
        self.process_handle.close()
        self._closed = True


class OwnerBridgeGroup:
    """Close sibling channels in each fork child so owner death is observable as EOF."""

    def __init__(self, bridges: tuple[OwnerBridge, ...]):
        self.bridges = bridges

    def _factory(self, rank, context):
        selected = self.bridges[rank]
        for bridge in self.bridges:
            if bridge is not selected:
                bridge._parent.close()
                bridge._child.close()
        return selected.factory(context)

    @property
    def factories(self):
        return {rank: partial(self._factory, rank) for rank in range(len(self.bridges))}


class _OwnerService:
    def __init__(
        self,
        context,
        channel,
        owner,
        hostname,
        native_timeout,
        profile_config,
    ):
        self.context, self.channel, self.owner = context, channel, owner
        self.hostname, self.native_timeout = hostname, native_timeout
        configure_profiler(
            profile_config,
            process_name=f"pd-transfer-owner-rank-{owner.rank_id}",
            initially_active=False,
        )
        self.tokens = []
        self.stopped = False
        self._acl = ctypes.CDLL("libascendcl.so")
        self._acl.aclrtGetCurrentContext.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self._acl.aclrtSetCurrentContext.argtypes = [ctypes.c_void_p]
        self._context = ctypes.c_void_p()
        if self._acl.aclrtGetCurrentContext(ctypes.byref(self._context)) != 0 or not self._context.value:
            raise RuntimeError("chip child has no ACL context")
        self._initialized = threading.Event()
        self.thread = threading.Thread(target=self._serve, name="owner-transfer", daemon=False)
        self.thread.start()
        if not self._initialized.wait(native_timeout + 5):
            raise TimeoutError("owner engine initialization did not complete")

    def _serve(self):
        sequence = 0
        try:
            if self._acl.aclrtSetCurrentContext(self._context) != 0:
                raise RuntimeError("cannot attach owner ACL context")
            os.environ["MC_TRANSFER_TIMEOUT"] = str(self.native_timeout)
            provider = MooncakeTransferProvider(self.hostname)
            self._initialized.set()
            while True:
                request = _receive(self.channel)
                if request.get("owner") != asdict(self.owner) or request.get("sequence") != sequence + 1:
                    raise RuntimeError("stale owner command")
                sequence += 1
                reply = dict(ok=True, sequence=sequence, owner=asdict(self.owner), pid=os.getpid(),
                             operation=request["operation"])
                operation = request["operation"]
                try:
                    if operation == "probe":
                        pass
                    elif operation == "register":
                        lease = _lease(request["lease"])
                        if lease.owner != self.owner:
                            raise ValueError("stale registration")
                        token = request["token"]
                        address, extent = self.context.pin(token, _descriptor(request["descriptor"]))
                        self.tokens.append(token)
                        if extent != lease.extent:
                            raise ValueError("registration extent mismatch")
                        reply["envelope"] = provider.register(lease, address)
                    elif operation == "destinations":
                        for entry in request["entries"]:
                            provider.install_destination(_lease(entry["lease"]), entry["envelope"])
                    elif operation == "write":
                        task = _task(request["task"])
                        if task.attempt.source != self.owner:
                            raise ValueError("stale write owner")
                        provider.write(task)
                    elif operation == "profile":
                        requested = request.get("active")
                        if type(requested) is not bool:
                            raise ValueError("owner profile state must be boolean")
                        profiler = get_profiler(
                            process_name=(
                                f"pd-transfer-owner-rank-{self.owner.rank_id}"
                            ),
                            initially_active=False,
                        )
                        try:
                            if requested:
                                # The chip child can have inherited an open
                                # pre-start fragment. Reopen it after the API
                                # process has cleared the previous profile run.
                                profiler.stop()
                                profiler.start()
                            else:
                                profiler.stop()
                        except Exception as exc:
                            reply["profile_error"] = (
                                f"{type(exc).__name__}: {exc}"
                            )
                        reply["profile_active"] = profiler.active
                    elif operation == "release":
                        provider.release()
                        for token in self.tokens:
                            self.context.unpin(token)
                        self.tokens.clear()
                    elif operation == "stop":
                        if self.tokens:
                            raise RuntimeError("cannot stop with registered arenas")
                        self.stopped = True
                    else:
                        raise ValueError("unknown owner operation")
                except TransferFailure as exc:
                    reply.update(ok=False, code=exc.error.code.value, certainty=exc.error.certainty.value,
                                 backend_code=exc.error.backend_code)
                    _send(self.channel, reply)
                    os._exit(76)
                except BaseException:
                    # A control/service exception makes the owner's lifecycle untrusted.
                    reply.update(ok=False, code=ErrorCode.OWNER_LOST.value, certainty=Certainty.UNKNOWN.value)
                    _send(self.channel, reply)
                    os._exit(76)
                _send(self.channel, reply)
                if self.stopped:
                    return
        except BaseException:
            os._exit(76)

    def close(self):
        if not self.stopped:
            raise RuntimeError("owner service was not explicitly stopped")
        self.thread.join(5)
        if self.thread.is_alive():
            raise RuntimeError("owner service thread still alive")
        self.channel.close()
