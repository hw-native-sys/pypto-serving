# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Owner-local registration and transfer. Native completion is never inferred from HTTP."""

import ctypes
from functools import partial
import importlib
import os
import select
import socket
import struct
import subprocess
import sys
import threading
from typing import Protocol

import msgspec

from pypto_serving.tools.profile import profile_span, configure_profiler, get_profiler
from .config import PROVIDERS
from .protocol import Copy, MAX_FRAME, decode


class TransferError(RuntimeError):
    def __init__(self, message, *, uncertain=False):
        super().__init__(message)
        self.uncertain = uncertain


class Region(msgspec.Struct, frozen=True):
    boot: str
    rank: int
    name: str
    extent: int
    handle: dict

    @property
    def key(self):
        return self.boot, self.rank, self.name


class Provider(Protocol):
    def register(self, boot: str, rank: int, name: str, address: int, extent: int) -> Region: ...
    def install(self, regions: tuple[Region, ...]) -> None: ...
    def write(self, peer: str, copies: tuple[Copy, ...]) -> None: ...
    def release(self) -> None: ...


class Mooncake:
    @staticmethod
    def check_available():
        # Import in a disposable process, before weights/fork, without starting an engine.
        result = subprocess.run([sys.executable, "-c", "from mooncake.engine import TransferEngine"],
                                capture_output=True, text=True, timeout=30)
        if result.returncode:
            raise ValueError("Mooncake Python/native import failed: " + result.stderr[-1000:])

    def __init__(self, hostname, engine_factory=None):
        self.owner = (os.getpid(), threading.get_ident())
        self.poisoned = False
        self.local, self.remote = {}, {}
        self.engine = (engine_factory or importlib.import_module("mooncake.engine").TransferEngine)()
        self._code(self.engine.initialize(hostname, "P2PHANDSHAKE", "ascend", ""))
        port = int(self.engine.get_rpc_port())
        if not 0 < port < 65536:
            raise ValueError("invalid Mooncake RPC port")
        self.endpoint = f"{hostname}:{port}"

    def _check(self):
        if self.poisoned or self.owner != (os.getpid(), threading.get_ident()):
            raise TransferError("provider unavailable or wrong owner thread", uncertain=self.poisoned)

    def _code(self, code, uncertain=False):
        if type(code) is not int or code != 0:
            self.poisoned = True
            raise TransferError("Mooncake operation failed", uncertain=uncertain)

    def register(self, boot, rank, name, address, extent):
        self._check()
        if (not boot or type(address) is not int or address <= 0 or address % 64
                or type(extent) is not int or extent <= 0 or extent % 64
                or address + extent > 1 << 64 or name in self.local):
            raise ValueError("invalid or duplicate device region")
        self._code(self.engine.register_memory(address, extent))
        region = Region(boot, rank, name, extent, {"endpoint": self.endpoint, "address": address})
        self.local[name] = region
        return region

    def install(self, regions):
        self._check()
        for region in regions:
            address, endpoint = region.handle.get("address"), region.handle.get("endpoint")
            if (not region.boot or region.rank < 0 or not isinstance(endpoint, str) or not endpoint
                    or type(address) is not int or address <= 0 or address % 64
                    or region.extent <= 0 or region.extent % 64 or address + region.extent > 1 << 64):
                raise ValueError("invalid destination region")
            old = self.remote.get(region.key)
            if old is not None and old != region:
                raise ValueError("registration changed within the same boot identity")
        self.remote.update((r.key, r) for r in regions)

    def write(self, peer, copies):
        self._check()
        source, destination, lengths, endpoints = [], [], [], set()
        for copy in copies:
            src = self.local[copy.component]
            dst = self.remote[peer, copy.destination_rank, copy.component]
            if (copy.source_rank != src.rank or min(copy.source_offset, copy.destination_offset) < 0
                    or copy.length <= 0 or copy.source_offset + copy.length > src.extent
                    or copy.destination_offset + copy.length > dst.extent):
                raise ValueError("transfer exceeds registered owner/region")
            source.append(src.handle["address"] + copy.source_offset)
            destination.append(dst.handle["address"] + copy.destination_offset)
            lengths.append(copy.length)
            endpoints.add(dst.handle["endpoint"])
        if not copies:
            return
        if len(endpoints) != 1:
            raise ValueError("one native batch must target one rank owner")
        try:
            with profile_span("pd.native.write", cat="native-transfer",
                              args={"bytes": sum(lengths), "segments": len(lengths)}):
                self._code(self.engine.batch_transfer_sync_write(
                    endpoints.pop(), source, destination, lengths), uncertain=True)
        except BaseException as exc:
            self.poisoned = True
            raise TransferError(f"native completion uncertain: {exc}", uncertain=True) from exc

    def release(self):
        self._check()
        for name, region in tuple(self.local.items()):
            self._code(self.engine.unregister_memory(region.handle["address"]), uncertain=True)
            del self.local[name]
        self.remote.clear()


def provider_class(name):
    if name not in PROVIDERS:
        raise ValueError(f"unsupported transfer provider {name}")
    module, implementation = PROVIDERS[name]
    return getattr(importlib.import_module(module), implementation)


def provider_factory(name, hostname):
    return provider_class(name)(hostname)


def _send(channel, value):
    payload = msgspec.msgpack.encode(value)
    if len(payload) > MAX_FRAME:
        raise ValueError("owner message too large")
    channel.sendall(struct.pack("!I", len(payload)) + payload)


def _receive(channel):
    def exact(size):
        parts = bytearray()
        while len(parts) < size:
            part = channel.recv(size - len(parts))
            if not part:
                raise EOFError("allocation owner exited")
            parts.extend(part)
        return bytes(parts)
    size, = struct.unpack("!I", exact(4))
    if not 0 < size <= MAX_FRAME:
        raise ValueError("invalid owner frame size")
    return msgspec.msgpack.decode(exact(size))


def _descriptor_wire(descriptor):
    return dict(owner=bytes(descriptor.identity.owner_instance_id),
                buffer_id=int(descriptor.identity.buffer_id), generation=int(descriptor.identity.generation),
                address_space=int(descriptor.address_space), access=int(descriptor.access),
                backend_kind=int(descriptor.backend_kind), nbytes=int(descriptor.nbytes),
                body=bytes(descriptor.body), owner_worker_path_id=int(descriptor.owner_worker_path_id))


def _descriptor(wire):
    from simpler.buffer import AccessMode, AddressSpace, BackendKind, BufferDescriptor, CanonicalIdentity
    return BufferDescriptor(identity=CanonicalIdentity(wire["owner"], wire["buffer_id"], wire["generation"]),
                            address_space=AddressSpace(wire["address_space"]), access=AccessMode(wire["access"]),
                            backend_kind=BackendKind(wire["backend_kind"]), nbytes=wire["nbytes"],
                            body=wire["body"], owner_worker_path_id=wire["owner_worker_path_id"])


class Owner:
    """Parent-side bridge; all chip-child services are installed before L3 prepare."""
    def __init__(self, boot, rank, hostname, provider="mooncake", factory=provider_factory):
        self.boot, self.rank = boot, rank
        self.hostname, self.provider, self.provider_factory = hostname, provider, factory
        self.parent, self.child = socket.socketpair()
        self.parent.settimeout(45)
        self.parent_pid = os.getpid()
        self.lock = threading.Lock()
        self.sequence = 0
        self.retained = []
        self.uncertain = False
        self.profile_config = get_profiler(initially_active=False).config
        self.monitor_stop = threading.Event()
        self.monitor = None

    def factory(self, context):
        if os.getpid() == self.parent_pid:
            raise RuntimeError("transfer service must run in allocation owner")
        self.parent.close()
        return OwnerService(context, self.child, self)

    def call(self, operation, **payload):
        with self.lock:
            if self.uncertain:
                raise TransferError("owner gated", uncertain=True)
            self.sequence += 1
            try:
                _send(self.parent, dict(sequence=self.sequence, boot=self.boot, operation=operation, **payload))
                reply = _receive(self.parent)
                if reply["sequence"] != self.sequence or reply["boot"] != self.boot:
                    raise ValueError("stale owner reply")
                if "error" in reply:
                    raise TransferError(reply["error"], uncertain=reply["uncertain"])
                return reply.get("result")
            except BaseException as exc:
                self.uncertain = not isinstance(exc, TransferError) or exc.uncertain
                if self.uncertain:
                    raise TransferError(f"owner completion unknown: {exc}", uncertain=True) from exc
                raise

    def ready(self):
        self.child.close()
        pid = self.call("probe")
        if pid == self.parent_pid:
            raise RuntimeError("registration service is not an allocation owner")
        self.pidfd = os.pidfd_open(pid)
        def observe():
            poller = select.poll()
            poller.register(self.pidfd, select.POLLIN)
            while not self.monitor_stop.is_set():
                if poller.poll(50):
                    self.uncertain = True
                    return
        self.monitor = threading.Thread(target=observe, name="pd-owner-health", daemon=True)
        self.monitor.start()
        return pid

    def register(self, runtime, tensor, name):
        token, descriptor = runtime.retain_service_tensor(tensor, self.rank)
        self.retained.append((runtime, token))
        return decode(self.call("register", token=token, descriptor=_descriptor_wire(descriptor),
                                name=name, extent=tensor.nbytes), Region)

    def release(self):
        self.call("release")
        for runtime, token in self.retained:
            runtime.release_service_tensor(token)
        self.retained.clear()

    def close(self):
        if self.retained:
            raise RuntimeError("cannot close registered owner")
        self.call("stop")
        self.monitor_stop.set()
        if self.monitor is not None:
            self.monitor.join(1)
            os.close(self.pidfd)
        self.parent.close()


def _owner_factory(owners, rank, context):
    for sibling in owners:
        if sibling is not owners[rank]:
            sibling.parent.close()
            sibling.child.close()
    return owners[rank].factory(context)


def owner_factories(owners):
    return {i: partial(_owner_factory, owners, i) for i in range(len(owners))}


class OwnerService:
    def __init__(self, context, channel, owner):
        self.context, self.channel, self.owner = context, channel, owner
        configure_profiler(owner.profile_config, process_name=f"pd-owner-{owner.rank}", initially_active=False)
        self.tokens = []
        self.stopped = False
        self.started = threading.Event()
        self.start_error = None
        self.acl = ctypes.CDLL("libascendcl.so")
        self.acl.aclrtGetCurrentContext.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.acl.aclrtSetCurrentContext.argtypes = [ctypes.c_void_p]
        self.acl_context = ctypes.c_void_p()
        if self.acl.aclrtGetCurrentContext(ctypes.byref(self.acl_context)) != 0 or not self.acl_context.value:
            raise RuntimeError("allocation owner has no ACL context")
        self.thread = threading.Thread(target=self._run, name="pd-transfer-owner", daemon=False)
        self.thread.start()
        if not self.started.wait(40):
            raise RuntimeError("owner transfer initialization timed out")
        if self.start_error is not None:
            self.thread.join(5)
            raise RuntimeError("owner transfer initialization failed") from self.start_error

    def _run(self):
        provider = None
        sequence = 0
        try:
            if self.acl.aclrtSetCurrentContext(self.acl_context) != 0:
                raise RuntimeError("cannot attach owner ACL context")
            os.environ["MC_TRANSFER_TIMEOUT"] = "30"
            provider = self.owner.provider_factory(self.owner.provider, self.owner.hostname)
            self.started.set()
            while not self.stopped:
                request = _receive(self.channel)
                sequence += 1
                if request["boot"] != self.owner.boot or request["sequence"] != sequence:
                    raise ValueError("stale owner command")
                reply = dict(boot=self.owner.boot, sequence=sequence)
                try:
                    op = request["operation"]
                    if op == "probe":
                        reply["result"] = os.getpid()
                    elif op == "register":
                        token = request["token"]
                        address, extent = self.context.pin(token, _descriptor(request["descriptor"]))
                        self.tokens.append(token)
                        if extent != request["extent"]:
                            raise ValueError("registered extent differs from allocation")
                        reply["result"] = provider.register(self.owner.boot, self.owner.rank,
                                                           request["name"], address, extent)
                    elif op == "install":
                        provider.install(tuple(decode(r, Region) for r in request["regions"]))
                    elif op == "write":
                        provider.write(request["peer"], tuple(decode(c, Copy) for c in request["copies"]))
                    elif op == "release":
                        provider.release()
                        for token in self.tokens:
                            self.context.unpin(token)
                        self.tokens.clear()
                    elif op == "profile":
                        profiler = get_profiler(initially_active=False)
                        if request["active"]:
                            profiler.stop()
                            profiler.start()
                        else:
                            profiler.stop()
                    elif op == "stop":
                        if self.tokens:
                            raise RuntimeError("registered owner cannot stop")
                        self.stopped = True
                    else:
                        raise ValueError("unknown owner operation")
                except Exception as exc:
                    reply.update(error=str(exc), uncertain=not isinstance(exc, ValueError)
                                 and (not isinstance(exc, TransferError) or exc.uncertain))
                _send(self.channel, reply)
        except Exception as exc:
            # Never unregister/unpin on a lost reply or uncertain native write.
            self.start_error = exc
            self.started.set()
            self.channel.close()

    def close(self):
        if not self.stopped:
            raise RuntimeError("owner must be safely retired before runtime shutdown")
        self.thread.join(5)
        if self.thread.is_alive():
            raise RuntimeError("owner progress thread still active")
        self.channel.close()
