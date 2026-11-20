# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""VMM imports in independent transfer processes; HTTP does not establish DMA completion."""

from dataclasses import asdict
import importlib
import logging
import multiprocessing
import os
import select
import socket
import struct
import subprocess
import sys
import threading
from typing import Protocol

import msgspec

from pypto_serving.serving.utils.native_buffers import DeviceContext
from pypto_serving.tools.observe import span
from pypto_serving.tools.profile import configure_profiler, get_profiler
from .config import PROVIDERS
from .protocol import Copy, MAX_FRAME, decode

logger = logging.getLogger(__name__)


def configure_ports():
    os.environ["HCCL_NPU_SOCKET_PORT_RANGE"] = "auto"
    os.environ["HCCL_HOST_SOCKET_PORT_RANGE"] = "auto"


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
        probe = """\
from inspect import signature
from mooncake.engine import TransferEngine

register = TransferEngine.register_memory
try:
    binding = signature(register)
except (TypeError, ValueError):
    # Pybind11 publishes the argument names in its generated signature.
    if "location:" not in (register.__doc__ or "").split("\\n", 1)[0]:
        raise TypeError("register_memory has no location argument")
else:
    binding.bind(None, 0, 0, "npu")
"""
        result = subprocess.run([sys.executable, "-c", probe],
                                capture_output=True, text=True, timeout=30)
        if result.returncode:
            raise ValueError("Mooncake Ascend must support register_memory(address, extent, location) "
                             "from Mooncake #2191: " + result.stderr[-1000:])

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
        self._code(self.engine.register_memory(address, extent, "npu"))
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
        with span(logger, "pd.provider.lower", log=False, thread_cpu=True,
                  peer=peer, segments=len(copies)):
            source, destination, lengths, endpoints = self._lower(peer, copies)
        if not copies:
            return
        try:
            with span(logger, "MooncakeTransferEngine.batch_transfer_sync_write", cat="native-transfer",
                      log=False, segments=len(copies), bytes=sum(lengths)):
                self._code(self.engine.batch_transfer_sync_write(
                    endpoints.pop(), source, destination, lengths), uncertain=True)
        except BaseException as exc:
            self.poisoned = True
            raise TransferError(f"native completion uncertain: {exc}", uncertain=True) from exc

    def _lower(self, peer, copies):
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
        if copies and len(endpoints) != 1:
            raise ValueError("one native batch must target one rank owner")
        return source, destination, lengths, endpoints

    def release(self):
        self._check()
        for name, region in tuple(self.local.items()):
            self._code(self.engine.unregister_memory(region.handle["address"]), uncertain=True)
            del self.local[name]
        self.remote.clear()
        # Disconnect native communication before unmapping consumer addresses.
        self.engine = None


def provider_class(name):
    if name not in PROVIDERS:
        raise ValueError(f"unsupported transfer provider {name}")
    module, implementation = PROVIDERS[name]
    return getattr(importlib.import_module(module), implementation)


def provider_factory(name, hostname):
    return provider_class(name)(hostname)


def _send(channel, value):
    with span(logger, "pd.owner.encode", log=False, thread_cpu=True,
              operation=value.get("operation", "reply"), sequence=value.get("sequence")):
        payload = msgspec.msgpack.encode(value)
    if len(payload) > MAX_FRAME:
        raise ValueError("transfer message too large")
    channel.sendall(struct.pack("!I", len(payload)) + payload)


def _receive(channel):
    def exact(size):
        parts = bytearray()
        while len(parts) < size:
            part = channel.recv(size - len(parts))
            if not part:
                raise EOFError("transfer process exited")
            parts.extend(part)
        return bytes(parts)
    size, = struct.unpack("!I", exact(4))
    if not 0 < size <= MAX_FRAME:
        raise ValueError("invalid transfer frame size")
    return msgspec.msgpack.decode(exact(size))


class TransferWorker:
    """Controller for one fixed transfer process and its allocation-owner leases."""
    def __init__(self, boot, rank, hostname, provider="mooncake", factory=provider_factory):
        self.boot, self.rank = boot, rank
        self.hostname, self.provider, self.provider_factory = hostname, provider, factory
        self.parent, self.child = socket.socketpair()
        self.parent.settimeout(45)
        self.lock = threading.Lock()
        self.sequence = 0
        self.retained = []
        self.uncertain = False
        self.profile_config = get_profiler(initially_active=False).config
        self.monitor_stop = threading.Event()
        self.monitor = None
        self.process = None

    def start(self, device_id):
        self.process = multiprocessing.get_context("spawn").Process(
            target=_serve, name=f"pd-transfer-{self.rank}",
            args=(self.child, self.boot, self.rank, device_id, self.hostname,
                  self.provider, self.provider_factory, self.profile_config))
        self.process.start()
        return self.ready()

    def call(self, operation, **payload):
        fields = dict(operation=operation, owner_rank=self.rank, **payload.get("observation", {}))
        with span(logger, "pd.owner.call", log=False, **fields), self.lock:
            if self.uncertain:
                raise TransferError("transfer process gated", uncertain=True)
            self.sequence += 1
            try:
                with span(logger, "pd.owner.rpc", log=False, **fields):
                    _send(self.parent, dict(sequence=self.sequence, boot=self.boot, operation=operation, **payload))
                    reply = _receive(self.parent)
                if reply["sequence"] != self.sequence or reply["boot"] != self.boot:
                    raise ValueError("stale transfer reply")
                if "error" in reply:
                    raise TransferError(reply["error"], uncertain=reply["uncertain"])
                return reply.get("result")
            except BaseException as exc:
                self.uncertain = not isinstance(exc, TransferError) or exc.uncertain
                if self.uncertain:
                    raise TransferError(f"transfer completion unknown: {exc}", uncertain=True) from exc
                raise

    def ready(self):
        self.child.close()
        pid = self.call("probe")
        if self.process is None or pid != self.process.pid:
            raise RuntimeError("registration service is not the expected transfer process")
        self.pidfd = os.pidfd_open(pid)
        def observe():
            poller = select.poll()
            poller.register(self.pidfd, select.POLLIN)
            while not self.monitor_stop.is_set():
                if poller.poll(50):
                    self.uncertain = True
                    return
        monitor = threading.Thread(target=observe, name="pd-transfer-health", daemon=True)
        try:
            monitor.start()
        except BaseException:
            os.close(self.pidfd)
            raise
        self.monitor = monitor
        return pid

    def register(self, runtime, tensor, name):
        from simpler.buffer_export import BufferExportError
        try:
            exported = runtime.export_buffer(tensor, worker_id=self.rank)
        except BufferExportError as error:
            self.retained.append((runtime, error.exported))
            raise
        self.retained.append((runtime, exported))
        return decode(self.call("register", exported=asdict(exported),
                                name=name, extent=tensor.nbytes), Region)

    def release(self):
        self.call("release")
        while self.retained:
            runtime, exported = self.retained[0]
            runtime.release_export(exported)
            self.retained.pop(0)

    def close(self, *, startup_failed=False):
        if self.retained:
            raise RuntimeError("cannot close registered transfer process")
        process = self.process
        if process is not None and process.pid is not None and not startup_failed:
            self.call("stop")
        self.monitor_stop.set()
        if self.monitor is not None:
            self.monitor.join(1)
            os.close(self.pidfd)
            self.monitor = None
        self.child.close()
        self.parent.close()
        if process is None or process.pid is None:
            return
        process.join(10)
        if startup_failed and process.is_alive():
            # The startup probe comes before exports or any possible writes.
            process.terminate()
            process.join(5)
        if process.is_alive() or (not startup_failed and process.exitcode != 0):
            raise RuntimeError("transfer process did not retire normally")
        process.close()
        self.process = None


def _serve(channel, boot, rank, device_id, hostname, provider_name, factory, profile_config):
    # Each TE is another HCCL process on the compute owner's device. Let the
    # OS reserve unused ports atomically instead of reusing HCCL's fixed 16666.
    configure_ports()
    from simpler.buffer_export import BufferExport, BufferImportError, BufferMapping

    mappings = []
    sequence = 0
    try:
        context = DeviceContext(device_id)
        configure_profiler(profile_config, process_name=f"pd-transfer-{rank}", initially_active=False)
        os.environ["MC_TRANSFER_TIMEOUT"] = "30"
        provider = factory(provider_name, hostname)

        def disconnect():
            nonlocal provider
            if provider is not None:
                provider.release()
                provider = None

        while True:
            try:
                request = _receive(channel)
            except EOFError:
                if mappings:
                    raise
                disconnect()
                context.close()
                channel.close()
                return
            sequence += 1
            if request["boot"] != boot or request["sequence"] != sequence:
                raise ValueError("stale transfer command")
            reply = dict(boot=boot, sequence=sequence)
            stopped = False
            try:
                context.bind()
                op = request["operation"]
                if op == "probe":
                    reply["result"] = os.getpid()
                elif op == "register":
                    exported = BufferExport(**request["exported"])
                    if exported.device_id != device_id:
                        raise ValueError("export belongs to a different device")
                    mapping = BufferMapping(exported)
                    mappings.append(mapping)
                    if mapping.descriptor.nbytes != request["extent"]:
                        raise ValueError("registered extent differs from allocation")
                    reply["result"] = provider.register(boot, rank, request["name"],
                                                        mapping.base, request["extent"])
                elif op == "install":
                    provider.install(tuple(decode(r, Region) for r in request["regions"]))
                elif op == "write":
                    with span(logger, "pd.owner.decode", log=False, thread_cpu=True,
                              **request["observation"]):
                        copies = tuple(decode(c, Copy) for c in request["copies"])
                    with span(logger, "pd.native.write", cat="native-transfer",
                              **request["observation"], bytes=sum(c.length for c in copies),
                              segments=len(copies)):
                        provider.write(request["peer"], copies)
                elif op == "release":
                    disconnect()
                    while mappings:
                        try:
                            mappings[0].close()
                        except Exception:
                            # The provider is disconnected. Retry only this import.
                            try:
                                mappings[0].close()
                            except Exception as exc:
                                raise TransferError(f"buffer unmap incomplete: {exc}") from exc
                        mappings.pop(0)
                elif op == "profile":
                    profiler = get_profiler(initially_active=False)
                    profiler.stop()
                    if request["active"]:
                        profiler.start()
                elif op == "stop":
                    if mappings:
                        raise RuntimeError("registered transfer process cannot stop")
                    disconnect()
                    context.close()
                    stopped = True
                else:
                    raise ValueError("unknown transfer operation")
            except Exception as exc:
                if isinstance(exc, BufferImportError):
                    mappings.append(exc.mapping)
                reply.update(error=str(exc), uncertain=not isinstance(exc, (ValueError, BufferImportError))
                             and (not isinstance(exc, TransferError) or exc.uncertain))
            _send(channel, reply)
            if stopped:
                channel.close()
                return
    except Exception:
        logging.exception("transfer process failed; imported buffers remain retained")
        channel.close()
        if mappings:
            # A lost control channel is not permission to unmap possible DMA targets.
            threading.Event().wait()
        raise
