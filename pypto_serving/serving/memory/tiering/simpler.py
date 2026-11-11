# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Owner-local DMA over long-lived Simpler host Buffers, outside compute submission."""

from __future__ import annotations

import ctypes
import threading

from .pool import advise


def descriptor_wire(descriptor):
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


class Location(ctypes.Structure):
    _fields_ = [("id", ctypes.c_uint32), ("type", ctypes.c_int)]


class CopyAttributes(ctypes.Structure):
    _fields_ = [("destination", Location), ("source", Location), ("reserved", ctypes.c_uint8 * 16)]


class Copy(ctypes.Structure):
    _fields_ = [("dst", ctypes.c_void_p), ("src", ctypes.c_void_p),
                ("width", ctypes.c_size_t), ("height", ctypes.c_size_t),
                ("dst_pitch", ctypes.c_size_t), ("src_pitch", ctypes.c_size_t)]


def rectangles(entries):
    entries = sorted(entries)
    result, index = [], 0
    while index < len(entries):
        name, device, host, width = entries[index]
        end, device_pitch, host_pitch = index + 1, width, width
        if end < len(entries) and entries[end][0] == name and entries[end][3] == width:
            device_pitch = entries[end][1] - device
            host_pitch = entries[end][2] - host
            if device_pitch >= width and host_pitch >= width:
                while end < len(entries) and entries[end] == (
                        name, device + (end - index) * device_pitch,
                        host + (end - index) * host_pitch, width):
                    end += 1
        if end == index + 1:
            device_pitch = host_pitch = width
        result.append((device, host, width, end - index, device_pitch, host_pitch))
        index = end
    return result


class NativeCopy:
    """One owner thread, one independent copy stream, allocation-scoped registrations."""

    def __init__(self, context):
        self.context = context
        self.thread = threading.get_ident()
        self.acl = ctypes.CDLL("libascendcl.so")
        pointer = ctypes.c_void_p
        signatures = {
            "aclrtCreateStream": [ctypes.POINTER(pointer)],
            "aclrtDestroyStream": [pointer],
            "aclrtSynchronizeStream": [pointer],
            "aclrtGetDevice": [ctypes.POINTER(ctypes.c_int)],
            "aclrtHostRegisterV2": [pointer, ctypes.c_uint64, ctypes.c_uint32],
            "aclrtHostUnregister": [pointer],
        }
        for name, args in signatures.items():
            function = getattr(self.acl, name)
            function.argtypes, function.restype = args, ctypes.c_int
        self.stream = pointer()
        self._call("aclrtCreateStream", ctypes.byref(self.stream))
        self.regions = {}
        self.pool = None
        self.poisoned = self.closed = False

    def _call(self, name, *args):
        code = getattr(self.acl, name)(*args)
        if code:
            raise RuntimeError(f"{name} failed: {code}")

    def _check(self):
        if self.thread != threading.get_ident() or self.closed or self.poisoned:
            raise RuntimeError("cache copy owner unavailable")

    def register(self, name, token, descriptor):
        self._check()
        if name in self.regions:
            raise ValueError("duplicate cache allocation")
        address, extent = self.context.pin(token, descriptor)
        self.regions[name] = (token, address, extent)

    def initialize_pool(self, wire):
        from simpler.buffer import ImportContext, ImportRegistry
        from simpler.comm_endpoints import HOST_CPU

        self._check()
        self.registry = ImportRegistry(ImportContext(deployment=HOST_CPU))
        self.pool = self.registry.materialize(_descriptor(wire))
        advise(self.pool.base, self.pool.nbytes)
        self.native = ctypes.CDLL(wire["copy_library"])
        self.native.prefault_read.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.native.prefault_read.restype = ctypes.c_uint64
        self.native.prefault_read(self.pool.base, self.pool.nbytes)
        self._call("aclrtHostRegisterV2", self.pool.base, self.pool.nbytes, 0x10000000)
        self.native.copy_cache_batch.argtypes = [ctypes.POINTER(Copy), ctypes.c_size_t, ctypes.c_int,
            ctypes.POINTER(CopyAttributes), ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
        self.native.copy_cache_batch.restype = ctypes.c_int
        self.device = ctypes.c_int()
        self._call("aclrtGetDevice", ctypes.byref(self.device))

    def copy_batch(self, operation, pages):
        """All descriptors are validated before any native use of the shared pool."""
        self._check()
        if operation not in ("store", "load") or not pages:
            raise ValueError("invalid cache batch")
        ranges, entries, handles = [], [], set()
        for handle, offset, allocation, spans in pages:
            if (type(handle) is not int or handle <= 0 or handle in handles
                    or any(type(v) is not int for v in (offset, allocation))
                    or offset < 0 or allocation <= 0 or offset + allocation > self.pool.nbytes):
                raise ValueError("invalid DDR range or storage generation")
            handles.add(handle)
            ranges.append((offset, offset + allocation))
            host_ranges, size = [], 0
            for name, device_offset, host_offset, width, rows, pitch in spans:
                if (any(type(v) is not int for v in (device_offset, host_offset, width, rows, pitch))
                        or device_offset < 0 or host_offset < 0 or width <= 0 or rows <= 0 or pitch < width):
                    raise ValueError("invalid copy span")
                _, address, extent = self.regions[name]
                if (device_offset + (rows - 1) * pitch + width > extent
                        or host_offset + rows * width > allocation):
                    raise ValueError("copy exceeds registered allocation")
                host_ranges.append((host_offset, host_offset + rows * width))
                for row in range(rows):
                    entries.append((name, address + device_offset + row * pitch,
                                    self.pool.base + offset + host_offset + row * width, width))
                size += rows * width
            if not size or any(left[1] > right[0] for left, right in
                               zip(sorted(host_ranges), sorted(host_ranges)[1:])):
                raise ValueError("empty or overlapping cache page")
        if (len(entries) > 8192 or any(left[1] > right[0] for left, right in
                                      zip(sorted(ranges), sorted(ranges)[1:]))):
            raise ValueError("overlapping DDR ranges or oversized native batch")
        store = operation == "store"
        planned = rectangles(entries)
        commands = (Copy * len(planned))(*(Copy(host if store else device, device if store else host,
            width, rows, host_pitch if store else device_pitch, device_pitch if store else host_pitch)
            for device, host, width, rows, device_pitch, host_pitch in planned))
        attributes = CopyAttributes()
        host, device = Location(0, 0), Location(self.device.value, 1)
        attributes.destination, attributes.source = (host, device) if store else (device, host)
        timing = (ctypes.c_uint64 * 3)()
        try:
            code = self.native.copy_cache_batch(commands, len(commands), 2 if store else 1,
                                                ctypes.byref(attributes), self.stream, timing)
            if code:
                raise RuntimeError(f"cache native copy failed: {code}")
        except BaseException:
            self.poisoned = True
            raise
        return dict(pages=tuple(dict(handle=handle, bytes=sum(s[3] * s[4] for s in spans),
                                     stopped=True, native_completed=True)
                                for handle, _, _, spans in pages),
                    started_ns=timing[0], submitted_ns=timing[1], completed_ns=timing[2],
                    native_commands=len(planned))

    def close(self):
        self._check()
        if self.pool is not None:
            self._call("aclrtHostUnregister", self.pool.base)
            self.registry.unregister(self.pool.identity)
            self.pool = None
        for token, _, _ in self.regions.values():
            self.context.unpin(token)
        self.regions.clear()
        self._call("aclrtDestroyStream", self.stream)
        self.closed = True


class CopyService:
    """Independent child-owner endpoint. All Buffer and native work stays in its thread."""

    def __init__(self, context, channel, pool):
        self.context, self.channel, self.pool = context, channel, pool
        acl = ctypes.CDLL("libascendcl.so")
        self.acl_context = ctypes.c_void_p()
        acl.aclrtGetCurrentContext.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        if acl.aclrtGetCurrentContext(ctypes.byref(self.acl_context)) or not self.acl_context.value:
            raise RuntimeError("cache service requires the allocation owner's ACL context")
        self.started = threading.Event()
        self.close_requested = threading.Event()
        self.error = None
        self.stopped = False
        self.thread = threading.Thread(target=self._run, name="cache-copy-owner", daemon=False)
        self.thread.start()
        import os
        if not self.started.wait(float(os.environ.get("PYPTO_WORKER_INIT_TIMEOUT", "600"))) or self.error is not None:
            raise RuntimeError("cache copy owner initialization failed") from self.error

    def _run(self):
        try:
            acl = ctypes.CDLL("libascendcl.so")
            acl.aclrtSetCurrentContext.argtypes = [ctypes.c_void_p]
            if acl.aclrtSetCurrentContext(self.acl_context):
                raise RuntimeError("cannot attach cache-copy thread to owner context")
            provider = NativeCopy(self.context)
            self.provider = provider
            provider.initialize_pool(self.pool)
            self.started.set()
            expected = 0
            while not self.stopped:
                if self.close_requested.is_set():
                    provider.close()  # Rejects uncertain DMA.
                    self.stopped = True
                    break
                if not self.channel.poll(.1):
                    continue
                try:
                    sequence, operation, payload = self.channel.recv()
                except EOFError:
                    provider.close()
                    self.stopped = True
                    break
                expected += 1
                if sequence != expected:
                    raise RuntimeError("stale cache-copy command")
                try:
                    if operation == "register":
                        name, token, descriptor = payload
                        result = provider.register(name, token, _descriptor(descriptor))
                    elif operation == "copy_batch":
                        result = provider.copy_batch(*payload)
                    elif operation == "stop":
                        result = provider.close()
                        self.stopped = True
                    else:
                        raise ValueError("unknown cache-copy operation")
                    self.channel.send((sequence, True, result))
                except Exception as error:
                    self.channel.send((sequence, False, (str(error), provider.poisoned)))
        except BaseException as error:
            self.error = error
            self.started.set()
        finally:
            self.channel.close()

    def close(self):
        self.close_requested.set()
        self.thread.join(5)
        if self.thread.is_alive():
            raise RuntimeError("cache copy thread is still active")
        if not self.stopped:
            raise RuntimeError("cache copy owner has not been safely retired") from self.error
