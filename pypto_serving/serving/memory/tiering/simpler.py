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
import time


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
            "aclrtMemcpy2dAsync": [pointer, ctypes.c_size_t, pointer, ctypes.c_size_t,
                                   ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int, pointer],
            "aclrtHostRegister": [pointer, ctypes.c_uint64, ctypes.c_int, ctypes.POINTER(pointer)],
            "aclrtHostUnregister": [pointer],
        }
        for name, args in signatures.items():
            function = getattr(self.acl, name)
            function.argtypes, function.restype = args, ctypes.c_int
        self.stream = pointer()
        self._call("aclrtCreateStream", ctypes.byref(self.stream))
        self.regions = {}
        self.buffers = {}
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

    def reserve(self, key, size):
        from simpler.buffer import create_host_shared_buffer, mint_owner_instance_id

        self._check()
        if key in self.buffers or type(size) is not int or size <= 0:
            raise ValueError("invalid DDR allocation")
        buffer = create_host_shared_buffer(size, mint_owner_instance_id(), 1)
        address = ctypes.addressof(ctypes.c_char.from_buffer(buffer.shm.buf))
        mapped = ctypes.c_void_p()
        try:
            self._call("aclrtHostRegister", address, size, 0, ctypes.byref(mapped))
        except BaseException:
            buffer.close()
            raise
        self.buffers[key] = (buffer, address)

    def copy(self, key, spans, *, store):
        """Spans are (region, device offset, DDR offset, row bytes, rows, device pitch)."""
        self._check()
        buffer, host = self.buffers[key]
        ranges = []
        for name, device_offset, host_offset, width, rows, pitch in spans:
            if (any(type(value) is not int for value in (device_offset, host_offset, width, rows, pitch))
                    or device_offset < 0 or host_offset < 0 or width <= 0 or rows <= 0 or pitch < width):
                raise ValueError("invalid copy span")
            _, address, extent = self.regions[name]
            if (device_offset + (rows - 1) * pitch + width > extent
                    or host_offset + rows * width > buffer.nbytes):
                raise ValueError("copy exceeds registered allocation")
            ranges.append((address + device_offset, host + host_offset, width, rows, pitch))
        if not ranges:
            raise ValueError("empty cache transfer")
        started = time.monotonic_ns()
        try:
            for device, host, width, rows, pitch in ranges:
                destination, source = (host, device) if store else (device, host)
                destination_pitch, source_pitch = (width, pitch) if store else (pitch, width)
                self._call("aclrtMemcpy2dAsync", destination, destination_pitch, source, source_pitch,
                           width, rows, 2 if store else 1, self.stream)
            # Only the dedicated copy thread waits; compute uses another stream and IPC lane.
            self._call("aclrtSynchronizeStream", self.stream)
        except BaseException:
            self.poisoned = True
            raise
        return dict(bytes=sum(width * rows for _, _, width, rows, _ in ranges), stopped=True, native_completed=True,
                    started_ns=started, completed_ns=time.monotonic_ns())

    def release(self, key):
        self._check()
        buffer, address = self.buffers[key]
        try:
            self._call("aclrtHostUnregister", address)
            buffer.close()
        except BaseException:
            self.poisoned = True
            raise
        del self.buffers[key]

    def close(self):
        self._check()
        if self.buffers:
            raise RuntimeError("DDR cache entries must retire before owner close")
        for token, _, _ in self.regions.values():
            self.context.unpin(token)
        self.regions.clear()
        self._call("aclrtDestroyStream", self.stream)
        self.closed = True


class CopyService:
    """Independent child-owner endpoint. All Buffer and native work stays in its thread."""

    def __init__(self, context, channel):
        self.context, self.channel = context, channel
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
        if not self.started.wait(30) or self.error is not None:
            raise RuntimeError("cache copy owner initialization failed") from self.error

    def _run(self):
        try:
            acl = ctypes.CDLL("libascendcl.so")
            acl.aclrtSetCurrentContext.argtypes = [ctypes.c_void_p]
            if acl.aclrtSetCurrentContext(self.acl_context):
                raise RuntimeError("cannot attach cache-copy thread to owner context")
            provider = NativeCopy(self.context)
            self.provider = provider
            self.started.set()
            expected = 0
            while not self.stopped:
                if self.close_requested.is_set():
                    provider.close()  # Rejects live backing or uncertain DMA.
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
                    elif operation == "reserve":
                        result = provider.reserve(*payload)
                    elif operation in ("store", "load"):
                        key, spans = payload
                        result = provider.copy(key, spans, store=operation == "store")
                    elif operation == "release":
                        result = provider.release(payload)
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
