# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Aligned DDR extents; content identity and eviction stay in CacheTier."""

from bisect import bisect_left
import ctypes
from pathlib import Path
from mmap import PAGESIZE


class Extents:
    def __init__(self, capacity):
        if type(capacity) is not int or capacity < PAGESIZE:
            raise ValueError("DDR pool must hold at least one physical page")
        self.capacity = capacity // PAGESIZE * PAGESIZE
        self.free = [(0, self.capacity)]
        self.allocations = {}

    def reserve(self, handle, copies, size):
        if handle in self.allocations or copies <= 0 or size <= 0:
            raise ValueError("invalid DDR storage generation")
        size = (size + PAGESIZE - 1) // PAGESIZE * PAGESIZE
        acquired = []
        for _ in range(copies):
            for index, (offset, available) in enumerate(self.free):
                if available >= size:
                    acquired.append((offset, size))
                    if available == size:
                        self.free.pop(index)
                    else:
                        self.free[index] = offset + size, available - size
                    break
            else:
                for extent in acquired:
                    self._return(*extent)
                return None
        result = tuple(acquired)
        self.allocations[handle] = result
        return result

    def _return(self, offset, size):
        index = bisect_left(self.free, (offset,))
        if index and sum(self.free[index - 1]) == offset:
            index -= 1
            previous, available = self.free.pop(index)
            offset, size = previous, size + available
        if index < len(self.free) and offset + size == self.free[index][0]:
            size += self.free.pop(index)[1]
        self.free.insert(index, (offset, size))

    def release(self, handle):
        for offset, size in self.allocations.pop(handle):
            self._return(offset, size)


def advise(address, size):
    libc = ctypes.CDLL(None, use_errno=True)
    libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    libc.madvise.restype = ctypes.c_int
    if libc.madvise(address, size, 14):  # Linux MADV_HUGEPAGE, on the persistent VMA.
        raise OSError(ctypes.get_errno(), "cannot prepare DDR hugepage mapping")


def allocate_pool(capacity):
    """Materialize shared backing without concentrating it on the launch CPU's NUMA node."""
    from simpler.buffer import create_host_shared_buffer, mint_owner_instance_id

    buffer = create_host_shared_buffer(capacity, mint_owner_instance_id(), 1)
    try:
        advise(buffer.base, buffer.nbytes)
        allowed = next(line.split(":", 1)[1].strip() for line in Path("/proc/self/status").read_text().splitlines()
                       if line.startswith("Mems_allowed_list:"))
        nodes = []
        for span in allowed.split(","):
            first, _, last = span.partition("-")
            nodes.extend(range(int(first), int(last or first) + 1))
        if len(nodes) > 1:
            bits = ctypes.sizeof(ctypes.c_ulong) * 8
            mask = (ctypes.c_ulong * (max(nodes) // bits + 1))()
            for node in nodes:
                mask[node // bits] |= 1 << (node % bits)
            numa = ctypes.CDLL("libnuma.so.1", use_errno=True)
            numa.mbind.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int,
                                  ctypes.POINTER(ctypes.c_ulong), ctypes.c_ulong, ctypes.c_uint]
            numa.mbind.restype = ctypes.c_int
            # Linux's legacy nodemask ABI excludes maxnode-1; include the final allowed node.
            if numa.mbind(buffer.base, buffer.nbytes, 3, mask, max(nodes) + 2, 0):
                raise OSError(ctypes.get_errno(), "cannot interleave DDR pool across allowed NUMA nodes")
        ctypes.memset(buffer.base, 0, buffer.nbytes)
    except BaseException:
        buffer.close()  # No owner has received this backing yet.
        raise
    return buffer
