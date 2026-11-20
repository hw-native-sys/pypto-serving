# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""An independent consumer context for exported device buffers."""

import ctypes


class DeviceContext:
    def __init__(self, device_id):
        self.acl = ctypes.CDLL("libascendcl.so")
        self.acl.aclInit.argtypes = [ctypes.c_char_p]
        self.acl.aclrtSetDevice.argtypes = [ctypes.c_int]
        self.acl.aclrtCreateContext.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
        self.acl.aclrtSetCurrentContext.argtypes = [ctypes.c_void_p]
        self.acl.aclrtDestroyContext.argtypes = [ctypes.c_void_p]
        initialized = self.acl.aclInit(None)
        if initialized not in (0, 100002):  # ACL_ERROR_REPEAT_INITIALIZE
            self.check(initialized)
        self.check(self.acl.aclrtSetDevice(device_id))
        self.context = ctypes.c_void_p()
        self.check(self.acl.aclrtCreateContext(ctypes.byref(self.context), device_id))
        self.bind()

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError(f"transfer ACL operation failed: {code}")

    def bind(self):
        self.check(self.acl.aclrtSetCurrentContext(self.context))

    def close(self):
        self.check(self.acl.aclrtDestroyContext(self.context))


def check_buffer_exports():
    from inspect import signature
    from pypto.runtime.distributed_runner import DistributedWorker
    from simpler.buffer_export import BufferMapping  # noqa: F401

    if ("exportable" not in signature(DistributedWorker.alloc_tensor).parameters
            or not all(hasattr(DistributedWorker, name) for name in ("export_buffer", "release_export"))):
        raise ValueError("PyPTO lacks the required VMM buffer export interfaces")
