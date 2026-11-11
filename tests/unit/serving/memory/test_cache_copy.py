# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compile the actual planner against CPU copy stubs; no devices are initialized."""
import ctypes as C
from pathlib import Path
import subprocess
import threading
from types import SimpleNamespace

import pytest

from pypto_serving.serving.memory.tiering import simpler
from pypto_serving.serving.memory.tiering.backend import BATCH_SPANS


@pytest.fixture(scope="module")
def helper(tmp_path_factory):
    root = tmp_path_factory.mktemp("cache-copy")
    (root / "acl").mkdir()
    (root / "acl/acl.h").write_text('''#pragma once
#include <cstddef>
using aclrtStream=void*;
enum aclrtMemcpyKind {ACL_MEMCPY_HOST_TO_DEVICE=1, ACL_MEMCPY_DEVICE_TO_HOST=2};
extern "C" {
int aclrtMemcpyAsync(void*,size_t,const void*,size_t,aclrtMemcpyKind,aclrtStream);
int aclrtMemcpy2dAsync(void*,size_t,const void*,size_t,size_t,size_t,aclrtMemcpyKind,aclrtStream);
int aclrtSynchronizeStream(aclrtStream);
}
''')
    (root / "stub.cpp").write_text('''#include "acl/acl.h"
#include <cstring>
int calls=0,fail=0,stop_error=0,syncs=0;
extern "C" void inject(int n,int code){calls=0;fail=n;stop_error=code;syncs=0;}
extern "C" int stopped(){return syncs;}
extern "C" int aclrtMemcpyAsync(void*d,size_t,const void*s,size_t n,aclrtMemcpyKind,aclrtStream){
    if(++calls==fail)return 77;std::memcpy(d,s,n);return 0;}
extern "C" int aclrtMemcpy2dAsync(void*d,size_t dp,const void*s,size_t sp,size_t w,size_t h,
    aclrtMemcpyKind,aclrtStream){if(++calls==fail)return 77;
    for(size_t i=0;i<h;++i)std::memcpy((char*)d+i*dp,(const char*)s+i*sp,w);return 0;}
extern "C" int aclrtSynchronizeStream(aclrtStream){++syncs;return stop_error;}
''')
    source = Path(simpler.__file__).with_name("copy.cpp")
    subprocess.run(["g++", "-std=c++17", "-O2", "-shared", "-fPIC", "-I"+str(root),
                    str(source), str(root / "stub.cpp"), "-o", str(root / "copy.so")], check=True)
    lib = C.CDLL(str(root / "copy.so"))
    lib.copy_cache_batch.argtypes = [C.POINTER(simpler.Span), C.c_size_t, C.c_int, C.c_void_p,
                                    C.POINTER(C.c_uint64), C.POINTER(C.c_size_t)]
    lib.copy_cache_batch.restype = C.c_int
    return lib


def owner(helper, device, host):
    native = simpler.NativeCopy.__new__(simpler.NativeCopy)
    native.thread = threading.get_ident()
    native.closed = native.poisoned = False
    native.pool = SimpleNamespace(base=C.addressof(host), nbytes=len(host))
    native.regions = {"kv": ("pin", C.addressof(device), len(device), 0)}
    native.commands = (simpler.Span * BATCH_SPANS)()
    native.native, native.stream = helper, None
    helper.inject(0, 0)
    return native


@pytest.mark.parametrize("fragmented", [False, True])
@pytest.mark.parametrize("width", [1, 128, 1024, 32768])
def test_full_bytes_guards_and_single_command_table_reuse(helper, fragmented, width):
    layers, blocks = 43, 8
    size = layers * blocks * width
    original = (bytes(range(251)) * (size//251+1))[:size]
    device = C.create_string_buffer(original, size)
    host = C.create_string_buffer(b"\xa5" * (size+16), size+16)
    native = owner(helper, device, host)
    table = native.commands
    pages = [3, 0, 2, 1] if fragmented else list(range(4))
    for load in (False, True):
        payload = tuple((i+1, i*layers*width, layers*width,
            (("kv", (i+4 if load else page)*width, 0, width, layers, blocks*width),))
            for i, page in enumerate(pages))
        proof = native.copy_batch("load" if load else "store", payload)
        assert all(p["native_completed"] and p["bytes"] == layers*width for p in proof["pages"])
        assert proof["started_ns"] <= proof["submitted_ns"] <= proof["completed_ns"]
    expected = bytearray(original)
    for layer in range(layers):
        for i, page in enumerate(pages):
            expected[(layer*blocks+i+4)*width:(layer*blocks+i+5)*width] = (
                original[(layer*blocks+page)*width:(layer*blocks+page+1)*width])
    assert bytes(device) == expected and bytes(host)[4*layers*width:] == b"\xa5"*(size+16-4*layers*width)
    assert native.commands is table and helper.stopped() == 2


@pytest.mark.parametrize("submit,stop", [(1, 0), (2, 0), (0, 99)])
def test_partial_submission_and_unknown_completion_keep_owner_isolated(helper, submit, stop):
    device, host = C.create_string_buffer(256), C.create_string_buffer(256)
    native = owner(helper, device, host)
    helper.inject(submit, stop)
    with pytest.raises(RuntimeError, match="failed"):
        native.copy_batch("store", ((1, 0, 8, (("kv", 0, 0, 4, 2, 32),)),
                                    (2, 16, 8, (("kv", 16, 0, 4, 2, 32),))))
    assert helper.stopped() == 1 and native.poisoned
    with pytest.raises(RuntimeError, match="unavailable"):
        native.close()
