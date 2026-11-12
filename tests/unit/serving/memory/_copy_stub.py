# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU ACL fixture for exercising the actual cache-copy planner without a device."""
import ctypes as C
from pathlib import Path
import subprocess

import pytest

from pypto_serving.serving.memory.tiering import simpler


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

