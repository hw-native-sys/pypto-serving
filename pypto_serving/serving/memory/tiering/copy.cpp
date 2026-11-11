// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
#include <acl/acl.h>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

struct Copy {
    void *dst, *src;
    size_t width, height, dst_pitch, src_pitch;
};
static uint64_t now()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
extern "C" uint64_t prefault_read(const void *base, size_t bytes)
{
    auto p = static_cast<const volatile uint8_t *>(base);
    uint64_t value = 0;
    for (size_t i = 0; i < bytes; i += (2UL << 20)) {
        value += p[i];
    }
    if (bytes) {
        value += p[bytes - 1];
    }
    return value;
}
extern "C" int copy_cache_batch(const Copy *commands, size_t count, int kind,
    aclrtMemcpyBatchAttr *attr, aclrtStream stream, uint64_t *ticks)
{
    std::vector<void *> dst, src;
    std::vector<size_t> sizes;
    dst.reserve(count);
    src.reserve(count);
    sizes.reserve(count);
    int code = 0;
    ticks[0] = now();
    for (size_t i = 0; i < count && !code; ++i) {
        const auto &c = commands[i];
        if (c.height == 1) {
            dst.push_back(c.dst);
            src.push_back(c.src);
            sizes.push_back(c.width);
        } else if (c.src_pitch == c.width && c.dst_pitch == c.width) {
            const auto bytes = c.width * c.height;
            code = aclrtMemcpyAsync(c.dst, bytes, c.src, bytes, static_cast<aclrtMemcpyKind>(kind), stream);
        } else {
            code = aclrtMemcpy2dAsync(c.dst, c.dst_pitch, c.src, c.src_pitch, c.width, c.height,
                                    static_cast<aclrtMemcpyKind>(kind), stream);
        }
    }
    if (!code && sizes.size() == 1) {
        code = aclrtMemcpyAsync(dst[0], sizes[0], src[0], sizes[0], static_cast<aclrtMemcpyKind>(kind), stream);
    } else if (!code && !sizes.empty()) {
        size_t index = 0, failed = SIZE_MAX;
        code = aclrtMemcpyBatchAsync(dst.data(), sizes.data(), src.data(), sizes.data(), sizes.size(),
                                    attr, &index, 1, &failed, stream);
    }
    ticks[1] = now();
    // Even an accepted prefix must be stopped before a failed batch can retire.
    const auto completed = aclrtSynchronizeStream(stream);
    ticks[2] = now();
    return code ? code : completed;
}
