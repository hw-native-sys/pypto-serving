// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
#include <acl/acl.h>
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <vector>

struct Span {
    uintptr_t device, host;
    size_t width, rows, pitch, region;
};
struct Copy {
    uintptr_t device, host;
    size_t width, rows, device_pitch, host_pitch;
};
struct Row {
    size_t region;
    uintptr_t device, host;
    size_t width;
    auto key() const { return std::tie(region, device, host, width); }
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
static std::vector<Copy> plan(const Span *spans, size_t count)
{
    std::vector<Row> rows;
    std::vector<Copy> pages, runs;
    for (size_t i = 0; i < count; ++i) {
        const auto &s = spans[i];
        pages.push_back({s.device, s.host, s.width, s.rows, s.pitch, s.width});
        for (size_t r = 0; r < s.rows; ++r) {
            rows.push_back({s.region, s.device + r*s.pitch, s.host + r*s.width, s.width});
        }
    }
    std::sort(rows.begin(), rows.end(), [](const Row &a, const Row &b) { return a.key() < b.key(); });
    for (size_t i = 0; i < rows.size();) {
        const auto a = rows[i];
        size_t end = i + 1, dp = a.width, hp = a.width;
        if (end < rows.size() && rows[end].region == a.region && rows[end].width == a.width &&
            rows[end].device >= a.device + a.width && rows[end].host >= a.host + a.width) {
            dp = rows[end].device - a.device;
            hp = rows[end].host - a.host;
            while (end < rows.size() && rows[end].region == a.region && rows[end].width == a.width &&
                   rows[end].device == a.device + (end-i)*dp && rows[end].host == a.host + (end-i)*hp) {
                ++end;
            }
        }
        runs.push_back({a.device, a.host, a.width, end-i, dp, hp});
        // More runs cannot improve a plan that already exceeds the page plan.
        if (runs.size() > pages.size()) return pages;
        i = end;
    }
    return runs.size() <= pages.size() ? runs : pages;
}
extern "C" int copy_cache_batch(const Span *spans, size_t count, int kind,
    aclrtStream stream, uint64_t *ticks, size_t *planned)
{
    ticks[0] = now();
    const auto commands = plan(spans, count);
    *planned = commands.size();
    int code = 0;
    const bool store = kind == ACL_MEMCPY_DEVICE_TO_HOST;
    for (const auto &c : commands) {
        void *dst = reinterpret_cast<void *>(store ? c.host : c.device);
        void *src = reinterpret_cast<void *>(store ? c.device : c.host);
        const size_t dp = store ? c.host_pitch : c.device_pitch;
        const size_t sp = store ? c.device_pitch : c.host_pitch;
        if (c.rows == 1 || (dp == c.width && sp == c.width)) {
            const auto bytes = c.width * c.rows;
            code = aclrtMemcpyAsync(dst, bytes, src, bytes, static_cast<aclrtMemcpyKind>(kind), stream);
        } else {
            code = aclrtMemcpy2dAsync(dst, dp, src, sp, c.width, c.rows,
                                    static_cast<aclrtMemcpyKind>(kind), stream);
        }
        if (code) break;
    }
    ticks[1] = now();
    // Even an accepted prefix must be stopped before a failed batch can retire.
    const auto completed = aclrtSynchronizeStream(stream);
    ticks[2] = now();
    return code ? code : completed;
}
