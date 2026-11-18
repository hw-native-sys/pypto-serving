# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Retain real prefill handles through publication, capture and failed drain."""


from .profiling import operation


class PublishedPrefill:
    def __init__(self, handle):
        self.handle = handle
        self.published = False

    @operation("lowlat.prefill.wait")
    def result(self, *args, **kwargs):
        result = self.handle.result(*args, **kwargs)
        self.published = True
        return result


class PrefillSubmissions:
    """Forward the original adapter API while recording every accepted handle."""

    def __init__(self, worker, state, transfers):
        self.worker, self.state, self.transfers = worker, state, transfers
        self.handles = []

    def __getattr__(self, name):
        return getattr(self.worker, name)

    @operation("lowlat.prefill.submit", category="kernel")
    def submit(self, *args, **kwargs):
        handle = self.worker.submit(*args, **kwargs)
        self.state.pending_handles.append(handle)
        published = PublishedPrefill(handle)
        self.handles.append(published)
        return published

    def ensure_published(self):
        if len(self.handles) != 1 or not all(handle.published for handle in self.handles):
            raise RuntimeError("prefill requires one successful whole-chunk publication")

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        if kind is not None:
            self.transfers.poison()
            return False
        try:
            self.ensure_published()
            for published in self.handles:
                self.state.pending_handles.remove(published.handle)
        except BaseException:
            self.transfers.poison()
            raise
        return False
