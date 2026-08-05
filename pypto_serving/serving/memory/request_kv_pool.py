# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import math
from collections.abc import Sequence


class RequestKVPool:
    """Request-to-page mappings with SGLang-compatible logical slot views."""

    def __init__(self, page_size: int) -> None:
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        self.page_size = int(page_size)
        self._request_pages: dict[str, list[int]] = {}

    def has_request(self, request_id: str) -> bool:
        return request_id in self._request_pages

    def set_pages(self, request_id: str, page_ids: Sequence[int]) -> None:
        pages = [int(page_id) for page_id in page_ids]
        if len(set(pages)) != len(pages):
            raise ValueError("a request cannot map the same physical page more than once")
        self._request_pages[request_id] = pages

    def extend_pages(self, request_id: str, page_ids: Sequence[int]) -> None:
        current = self._request_pages.setdefault(request_id, [])
        new_pages = [int(page_id) for page_id in page_ids]
        if set(current).intersection(new_pages) or len(set(new_pages)) != len(new_pages):
            raise ValueError("request page mappings must be unique")
        current.extend(new_pages)

    def page_ids(self, request_id: str) -> list[int]:
        return list(self._request_pages.get(request_id, []))

    def capacity(self, request_id: str) -> int:
        return len(self._request_pages.get(request_id, [])) * self.page_size

    def slot_indices(self, request_id: str, token_count: int) -> list[int]:
        if token_count < 0:
            raise ValueError("token_count must be non-negative")
        pages = self._request_pages.get(request_id, [])
        if token_count > len(pages) * self.page_size:
            raise ValueError(
                f"request {request_id} has capacity {len(pages) * self.page_size}, "
                f"cannot map {token_count} tokens"
            )
        return [
            pages[position // self.page_size] * self.page_size + position % self.page_size
            for position in range(token_count)
        ]

    def free(self, request_id: str) -> list[int]:
        return self._request_pages.pop(request_id, [])

    def clear(self) -> None:
        self._request_pages.clear()

    def page_ids_from_slots(self, slot_indices: Sequence[int]) -> list[int]:
        if len(slot_indices) % self.page_size:
            raise ValueError("canonical prefix slots must be page aligned")
        pages: list[int] = []
        for start in range(0, len(slot_indices), self.page_size):
            first = int(slot_indices[start])
            page_id = first // self.page_size
            expected = [page_id * self.page_size + offset for offset in range(self.page_size)]
            actual = [int(slot) for slot in slot_indices[start : start + self.page_size]]
            if actual != expected:
                raise ValueError("canonical prefix slots must contain contiguous complete pages")
            pages.append(page_id)
        return pages

    def blocks_needed(self, request_id: str, token_count: int) -> int:
        needed = math.ceil(token_count / self.page_size) if token_count else 0
        return max(0, needed - len(self._request_pages.get(request_id, [])))
