# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Lowlat prefix resume and address-free execution metadata."""

from dataclasses import dataclass

from pypto_serving.serving.memory.protocol import CachePageIdentity, GroupPageRange


@dataclass(frozen=True)
class PrefixResumeDescriptor:
    namespace: str
    request_id: str
    execution_generation: int
    hit_tokens: int
    replay_tokens: int
    mapping_version: int
    pages: tuple[CachePageIdentity, ...]
    valid_ranges: tuple[GroupPageRange, ...]
    tables: dict[str, tuple[int, ...]]

    def __post_init__(self):
        if (
            not self.namespace
            or not self.request_id
            or any(
                type(v) is not int
                for v in (
                    self.execution_generation,
                    self.hit_tokens,
                    self.replay_tokens,
                    self.mapping_version,
                )
            )
            or min(self.execution_generation, self.hit_tokens, self.mapping_version) <= 0
            or self.replay_tokens < 128
            or self.hit_tokens % 128
            or len(set(self.pages)) != len(self.pages)
        ):
            raise ValueError("invalid prefix resume descriptor")
