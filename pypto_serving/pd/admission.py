# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Receiver requirements only; Serving still owns matching and allocation."""


class DFirst:
    """Require a stable HBM prompt reservation before source computation."""

    def requirements(self, prompt_tokens, max_new_tokens, prefix):
        return dict(token_capacity=prompt_tokens + max_new_tokens,
                    prompt_token_count=prompt_tokens,
                    prefix_block_hashes={name: [bytes.fromhex(value) for value in hashes]
                                         for name, hashes in prefix.hashes.items()},
                    shareable_group_names=tuple(prefix.hashes))

    def accepts(self, active, limit):
        return active < limit
