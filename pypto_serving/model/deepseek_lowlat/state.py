# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""One synchronous request generation, with completion-driven KV cursors."""

from dataclasses import dataclass, field


@dataclass
class LowlatRequestState:
    request_id: str
    generation: int
    prompt_len: int
    prefilled_tokens: int = 0
    kv_committed_tokens: int = 0
    last_sampled_token: int | None = None
    phase: str = "prefill"
    pending_handles: list = field(default_factory=list)

    def check_prefill(self, start, length):
        if (
            self.phase != "prefill"
            or start != self.prefilled_tokens
            or not 0 < length <= self.prompt_len - start
        ):
            raise ValueError("lowlat prefill cursor or request phase mismatch")

    def commit_prefill(self, end):
        self.prefilled_tokens = self.kv_committed_tokens = end

    def check_decode(self, position, token):
        if self.phase != "decode" or position != self.kv_committed_tokens or token != self.last_sampled_token:
            raise ValueError("lowlat decode cursor/token does not match the committed request")
