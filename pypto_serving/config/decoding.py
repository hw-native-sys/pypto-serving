# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Address-free decode reservations and confirmed execution receipts."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DecodeReservation:
    reservation_id: int
    request_id: str
    generation: int
    cycles: int
    max_output_tokens: int
    physical_input_end: int
    remaining_lower_bound: int
    frame_id: int

    def __post_init__(self):
        fields = (
            self.reservation_id,
            self.generation,
            self.cycles,
            self.max_output_tokens,
            self.physical_input_end,
            self.remaining_lower_bound,
            self.frame_id,
        )
        if (
            any(type(v) is not int for v in fields)
            or not isinstance(self.request_id, str)
            or not self.request_id
            or min(fields[:2]) <= 0
            or self.cycles <= 0
            or self.frame_id < 0
            or self.physical_input_end <= 0
            or self.remaining_lower_bound <= 0
            or not 0 < self.max_output_tokens <= self.remaining_lower_bound
        ):
            raise ValueError("invalid decode reservation")
