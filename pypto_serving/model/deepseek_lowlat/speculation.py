# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate native K5 control packets independently of request scheduling."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class NativeState:
    epoch: int
    anchor: int
    remaining: int
    last_token: int

    def __post_init__(self):
        if (any(type(v) is not int for v in vars(self).values()) or self.epoch <= 0
                or self.anchor < 0 or not 0 <= self.remaining <= 2047
                or not 0 <= self.last_token < 129280):
            raise ValueError("invalid native request state")


@dataclass(frozen=True)
class NativeCommit:
    state: NativeState
    token_ids: tuple[int, ...]
    advances: tuple[int, ...]
    accepted: tuple[int, ...]
    controls: tuple[tuple[int, ...], ...]


def validate_packets(control, emitted, *, state, cycles):
    """Check only published cycle rows; rank0 owns commits and other ranks clear outputs."""
    if type(cycles) is not int or cycles not in (1, 2):
        raise ValueError("one or two native cycles required")
    if state.remaining < (7 if cycles == 2 else 1):
        raise ValueError("native parent exceeds the proven remaining budget")
    for value, shape in ((control, (8, 2, 16)), (emitted, (8, 2, 6))):
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                or tuple(value.shape) != shape or value.dtype != torch.int64):
            raise ValueError("native packets require exact CPU INT64 shapes")
    # Whole-parent publication is checked by the frame before this snapshot.
    controls, outputs = control[:, :cycles].tolist(), emitted[:, :cycles].tolist()
    if any(any(field != 0 for field in row) for rank in controls[1:] for row in rank):
        raise ValueError("nonowner control fields were not cleared")
    if any(any(token != -1 for token in row) for rank in outputs[1:] for row in rank):
        raise ValueError("nonowner emitted fields were not cleared")
    tokens, advances, accepted_counts, committed = [], [], [], []
    current = state
    for values, ids in zip(controls[0], outputs[0], strict=True):
        (epoch, status, anchor, accepted, advance, actual, next_anchor,
         bonus, last, remaining, ring_done, feature_start, feature_rows,
         terminated, reserved14, reserved15) = values
        if (epoch != current.epoch or anchor != current.anchor or status != 1 or ring_done != 1
                or not 0 <= accepted <= 5 or advance != accepted+1
                or next_anchor != anchor+advance or feature_start != anchor+1 or feature_rows != advance
                or actual != min(advance, current.remaining) or actual <= 0
                or remaining != current.remaining-actual or terminated != int(remaining == 0)
                or reserved14 or reserved15 or not 0 <= bonus < 129280):
            raise ValueError("native commit does not close its predecessor state/budget")
        active = ids[:actual]
        if (any(not 0 <= token < 129280 for token in active)
                or any(token != -1 for token in ids[actual:]) or active[-1] != last
                or remaining and (actual != advance or bonus != last)):
            raise ValueError("native emitted IDs disagree with the control packet")
        current = NativeState(epoch+1, next_anchor, remaining, last)
        tokens.extend(active)
        advances.append(advance)
        accepted_counts.append(accepted)
        committed.append(tuple(values))
    return NativeCommit(current, tuple(tokens), tuple(advances), tuple(accepted_counts), tuple(committed))


def validate_native_capacity(*, prompt_tokens, max_output_tokens, physical_capacity):
    """Require full final-cycle lookahead; never switch to K0 with unproven speculative state."""
    if (any(type(v) is not int or v <= 0 for v in
            (prompt_tokens, max_output_tokens, physical_capacity)) or max_output_tokens > 2048):
        raise ValueError("invalid native request capacity")
    lookahead = 0 if max_output_tokens == 1 else 6
    if prompt_tokens+max_output_tokens+lookahead > physical_capacity:
        raise ValueError("native K5 needs six physical lookahead positions beyond the logical budget")
