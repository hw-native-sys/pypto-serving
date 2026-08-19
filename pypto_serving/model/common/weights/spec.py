# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Declarative description of one layer's checkpoint-to-kernel mapping.

A family says *what* its weights are — which checkpoint tensor feeds which kernel name,
in which dtype, with which orientation — and the packer decides *how* to produce them. The
point of keeping this as data is that adding a model family should not mean adding another
hand-written pack function; the point of keeping it small is that every field here has to
be reproducible byte-for-byte by the evaluator.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LayerContext:
    """Everything a rule may need to know about the layer being packed."""

    layer_id: int
    prefix: str
    ranks: int
    compress_ratio: int = 0
    n_routed_experts: int = 0
    include_tid2eid: bool = False
    include_gate_bias: bool = False

    def source_name(self, suffix: str) -> str:
        """Return the checkpoint name for ``suffix`` under this layer's prefix."""
        return f"{self.prefix}.{suffix}"


@dataclass(frozen=True)
class LayerWeightRule:
    """One kernel weight: where it comes from and the shape-preserving edits it needs.

    ``transpose`` and ``reshape_groups`` are the only transforms expressed here on purpose.
    Both are pure re-orientations of the same bytes, which is what lets the evaluator stay
    generic; anything that computes new values belongs in a family's own code, not in a
    field that looks declarative but hides arithmetic.
    """

    name: str
    source: str
    dtype: torch.dtype
    transpose: bool = False
    reshape_groups: int | None = None
