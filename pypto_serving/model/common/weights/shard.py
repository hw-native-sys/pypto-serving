# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Rank-shard policies: how one packed tensor is distributed across the rank axis."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Replicate:
    """Give every rank the same tensor, as ``[ranks, *tensor.shape]``.

    Two paths, and they are not interchangeable: with a ``destination`` the tensor is copied
    into a slice of a preallocated whole-model slab, which is what keeps the stacker from
    ever calling ``torch.cat``; without one a fresh buffer is expanded and made contiguous.

    The dtype conversion differs in *form* between them — explicit ``.to(dtype)`` on the
    direct path, implicit inside ``copy_()`` on the destination path — and callers rely on
    the two agreeing bit for bit. They do for the conversions in use, but it is a property
    worth a test rather than an assumption, so the parity suite exercises both paths.
    """

    ranks: int
    mismatch_error: str = "packed destination {name} shape/dtype mismatch: expected={expected}, got={got}"

    def apply(
        self,
        name: str,
        tensor: torch.Tensor,
        *,
        dtype: torch.dtype | None,
        destination: torch.Tensor | None,
    ) -> torch.Tensor:
        """Replicate ``tensor`` across ranks, into ``destination`` when one is given."""
        source = tensor.cpu() if tensor.device.type != "cpu" else tensor
        output_dtype = source.dtype if dtype is None else dtype
        expected_shape = (self.ranks, *source.shape)
        if destination is not None:
            if tuple(destination.shape) != expected_shape or destination.dtype != output_dtype:
                raise ValueError(
                    self.mismatch_error.format(
                        name=name,
                        expected=f"{expected_shape}/{output_dtype}",
                        got=f"{tuple(destination.shape)}/{destination.dtype}",
                    )
                )
            destination.copy_(source.unsqueeze(0))
            return destination
        if dtype is not None:
            source = source.to(dtype=dtype)
        return source.contiguous().unsqueeze(0).expand(self.ranks, *source.shape).contiguous()
