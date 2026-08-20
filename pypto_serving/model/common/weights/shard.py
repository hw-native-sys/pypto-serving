# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Rank-shard policies: how one packed tensor is distributed across the rank axis."""

from collections.abc import Callable, Sequence
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


@dataclass(frozen=True)
class ExpertParallel:
    """Give each rank its own slice of the experts, as ``[ranks, local, *expert.shape]``.

    Placement is injected rather than assumed: which experts a rank owns is a model
    decision, and a policy that hard-coded "contiguous block" would silently reshuffle a
    family that numbers them differently. The two paths mirror ``Replicate``'s — into slab
    slices, or stacking fresh buffers — and carry the same cast asymmetry, explicit on the
    direct path and implicit inside ``copy_()`` on the destination one.
    """

    ranks: int
    n_experts: int
    local_ids: Callable[..., Sequence[int]]
    mismatch_error: str = "packed destination {name} shape/dtype mismatch: expected={expected}, got={got}"

    def _ids_for(self, rank: int) -> Sequence[int]:
        return self.local_ids(rank=rank, ranks=self.ranks, n_routed_experts=self.n_experts)

    def apply(
        self,
        name: str,
        lookup: Callable[[int], torch.Tensor],
        *,
        dtype: torch.dtype,
        destination: torch.Tensor | None,
    ) -> torch.Tensor:
        """Stack each rank's local experts, into ``destination`` when one is given."""
        if destination is not None:
            local_experts = self.n_experts // self.ranks
            expected_shape = (self.ranks, local_experts, *lookup(0).shape)
            if tuple(destination.shape) != expected_shape or destination.dtype != dtype:
                raise ValueError(
                    self.mismatch_error.format(
                        name=name,
                        expected=f"{expected_shape}/{dtype}",
                        got=f"{tuple(destination.shape)}/{destination.dtype}",
                    )
                )
            for rank in range(self.ranks):
                for local_index, expert_id in enumerate(self._ids_for(rank)):
                    destination[rank, local_index].copy_(lookup(expert_id))
            return destination

        per_rank = [
            torch.stack(
                [lookup(expert_id).to(dtype=dtype).contiguous().cpu() for expert_id in self._ids_for(rank)],
                dim=0,
            )
            for rank in range(self.ranks)
        ]
        return torch.stack(per_rank, dim=0).contiguous()


@dataclass(frozen=True)
class NoShard:
    """Keep the tensor as it is — no rank axis at all.

    For a family whose kernels take one copy of each weight, so there is nothing to distribute.
    It is a distinct policy rather than ``Replicate(ranks=1)`` because the shapes differ: a
    replicated weight is ``[1, *shape]`` and a rank-less one is ``[*shape]``, and a slab built
    from the wrong one is the right size with an extra axis nothing indexes.
    """

    mismatch_error: str = "packed destination {name} shape/dtype mismatch: expected={expected}, got={got}"

    def apply(
        self,
        name: str,
        tensor: torch.Tensor,
        *,
        dtype: torch.dtype | None,
        destination: torch.Tensor | None,
    ) -> torch.Tensor:
        """Cast *tensor*, writing into ``destination`` when one is given."""
        source = tensor.cpu() if tensor.device.type != "cpu" else tensor
        output_dtype = source.dtype if dtype is None else dtype
        if destination is not None:
            if tuple(destination.shape) != tuple(source.shape) or destination.dtype != output_dtype:
                raise ValueError(
                    self.mismatch_error.format(
                        name=name,
                        expected=f"{tuple(source.shape)}/{output_dtype}",
                        got=f"{tuple(destination.shape)}/{destination.dtype}",
                    )
                )
            # The cast rides along inside copy_, as it does on the replicated path.
            destination.copy_(source)
            return destination
        if dtype is not None:
            source = source.to(dtype=dtype)
        return source.contiguous()
