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
class TensorParallel:
    """Shard one tensor axis across each TP group and repeat it across DP groups.

    Physical rank ``r`` consumes shard ``r % tp_size``. This is the layout used
    by model programs whose outer rank axis spans the full DP x TP world while
    a dense projection is partitioned only inside each TP group.
    """

    ranks: int
    tp_size: int
    axis: int
    mismatch_error: str = "packed destination {name} shape/dtype mismatch: expected={expected}, got={got}"

    def apply(
        self,
        name: str,
        tensor: torch.Tensor,
        *,
        dtype: torch.dtype | None,
        destination: torch.Tensor | None,
    ) -> torch.Tensor:
        """Shard ``tensor`` into the rank-major DP x TP layout."""
        if self.ranks <= 0 or self.tp_size <= 0 or self.ranks % self.tp_size != 0:
            raise ValueError(
                f"{name} requires positive ranks divisible by tp_size, "
                f"got ranks={self.ranks}, tp_size={self.tp_size}"
            )
        source = tensor.cpu() if tensor.device.type != "cpu" else tensor
        axis = self.axis if self.axis >= 0 else source.ndim + self.axis
        if not 0 <= axis < source.ndim:
            raise ValueError(f"{name} TP axis {self.axis} is invalid for shape={tuple(source.shape)}")
        width = int(source.shape[axis])
        if width % self.tp_size != 0:
            raise ValueError(
                f"{name} dimension {width} on axis {axis} must divide by tp_size={self.tp_size}"
            )

        output_dtype = source.dtype if dtype is None else dtype
        shard_shape = list(source.shape)
        shard_shape[axis] //= self.tp_size
        expected_shape = (self.ranks, *shard_shape)
        if destination is not None:
            if tuple(destination.shape) != expected_shape or destination.dtype != output_dtype:
                raise ValueError(
                    self.mismatch_error.format(
                        name=name,
                        expected=f"{expected_shape}/{output_dtype}",
                        got=f"{tuple(destination.shape)}/{destination.dtype}",
                    )
                )
            shards = source.chunk(self.tp_size, dim=axis)
            for rank in range(self.ranks):
                destination[rank].copy_(shards[rank % self.tp_size])
            return destination

        if dtype is not None:
            source = source.to(dtype=dtype)
        shards = source.chunk(self.tp_size, dim=axis)
        return torch.stack([shards[rank % self.tp_size] for rank in range(self.ranks)], dim=0).contiguous()


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
