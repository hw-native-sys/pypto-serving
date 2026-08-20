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

from collections.abc import Mapping, Sequence
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

# A dimension is either a literal or the name of a `LayerContext` field to read, which keeps
# a shape that depends on the model config expressible as data rather than as a lambda.
Dim = int | str


def resolve_shape(shape: Sequence[Dim], context: "LayerContext") -> tuple[int, ...]:
    """Resolve literal and context-derived dimensions into a concrete shape."""
    resolved: list[int] = []
    for dim in shape:
        if isinstance(dim, str):
            resolved.append(int(getattr(context, dim)))
        else:
            resolved.append(int(dim))
    return tuple(resolved)


@dataclass(frozen=True)
class OptionalWeightRule:
    """A weight that exists only for some attention kinds, and is zero-filled otherwise.

    The inactive branch is not skipped: it is written as zeros at ``absent_shape``, because
    every layer must present the same kernel signature. That shape is fixed by the model
    rather than derived from the checkpoint — which is also why a synthetic checkpoint has to
    use production dimensions for these tensors.
    """

    name: str
    source: str
    dtype: torch.dtype
    absent_shape: tuple[Dim, ...]
    enabled_ratios: tuple[int, ...]
    transpose: bool = False

    def enabled_for(self, compress_ratio: int) -> bool:
        """Return whether the checkpoint carries this weight at ``compress_ratio``."""
        return int(compress_ratio) in self.enabled_ratios


@dataclass(frozen=True)
class DefaultedWeightRule:
    """A weight the checkpoint may omit, with a zero default and a per-layer requirement.

    ``required_when`` names the ``LayerContext`` flag that decides whether an absent source is
    an error or simply means "this layer does not use that router mode".
    """

    name: str
    source: str
    dtype: torch.dtype
    default_shape: tuple[Dim, ...]
    required_when: str


@dataclass(frozen=True)
class SyntheticWeightRule:
    """A weight computed rather than read — the factory is looked up by key, not embedded."""

    name: str
    dtype: torch.dtype
    factory: str


@dataclass(frozen=True)
class ExpertWeightRule:
    """One expert weight, stacked rank-major over each rank's local expert slice."""

    name: str
    source: str
    dtype: torch.dtype


LayerRule = LayerWeightRule | OptionalWeightRule | DefaultedWeightRule | SyntheticWeightRule | ExpertWeightRule


@dataclass(frozen=True)
class GlobalWeightRule:
    """A whole-model weight — embedding, LM head, final norm — and how it is conditioned.

    These differ from layer weights in three ways that all have to be expressible, because
    getting any of them wrong is silent rather than loud:

    * ``fallback_source`` — a checkpoint with tied embeddings ships no ``lm_head``, and the
      embedding is used in its place. Falling back is correct; inventing zeros is not.
    * ``pad_to_multiple`` — the fused LM head hard-codes a padded vocabulary, so the weight has
      to be grown to match it. The padded rows are never selected at runtime, but they are
      matmul operands, so what goes in them still matters numerically.
    * ``pad_fill`` — and it is not the same for the two: the embedding pads with zeros, while
      the LM head pads by **replicating its first row**. Zero rows in an LM head would score
      every padded token identically at logit 0, which is a plausible-looking value rather than
      an impossible one, so a wrong choice here survives review and shows up as sampling noise.
    """

    name: str
    source: str
    dtype: torch.dtype
    fallback_source: str | None = None
    pad_to_multiple: int | None = None
    pad_fill: str = "zeros"
    flatten_to_row: bool = False

    def resolve(self, available: "Mapping[str, torch.Tensor]") -> torch.Tensor:
        """Pick this weight's source tensor, honouring the tied-weight fallback."""
        tensor = available.get(self.source)
        if tensor is not None:
            return tensor
        if self.fallback_source is not None:
            fallback = available.get(self.fallback_source)
            if fallback is not None:
                return fallback
        raise KeyError(
            f"missing global weight {self.name}: neither {self.source!r} nor its fallback "
            f"{self.fallback_source!r} is present"
        )
