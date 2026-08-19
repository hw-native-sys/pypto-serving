# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Generic per-layer evaluator: rules plus raw tensors in, packed kernel weights out."""

from collections.abc import Mapping, Sequence

import torch

from .shard import Replicate
from .spec import LayerContext, LayerWeightRule


def _reshaped_groups(name: str, weight: torch.Tensor, groups: int) -> torch.Tensor:
    """Split a flattened leading dimension into ``[groups, rows // groups, cols]``."""
    if weight.ndim != 2:
        raise ValueError(f"{name} weight must be rank-2, got shape={tuple(weight.shape)}")
    rows = int(weight.shape[0])
    if rows % groups != 0:
        raise ValueError(f"{name} first dimension {rows} must divide by {groups}")
    return weight.reshape(groups, rows // groups, int(weight.shape[1]))


def pack_layer(
    rules: Sequence[LayerWeightRule],
    raw: Mapping[str, torch.Tensor],
    context: LayerContext,
    *,
    policy: Replicate,
    destinations: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Evaluate ``rules`` against ``raw``, writing into ``destinations`` when given.

    Rule order is the output order, which matters beyond aesthetics: the slab allocator walks
    the packed mapping to lay out the whole-model tensors, and the prepacked sidecar records
    the resulting name-to-offset map. Reordering the rules would silently invalidate every
    sidecar already on disk.

    A rule whose destination is absent is skipped rather than packed and thrown away — that is
    how a caller stages a subset (one group of a multi-group layout) without paying for the rest.
    """
    packed: dict[str, torch.Tensor] = {}
    for rule in rules:
        if destinations is not None and rule.name not in destinations:
            continue
        source_name = context.source_name(rule.source)
        try:
            tensor = raw[source_name]
        except KeyError as exc:
            raise KeyError(f"missing raw layer tensor: {source_name}") from exc
        if rule.reshape_groups is not None:
            tensor = _reshaped_groups(rule.name, tensor, rule.reshape_groups)
        if rule.transpose:
            tensor = tensor.transpose(0, 1)
        packed[rule.name] = policy.apply(
            rule.name,
            tensor,
            dtype=rule.dtype,
            destination=None if destinations is None else destinations[rule.name],
        )
    return packed
