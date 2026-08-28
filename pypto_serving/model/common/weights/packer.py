# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Generic per-layer evaluator: rules plus raw tensors in, packed kernel weights out."""

from collections.abc import Callable, Mapping, Sequence

import torch

from .shard import ExpertParallel, NoShard, Replicate, TensorParallel
from .spec import (
    DefaultedWeightRule,
    ExpertWeightRule,
    GlobalWeightRule,
    LayerContext,
    LayerRule,
    LayerWeightRule,
    OptionalWeightRule,
    SyntheticWeightRule,
    resolve_shape,
)


def _reshaped_groups(name: str, weight: torch.Tensor, groups: int) -> torch.Tensor:
    """Split a flattened leading dimension into ``[groups, rows // groups, cols]``."""
    if weight.ndim != 2:
        raise ValueError(f"{name} weight must be rank-2, got shape={tuple(weight.shape)}")
    rows = int(weight.shape[0])
    if rows % groups != 0:
        raise ValueError(f"{name} first dimension {rows} must divide by {groups}")
    return weight.reshape(groups, rows // groups, int(weight.shape[1]))


def pack_layer(
    rules: Sequence[LayerRule],
    raw: Mapping[str, torch.Tensor],
    context: LayerContext,
    *,
    policy: Replicate | NoShard,
    expert_policy: ExpertParallel | None = None,
    policies: Mapping[str, Replicate | NoShard | TensorParallel] | None = None,
    factories: Mapping[str, Callable[[], torch.Tensor]] | None = None,
    destinations: Mapping[str, torch.Tensor] | None = None,
    missing_source_error: str = "missing raw layer tensor: {name}",
    missing_expert_error: str = "missing raw expert tensor: {name}",
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
        destination = None if destinations is None else destinations[rule.name]
        selected_policy = (policies or {}).get(rule.name, policy)

        if isinstance(rule, ExpertWeightRule):
            if expert_policy is None:
                raise ValueError(f"{rule.name} needs an expert policy but none was given")

            def _expert(expert_id: int, rule: ExpertWeightRule = rule) -> torch.Tensor:
                name = context.source_name(f"ffn.experts.{expert_id}.{rule.source}")
                try:
                    return raw[name]
                except KeyError as exc:
                    raise KeyError(missing_expert_error.format(name=name)) from exc

            packed[rule.name] = expert_policy.apply(
                rule.name, _expert, dtype=rule.dtype, destination=destination
            )
            continue

        if isinstance(rule, SyntheticWeightRule):
            factory = (factories or {}).get(rule.factory)
            if factory is None:
                raise ValueError(f"{rule.name} needs the {rule.factory!r} factory but none was given")
            packed[rule.name] = selected_policy.apply(
                rule.name, factory(), dtype=rule.dtype, destination=destination
            )
            continue

        if isinstance(rule, OptionalWeightRule):
            tensor = (
                raw.get(context.source_name(rule.source))
                if rule.enabled_for(context.compress_ratio)
                else None
            )
            if tensor is None:
                packed[rule.name] = _zero_filled(
                    rule.name,
                    resolve_shape(rule.absent_shape, context),
                    rule.dtype,
                    ranks=selected_policy.ranks,
                    destination=destination,
                    mismatch_error=selected_policy.mismatch_error,
                )
                continue
            if rule.transpose:
                tensor = tensor.transpose(0, 1)
            packed[rule.name] = selected_policy.apply(
                rule.name, tensor, dtype=rule.dtype, destination=destination
            )
            continue

        if isinstance(rule, DefaultedWeightRule):
            tensor = raw.get(context.source_name(rule.source))
            if tensor is None:
                if rule.required_when is not None and getattr(context, rule.required_when):
                    raise KeyError(missing_source_error.format(name=context.source_name(rule.source)))
                shape = resolve_shape(rule.default_shape, context)
                if rule.default_fill == "zeros":
                    tensor = torch.zeros(shape, dtype=rule.dtype)
                elif rule.default_fill == "ones":
                    tensor = torch.ones(shape, dtype=rule.dtype)
                else:
                    raise ValueError(f"{rule.name} has unsupported default fill {rule.default_fill!r}")
            if rule.flatten_to_row:
                tensor = tensor.reshape(1, -1)
            packed[rule.name] = selected_policy.apply(
                rule.name, tensor, dtype=rule.dtype, destination=destination
            )
            continue

        if not isinstance(rule, LayerWeightRule):  # pragma: no cover - guards a new rule kind
            raise TypeError(f"unsupported weight rule: {type(rule).__name__}")

        source_name = context.source_name(rule.source)
        try:
            tensor = raw[source_name]
        except KeyError as exc:
            raise KeyError(missing_source_error.format(name=source_name)) from exc
        if rule.reshape_groups is not None:
            tensor = _reshaped_groups(rule.name, tensor, rule.reshape_groups)
        if rule.transpose:
            tensor = tensor.transpose(0, 1)
        if rule.flatten_to_row:
            # reshape, not view: a gamma is 1-D contiguous today, but reshape also handles a
            # non-contiguous source instead of raising.
            tensor = tensor.reshape(1, -1)
        packed[rule.name] = selected_policy.apply(
            rule.name, tensor, dtype=rule.dtype, destination=destination
        )
    return packed


def _zero_filled(
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    *,
    ranks: int,
    destination: torch.Tensor | None,
    mismatch_error: str,
) -> torch.Tensor:
    """Zero an inactive branch, validating the destination's shape before touching it.

    The validation comes first on purpose: a mismatch here means the slab was allocated from
    a template that disagrees with this rule, and zeroing it anyway would hide that behind
    plausible-looking output.
    """
    expected_shape = (ranks, *shape)
    if destination is None:
        return torch.zeros(expected_shape, dtype=dtype)
    if tuple(destination.shape) != expected_shape or destination.dtype != dtype:
        raise ValueError(
            mismatch_error.format(
                name=name,
                expected=f"{expected_shape}/{dtype}",
                got=f"{tuple(destination.shape)}/{destination.dtype}",
            )
        )
    destination.zero_()
    return destination


def _round_up(value: int, multiple: int) -> int:
    return -(-int(value) // int(multiple)) * int(multiple)


def pad_rows(
    name: str,
    weight: torch.Tensor,
    *,
    rows: int,
    fill: str,
) -> torch.Tensor:
    """Grow *weight* to ``rows`` along dim 0, filling the new rows per ``fill``.

    ``"zeros"`` is the neutral choice for an embedding: a padded token id is never looked up.
    ``"first_row"`` replicates row 0, which is what an LM head needs — zero rows there would
    give every padded vocabulary entry the same finite logit rather than an impossible one, so
    the error would look like sampling noise instead of a crash.
    """
    have = int(weight.shape[0])
    if rows == have:
        return weight
    if rows < have:
        raise ValueError(f"{name} has {have} rows, cannot pad down to {rows}")
    missing = rows - have
    if fill == "zeros":
        padding = torch.zeros((missing, *weight.shape[1:]), dtype=weight.dtype, device=weight.device)
    elif fill == "first_row":
        padding = weight[:1].expand(missing, *weight.shape[1:]).clone()
    else:
        raise ValueError(f"{name} has unsupported pad fill {fill!r}")
    return torch.cat([weight, padding], dim=0)


def pack_globals(
    rules: Sequence[GlobalWeightRule],
    available: Mapping[str, torch.Tensor],
    *,
    padded_rows: Mapping[str, int] | None = None,
) -> dict[str, torch.Tensor]:
    """Resolve, pad and cast the whole-model weights named by ``rules``.

    ``padded_rows`` lets a caller state the target row count directly — a fused kernel's
    hard-coded vocabulary, say — instead of it being derived here; a rule with
    ``pad_to_multiple`` and no explicit target rounds its own row count up.

    ``torch.cat`` appears here, unlike in the stacker, and deliberately: padding grows a weight
    that has no preallocated destination, and the whole-model weights are a handful of tensors
    rather than the bulk of the model.
    """
    packed: dict[str, torch.Tensor] = {}
    for rule in rules:
        tensor = rule.resolve(available)
        target = None if padded_rows is None else padded_rows.get(rule.name)
        if target is None and rule.pad_to_multiple is not None:
            target = _round_up(int(tensor.shape[0]), rule.pad_to_multiple)
        if target is not None:
            tensor = pad_rows(rule.name, tensor, rows=target, fill=rule.pad_fill)
        if rule.flatten_to_row:
            tensor = tensor.view(1, -1)
        packed[rule.name] = tensor.to(dtype=rule.dtype).contiguous().cpu()
    return packed
