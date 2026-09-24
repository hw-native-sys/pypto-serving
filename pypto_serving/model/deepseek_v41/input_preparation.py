# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded CPU embedding preparation using the checkpoint's vocabulary shards.

Like V4's executor lookup, this is explicit host input preparation. It is not a
device embedding implementation or a fallback for a missing lib composite.
Residual expansion and the initial delayed pre_mix belong to the selected lib
initialization contract; an Attention golden fixture does not establish them.
"""

from collections.abc import Sequence

import torch

from .weight_loader import V41WeightLoader


def lookup_token_embeddings(
    loaders: Sequence[V41WeightLoader],
    token_ids: torch.Tensor,
    *,
    max_prepare_bytes: int = 256 << 20,
    max_rows_per_read: int = 64,
) -> torch.Tensor:
    """Return owned CPU BF16 rows with shape ``(*token_ids.shape, hidden_size)``.

    Supply one complete set of TP vocabulary shards from the same checkpoint,
    in any order. DP groups share this checkpoint table; token order and all
    leading dimensions are preserved, without duplicating rows for TP ranks.
    IDs must be CPU int32/int64 with at least one dimension. Empty inputs are
    allowed, but padding sentinels such as -1 are not vocabulary indices.

    Repeated IDs are read once. Adjacent unique rows in the same TP shard are
    read together, with at most max_rows_per_read rows per load_rows call.
    max_prepare_bytes bounds conservative preparation tensor storage, including
    the result, unique rows, indexing scratch and a read buffer; the loader's
    separate max_load_bytes still bounds its own read/conversion operation.
    Neither budget represents total process memory or previously returned data.
    """
    for value, name in ((max_prepare_bytes, "max_prepare_bytes"),
                        (max_rows_per_read, "max_rows_per_read")):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    shards = tuple(loaders)
    if not shards or any(not isinstance(loader, V41WeightLoader) for loader in shards):
        raise ValueError("embedding lookup requires V41WeightLoader TP shards")
    first = shards[0]
    if len(shards) != first.tp_size or {loader.tp_rank for loader in shards} != set(range(first.tp_size)):
        raise ValueError("embedding lookup requires exactly one loader per TP shard")
    if any(loader.model_dir != first.model_dir or loader.config != first.config
           or loader.tp_size != first.tp_size or loader.ep_size != first.ep_size for loader in shards):
        raise ValueError("embedding TP shards must use the same checkpoint and parallel sizes")
    if not isinstance(token_ids, torch.Tensor) or token_ids.layout != torch.strided:
        raise ValueError("token_ids must be a strided CPU tensor")
    if token_ids.device.type != "cpu" or token_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("token_ids must be a CPU int32/int64 tensor")
    if token_ids.ndim == 0:
        raise ValueError("token_ids must have at least one dimension")

    hidden = first.config.hidden_size
    count = token_ids.numel()
    if count == 0:
        return torch.empty((*token_ids.shape, hidden), dtype=torch.bfloat16)
    # Bound allocations before flattening or sorting. Worst case every ID is
    # unique; count*128 allows conservative int64 indexing/sorting scratch.
    row_bytes = hidden * 2
    estimate = (2 * count + min(count, max_rows_per_read)) * row_bytes + count * 128
    if estimate > max_prepare_bytes:
        raise ValueError(
            f"embedding preparation requires estimated {estimate} bytes; budget={max_prepare_bytes}"
        )
    if bool(((token_ids < 0) | (token_ids >= first.config.vocab_size)).any()):
        raise ValueError("token_ids must be inside the checkpoint vocabulary")

    flat = token_ids.reshape(-1).to(torch.int64)
    unique, inverse = torch.unique(flat, sorted=True, return_inverse=True)
    unique_rows = torch.empty((unique.numel(), hidden), dtype=torch.bfloat16)
    by_rank = {loader.tp_rank: loader for loader in shards}
    shard_size = first.config.vocab_size // first.tp_size
    start = 0
    while start < unique.numel():
        token = int(unique[start])
        rank, local_row = divmod(token, shard_size)
        stop = start + 1
        limit = min(unique.numel(), start + max_rows_per_read, start + shard_size - local_row)
        while stop < limit and int(unique[stop]) == token + stop - start:
            stop += 1
        bundle = by_rank[rank].load_rows("embed.weight", local_row, local_row + stop - start)
        unique_rows[start:stop].copy_(bundle.weight)
        del bundle
        start = stop
    return unique_rows.index_select(0, inverse).reshape(*token_ids.shape, hidden)
