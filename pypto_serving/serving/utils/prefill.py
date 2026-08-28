# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from pypto_serving.config.types import KvAllocation, PrefillBatch


def pack_prefill_batch(
    *,
    request_ids: Sequence[str],
    token_chunks: Sequence[Sequence[int]],
    seq_lens: Sequence[int],
    chunk_starts: Sequence[int],
    device: str | torch.device,
    embedding_lookup: Callable[[torch.Tensor], torch.Tensor] | None = None,
    allow_device_greedy_sampling: bool = False,
    allow_device_topk_sampling: bool = False,
    kv_allocations: Sequence[KvAllocation] = (),
    block_ids: Sequence[Sequence[int]] = (),
    block_ids_by_group: Sequence[dict[str, list[int]]] = (),
    cache_partitions: Sequence[int | None] = (),
    next_prefill_token_ids: Sequence[int | None] = (),
) -> PrefillBatch:
    """Pack request chunks and optionally look up all host embeddings once."""
    chunk_lens = [len(chunk) for chunk in token_chunks]
    chunk_offsets: list[int] = []
    next_offset = 0
    for chunk_len in chunk_lens:
        chunk_offsets.append(next_offset)
        next_offset += chunk_len

    packed_token_ids = torch.tensor(
        [token_id for chunk in token_chunks for token_id in chunk],
        dtype=torch.long,
        device=device,
    )
    input_embeddings = (
        None if embedding_lookup is None else embedding_lookup(packed_token_ids)
    )
    return PrefillBatch(
        request_ids=list(request_ids),
        token_ids=packed_token_ids,
        input_embeddings=input_embeddings,
        seq_lens=list(seq_lens),
        chunk_lens=chunk_lens,
        chunk_offsets=chunk_offsets,
        chunk_starts=list(chunk_starts),
        allow_device_greedy_sampling=allow_device_greedy_sampling,
        allow_device_topk_sampling=allow_device_topk_sampling,
        kv_allocations=list(kv_allocations),
        block_ids=[list(row) for row in block_ids],
        block_ids_by_group=list(block_ids_by_group),
        cache_partitions=list(cache_partitions),
        next_prefill_token_ids=list(next_prefill_token_ids),
    )
