# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate scheduler metadata before taking request/cache state ownership.

Positions and page tables stay logical here; the lib composite adapter lowers
them to its own physical buffers. There is no implicit assignment from batch
row to DP partition or compressor slot.
"""

from collections.abc import Mapping, Sequence
import math

import torch

from pypto_serving.config.types import KVCacheGroupSpec
from .composite import MissingCompositeInterface


def validate_page_tables(rows, partitions, ends, groups):
    """Copy private full-history page tables and reject active request aliases.

    Physical IDs are local to each DP partition. Without an explicit num_blocks
    the device allocator must check its actual capacity before launching work.
    Rolling groups require a position-to-ring-slot ABI and are not guessed here.
    """
    if not groups or any(not isinstance(group, KVCacheGroupSpec) for group in groups):
        raise MissingCompositeInterface("V4.1 requires explicit grouped cache specifications")
    if len({group.name for group in groups}) != len(groups):
        raise ValueError("cache group names must be unique")
    if any(group.num_partitions != 2 for group in groups):
        raise ValueError("V4.1 cache groups require two DP partitions")
    if any(group.sliding_window is not None for group in groups):
        raise MissingCompositeInterface("rolling cache page lowering requires a verified composite contract")
    if len(rows) != len(partitions) or len(rows) != len(ends):
        raise ValueError("page tables, partitions and request extents must have matching rows")
    required = {group.name for group in groups}
    owners, result = set(), []
    for pages, partition, end in zip(rows, partitions, ends):
        if type(partition) is not int or not 0 <= partition < 2:
            raise ValueError("requests require an explicit DP cache partition in [0, 2)")
        if type(end) is not int or end <= 0:
            raise ValueError("cache request extent must be a positive integer")
        if not isinstance(pages, Mapping) or set(pages) != required:
            raise ValueError("request page tables must match the declared cache groups")
        copied = {}
        for group in groups:
            ids = pages[group.name]
            if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes)):
                raise ValueError("a cache page table must be a sequence of physical IDs")
            ids = tuple(ids)
            if any(type(page) is not int or page < 0 for page in ids):
                raise ValueError("physical page IDs must be nonnegative integers")
            if group.num_blocks is not None and any(page >= group.num_blocks for page in ids):
                raise ValueError("physical page ID exceeds the configured cache pool")
            required_pages = math.ceil(end / group.spec.token_capacity)
            if not required_pages <= len(ids) <= group.max_blocks_per_seq:
                raise ValueError("cache page table does not cover the request extent within its capacity")
            if len(set(ids)) != len(ids):
                raise ValueError("a full-history request must not alias its own cache pages")
            addresses = {(partition, group.name, page) for page in ids}
            if owners.intersection(addresses):
                raise ValueError("active requests must not share writable cache pages")
            owners.update(addresses)
            copied[group.name] = ids
        result.append(copied)
    return tuple(result)


def prefill_requests(batch, config, runtime, groups):
    """Build RequestLedger.begin_prefill items from the shared packed batch.

    seq_lens is the end of the current chunk; prompt_lens is the original total
    prompt length. Confusing them would mark the first chunk as terminal and
    prevent subsequent chunks from continuing their persistent state.
    """
    count = len(batch.request_ids)
    if not count or count > runtime.max_batch_size:
        raise ValueError("prefill request count exceeds the configured batch capacity")
    if any(not isinstance(key, str) or not key for key in batch.request_ids):
        raise ValueError("request IDs must be nonempty strings")
    if len(set(batch.request_ids)) != count:
        raise ValueError("prefill requests must have distinct IDs")
    for name in ("chunk_lens", "chunk_offsets", "chunk_starts", "seq_lens", "prompt_lens",
                 "block_ids_by_group", "cache_partitions"):
        if len(getattr(batch, name)) != count:
            raise ValueError(f"prefill {name} must contain one entry per request")
    tokens = batch.token_ids
    if not isinstance(tokens, torch.Tensor) or tokens.device.type != "cpu" or tokens.ndim != 1:
        raise ValueError("prefill token IDs must be a flat CPU tensor")
    if tokens.dtype not in (torch.int32, torch.int64):
        raise ValueError("prefill token IDs must use int32 or int64")
    if tokens.numel() > runtime.max_num_batched_tokens:
        raise ValueError("prefill tokens exceed the configured dispatch capacity")
    token_ids = tokens.tolist()
    if any(not 0 <= token < config.vocab_size for token in token_ids):
        raise ValueError("prefill token ID is outside the vocabulary")

    cursor, chunks = 0, []
    for size, offset, start, end, prompt in zip(
        batch.chunk_lens, batch.chunk_offsets, batch.chunk_starts, batch.seq_lens, batch.prompt_lens,
    ):
        if any(type(value) is not int for value in (size, offset, start, end, prompt)):
            raise ValueError("prefill offsets and lengths must be integers")
        if size <= 0 or offset != cursor or offset + size > len(token_ids):
            raise ValueError("prefill chunks must cover consecutive packed token spans")
        if not 0 <= start < end == start + size <= prompt <= min(
            runtime.max_seq_len, config.max_position_embeddings,
        ):
            raise ValueError("prefill chunk extent or total prompt length is invalid")
        if runtime.max_prefill_tokens_per_request is not None and size > runtime.max_prefill_tokens_per_request:
            raise ValueError("prefill chunk exceeds the per-request capacity")
        chunks.append(tuple(token_ids[offset:offset + size]))
        cursor += size
    if cursor != len(token_ids):
        raise ValueError("prefill chunk spans leave unclaimed packed tokens")
    pages = validate_page_tables(
        batch.block_ids_by_group, batch.cache_partitions, batch.seq_lens, groups,
    )
    return [
        (key, partition, start, chunk, prompt, table)
        for key, partition, start, chunk, prompt, table in zip(
            batch.request_ids, batch.cache_partitions, batch.chunk_starts, chunks, batch.prompt_lens, pages,
        )
    ]


def decode_requests(batch, config, runtime, groups):
    """Decode consumes exactly one supplied token per request; no padding rows."""
    count = len(batch.request_ids)
    if not count or count > min(runtime.max_batch_size, runtime.max_num_batched_tokens):
        raise ValueError("decode request count exceeds the configured batch capacity")
    if len(set(batch.request_ids)) != count or any(not isinstance(k, str) or not k for k in batch.request_ids):
        raise ValueError("decode requires distinct nonempty request IDs")
    tokens, lengths = batch.token_ids, batch.seq_lens
    for value in (tokens, lengths):
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype not in (
            torch.int32, torch.int64,
        ):
            raise ValueError("decode tokens and lengths must be CPU integer tensors")
    if tuple(tokens.shape) not in ((count,), (count, 1)) or tuple(lengths.shape) != (count,):
        raise ValueError("decode requires one token and one sequence length per request")
    ids, ends = tokens.reshape(-1).tolist(), lengths.tolist()
    if any(not 0 <= token < config.vocab_size for token in ids):
        raise ValueError("decode token ID is outside the vocabulary")
    if any(not 1 <= end <= min(runtime.max_seq_len, config.max_position_embeddings) for end in ends):
        raise ValueError("decode sequence length exceeds model capacity")
    if len(batch.cache_partitions) != count or len(batch.block_ids_by_group) != count:
        raise ValueError("decode requires one partition and grouped page table per request")
    pages = validate_page_tables(batch.block_ids_by_group, batch.cache_partitions, ends, groups)
    return [(key, partition, end - 1, token, table)
            for key, partition, end, token, table in zip(
                batch.request_ids, batch.cache_partitions, ends, ids, pages)]
