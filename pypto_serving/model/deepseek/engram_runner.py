# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host-side engram lookup owner, independent of any model runner.

The engram embed tables stay host-resident as read-only shared mappings of
the original checkpoint (each layer is close to a hundred gigabytes, so
nothing is ever uploaded wholesale). :class:`EngramModelRunner` builds the
n-gram hasher once, serves per-token hash ids at input preparation, and
gathers every TP rank's zero-padded partial at prepare time; the device
kernel dequantizes and all-reduces the partials -- matching the reference
``ParallelEngramEmbedding`` sharding without device-resident tables.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

from pypto_serving.model.deepseek.engram_hash import (
    EngramHashLayout,
    NgramHasher,
    build_compressed_token_map,
)
from pypto_serving.model.deepseek.engram_loader import (
    EngramEmbedCheckpoint,
    EngramEmbedLayout,
    EngramEmbedPartial,
)

logger = logging.getLogger(__name__)

__all__ = ["EngramEmbedSource", "EngramModelRunner"]


@dataclass(frozen=True)
class EngramEmbedSource:
    """Where the engram embed tables live and how lookups shard over the TP ranks.

    The tables are absent from the W8A8 checkpoint (the conversion drops them),
    so the source points at the original DeepSeek-V4.1-Flash release.
    ``hash_layout`` carries the rebuilt n-gram bucket geometry the host-side
    hasher needs.
    """

    checkpoint_dir: str
    tp_size: int
    layouts: Mapping[int, EngramEmbedLayout]
    hash_layout: EngramHashLayout


class EngramModelRunner:
    """Owns the host-side engram state: hasher plus mapped embed tables."""

    def __init__(self, source: EngramEmbedSource) -> None:
        self.source = source
        self._hasher: NgramHasher | None = None
        self._tables: EngramEmbedCheckpoint | None = None

    def prepare(self) -> None:
        """Build the hasher and map the tables ahead of the first request."""
        self._require_host_state()

    def _require_host_state(self) -> tuple[NgramHasher, EngramEmbedCheckpoint]:
        """Build (once) the n-gram hasher and the mapped engram tables.

        The hasher constants (bucket primes, multipliers, compressed token
        map) are rebuilt from the config and the checkpoint's raw
        tokenizer -- the same tokenizer training decoded with.
        """
        if self._hasher is not None and self._tables is not None:
            return self._hasher, self._tables
        from tokenizers import Tokenizer

        checkpoint_dir = Path(self.source.checkpoint_dir)
        backend = Tokenizer.from_file(str(checkpoint_dir / "tokenizer.json"))
        token_map = build_compressed_token_map(backend)
        hasher = NgramHasher(
            self.source.hash_layout,
            token_map,
            pad_token_id=self.source.hash_layout.pad_token_id,
        )
        tables = EngramEmbedCheckpoint(checkpoint_dir)
        self._hasher = hasher
        self._tables = tables
        logger.info(
            "Engram host lookup ready: %d layers mapped from %s",
            len(self.source.layouts),
            checkpoint_dir,
        )
        return hasher, tables

    def hasher(self) -> NgramHasher:
        """Return the host-side n-gram hasher for input preparation."""
        return self._require_host_state()[0]

    def gather_partials(
        self,
        hash_ids: Mapping[int, torch.Tensor],
    ) -> dict[int, dict[int, EngramEmbedPartial]]:
        """Gather every engram layer's zero-padded partial for every TP rank.

        Args:
            hash_ids: per-layer ``[tokens, n_hash_cols]`` int64 global table
                row ids, as produced by :meth:`hasher`.

        Returns:
            ``{tp_rank: {layer_id: EngramEmbedPartial}}``. A physical rank
            uses the partial of ``tp_rank = rank % tp_size``; DP groups share
            the same set. Each partial carries the raw FP8 bytes and E8M0
            scales of its shard's rows and zeros elsewhere, ready to stage
            into a HOST_SHARED slot for the kernel's dequantize + all-reduce.
        """
        _, tables = self._require_host_state()
        unexpected = set(hash_ids) - set(self.source.layouts)
        if unexpected:
            raise KeyError(f"Hash ids for unknown engram layers: {sorted(unexpected)}")
        partials: dict[int, dict[int, EngramEmbedPartial]] = {
            tp_rank: {} for tp_rank in range(self.source.tp_size)
        }
        for layer_id, layout in self.source.layouts.items():
            rows = hash_ids.get(layer_id)
            if rows is None:
                raise KeyError(f"Missing hash ids for engram layer {layer_id}")
            for tp_rank in range(self.source.tp_size):
                partials[tp_rank][layer_id] = tables.gather_partial_rows(
                    layer_id, layout, rows, tp_rank=tp_rank
                )
        return partials
