# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Engram embed table loading from the original DeepSeek-V4.1-Flash checkpoint.

The released V4.1 checkpoint stores each engram layer's hash table as an FP8
matrix plus per-row, group-of-32 E8M0 scales::

    layers.{L}.engram.embed.weight  [num_embeddings, head_dim]   F8_E4M3
    layers.{L}.engram.embed.scale   [num_embeddings, head_dim/32] F8_E8M0

One table is close to a hundred gigabytes, so it can never be materialized on
the host. This module maps the owning safetensors shard read-only and returns
**zero-copy row-range views** of exactly the rows one TP rank owns: rank ``r``
holds rows ``[r*rows_per_rank, (r+1)*rows_per_rank)`` with
``rows_per_rank = ceil(num_embeddings / tp_size)``, mirroring the reference
``ParallelEngramEmbedding`` sharding (the last rank's tail slots beyond the
table are zero-filled so every shard presents one uniform shape).

The read-only shared mapping is also what the device upload wants: the H2D DMA
reads the mapped pages directly, without breaking copy-on-write the way a
private ``safe_open`` mapping would.
"""

from __future__ import annotations

import json
import logging
import mmap
import os
import struct
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

# The E8M0 scale group along the head dimension (config ``weight_block_size``).
ENGRAM_EMBED_BLOCK_SIZE = 32
# safetensors header dtype strings the released checkpoint uses for the table.
ENGRAM_EMBED_WEIGHT_DTYPE = "F8_E4M3"
ENGRAM_EMBED_SCALE_DTYPE = "F8_E8M0"

_SAFETENSORS_TORCH_DTYPE = {
    ENGRAM_EMBED_WEIGHT_DTYPE: torch.float8_e4m3fn,
    ENGRAM_EMBED_SCALE_DTYPE: torch.float8_e8m0fnu,
    # The engram q/k gate weights ride along in the same shards as BF16.
    "BF16": torch.bfloat16,
}

_ENGRAM_MISSING_ERROR = "Engram embed checkpoint is missing required tensors: {names}"
_ENGRAM_DTYPE_ERROR = (
    "Engram embed tensor {name} must be safetensors {expected}, got {actual}"
)
_ENGRAM_SHAPE_ERROR = (
    "Engram embed tensor {name} must have shape {expected}, got {actual}"
)


@dataclass(frozen=True)
class EngramEmbedLayout:
    """Row-shard geometry of one engram embed table across the TP ranks."""

    num_embeddings: int
    head_dim: int
    tp_size: int

    def __post_init__(self) -> None:
        if self.num_embeddings <= 0:
            raise ValueError(f"num_embeddings must be positive, got {self.num_embeddings}")
        if self.head_dim <= 0 or self.head_dim % ENGRAM_EMBED_BLOCK_SIZE:
            raise ValueError(
                f"head_dim must be a positive multiple of {ENGRAM_EMBED_BLOCK_SIZE}, "
                f"got {self.head_dim}"
            )
        if self.tp_size <= 0:
            raise ValueError(f"tp_size must be positive, got {self.tp_size}")

    @property
    def rows_per_rank(self) -> int:
        """Uniform shard height; the last rank's tail may be zero-filled."""
        return -(-self.num_embeddings // self.tp_size)

    @property
    def scale_groups(self) -> int:
        return self.head_dim // ENGRAM_EMBED_BLOCK_SIZE

    def row_range(self, tp_rank: int) -> tuple[int, int]:
        """Return the ``[start, end)`` table rows rank ``tp_rank`` owns."""
        if not 0 <= tp_rank < self.tp_size:
            raise ValueError(f"tp_rank {tp_rank} out of range for TP={self.tp_size}")
        start = tp_rank * self.rows_per_rank
        return start, min(start + self.rows_per_rank, self.num_embeddings)


def engram_embed_tensor_names(layer_id: int) -> tuple[str, str]:
    """Return the checkpoint names of one layer's table payload and scales."""
    return (
        f"layers.{layer_id}.engram.embed.weight",
        f"layers.{layer_id}.engram.embed.scale",
    )


def parse_engram_embed_layouts(
    config_data: Mapping[str, object],
    *,
    tp_size: int,
) -> dict[int, EngramEmbedLayout]:
    """Read the engram embed geometry from a V4.1 config, flat or nested.

    The released checkpoint nests the text-model keys under ``text_config``;
    converted or trimmed variants may carry them at the top level. Both are
    accepted, with the nesting taking precedence. A config without engram keys
    yields an empty mapping rather than an error: the engram is optional.
    """
    nested = config_data.get("text_config")
    text_config: Mapping[str, object] = (
        nested if isinstance(nested, dict) else config_data
    )
    layer_ids = [int(layer) for layer in text_config.get("engram_layer_ids", ())]
    num_embeddings = [int(rows) for rows in text_config.get("engram_num_embeddings", ())]
    if not layer_ids:
        return {}
    if len(layer_ids) != len(num_embeddings):
        raise ValueError(
            "engram_layer_ids and engram_num_embeddings must have the same length, "
            f"got {len(layer_ids)} and {len(num_embeddings)}"
        )
    head_dim = int(text_config["engram_head_dim"])
    return {
        layer: EngramEmbedLayout(
            num_embeddings=rows,
            head_dim=head_dim,
            tp_size=tp_size,
        )
        for layer, rows in zip(layer_ids, num_embeddings, strict=True)
    }


@dataclass(frozen=True)
class EngramEmbedShard:
    """One TP rank's shard: FP8 payload rows plus matching E8M0 scale rows.

    Both tensors are ``[rows_per_rank, ...]`` contiguous CPU storage. The
    views keep the checkpoint's shared mapping alive through their numpy base
    chain, so the caller may hold them indefinitely without copying.
    """

    layer_id: int
    tp_rank: int
    layout: EngramEmbedLayout
    weight: torch.Tensor
    scale: torch.Tensor


@dataclass(frozen=True)
class EngramEmbedPartial:
    """One TP rank's zero-padded partial gather of one layer's table rows.

    ``weight`` is ``[tokens, n_hash_cols, head_dim]`` and ``scale`` is
    ``[tokens, n_hash_cols, scale_groups]``. Columns whose hash id falls in
    this rank's shard carry the table bytes; every other column stays zero.
    Dequantized and summed over ranks, the partials reproduce the full
    lookup, which is the host-side equivalent of the reference
    ``ParallelEngramEmbedding`` masked gather feeding its all-reduce.
    """

    layer_id: int
    tp_rank: int
    layout: EngramEmbedLayout
    weight: torch.Tensor
    scale: torch.Tensor


class _MappedShard:
    """One read-only shared mapping of a safetensors shard file."""

    def __init__(self, path: Path) -> None:
        fd = os.open(path, os.O_RDONLY)
        try:
            header_len = struct.unpack("<Q", os.pread(fd, 8, 0))[0]
            header = json.loads(os.pread(fd, header_len, 8))
            self._mapping = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
        finally:
            # Closing the descriptor does not unmap the file.
            os.close(fd)
        self.header = {
            name: spec
            for name, spec in header.items()
            if name != "__metadata__"
        }
        self.data_start = 8 + header_len

    def row_view(
        self,
        name: str,
        *,
        row_start: int,
        row_count: int,
        row_bytes: int,
        columns: int,
        dtype: str,
    ) -> torch.Tensor:
        """Return a zero-copy view of ``row_count`` rows starting at ``row_start``."""
        spec = self.header[name]
        if spec["dtype"] != dtype:
            raise ValueError(
                _ENGRAM_DTYPE_ERROR.format(name=name, expected=dtype, actual=spec["dtype"])
            )
        begin, end = spec["data_offsets"]
        offset = self.data_start + begin + row_start * row_bytes
        array = np.frombuffer(
            self._mapping, dtype=np.uint8, count=row_count * row_bytes, offset=offset
        )
        if array.nbytes > end - begin:
            raise ValueError(
                f"Engram embed tensor {name} row range [{row_start}, "
                f"{row_start + row_count}) exceeds its {end - begin} byte payload"
            )
        # ``np.frombuffer`` keeps the mapping alive through the array's base
        # chain, and so does the torch tensor built on it.
        with warnings.catch_warnings():
            # The mapping is read-only on purpose, so the tensor is non-writable.
            warnings.simplefilter("ignore")
            tensor = torch.from_numpy(array)
        return (
            tensor.view(_SAFETENSORS_TORCH_DTYPE[dtype]).reshape(row_count, columns)
        )


class EngramEmbedCheckpoint:
    """Name-addressed, mapping-cached reader over the original V4.1 checkpoint.

    One instance may serve every layer and every rank; shard files are mapped
    at most once and the mappings stay alive as long as any issued view is.
    """

    def __init__(self, checkpoint_dir: str | Path) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        index_path = self.checkpoint_dir / "model.safetensors.index.json"
        if not index_path.exists():
            raise FileNotFoundError(
                f"Engram checkpoint has no model.safetensors.index.json: {self.checkpoint_dir}"
            )
        index_data = json.loads(index_path.read_text())
        self.weight_map: dict[str, str] = dict(index_data.get("weight_map", {}))
        self._shards: dict[str, _MappedShard] = {}

    def _mapped_shard(self, name: str) -> _MappedShard:
        filename = self.weight_map.get(name)
        if filename is None:
            raise KeyError(_ENGRAM_MISSING_ERROR.format(names=name))
        shard = self._shards.get(filename)
        if shard is None:
            path = self.checkpoint_dir / filename
            if not path.exists():
                raise FileNotFoundError(
                    f"Engram checkpoint shard {filename} is missing: {path}"
                )
            shard = _MappedShard(path)
            self._shards[filename] = shard
        return shard

    def require_layers(self, layouts: Mapping[int, EngramEmbedLayout]) -> None:
        """Fail early when any table payload or scale tensor is absent."""
        required = [
            name
            for layer in layouts
            for name in engram_embed_tensor_names(layer)
        ]
        missing = [name for name in required if name not in self.weight_map]
        if missing:
            raise KeyError(_ENGRAM_MISSING_ERROR.format(names=", ".join(missing)))

    def _validate_spec(
        self,
        name: str,
        layout: EngramEmbedLayout,
        *,
        columns: int,
        dtype: str,
    ) -> None:
        spec = self._mapped_shard(name).header[name]
        if spec["dtype"] != dtype:
            raise ValueError(
                _ENGRAM_DTYPE_ERROR.format(name=name, expected=dtype, actual=spec["dtype"])
            )
        expected = (layout.num_embeddings, columns)
        actual = tuple(int(dim) for dim in spec["shape"])
        if actual != expected:
            raise ValueError(
                _ENGRAM_SHAPE_ERROR.format(name=name, expected=expected, actual=actual)
            )

    def load_shard(
        self,
        layer_id: int,
        layout: EngramEmbedLayout,
        *,
        tp_rank: int,
    ) -> EngramEmbedShard:
        """Load one rank's row shard of one layer's table, zero-copy where possible.

        The last rank of a non-evenly-sharded table owns fewer valid rows; its
        tail slots are zero-filled so the shard keeps the uniform
        ``rows_per_rank`` height the kernel banks expect.
        """
        weight_name, scale_name = engram_embed_tensor_names(layer_id)
        self._validate_spec(weight_name, layout, columns=layout.head_dim, dtype=ENGRAM_EMBED_WEIGHT_DTYPE)
        self._validate_spec(scale_name, layout, columns=layout.scale_groups, dtype=ENGRAM_EMBED_SCALE_DTYPE)

        start, end = layout.row_range(tp_rank)
        valid_rows = end - start
        weight_shard = self._mapped_shard(weight_name).row_view(
            weight_name,
            row_start=start,
            row_count=valid_rows,
            row_bytes=layout.head_dim,
            columns=layout.head_dim,
            dtype=ENGRAM_EMBED_WEIGHT_DTYPE,
        )
        scale_shard = self._mapped_shard(scale_name).row_view(
            scale_name,
            row_start=start,
            row_count=valid_rows,
            row_bytes=layout.scale_groups,
            columns=layout.scale_groups,
            dtype=ENGRAM_EMBED_SCALE_DTYPE,
        )
        if valid_rows == layout.rows_per_rank:
            return EngramEmbedShard(
                layer_id=layer_id,
                tp_rank=tp_rank,
                layout=layout,
                weight=weight_shard,
                scale=scale_shard,
            )
        padded_weight = torch.zeros(
            (layout.rows_per_rank, layout.head_dim), dtype=torch.float8_e4m3fn
        )
        padded_scale = torch.zeros(
            (layout.rows_per_rank, layout.scale_groups), dtype=torch.float8_e8m0fnu
        )
        padded_weight[:valid_rows].copy_(weight_shard)
        padded_scale[:valid_rows].copy_(scale_shard)
        return EngramEmbedShard(
            layer_id=layer_id,
            tp_rank=tp_rank,
            layout=layout,
            weight=padded_weight,
            scale=padded_scale,
        )


    def gather_partial_rows(
        self,
        layer_id: int,
        layout: EngramEmbedLayout,
        hash_ids: torch.Tensor,
        *,
        tp_rank: int,
    ) -> EngramEmbedPartial:
        """Gather one TP rank's share of a hash-id batch, zero-padded.

        Args:
            layer_id: engram layer whose table is read.
            layout: the layer's row-shard geometry.
            hash_ids: ``[tokens, n_hash_cols]`` int64 global table row ids
                (one per (n-gram size, head) bucket, as produced by the
                host-side hasher).
            tp_rank: rows inside ``layout.row_range(tp_rank)`` are fetched
                from the mapped table; all other columns stay zero.

        The output keeps the checkpoint's raw byte layout (FP8 payload plus
        E8M0 scales), so it can be staged straight into a HOST_SHARED slot
        and dequantized on the device after the all-reduce.
        """
        if hash_ids.ndim != 2 or hash_ids.dtype != torch.int64:
            raise ValueError(
                f"hash_ids must be 2-D int64, got {hash_ids.ndim}-D {hash_ids.dtype}"
            )
        tokens, columns = tuple(hash_ids.shape)
        if (hash_ids < 0).any() or (hash_ids >= layout.num_embeddings).any():
            raise ValueError(
                f"hash ids for layer {layer_id} fall outside "
                f"[0, {layout.num_embeddings})"
            )
        weight_name, scale_name = engram_embed_tensor_names(layer_id)
        self._validate_spec(
            weight_name, layout, columns=layout.head_dim, dtype=ENGRAM_EMBED_WEIGHT_DTYPE
        )
        self._validate_spec(
            scale_name, layout, columns=layout.scale_groups, dtype=ENGRAM_EMBED_SCALE_DTYPE
        )
        start, end = layout.row_range(tp_rank)
        valid_rows = end - start
        local = hash_ids - start
        hit = (local >= 0) & (local < valid_rows)
        # Work in the uint8 domain: FP8 dtypes have no index_put.
        weight_bytes = torch.zeros((tokens, columns, layout.head_dim), dtype=torch.uint8)
        scale_bytes = torch.zeros(
            (tokens, columns, layout.scale_groups), dtype=torch.uint8
        )
        if not hit.any():
            return EngramEmbedPartial(
                layer_id=layer_id,
                tp_rank=tp_rank,
                layout=layout,
                weight=weight_bytes.view(torch.float8_e4m3fn),
                scale=scale_bytes.view(torch.float8_e8m0fnu),
            )
        # Clamp so out-of-shard ids index a safe row; the hit mask keeps
        # their values out of the output.
        rows = local.clamp(0, valid_rows - 1)[hit]
        weight_shard = self._mapped_shard(weight_name).row_view(
            weight_name,
            row_start=start,
            row_count=valid_rows,
            row_bytes=layout.head_dim,
            columns=layout.head_dim,
            dtype=ENGRAM_EMBED_WEIGHT_DTYPE,
        )
        scale_shard = self._mapped_shard(scale_name).row_view(
            scale_name,
            row_start=start,
            row_count=valid_rows,
            row_bytes=layout.scale_groups,
            columns=layout.scale_groups,
            dtype=ENGRAM_EMBED_SCALE_DTYPE,
        )
        weight_bytes[hit] = weight_shard.view(torch.uint8)[rows]
        scale_bytes[hit] = scale_shard.view(torch.uint8)[rows]
        return EngramEmbedPartial(
            layer_id=layer_id,
            tp_rank=tp_rank,
            layout=layout,
            weight=weight_bytes.view(torch.float8_e4m3fn),
            scale=scale_bytes.view(torch.float8_e8m0fnu),
        )


def load_engram_embed_shards(
    checkpoint: EngramEmbedCheckpoint,
    layouts: Mapping[int, EngramEmbedLayout],
    *,
    tp_rank: int,
) -> dict[int, EngramEmbedShard]:
    """Load every engram layer's shard for one rank, keyed by layer id."""
    checkpoint.require_layers(layouts)
    return {
        layer: checkpoint.load_shard(layer, layout, tp_rank=tp_rank)
        for layer, layout in layouts.items()
    }
