# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the original-checkpoint engram embed loader."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest
import torch

from pypto_serving.model.deepseek.engram_loader import (
    ENGRAM_EMBED_BLOCK_SIZE,
    EngramEmbedCheckpoint,
    EngramEmbedLayout,
    engram_embed_tensor_names,
    load_engram_embed_shards,
    parse_engram_embed_layouts,
)


def _write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    """Write one minimal safetensors shard (header JSON + raw payloads)."""
    header: dict[str, dict[str, object]] = {}
    blob = bytearray()
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [len(blob), len(blob) + len(data)],
        }
        blob += data
    header_bytes = json.dumps(header).encode()
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        handle.write(blob)


def _build_checkpoint(
    root: Path,
    *,
    layer_id: int,
    num_embeddings: int,
    head_dim: int,
    weight_dtype: str = "F8_E4M3",
    scale_dtype: str = "F8_E8M0",
    omit_scale: bool = False,
    weight_shape: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize a synthetic V4.1 engram checkpoint and return its payloads."""
    scale_groups = head_dim // ENGRAM_EMBED_BLOCK_SIZE
    weight = torch.randint(0, 256, (num_embeddings, head_dim), dtype=torch.uint8)
    scale = torch.randint(0, 256, (num_embeddings, scale_groups), dtype=torch.uint8)
    tensors: dict[str, tuple[str, list[int], bytes]] = {
        "layers."
        f"{layer_id}.engram.embed.weight": (
            weight_dtype,
            list(weight_shape or (num_embeddings, head_dim)),
            weight.contiguous().numpy().tobytes(),
        )
    }
    if not omit_scale:
        tensors[f"layers.{layer_id}.engram.embed.scale"] = (
            scale_dtype,
            [num_embeddings, scale_groups],
            scale.contiguous().numpy().tobytes(),
        )
    _write_safetensors(root / "model-00001-of-00001.safetensors", tensors)
    index = {name: "model-00001-of-00001.safetensors" for name in tensors}
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": index})
    )
    return weight, scale


class TestEngramEmbedLayout:
    def test_row_ranges_even_shard(self):
        layout = EngramEmbedLayout(num_embeddings=100, head_dim=256, tp_size=4)
        assert layout.rows_per_rank == 25
        assert layout.scale_groups == 8
        assert layout.row_range(0) == (0, 25)
        assert layout.row_range(3) == (75, 100)

    def test_row_ranges_uneven_tail_zero_filled(self):
        layout = EngramEmbedLayout(num_embeddings=10, head_dim=64, tp_size=4)
        assert layout.rows_per_rank == 3
        assert layout.row_range(0) == (0, 3)
        assert layout.row_range(3) == (9, 10)

    def test_rejects_invalid_geometry(self):
        with pytest.raises(ValueError, match="head_dim"):
            EngramEmbedLayout(num_embeddings=10, head_dim=33, tp_size=4)
        with pytest.raises(ValueError, match="tp_size"):
            EngramEmbedLayout(num_embeddings=10, head_dim=64, tp_size=0)
        layout = EngramEmbedLayout(num_embeddings=10, head_dim=64, tp_size=4)
        with pytest.raises(ValueError, match="out of range"):
            layout.row_range(4)


class TestParseEngramEmbedLayouts:
    def test_nested_text_config(self):
        config = {"text_config": {"engram_layer_ids": [1, 14], "engram_num_embeddings": [40, 41],
                                  "engram_head_dim": 256}}
        layouts = parse_engram_embed_layouts(config, tp_size=4)
        assert set(layouts) == {1, 14}
        assert layouts[1].num_embeddings == 40
        assert layouts[14].rows_per_rank == 11

    def test_flat_config(self):
        config = {"engram_layer_ids": [1], "engram_num_embeddings": [8], "engram_head_dim": 64}
        layouts = parse_engram_embed_layouts(config, tp_size=2)
        assert set(layouts) == {1}
        assert layouts[1].rows_per_rank == 4

    def test_no_engram_keys_is_empty(self):
        assert parse_engram_embed_layouts({"text_config": {}}, tp_size=4) == {}

    def test_mismatched_lengths_raise(self):
        config = {"engram_layer_ids": [1, 14], "engram_num_embeddings": [40], "engram_head_dim": 256}
        with pytest.raises(ValueError, match="same length"):
            parse_engram_embed_layouts(config, tp_size=4)


class TestEngramEmbedCheckpoint:
    def test_shard_roundtrip_and_tail_zero_fill(self, tmp_path):
        num_embeddings, head_dim, tp_size = 10, 64, 4
        weight, scale = _build_checkpoint(
            tmp_path, layer_id=5, num_embeddings=num_embeddings, head_dim=head_dim
        )
        layouts = {5: EngramEmbedLayout(num_embeddings, head_dim, tp_size)}
        checkpoint = EngramEmbedCheckpoint(tmp_path)

        shards = load_engram_embed_shards(checkpoint, layouts, tp_rank=0)
        shard = shards[5]
        assert shard.weight.dtype is torch.float8_e4m3fn
        assert shard.scale.dtype is torch.float8_e8m0fnu
        assert tuple(shard.weight.shape) == (3, head_dim)
        assert tuple(shard.scale.shape) == (3, 2)
        assert torch.equal(shard.weight.view(torch.uint8), weight[0:3])
        assert torch.equal(shard.scale.view(torch.uint8), scale[0:3])

        tail = checkpoint.load_shard(5, layouts[5], tp_rank=3)
        assert tuple(tail.weight.shape) == (3, head_dim)
        assert torch.equal(tail.weight.view(torch.uint8)[0:1], weight[9:10])
        assert torch.equal(tail.weight.view(torch.uint8)[1:], torch.zeros(2, head_dim, dtype=torch.uint8))
        assert torch.equal(tail.scale.view(torch.uint8)[0:1], scale[9:10])

    def test_every_rank_covers_the_table_exactly_once(self, tmp_path):
        num_embeddings, head_dim, tp_size = 37, 128, 4
        weight, _ = _build_checkpoint(
            tmp_path, layer_id=1, num_embeddings=num_embeddings, head_dim=head_dim
        )
        layout = EngramEmbedLayout(num_embeddings, head_dim, tp_size)
        checkpoint = EngramEmbedCheckpoint(tmp_path)
        gathered = torch.cat(
            [
                checkpoint.load_shard(1, layout, tp_rank=rank).weight.view(torch.uint8)[
                    : layout.row_range(rank)[1] - layout.row_range(rank)[0]
                ]
                for rank in range(tp_size)
            ]
        )
        assert torch.equal(gathered, weight)

    def test_missing_scale_raises(self, tmp_path):
        _build_checkpoint(
            tmp_path, layer_id=1, num_embeddings=8, head_dim=64, omit_scale=True
        )
        layout = EngramEmbedLayout(8, 64, 2)
        checkpoint = EngramEmbedCheckpoint(tmp_path)
        with pytest.raises(KeyError, match="missing required tensors"):
            checkpoint.load_shard(1, layout, tp_rank=0)

    def test_wrong_weight_dtype_raises(self, tmp_path):
        weight, _ = _build_checkpoint(
            tmp_path,
            layer_id=1,
            num_embeddings=8,
            head_dim=64,
            weight_dtype="BF16",
        )
        assert weight.dtype == torch.uint8
        layout = EngramEmbedLayout(8, 64, 2)
        checkpoint = EngramEmbedCheckpoint(tmp_path)
        with pytest.raises(ValueError, match="F8_E4M3"):
            checkpoint.load_shard(1, layout, tp_rank=0)

    def test_wrong_shape_raises(self, tmp_path):
        _build_checkpoint(
            tmp_path,
            layer_id=1,
            num_embeddings=8,
            head_dim=64,
            weight_shape=(7, 64),
        )
        layout = EngramEmbedLayout(8, 64, 2)
        checkpoint = EngramEmbedCheckpoint(tmp_path)
        with pytest.raises(ValueError, match="shape"):
            checkpoint.load_shard(1, layout, tp_rank=0)

    def test_require_layers_reports_missing_names(self, tmp_path):
        _build_checkpoint(tmp_path, layer_id=1, num_embeddings=8, head_dim=64)
        checkpoint = EngramEmbedCheckpoint(tmp_path)
        layouts = {1: EngramEmbedLayout(8, 64, 2), 14: EngramEmbedLayout(9, 64, 2)}
        with pytest.raises(KeyError, match="layers.14.engram.embed.weight"):
            checkpoint.require_layers(layouts)

    def test_missing_index_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="index"):
            EngramEmbedCheckpoint(tmp_path)

    def test_tensor_names(self):
        assert engram_embed_tensor_names(14) == (
            "layers.14.engram.embed.weight",
            "layers.14.engram.embed.scale",
        )


class TestGatherPartialRows:
    """TP-sharded zero-padded gathers: the host half of the engram lookup."""

    def _gather_all_ranks(self, checkpoint, layout, layer_id, hash_ids, tp_size):
        return [
            checkpoint.gather_partial_rows(layer_id, layout, hash_ids, tp_rank=rank)
            for rank in range(tp_size)
        ]

    def test_partial_hits_own_shard_and_zeroes_rest(self, tmp_path):
        rows, head_dim, tp_size = 100, 64, 4
        weight, scale = _build_checkpoint(
            tmp_path, layer_id=1, num_embeddings=rows, head_dim=head_dim
        )
        checkpoint = EngramEmbedCheckpoint(tmp_path)
        layout = EngramEmbedLayout(rows, head_dim, tp_size)
        generator = torch.Generator().manual_seed(3)
        hash_ids = torch.randint(0, rows, (7, 5), dtype=torch.int64, generator=generator)

        partials = self._gather_all_ranks(checkpoint, layout, 1, hash_ids, tp_size)
        for rank, partial in enumerate(partials):
            start, end = layout.row_range(rank)
            hit = (hash_ids >= start) & (hash_ids < end)
            assert partial.weight.shape == (7, 5, head_dim)
            assert partial.scale.shape == (7, 5, head_dim // ENGRAM_EMBED_BLOCK_SIZE)
            weight_bytes = partial.weight.view(torch.uint8)
            scale_bytes = partial.scale.view(torch.uint8)
            # Hit columns carry the table bytes.
            assert torch.equal(weight_bytes[hit], weight[hash_ids[hit]])
            assert torch.equal(scale_bytes[hit], scale[hash_ids[hit]])
            # Missed columns stay zero.
            assert bool((weight_bytes[~hit] == 0).all())
            assert bool((scale_bytes[~hit] == 0).all())

    def test_partials_sum_to_full_gather(self, tmp_path):
        rows, head_dim, tp_size = 97, 64, 4  # uneven tail shard
        weight, scale = _build_checkpoint(
            tmp_path, layer_id=14, num_embeddings=rows, head_dim=head_dim
        )
        checkpoint = EngramEmbedCheckpoint(tmp_path)
        layout = EngramEmbedLayout(rows, head_dim, tp_size)
        generator = torch.Generator().manual_seed(5)
        hash_ids = torch.randint(0, rows, (11, 6), dtype=torch.int64, generator=generator)

        partials = self._gather_all_ranks(checkpoint, layout, 14, hash_ids, tp_size)
        weight_sum = torch.zeros((11, 6, head_dim), dtype=torch.uint8)
        scale_sum = torch.zeros(
            (11, 6, head_dim // ENGRAM_EMBED_BLOCK_SIZE), dtype=torch.uint8
        )
        for partial in partials:
            weight_sum += partial.weight.view(torch.uint8)
            scale_sum += partial.scale.view(torch.uint8)
        # Each column is non-zero on exactly one rank, so the byte sums equal
        # the direct full-table gather.
        assert torch.equal(weight_sum, weight[hash_ids])
        assert torch.equal(scale_sum, scale[hash_ids])

    def test_all_misses_yield_zero_partial(self, tmp_path):
        rows, head_dim = 100, 64
        _build_checkpoint(tmp_path, layer_id=1, num_embeddings=rows, head_dim=head_dim)
        checkpoint = EngramEmbedCheckpoint(tmp_path)
        layout = EngramEmbedLayout(rows, head_dim, 4)
        # Rank 0 owns [0, 25); ids beyond that never hit.
        hash_ids = torch.full((3, 4), 90, dtype=torch.int64)
        partial = checkpoint.gather_partial_rows(1, layout, hash_ids, tp_rank=0)
        assert bool((partial.weight.view(torch.uint8) == 0).all())
        assert bool((partial.scale.view(torch.uint8) == 0).all())

    def test_rejects_out_of_range_hash_ids(self, tmp_path):
        rows, head_dim = 100, 64
        _build_checkpoint(tmp_path, layer_id=1, num_embeddings=rows, head_dim=head_dim)
        checkpoint = EngramEmbedCheckpoint(tmp_path)
        layout = EngramEmbedLayout(rows, head_dim, 4)
        with pytest.raises(ValueError, match="outside"):
            checkpoint.gather_partial_rows(
                1, layout, torch.full((2, 2), rows, dtype=torch.int64), tp_rank=0
            )
        with pytest.raises(ValueError, match="2-D int64"):
            checkpoint.gather_partial_rows(
                1, layout, torch.zeros(4, dtype=torch.int64), tp_rank=0
            )

    def test_tail_rank_partial_covers_residual_rows(self, tmp_path):
        rows, head_dim, tp_size = 10, 64, 4  # rank 3 owns only row 9
        weight, scale = _build_checkpoint(
            tmp_path, layer_id=1, num_embeddings=rows, head_dim=head_dim
        )
        checkpoint = EngramEmbedCheckpoint(tmp_path)
        layout = EngramEmbedLayout(rows, head_dim, tp_size)
        hash_ids = torch.tensor([[9, 9], [0, 9]], dtype=torch.int64)
        partial = checkpoint.gather_partial_rows(1, layout, hash_ids, tp_rank=3)
        hit = torch.tensor([[True, True], [False, True]])
        assert torch.equal(partial.weight.view(torch.uint8)[hit], weight[hash_ids[hit]])
        assert torch.equal(partial.scale.view(torch.uint8)[hit], scale[hash_ids[hit]])
        assert bool((partial.weight.view(torch.uint8)[~hit] == 0).all())
