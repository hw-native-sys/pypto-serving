# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Bit-exact parity tests for the host-side engram n-gram hasher.

The reference implementation lives in the DeepSeek-V4.1-Flash inference
directory; its ``NgramHashState.__init__`` needs the training tokenizer, so
the reference instances here are built through ``object.__new__`` with the
buffers filled by hand. Everything the hasher derives (bucket primes,
offsets, multipliers, rolling-history semantics across chunked calls) must
match the reference bit for bit, because a single differing hash id reads a
wrong row of a hundred-gigabyte table.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

_REFERENCE_DIR = Path("/home/pyptouser/liuchao/source/DeepSeek-V4.1-Flash/inference")
if str(_REFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(_REFERENCE_DIR))

from engram import (  # noqa: E402  (reference module, outside the repo)
    EngramLayout,
    NgramHashState,
    compute_hash_multipliers as reference_multipliers,
)

from pypto_serving.model.deepseek.engram_hash import (  # noqa: E402
    ENGRAM_HASH_DEAD,
    EngramHashLayout,
    NgramHasher,
    build_compressed_token_map,
    compute_hash_multipliers,
)

_V41_CONFIG_PATH = Path("/home/pyptouser/liuchao/source/DeepSeek-V4.1-Flash/config.json")

# A small geometry so prime search stays fast; two layers exercise the
# per-layer RNG and disjoint prime draws.
_SMALL = SimpleNamespace(
    engram_layer_ids=(5, 9),
    # Placeholder; overwritten with the drawn bucket sums right below, since
    # the reference layout derives primes before reading the row counts.
    engram_num_embeddings=(1, 1),
    engram_max_ngram_size=3,
    engram_vocab_size=101,
    engram_n_heads=2,
    engram_head_dim=64,
)


def _fill_small_num_embeddings() -> None:
    layout = EngramLayout.from_args(_SMALL)
    _SMALL.engram_num_embeddings = tuple(
        sum(prime for per_ngram in layer for prime in per_ngram)
        for layer in layout.primes
    )


_fill_small_num_embeddings()


def _small_config() -> dict[str, object]:
    return {
        "text_config": {
            "engram_layer_ids": list(_SMALL.engram_layer_ids),
            "engram_num_embeddings": list(_SMALL.engram_num_embeddings),
            "engram_max_ngram_size": _SMALL.engram_max_ngram_size,
            "engram_vocab_size": _SMALL.engram_vocab_size,
            "engram_n_heads": _SMALL.engram_n_heads,
            "engram_head_dim": _SMALL.engram_head_dim,
            "engram_compressed_vocab_size": 50,
        }
    }


def _reference_state(
    layout: EngramLayout,
    token_map: torch.Tensor,
    pad_token_id: int,
    *,
    max_batch: int,
    max_seq: int,
) -> NgramHashState:
    state = object.__new__(NgramHashState)
    state.layout = layout
    state.pad_id = int(token_map[pad_token_id].item())
    state.primes = torch.tensor(layout.primes)
    flat = [[p for per_ngram in layer for p in per_ngram] for layer in layout.primes]
    offsets = [np.cumsum([0, *sizes[:-1]]) for sizes in flat]
    state.offsets = torch.tensor(np.array(offsets))
    state.multipliers = reference_multipliers(
        layout.layer_ids, layout.max_ngram_size, 50
    )
    state.token_map = token_map
    state.cache = torch.empty(max_batch, max_seq, dtype=torch.int64)
    return state


def _hasher(pad_token_id: int = 2) -> tuple[NgramHasher, torch.Tensor]:
    # Full-coverage map: the hasher asserts the compressed ids are exactly
    # the declared 50, so the synthetic map must hit every id.
    token_map = torch.arange(80, dtype=torch.int64) % 50
    layout = EngramHashLayout.from_config(_small_config())
    return NgramHasher(layout, token_map, pad_token_id=pad_token_id), token_map


def test_primes_and_offsets_match_reference() -> None:
    reference = EngramLayout.from_args(_SMALL)
    ours = EngramHashLayout.from_config(_small_config())
    assert ours.layer_ids == reference.layer_ids
    assert ours.num_embeddings == reference.num_embeddings
    assert ours.max_ngram_size == reference.max_ngram_size
    assert ours.n_hash_cols == (reference.max_ngram_size - 1) * reference.n_heads
    assert ours.primes == reference.primes
    assert torch.equal(ours.offsets(), _reference_state(
        reference, torch.zeros(80, dtype=torch.int64), 0, max_batch=1, max_seq=1
    ).offsets)


def test_multipliers_match_reference() -> None:
    ours = compute_hash_multipliers(_SMALL.engram_layer_ids, _SMALL.engram_max_ngram_size, 50)
    reference = reference_multipliers(_SMALL.engram_layer_ids, _SMALL.engram_max_ngram_size, 50)
    assert torch.equal(ours, reference)
    assert ours.dtype == torch.int64
    assert torch.equal(ours % 2, torch.ones_like(ours))


def test_hash_full_sequence_matches_reference() -> None:
    hasher, token_map = _hasher()
    reference = _reference_state(
        EngramLayout.from_args(_SMALL), token_map, 2, max_batch=1, max_seq=128
    )
    generator = torch.Generator().manual_seed(11)
    tokens = torch.randint(0, 80, (40,), dtype=torch.int64, generator=generator)
    expected = reference.forward(tokens[None], start_pos=0)[0]
    actual, _ = hasher.hash_tokens(tokens)
    assert actual.shape == (40, len(hasher.layout.layer_ids), hasher.layout.n_hash_cols)
    assert torch.equal(actual, expected)


def test_hash_chunked_matches_reference_and_full() -> None:
    hasher, token_map = _hasher()
    reference = _reference_state(
        EngramLayout.from_args(_SMALL), token_map, 2, max_batch=1, max_seq=128
    )
    generator = torch.Generator().manual_seed(13)
    tokens = torch.randint(0, 80, (23,), dtype=torch.int64, generator=generator)
    full, _ = hasher.hash_tokens(tokens)
    expected_full = reference.forward(tokens[None], start_pos=0)[0]
    assert torch.equal(full, expected_full)

    # Chunked: a prefill chunk longer than the lookback, then single-token
    # decode steps, then one chunk shorter than the lookback (sequence tail).
    chunks = [tokens[:7], tokens[7:8], tokens[8:9], tokens[9:22], tokens[22:23]]
    history: torch.Tensor | None = None
    pieces = []
    for chunk in chunks:
        ids, history = hasher.hash_tokens(chunk, history=history)
        pieces.append(ids)
    assert torch.equal(torch.cat(pieces), full)

    # The reference must agree on the same chunk boundaries too.
    start = 0
    reference.cache.fill_(0)
    for piece, chunk in zip(pieces, chunks):
        expected = reference.forward(chunk[None], start_pos=start)[0]
        assert torch.equal(piece, expected)
        start += chunk.numel()


def test_hash_from_first_token_matches_reference() -> None:
    # Chunks shorter than the lookback right from the sequence start: the
    # DEAD-filled initial history must block the same lookbacks the
    # reference blocks through `positions < shift`.
    hasher, token_map = _hasher()
    reference = _reference_state(
        EngramLayout.from_args(_SMALL), token_map, 2, max_batch=1, max_seq=64
    )
    generator = torch.Generator().manual_seed(17)
    tokens = torch.randint(0, 80, (9,), dtype=torch.int64, generator=generator)
    full, _ = hasher.hash_tokens(tokens)
    expected = reference.forward(tokens[None], start_pos=0)[0]
    assert torch.equal(full, expected)

    history: torch.Tensor | None = None
    pieces = []
    consumed = 0
    for width in (1, 1, 1, 6):
        chunk = tokens[consumed : consumed + width]
        ids, history = hasher.hash_tokens(chunk, history=history)
        pieces.append(ids)
        consumed += width
    assert torch.equal(torch.cat(pieces), full)


def test_dead_mask_matches_reference() -> None:
    hasher, token_map = _hasher()
    reference = _reference_state(
        EngramLayout.from_args(_SMALL), token_map, 2, max_batch=1, max_seq=128
    )
    generator = torch.Generator().manual_seed(19)
    tokens = torch.randint(0, 80, (32,), dtype=torch.int64, generator=generator)
    mask = torch.ones(32, dtype=torch.bool)
    mask[5:9] = False  # an image span
    mask[20] = False
    expected = reference.forward(tokens[None], start_pos=0, token_mask=mask[None])[0]
    actual, _ = hasher.hash_tokens(tokens, token_mask=mask)
    assert torch.equal(actual, expected)

    # A dead token keeps blocking the following positions across the chunk
    # boundary: continue hashing with no mask and compare against the
    # reference run over the same concatenated sequence.
    more = torch.randint(0, 80, (4,), dtype=torch.int64, generator=generator)
    _, history = hasher.hash_tokens(tokens, token_mask=mask)
    tail, _ = hasher.hash_tokens(more, history=history)
    expected_tail = reference.forward(more[None], start_pos=32)[0]
    assert torch.equal(tail, expected_tail)


def test_real_v41_config_layout() -> None:
    import json

    config = json.loads(_V41_CONFIG_PATH.read_text())
    layout = EngramHashLayout.from_config(config)
    assert layout.layer_ids == (1, 14)
    assert layout.max_ngram_size == 4
    assert layout.n_heads == 8
    assert layout.head_dim == 256
    assert layout.n_hash_cols == 24
    # from_config itself asserts sum(primes) == num_embeddings per layer.
    assert layout.num_embeddings == (384006168, 384016682)
    multipliers = compute_hash_multipliers(layout.layer_ids, 4, layout.compressed_vocab_size)
    assert multipliers.shape == (2, 4)
    assert torch.equal(multipliers % 2, torch.ones_like(multipliers))
    hasher = NgramHasher(
        layout,
        torch.arange(129280, dtype=torch.int64) % layout.compressed_vocab_size,
        pad_token_id=2,
    )
    assert hasher.primes.shape == (2, 3, 8)
    assert hasher.offsets.shape == (2, 24)
    ids, history = hasher.hash_tokens(torch.tensor([0, 1, 2, 3]))
    assert ids.shape == (4, 2, 24)
    assert history.shape == (3,)
    assert bool((ids >= 0).all()) and bool((ids < 384006168).all())


def test_hasher_validation() -> None:
    layout = EngramHashLayout.from_config(_small_config())
    token_map = torch.arange(80, dtype=torch.int64) % 50
    with pytest.raises(ValueError, match="1-D int64"):
        NgramHasher(layout, token_map[None, :], pad_token_id=2)
    # Covers 100 ids vs the declared 50.
    with pytest.raises(ValueError, match="compressed id"):
        NgramHasher(layout, torch.full((80,), 99, dtype=torch.int64), pad_token_id=2)
    # Covers 1 id vs the declared 50.
    with pytest.raises(ValueError, match="compressed id"):
        NgramHasher(layout, torch.zeros(80, dtype=torch.int64), pad_token_id=2)
    with pytest.raises(ValueError, match="pad_token_id"):
        NgramHasher(layout, token_map, pad_token_id=80)
    hasher = NgramHasher(layout, token_map, pad_token_id=2)
    with pytest.raises(ValueError, match="input_ids must be 1-D"):
        hasher.hash_tokens(torch.zeros(2, 2, dtype=torch.int64))
    with pytest.raises(ValueError, match="history"):
        hasher.hash_tokens(torch.zeros(3, dtype=torch.int64), history=torch.zeros(4, dtype=torch.int64))
    with pytest.raises(ValueError, match="token_mask shape"):
        hasher.hash_tokens(torch.zeros(3, dtype=torch.int64), token_mask=torch.ones(4, dtype=torch.bool))
    empty_ids, history = hasher.hash_tokens(torch.zeros(0, dtype=torch.int64))
    assert empty_ids.shape == (0, 2, 4)
    assert torch.equal(history, hasher.initial_history())


def test_build_compressed_token_map_needs_tokenizers() -> None:
    import importlib.util

    if importlib.util.find_spec("tokenizers") is not None:
        pytest.skip("tokenizers is installed; the lazy import path is not exercised")
    with pytest.raises(ImportError):
        build_compressed_token_map(object())


def test_dead_sentinel_is_negative() -> None:
    # The rolling history stores compressed ids next to DEAD, so the sentinel
    # must never collide with a real compressed id (they are non-negative).
    assert ENGRAM_HASH_DEAD < 0
