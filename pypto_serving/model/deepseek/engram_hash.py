# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Host-side n-gram hashing for the engram tables.

Port of the reference ``NgramHashState`` (DeepSeek-V4.1-Flash
``inference/engram.py``) to the serving runner. The reference keeps every
hash constant as a non-persistent buffer, so nothing lives in the released
checkpoint and each constant must be rebuilt deterministically:

* bucket primes and offsets -- derived from ``engram_vocab_size`` exactly the
  way ``EngramLayout.from_args`` draws them (one prime per (n-gram size,
  head) pair, drawn in order and never reused, so bucket ranges stay
  disjoint);
* per-(layer, lookback) multipliers -- drawn from the per-layer RNG seeded
  with ``10007 * layer_id`` (numpy PCG64, whose stream is frozen across
  releases);
* the compressed token map -- every token id folded onto a smaller id space
  where tokens that normalize alike collapse together. It is rebuilt at
  load time from the checkpoint's raw Rust tokenizer, exactly what
  training decoded with.

The public entry point :meth:`NgramHasher.hash_tokens` hashes one request's
new tokens given the compressed ids of the tokens immediately before them
(the rolling history). Slots before the sequence start, and any dead token
(an image span), block the lookback and hash as the pad id, matching the
reference semantics bit for bit.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger(__name__)

__all__ = [
    "ENGRAM_HASH_DEAD",
    "EngramHashLayout",
    "NgramHasher",
    "build_compressed_token_map",
    "compute_hash_multipliers",
    "find_next_prime",
    "is_prime",
]

# Compressed-id sentinel for a slot that takes no part in any n-gram: sequence
# starts and image spans both leave DEAD entries in the rolling history.
ENGRAM_HASH_DEAD = -1

# Per-layer RNG seed base; layer ``L`` uses ``10007 * L`` (reference constant).
_ENGRAM_MULTIPLIER_SEED = 10007

# Miller-Rabin bases that make the test deterministic for every n < 3.3e24;
# the bucket primes sit around engam_vocab_size (~1.6e7 for V4.1).
_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def is_prime(value: int) -> bool:
    """Deterministic primality test agreeing with ``sympy.isprime``."""
    if value < 2:
        return False
    for small in _MR_BASES:
        if value % small == 0:
            return value == small
    exponent = 0
    remainder = value - 1
    while remainder % 2 == 0:
        remainder //= 2
        exponent += 1
    for base in _MR_BASES:
        witness = pow(base, remainder, value)
        if witness in (1, value - 1):
            continue
        for _ in range(exponent - 1):
            witness = witness * witness % value
            if witness == value - 1:
                break
        else:
            return False
    return True


def find_next_prime(start: int, seen_primes: set[int]) -> int:
    """The smallest prime above ``start`` that has not been handed out yet."""
    candidate = start + 1
    while not is_prime(candidate) or candidate in seen_primes:
        candidate += 1
    return candidate


@dataclass(frozen=True)
class EngramHashLayout:
    """Bucket geometry of the engram n-gram hash tables, per layer.

    A position is hashed as ``max_ngram_size - 1`` n-grams (2-gram ..
    max_ngram_size-gram), each split over ``n_heads`` heads. Every (n-gram
    size, head) pair owns a prime-sized bucket range in the layer's table;
    the primes are drawn in order and never reused, which keeps the ranges
    disjoint.
    """

    layer_ids: tuple[int, ...]
    num_embeddings: tuple[int, ...]
    max_ngram_size: int
    n_heads: int
    head_dim: int
    # [layer][n-gram size][head] bucket modulus, in draw order.
    primes: tuple[tuple[tuple[int, ...], ...], ...]
    # Size of the compressed tokenizer vocabulary; every hash multiplier is
    # derived from it, so a mismatch would silently rehash the whole table.
    compressed_vocab_size: int
    # Token that fills n-gram slots with no history; matches training.
    pad_token_id: int = 2

    def __post_init__(self) -> None:
        if self.max_ngram_size < 2:
            raise ValueError(
                f"max_ngram_size must be at least 2 (1-grams hash nothing), got {self.max_ngram_size}"
            )
        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {self.n_heads}")
        if len(self.layer_ids) != len(self.num_embeddings):
            raise ValueError(
                "layer_ids and num_embeddings must have the same length, "
                f"got {len(self.layer_ids)} and {len(self.num_embeddings)}"
            )
        if len(self.primes) != len(self.layer_ids):
            raise ValueError(
                "primes must be drawn per layer, got "
                f"{len(self.primes)} for {len(self.layer_ids)} layers"
            )

    @property
    def n_hash_cols(self) -> int:
        """Hash columns per layer: one bucket per (n-gram size, head) pair."""
        return (self.max_ngram_size - 1) * self.n_heads

    def offsets(self) -> torch.Tensor:
        """Per-layer bucket base addresses, ``[n_layers, n_hash_cols]`` int64."""
        rows = []
        for layer_primes in self.primes:
            flat = [prime for per_ngram in layer_primes for prime in per_ngram]
            base = 0
            row = []
            for prime in flat[:-1]:
                row.append(base)
                base += prime
            row.append(base)
            rows.append(row)
        return torch.tensor(rows, dtype=torch.int64)

    @classmethod
    def from_config(cls, config_data: Mapping[str, object]) -> EngramHashLayout:
        """Rebuild the bucket geometry from a V4.1 config, flat or nested.

        The released checkpoint nests the text-model keys under
        ``text_config``; converted variants may carry them at the top level.
        Both are accepted, with the nesting taking precedence.
        """
        nested = config_data.get("text_config")
        text_config: Mapping[str, object] = (
            nested if isinstance(nested, dict) else config_data
        )
        layer_ids = tuple(int(layer) for layer in text_config.get("engram_layer_ids", ()))
        if not layer_ids:
            raise ValueError("config declares no engram_layer_ids")
        num_embeddings = tuple(
            int(rows) for rows in text_config.get("engram_num_embeddings", ())
        )
        max_ngram_size = int(text_config["engram_max_ngram_size"])
        n_heads = int(text_config["engram_n_heads"])
        head_dim = int(text_config["engram_head_dim"])
        vocab_size = int(text_config["engram_vocab_size"])
        compressed_vocab_size = int(text_config["engram_compressed_vocab_size"])
        primes: list[tuple[tuple[int, ...], ...]] = []
        seen: set[int] = set()
        for _layer in layer_ids:
            per_ngram = []
            for _ in range(max_ngram_size - 1):
                sizes = []
                current = vocab_size - 1
                for _ in range(n_heads):
                    current = find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))
        for layer, (rows, layer_primes) in enumerate(zip(num_embeddings, primes, strict=True)):
            drawn = sum(prime for per_ngram in layer_primes for prime in per_ngram)
            if rows != drawn:
                raise ValueError(
                    f"engram layer {layer_ids[layer]} declares {rows} table rows but its "
                    f"rebuilt buckets sum to {drawn}; engram_vocab_size mismatch?"
                )
        return cls(
            layer_ids=layer_ids,
            num_embeddings=num_embeddings,
            max_ngram_size=max_ngram_size,
            n_heads=n_heads,
            head_dim=head_dim,
            primes=tuple(primes),
            compressed_vocab_size=compressed_vocab_size,
            pad_token_id=int(text_config.get("engram_pad_token_id", 2)),
        )


def compute_hash_multipliers(
    layer_ids: Sequence[int],
    max_ngram_size: int,
    compressed_vocab_size: int,
) -> torch.Tensor:
    """One odd multiplier per (layer, lookback), from a per-layer RNG.

    Kept odd, and bounded so that ``token_id * multiplier`` cannot overflow
    int64. Bit-for-bit port of the reference ``compute_hash_multipliers``.
    """
    max_long = np.iinfo(np.int64).max
    multiplier_bound = max(1, (max_long // compressed_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(_ENGRAM_MULTIPLIER_SEED * int(layer_id))
        values = generator.integers(
            low=0,
            high=multiplier_bound,
            size=(max_ngram_size,),
            dtype=np.int64,
        )
        rows.append(torch.tensor(values * 2 + 1))
    return torch.stack(rows)


def build_compressed_token_map(tokenizer) -> torch.Tensor:
    """Map every token id onto the compressed id space, matching training.

    Tokens that normalize alike collapse together, so " The", "the" and
    "THE" all hash the same way. Bit-for-bit port of the reference
    ``build_compressed_token_map`` (which additionally returns the compressed
    vocabulary size; the hasher re-derives that count from the map itself).
    ``tokenizer`` is the raw Rust backend (``tokenizers.Tokenizer``),
    exactly what training decoded with.
    """
    from tokenizers import Regex, normalizers

    # a private-use char, so a token that is exactly one space survives Strip() instead of
    # collapsing to the empty string and merging with unrelated tokens
    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    key_to_new: dict[str, int] = {}
    lookup = [0] * tokenizer.get_vocab_size()
    for token_id in range(tokenizer.get_vocab_size()):
        text = tokenizer.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            # a partial UTF-8 byte token: nothing to normalize, so key it by its raw form
            key = tokenizer.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return torch.tensor(lookup, dtype=torch.int64)


class NgramHasher:
    """Maps each position to the hash ids of the n-grams ending there.

    Ids go through the compressed table, then each position is hashed with
    the ``max_ngram_size - 1`` tokens before it. The look-back stops at the
    sequence start and at any dead token (an image span), so an n-gram never
    spans one; the rolling history carries all of this across the
    prefill/decode split.
    """

    def __init__(
        self,
        layout: EngramHashLayout,
        token_map: torch.Tensor,
        *,
        pad_token_id: int,
    ) -> None:
        if token_map.ndim != 1 or token_map.dtype != torch.int64:
            raise ValueError(
                f"token_map must be a 1-D int64 tensor, got {token_map.ndim}-D {token_map.dtype}"
            )
        largest = int(token_map.max().item()) if token_map.numel() else -1
        if largest + 1 != layout.compressed_vocab_size:
            raise ValueError(
                f"token map covers {largest + 1} compressed ids but the layout "
                f"declares {layout.compressed_vocab_size}; a tokenizer mismatch "
                "would silently rehash the whole table"
            )
        if not 0 <= pad_token_id < token_map.numel():
            raise ValueError(f"pad_token_id {pad_token_id} out of token map range")
        self.layout = layout
        self.token_map = token_map
        self.pad_id = int(token_map[pad_token_id].item())
        # [n_layers, max_ngram_size - 1, n_heads] bucket moduli.
        self.primes = torch.tensor(layout.primes, dtype=torch.int64)
        # [n_layers, n_hash_cols] bucket base addresses.
        self.offsets = layout.offsets()
        # [n_layers, max_ngram_size] per-lookback odd multipliers.
        self.multipliers = compute_hash_multipliers(
            layout.layer_ids, layout.max_ngram_size, layout.compressed_vocab_size
        )

    def initial_history(self) -> torch.Tensor:
        """Rolling history for a sequence that starts here: all DEAD."""
        return torch.full(
            (self.layout.max_ngram_size - 1,), ENGRAM_HASH_DEAD, dtype=torch.int64
        )

    def hash_tokens(
        self,
        input_ids: torch.Tensor,
        *,
        history: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Hash one request's new tokens.

        Args:
            input_ids: ``[new_len]`` int64 CPU tensor of this step's token ids
                (a prefill chunk or a decode window).
            history: ``[max_ngram_size - 1]`` int64 rolling history of the
                compressed ids immediately before ``input_ids`` (oldest
                first), as returned by a previous call. ``None`` means the
                sequence starts here.
            token_mask: ``[new_len]`` bool tensor, ``False`` for tokens that
                take no part in an n-gram (image spans).

        Returns:
            ``(hash_ids, history)`` where ``hash_ids`` is
            ``[new_len, n_layers, n_hash_cols]`` int64 global table row ids
            and ``history`` is the updated rolling tail for the next call.
        """
        if input_ids.ndim != 1:
            raise ValueError(f"input_ids must be 1-D, got {input_ids.ndim}-D")
        if input_ids.dtype != torch.int64:
            input_ids = input_ids.to(torch.int64)
        lookback = self.layout.max_ngram_size - 1
        if history is None:
            history = self.initial_history()
        elif history.shape != (lookback,) or history.dtype != torch.int64:
            raise ValueError(
                f"history must be int64 [{lookback}], got {history.dtype} {tuple(history.shape)}"
            )
        compressed = self.token_map[input_ids]
        if token_mask is not None:
            if token_mask.shape != input_ids.shape:
                raise ValueError(
                    f"token_mask shape {tuple(token_mask.shape)} does not match "
                    f"input_ids {tuple(input_ids.shape)}"
                )
            compressed = torch.where(token_mask, compressed, ENGRAM_HASH_DEAD)
        # Window sequence: the lookback slots plus the new compressed ids.
        # Slots before the sequence start are DEAD in the history, which
        # blocks the same lookbacks the reference blocks via `positions < shift`.
        window = torch.cat([history, compressed])
        new_len = input_ids.numel()
        positions = torch.arange(lookback, lookback + new_len, dtype=torch.int64)
        tokens: list[torch.Tensor] = []
        blocked = torch.zeros(new_len, dtype=torch.bool)
        for shift in range(self.layout.max_ngram_size):
            source = window[positions - shift]
            blocked = blocked | (source == ENGRAM_HASH_DEAD)
            tokens.append(torch.where(blocked, self.pad_id, source))
        tokens = torch.stack(tokens, dim=-1)  # [new_len, max_ngram_size]

        # XOR the multiplied ids together one lookback at a time, so the
        # running value after step i is the hash of the (i+1)-gram; each
        # lands in its own prime-sized bucket range.
        products = tokens.unsqueeze(1) * self.multipliers  # [new_len, L, max]
        rolling = products[..., 0]
        hashes = []
        for i in range(1, self.layout.max_ngram_size):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, i - 1])
        hash_ids = torch.cat(hashes, dim=-1) + self.offsets
        if new_len:
            new_history = window[-lookback:] if lookback else history
        else:
            new_history = history
        return hash_ids, new_history
