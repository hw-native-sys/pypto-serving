# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host-side DeepSeek V4.1 Engram hashing and row lookup.

The text path follows the readable reference implementation shipped with the
official DeepSeek V4.1 model. Serving owns the host state and table reads;
pypto-lib is only used for the device Engram kernels and their ABI.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def find_next_prime(start: int, seen_primes: set[int]) -> int:
    """Return the first unused prime above ``start`` (official reference behavior)."""
    candidate = start + 1
    while not _is_prime(candidate) or candidate in seen_primes:
        candidate += 1
    return candidate


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Build the official normalized token map from the raw tokenizer backend."""
    from tokenizers import Regex, normalizers

    sentinel = "\ue000"
    normalizer = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.NFD(),
        normalizers.StripAccents(),
        normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
        normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(),
        normalizers.Replace(sentinel, " "),
    ])
    # ``PreTrainedTokenizerFast`` exposes the Rust tokenizer as
    # ``backend_tokenizer`` while the serving loader may pass that Rust
    # ``tokenizers.Tokenizer`` directly.  The official reference operates on
    # the backend methods, so accept both forms here.
    backend = getattr(tokenizer, "backend_tokenizer", tokenizer)
    if hasattr(tokenizer, "__len__"):
        vocab_size = len(tokenizer)
    elif hasattr(backend, "get_vocab_size"):
        vocab_size = backend.get_vocab_size(with_added_tokens=True)
    else:
        vocab_size = len(backend.get_vocab())
    key_to_new: dict[str, int] = {}
    lookup = [0] * vocab_size
    for token_id in range(vocab_size):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return lookup, len(key_to_new)


def compute_hash_multipliers(
    layer_ids: tuple[int, ...], max_ngram_size: int, tokenizer_vocab_size: int,
) -> torch.Tensor:
    """Build the official per-layer odd hash multipliers."""
    max_long = np.iinfo(np.int64).max
    multiplier_bound = max(1, (max_long // tokenizer_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(0, multiplier_bound, size=(max_ngram_size,), dtype=np.int64)
        rows.append(torch.tensor(values * 2 + 1))
    return torch.stack(rows)


@dataclass(frozen=True)
class EngramLayout:
    """Prime bucket and projection layout from ``text_config``."""

    max_ngram_size: int
    layer_ids: tuple[int, ...]
    num_embeddings: tuple[int, ...]
    primes: tuple[tuple[tuple[int, ...], ...], ...]
    n_heads: int
    head_dim: int

    @classmethod
    def from_config(cls, config: dict) -> "EngramLayout | None":
        layer_ids = tuple(config.get("engram_layer_ids", ()))
        if not layer_ids:
            return None
        max_ngram_size = config["engram_max_ngram_size"]
        n_heads = config["engram_n_heads"]
        vocab_size = config["engram_vocab_size"]
        primes, seen = [], set()
        for _ in layer_ids:
            per_ngram = []
            for _ in range(max_ngram_size - 1):
                current, sizes = vocab_size - 1, []
                for _ in range(n_heads):
                    current = find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))
        return cls(
            max_ngram_size=max_ngram_size,
            layer_ids=layer_ids,
            num_embeddings=tuple(config["engram_num_embeddings"]),
            primes=tuple(primes),
            n_heads=n_heads,
            head_dim=config["engram_head_dim"],
        )

    @property
    def hash_columns(self) -> int:
        return (self.max_ngram_size - 1) * self.n_heads

    @property
    def lookup_width(self) -> int:
        return self.hash_columns * self.head_dim


class HostEngram:
    """Official text-only n-gram hash state plus host safetensors lookup."""

    def __init__(self, checkpoint, tokenizer, config: dict):
        self.checkpoint = checkpoint
        self.layout = EngramLayout.from_config(config)
        if self.layout is None:
            raise ValueError("Engram configuration is missing")
        token_map, vocab_size = build_compressed_token_map(tokenizer)
        expected_vocab = config["engram_compressed_vocab_size"]
        if vocab_size != expected_vocab:
            raise ValueError(f"Engram compressed vocabulary has {vocab_size} entries, expected {expected_vocab}")
        pad_token_id = config.get("engram_pad_id", config.get("engram_pad_token_id", 2))
        self.token_map = torch.tensor(token_map, dtype=torch.int64)
        self.pad_id = token_map[pad_token_id]
        flat = [[prime for per_ngram in layer for prime in per_ngram] for layer in self.layout.primes]
        offsets = [np.cumsum([0, *sizes[:-1]]) for sizes in flat]
        self.primes = torch.tensor(self.layout.primes, dtype=torch.int64)
        self.offsets = torch.tensor(np.array(offsets), dtype=torch.int64)
        self.multipliers = compute_hash_multipliers(self.layout.layer_ids, self.layout.max_ngram_size, vocab_size)
        self.history: dict[object, torch.Tensor] = {}
        self.kv_out = (config["hc_mult"] + 1) * config["hidden_size"]

    def release_request(self, request_id) -> None:
        self.history.pop(request_id, None)

    def hash_ids(self, token_ids, start_pos: int = 0, *, request_id=0) -> torch.Tensor:
        ids = torch.as_tensor(token_ids, dtype=torch.int64, device="cpu")
        if ids.ndim != 1 or bool(((ids < 0) | (ids >= self.token_map.numel())).any()):
            raise ValueError("Engram requires a one-dimensional sequence of valid token IDs")
        history = self.history.get(request_id, torch.empty(0, dtype=torch.int64))
        if isinstance(start_pos, bool) or not isinstance(start_pos, int) or not 0 <= start_pos <= history.numel():
            raise ValueError("Engram continuation requires all preceding token history")
        compressed = torch.cat((history[:start_pos], self.token_map[ids]))
        positions = torch.arange(start_pos, compressed.numel(), dtype=torch.int64)
        lookback = []
        for shift in range(self.layout.max_ngram_size):
            source = (positions - shift).clamp_min(0)
            values = compressed[source]
            lookback.append(torch.where(positions >= shift, values, self.pad_id))
        tokens = torch.stack(lookback, dim=-1)
        products = tokens.unsqueeze(0) * self.multipliers.unsqueeze(1)
        rolling, hashes = products[..., 0], []
        for shift in range(1, self.layout.max_ngram_size):
            rolling = torch.bitwise_xor(rolling, products[..., shift])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, shift - 1].unsqueeze(1))
        self.history[request_id] = compressed
        return torch.cat(hashes, dim=-1) + self.offsets.unsqueeze(1)

    def lookup(self, token_ids, capacity: int, start_pos: int = 0, *, request_id=0) -> torch.Tensor:
        if len(token_ids) > capacity or capacity <= 0:
            raise ValueError("Engram rows must fit the current-token capacity")
        hashes = self.hash_ids(token_ids, start_pos, request_id=request_id)
        result = torch.zeros((len(self.layout.layer_ids), capacity, self.layout.lookup_width), dtype=torch.bfloat16)
        for slot, layer in enumerate(self.layout.layer_ids):
            prefix = f"layers.{layer}.engram.embed."
            rows = self.checkpoint.tensor_rows(prefix + "weight", hashes[slot])
            scales = self.checkpoint.tensor_rows(prefix + "scale", hashes[slot])
            if rows.dtype != torch.float8_e4m3fn or rows.shape[-1] != self.layout.head_dim:
                raise ValueError(f"unsupported native Engram table layout: {prefix}weight")
            if scales.dtype != torch.float8_e8m0fnu or scales.shape[-1] != self.layout.head_dim // 32:
                raise ValueError(f"unsupported native Engram table scales: {prefix}scale")
            scale_values = torch.exp2(scales.view(torch.uint8).float() - 127.0)
            values = rows.float() * scale_values.repeat_interleave(32, dim=-1)
            result[slot, :len(token_ids)] = values.flatten(1).to(torch.bfloat16)
        return result
