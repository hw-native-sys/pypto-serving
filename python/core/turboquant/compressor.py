"""MSE-optimal KV cache compressor using random rotation + Lloyd-Max quantization.

Adapted from TurboQuant V3 (MSE-only, no QJL).
Provides per-layer compressors with asymmetric key/value bit-widths.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .lloyd_max import LloydMaxCodebook


def _generate_rotation_matrix(d: int, seed: int, device: str = "cpu") -> torch.Tensor:
    """Generate a random orthogonal rotation matrix via QR decomposition."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    G = torch.randn(d, d, generator=gen)
    Q, R = torch.linalg.qr(G)
    diag_sign = torch.sign(torch.diag(R))
    diag_sign[diag_sign == 0] = 1.0
    Q = Q * diag_sign.unsqueeze(0)
    return Q.to(device)


class MSECompressor:
    """Single-stage MSE-optimal compressor for one side (keys or values).

    Compress normalizes to unit sphere, quantizes with Lloyd-Max,
    and stores bit-packed indices + norms.
    """

    def __init__(self, head_dim: int, bits: int, seed: int, device: str = "cpu"):
        self.head_dim = head_dim
        self.bits = bits
        self.device = device

        self.Pi = _generate_rotation_matrix(head_dim, seed=seed, device=device)
        self.centroids = LloydMaxCodebook(head_dim, bits).centroids.to(device)

    @torch.no_grad()
    def compress(self, states: torch.Tensor) -> dict:
        """Compress (B, H, S, D) -> dict with bit-packed indices + norms."""
        B, H, S, D = states.shape
        N = B * H * S
        flat = states.reshape(N, D).float()

        # Normalize to unit sphere, store norms
        vec_norms = torch.norm(flat, dim=-1)  # (N,)
        flat_norm = flat / (vec_norms.unsqueeze(-1) + 1e-8)

        # Rotate + quantize
        rotated = flat_norm @ self.Pi.T
        diffs = rotated.unsqueeze(-1) - self.centroids  # (N, D, levels)
        indices = diffs.abs().argmin(dim=-1).to(torch.uint8)  # (N, D)

        # Bit-pack indices
        indices_per_byte = 8 // self.bits
        idx_pad = (indices_per_byte - D % indices_per_byte) % indices_per_byte
        idx_flat = indices.long()
        if idx_pad:
            idx_flat = F.pad(idx_flat, (0, idx_pad))
        n_groups = idx_flat.shape[-1] // indices_per_byte
        idx_powers = torch.tensor(
            [2 ** (self.bits * i) for i in range(indices_per_byte - 1, -1, -1)],
            dtype=torch.long,
            device=idx_flat.device,
        )
        idx_bytes = (
            (idx_flat.reshape(N, n_groups, indices_per_byte) * idx_powers)
            .sum(-1)
            .to(torch.uint8)
        )

        return {
            "idx_bytes": idx_bytes.reshape(B, H, S, n_groups),
            "vec_norms": vec_norms.to(torch.float16).reshape(B, H, S),
            "shape": (B, H, S, D),
            "idx_pad": idx_pad,
        }

    @torch.no_grad()
    def decompress(self, compressed: dict) -> torch.Tensor:
        """Decompress back to (B, H, S, D) tensor."""
        B, H, S, D = compressed["shape"]
        N = B * H * S
        idx_bytes = compressed["idx_bytes"].reshape(N, -1)
        vec_norms = compressed["vec_norms"].reshape(N, 1).float()
        idx_pad = compressed["idx_pad"]

        # Unpack indices
        indices_per_byte = 8 // self.bits
        mask = (1 << self.bits) - 1
        idx_shifts = torch.tensor(
            [self.bits * i for i in range(indices_per_byte - 1, -1, -1)],
            dtype=torch.long,
            device=idx_bytes.device,
        )
        indices = (
            (idx_bytes.long().unsqueeze(-1) >> idx_shifts) & mask
        ).reshape(N, -1)
        if idx_pad:
            indices = indices[:, :D]

        # Reconstruct
        reconstructed = (self.centroids[indices] @ self.Pi) * vec_norms
        return reconstructed.reshape(B, H, S, D)


@dataclass
class KvQuantConfig:
    """Configuration for KV cache quantization."""

    enabled: bool = False
    key_bits: int = 4
    value_bits: int = 2
    residual_window: int = 128
    protected_layers: int = 4
    protected_bits: int = 8


class KVCompressor:
    """Per-layer KV cache compressor with asymmetric key/value bit-widths.

    Each layer gets its own compressor instance with a unique seed,
    enabling layer-adaptive precision (protected layers get more bits).
    """

    def __init__(
        self,
        head_dim: int,
        num_layers: int,
        config: KvQuantConfig,
        seed: int = 42,
        device: str = "cpu",
    ):
        self.head_dim = head_dim
        self.config = config

        self.key_compressors: list[MSECompressor] = []
        self.val_compressors: list[MSECompressor] = []

        for layer_idx in range(num_layers):
            is_protected = (
                layer_idx < config.protected_layers
                or layer_idx >= (num_layers - config.protected_layers)
            )
            effective_key_bits = config.protected_bits if is_protected else config.key_bits
            effective_val_bits = config.protected_bits if is_protected else config.value_bits
            effective_key_bits = min(effective_key_bits, 8)
            effective_val_bits = min(effective_val_bits, 8)

            seed_base = seed + layer_idx * 1000
            self.key_compressors.append(
                MSECompressor(head_dim, effective_key_bits, seed=seed_base, device=device)
            )
            self.val_compressors.append(
                MSECompressor(head_dim, effective_val_bits, seed=seed_base + 500, device=device)
            )

    @torch.no_grad()
    def compress_layer(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[dict, dict]:
        """Compress keys/values for one layer.

        Args:
            keys: (S, H, D) — old tokens to compress
            values: (S, H, D) — old tokens to compress

        Returns:
            (compressed_k, compressed_v) dicts
        """
        # Reshape (S, H, D) -> (1, H, S, D) for MSECompressor
        keys_4d = keys.permute(1, 0, 2).unsqueeze(0)
        values_4d = values.permute(1, 0, 2).unsqueeze(0)

        compressed_k = self.key_compressors[layer_idx].compress(keys_4d)
        compressed_v = self.val_compressors[layer_idx].compress(values_4d)
        return compressed_k, compressed_v

    @torch.no_grad()
    def decompress_layer(
        self,
        layer_idx: int,
        compressed_k: dict,
        compressed_v: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decompress keys/values for one layer.

        Returns:
            keys: (S, H, D), values: (S, H, D)
        """
        keys_4d = self.key_compressors[layer_idx].decompress(compressed_k)
        values_4d = self.val_compressors[layer_idx].decompress(compressed_v)

        # (1, H, S, D) -> (S, H, D)
        keys = keys_4d.squeeze(0).permute(1, 0, 2)
        values = values_4d.squeeze(0).permute(1, 0, 2)
        return keys, values
