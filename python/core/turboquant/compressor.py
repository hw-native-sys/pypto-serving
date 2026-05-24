"""TurboQuant KV cache compressor: rotation matrix generation + bit-packing utilities.

The actual KV quantization is performed by NPU kernels (prefill_tq / decode_tq)
which write directly to compressed format. This module provides:
  - Rotation matrix generation (seeded per-layer)
  - Bit-packing / bit-unpacking for sub-8-bit storage
  - KVCompressor for layer-adaptive precision configuration and rotation matrix export
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

from ..types import KvQuantConfig
from .lloyd_max import LloydMaxCodebook


def generate_rotation_matrix(d: int, seed: int = 42, device: str = "cpu") -> torch.Tensor:
    """Generate a random orthogonal rotation matrix via QR decomposition."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    G = torch.randn(d, d, generator=gen)
    Q, R = torch.linalg.qr(G)
    diag_sign = torch.sign(torch.diag(R))
    diag_sign[diag_sign == 0] = 1.0
    return (Q * diag_sign.unsqueeze(0)).to(device)


class TurboQuantCompressor:
    """Per-vector TurboQuant compressor configuration.

    Stores the rotation matrix and quantization parameters for one layer.
    The actual compress/decompress is done by NPU kernels; this class
    provides bit-packing utilities for host-side storage.
    """

    def __init__(self, head_dim: int, bits: int = 4, seed: int = 42, device: str = "cpu"):
        self.head_dim = head_dim
        self.bits = min(bits, 8)
        self.device = device
        self.n_levels = 2 ** self.bits

        # Rotation matrix (fixed at init).
        self.Pi = generate_rotation_matrix(head_dim, seed=seed, device=device)
        self.PiT = self.Pi.T.contiguous()

        # Quantization range for normalized vectors (~N(0, 1/d)).
        sigma = 1.0 / math.sqrt(head_dim)
        self.lo = -3.5 * sigma
        self.hi = 3.5 * sigma

        # Uniform centroids (matching NPU kernel's uniform quantization).
        self.uniform_centroids = torch.linspace(self.lo, self.hi, self.n_levels, dtype=torch.float32)

    def _bit_pack(self, indices: torch.Tensor, N: int, D: int) -> tuple[torch.Tensor, int]:
        """Pack UINT8 indices into bit-packed bytes for sub-8-bit."""
        if self.bits >= 8:
            return indices.reshape(N, D), 0

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
        return idx_bytes, idx_pad

    def _bit_unpack(self, idx_bytes: torch.Tensor, N: int, D: int, idx_pad: int) -> torch.Tensor:
        """Unpack bit-packed bytes back to UINT8 indices."""
        if self.bits >= 8:
            return idx_bytes.reshape(N, D)

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
        return indices.to(torch.uint8)


class KVCompressor:
    """Per-layer KV cache compressor configuration.

    Each layer gets its own TurboQuantCompressor with a unique rotation
    matrix and configurable bit width, enabling layer-adaptive precision.
    The NPU kernels (prefill_tq / decode_tq) handle all compression/decompression.
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

        self.key_compressors: list[TurboQuantCompressor] = []
        self.val_compressors: list[TurboQuantCompressor] = []

        print(f"[TQ] Initializing: {num_layers} layers, head_dim={head_dim}, "
              f"key_bits={config.key_bits}, val_bits={config.value_bits}", flush=True)
        t_total = time.perf_counter()
        for layer_idx in range(num_layers):
            is_protected = (
                layer_idx < config.protected_layers
                or layer_idx >= (num_layers - config.protected_layers)
            )
            effective_key_bits = config.protected_bits if is_protected else config.key_bits
            effective_val_bits = config.protected_bits if is_protected else config.value_bits
            effective_key_bits = min(effective_key_bits, 8)
            effective_val_bits = min(effective_val_bits, 8)

            self.key_compressors.append(
                TurboQuantCompressor(head_dim, effective_key_bits,
                                     seed=seed + layer_idx * 1000, device=device)
            )
            self.val_compressors.append(
                TurboQuantCompressor(head_dim, effective_val_bits,
                                     seed=seed + layer_idx * 1000, device=device)
            )
            print(f"[TQ]   layer {layer_idx}: key_bits={effective_key_bits}, "
                  f"val_bits={effective_val_bits}", flush=True)
        dt_total = (time.perf_counter() - t_total) * 1000
        print(f"[TQ] Initialization complete: {dt_total:.1f} ms", flush=True)

    def get_rot_matrices(self, device: str = "cpu") -> torch.Tensor:
        """Stack all per-layer rotation matrices for NPU upload.

        Returns: (num_layers, head_dim, head_dim) BF16 tensor.
        """
        return torch.stack([c.Pi for c in self.key_compressors]).bfloat16().to(device)
