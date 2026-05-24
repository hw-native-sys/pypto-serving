#!/usr/bin/env python3
"""Compare FP and TurboQuant KV cache + logits dumps from the NPU runner.

Loads the quantized KV cache, dequantizes it using the saved codebook and
rotation matrices, and compares against the FP KV cache.

Usage:
    python compare_npu_dump.py /tmp/dump_fp /tmp/dump_tq
    python compare_npu_dump.py /tmp/dump_fp /tmp/dump_tq --phase decode --step 3
    python compare_npu_dump.py /tmp/dump_fp /tmp/dump_tq --layer 0
    python compare_npu_dump.py /tmp/dump_fp /tmp/dump_tq --all-steps
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Dequantization logic
# ---------------------------------------------------------------------------

def generate_rotation_matrix(d: int, seed: int) -> np.ndarray:
    """Generate orthogonal rotation matrix matching TurboQuant compressor."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    G = torch.randn(d, d, generator=gen)
    Q, R = torch.linalg.qr(G)
    diag_sign = torch.sign(torch.diag(R))
    diag_sign[diag_sign == 0] = 1.0
    return (Q * diag_sign.unsqueeze(0)).numpy().astype(np.float32)


# Fallback codebook: Lloyd-Max centroids for HEAD_DIM=128, N_LEVELS=16 (int4).
# Used only when codebook.npy is not present in the dump directory.
_FALLBACK_CODEBOOK = np.array([
    -0.24156573, -0.18291856, -0.14305916, -0.11107627,
    -0.08332659, -0.05807660, -0.03431584, -0.01135439,
     0.01135439,  0.03431582,  0.05807661,  0.08332659,
     0.11107642,  0.14305905,  0.18291837,  0.24156609,
], dtype=np.float32)

_FALLBACK_HEAD_DIM = 128
_FALLBACK_SEED_BASE = 42


def load_codebook(tq_dir: Path) -> np.ndarray:
    """Load codebook from dump, or fall back to hardcoded values."""
    cb_path = tq_dir / "codebook.npy"
    if cb_path.exists():
        cb = np.load(cb_path)
        # codebook may be [1, N] or [N]; flatten to 1-D.
        return cb.ravel()
    return _FALLBACK_CODEBOOK


def load_rot_matrices(tq_dir: Path, num_layers: int) -> np.ndarray:
    """Load rotation matrices from dump, or regenerate from known seeds.

    Returns shape [num_layers, head_dim, head_dim].
    """
    rot_path = tq_dir / "rot_matrices.npy"
    if rot_path.exists():
        return np.load(rot_path).astype(np.float32)
    head_dim = _FALLBACK_HEAD_DIM
    return np.stack([
        generate_rotation_matrix(head_dim, seed=_FALLBACK_SEED_BASE + l * 1000)
        for l in range(num_layers)
    ])


def dequantize_kv(
    quant_indices: np.ndarray,
    quant_norms: np.ndarray,
    codebook: np.ndarray,
    rot_matrix: np.ndarray,
) -> np.ndarray:
    """Dequantize one layer's KV cache.

    Args:
        quant_indices: [kv_heads, page_size, head_dim] uint8
        quant_norms:   [kv_heads, page_size, 1]       float32
        codebook:      [N_LEVELS]                      float32
        rot_matrix:    [head_dim, head_dim]            float32

    Returns:
        [kv_heads, page_size, head_dim] float32
    """
    # Codebook lookup → rotated normalized approximation
    dequant = codebook[quant_indices]      # [kv_heads, page_size, head_dim]
    # Inverse rotation
    dequant = dequant @ rot_matrix.T       # [kv_heads, page_size, head_dim]
    # Rescale by L2 norm
    dequant = dequant * quant_norms        # [kv_heads, page_size, head_dim]
    return dequant


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(fp: np.ndarray, dq: np.ndarray) -> dict:
    """Compute comparison metrics between FP and dequantized tensors."""
    diff = fp - dq
    abs_diff = np.abs(diff)
    cos_sim = float(
        np.sum(fp * dq) / (np.linalg.norm(fp) * np.linalg.norm(dq) + 1e-12)
    )
    return {
        "cosine": cos_sim,
        "mse": float((diff ** 2).mean()),
        "mae": float(abs_diff.mean()),
        "max_err": float(abs_diff.max()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compare_step(
    fp_dir: Path,
    tq_dir: Path,
    codebook: np.ndarray,
    rot_matrices: np.ndarray,
    phase: str,
    step: int,
    layer_filter: int | None = None,
) -> None:
    """Compare one prefill/decode step."""
    tag = f"{phase}_step{step:04d}"

    # ---- Load FP KV cache ----
    fp_k_path = fp_dir / f"{tag}_k_cache.npy"
    fp_v_path = fp_dir / f"{tag}_v_cache.npy"
    if not fp_k_path.exists():
        print(f"  [skip] {tag}: FP k_cache not found")
        return
    fp_k = np.load(fp_k_path)   # [layers, kv_heads, page_size, head_dim]
    fp_v = np.load(fp_v_path)

    # ---- Load TQ KV cache ----
    tq_qk_path = tq_dir / f"{tag}_quant_k.npy"
    tq_qv_path = tq_dir / f"{tag}_quant_v.npy"
    tq_kn_path = tq_dir / f"{tag}_quant_k_norm.npy"
    tq_vn_path = tq_dir / f"{tag}_quant_v_norm.npy"
    if not tq_qk_path.exists():
        print(f"  [skip] {tag}: TQ quant_k not found")
        return
    tq_qk = np.load(tq_qk_path)
    tq_qv = np.load(tq_qv_path)
    tq_kn = np.load(tq_kn_path)
    tq_vn = np.load(tq_vn_path)

    num_layers = fp_k.shape[0]
    layers = range(num_layers) if layer_filter is None else [layer_filter]

    print(f"\n{'=' * 90}")
    print(f"  {tag}  |  {num_layers} layers  |  KV shape per layer: {fp_k.shape[1:]}")
    print(f"{'=' * 90}")
    print(
        f"  {'Layer':>6s}  │  {'K cos':>8s}  {'K MSE':>10s}  {'K MAE':>10s}  {'K max':>10s}"
        f"  │  {'V cos':>8s}  {'V MSE':>10s}  {'V MAE':>10s}  {'V max':>10s}"
    )
    print(f"  {'─' * 88}")

    all_k_diff = []
    all_v_diff = []

    for l in layers:
        k_dq = dequantize_kv(tq_qk[l], tq_kn[l], codebook, rot_matrices[l])
        v_dq = dequantize_kv(tq_qv[l], tq_vn[l], codebook, rot_matrices[l])

        km = compute_metrics(fp_k[l], k_dq)
        vm = compute_metrics(fp_v[l], v_dq)

        all_k_diff.append(fp_k[l] - k_dq)
        all_v_diff.append(fp_v[l] - v_dq)

        flag = " ⚠" if km["cosine"] < 0.99 or vm["cosine"] < 0.99 else ""
        print(
            f"  {l:>6d}  │  {km['cosine']:>8.6f}  {km['mse']:>10.2e}  {km['mae']:>10.2e}  {km['max_err']:>10.2e}"
            f"  │  {vm['cosine']:>8.6f}  {vm['mse']:>10.2e}  {vm['mae']:>10.2e}  {vm['max_err']:>10.2e}"
            f"{flag}"
        )

    # ---- Overall ----
    all_k_dq = np.concatenate([
        dequantize_kv(tq_qk[l], tq_kn[l], codebook, rot_matrices[l]).ravel()
        for l in layers
    ])
    all_v_dq = np.concatenate([
        dequantize_kv(tq_qv[l], tq_vn[l], codebook, rot_matrices[l]).ravel()
        for l in layers
    ])
    all_k_fp = np.concatenate([fp_k[l].ravel() for l in layers])
    all_v_fp = np.concatenate([fp_v[l].ravel() for l in layers])

    km_all = compute_metrics(all_k_fp, all_k_dq)
    vm_all = compute_metrics(all_v_fp, all_v_dq)

    print(f"  {'─' * 88}")
    print(
        f"  {'total':>6s}  │  {km_all['cosine']:>8.6f}  {km_all['mse']:>10.2e}  {km_all['mae']:>10.2e}  {km_all['max_err']:>10.2e}"
        f"  │  {vm_all['cosine']:>8.6f}  {vm_all['mse']:>10.2e}  {vm_all['mae']:>10.2e}  {vm_all['max_err']:>10.2e}"
    )

    # ---- Logits comparison (if available) ----
    fp_logits_path = fp_dir / f"{tag}_logits.npy"
    tq_logits_path = tq_dir / f"{tag}_logits.npy"
    if fp_logits_path.exists() and tq_logits_path.exists():
        fp_logits = np.load(fp_logits_path)
        tq_logits = np.load(tq_logits_path)
        ld = np.abs(fp_logits - tq_logits)
        fp_top1 = np.argmax(fp_logits, axis=-1)
        tq_top1 = np.argmax(tq_logits, axis=-1)
        match = "✓" if np.array_equal(fp_top1, tq_top1) else "✗"
        print(f"\n  Logits: max_diff={ld.max():.6f}  mean_diff={ld.mean():.6f}"
              f"  top-1 {match} (fp={fp_top1.tolist()} tq={tq_top1.tolist()})")

    # ---- Save dequantized KV for further analysis ----
    dq_k = np.stack([
        dequantize_kv(tq_qk[l], tq_kn[l], codebook, rot_matrices[l])
        for l in layers
    ])
    dq_v = np.stack([
        dequantize_kv(tq_qv[l], tq_vn[l], codebook, rot_matrices[l])
        for l in layers
    ])
    out_dir = tq_dir / "dequantized"
    out_dir.mkdir(exist_ok=True)
    np.save(out_dir / f"{tag}_k_dequant.npy", dq_k)
    np.save(out_dir / f"{tag}_v_dequant.npy", dq_v)
    print(f"\n  Dequantized KV saved to {out_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare FP and TurboQuant KV cache + logits dumps from the NPU runner.",
    )
    parser.add_argument("fp_dir", help="FP dump directory")
    parser.add_argument("tq_dir", help="TQ dump directory")
    parser.add_argument("--phase", default="prefill", choices=["prefill", "decode"])
    parser.add_argument("--step", type=int, default=0, help="Decode step index (default: 0)")
    parser.add_argument("--layer", type=int, default=None, help="Only compare one layer")
    parser.add_argument(
        "--all-steps", action="store_true",
        help="Compare all available steps (auto-detect from files).",
    )
    args = parser.parse_args()

    fp_dir = Path(args.fp_dir)
    tq_dir = Path(args.tq_dir)

    if not fp_dir.is_dir():
        print(f"Error: FP dump directory not found: {fp_dir}", file=sys.stderr)
        sys.exit(1)
    if not tq_dir.is_dir():
        print(f"Error: TQ dump directory not found: {tq_dir}", file=sys.stderr)
        sys.exit(1)

    codebook = load_codebook(tq_dir)
    print(f"Codebook: {len(codebook)} levels, range [{codebook.min():.6f}, {codebook.max():.6f}]")

    # Determine num_layers from FP dump to load/generate rotation matrices.
    fp_sample = sorted(fp_dir.glob("prefill_step0000_k_cache.npy"))
    if fp_sample:
        num_layers = np.load(fp_sample[0]).shape[0]
    else:
        num_layers = 40
        print(f"Warning: could not detect num_layers, assuming {num_layers}")

    rot_matrices = load_rot_matrices(tq_dir, num_layers)
    print(f"Rotation matrices: {rot_matrices.shape}")

    if args.all_steps:
        # Auto-detect all steps from filenames.
        steps = set()
        phase = args.phase
        for p in sorted(fp_dir.glob(f"{phase}_step*_k_cache.npy")):
            name = p.stem  # e.g. "prefill_step0000_k_cache"
            step_str = name.split("_step")[1].split("_")[0]
            steps.add(int(step_str))
        for step in sorted(steps):
            compare_step(fp_dir, tq_dir, codebook, rot_matrices, phase, step, args.layer)
    else:
        compare_step(fp_dir, tq_dir, codebook, rot_matrices, args.phase, args.step, args.layer)


if __name__ == "__main__":
    main()
