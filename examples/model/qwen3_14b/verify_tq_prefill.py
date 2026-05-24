#!/usr/bin/env python3
"""Verify TQ prefill: print NPU vs CPU scales and indices."""
import argparse
from pathlib import Path

import numpy as np
import torch


def generate_rotation_matrix(d: int, seed: int) -> np.ndarray:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    G = torch.randn(d, d, generator=gen)
    Q, R = torch.linalg.qr(G)
    diag_sign = torch.sign(torch.diag(R))
    diag_sign[diag_sign == 0] = 1.0
    return (Q * diag_sign.unsqueeze(0)).numpy().astype(np.float32)


def load_codebook(tq_dir: Path) -> np.ndarray:
    cb_path = tq_dir / "codebook.npy"
    if cb_path.exists():
        return np.load(cb_path).ravel()
    return np.array([
        -0.24156573, -0.18291856, -0.14305916, -0.11107627,
        -0.08332659, -0.05807660, -0.03431584, -0.01135439,
         0.01135439,  0.03431582,  0.05807661,  0.08332659,
         0.11107642,  0.14305905,  0.18291837,  0.24156609,
    ], dtype=np.float32)


HEAD_DIM = 128


def cpu_quantize(fp_vec, rot, codebook):
    norm = np.linalg.norm(fp_vec)
    rotated = fp_vec @ rot
    unit = rotated / norm
    indices = np.argmin(np.abs(unit[:, None] - codebook[None, :]), axis=-1).astype(np.uint8)
    return indices, norm


def analyze(kv, fp_dir, tq_dir, L, H, phase, step):
    tag = f"{phase}_step{step:04d}"
    codebook = load_codebook(tq_dir)
    rot = generate_rotation_matrix(HEAD_DIM, seed=42 + L * 1000)

    if kv == "k":
        fp = np.load(fp_dir / f"{tag}_k_cache.npy")
        npu_idx = np.load(tq_dir / f"{tag}_quant_k.npy")
        npu_scl = np.load(tq_dir / f"{tag}_quant_k_norm.npy")
    else:
        fp = np.load(fp_dir / f"{tag}_v_cache.npy")
        npu_idx = np.load(tq_dir / f"{tag}_quant_v.npy")
        npu_scl = np.load(tq_dir / f"{tag}_quant_v_norm.npy")

    valid_pos = int(np.count_nonzero(npu_scl[L, H, :, 0]))
    print(f"\n{'='*60}")
    print(f"  {kv.upper()}  Layer {L}  Head {H}  valid_pos={valid_pos}")
    print(f"{'='*60}")

    # ── SCALES ──
    print(f"\n--- SCALES ---")
    print(f"{'pos':>3} | {'CPU_scale':>14} | {'NPU_scale':>14}")
    print("-" * 40)
    for pos in range(valid_pos):
        _, cpu_s = cpu_quantize(fp[L, H, pos], rot, codebook)
        npu_s = float(npu_scl[L, H, pos, 0])
        print(f"{pos:3d} | {cpu_s:14.6f} | {npu_s:14.6f}")

    # ── INDICES ──
    print(f"\n--- INDICES ---")
    cpu_indices = []
    for pos in range(valid_pos):
        idx, _ = cpu_quantize(fp[L, H, pos], rot, codebook)
        cpu_indices.append(idx)

    hdr = f"{'i':>3} |"
    for p in range(valid_pos):
        hdr += f" NPU{p} CPU{p} |"
    print(hdr)
    print("-" * (5 + valid_pos * 10))
    for i in range(HEAD_DIM):
        row = f"{i:3d} |"
        for p in range(valid_pos):
            row += f"   {npu_idx[L, H, p, i]:2d}   {cpu_indices[p][i]:2d} |"
        print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fp_dir")
    parser.add_argument("tq_dir")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--head", type=int, default=0)
    parser.add_argument("--phase", default="prefill", choices=["prefill", "decode"])
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--kv", default="k", choices=["k", "v", "both"])
    parser.add_argument("--all-heads", action="store_true", help="Print all heads")
    args = parser.parse_args()

    fp_dir = Path(args.fp_dir)
    tq_dir = Path(args.tq_dir)

    heads = range(8) if args.all_heads else [args.head]
    for h in heads:
        if args.kv in ("k", "both"):
            analyze("k", fp_dir, tq_dir, args.layer, h, args.phase, args.step)
        if args.kv in ("v", "both"):
            analyze("v", fp_dir, tq_dir, args.layer, h, args.phase, args.step)


if __name__ == "__main__":
    main()
