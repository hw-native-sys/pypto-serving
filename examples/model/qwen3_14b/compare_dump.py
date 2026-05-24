# Copyright (c) PyPTO Contributors.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compare intermediate tensor dumps between FP and TQ executors.

Usage:
    # Compare prefill dumps (runs on both TQ and FP first):
    python cpu_generate.py \\
        --model-dir /path/to/Qwen3-14B \\
        --prompt "Hello" --max-new-tokens 1 \\
        --compare --dump-dir ./dump_output

    # Then analyze:
    python compare_dump.py --fp-dir ./dump_output/fp --tq-dir ./dump_output/tq

    # Focus on specific layers or decode steps:
    python compare_dump.py --fp-dir ./dump_output/fp --tq-dir ./dump_output/tq --layers 0,15,39
    python compare_dump.py --fp-dir ./dump_output/fp --tq-dir ./dump_output/tq --step decode5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two flattened tensors."""
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    dot = (a_flat * b_flat).sum()
    norm_a = a_flat.norm()
    norm_b = b_flat.norm()
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return (dot / (norm_a * norm_b)).item()


def compute_metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Compute comparison metrics between two tensors."""
    a_f = a.float()
    b_f = b.float()
    diff = a_f - b_f
    mse = (diff ** 2).mean().item()
    mae = diff.abs().mean().item()
    max_err = diff.abs().max().item()
    rel_err = (diff.abs() / (a_f.abs() + 1e-12)).mean().item()
    cos = cosine_sim(a, b)
    return {
        "cosine": cos,
        "mse": mse,
        "mae": mae,
        "max_err": max_err,
        "rel_err": rel_err,
    }


def find_matching_files(fp_dir: Path, tq_dir: Path, step_filter: str | None = None):
    """Find matching .pt files in both directories."""
    fp_files = sorted(fp_dir.glob("*.pt"))
    matches = []
    for fp_file in fp_files:
        name = fp_file.name
        tq_file = tq_dir / name
        if not tq_file.exists():
            continue
        if step_filter and step_filter not in name:
            continue
        matches.append((name, fp_file, tq_file))
    return matches


def compare_tensors_in_file(fp_file: Path, tq_file: Path) -> list[dict]:
    """Load two .pt files and compare matching tensor keys."""
    fp_data = torch.load(fp_file, map_location="cpu", weights_only=False)
    tq_data = torch.load(tq_file, map_location="cpu", weights_only=False)

    results = []
    all_keys = sorted(set(fp_data.keys()) | set(tq_data.keys()))

    for key in all_keys:
        fp_tensor = fp_data.get(key)
        tq_tensor = tq_data.get(key)

        if fp_tensor is None:
            results.append({"key": key, "status": "MISSING_FP", "metrics": None})
            continue
        if tq_tensor is None:
            results.append({"key": key, "status": "MISSING_TQ", "metrics": None})
            continue

        # Align shapes: TQ kv_dequant may have different layout
        if fp_tensor.shape != tq_tensor.shape:
            results.append({
                "key": key,
                "status": "SHAPE_MISMATCH",
                "fp_shape": list(fp_tensor.shape),
                "tq_shape": list(tq_tensor.shape),
                "metrics": None,
            })
            continue

        metrics = compute_metrics(fp_tensor, tq_tensor)
        results.append({"key": key, "status": "OK", "metrics": metrics})

    return results


def print_layer_report(layer_results: list[dict], file_name: str) -> None:
    """Print a formatted comparison report for one layer file."""
    print(f"\n{'─' * 80}")
    print(f"  {file_name}")
    print(f"{'─' * 80}")
    header = f"  {'Tensor':<20s} {'Status':<16s} {'CosSim':>8s} {'MSE':>12s} {'MAE':>12s} {'MaxErr':>12s} {'RelErr':>10s}"
    print(header)
    print(f"  {'─' * 78}")

    for r in layer_results:
        key = r["key"]
        status = r["status"]

        if status == "OK":
            m = r["metrics"]
            print(f"  {key:<20s} {'OK':<16s} {m['cosine']:>8.6f} {m['mse']:>12.2e} "
                  f"{m['mae']:>12.2e} {m['max_err']:>12.2e} {m['rel_err']:>10.2e}")
        elif status == "SHAPE_MISMATCH":
            print(f"  {key:<20s} {'SHAPE_MISMATCH':<16s} FP={r['fp_shape']} TQ={r['tq_shape']}")
        elif status.startswith("MISSING"):
            print(f"  {key:<20s} {status:<16s}")


def print_summary(all_results: list[tuple[str, list[dict]]]) -> None:
    """Print a compact summary across all layers."""
    print(f"\n{'=' * 80}")
    print("  SUMMARY: Per-layer cosine similarity")
    print(f"{'=' * 80}")

    # Group by layer
    layer_metrics: dict[str, dict[str, float]] = {}
    for file_name, results in all_results:
        # Extract layer and phase from filename like "layer00_prefill_input.pt"
        parts = file_name.replace(".pt", "").split("_", 1)
        layer = parts[0] if parts else file_name
        phase = parts[1] if len(parts) > 1 else ""

        for r in results:
            if r["status"] == "OK" and r["metrics"] is not None:
                key = f"{layer}/{r['key']}"
                layer_metrics[key] = r["metrics"]["cosine"]

    # Sort and print
    items = sorted(layer_metrics.items())
    # Find worst entries
    if items:
        worst = sorted(items, key=lambda x: x[1])[:10]
        print(f"\n  Top-10 lowest cosine similarity:")
        print(f"  {'Tensor':<35s} {'CosSim':>10s}")
        print(f"  {'─' * 47}")
        for name, cos in worst:
            flag = " ⚠️" if cos < 0.99 else ""
            print(f"  {name:<35s} {cos:>10.6f}{flag}")

    # Count issues
    total = 0
    issues = 0
    for file_name, results in all_results:
        for r in results:
            if r["status"] == "OK":
                total += 1
                if r["metrics"]["cosine"] < 0.99:
                    issues += 1
            elif r["status"] != "OK":
                issues += 1
                total += 1

    print(f"\n  Total comparisons: {total}, issues (cos < 0.99 or mismatch): {issues}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare intermediate tensor dumps between FP and TQ executors.",
    )
    parser.add_argument("--fp-dir", required=True, help="Directory with FP executor dumps.")
    parser.add_argument("--tq-dir", required=True, help="Directory with TQ executor dumps.")
    parser.add_argument("--layers", default=None,
                        help="Comma-separated layer indices to compare (e.g. '0,15,39').")
    parser.add_argument("--step", default=None,
                        help="Only compare files containing this step string (e.g. 'prefill', 'decode5').")
    parser.add_argument("--summary-only", action="store_true",
                        help="Only print the summary, skip per-file details.")
    args = parser.parse_args()

    fp_dir = Path(args.fp_dir)
    tq_dir = Path(args.tq_dir)

    if not fp_dir.is_dir():
        print(f"Error: FP dump directory not found: {fp_dir}", file=sys.stderr)
        sys.exit(1)
    if not tq_dir.is_dir():
        print(f"Error: TQ dump directory not found: {tq_dir}", file=sys.stderr)
        sys.exit(1)

    # Find matching files
    matches = find_matching_files(fp_dir, tq_dir, step_filter=args.step)

    # Filter by layers
    if args.layers:
        selected = set(f"layer{int(l):02d}" for l in args.layers.split(","))
        matches = [(n, f, t) for n, f, t in matches
                    if any(n.startswith(s) for s in selected)]

    if not matches:
        print("No matching dump files found.")
        sys.exit(0)

    print(f"Found {len(matches)} matching dump files.")
    print(f"FP dir: {fp_dir}")
    print(f"TQ dir: {tq_dir}")

    # Compare each file
    all_results: list[tuple[str, list[dict]]] = []
    for name, fp_file, tq_file in matches:
        results = compare_tensors_in_file(fp_file, tq_file)
        all_results.append((name, results))
        if not args.summary_only:
            print_layer_report(results, name)

    # Print summary
    print_summary(all_results)


if __name__ == "__main__":
    main()
