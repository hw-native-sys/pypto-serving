#!/usr/bin/env python3
"""Compare serial-transfer and overlap runs with an explicit regression gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial-summary", required=True, type=Path)
    parser.add_argument("--overlap-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-mean-latency-regression", type=float, default=0.10)
    args = parser.parse_args()
    serial = json.loads(args.serial_summary.read_text())
    overlap = json.loads(args.overlap_summary.read_text())
    for name in ("requests", "concurrency", "repeat"):
        if serial[name] != overlap[name]:
            raise ValueError(f"performance summaries differ in {name}")
    serial_latency = serial["client_latency_seconds"]["mean"]
    overlap_latency = overlap["client_latency_seconds"]["mean"]
    latency_ratio = overlap_latency / serial_latency if serial_latency else 0.0
    throughput_ratio = (
        overlap["throughput_requests_per_second"]
        / serial["throughput_requests_per_second"]
        if serial["throughput_requests_per_second"]
        else 0.0
    )
    errors = []
    if serial["result"] != "PASS" or overlap["result"] != "PASS":
        errors.append("one or both source runs failed correctness")
    if overlap["transfer"]["overlap_completed"] < 1:
        errors.append("overlap run completed no asynchronous transfer")
    if latency_ratio > 1.0 + args.max_mean_latency_regression:
        errors.append("overlap mean latency exceeded the frozen regression bound")
    result = {
        "schema_version": 1,
        "latency_ratio_overlap_over_serial": latency_ratio,
        "throughput_ratio_overlap_over_serial": throughput_ratio,
        "max_mean_latency_regression": args.max_mean_latency_regression,
        "errors": errors,
        "result": "PASS" if not errors else "FAIL",
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
