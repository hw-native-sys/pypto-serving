#!/usr/bin/env python3
"""Create a compact, reproducible Phase F evidence summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    suite = args.run_dir / "suite"
    audit = json.loads((suite / "audit.json").read_text())
    manifest = json.loads((suite / "manifest.json").read_text())
    snapshots = json.loads((suite / "snapshots.json").read_text())
    orchestration = json.loads((args.run_dir / "orchestration.json").read_text())
    results = []
    for path in sorted(suite.glob("*.json")):
        value = json.loads(path.read_text())
        if "elapsed_ns" in value:
            results.append(value)
    latency = [item["elapsed_ns"] / 1e9 for item in results if not item["errors"]]
    starts = [item["started_ns"] for item in results if not item["errors"]]
    finishes = [
        item["started_ns"] + item["elapsed_ns"]
        for item in results
        if not item["errors"]
    ]
    wall_seconds = (max(finishes) - min(starts)) / 1e9 if starts else 0.0
    router = snapshots["router_metrics"]
    prefill = snapshots["prefill_metrics"]
    transfer = prefill.get("durations", {}).get("transfer.duration", {})
    transfer_seconds = transfer.get("sum_ns", 0) / 1e9
    transfer_bytes = prefill.get("counters", {}).get("transfer.bytes", 0)
    summary = {
        "schema_version": 1,
        "result": audit["result"],
        "requests": audit["requests"],
        "passed": audit["passed"],
        "concurrency": manifest["concurrency"],
        "repeat": manifest["repeat"],
        "wall_seconds": wall_seconds,
        "throughput_requests_per_second": (
            len(latency) / wall_seconds if wall_seconds else 0.0
        ),
        "client_latency_seconds": {
            "mean": statistics.fmean(latency) if latency else 0.0,
            "p50": _percentile(latency, 0.50),
            "p95": _percentile(latency, 0.95),
            "max": max(latency, default=0.0),
        },
        "router": {
            "active_peak": router.get("gauges", {}).get("handoffs.active_peak", 0),
            "queued_peak": router.get("gauges", {}).get("handoffs.queued_peak", 0),
            "ttft_mean_seconds": (
                router["durations"]["request.ttft"]["sum_ns"]
                / router["durations"]["request.ttft"]["count"]
                / 1e9
            ),
            "ttft_max_seconds": router["durations"]["request.ttft"]["max_ns"]
            / 1e9,
        },
        "transfer": {
            "chunks": prefill.get("counters", {}).get("transfer.chunks", 0),
            "bytes": transfer_bytes,
            "seconds_sum": transfer_seconds,
            "aggregate_gib_per_second": (
                transfer_bytes / transfer_seconds / (1 << 30)
                if transfer_seconds
                else 0.0
            ),
            "overlap_submitted": prefill.get("counters", {}).get(
                "overlap.submitted", 0
            ),
            "overlap_completed": prefill.get("counters", {}).get(
                "overlap.completed", 0
            ),
        },
        "k7": audit["k7"],
        "teardown": {
            "forced_count": sum(bool(stop.get("forced")) for stop in orchestration["stops"]),
            "states": [stop["state"] for stop in orchestration["stops"]],
        },
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
