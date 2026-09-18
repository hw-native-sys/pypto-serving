#!/usr/bin/env python3
"""Audit Phase F merged traces for real next-chunk/transfer overlap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _events(root: Path) -> list[dict]:
    events = []
    for path in sorted(root.rglob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError):
            continue
        if isinstance(value, dict):
            value = value.get("traceEvents", ())
        if isinstance(value, list):
            events.extend(event for event in value if isinstance(event, dict))
    return events


def _intervals(
    events: list[dict],
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], list[tuple[float, float]]]:
    begins: dict[tuple[object, object], list[float]] = {}
    transfers = []
    for event in sorted(events, key=lambda item: float(item.get("ts", 0))):
        name = event.get("name")
        args = event.get("args") or {}
        key = (args.get("request_id"), args.get("chunk_id"))
        timestamp = float(event.get("ts", 0))
        if name == "pd.transfer.begin":
            begins.setdefault(key, []).append(timestamp)
        elif name == "pd.transfer.end" and begins.get(key):
            transfers.append((begins[key].pop(0), timestamp))
    compute = []
    native = []
    for event in events:
        if event.get("ph") != "X" or float(event.get("dur", 0)) <= 0:
            continue
        start = float(event["ts"])
        if event.get("name") == "DSparkModelRunner.prefill.l3_dispatch":
            compute.append((start, start + float(event["dur"])))
        elif event.get("name") == "MooncakeTransferEngine.batch_transfer_sync_write":
            native.append((start, start + float(event["dur"])))
    return transfers, native, compute


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    events = _events(args.trace_dir)
    transfers, native, compute = _intervals(events)
    host_overlaps = [
        {"host_transfer": transfer, "prefill": dispatch}
        for transfer in transfers
        for dispatch in compute
        if max(transfer[0], dispatch[0]) < min(transfer[1], dispatch[1])
    ]
    native_overlaps = [
        {"native_transfer": transfer, "prefill": dispatch}
        for transfer in native
        for dispatch in compute
        if max(transfer[0], dispatch[0]) < min(transfer[1], dispatch[1])
    ]
    native_names = sorted(
        {
            str(event.get("name", ""))
            for event in events
            if not str(event.get("name", "")).startswith("pd.transfer")
            and any(
                token in str(event.get("name", "")).lower()
                for token in ("roce", "rdma", "adxl", "mooncake")
            )
        }
    )
    result = {
        "schema_version": 1,
        "trace_events": len(events),
        "transfer_intervals": len(transfers),
        "native_transfer_intervals": len(native),
        "prefill_dispatch_intervals": len(compute),
        "host_overlap_pairs": len(host_overlaps),
        "native_overlap_pairs": len(native_overlaps),
        "native_transfer_event_names": native_names,
        "result": "PASS" if native_overlaps else "FAIL",
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
