# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Offline P0 report. Planned load and host envelopes never imply device cost."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


def analyze_capture(path: Path) -> tuple[dict, dict]:
    counts = Counter()
    placements = Counter()
    reasons = Counter()
    steps = defaultdict(dict)
    captures = set()
    boundaries = defaultdict(dict)
    trace = []
    process_tracks = {}
    malformed = 0
    sources = sorted(path.glob("*.jsonl"))
    manifests = []
    process_boundaries = set()
    for source in sources:
        manifest = source.with_suffix(".manifest.json")
        try:
            state = json.loads(manifest.read_text())
            if not isinstance(state, dict):
                raise ValueError("Invalid manifest object")
        except (OSError, ValueError) as exc:
            state = dict(closed=False, error=str(exc), dropped=None)
        manifests.append(state)
        for line in source.read_text().splitlines():
            try:
                event = json.loads(line)
            except (ValueError, TypeError):
                malformed += 1
                continue
            if not isinstance(event, dict) or not {"capture_id", "kind", "role", "replica",
                                                  "clock_domain", "monotonic_ns"} <= event.keys():
                malformed += 1
                continue
            capture = event["capture_id"]
            captures.add(capture)
            kind = event["kind"]
            counts[kind] += 1
            if kind in {"capture_start", "capture_stop"}:
                process_boundaries.add((event["role"], event["replica"], kind))
            replica = event["replica"]
            domain = event["clock_domain"]
            if domain not in process_tracks:
                process_tracks[domain] = len(process_tracks) + 1
                trace.append(dict(name="process_name", ph="M", pid=process_tracks[domain], tid=0,
                                  args=dict(name=domain)))
            process_track = process_tracks[domain]
            trace.append(dict(name=kind, cat="dplb_host", ph="i", s="t",
                              ts=event["monotonic_ns"] / 1000,
                              pid=process_track, tid=replica,
                              args={k: v for k, v in event.items() if k != "monotonic_ns"}))
            if kind == "ownership" and event.get("action") in {"admit", "prefix_admit"}:
                placements[str(event["group"])] += 1
            if kind == "placement_rank":
                reasons["short" if event["short"] else "long"] += 1
            if kind == "scheduled_step":
                for group in event["groups"]:
                    trace.append(dict(name="planned_group_load", cat="dplb_planned", ph="i", s="t",
                                      ts=event["monotonic_ns"] / 1000,
                                      pid=process_track, tid=1000000 + replica * 1000 + group["group"],
                                      args=group))
            if event["role"] == "engine" and kind in {"capture_start", "capture_stop"}:
                boundaries[replica][kind] = event
            if "step_id" in event:
                key = (capture, replica, event["step_id"])
                steps[key][kind] = event
    if len(captures) > 1:
        raise ValueError("Analyze one capture directory at a time")
    missing = Counter()
    ratios = []
    for events in steps.values():
        for kind in ("scheduled_step", "worker_prepared", "worker_command_begin",
                     "worker_command_reclaimed", "step_outcome"):
            if kind not in events:
                missing[kind] += 1
        planned = events.get("scheduled_step", {})
        lengths = [g["sum_seq_len"] for g in planned.get("groups", [])]
        if lengths and sum(lengths):
            ratios.append(max(lengths) * len(lengths) / sum(lengths))
    itl_sum = itl_count = generated = 0
    windows = []
    for replica, edges in boundaries.items():
        start, stop = edges.get("capture_start"), edges.get("capture_stop")
        if start is None or stop is None:
            continue
        if start.get("counters") is None or stop.get("counters") is None:
            continue
        def metrics(event):
            return next((item for item in event["counters"]["replicas"]
                         if item["engine"] == replica), {})
        before, after = metrics(start), metrics(stop)
        old = before.get("histograms", {}).get("itl", {"sum": 0, "count": 0})
        new = after.get("histograms", {}).get("itl", {"sum": 0, "count": 0})
        itl_sum += new["sum"] - old["sum"]
        itl_count += new["count"] - old["count"]
        generated += (after.get("counters", {}).get("generation_tokens", 0) -
                      before.get("counters", {}).get("generation_tokens", 0))
        windows.append(dict(replica=replica, start_ns=start["monotonic_ns"], stop_ns=stop["monotonic_ns"]))
    complete = bool(sources and boundaries and len(windows) == len(boundaries))
    complete = complete and all((role, replica, kind) in process_boundaries
                                for replica in boundaries for role in ("engine", "worker")
                                for kind in ("capture_start", "capture_stop"))
    complete = complete and not malformed and not missing and all(
        m.get("closed") and not m.get("error") and m.get("dropped") == 0 for m in manifests)
    summary = dict(schema_version=1, capture_ids=sorted(captures),
                   host_capture_status="complete" if complete else "partial",
                   device_timing="unavailable", device_coverage=0,
                   files=len(sources), malformed_lines=malformed, manifests=manifests,
                   events=dict(counts), placements_by_group=dict(placements),
                   ranking_branches=dict(reasons), sampled_steps=len(steps), missing=dict(missing),
                   mean_planned_decode_sum_length_imbalance=(sum(ratios) / len(ratios) if ratios else None),
                   observed_generation_tokens=generated if windows else None,
                   observed_itl_sum_seconds=itl_sum if windows else None,
                   observed_itl_count=itl_count if windows else None,
                   observed_token_weighted_tpot_ms=1000 * itl_sum / itl_count if itl_count else None,
                   windows=windows)
    return summary, dict(traceEvents=trace, displayTimeUnit="ms")


def write_report(path: Path) -> dict:
    summary, trace = analyze_capture(path)
    (path / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (path / "trace.json").write_text(json.dumps(trace) + "\n")
    (path / "report.md").write_text(
        "# DPLB P0 capture\n\n"
        f"Host capture: {summary['host_capture_status']}. Sampled steps: {summary['sampled_steps']}.\n\n"
        f"Observed token-weighted TPOT (ms): {summary['observed_token_weighted_tpot_ms']}.\n\n"
        "This is the delta of cumulative ITL sum divided by ITL count, including observed tokens "
        "of unfinished requests. Unobserved tails are excluded. Zero intervals in multi-token "
        "updates follow serving's existing accounting.\n\n"
        "Device timing is unavailable. Planned length imbalance and worker command host events "
        "cannot establish attention cost, device waits, or DPLB speedup. Worker events describe "
        "commands; fused bootstrap may launch multiple runtime invocations inside a command.\n\n"
        "Inspect summary.json for dropped events, missing joins and capture boundaries. "
        "Perfetto trace.json uses separate process clock domains.\n")
    return summary
