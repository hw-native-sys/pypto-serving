# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate and summarize the fixed DSV4 GBS32 serving profile."""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from aligned_prompts import validate_prompt_manifest


REPO_ROOT = Path(__file__).resolve().parents[4]
REQUEST_COUNT = 32
OUTPUT_TOKENS = 256

try:
    from simpler_setup.tools.strace_timing import (
        bucket_by_hid,
        group_invocations,
        parse_spans,
        to_chrome_trace,
    )
except ModuleNotFoundError:
    candidates = [
        Path(value)
        for value in (
            os.environ.get("PYPTO_RUNTIME_ROOT"),
            str(REPO_ROOT.parent / "pypto" / "runtime"),
        )
        if value
    ]
    runtime_root = next(
        (candidate for candidate in candidates if (candidate / "simpler_setup").is_dir()),
        None,
    )
    if runtime_root is None:
        raise RuntimeError(
            "cannot import simpler_setup; install the runtime package or set PYPTO_RUNTIME_ROOT"
        ) from None
    sys.path.insert(0, str(runtime_root))
    from simpler_setup.tools.strace_timing import (  # noqa: E402
        bucket_by_hid,
        group_invocations,
        parse_spans,
        to_chrome_trace,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--run-id", "--task-id", dest="run_id", default="unknown")
    return parser.parse_args()


def stats(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def span_ms(span) -> float:
    return span.dur / 1_000_000.0


def span_by_alias(names: dict, *aliases: str):
    for alias in aliases:
        span = names.get(alias)
        if span is not None:
            return span
    raise KeyError(f"missing span aliases: {aliases}")


def classify_callables(invocations: list) -> tuple[dict[str, str], dict[str, int]]:
    counts = Counter(invocation.hid for invocation in invocations)
    first_ts = {
        hid: min(invocation.root().ts for invocation in invocations if invocation.hid == hid)
        for hid in counts
    }
    groups: dict[int, list[str]] = defaultdict(list)
    for hid, count in counts.items():
        groups[count].append(hid)
    ordered = sorted(groups.items())
    if len(ordered) != 3 or [len(hids) for _count, hids in ordered] != [2, 2, 1]:
        raise RuntimeError(f"cannot classify aligned GBS32 callables: {counts}")
    main_hids = sorted(ordered[0][1], key=first_ts.get)
    mtp_hids = sorted(ordered[1][1], key=first_ts.get)
    decode_hid = ordered[2][1][0]
    labels = {
        main_hids[0]: "prefill.main",
        main_hids[1]: "prefill.main.lm_head",
        mtp_hids[0]: "prefill.mtp",
        mtp_hids[1]: "prefill.mtp.lm_head",
        decode_hid: "decode.main+verify+mtp",
    }
    return labels, dict(counts)


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir.resolve()
    server_log_path = artifact_dir / "server.log"
    serving_trace_path = artifact_dir / "serving-trace" / "trace.json"
    performance_path = artifact_dir / "performance_summary.json"
    responses_path = artifact_dir / "responses.json"
    server_log = server_log_path.read_text(encoding="utf-8", errors="replace")
    serving_payload = json.loads(serving_trace_path.read_text(encoding="utf-8"))
    performance = json.loads(performance_path.read_text(encoding="utf-8"))
    responses = json.loads(responses_path.read_text(encoding="utf-8"))
    prompt_manifest = json.loads((artifact_dir / "prompt_manifest.json").read_text(encoding="utf-8"))
    prompt_digest = validate_prompt_manifest(prompt_manifest)

    if len(responses) != REQUEST_COUNT:
        raise RuntimeError(f"expected {REQUEST_COUNT} responses, got {len(responses)}")
    lengths = [len(item["token_ids"]) for item in responses]
    if lengths != [OUTPUT_TOKENS] * REQUEST_COUNT:
        raise RuntimeError(f"unexpected profiled output lengths: {lengths}")

    serving_events = serving_payload["traceEvents"]
    serving_spans = [event for event in serving_events if event.get("ph") == "X"]
    categories = Counter(str(event.get("cat", "")) for event in serving_spans)
    required_categories = {"scheduler", "worker", "executor", "kernel", "serving"}
    missing_categories = required_categories - categories.keys()
    if missing_categories:
        raise RuntimeError(f"missing serving profile categories: {sorted(missing_categories)}")
    serving_start_us = min(float(event["ts"]) for event in serving_spans)
    serving_end_us = max(
        float(event["ts"]) + float(event.get("dur", 0)) for event in serving_spans
    )

    pid_to_device = {
        int(pid): int(device)
        for pid, device in re.findall(
            r"\[chip_process pid=(\d+) dev=(\d+)\] ready", server_log
        )
    }
    devices = sorted(pid_to_device.values())
    if devices != sorted(set(devices)) or len(devices) != 8:
        raise RuntimeError(f"expected eight unique device mappings, got {pid_to_device}")

    split_log = server_log.replace("[STRACE]", "\n[STRACE]")
    spans = list(parse_spans(split_log.splitlines()))
    if any(span.is_device for span in spans):
        raise RuntimeError("device STRACE is unsupported; use Host STRACE only")
    invocations = group_invocations(spans)
    selected = []
    for invocation in invocations:
        root = invocation.root()
        if root is None or invocation.pid not in pid_to_device:
            continue
        root_start_us = root.ts / 1000.0
        root_end_us = (root.ts + root.dur) / 1000.0
        if root_start_us <= serving_end_us and root_end_us >= serving_start_us:
            selected.append(invocation)
    if not selected:
        raise RuntimeError("no Host STRACE invocation overlaps the formal serving profile")

    by_pid: dict[int, list] = defaultdict(list)
    for invocation in selected:
        by_pid[invocation.pid].append(invocation)
    if set(by_pid) != set(pid_to_device):
        raise RuntimeError(f"incomplete profiled rank set: {sorted(by_pid)}")
    signatures = {
        tuple(sorted(Counter(item.hid for item in rows).values()))
        for rows in by_pid.values()
    }
    if len(signatures) != 1:
        raise RuntimeError("profiled callable counts differ across ranks")

    reference_pid = min(by_pid, key=pid_to_device.get)
    labels, callable_counts = classify_callables(by_pid[reference_pid])
    decode_hid = next(hid for hid, label in labels.items() if label.startswith("decode"))
    per_rank_decode = {
        pid: sorted(
            (item for item in rows if item.hid == decode_hid),
            key=lambda item: item.root().ts,
        )
        for pid, rows in by_pid.items()
    }
    decode_counts = {pid_to_device[pid]: len(rows) for pid, rows in per_rank_decode.items()}
    unique_decode_counts = set(decode_counts.values())
    if len(unique_decode_counts) != 1 or not unique_decode_counts or 0 in unique_decode_counts:
        raise RuntimeError(f"decode step counts differ across ranks: {decode_counts}")
    decode_step_count = unique_decode_counts.pop()

    decode_rows = []
    ordered_pids = sorted(pid_to_device, key=pid_to_device.get)
    for step in range(decode_step_count):
        rank_rows = []
        for pid in ordered_pids:
            invocation = per_rank_decode[pid][step]
            names = invocation.by_name()
            root = invocation.root()
            rank_rows.append(
                {
                    "device": pid_to_device[pid],
                    "source_invocation": invocation.inv,
                    "host_ms": span_ms(root),
                    "bind_ms": span_ms(span_by_alias(names, "chip.run.bind", "simpler_run.bind")),
                    "runner_run_ms": span_ms(
                        span_by_alias(names, "chip.run.runner_run", "simpler_run.runner_run")
                    ),
                    "validate_ms": span_ms(
                        span_by_alias(names, "chip.run.validate", "simpler_run.validate")
                    ),
                }
            )
        decode_rows.append(
            {
                "step": step + 1,
                "critical_device": max(rank_rows, key=lambda row: row["host_ms"])["device"],
                "critical_host_ms": max(row["host_ms"] for row in rank_rows),
                "critical_bind_ms": max(row["bind_ms"] for row in rank_rows),
                "critical_runner_run_ms": max(row["runner_run_ms"] for row in rank_rows),
                "critical_validate_ms": max(row["validate_ms"] for row in rank_rows),
                "per_rank": rank_rows,
            }
        )

    simpler_payload = to_chrome_trace(selected, bucket_by_hid(selected))
    (artifact_dir / "simpler-swimlane.json").write_text(
        json.dumps(simpler_payload, indent=2) + "\n", encoding="utf-8"
    )

    phase_summary = {
        "step_ms": stats([row["critical_host_ms"] for row in decode_rows]),
        "bind_ms": stats([row["critical_bind_ms"] for row in decode_rows]),
        "runner_run_ms": stats([row["critical_runner_run_ms"] for row in decode_rows]),
        "validate_ms": stats([row["critical_validate_ms"] for row in decode_rows]),
    }
    summary = {
        "run_id": args.run_id,
        "workload": {
            "gbs": REQUEST_COUNT,
            "dp": 8,
            "per_rank_batch": 4,
            "prompt_tokens": 64,
            "unique_prompts": len(prompt_manifest),
            "prompt_token_matrix_sha256": prompt_digest,
            "output_tokens": OUTPUT_TOKENS,
            "mtp_k": 1,
            "temperature": 0,
        },
        "devices": devices,
        "performance": performance,
        "serving_profile": {
            "event_count": len(serving_events),
            "span_count": len(serving_spans),
            "span_categories": dict(sorted(categories.items())),
        },
        "simpler": {
            "device_effective_available": False,
            "selected_invocations": len(selected),
            "selected_invocations_per_rank": {
                str(pid_to_device[pid]): len(rows) for pid, rows in by_pid.items()
            },
            "callable_counts_per_rank": callable_counts,
            "callable_labels": labels,
            "decode_steps": decode_rows,
            "critical_phase_summary": phase_summary,
        },
    }
    (artifact_dir / "profile-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    report = f"""# DeepSeek V4 aligned GBS32 serving profile

- Run: `{args.run_id}`
- Workload: `GBS32 / DP8 / per-rank batch4 / Seq64 / Output256 / MTP k=1 / temperature=0`
- Inputs: 32 distinct attraction prompts; token matrix SHA256 `{prompt_digest}`
- Serving categories: `{dict(sorted(categories.items()))}`
- Host STRACE invocations: `{len(selected)}` (`{len(selected) // 8}` per rank)
- Decode steps: `{decode_step_count}`
- Critical host Step mean: `{phase_summary['step_ms']['mean']:.3f} ms`
- Device Effective: unavailable (Device STRACE intentionally disabled)

Open `serving-strace-swimlane.json` in Perfetto for the combined serving and 8-rank Host view.
"""
    (artifact_dir / "profile-summary.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
