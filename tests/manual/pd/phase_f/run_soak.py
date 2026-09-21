# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded fixed-pair soak with resource-watermark and terminal audits."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import time

from run_suite import _get, _one


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-url", required=True)
    parser.add_argument("--prefill-url", required=True)
    parser.add_argument("--decode-url", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-rss-growth-bytes", type=int, default=512 << 20)
    parser.add_argument("--max-fd-growth", type=int, default=32)
    parser.add_argument("--max-thread-growth", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    return parser


def _snapshots(args) -> dict:
    return {
        "timestamp_ns": time.time_ns(),
        "router": _get(f"{args.router_url.rstrip('/')}/metrics", 30),
        "recovery": _get(f"{args.router_url.rstrip('/')}/recovery", 30),
        "prefill_capacity": _get(
            f"{args.prefill_url.rstrip('/')}/internal/pd/capacity", 30
        ),
        "decode_capacity": _get(
            f"{args.decode_url.rstrip('/')}/internal/pd/capacity", 30
        ),
        "prefill": _get(
            f"{args.prefill_url.rstrip('/')}/internal/pd/metrics", 30
        ),
        "decode": _get(
            f"{args.decode_url.rstrip('/')}/internal/pd/metrics", 30
        ),
    }


def main() -> int:
    args = _parser().parse_args()
    if args.requests < 1 or args.concurrency < 1:
        raise ValueError("requests and concurrency must be positive")
    evidence = Path(args.evidence_dir)
    if evidence.exists():
        raise FileExistsError(f"evidence directory already exists: {evidence}")
    evidence.mkdir(parents=True)
    case = {
        "name": "soak-forbidden-city",
        "payload": {
            "model": "dsv4-flash-dspark-w8a8",
            "prompt": "紫禁城",
            "max_tokens": 128,
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": False,
        },
        "expected_completion_tokens": 128,
        "expected_finish_reason": "length",
    }
    samples = [_snapshots(args)]
    results = []
    started_ns = time.time_ns()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for start in range(0, args.requests, args.concurrency):
            batch = range(start, min(args.requests, start + args.concurrency))
            futures = [
                pool.submit(
                    _one,
                    args.router_url.rstrip("/"),
                    "",
                    case,
                    index,
                    args.timeout_seconds,
                )
                for index in batch
            ]
            batch_results = [future.result() for future in futures]
            results.extend(batch_results)
            if any(result["errors"] for result in batch_results):
                break
            samples.append(_snapshots(args))

    with (evidence / "results.jsonl").open("w", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
    (evidence / "samples.json").write_text(
        json.dumps(samples, indent=2, sort_keys=True), encoding="utf-8"
    )

    errors = [error for result in results for error in result["errors"]]
    if len(results) != args.requests:
        errors.append(f"only {len(results)} of {args.requests} requests completed")
    for index, sample in enumerate(samples):
        if sample["recovery"]["phase"] != "RUNNING":
            errors.append(f"sample {index}: Router recovery is not RUNNING")
        for role in ("prefill_capacity", "decode_capacity"):
            capacity = sample[role]
            if (
                capacity["active_handoffs"]
                or capacity["quarantined_reservations"]
                or capacity["reservations"]
                or capacity["prepared_requests"]
            ):
                errors.append(f"sample {index}: {role} retained unsafe state")
        for role in ("prefill", "decode"):
            worker = sample[role].get("worker")
            if worker is None:
                errors.append(f"sample {index}: {role} worker metrics unavailable")
            elif worker.get("pd_transfer_jobs_active", 0) or worker.get(
                "active_requests", 0
            ):
                errors.append(f"sample {index}: {role} retained worker state")
    first, last = samples[0], samples[-1]
    for role in ("router", "prefill", "decode"):
        before = first[role]["process"]
        after = last[role]["process"]
        if after["rss_bytes"] - before["rss_bytes"] > args.max_rss_growth_bytes:
            errors.append(f"{role} RSS growth exceeded the configured bound")
        if after["fds"] - before["fds"] > args.max_fd_growth:
            errors.append(f"{role} FD growth exceeded the configured bound")
        if after["threads"] - before["threads"] > args.max_thread_growth:
            errors.append(f"{role} thread growth exceeded the configured bound")
    elapsed = [result["elapsed_ns"] / 1e9 for result in results if not result["errors"]]
    audit = {
        "schema_version": 1,
        "started_ns": started_ns,
        "finished_ns": time.time_ns(),
        "requested": args.requests,
        "completed": len(results),
        "concurrency": args.concurrency,
        "latency_seconds": {
            "mean": statistics.fmean(elapsed) if elapsed else 0,
            "max": max(elapsed, default=0),
        },
        "k7_final": samples[-1]["decode"].get("worker", {}),
        "errors": errors,
        "result": "PASS" if not errors else "FAIL",
    }
    (evidence / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(audit, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
