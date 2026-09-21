# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise bounded Router concurrency and explicit overload backpressure."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request


def _get(url: str, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def _one(
    router_url: str,
    barrier: threading.Barrier,
    index: int,
    timeout: float,
) -> dict:
    payload = json.dumps(
        {
            "model": "dsv4-flash-dspark-w8a8",
            "prompt": "紫禁城",
            "max_tokens": 128,
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": False,
        },
        ensure_ascii=False,
    ).encode()
    request = urllib.request.Request(
        f"{router_url.rstrip('/')}/v1/completions",
        data=payload,
        method="POST",
        headers={"content-type": "application/json"},
    )
    barrier.wait(timeout=30)
    started_ns = time.time_ns()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.load(response)
            return {
                "index": index,
                "started_ns": started_ns,
                "finished_ns": time.time_ns(),
                "status": response.status,
                "token_ids_sha256": response.headers.get(
                    "x-pypto-token-ids-sha256", ""
                ),
                "response": body,
            }
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            body = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            body = {"raw": raw.decode(errors="replace")}
        return {
            "index": index,
            "started_ns": started_ns,
            "finished_ns": time.time_ns(),
            "status": exc.code,
            "retry_after": exc.headers.get("Retry-After", ""),
            "response": body,
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-url", required=True)
    parser.add_argument("--prefill-url", required=True)
    parser.add_argument("--decode-url", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument("--expected-successes", type=int, required=True)
    parser.add_argument("--expected-active-peak", type=int, required=True)
    parser.add_argument("--expected-queued-peak", type=int, required=True)
    parser.add_argument("--expected-token-digest", default="")
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.requests < 1 or args.concurrency != args.requests:
        raise ValueError("concurrency probe requires one simultaneous client per request")
    evidence = Path(args.evidence_dir)
    if evidence.exists():
        raise FileExistsError(f"evidence directory already exists: {evidence}")
    evidence.mkdir(parents=True)
    (evidence / "manifest.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    barrier = threading.Barrier(args.requests)
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(
                _one,
                args.router_url,
                barrier,
                index,
                args.timeout_seconds,
            )
            for index in range(args.requests)
        ]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: item["index"])
    (evidence / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    snapshots = {
        "router_metrics": _get(f"{args.router_url.rstrip('/')}/metrics", 30),
        "router_recovery": _get(f"{args.router_url.rstrip('/')}/recovery", 30),
        "prefill_capacity": _get(
            f"{args.prefill_url.rstrip('/')}/internal/pd/capacity", 30
        ),
        "decode_capacity": _get(
            f"{args.decode_url.rstrip('/')}/internal/pd/capacity", 30
        ),
        "prefill_metrics": _get(
            f"{args.prefill_url.rstrip('/')}/internal/pd/metrics", 30
        ),
        "decode_metrics": _get(
            f"{args.decode_url.rstrip('/')}/internal/pd/metrics", 30
        ),
    }
    (evidence / "snapshots.json").write_text(
        json.dumps(snapshots, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    errors = []
    successes = [item for item in results if item["status"] == 200]
    backpressure = [item for item in results if item["status"] == 503]
    expected_backpressure = args.requests - args.expected_successes
    if len(successes) != args.expected_successes:
        errors.append(
            f"expected {args.expected_successes} successes, got {len(successes)}"
        )
    if len(backpressure) != expected_backpressure:
        errors.append(
            f"expected {expected_backpressure} backpressure responses, "
            f"got {len(backpressure)}"
        )
    unexpected = [item["status"] for item in results if item["status"] not in (200, 503)]
    if unexpected:
        errors.append(f"unexpected HTTP statuses: {unexpected}")
    for item in successes:
        response = item["response"]
        choice = response.get("choices", [{}])[0]
        usage = response.get("usage", {})
        if choice.get("finish_reason") != "length":
            errors.append(f"request {item['index']} finish reason mismatch")
        if usage.get("completion_tokens") != 128:
            errors.append(f"request {item['index']} completion count mismatch")
        if not item.get("token_ids_sha256"):
            errors.append(f"request {item['index']} has no token digest")
    digests = {item.get("token_ids_sha256", "") for item in successes}
    digests.discard("")
    if len(digests) != 1:
        errors.append("successful concurrent requests produced different token digests")
    if args.expected_token_digest and digests != {args.expected_token_digest}:
        errors.append("concurrent token digest differs from the frozen G0/G1 baseline")
    for item in backpressure:
        if item.get("retry_after") != "1":
            errors.append(f"request {item['index']} has no Retry-After: 1")
        message = str(item.get("response", {}).get("message", ""))
        if "configured limit" not in message:
            errors.append(f"request {item['index']} has an unexpected 503 reason")

    gauges = snapshots["router_metrics"].get("gauges", {})
    if gauges.get("handoffs.active_peak", 0) < args.expected_active_peak:
        errors.append("Router did not reach the configured active handoff limit")
    if gauges.get("handoffs.queued_peak", 0) < args.expected_queued_peak:
        errors.append("Router did not expose the configured queued handoff waterline")
    if gauges.get("handoffs.active", -1) != 0 or gauges.get("handoffs.queued", -1) != 0:
        errors.append("Router retained active or queued handoffs")
    if snapshots["router_recovery"].get("phase") != "RUNNING":
        errors.append("Router recovery gate is not RUNNING")
    for role in ("prefill_capacity", "decode_capacity"):
        capacity = snapshots[role]
        for name in (
            "active_handoffs",
            "prepared_requests",
            "reservations",
            "quarantined_reservations",
        ):
            if capacity.get(name):
                errors.append(f"{role} retained {name}")
    for role in ("prefill_metrics", "decode_metrics"):
        worker = snapshots[role].get("worker", {})
        if worker.get("pd_transfer_jobs_active", 0):
            errors.append(f"{role} retained transfer jobs")
        if worker.get("active_requests", 0):
            errors.append(f"{role} retained active model requests")

    audit = {
        "schema_version": 1,
        "requests": args.requests,
        "successes": len(successes),
        "backpressure": len(backpressure),
        "token_ids_sha256": sorted(digests),
        "active_peak": gauges.get("handoffs.active_peak", 0),
        "queued_peak": gauges.get("handoffs.queued_peak", 0),
        "errors": errors,
        "result": "PASS" if not errors else "FAIL",
    }
    (evidence / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(audit, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
