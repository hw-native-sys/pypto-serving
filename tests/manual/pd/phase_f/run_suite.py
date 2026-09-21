# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run a bounded Phase F matrix and archive structured three-plane evidence."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import platform
import time
import urllib.request


def _request(url: str, payload: dict, timeout: float, headers=None) -> dict:
    wire = json.dumps(payload, ensure_ascii=False).encode()
    request = urllib.request.Request(
        url,
        data=wire,
        method="POST",
        headers={"content-type": "application/json", **(headers or {})},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
        digest = response.headers.get("x-pypto-token-ids-sha256", "")
        if digest:
            payload["_pypto_token_ids_sha256"] = digest
        return payload


def _get(url: str, timeout: float, headers=None) -> dict:
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _post_empty(url: str, timeout: float, headers=None) -> None:
    request = urllib.request.Request(url, data=b"", method="POST", headers=headers or {})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response.read()


def _one(
    router_url: str,
    baseline_url: str,
    case: dict,
    index: int,
    timeout: float,
) -> dict:
    started_ns = time.time_ns()
    monotonic = time.monotonic_ns()
    payload = dict(case["payload"])
    if "prompt_token_count" in case:
        payload["prompt"] = [
            int(case.get("prompt_token_id", 100))
        ] * int(case["prompt_token_count"])
    response = _request(
        f"{router_url}/v1/completions",
        payload,
        timeout,
    )
    elapsed_ns = time.monotonic_ns() - monotonic
    choice = response["choices"][0]
    usage = response["usage"]
    errors = []
    # A valid single generated token can be a zero-width/special token when
    # the prompt itself is supplied as exact token IDs.  Correctness for these
    # cases is established by completion count plus the engine's terminal
    # token-ID digest, not by rendered text length.
    if "prompt_token_count" not in case and not choice.get("text"):
        errors.append("empty output text")
    if choice.get("finish_reason") != case["expected_finish_reason"]:
        errors.append("finish reason mismatch")
    if usage.get("completion_tokens") != case["expected_completion_tokens"]:
        errors.append("completion token count mismatch")
    if "prompt_token_count" in case and usage.get("prompt_tokens") != case[
        "prompt_token_count"
    ]:
        errors.append("prompt token count mismatch")
    baseline = None
    if baseline_url:
        baseline = _request(
            f"{baseline_url}/v1/completions",
            payload,
            timeout,
        )
        baseline_choice = baseline["choices"][0]
        if choice.get("text") != baseline_choice.get("text"):
            errors.append("PD/non-PD greedy output text mismatch")
        if choice.get("finish_reason") != baseline_choice.get("finish_reason"):
            errors.append("PD/non-PD finish reason mismatch")
        if usage != baseline.get("usage"):
            errors.append("PD/non-PD usage mismatch")
        pd_digest = response.get("_pypto_token_ids_sha256", "")
        baseline_digest = baseline.get("_pypto_token_ids_sha256", "")
        if not pd_digest or not baseline_digest:
            errors.append("PD/non-PD token-ID digest header missing")
        elif pd_digest != baseline_digest:
            errors.append("PD/non-PD greedy token-ID digest mismatch")
    return {
        "case": case["name"],
        "iteration": index,
        "started_ns": started_ns,
        "elapsed_ns": elapsed_ns,
        "request_id": response.get("id", ""),
        "response": response,
        "baseline_response": baseline,
        "errors": errors,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-url", required=True)
    parser.add_argument("--prefill-url", required=True)
    parser.add_argument("--decode-url", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument(
        "--cases",
        default=str(Path(__file__).with_name("cases.json")),
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    parser.add_argument("--baseline-url", default="")
    parser.add_argument("--profile", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.concurrency < 1 or args.repeat < 1:
        raise ValueError("concurrency and repeat must be positive")
    evidence = Path(args.evidence_dir)
    if evidence.exists():
        raise FileExistsError(f"evidence directory already exists: {evidence}")
    evidence.mkdir(parents=True)
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    work = [
        (case, iteration)
        for iteration in range(args.repeat)
        for case in cases
    ]
    manifest = {
        "schema_version": 1,
        "started_ns": time.time_ns(),
        "router_url": args.router_url,
        "prefill_url": args.prefill_url,
        "decode_url": args.decode_url,
        "concurrency": args.concurrency,
        "repeat": args.repeat,
        "case_names": [case["name"] for case in cases],
        "python": platform.python_version(),
        "host": platform.node(),
    }
    (evidence / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    results = []
    if args.profile:
        for url in (args.prefill_url, args.decode_url):
            _post_empty(
                f"{url.rstrip('/')}/internal/pd/start-profile",
                120,
            )
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(
                _one,
                args.router_url.rstrip("/"),
                args.baseline_url.rstrip("/"),
                case,
                iteration,
                args.timeout_seconds,
            ): (case, iteration)
            for case, iteration in work
        }
        for future in as_completed(futures):
            case, iteration = futures[future]
            try:
                result = future.result()
            except BaseException as exc:
                result = {
                    "case": case["name"],
                    "iteration": iteration,
                    "errors": [f"{type(exc).__name__}: {exc}"],
                }
            results.append(result)
            path = evidence / f"{case['name']}-{iteration:04d}.json"
            path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
    if args.profile:
        for url in (args.prefill_url, args.decode_url):
            _post_empty(
                f"{url.rstrip('/')}/internal/pd/stop-profile",
                300,
            )
    snapshots = {
        "router_metrics": _get(f"{args.router_url.rstrip('/')}/metrics", 30),
        "router_recovery": _get(f"{args.router_url.rstrip('/')}/recovery", 30),
        "prefill_capacity": _get(
            f"{args.prefill_url.rstrip('/')}/internal/pd/capacity",
            30,
        ),
        "decode_capacity": _get(
            f"{args.decode_url.rstrip('/')}/internal/pd/capacity",
            30,
        ),
        "prefill_metrics": _get(
            f"{args.prefill_url.rstrip('/')}/internal/pd/metrics",
            30,
        ),
        "decode_metrics": _get(
            f"{args.decode_url.rstrip('/')}/internal/pd/metrics",
            30,
        ),
    }
    (evidence / "snapshots.json").write_text(
        json.dumps(snapshots, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    errors = [error for result in results for error in result.get("errors", ())]
    terminal_by_request = {
        item["request_id"]: item
        for item in snapshots["router_metrics"].get("recent_terminal", ())
    }
    digests_by_case: dict[str, set[str]] = {}
    for result in results:
        request_id = result.get("request_id", "")
        if result.get("errors"):
            continue
        terminal = terminal_by_request.get(request_id)
        if terminal is None:
            errors.append(f"missing Router terminal metrics for {request_id}")
        elif not terminal.get("token_ids_sha256"):
            errors.append(f"missing final token-id digest for {request_id}")
        else:
            digest = terminal["token_ids_sha256"]
            result["token_ids_sha256"] = digest
            response_digest = result["response"].get(
                "_pypto_token_ids_sha256", ""
            )
            if response_digest != digest:
                errors.append(
                    f"Router response/terminal token-ID digest differs for {request_id}"
                )
            digests_by_case.setdefault(result["case"], set()).add(digest)
            path = evidence / f"{result['case']}-{result['iteration']:04d}.json"
            path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
    if args.repeat > 1:
        for case_name, digests in sorted(digests_by_case.items()):
            if len(digests) != 1:
                errors.append(
                    f"{case_name}: repeated greedy token-ID digests differ"
                )
    comparison_groups: dict[str, set[str]] = {}
    for case in cases:
        group = case.get("comparison_group", "")
        if not group:
            continue
        comparison_groups.setdefault(group, set()).update(
            digests_by_case.get(case["name"], set())
        )
    for group, digests in sorted(comparison_groups.items()):
        if len(digests) != 1:
            errors.append(
                f"comparison group {group}: greedy token-ID digests differ"
            )
    if snapshots["router_recovery"]["phase"] != "RUNNING":
        errors.append("Router recovery gate is not RUNNING")
    for role in ("prefill_capacity", "decode_capacity"):
        if snapshots[role]["active_handoffs"]:
            errors.append(f"{role} retained active handoffs")
        if snapshots[role]["quarantined_reservations"]:
            errors.append(f"{role} contains quarantined reservations")
        if snapshots[role]["reservations"]:
            errors.append(f"{role} retained terminal allocator reservations")
        if snapshots[role]["prepared_requests"]:
            errors.append(f"{role} retained prepared requests")
    for role in ("prefill_metrics", "decode_metrics"):
        worker = snapshots[role].get("worker")
        if worker is None:
            errors.append(f"{role} has no worker-resident metrics")
            continue
        if worker.get("pd_transfer_jobs_active", 0):
            errors.append(f"{role} retained active transfer jobs")
        if worker.get("active_requests", 0):
            errors.append(f"{role} retained active K7 request state")
    decode_worker = snapshots["decode_metrics"].get("worker", {})
    if results and decode_worker.get("verify_steps", 0) <= 0:
        errors.append("Decode worker reported no K7 verification steps")
    audit = {
        "schema_version": 1,
        "finished_ns": time.time_ns(),
        "requests": len(results),
        "passed": len(results) - len([r for r in results if r.get("errors")]),
        "k7": {
            name: decode_worker.get(name, 0)
            for name in (
                "verify_steps",
                "proposed_drafts",
                "matched_drafts",
                "accepted_tokens",
                "fallback_steps",
                "mean_accepted_length",
            )
        },
        "errors": errors,
        "comparison_groups": {
            group: sorted(digests)
            for group, digests in sorted(comparison_groups.items())
        },
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
