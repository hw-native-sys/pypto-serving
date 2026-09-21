# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run a repeatable non-PD K7 correctness baseline with exact token digests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from run_suite import _request


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    args = parser.parse_args()
    if args.repeat < 1:
        raise ValueError("baseline repeat must be positive")
    if args.evidence_dir.exists():
        raise FileExistsError(args.evidence_dir)
    args.evidence_dir.mkdir(parents=True)
    cases = (
        json.loads(args.cases.read_text(encoding="utf-8"))
        if args.cases is not None
        else [
            {
                "name": "baseline-forbidden-city",
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
        ]
    )
    results = []
    for index in range(args.repeat):
        for case in cases:
            payload = dict(case["payload"])
            if "prompt_token_count" in case:
                payload["prompt"] = [
                    int(case.get("prompt_token_id", 100))
                ] * int(case["prompt_token_count"])
            started = time.monotonic_ns()
            response = _request(
                f"{args.url.rstrip('/')}/v1/completions",
                payload,
                args.timeout_seconds,
            )
            text = response["choices"][0]["text"]
            result = {
                "case": case["name"],
                "iteration": index,
                "elapsed_ns": time.monotonic_ns() - started,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "response": response,
            }
            results.append(result)
            (args.evidence_dir / f"{case['name']}-{index:04d}.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
    errors = []
    for result in results:
        response = result["response"]
        case = next(item for item in cases if item["name"] == result["case"])
        if response["choices"][0]["finish_reason"] != case["expected_finish_reason"]:
            errors.append(f"{result['case']}: finish reason mismatch")
        if response["usage"]["completion_tokens"] != case["expected_completion_tokens"]:
            errors.append(f"{result['case']}: completion token count mismatch")
        if "prompt_token_count" in case and response["usage"]["prompt_tokens"] != case[
            "prompt_token_count"
        ]:
            errors.append(f"{result['case']}: prompt token count mismatch")
        if not response.get("_pypto_token_ids_sha256", ""):
            errors.append(f"{result['case']}: token-ID digest header missing")
    digests_by_case = {
        case["name"]: {
            item["response"].get("_pypto_token_ids_sha256", "")
            for item in results
            if item["case"] == case["name"]
            and item["response"].get("_pypto_token_ids_sha256", "")
        }
        for case in cases
    }
    if args.repeat > 1:
        for case_name, digests in digests_by_case.items():
            if len(digests) != 1:
                errors.append(f"{case_name}: repeated greedy token-ID digests differ")
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
    audit = {
        "schema_version": 1,
        "repeat": args.repeat,
        "requests": len(results),
        "case_names": [case["name"] for case in cases],
        "unique_text_digests_by_case": {
            case["name"]: len(
                {item["text_sha256"] for item in results if item["case"] == case["name"]}
            )
            for case in cases
        },
        "unique_token_id_digests_by_case": {
            name: len(digests) for name, digests in digests_by_case.items()
        },
        "comparison_groups": {
            group: sorted(digests)
            for group, digests in sorted(comparison_groups.items())
        },
        "errors": errors,
        "result": "PASS" if not errors else "FAIL",
    }
    (args.evidence_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(audit, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
