#!/usr/bin/env python3
"""Compare archived PD and non-PD greedy results by case and iteration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _results(root: Path) -> dict[tuple[str, int], dict]:
    values: dict[tuple[str, int], dict] = {}
    for path in sorted(root.glob("*.json")):
        if path.name in {"audit.json", "manifest.json", "snapshots.json"}:
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        if "case" not in value or "iteration" not in value:
            continue
        key = (str(value["case"]), int(value["iteration"]))
        if key in values:
            raise ValueError(f"duplicate archived result {key}")
        values[key] = value
    return values


def _digest(value: dict) -> str:
    response = value.get("response", {})
    return str(
        value.get("token_ids_sha256")
        or response.get("_pypto_token_ids_sha256")
        or ""
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pd-suite", required=True, type=Path)
    parser.add_argument("--baseline-suite", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    pd = _results(args.pd_suite)
    baseline = _results(args.baseline_suite)
    errors: list[str] = []
    if set(pd) != set(baseline):
        missing_pd = sorted(set(baseline) - set(pd))
        missing_baseline = sorted(set(pd) - set(baseline))
        if missing_pd:
            errors.append(f"PD results missing {missing_pd}")
        if missing_baseline:
            errors.append(f"baseline results missing {missing_baseline}")

    comparisons = []
    for key in sorted(set(pd) & set(baseline)):
        pd_response = pd[key].get("response", {})
        baseline_response = baseline[key].get("response", {})
        pd_digest = _digest(pd[key])
        baseline_digest = _digest(baseline[key])
        item_errors = []
        if not pd_digest or not baseline_digest:
            item_errors.append("token-ID digest missing")
        elif pd_digest != baseline_digest:
            item_errors.append("token-ID digest mismatch")
        if pd_response.get("usage") != baseline_response.get("usage"):
            item_errors.append("usage mismatch")
        pd_choice = (pd_response.get("choices") or [{}])[0]
        baseline_choice = (baseline_response.get("choices") or [{}])[0]
        if pd_choice.get("finish_reason") != baseline_choice.get("finish_reason"):
            item_errors.append("finish reason mismatch")
        if pd_choice.get("text") != baseline_choice.get("text"):
            item_errors.append("text mismatch")
        errors.extend(f"{key}: {error}" for error in item_errors)
        comparisons.append(
            {
                "case": key[0],
                "iteration": key[1],
                "pd_token_ids_sha256": pd_digest,
                "baseline_token_ids_sha256": baseline_digest,
                "errors": item_errors,
            }
        )

    report = {
        "schema_version": 1,
        "pd_suite": str(args.pd_suite),
        "baseline_suite": str(args.baseline_suite),
        "comparisons": comparisons,
        "errors": errors,
        "result": "PASS" if not errors else "FAIL",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
