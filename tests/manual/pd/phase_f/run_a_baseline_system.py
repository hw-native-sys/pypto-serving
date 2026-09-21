#!/usr/bin/env python3
"""One-command exact-PID non-PD K7 baseline on a selected validation host."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

from run_ab_system import (
    _IDENTITY,
    _copy_container_tree,
    _default_run_id,
    _docker,
    _start,
    _start_suite,
    _stop_with_retries,
    _wait_health,
    _wait_suite,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id",
        default=_default_run_id("nonpd"),
        help="run identity (default: timestamped unique tag)",
    )
    parser.add_argument("--repo", default="/workspace/pypto-serving")
    parser.add_argument("--run-root", default="/workspace/phase-g-runs")
    parser.add_argument(
        "--env-file",
        default=os.environ.get("PYPTO_STACK_ENV_FILE", "/workspace/env_all.sh"),
    )
    parser.add_argument(
        "--model-dir", default="/models/dsv4-flash-0731-dspark-w8a8"
    )
    parser.add_argument("--max-model-len", type=int, default=524288)
    parser.add_argument("--container", default="openeuler-2403-DS")
    parser.add_argument(
        "--ssh-host", default=os.environ.get("PYPTO_PD_BASELINE_SSH", "")
    )
    parser.add_argument("--port", type=int, default=8113)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--cases-file", default="cases_determinism.json")
    parser.add_argument("--startup-timeout-seconds", type=float, default=1800)
    parser.add_argument("--request-timeout-seconds", type=float, default=7200)
    parser.add_argument("--local-evidence-dir", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not args.ssh_host:
        raise ValueError(
            "--ssh-host (or PYPTO_PD_BASELINE_SSH) must select a validation host"
        )
    if not _IDENTITY.fullmatch(args.run_id):
        raise ValueError("run-id contains unsupported characters")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+\.json", args.cases_file):
        raise ValueError("cases-file must be a JSON basename inside phase_f")
    if args.repeat < 1 or args.max_model_len < 1:
        raise ValueError("repeat and max-model-len must be positive")
    for name, value in (
        ("repo", args.repo),
        ("run-root", args.run_root),
        ("env-file", args.env_file),
        ("model-dir", args.model_dir),
    ):
        path = Path(value)
        if not path.is_absolute() or path == Path("/"):
            raise ValueError(f"{name} must be an explicit absolute path")
    if not Path(args.run_root).name.startswith("phase-g-"):
        raise ValueError("run-root basename must start with phase-g-")
    if args.local_evidence_dir.exists():
        raise FileExistsError(args.local_evidence_dir)

    repo = args.repo.rstrip("/")
    run_root = args.run_root.rstrip("/")
    service_name = f"{args.run_id}-baseline"
    suite_name = f"{args.run_id}-suite"
    control_name = f"{args.run_id}-suite-control"
    stop_records: list[dict[str, object]] = []
    result = "FAIL"
    try:
        _docker(
            args.ssh_host,
            args.container,
            ["mkdir", "-p", run_root],
            timeout=30,
        )
        _start(
            args.ssh_host,
            args.container,
            f"{repo}/tests/manual/pd/phase_f/run_non_pd_k7_node.sh",
            {
                "BASELINE_RUN_ID": args.run_id,
                "BASELINE_API_PORT": args.port,
                "BASELINE_EVIDENCE_NAME": service_name,
                "BASELINE_EVIDENCE_ROOT": run_root,
                "PYPTO_STACK_ENV_FILE": args.env_file,
                "PYPTO_DSV4_DSPARK_MODEL_DIR": args.model_dir,
                "PYPTO_MAX_MODEL_LEN": args.max_model_len,
            },
        )
        _wait_health(
            args.ssh_host,
            args.container,
            f"http://127.0.0.1:{args.port}/health",
            args.startup_timeout_seconds,
            expected={},
        )
        suite_args = [
            "--url",
            f"http://127.0.0.1:{args.port}",
            "--evidence-dir",
            f"{run_root}/{suite_name}",
            "--repeat",
            str(args.repeat),
            "--cases",
            f"{repo}/tests/manual/pd/phase_f/{args.cases_file}",
        ]
        _start_suite(
            args.ssh_host,
            args.container,
            repo,
            f"{run_root}/{control_name}",
            "run_non_pd_suite.py",
            suite_args,
        )
        _wait_suite(
            args.ssh_host,
            args.container,
            f"{run_root}/{control_name}",
            args.request_timeout_seconds,
        )
        result = "PASS"
    finally:
        stop_records.append(
            _stop_with_retries(
                args.ssh_host,
                args.container,
                f"{run_root}/{control_name}/suite-service.pid",
            )
        )
        stop_records.append(
            _stop_with_retries(
                args.ssh_host,
                args.container,
                f"{run_root}/{service_name}/service.pid",
            )
        )
        args.local_evidence_dir.mkdir(parents=True, exist_ok=False)
        for label, name in (
            ("service", service_name),
            ("suite", suite_name),
            ("suite-control", control_name),
        ):
            try:
                _copy_container_tree(
                    args.ssh_host,
                    args.container,
                    f"{run_root}/{name}",
                    args.local_evidence_dir / label,
                )
            except Exception as exc:  # Preserve the primary failure evidence.
                (args.local_evidence_dir / f"{label}-copy-error.txt").write_text(
                    f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
                )
        (args.local_evidence_dir / "orchestration.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "run_id": args.run_id,
                    "result": result,
                    "stops": stop_records,
                    "finished_ns": time.time_ns(),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    allowed_states = {"absent", "stopped", "already-stopped"}
    return 0 if result == "PASS" and all(
        record["state"] in allowed_states for record in stop_records
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
