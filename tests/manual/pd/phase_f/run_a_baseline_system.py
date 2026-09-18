#!/usr/bin/env python3
"""One-command exact-PID non-PD K7 baseline on serving-a."""

from __future__ import annotations

import argparse
import json
import secrets
import time
from pathlib import Path

from run_ab_system import (
    _IDENTITY,
    _docker,
    _run,
    _start,
    _start_suite,
    _stop_with_retries,
    _wait_health,
    _wait_suite,
    _write_secret_file,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", default="/home/sj/git/phase-f-system-20260913")
    parser.add_argument("--container", default="openeuler-2403-DS")
    parser.add_argument("--ssh-host", default="serving-a")
    parser.add_argument("--port", type=int, default=8113)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--cases-file", default="")
    parser.add_argument("--verify-drafter-scrub", action="store_true")
    parser.add_argument("--reset-persistent-windows", action="store_true")
    parser.add_argument("--delay-drafter-lease-reuse", action="store_true")
    parser.add_argument("--startup-timeout-seconds", type=float, default=1800)
    parser.add_argument("--request-timeout-seconds", type=float, default=7200)
    parser.add_argument("--local-evidence-dir", required=True, type=Path)
    args = parser.parse_args()
    if not _IDENTITY.fullmatch(args.run_id):
        raise ValueError("run-id contains unsupported characters")
    if not args.stage.startswith("/home/sj/git/phase-f-"):
        raise ValueError("stage must be an explicit /home/sj/git/phase-f-* directory")
    if args.local_evidence_dir.exists():
        raise FileExistsError(args.local_evidence_dir)

    repo = f"{args.stage}/pypto-serving"
    secret_file = f"{args.stage}/.{args.run_id}.secrets"
    service_name = f"{args.run_id}-baseline"
    suite_name = f"{args.run_id}-suite"
    control_name = f"{args.run_id}-suite-control"
    stop_records = []
    result = "FAIL"
    try:
        _write_secret_file(
            args.ssh_host,
            args.container,
            secret_file,
            secrets.token_hex(32),
            secrets.token_hex(32),
        )
        _start(
            args.ssh_host,
            args.container,
            secret_file,
            f"{repo}/tests/manual/pd/phase_f/run_non_pd_k7_node.sh",
            {
                "BASELINE_RUN_ID": args.run_id,
                "BASELINE_API_PORT": args.port,
                "BASELINE_EVIDENCE_NAME": service_name,
                "PYPTO_DSPARK_VERIFY_DRAFTER_SCRUB": int(
                    args.verify_drafter_scrub
                ),
                "PYPTO_DSPARK_RESET_PERSISTENT_WINDOWS": int(
                    args.reset_persistent_windows
                ),
                "PYPTO_DSPARK_DELAY_DRAFTER_LEASE_REUSE": int(
                    args.delay_drafter_lease_reuse
                ),
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
            f"{args.stage}/{suite_name}",
            "--repeat",
            str(args.repeat),
        ]
        if args.cases_file:
            if Path(args.cases_file).name != args.cases_file or not args.cases_file.endswith(
                ".json"
            ):
                raise ValueError("cases-file must be a JSON basename inside phase_f")
            suite_args.extend(
                ["--cases", f"{repo}/tests/manual/pd/phase_f/{args.cases_file}"]
            )
        _start_suite(
            args.ssh_host,
            args.container,
            secret_file,
            repo,
            f"{args.stage}/{control_name}",
            "run_non_pd_suite.py",
            suite_args,
        )
        _wait_suite(
            args.ssh_host,
            args.container,
            f"{args.stage}/{control_name}",
            args.request_timeout_seconds,
        )
        result = "PASS"
    finally:
        stop_records.append(
            _stop_with_retries(
                args.ssh_host,
                args.container,
                f"{args.stage}/{control_name}/suite-service.pid",
            )
        )
        stop_records.append(
            _stop_with_retries(
                args.ssh_host,
                args.container,
                f"{args.stage}/{service_name}/service.pid",
            )
        )
        try:
            _docker(args.ssh_host, args.container, ["rm", "-f", secret_file], timeout=15)
        except Exception:
            pass
        args.local_evidence_dir.mkdir(parents=True, exist_ok=False)
        for label, name in (
            ("service", service_name),
            ("suite", suite_name),
            ("suite-control", control_name),
        ):
            try:
                _run(
                    [
                        "scp",
                        "-r",
                        f"{args.ssh_host}:{args.stage}/{name}",
                        str(args.local_evidence_dir / label),
                    ],
                    timeout=600,
                )
            except Exception as exc:
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
    return 0 if result == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
