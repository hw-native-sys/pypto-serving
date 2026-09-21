# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise deterministic cancellation and uncertain fixed-pair fail-close."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import time
import urllib.error
import urllib.request

from run_suite import _get, _request


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-url", required=True)
    parser.add_argument("--prefill-url", required=True)
    parser.add_argument("--decode-url", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--scenario",
        choices=("cancel-after-output", "kill-prefill"),
        required=True,
    )
    parser.add_argument("--prefill-python-pid-file", default="")
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    return parser


def _optional_get(url: str) -> dict[str, object]:
    try:
        return {"ok": True, "value": _get(url, 15)}
    except BaseException as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _snapshot(args) -> dict[str, object]:
    return {
        "timestamp_ns": time.time_ns(),
        "router_recovery": _optional_get(
            f"{args.router_url.rstrip('/')}/recovery"
        ),
        "router_metrics": _optional_get(f"{args.router_url.rstrip('/')}/metrics"),
        "prefill_capacity": _optional_get(
            f"{args.prefill_url.rstrip('/')}/internal/pd/capacity"
        ),
        "decode_capacity": _optional_get(
            f"{args.decode_url.rstrip('/')}/internal/pd/capacity"
        ),
    }


def _wait_until(predicate, timeout: float, description: str) -> None:
    deadline = time.monotonic() + timeout
    last = "not sampled"
    while time.monotonic() < deadline:
        try:
            value = predicate()
            last = repr(value)
            if value:
                return
        except BaseException as exc:
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(0.25)
    raise TimeoutError(f"timed out waiting for {description}: {last}")


def _open_then_cancel(router_url: str, timeout: float) -> dict[str, object]:
    payload = {
        "model": "dsv4-flash-dspark-w8a8",
        "prompt": "紫禁城",
        "max_tokens": 512,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": True,
    }
    request = urllib.request.Request(
        f"{router_url.rstrip('/')}/v1/completions",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        method="POST",
        headers={"content-type": "application/json"},
    )
    response = urllib.request.urlopen(request, timeout=timeout)
    first = b""
    try:
        while True:
            line = response.readline()
            if not line:
                raise EOFError("stream ended before the first SSE event")
            if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                first = line
                break
    finally:
        response.close()
    return {
        "first_event": first.decode(errors="replace").strip(),
        "closed_ns": time.time_ns(),
    }


def _capacity_is_zero(url: str) -> bool:
    value = _get(url, 15)
    return not any(
        value[name]
        for name in (
            "active_handoffs",
            "prepared_requests",
            "reservations",
            "quarantined_reservations",
            "inflight_transfer_bytes",
        )
    )


def _run_cancel(args) -> tuple[list[str], dict[str, object]]:
    details = {"cancel": _open_then_cancel(args.router_url, args.timeout_seconds)}
    _wait_until(
        lambda: _get(f"{args.router_url.rstrip('/')}/recovery", 15)["phase"]
        == "RUNNING",
        60,
        "Router to remain RUNNING after deterministic cancellation",
    )
    _wait_until(
        lambda: _capacity_is_zero(
            f"{args.prefill_url.rstrip('/')}/internal/pd/capacity"
        ),
        120,
        "Prefill resources to drain after cancellation",
    )
    _wait_until(
        lambda: _capacity_is_zero(
            f"{args.decode_url.rstrip('/')}/internal/pd/capacity"
        ),
        120,
        "Decode resources to drain after cancellation",
    )
    errors = []
    try:
        followup = _request(
            f"{args.router_url.rstrip('/')}/v1/completions",
            {
                "model": "dsv4-flash-dspark-w8a8",
                "prompt": "紫禁城",
                "max_tokens": 32,
                "temperature": 0.0,
                "top_p": 1.0,
                "stream": False,
            },
            args.timeout_seconds,
        )
    except urllib.error.HTTPError as exc:
        details["followup_error"] = {
            "status": exc.code,
            "body": exc.read(4096).decode(errors="replace"),
        }
        errors.append(f"follow-up request failed with HTTP {exc.code}")
    else:
        details["followup"] = followup
        if followup.get("usage", {}).get("completion_tokens") != 32:
            errors.append("follow-up request did not complete 32 tokens")
        if followup.get("choices", [{}])[0].get("finish_reason") != "length":
            errors.append("follow-up request finish reason is not length")
    return errors, details


def _read_owned_prefill_pid(path: str) -> tuple[int, str]:
    if not path or not Path(path).is_absolute():
        raise ValueError("kill-prefill requires an absolute Prefill python PID file")
    value = Path(path).read_text(encoding="utf-8").strip()
    if not value.isdigit() or int(value) < 2:
        raise RuntimeError("Prefill python PID file is invalid")
    pid = int(value)
    cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    if "pypto_serving.cli" not in cmdline or "--pd-role prefill" not in cmdline:
        raise RuntimeError("PID file does not identify the owned Prefill service")
    return pid, cmdline


def _run_kill_prefill(args) -> tuple[list[str], dict[str, object]]:
    payload = {
        "model": "dsv4-flash-dspark-w8a8",
        "prompt": [100] * 8192,
        "max_tokens": 128,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False,
    }
    details: dict[str, object] = {}
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            _request,
            f"{args.router_url.rstrip('/')}/v1/completions",
            payload,
            args.timeout_seconds,
        )
        _wait_until(
            lambda: _get(
                f"{args.prefill_url.rstrip('/')}/internal/pd/capacity", 15
            )["active_handoffs"]
            > 0,
            300,
            "Prefill handoff to become active",
        )
        pid, cmdline = _read_owned_prefill_pid(args.prefill_python_pid_file)
        details["fault"] = {
            "pid": pid,
            "cmdline": cmdline,
            "signal": "SIGKILL",
            "injected_ns": time.time_ns(),
        }
        os.kill(pid, signal.SIGKILL)
        try:
            details["request_result"] = future.result(timeout=120)
        except BaseException as exc:
            details["request_error"] = f"{type(exc).__name__}: {exc}"

    _wait_until(
        lambda: _get(f"{args.router_url.rstrip('/')}/recovery", 15)["phase"]
        == "RECOVERY_REQUIRED",
        120,
        "Router fail-closed recovery gate",
    )
    errors = []
    if "request_error" not in details:
        errors.append("faulted request unexpectedly completed")
    return errors, details


def main() -> int:
    args = _parser().parse_args()
    if args.concurrency != 1:
        raise ValueError("fault scenarios require concurrency=1")
    evidence = Path(args.evidence_dir)
    if evidence.exists():
        raise FileExistsError(f"evidence directory already exists: {evidence}")
    evidence.mkdir(parents=True)
    before = _snapshot(args)
    started_ns = time.time_ns()
    if args.scenario == "cancel-after-output":
        errors, details = _run_cancel(args)
    else:
        errors, details = _run_kill_prefill(args)
    after = _snapshot(args)
    audit = {
        "schema_version": 1,
        "scenario": args.scenario,
        "started_ns": started_ns,
        "finished_ns": time.time_ns(),
        "before": before,
        "after": after,
        "details": details,
        "errors": errors,
        "result": "PASS" if not errors else "FAIL",
    }
    (evidence / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
