# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run a bounded DSpark K7 Host-DFX smoke through an eight- or sixteen-device task.

Use --check-only to validate configuration without allocating devices. Real runs
require TASK_DEVICE from task-submit. This is a smoke, not a performance test.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_deepseek_v4_accuracy import (  # noqa: E402
    _request_json,
    _stop_process_group,
    _unused_local_port,
    _wait_for_health,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dplb_scheduled_validation import validate_scheduled_load  # noqa: E402

MODEL_ID = "dplb-dfx-smoke"


def validate_capture(
    path: Path, expected_groups: int, *, require_prefix_hit: bool = False, require_scheduled_load: bool = False,
) -> dict:
    manifests = sorted(path.glob("dplb/*/*.manifest.json"))
    if len(manifests) != 2:
        raise RuntimeError(f"expected two completed captures, found {len(manifests)}")
    summaries = []
    prefix_hits = 0
    for manifest_path in manifests:
        status = json.loads(manifest_path.read_text())
        data_path = manifest_path.with_name(manifest_path.name.removesuffix(".manifest.json") + ".jsonl")
        events = [json.loads(line) for line in data_path.read_text().splitlines()]
        if not status["complete"] or status["dropped"] or status["error"]:
            raise RuntimeError(f"incomplete capture: {manifest_path}")
        if status["bytes_written"] != data_path.stat().st_size:
            raise RuntimeError("capture byte accounting mismatch")
        if [event["event_seq"] for event in events] != list(range(1, len(events) + 1)):
            raise RuntimeError("capture has sequence gaps")
        if len(status["topology"]) != expected_groups:
            raise RuntimeError("unexpected grouped-cache topology")
        if any(event["capture_id"] != status["capture_id"] for event in events):
            raise RuntimeError("event belongs to a different capture")
        if sum(status["written"].values()) != len(events):
            raise RuntimeError("manifest record accounting mismatch")
        commits = [event for event in events if event["event"] == "placement_commit"]
        prefix_hits += sum(
            event.get("hit_tokens", 0) > 0 for event in events if event["event"] == "prefix_lookup"
        )
        terminals = {event["request_uid"] for event in events if event["event"] == "request_terminal"}
        if not commits or any(event["request_uid"] not in terminals for event in commits):
            raise RuntimeError("placement without an observed terminal event")
        if status["open_request_count"] != 0:
            raise RuntimeError("requests remain open after the completed workload")
        releases = [
            event for event in events if event["event"] == "owner_change" and event["action"] == "released"
        ]
        if not releases or any(
            row["request_count"] or row["sum_logical_tokens"] or row["max_logical_tokens"]
            for row in releases[-1]["groups"]
        ):
            raise RuntimeError("group load did not return to zero")
        scheduled = validate_scheduled_load(events, status["topology"]) if require_scheduled_load else None
        summaries.append(
            dict(
                capture_id=status["capture_id"],
                events=len(events),
                groups=sorted({event["selected_group"] for event in commits}),
                commits=len(commits),
                terminals=len(terminals),
                scheduled_load=scheduled,
            )
        )
    if require_prefix_hit and not prefix_hits:
        raise RuntimeError("no actual prefix hit was recorded")
    return dict(status="PASS_HOST_CAPTURE_SMOKE", captures=summaries, prefix_hits=prefix_hits)


def server_command(args, devices: tuple[int, ...], port: int) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "pypto_serving.cli",
        "--model",
        str(args.model),
        "--served-model-name",
        MODEL_ID,
        "--platform",
        "a2a3",
        "--devices",
        ",".join(map(str, devices)),
        "--dp",
        str(args.ep // 4),
        "--ep",
        str(args.ep),
        "--tp",
        "4",
        "--block-size",
        "32",
        "--max-model-len",
        "32768",
        "--max-num-seqs",
        "8",
        "--max-num-batched-tokens",
        "8192",
        "--long-prefill-token-threshold",
        "128",
        "--ring-heap",
        "2147483648,2147483648,4294967296,8589934592",
        "--speculative-config",
        '{"method":"dspark","num_speculative_tokens":7}',
        "--enable-prefix-caching",
        "--enable-dplb" if args.dplb else "--no-enable-dplb",
        "--profile",
        "--profile-output",
        str(args.output_dir / "profile"),
        "--dplb-dfx-level",
        args.dfx_level,
        "--dplb-dfx-max-seconds",
        "600",
        "--port",
        str(port),
        "--show-startup-logs",
    ]
    return command


def run_workload(process, port: int, output: Path, timeout: int, *, concurrency: int = 4) -> dict:
    deadline = time.monotonic() + timeout

    def post(endpoint, payload):
        return _request_json(
            process, port, deadline, endpoint=endpoint, request_kind="DFX smoke", payload=payload
        )

    def completion(prompt):
        result = post(
            "/v1/completions",
            dict(model=MODEL_ID, prompt=prompt, max_tokens=32, temperature=0.0, ignore_eos=True),
        )
        if (
            not result.get("choices")
            or result.get("usage", {}).get("completion_tokens") != 32
            or result["choices"][0].get("finish_reason") != "length"
        ):
            raise RuntimeError(f"generation contract failed: {result}")
        return result

    # Warm compilation outside the capture; then mix lengths and repeat a prefix.
    completion("Huawei is")
    prompts = [
        "Explain data parallel inference in one sentence.",
        "The Longevity Hill overlooks Kunming Lake. " * 64,
        "Describe a small garden. " * 16,
        "The Seventeen-Arch Bridge joins the island. " * 128,
    ]
    results = []
    measurements = []
    for window in range(2):
        # These control endpoints return an empty 200 response.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for endpoint in ("/start_profile", "/start_profile"):
            with opener.open(
                urllib.request.Request(
                    f"http://127.0.0.1:{port}{endpoint}",
                    data=b"",
                    method="POST",
                ),
                timeout=max(1, deadline - time.monotonic()),
            ) as response:
                if response.status != 200:
                    raise RuntimeError("profile start failed")
        try:
            started = time.monotonic()
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                batch = list(pool.map(completion, prompts))
            # Exact same prefix is sent after its producer has completed.
            batch.append(completion(prompts[1]))
            elapsed = time.monotonic() - started
            measurements.append(dict(
                window=window, elapsed_seconds=elapsed, completed_requests=len(batch),
                output_tokens=32 * len(batch), output_tokens_per_second=32 * len(batch) / elapsed,
            ))
            results.append(batch)
            (output / f"responses-{window}.json").write_text(json.dumps(batch, indent=2) + "\n")
        finally:
            with opener.open(
                urllib.request.Request(
                    f"http://127.0.0.1:{port}/stop_profile",
                    data=b"",
                    method="POST",
                ),
                timeout=max(1, deadline - time.monotonic()),
            ) as response:
                if response.status != 200:
                    raise RuntimeError("profile stop failed")
    return dict(
        completed_requests=sum(map(len, results)), output_tokens=32 * sum(map(len, results)),
        workload_concurrency=concurrency,
        workload_measurements=measurements,
        measurement_scope="smoke_wall_time_after_compilation_warmup_not_formal_overhead",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ep", type=int, choices=(8, 16), default=8)
    # task-submit appends the allocated IDs to the command. The smoke reads
    # TASK_DEVICE, but accepts this queue-owned argument for compatibility.
    parser.add_argument("--device", default=None)
    parser.add_argument("--dplb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dfx-level", choices=("off", "load"), default="load")
    parser.add_argument("--startup-seconds", type=int, default=2400)
    parser.add_argument("--workload-seconds", type=int, default=1200)
    parser.add_argument("--concurrency", type=int, choices=(1, 4), default=4)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    args.model = args.model.resolve()
    args.output_dir = args.output_dir.resolve()
    if not (args.model / "config.json").is_file():
        parser.error("model config.json is missing")
    if args.startup_seconds <= 0 or args.workload_seconds <= 0:
        parser.error("timeouts must be positive")
    raw = os.environ.get("TASK_DEVICE", "")
    devices = tuple(int(value) for value in raw.split(",") if value.strip())
    if args.check_only:
        devices = tuple(range(args.ep))
    elif len(devices) != args.ep or len(set(devices)) != args.ep or min(devices) < 0:
        parser.error(f"run through task-submit with {args.ep} unique devices")
    command = server_command(args, devices, _unused_local_port())
    from pypto_serving.cli.main import build_parser, build_serving_engine_config
    from pypto_serving.model.deepseek_dspark.npu_executor import _find_pypto_lib_dspark_dir
    import pypto_serving

    if Path(pypto_serving.__file__).resolve().parent.parent != ROOT:
        raise RuntimeError("smoke imported a different serving checkout")
    config = build_serving_engine_config(build_parser().parse_args(command[3:]))
    kernel_dir = _find_pypto_lib_dspark_dir()
    evidence = dict(
        command=command,
        devices=(None if args.check_only else devices),
        interpreter=sys.executable,
        serving_source=str(ROOT),
        kernel_dir=str(kernel_dir),
        policy_enabled=config.enable_dplb,
        dfx_level=config.dplb_dfx_config.level,
        workload_concurrency=args.concurrency,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    program_build_dir = None
    if not args.check_only:
        build_base = Path(os.environ.get("PYPTO_PROG_BUILD_DIR") or args.output_dir / "build").resolve()
        build_base.mkdir(parents=True, exist_ok=True)
        # A regenerated kernel table must never be paired with a previous run's
        # source-adjacent orchestration .so, even under a caller-supplied base.
        program_build_dir = tempfile.mkdtemp(prefix="dfx-smoke-", dir=build_base)
    evidence["program_build_dir"] = program_build_dir
    (args.output_dir / "configuration.json").write_text(json.dumps(evidence, indent=2) + "\n")
    if args.check_only:
        print(json.dumps(dict(status="PASS_CONFIGURATION_ONLY", **evidence), indent=2))
        return
    with (args.output_dir / "server.log").open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT), "PYPTO_PROG_BUILD_DIR": program_build_dir},
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        result = dict(status="INCOMPLETE")
        try:
            port = int(command[command.index("--port") + 1])
            _wait_for_health(process, port, time.monotonic() + args.startup_seconds)
            result.update(run_workload(
                process, port, args.output_dir, args.workload_seconds, concurrency=args.concurrency,
            ))
            if args.dfx_level == "load":
                result.update(
                    validate_capture(
                        args.output_dir / "profile", args.ep // 4,
                        require_prefix_hit=True, require_scheduled_load=True,
                    )
                )
            elif list((args.output_dir / "profile").glob("dplb/*/*.jsonl")):
                raise RuntimeError("DFX off wrote a capture")
            else:
                result["status"] = "PASS_GENERATION_DFX_OFF"
        except BaseException as exc:
            result.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            primary_failed = sys.exc_info()[0] is not None
            try:
                _stop_process_group(process)
                result["server_returncode"] = process.poll()
                if process.poll() is None or process.returncode != 0:
                    result["cleanup_error"] = "server did not shut down successfully"
                    if not primary_failed:
                        result["status"] = "FAIL_CLEANUP"
                        raise RuntimeError(result["cleanup_error"])
            finally:
                (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
