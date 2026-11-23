# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Collect or reprocess a lowlat TP8/K5 native-pipeline Host profile."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def request_json(base, path, payload=None, timeout=1800):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    # Local profiling must not route loopback requests through a proxy.
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=timeout) as response:
        body = response.read()
        return json.loads(body) if body else {}


def completion(base, args, prompt_tokens):
    payload = {
        "prompt": args.prompt,
        "max_tokens": args.output_tokens,
        "temperature": 0,
        "top_p": 1,
        "stream": True,
    }
    req = urllib.request.Request(
        base + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start, first, last, usage, finish, done = time.perf_counter(), None, None, None, None, False
    pieces = []
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
        req, timeout=args.request_timeout
    ) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            if line[6:] == "[DONE]":
                done = True
                break
            chunk = json.loads(line[6:])
            if "error" in chunk:
                raise RuntimeError(chunk["error"])
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                text = choice.get("text", "")
                if text:
                    last = time.perf_counter()
                    first = first if first is not None else last
                    pieces.append(text)
                finish = choice.get("finish_reason") or finish
    end = time.perf_counter()
    if (
        not done
        or first is None
        or finish != "length"
        or usage is None
        or usage.get("prompt_tokens") != prompt_tokens
        or usage.get("completion_tokens") != args.output_tokens
    ):
        raise RuntimeError(f"incomplete lowlat response: done={done}, finish={finish}, usage={usage}")
    return dict(
        text="".join(pieces),
        usage=usage,
        finish_reason=finish,
        ttft_ms=(first - start) * 1000,
        wall_ms=(end - start) * 1000,
        tpot_ms=(last - first) * 1000 / (args.output_tokens - 1),
        timing_scope="HTTP first-to-last nonempty streamed delta; excludes final usage/DONE tail",
    )


def server_command(args, devices, port):
    entry = (
        [str(Path(__file__).with_name("lowlat_prefix_audit.py"))]
        if getattr(args, "audit_prefix_state", False)
        else ["-m", "pypto_serving.cli"]
    )
    return [
        sys.executable,
        *entry,
        "--model",
        str(args.model_dir.resolve()),
        "--platform",
        "a2a3",
        "--devices",
        ",".join(map(str, devices)),
        "--dp",
        "1",
        "--tp",
        "8",
        "--ep",
        "1",
        "--lm-head-tp",
        "8",
        "--speculative-config",
        '{"method":"dspark","num_speculative_tokens":5}',
        "--max-num-seqs",
        "1",
        "--block-size",
        "128",
        "--max-model-len",
        str(getattr(args, "max_model_len", 16384)),
        "--max-num-batched-tokens",
        str(getattr(args, "prefill_chunk_size", 512)),
        "--long-prefill-token-threshold",
        str(getattr(args, "prefill_chunk_size", 512)),
        "--npu-memory-utilization",
        str(getattr(args, "npu_memory_utilization", 0.9)),
        "--enable-prefix-caching"
        if getattr(args, "prefix_cache_mode", "off") == "hot"
        else "--no-enable-prefix-caching",
        "--enable-chunked-prefill",
        "--ring-heap",
        getattr(args, "ring_heap", "1073741824,268435456,1073741824,1073741824"),
        "--ring-task-window",
        "131072",
        "--ring-dep-pool",
        "131072",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--show-startup-logs",
        "--profile",
        "--profile-level",
        "verbose",
        "--profile-output",
        str(args.artifact_dir / "serving-trace"),
    ]


def source_fingerprint():
    serving = next(
        parent for parent in Path(__file__).resolve().parents if (parent / "pypto_serving").is_dir()
    )
    lib = Path(os.environ.get("PYPTO_LIB_ROOT", serving / "pypto-lib"))
    result = {}
    for label, directory in (("serving", serving / "pypto_serving"), ("lib", lib / "models")):
        for path in sorted(directory.rglob("*.py")):
            result[label + "/" + str(path.relative_to(directory))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return result


def run(args):
    devices = [int(value) for value in args.devices.split(",") if value.strip()]
    if len(devices) != 8 or len(set(devices)) != 8 or min(devices) < 0:
        raise ValueError("lowlat profiling requires eight distinct assigned devices")
    assigned = os.environ.get("TASK_DEVICE")
    if assigned and devices != [int(value) for value in assigned.split(",")]:
        raise ValueError("--devices differs from TASK_DEVICE allocation/order")
    if (
        not args.model_dir.is_dir()
        or not 25 <= args.output_tokens <= 2048
        or type(args.warmup_count) is not int
        or not 1 <= args.warmup_count <= 2
    ):
        raise ValueError("real model directory, output budget [25,2048] and warmup-count [1,2] required")
    if not 0 < args.npu_memory_utilization < 1 or (
        args.prefix_cache_mode == "hot" and args.warmup_count != 2
    ):
        raise ValueError("memory utilization must be in (0,1); hot profiles require two unprofiled requests")
    if getattr(args, "audit_prefix_state", False) and (
        args.benchmark_only or args.prefix_cache_mode != "hot"
    ):
        raise ValueError("Prefix state audit requires hot profiling and cannot measure benchmark TPOT")
    args.artifact_dir = args.artifact_dir.resolve()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    if any(args.artifact_dir.iterdir()):
        raise ValueError("use a fresh empty artifact directory")
    from pypto_serving.model.tokenizer import load_tokenizer

    from pypto_serving.model.deepseek_lowlat.capacity import LowlatCapacityPlan

    LowlatCapacityPlan(
        args.max_model_len, args.prefill_chunk_size, "rolling", args.prefix_cache_mode == "hot"
    )
    if args.prompt_file is not None:
        args.prompt = args.prompt_file.read_text(encoding="utf-8")
    ids = load_tokenizer(args.model_dir).encode(args.prompt)
    if not ids or len(ids) + args.output_tokens + 6 > args.max_model_len:
        raise ValueError("prompt/output budget exceeds native context/lookahead")
    boundary = max(0, len(ids) - 128) // 128 * 128
    if args.prefix_cache_mode == "hot" and not boundary:
        raise ValueError("hot profiling requires a cacheable prefix and 128-token replay")
    if args.warmup_prompt is not None and args.prefix_cache_mode != "off":
        raise ValueError("A different warmup prompt requires prefix caching disabled")
    warmup_args = copy.copy(args)
    warmup_tokens = len(ids)
    if args.warmup_prompt is not None:
        warmup_args.prompt = args.warmup_prompt
        warmup_args.output_tokens = min(64, args.output_tokens)
        warmup_tokens = len(load_tokenizer(args.model_dir).encode(warmup_args.prompt))
    before = source_fingerprint()
    write_json(args.artifact_dir / "source-manifest.json", before)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = server_command(args, devices, port)
    write_json(args.artifact_dir / "command.json", command)
    env = dict(
        os.environ,
        PYPTO_RUNTIME_LOG="error" if args.benchmark_only else "v9",
        # The audit entry imports runtime before CLI/worker logging setup.
        # Keep its final threshold explicit regardless of that import order.
        PYPTO_SIMPLER_LOG_LEVEL="ERROR" if args.benchmark_only else "INFO",
        SIMPLER_DEVICE_STRACE_ENABLE="0",
        PYPTO_DISABLE_DEVICE_LOG="1",
        PYPTO_CACHE="0",
        PYPTO_PROG_BUILD_DIR=str(args.artifact_dir / "build"),
        ASCEND_PROCESS_LOG_PATH=str(args.artifact_dir / "ascend"),
    )
    if getattr(args, "audit_prefix_state", False):
        env["LOWLAT_PREFIX_AUDIT_DIR"] = str(args.artifact_dir / "prefix-audit")
    else:
        env.pop("LOWLAT_PREFIX_AUDIT_DIR", None)
    for key, value in {
        "SIMPLER_OP_EXECUTE_TIMEOUT_US": "400000000",
        "SIMPLER_STREAM_SYNC_TIMEOUT_MS": "440000",
        "SIMPLER_SCHEDULER_TIMEOUT_MS": "320000",
        "PYPTO_WORKER_INIT_TIMEOUT": "7200",
        "SERVING_WORKER_STEP_TIMEOUT": "1800",
    }.items():
        env.setdefault(key, value)
    receipt = dict(
        mode="benchmark" if args.benchmark_only else "profile",
        warmup_prompt_tokens=warmup_tokens,
        warmup_output_tokens=warmup_args.output_tokens,
        reference_matches_workload=args.warmup_prompt is None or warmup_args.prompt == args.prompt
        and warmup_args.output_tokens == args.output_tokens,
        status="FAIL",
        run_id=args.run_id,
        devices=devices,
        prompt_ids=ids,
        output_tokens=args.output_tokens,
        topology="DP1/TP8/EP1/B1/K5/rolling/pipeline2/launch1",
        reference_response_files=(
            ["warmup-responses.json"]
            if args.warmup_count == 1
            else ["warmup-responses.json", "baseline-responses.json"]
        ),
        npu_memory_utilization=args.npu_memory_utilization,
        device_strace=False,
        compile_cache=False,
        max_model_len=args.max_model_len,
        prefill_chunk_size=args.prefill_chunk_size,
        prefix_cache_mode=args.prefix_cache_mode,
        expected_prefix_hit_tokens=boundary if args.prefix_cache_mode == "hot" else 0,
        prompt_sha256=hashlib.sha256(args.prompt.encode()).hexdigest(),
    )
    process, error = None, None
    base = f"http://127.0.0.1:{port}"
    try:
        with (args.artifact_dir / "server.log").open("w") as log:
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True
            )
            deadline = time.monotonic() + args.startup_timeout
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"lowlat server startup exited {process.returncode}")
                try:
                    if request_json(base, "/health", timeout=2).get("status") == "ok":
                        break
                except (OSError, ValueError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError("lowlat startup timeout")
                time.sleep(1)
            responses = {}
            for index in range(args.warmup_count):
                name = "warmup" if args.warmup_count == 1 else ("warmup" if index == 0 else "baseline")
                response = completion(base, warmup_args, warmup_tokens)
                responses[name] = response
                write_json(args.artifact_dir / (name + "-responses.json"), response)
                print(name, response["tpot_ms"], flush=True)
            if args.benchmark_only:
                response = completion(base, args, len(ids))
                write_json(args.artifact_dir / "responses.json", response)
            else:
                request_json(base, "/start_profile", {})
                try:
                    response = completion(base, args, len(ids))
                    write_json(args.artifact_dir / "responses.json", response)
                finally:
                    request_json(base, "/stop_profile", {}, timeout=180)
            measurement_label = "benchmark" if args.benchmark_only else "profile"
            print(measurement_label, response["tpot_ms"], flush=True)
            receipt["measurements"] = {
                name: {
                    key: value
                    for key, value in item.items()
                    if key in ("ttft_ms", "wall_ms", "tpot_ms", "usage")
                }
                for name, item in (*responses.items(), (measurement_label, response))
            }
            if args.prefix_cache_mode == "hot":
                import re

                rows = [
                    json.loads(m.group(1))
                    for m in re.finditer(
                        r"LOWLAT_PREFIX_RESTORE (\{[^\n]*\})", (args.artifact_dir / "server.log").read_text()
                    )
                ]
                if len(rows) != 2 or any(r["hit_tokens"] != boundary for r in rows):
                    raise RuntimeError("baseline/profile did not both restore the expected prefix")
                warmup = json.loads((args.artifact_dir / "warmup-responses.json").read_text())
                baseline = json.loads((args.artifact_dir / "baseline-responses.json").read_text())
                if not warmup["text"] == baseline["text"] == response["text"]:
                    raise RuntimeError("cold/hot profile responses differ")
                receipt["prefix_restores"] = rows
            if getattr(args, "audit_prefix_state", False):
                from lowlat_prefix_audit import validate

                receipt["prefix_state_audit"] = validate(args.artifact_dir / "prefix-audit")
                receipt["timing_scope"] = "Diagnostic readbacks enabled; do not report these as TPOT benchmarks"
            receipt["status"] = "PASS"
    except BaseException as exc:
        error = exc
        receipt["error"] = str(exc)
    finally:
        if process is not None:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=55)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=30)
            receipt["server_exit"] = process.returncode
        after = source_fingerprint()
        receipt["source_unchanged"] = before == after
        if before != after:
            receipt["status"] = "FAIL"
            error = RuntimeError("profiled sources changed during execution")
        write_json(args.artifact_dir / "run-summary.json", receipt)
    if error is not None:
        raise error
    if not args.benchmark_only:
        from lowlat_trace import analyze

        analyze(args.artifact_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    collect = subs.add_parser("run", help="run HTTP warmup, baseline and formal profile inside an allocation")
    collect.add_argument("--model-dir", type=Path, required=True)
    collect.add_argument("--devices", default=os.environ.get("TASK_DEVICE", ""))
    collect.add_argument("--artifact-dir", type=Path, required=True)
    collect.add_argument("--run-id", default=os.environ.get("TASK_ID", "unknown"))
    collect.add_argument("--prompt", default="Huawei is" * 32)
    collect.add_argument("--output-tokens", type=int, default=64)
    collect.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Measure one unprofiled request without collecting STRACE",
    )
    collect.add_argument("--warmup-prompt", help="Short warmup prompt for benchmark-only mode")
    collect.add_argument(
        "--warmup-count",
        type=int,
        default=2,
        choices=(1, 2),
        help="Unprofiled requests before the formal profile",
    )
    collect.add_argument("--npu-memory-utilization", type=float, default=0.9)
    collect.add_argument("--max-model-len", type=int, default=16384)
    collect.add_argument("--prefill-chunk-size", type=int, default=512)
    collect.add_argument("--prefix-cache-mode", choices=("off", "hot"), default="off")
    collect.add_argument("--audit-prefix-state", action="store_true",
                         help="Read back cold/hot prefix pages and native seeds in a separate diagnostic run")
    collect.add_argument("--prompt-file", type=Path, help="UTF-8 prompt file for long-context requests")
    collect.add_argument("--ring-heap", default="1073741824,268435456,1073741824,1073741824")
    collect.add_argument("--startup-timeout", type=float, default=7200)
    collect.add_argument("--request-timeout", type=float, default=1800)
    offline = subs.add_parser("analyze", help="rebuild and validate traces without NPU access")
    offline.add_argument("artifact_dir", type=Path)
    args = parser.parse_args()
    if args.action == "run":
        run(args)
    else:
        from lowlat_trace import analyze

        analyze(args.artifact_dir)


if __name__ == "__main__":
    main()
