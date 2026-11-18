# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Run a bounded DSpark PD handoff and EPLB check on local or existing nodes.

Run inside a task-submit allocation of twice the EP device count. Each mode starts P and
D once, reuses the nodes across requests, and saves responses and node logs. Existing-node
checks start only a Router and can run without a device allocation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def json_request(url, payload=None, timeout=300):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with OPENER.open(request, timeout=timeout) as response:
        return json.load(response)


def stream_request(url, payload, *, cancel_after_first=False):
    started = time.monotonic()
    first = last = None
    pieces, usage, request_id, finish_reason = [], None, None, None
    request = urllib.request.Request(url, data=json.dumps({**payload, "stream": True}).encode(),
                                     headers={"Content-Type": "application/json"})
    with OPENER.open(request, timeout=900) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            value = json.loads(data)
            if "error" in value:
                raise RuntimeError(f"stream generation failed: {value['error']}")
            request_id = value.get("id", request_id)
            if value.get("usage") is not None:
                usage = value["usage"]
            for choice in value.get("choices", []):
                if choice.get("text"):
                    last = time.monotonic()
                    first = first or last
                    pieces.append(choice["text"])
                    if cancel_after_first:
                        return dict(id=request_id, text="".join(pieces))
                finish_reason = choice.get("finish_reason") or finish_reason
    if not usage or finish_reason == "error" or not pieces:
        raise RuntimeError("stream lacks a successful completion and authoritative usage")
    tokens = usage["completion_tokens"]
    return dict(id=request_id, choices=[dict(text="".join(pieces), finish_reason=finish_reason)], usage=usage,
                ttft_seconds=first - started,
                tpot_seconds=(last - first) / (tokens - 1) if tokens > 1 else None)


def check_cancellation(router, urls, prompt):
    partial = stream_request(router + "/v1/completions", dict(model="pd-eplb", prompt=prompt,
        max_tokens=128, temperature=0, seed=0), cancel_after_first=True)
    request_id = partial["id"]
    deadline = time.monotonic() + 60
    while True:
        statuses = [json_request(url + "/internal/pd/requests/" + request_id) for url in urls]
        if all(not status["entry_present"] for status in statuses):
            if statuses[0]["source_pin"] or statuses[1]["reservation"]["present"]:
                raise AssertionError("cancelled PD request retained a source pin or destination reservation")
            return dict(request=request_id, status=statuses)
        if time.monotonic() >= deadline:
            raise TimeoutError(f"cancelled PD request did not retire: {statuses}")
        time.sleep(.1)


def free_port():
    with socket.socket() as channel:
        channel.bind(("127.0.0.1", 0))
        return channel.getsockname()[1]


def latency_summary(values):
    values = sorted(value for value in values if value is not None)
    if not values:
        return dict(count=0)
    result = dict(count=len(values), mean=sum(values) / len(values), maximum=values[-1])
    for percentile in (50, 95, 99):
        result[f"p{percentile}"] = values[math.ceil(percentile * len(values) / 100) - 1]
    return result


def stop(process):
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(60)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(10)


def wait_ready(processes, urls, deadline):
    pending = set(urls)
    next_report = 0.0
    while pending:
        for process in processes:
            if process.poll() is not None:
                raise RuntimeError(f"server exited during startup: {process.returncode}")
        for url in tuple(pending):
            try:
                if json_request(url + "/health", timeout=2).get("status") == "ok":
                    pending.remove(url)
            except (OSError, ValueError):
                pass
        if time.monotonic() >= deadline:
            raise TimeoutError("PD server readiness deadline")
        if pending:
            if time.monotonic() >= next_report:
                print("Waiting for PD servers: " + ", ".join(sorted(pending)), flush=True)
                next_report = time.monotonic() + 30
            time.sleep(1)


def run_mode(args, devices, mode, reference):
    directory = args.output / mode
    directory.mkdir(parents=True, exist_ok=False)
    ports = [free_port() for _ in range(5)]
    replicas = [dict(name=role, role=role, host="127.0.0.1", port=ports[index],
                     control_port=ports[index + 3], transfer_hostname=args.transfer_hostname)
                for index, role in enumerate(("prefill", "decode"))]
    config = directory / "pd.json"
    config.write_text(json.dumps(dict(replicas=replicas, pd=dict(handoff_timeout=300, request_timeout=900),
                                     observability=dict(enabled=True, root=str(directory / "pd-log")))))
    processes, files = [], []
    p_enabled = mode in ("prefill", "both", "frozen")
    d_enabled = mode in ("decode", "both", "frozen")
    try:
        for index, role in enumerate(("prefill", "decode")):
            command = [sys.executable, "-m", "pypto_serving.cli", "--model", str(args.model),
                "--served-model-name", "pd-eplb", "--platform", "a2a3", "--backend", "npu",
                "--devices", ",".join(map(str, devices[index * args.ep_size:(index + 1) * args.ep_size])),
                "--tp", str((args.prefill_tp, args.decode_tp)[index]),
                "--dp", str(args.ep_size // (args.prefill_tp, args.decode_tp)[index]),
                "--ep", str(args.ep_size), "--lm-head-tp", "4", "--block-size", "32",
                "--max-model-len", "2048", "--max-num-seqs", "4", "--max-num-batched-tokens", "1024",
                "--long-prefill-token-threshold", "128", "--enable-prefix-caching",
                "--speculative-config", '{"method":"dspark","num_speculative_tokens":7}',
                "--ring-heap", args.ring_heap or (args.prefill_ring_heap, args.decode_ring_heap)[index],
                "--pd-config", str(config), "--pd-role", role,
                "--port", str(ports[index]), "--show-startup-logs"]
            if (p_enabled, d_enabled)[index]:
                options = dict(num_redundant_experts=args.ep_size if index == 0 else 0,
                    max_migration_bytes=1 << 34, window_size=2, step_interval=2,
                    min_relative_gain=0, cooldown_seconds=0, max_sample_age_seconds=300,
                    maintenance_timeout_seconds=1, amortization_steps=1000000)
                if mode == "frozen":
                    options["max_rebalances"] = 1
                command.extend(("--enable-eplb", "--eplb-config", json.dumps(options)))
            log = (directory / (role + ".log")).open("w")
            files.append(log)
            processes.append(subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True))
        urls = [f"http://127.0.0.1:{port}" for port in ports[:2]]
        return check_nodes(args, directory, urls, config, mode, reference, processes, files)
    finally:
        # Stop Router, then senders, then receivers so peer registrations can retire.
        for process in processes[2:] + processes[:2]:
            stop(process)
        for stream in files:
            stream.close()


def check_nodes(args, directory, urls, config, mode, reference, processes, files):
    p_enabled = mode in ("prefill", "both", "frozen")
    d_enabled = mode in ("decode", "both", "frozen")
    wait_ready(processes, urls, time.monotonic() + args.startup_timeout)
    descriptors = [json_request(url + "/internal/pd/descriptor") for url in urls]
    node_logs = {role: getattr(args, role + "_log") or directory / (role + ".log")
                 for role in ("prefill", "decode")}
    log_offsets = {role: path.stat().st_size for role, path in node_logs.items()}
    if args.collect_metrics:
        (directory / "metrics_before.json").write_text(json.dumps(
            [json_request(url + "/metrics/json") for url in urls], indent=2))
    router_port = free_port()
    router_log = (directory / "router.log").open("w")
    files.append(router_log)
    processes.append(subprocess.Popen([sys.executable, "-m", "pypto_serving.router.app",
        "--replicas", str(config), "--port", str(router_port)], cwd=ROOT,
        stdout=router_log, stderr=subprocess.STDOUT, start_new_session=True))
    router = f"http://127.0.0.1:{router_port}"
    wait_ready(processes, [router], time.monotonic() + 60)
    prompts = ["Explain why the sky looks blue in a few sentences.",
               "What is the capital of France? Answer briefly.",
               "Continue this explanation: " + "A cache stores recently used information. " * 40]
    responses, samples = [], []
    def generate(index):
        started = time.monotonic()
        payload = dict(model="pd-eplb", prompt=prompts[index % len(prompts)],
                       max_tokens=args.max_tokens, temperature=0, seed=0)
        streamed = args.exercise_lifecycle and index >= len(prompts)
        result = (stream_request(router + "/v1/completions", payload) if streamed
                  else json_request(router + "/v1/completions", payload, timeout=900))
        if "error" in result or not result.get("choices"):
            raise RuntimeError(f"generation failed: {result}")
        text = result["choices"][0]["text"]
        if not text or result["choices"][0].get("finish_reason") == "error":
            raise RuntimeError("empty or failed generation")
        return dict(prompt=index % len(prompts), text=text, usage=result["usage"], streamed=streamed,
                    ttft_seconds=result.get("ttft_seconds"), tpot_seconds=result.get("tpot_seconds")), time.monotonic() - started
    measured_start = time.monotonic()
    for first in range(0, args.requests, args.concurrency):
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            for result, elapsed in pool.map(generate, range(first, min(first + args.concurrency, args.requests))):
                responses.append(result)
                samples.append(elapsed)
        (directory / "responses.json").write_text(json.dumps(responses, indent=2))
        print(f"{mode}: completed {len(responses)}/{args.requests}", flush=True)
    measured_seconds = time.monotonic() - measured_start
    canonical = {}
    for response in responses:
        prompt = response["prompt"]
        if prompt in canonical and response["text"] != canonical[prompt]:
            raise AssertionError(f"generation changed across layout/cache reuse for prompt {prompt}")
        canonical[prompt] = response["text"]
        if reference is not None and response["text"] != reference[str(prompt)]:
            raise AssertionError(f"generation differs from disabled baseline for prompt {prompt}")
    lifecycle = None
    if args.exercise_lifecycle:
        lifecycle = check_cancellation(router, urls, prompts[2])
        restored, _ = generate(0)
        if restored["text"] != canonical[0]:
            raise AssertionError("generation changed after cancelled request retirement")
    generations, committed_generations = {}, {}
    for role, enabled in zip(("prefill", "decode"), (p_enabled, d_enabled)):
        with node_logs[role].open("rb") as log:
            log.seek(log_offsets[role])
            text = log.read().decode(errors="replace")
        found = [int(value) for value in re.findall(r"\[eplb\] committed[^\n]*generation=(\d+)", text)]
        generations[role] = max(found, default=0)
        committed_generations[role] = sorted(set(found))
        required = (1 if mode == "frozen" else args.require_generations) if enabled else 0
        if len(committed_generations[role]) < required:
            raise AssertionError(f"{role} committed {len(committed_generations[role])} generations "
                                 f"during this check, requires {required}")
    after = [json_request(url + "/internal/pd/descriptor") for url in urls]
    if args.collect_metrics:
        (directory / "metrics_after.json").write_text(json.dumps(
            [json_request(url + "/metrics/json") for url in urls], indent=2))
    for before, current in zip(descriptors, after):
        for key in ("node_id", "boot", "registry", "contract", "topology"):
            if before[key] != current[key]:
                raise AssertionError(f"PD node {key} changed during expert migration")
    result = dict(mode=mode, descriptors=descriptors, descriptors_after=after, lifecycle=lifecycle,
                  generations=generations, committed_generations=committed_generations, responses=responses,
                  request_seconds=samples, canonical=canonical, measured_seconds=measured_seconds,
                  output_tokens_per_second=sum(r["usage"]["completion_tokens"] for r in responses)
                  / measured_seconds,
                  latency_seconds=dict(request=latency_summary(samples),
                      client_ttft=latency_summary(r["ttft_seconds"] for r in responses),
                      request_mean_tpot=latency_summary(r["tpot_seconds"] for r in responses)))
    (directory / "result.json").write_text(json.dumps(result, indent=2))
    return {str(key): value for key, value in canonical.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--devices", default=os.environ.get("TASK_DEVICE", ""))
    parser.add_argument("--prefill-url", help="Check an already running Prefill node.")
    parser.add_argument("--decode-url", help="Check an already running Decode node.")
    parser.add_argument("--pd-config", type=Path, help="Shared node configuration in existing-node mode.")
    parser.add_argument("--prefill-log", type=Path)
    parser.add_argument("--decode-log", type=Path)
    parser.add_argument("--reference-responses", type=Path,
                        help="Compare every response with a previous disabled-mode responses.json.")
    parser.add_argument("--collect-metrics", action="store_true",
                        help="Save node /metrics/json snapshots before and after the check.")
    parser.add_argument("--exercise-lifecycle", action="store_true", help="Check streaming, cancellation and recovery.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ep-size", type=int, choices=(8, 16), default=8)
    parser.add_argument("--prefill-tp", type=int, choices=(4,), default=4)
    parser.add_argument("--decode-tp", type=int, choices=(1, 4), default=1)
    parser.add_argument("--modes", default="disabled,prefill,decode,both")
    parser.add_argument("--transfer-hostname", default="127.0.0.1")
    parser.add_argument("--requests", type=int, default=12)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--require-generations", type=int, default=2)
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument("--ring-heap", default=None, help="Override both role-specific ring heaps.")
    parser.add_argument("--prefill-ring-heap", default="1073741824,1073741824,1073741824,4294967296")
    parser.add_argument("--decode-ring-heap", default="1073741824")
    args = parser.parse_args()
    devices = [int(value) for value in args.devices.split(",") if value]
    attached = bool(args.prefill_url or args.decode_url)
    if not attached and (len(devices) != 2 * args.ep_size or len(set(devices)) != len(devices)
            or any(value < 0 for value in devices)):
        parser.error("requires twice the EP size in distinct allocated device IDs")
    modes = args.modes.split(",")
    if any(mode not in ("disabled", "prefill", "decode", "both", "frozen") for mode in modes):
        parser.error("modes must be disabled,prefill,decode,both or frozen")
    if args.requests < 3 or args.concurrency < 1 or args.max_tokens < 1:
        parser.error("requires at least three requests and positive concurrency/token limits")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    reference = None
    if args.reference_responses:
        reference = {}
        for response in json.loads(args.reference_responses.read_text()):
            key = str(response["prompt"])
            if key in reference and reference[key] != response["text"]:
                raise ValueError("reference responses disagree for the same prompt")
            reference[key] = response["text"]
    if attached:
        if not (args.prefill_url and args.decode_url and args.pd_config
                and args.prefill_log and args.decode_log and len(modes) == 1):
            parser.error("existing-node checks need both URLs, --pd-config, both node logs and one --modes value")
        directory = args.output / modes[0]
        directory.mkdir(parents=True, exist_ok=False)
        processes, files = [], []
        try:
            check_nodes(args, directory, [args.prefill_url, args.decode_url], args.pd_config,
                        modes[0], reference, processes, files)
        finally:
            for process in processes:
                stop(process)
            for stream in files:
                stream.close()
        print("PASS: existing PD nodes completed the selected check", flush=True)
        return
    if args.model is None:
        parser.error("--model is required when starting nodes")
    for mode in modes:
        canonical = run_mode(args, devices, mode, reference)
        if mode == "disabled":
            reference = canonical
    print("PASS: selected PD EPLB modes completed", flush=True)


if __name__ == "__main__":
    def terminate(signum, _frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    main()
