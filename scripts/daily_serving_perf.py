# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Daily DSpark HTTP benchmark; run inside a task-submit allocation."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name("daily_serving_perf_cases.json")
MODEL_ID = "dsv4-flash-dspark-w8a8"
V41_PROFILE = "v41-a5"
WORDS = (" and", " the", " a", " to", " in", " of", " for", " with", " from", " on", " at", " is")
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def v41_case() -> tuple[dict, dict]:
    manifest = json.loads(MANIFEST.with_name("daily_v41_perf_cases.json").read_text(encoding="utf-8"))
    cases = manifest.get("cases", [])
    if len(cases) != 1:
        raise ValueError("daily V4.1 manifest must contain exactly one case")
    case = cases[0]
    prompts = manifest.get("prompts", [])
    if len(prompts) != case.get("requests"):
        raise ValueError(
            f"case {case.get('id')!r} requests={case.get('requests')} but prompts={len(prompts)}"
        )
    return manifest, case


def task_devices(counts: tuple[int, ...] = (16,)) -> tuple[int, ...]:
    raw = os.environ.get("TASK_DEVICE", "")
    try:
        devices = tuple(int(item) for item in raw.split(","))
    except ValueError as exc:
        raise ValueError(f"TASK_DEVICE must contain device IDs, got {raw!r}") from exc
    if len(devices) not in counts or len(set(devices)) != len(devices) or any(device < 0 for device in devices):
        raise ValueError(f"TASK_DEVICE requires a distinct device count in {counts}, got {raw!r}")
    return devices


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def server_command(
    model: Path, devices: tuple[int, ...], port: int, *, v41_a5: bool = False, case: dict | None = None
) -> list[str]:
    if v41_a5:
        if case is None:
            _, case = v41_case()
        max_model_len = case["input_tokens"] + case["output_tokens"]
        max_num_seqs = case["requests"]
        max_num_batched_tokens = case.get("max_num_batched_tokens", 4096)
        return [
            sys.executable, "-m", "pypto_serving.cli", "--model", str(model),
            "--served-model-name", case["model"], "--backend", "npu", "--platform", "a5",
            "--devices", ",".join(map(str, devices)), "--tp", "1", "--dp", str(len(devices)),
            "--ep", str(len(devices)), "--block-size", "128", "--max-model-len", str(max_model_len),
            "--max-num-seqs", str(max_num_seqs), "--max-num-batched-tokens", str(max_num_batched_tokens),
            "--enable-chunked-prefill", "--long-prefill-token-threshold", str(max_num_batched_tokens),
            "--no-enable-prefix-caching", "--speculative-config",
            '{"method":"dspark","num_speculative_tokens":5}',
            "--generate-config", '{"temperature":0,"top_p":1,"seed":123}',
            "--host", "127.0.0.1", "--port", str(port),
        ]
    return [
        sys.executable,
        "-m",
        "pypto_serving.cli",
        "--model",
        str(model),
        "--served-model-name",
        MODEL_ID,
        "--backend",
        "npu",
        "--platform",
        "a2a3",
        "--devices",
        ",".join(map(str, devices)),
        "--dp",
        "4",
        "--ep",
        "16",
        "--tp",
        "4",
        "--block-size",
        "32",
        "--max-model-len",
        "262144",
        "--max-num-seqs",
        "256",
        "--max-num-batched-tokens",
        "10240",
        "--long-prefill-token-threshold",
        "8192",
        "--no-enable-prefix-caching",
        "--speculative-config",
        '{"method":"dspark","num_speculative_tokens":7}',
        "--ring-heap",
        "2147483648,2147483648,4294967296,8589934592",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]


def wait_for_health(process: subprocess.Popen, port: int, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/health"
    last_error = "no response"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited during startup with status {process.returncode}")
        try:
            with HTTP.open(url, timeout=5) as response:
                if response.status == 200 and json.load(response) == {"status": "ok"}:
                    return
        except (OSError, ValueError, urllib.error.URLError) as exc:
            last_error = str(exc)
        time.sleep(2)
    raise TimeoutError(f"server did not become healthy within {timeout}s: {last_error}")


def stop_server(process: subprocess.Popen) -> None:
    if process.poll() is None:
        try:
            os.kill(process.pid, signal.SIGINT)
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        except ProcessLookupError:
            pass
    # The parent can exit before a worker; the task allocation must not retain it.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def build_prompt(tokenizer, target: int, seed: int, *, suffix: str = "") -> str:
    words = tuple(word for word in WORDS if len(tokenizer.encode(word)) == 1)
    if not words:
        raise ValueError("no single-token prompt words available")
    prefix = f"Benchmark request {seed}:"
    rng = random.Random(seed)
    pieces: list[str] = []
    count = target - len(tokenizer.encode(prefix + suffix))
    for _ in range(12):
        if count < 0:
            raise ValueError(f"prompt prefix exceeds target of {target} tokens")
        pieces.extend(rng.choice(words) for _ in range(max(0, count - len(pieces))))
        prompt = prefix + "".join(pieces[:count]) + suffix
        actual = len(tokenizer.encode(prompt))
        if actual == target:
            return prompt
        count += target - actual
    raise ValueError(f"could not construct a prompt of exactly {target} tokens (got {actual})")


def read_stream(response, request_start: float) -> dict:
    first_token = None
    last_token = None
    usage = None
    finish_reason = None
    done = False
    event = []
    for raw_line in response:
        line = raw_line.decode("utf-8").strip()
        if line.startswith("data:"):
            event.append(line[5:].strip())
        if line and not line.startswith("data:"):
            continue
        if line or not event:
            continue
        payload = "\n".join(event)
        event.clear()
        if payload == "[DONE]":
            done = True
            break
        chunk = json.loads(payload)
        if "error" in chunk:
            raise RuntimeError(f"stream error: {chunk['error']}")
        if chunk.get("usage") is not None:
            usage = chunk["usage"]
        for choice in chunk.get("choices", []):
            if choice.get("text"):
                last_token = time.monotonic() - request_start
                if first_token is None:
                    first_token = last_token
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
    if not done or usage is None or first_token is None:
        raise ValueError("incomplete stream: missing [DONE], usage, or first output token")
    return {
        "ttft_s": first_token,
        "latency_s": time.monotonic() - request_start,
        "last_token_s": last_token,
        "usage": usage,
        "finish_reason": finish_reason,
    }


def request_one(
    url: str, prompt: str, output_tokens: int, input_tokens: int, gate: threading.Barrier, timeout: int,
    model_id: str = MODEL_ID,
) -> dict:
    body = json.dumps(
        {"model": model_id, "prompt": prompt, "max_tokens": output_tokens, "temperature": 0, "stream": True}
    ).encode("utf-8")
    request = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    gate.wait(timeout=120)
    started = time.monotonic()
    try:
        with HTTP.open(request, timeout=timeout) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}")
            result = read_stream(response, started)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read(2048).decode('utf-8', 'replace')}") from exc
    usage = result["usage"]
    if usage.get("prompt_tokens") != input_tokens or usage.get("completion_tokens") != output_tokens:
        raise ValueError(f"unexpected token counts: {usage}")
    if result["finish_reason"] != "length":
        raise ValueError(f"unexpected finish reason: {result['finish_reason']}")
    return result


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(len(ordered) * percent) - 1)]


def speculation_counters(url: str) -> dict:
    metrics_url = urllib.parse.urljoin(url, "/metrics/json")
    with HTTP.open(metrics_url, timeout=10) as response:
        snapshot = json.load(response)
    if snapshot.get("schema_version") != 1 or not snapshot.get("replicas"):
        raise ValueError("unexpected speculation metrics schema")
    totals = {"server_id": snapshot["server_id"], "draft_tokens": 0, "accepted_tokens": 0}
    for replica in snapshot["replicas"]:
        for key in ("draft_tokens", "accepted_tokens"):
            count = replica["counters"][key]
            if type(count) is not int or count < 0:
                raise ValueError(f"invalid speculation counter {key}: {count!r}")
            totals[key] += count
    return totals


def run_round(url: str, prompts: list[str], case: dict, timeout: int) -> dict:
    counters_before = speculation_counters(url)
    gate = threading.Barrier(len(prompts) + 1)
    results: list[dict] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        futures = {
            pool.submit(
                request_one, url, prompt, case["output_tokens"], case["input_tokens"], gate, timeout,
                case.get("model", MODEL_ID),
            ): i
            for i, prompt in enumerate(prompts)
        }
        started = time.monotonic()
        gate.wait(timeout=120)
        for future in as_completed(futures):
            try:
                result = future.result()
                result["request"] = futures[future]
                results.append(result)
            except Exception as exc:
                errors.append(f"request {futures[future]}: {type(exc).__name__}: {exc}")
    wall_s = max(time.monotonic() - started, time.get_clock_info("monotonic").resolution)
    counters_after = speculation_counters(url)
    draft_tokens = counters_after["draft_tokens"] - counters_before["draft_tokens"]
    accepted_tokens = counters_after["accepted_tokens"] - counters_before["accepted_tokens"]
    if counters_after["server_id"] != counters_before["server_id"]:
        errors.append("server restarted during the round")
    if draft_tokens < 0 or not 0 <= accepted_tokens <= draft_tokens:
        errors.append("speculation counters decreased or accepted tokens exceeded draft tokens")
    result = {
        "case": case["id"],
        "requests": len(prompts),
        "input_tokens": case["input_tokens"],
        "output_tokens": case["output_tokens"],
        "wall_s": wall_s,
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "request_results": results,
        "speculation_counters_before": counters_before,
        "speculation_counters_after": counters_after,
        "draft_tokens": draft_tokens,
        "accepted_tokens": accepted_tokens,
    }
    if results and not errors:
        result.update(
            {
                "ttft_p50_s": statistics.median(item["ttft_s"] for item in results),
                "ttft_p95_s": percentile([item["ttft_s"] for item in results], 0.95),
                "latency_p50_s": statistics.median(item["latency_s"] for item in results),
                "latency_p95_s": percentile([item["latency_s"] for item in results], 0.95),
                "output_tokens_per_s": sum(item["usage"]["completion_tokens"] for item in results) / wall_s,
                "total_tokens_per_s": sum(
                    item["usage"]["prompt_tokens"] + item["usage"]["completion_tokens"] for item in results
                ) / wall_s,
                "acceptance_rate": accepted_tokens / draft_tokens if draft_tokens else None,
            }
        )
        end_key = "last_token_s" if case.get("token_event_tpot", False) else "latency_s"
        result["tpot_p50_ms"] = statistics.median(
            1000 * (item[end_key] - item["ttft_s"]) / max(1, item["usage"]["completion_tokens"] - 1)
            for item in results
        )
    return result


def render_report(results: dict) -> str:
    lines = [
        "## Daily DSpark Serving Performance",
        "",
        "Prefix caching: disabled. Rounds share one service; round 1 may include shape compilation.",
        "Total tok/s counts prompt and completion tokens; acceptance is accepted/proposed draft tokens "
        "across replicas for each round (- means no drafts).",
        "",
        "| Case | Round | Status | Requests | TTFT p50/p95 (s) | E2E p50/p95 (s) | TPOT p50 (ms) | Output tok/s | Total tok/s | Acceptance (%) |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if results.get("profile") == "v41-a5":
        workload = results["workload"]
        lines[2] = (
            f"A5 V4.1 K5, {workload['requests']} concurrent requests, {workload['input_tokens']} input tokens, "
            f"{workload['output_tokens']} output tokens (completion API). "
            f"max-num-seqs={workload['requests']}. max-num-batched-tokens={workload['max_num_batched_tokens']}. "
            "Prefix caching disabled. "
            "One complete warmup round precedes two measured rounds. TTFT includes bootstrap. "
            "TPOT uses first/last nonempty output events and actual output tokens minus one."
        )
    for item in results["rounds"]:
        fmt = lambda key: f"{item[key]:.2f}" if key in item else "-"
        acceptance = item.get("acceptance_rate")
        acceptance_text = f"{100 * acceptance:.2f}" if acceptance is not None else "-"
        lines.append(
            f"| {item['case']} | {item['round']} | {item['status']} | {item['requests']} | "
            f"{fmt('ttft_p50_s')}/{fmt('ttft_p95_s')} | "
            f"{fmt('latency_p50_s')}/{fmt('latency_p95_s')} | "
            f"{fmt('tpot_p50_ms')} | {fmt('output_tokens_per_s')} | "
            f"{fmt('total_tokens_per_s')} | {acceptance_text} |"
        )
    lines += ["", f"Serving: `{results['serving_commit']}`; pypto-lib: `{results['pypto_lib_commit']}`;"]
    lines += [f"PyPTO: `{results['pypto_commit']}`", ""]
    for item in results["rounds"]:
        for error in item["errors"]:
            lines.append(f"- {item['case']} round {item['round']}: {error}")
    if results.get("error"):
        lines.append(f"- Setup: {results['error']}")
    return "\n".join(lines) + "\n"


def revision(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def run(args: argparse.Namespace) -> int:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    v41_a5 = args.profile == "v41-a5"
    manifest_path = MANIFEST.with_name("daily_v41_perf_cases.json") if v41_a5 else MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    v41_workload = None
    if v41_a5:
        manifest, v41_workload = v41_case()
    results = {
        "model": v41_workload["model"] if v41_a5 else MODEL_ID,
        "profile": args.profile,
        "model_dir": str(args.model),
        "prefix_caching": False,
        "seed": manifest["seed"],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "serving_commit": "unavailable",
        "pypto_lib_commit": "unavailable",
        "pypto_commit": "unavailable",
        "rounds": [],
        "warmups": [],
        "workload": dict(v41_workload) if v41_a5 else {},
    }

    def publish() -> None:
        (output / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        (output / "report.md").write_text(render_report(results), encoding="utf-8")

    publish()
    process = None
    try:
        from pypto_serving.model.tokenizer import load_tokenizer

        results["serving_commit"] = revision(ROOT)
        results["pypto_lib_commit"] = revision(ROOT / "pypto-lib")
        results["pypto_commit"] = revision(Path(os.environ["PYPTO_SRC"]))
        devices = task_devices((4, 8) if v41_a5 else (16,))
        results["devices"] = devices
        if not args.model.is_dir():
            raise FileNotFoundError(f"model directory does not exist: {args.model}")
        tokenizer = load_tokenizer(args.model)
        fixture = manifest.get("prompts", [])
        if v41_a5 and len(fixture) != v41_workload["requests"]:
            raise ValueError(f"case {v41_workload['id']!r} prompt count does not match requests")
        prompts = {}
        for case in manifest["cases"]:
            prompts[case["id"]] = [
                build_prompt(tokenizer, case["input_tokens"], manifest["seed"] + i,
                             suffix="\n\n" + fixture[i] if fixture else "")
                for i in range(case["requests"])
            ]
        results["prompt_sha256"] = {
            name: [hashlib.sha256(prompt.encode()).hexdigest() for prompt in items]
            for name, items in prompts.items()
        }
        if v41_a5:
            results["python_version"] = sys.version
            results["package_versions"] = {}
            for package in ("torch", "pypto", "simpler", "ptoas"):
                try:
                    results["package_versions"][package] = importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError:
                    results["package_versions"][package] = "unavailable"
            cann_root = os.environ.get("CANN_ROOT")
            results["cann_root"] = cann_root or "unavailable"
            if cann_root:
                for path in (Path(cann_root) / "version.info", Path(cann_root).parent / "version.info"):
                    if path.is_file():
                        results["cann_version_info"] = path.read_text(encoding="utf-8")
                        break
            results["model_asset_sha256"] = {
                name: hashlib.sha256((args.model / name).read_bytes()).hexdigest()
                for name in ("config.json", "tokenizer.json", "tokenizer_config.json")
                if (args.model / name).is_file()
            }
            inputs = {
                name: [{"prompt": prompt, "input_ids": tokenizer.encode(prompt)} for prompt in items]
                for name, items in prompts.items()
            }
            encoded = json.dumps(inputs, ensure_ascii=False, sort_keys=True).encode("utf-8")
            (output / "inputs.json").write_bytes(encoded)
            results["inputs_sha256"] = hashlib.sha256(encoded).hexdigest()
            results["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        port = free_port()
        results["server_command"] = server_command(
            args.model, devices, port, v41_a5=v41_a5, case=v41_workload
        )
        server_env = os.environ.copy()
        if v41_a5:
            server_env["PYPTO_DSV41_DP_SIZE"] = str(len(devices))
            # A runner's ordinary smoke-test sidecar may be packed for DP4.
            dp_sidecar = os.environ.get(f"PYPTO_V41_DAILY_DP{len(devices)}_PREPACKED_WEIGHTS")
            if dp_sidecar:
                server_env["PYPTO_V41_PREPACKED_WEIGHTS"] = dp_sidecar
            elif os.environ.get("PYPTO_V41_PREPACKED_WEIGHTS"):
                server_env["PYPTO_V41_PREPACKED_WEIGHTS"] = os.environ["PYPTO_V41_PREPACKED_WEIGHTS"]
        with (output / "server.log").open("w", encoding="utf-8") as log:
            startup_start = time.monotonic()
            process = subprocess.Popen(
                results["server_command"],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=server_env,
            )
            try:
                wait_for_health(process, port, args.startup_timeout)
                results["startup_wait_s"] = time.monotonic() - startup_start
                url = f"http://127.0.0.1:{port}/v1/completions"
                for case in manifest["cases"]:
                    for _ in range(manifest.get("warmup_rounds", 0)):
                        warmup = run_round(url, prompts[case["id"]], case, args.request_timeout)
                        results["warmups"].append(warmup)
                        publish()
                        if warmup["status"] != "pass":
                            raise RuntimeError(f"warmup failed for {case['id']}: {warmup['errors']}")
                    for round_number in range(1, manifest["rounds"] + 1):
                        if process.poll() is not None:
                            raise RuntimeError(f"server exited before {case['id']} round {round_number}")
                        print(f"[perf] {case['id']} round {round_number}", flush=True)
                        item = run_round(url, prompts[case["id"]], case, args.request_timeout)
                        item["round"] = round_number
                        results["rounds"].append(item)
                        publish()
                        print(
                            f"[perf] {case['id']} round {round_number}: {item['status']}, "
                            f"{item.get('output_tokens_per_s', 0):.2f} output tok/s",
                            flush=True,
                        )
            finally:
                stop_server(process)
                process = None
    except Exception as exc:
        results["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[perf] {results['error']}", file=sys.stderr, flush=True)
    finally:
        if process is not None:
            stop_server(process)
        publish()
    return 1 if results.get("error") or any(item["status"] != "pass" for item in results["rounds"]) else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--startup-timeout", type=int, default=3600)
    parser.add_argument("--request-timeout", type=int, default=3600)
    parser.add_argument("--profile", choices=("dspark", "v41-a5"), default="dspark")
    return run(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
