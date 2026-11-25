# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Four-device V4.1 HTTP answer smoke test."""

from concurrent.futures import ThreadPoolExecutor
import json
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

import pytest


MODEL_NAME = "deepseek-v41-st"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _prompt(tokenizer, question, max_tokens=1024):
    from pypto_serving.model.deepseek_v41.encoding import encode_messages

    prefix = ("Read this context and then answer the final question.\n"
              "The two factors for the final calculation are 17 and 19. Remember both factors.\n")
    suffix = "\nQuestion: " + question
    # Separate the initial facts from the final question across prefill chunks.
    unit = " The context is background information."
    lo, hi = 0, 4000
    while lo < hi:
        middle = (lo + hi + 1) // 2
        content = prefix + unit * middle + suffix
        count = len(tokenizer.encode(encode_messages([{"role": "user", "content": content}])))
        if count <= max_tokens:
            lo = middle
        else:
            hi = middle - 1
    content = prefix + unit * lo + suffix
    count = len(tokenizer.encode(encode_messages([{"role": "user", "content": content}])))
    assert max_tokens - 96 <= count <= max_tokens, count
    return content


def _stop_server(process):
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


@pytest.mark.parametrize("num_speculative_tokens", [0, 5], ids=["k0", "k5"])
def test_v41_dp4_http_answer(tmp_path, num_speculative_tokens):
    model_path = os.environ.get("PYPTO_DSV41_MODEL_DIR")
    if not model_path or not Path(model_path).is_dir():
        pytest.fail("PYPTO_DSV41_MODEL_DIR is not set to a model directory")
    model_dir = Path(model_path)
    prefill_tokens = int(os.environ.get("PYPTO_DSV41_PREFILL_TOKENS", "1024"))
    assert 128 <= prefill_tokens <= 4096
    prompt_tokens = int(os.environ.get("PYPTO_DSV41_PROMPT_TOKENS", str(min(8192, 3 * prefill_tokens))))
    assert prefill_tokens < prompt_tokens <= 8192
    devices = tuple(part.strip() for part in os.environ.get("TASK_DEVICE", "").split(","))
    assert len(devices) == 4 and len(set(devices)) == 4 and all(part.isdecimal() for part in devices)

    from pypto_serving.model.tokenizer import load_tokenizer

    tokenizer = load_tokenizer(model_dir)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [
        sys.executable, "-m", "pypto_serving.cli", "--model", str(model_dir),
        "--served-model-name", MODEL_NAME, "--backend", "npu", "--platform", "a5",
        "--devices", ",".join(devices), "--tp", "1", "--dp", "4", "--ep", "4",
        "--block-size", "128", "--max-model-len", "8320", "--max-num-seqs", "8",
        "--max-num-batched-tokens", str(prefill_tokens), "--host", "127.0.0.1", "--port", str(port),
        "--enable-chunked-prefill", "--long-prefill-token-threshold", str(prefill_tokens),
        "--show-startup-logs",
    ]
    env = os.environ.copy()
    probe_dir = tmp_path / "prefix-probe"
    probe_dir.mkdir()
    (probe_dir / "sitecustomize.py").write_text(
        "from tests.v41_prefix_probe import install\n" + f"install({str(probe_dir)!r})\n",
        encoding="utf-8",
    )
    env["PYTHONPATH"] = os.pathsep.join((
        str(probe_dir), str(Path(__file__).resolve().parents[1]), env.get("PYTHONPATH", ""),
    ))
    metrics_dir = tmp_path / "decode-metrics"
    if num_speculative_tokens:
        command.extend([
            "--speculative-config",
            json.dumps({"method": "dspark", "num_speculative_tokens": num_speculative_tokens}),
        ])
        env["PYPTO_SERVING_DECODE_METRICS_DIR"] = str(metrics_dir)
    log_path = tmp_path / "server.log"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            worker_timeout = float(env.get("PYPTO_WORKER_INIT_TIMEOUT", "0"))
            deadline = started + max(2400, worker_timeout + 120)
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    pytest.fail(f"V4.1 server exited {process.returncode}; log: {log_path}")
                try:
                    with OPENER.open(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                        if response.status == 200:
                            print(f"STARTUP_HEALTH_SECONDS {time.monotonic() - started:.3f}", flush=True)
                            break
                except (OSError, urllib.error.URLError):
                    time.sleep(2)
            else:
                pytest.fail(f"V4.1 server did not become healthy; log: {log_path}")

            question = ("What is the product of the two factors given at the beginning of the context? "
                        "Answer with the number and one sentence.")
            payload = {"model": MODEL_NAME, "messages": [{"role": "user", "content": _prompt(tokenizer, question, prompt_tokens)}],
                       "max_tokens": 128, "temperature": 0.0, "top_p": 1.0, "stream": False}
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}, method="POST")
            with OPENER.open(request, timeout=400) as response:
                assert response.status == 200
                result = json.load(response)
            assert result["model"] == MODEL_NAME, result
            assert len(result["choices"]) == 1, result
            choice = result["choices"][0]
            answer = choice["message"]["content"]
            print("ANSWER", json.dumps({"question": question, "answer": answer,
                "usage": result["usage"], "finish_reason": choice["finish_reason"]}, ensure_ascii=False), flush=True)
            assert choice["finish_reason"] == "stop", result
            assert re.search(r"\b323\b", answer), result
            assert prompt_tokens - 96 <= result["usage"]["prompt_tokens"] <= prompt_tokens
            assert result["usage"]["prompt_tokens"] > prefill_tokens
            assert result["usage"]["completion_tokens"] > 0
            assert result["usage"]["total_tokens"] == (
                result["usage"]["prompt_tokens"] + result["usage"]["completion_tokens"]), result

            cold_answer = answer
            def repeated_answer(_):
                repeated = urllib.request.Request(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"}, method="POST")
                with OPENER.open(repeated, timeout=400) as response:
                    warmed = json.load(response)
                warmed_choice = warmed["choices"][0]
                warmed_answer = warmed_choice["message"]["content"]
                print("PREFIX_AB", json.dumps({
                    "cold_answer": cold_answer, "warm_answer": warmed_answer,
                    "identical": cold_answer == warmed_answer,
                    "correct": bool(re.search(r"\b323\b", warmed_answer)),
                    "usage": warmed["usage"], "finish_reason": warmed_choice["finish_reason"],
                }, ensure_ascii=False), flush=True)
                assert warmed_choice["finish_reason"] == "stop", warmed
                assert re.search(r"\b323\b", warmed_answer), warmed
                return warmed

            repeated_answer(0)
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(repeated_answer, range(2)))
            hits = re.findall(r"prefix_cache_hit_tokens=(\d+)", log_path.read_text(encoding="utf-8"))
            assert hits and max(map(int, hits)) >= 128, "repeated long prompts must reuse compressed prefix pages"

            if num_speculative_tokens:
                stochastic = dict(payload, temperature=0.7, top_p=0.9, top_k=32, seed=42)
                sampled_request = urllib.request.Request(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    data=json.dumps(stochastic).encode("utf-8"),
                    headers={"Content-Type": "application/json"}, method="POST")
                with OPENER.open(sampled_request, timeout=400) as response:
                    sampled = json.load(response)
                assert sampled["usage"]["completion_tokens"] > 1, sampled
                assert any(
                    item["request_id"] == sampled["id"] and item["proposed_drafts"] == num_speculative_tokens
                    for path in metrics_dir.glob("decode-steps.*.jsonl")
                    for line in path.read_text(encoding="utf-8").splitlines()
                    for item in json.loads(line)["requests"]
                ), f"Stochastic request {sampled['id']} did not verify a K=5 draft block"
                print("STOCHASTIC_K5", sampled["id"], sampled["usage"], flush=True)

            def concurrent_answer(index):
                number = 20 + index
                body = {"model": MODEL_NAME, "messages": [{"role": "user", "content":
                    f"What is {number} times 19? Answer with the number and one sentence."}],
                    "max_tokens": 128, "temperature": 0.0, "stream": False}
                request = urllib.request.Request(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    data=json.dumps(body).encode("utf-8"),
                    headers={"Content-Type": "application/json"}, method="POST")
                with OPENER.open(request, timeout=400) as response:
                    result = json.load(response)
                answer = result["choices"][0]["message"]["content"]
                assert re.search(rf"\b{number * 19}\b", answer), result
                assert result["usage"]["completion_tokens"] > 0, result
                assert result["usage"]["total_tokens"] == (
                    result["usage"]["prompt_tokens"] + result["usage"]["completion_tokens"]), result
                return result

            for wave in range(2):
                with ThreadPoolExecutor(max_workers=8) as pool:
                    results = list(pool.map(concurrent_answer, range(wave * 8, (wave + 1) * 8)))
                print("CONCURRENT_WAVE", wave, json.dumps(results, ensure_ascii=False), flush=True)
            records = [json.loads(line) for path in probe_dir.glob("pages.*.jsonl")
                       for line in path.read_text(encoding="utf-8").splitlines()]
            warm_prefills = [record for record in records if record["kind"] == "prefill"]
            assert len(warm_prefills) >= 2, records
            assert all(record["page_snapshots"] > 0 and record["unchanged"] for record in records), records
            assert all(set(record["fields"]) == {
                "compressed_cache_pool", "index_cache_pool", "index_cache_scale_pool"}
                for record in records), records
            crossings = [request for record in warm_prefills for request in record["requests"]
                         if request["start"] < request["hit"] < request["end"]]
            if prefill_tokens > 128:
                assert crossings, records
            else:
                assert any(request["end"] == request["hit"] for record in warm_prefills
                           for request in record["requests"])
                assert any(request["start"] == request["hit"] for record in warm_prefills
                           for request in record["requests"]), records
            assert any(record["kind"] == "reset" for record in records), records
            print("PREFIX_PAGE_GUARD", json.dumps(records), flush=True)
        finally:
            _stop_server(process)
            print("\n--- V4.1 server log (tail) ---", flush=True)
            print(log_path.read_text(encoding="utf-8", errors="replace")[-20000:], flush=True)

    if num_speculative_tokens:
        assert any(
            request["proposed_drafts"] > 0
            for path in metrics_dir.glob("decode-steps.*.jsonl")
            for line in path.read_text(encoding="utf-8").splitlines()
            for request in json.loads(line)["requests"]
        ), "The HTTP request did not verify any DSpark drafts"
