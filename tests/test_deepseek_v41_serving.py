# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Four-device V4.1 HTTP answer smoke test."""

import json
import os
from pathlib import Path
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


def _prompt(tokenizer, question):
    from pypto_serving.model.deepseek_v41.encoding import encode_messages

    prefix = "Read this context and then answer the final question.\n"
    suffix = "\nQuestion: " + question
    # A long but semantically neutral context exercises the 8K prefill path.
    unit = " The context is background information."
    lo, hi = 0, 4000
    while lo < hi:
        middle = (lo + hi + 1) // 2
        content = prefix + unit * middle + suffix
        count = len(tokenizer.encode(encode_messages([{"role": "user", "content": content}])))
        if count <= 8192:
            lo = middle
        else:
            hi = middle - 1
    content = prefix + unit * lo + suffix
    count = len(tokenizer.encode(encode_messages([{"role": "user", "content": content}])))
    assert 8000 <= count <= 8192, count
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


def test_v41_dp4_http_answers(tmp_path):
    model_path = os.environ.get("PYPTO_DSV41_MODEL_DIR")
    if not model_path or not Path(model_path).is_dir():
        pytest.skip("PYPTO_DSV41_MODEL_DIR is not set to a model directory")
    model_dir = Path(model_path)
    devices = tuple(part.strip() for part in os.environ.get("TASK_DEVICE", "").split(","))
    assert len(devices) == 4 and len(set(devices)) == 4 and all(part.isdecimal() for part in devices)

    from pypto_serving.model.tokenizer import load_tokenizer

    tokenizer = load_tokenizer(model_dir)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [
        sys.executable, "-S", "-m", "pypto_serving.cli", "--model", str(model_dir),
        "--served-model-name", MODEL_NAME, "--backend", "npu", "--platform", "a5",
        "--devices", ",".join(devices), "--tp", "1", "--dp", "4", "--ep", "4",
        "--block-size", "128", "--max-model-len", "8320", "--max-num-seqs", "1",
        "--max-num-batched-tokens", "80", "--host", "127.0.0.1", "--port", str(port),
        "--show-startup-logs",
    ]
    log_path = tmp_path / "server.log"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            deadline = started + 2400
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

            for question, expected in (
                ("What is 17 times 19? Answer with the number and one sentence.", "323"),
                ("What is 12 times 13? Answer with the number and one sentence.", "156"),
            ):
                payload = {"model": MODEL_NAME, "messages": [{"role": "user", "content": _prompt(tokenizer, question)}],
                           "max_tokens": 128, "temperature": 0.0, "top_p": 1.0, "stream": False}
                request = urllib.request.Request(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"}, method="POST")
                with OPENER.open(request, timeout=800) as response:
                    assert response.status == 200
                    result = json.load(response)
                choice = result["choices"][0]
                answer = choice["message"]["content"]
                print("ANSWER", json.dumps({"question": question, "answer": answer,
                    "usage": result["usage"], "finish_reason": choice["finish_reason"]}, ensure_ascii=False), flush=True)
                assert choice["finish_reason"] == "stop", result
                assert expected in answer, result
                assert 8000 <= result["usage"]["prompt_tokens"] <= 8192
        finally:
            _stop_server(process)
