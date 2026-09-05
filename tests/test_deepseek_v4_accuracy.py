# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""DeepSeek V4 HTTP generation accuracy guard for CI."""

from __future__ import annotations

import io
import json
import os
import queue
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "dsv4-flash-w8a8"


@dataclass(frozen=True)
class MtpAccuracyCase:
    num_speculative_tokens: int
    prompt: str
    prompt_tokens: int | None
    max_new_tokens: int
    expected_text: str | None
    temperature: float = 0.0
    top_k: int | None = None
    seed: int | None = None
    enable_prefix_caching: bool = False
    validate_chat_template: bool = False
    # Known-valid continuations for cases whose target model hits a documented
    # near-tie; each run must land inside the set instead of matching another
    # run token-for-token.
    acceptable_texts: tuple[str, ...] | None = None


# K=1 uses EAGLE look-ahead, so a reusable 128-token prefix needs another
# complete page after it. Repeating a common single-token fragment also keeps
# this case above one 1024-token serving chunk and below the 2048-token limit.
PREFIX_PROMPT = " and" * 1200 + " Huawei is"

# The K=1 target model has a near-tied argmax after " a leading global"
# ("provider" vs "information"), and the accepted NPU kernel nondeterminism
# flips it between otherwise identical greedy runs. Both continuations are
# valid model output, so the prefix-cache case pins the set instead of
# demanding that the cached run replay the cold run token-for-token.
PREFIX_CACHE_ACCEPTABLE_TEXTS = (
    " a leading global provider of information and communications technology (",
    " a leading global information and communications technology (ICT)",
)

CHAT_TEMPLATE_CONTENT = "What is 1+1?"
DEFAULT_CHAT_PROMPT = (
    "<\uff5cbegin\u2581of\u2581sentence\uff5c>"
    "<\uff5cUser\uff5c>What is 1+1?"
    "<\uff5cAssistant\uff5c></think>"
)

# Keep the fused K=1 baseline, one standalone DeepSeek MTP decode shape, and
# one NPU prefix-cache case. K=3 selects the S=4/B=4 standalone tile.
# Multi-request state and other MTP depths are covered by focused unit guards
# without expanding this hardware feature gate.
MTP_CASES = (
    MtpAccuracyCase(
        num_speculative_tokens=1,
        # Official Palace Museum introduction: https://www.dpm.org.cn/Explore.html
        # 新版 prompt：
        # 紫禁城南北长961米，东西宽753米，四面围有高10米的城墙，城外有宽52米的护城河。
        # 紫禁城有四座城门，南面为午门，北面为神武门，东面为东华门，西面为西华门。
        prompt=(
            "\u7d2b\u7981\u57ce\u5357\u5317\u957f961\u7c73\uff0c"
            "\u4e1c\u897f\u5bbd753\u7c73\uff0c"
            "\u56db\u9762\u56f4\u6709\u9ad810\u7c73\u7684\u57ce\u5899\uff0c"
            "\u57ce\u5916\u6709\u5bbd52\u7c73\u7684\u62a4\u57ce\u6cb3\u3002"
            "\u7d2b\u7981\u57ce\u6709\u56db\u5ea7\u57ce\u95e8\uff0c"
            "\u5357\u9762\u4e3a\u5348\u95e8\uff0c"
            "\u5317\u9762\u4e3a\u795e\u6b66\u95e8\uff0c"
            "\u4e1c\u9762\u4e3a\u4e1c\u534e\u95e8\uff0c"
            "\u897f\u9762\u4e3a\u897f\u534e\u95e8\u3002"
        ),
        prompt_tokens=64,
        max_new_tokens=128,
        # Exercise fused device sampling with a non-trivial candidate set.
        temperature=0.8,
        top_k=32,
        seed=42,
        validate_chat_template=True,
        # temperature=0.8, top-k=32, seed=42 的 expected text：
        # 城墙的四角，各有一座风姿绰约的角楼，民间有九梁十八柱七十二条脊之说，形容其结构的复杂。
        # 紫禁城内的建筑分为外朝和内廷两部分。外朝的中心为太和殿、中和殿、保和殿，统称三大殿，
        # 是国家举行大典礼的场所。内廷的中心是乾清宫、交泰殿、坤宁宫，统称后三宫，是皇帝和皇后
        # 居住的正宫。其后为御花园。后三宫两侧排列着东、西六宫，是后妃们居住休息的地方。东六宫东
        expected_text=(
            "\u57ce\u5899\u7684\u56db\u89d2\uff0c\u5404\u6709\u4e00\u5ea7\u98ce\u59ff\u7ef0\u7ea6\u7684\u89d2\u697c\uff0c\u6c11\u95f4"
            "\u6709\u4e5d\u6881\u5341\u516b\u67f1\u4e03\u5341\u4e8c\u6761\u810a\u4e4b\u8bf4\uff0c\u5f62\u5bb9\u5176\u7ed3\u6784\u7684"
            "\u590d\u6742\u3002\u7d2b\u7981\u57ce\u5185\u7684\u5efa\u7b51\u5206\u4e3a\u5916\u671d\u548c\u5185\u5ef7\u4e24\u90e8\u5206"
            "\u3002\u5916\u671d\u7684\u4e2d\u5fc3\u4e3a\u592a\u548c\u6bbf\u3001\u4e2d\u548c\u6bbf\u3001\u4fdd\u548c\u6bbf\uff0c\u7edf"
            "\u79f0\u4e09\u5927\u6bbf\uff0c\u662f\u56fd\u5bb6\u4e3e\u884c\u5927\u5178\u793c\u7684\u573a\u6240\u3002\u5185\u5ef7\u7684"
            "\u4e2d\u5fc3\u662f\u4e7e\u6e05\u5bab\u3001\u4ea4\u6cf0\u6bbf\u3001\u5764\u5b81\u5bab\uff0c\u7edf\u79f0\u540e\u4e09\u5bab"
            "\uff0c\u662f\u7687\u5e1d\u548c\u7687\u540e\u5c45\u4f4f\u7684\u6b63\u5bab\u3002\u5176\u540e\u4e3a\u5fa1\u82b1\u56ed\u3002"
            "\u540e\u4e09\u5bab\u4e24\u4fa7\u6392\u5217\u7740\u4e1c\u3001\u897f\u516d\u5bab\uff0c\u662f\u540e\u5983\u4eec\u5c45\u4f4f"
            "\u4f11\u606f\u7684\u5730\u65b9\u3002\u4e1c\u516d\u5bab\u4e1c"
        ),
    ),
    MtpAccuracyCase(
        num_speculative_tokens=3,
        prompt="Huawei is",
        prompt_tokens=4,
        max_new_tokens=10,
        expected_text=" a leading global information and communications technology (ICT)",
    ),
    MtpAccuracyCase(
        num_speculative_tokens=1,
        prompt=PREFIX_PROMPT,
        prompt_tokens=None,
        max_new_tokens=10,
        expected_text=None,
        enable_prefix_caching=True,
        acceptable_texts=PREFIX_CACHE_ACCEPTABLE_TEXTS,
    ),
)
MTP_CASE_IDS = ("k1-fused", "k3-s4-b4", "k1-prefix-cache")

STARTUP_TIMEOUT_SECONDS = int(os.environ.get("PYPTO_DSV4_STARTUP_TIMEOUT_SECONDS", "1800"))
OVERALL_TIMEOUT_SECONDS = int(os.environ.get("PYPTO_DSV4_OVERALL_TIMEOUT_SECONDS", "2400"))
HEARTBEAT_SECONDS = 30
LOCAL_URL_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _task_devices() -> tuple[int, ...]:
    raw_devices = os.environ.get("TASK_DEVICE", "")
    try:
        devices = tuple(int(value.strip()) for value in raw_devices.split(",") if value.strip())
    except ValueError:
        pytest.fail(f"TASK_DEVICE must contain comma-separated integer device IDs, got {raw_devices!r}")
    if len(devices) != 8 or len(set(devices)) != 8 or any(device < 0 for device in devices):
        pytest.fail(f"TASK_DEVICE must contain exactly 8 unique non-negative device IDs, got {raw_devices!r}")
    return devices


def _unused_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _server_command(
    model_dir: Path,
    devices: tuple[int, ...],
    port: int,
    num_speculative_tokens: int,
    enable_prefix_caching: bool = False,
) -> list[str]:
    # CI substitutes only the checkpoint, task-submit devices, and free port.
    # Scalar ring values are broadcast to all four levels, prewarming the same
    # 4 GiB arena used by later dispatches.
    return [
        sys.executable,
        "-m",
        "pypto_serving.cli",
        "--model",
        str(model_dir),
        "--served-model-name",
        MODEL_ID,
        "--backend",
        "npu",
        "--platform",
        "a2a3",
        "--devices",
        ",".join(str(device) for device in devices),
        "--dp",
        "8",
        "--ep",
        "8",
        "--block-size",
        "128",
        "--max-model-len",
        "2048" if enable_prefix_caching else "260",
        "--max-num-seqs",
        "8",
        "--max-num-batched-tokens",
        "1024",
        "--npu-memory-utilization",
        "0.90",
        "--long-prefill-token-threshold",
        "1024",
        "--num-speculative-tokens",
        str(num_speculative_tokens),
        "--enable-prefix-caching" if enable_prefix_caching else "--no-enable-prefix-caching",
        "--ring-dep-pool",
        "16384",
        "--ring-task-window",
        "16384",
        "--ring-heap",
        "1073741824",
        "--port",
        str(port),
        "--show-startup-logs",
    ]


def _wait_for_health(process: subprocess.Popen, port: int, deadline: float) -> None:
    url = f"http://127.0.0.1:{port}/health"
    startup_deadline = min(deadline, time.monotonic() + STARTUP_TIMEOUT_SECONDS)
    next_heartbeat = time.monotonic()
    last_error: BaseException | None = None

    while time.monotonic() < startup_deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"DeepSeek server exited before becoming healthy (code={return_code})")
        try:
            with LOCAL_URL_OPENER.open(url, timeout=5) as response:
                payload = json.loads(response.read())
            if response.status == 200 and payload == {"status": "ok"}:
                print("DeepSeek server is healthy", flush=True)
                return
        except (OSError, TimeoutError, ValueError, urllib.error.URLError) as exc:
            last_error = exc

        now = time.monotonic()
        if now >= next_heartbeat:
            print("Waiting for DeepSeek server startup...", flush=True)
            next_heartbeat = now + HEARTBEAT_SECONDS
        time.sleep(2)

    raise TimeoutError(f"DeepSeek server did not become healthy: {last_error}")


def _request_completion(
    process: subprocess.Popen,
    port: int,
    deadline: float,
    *,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    seed: int | None = None,
    model: str = MODEL_ID,
) -> dict:
    return _request_json(
        process,
        port,
        deadline,
        endpoint="/v1/completions",
        request_kind="completion",
        payload={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": 1.0,
            "top_k": top_k,
            "seed": seed,
        },
    )


def _request_chat_completion(
    process: subprocess.Popen,
    port: int,
    deadline: float,
    *,
    content: str,
    max_new_tokens: int,
) -> dict:
    return _request_json(
        process,
        port,
        deadline,
        endpoint="/v1/chat/completions",
        request_kind="chat completion",
        payload={
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
        },
    )


def _request_json(
    process: subprocess.Popen,
    port: int,
    deadline: float,
    *,
    endpoint: str,
    request_kind: str,
    payload: dict[str, object],
) -> dict:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{endpoint}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    results: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)

    def send_request() -> None:
        try:
            timeout = max(1.0, deadline - time.monotonic())
            with LOCAL_URL_OPENER.open(request, timeout=timeout) as response:
                body = response.read().decode("utf-8")
                results.put((True, json.loads(body)))
        except urllib.error.HTTPError as exc:
            try:
                error_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                error_body = "<failed to read error body>"
            results.put(
                (False, RuntimeError(f"{request_kind} request returned HTTP {exc.code}: {error_body}"))
            )
        except BaseException as exc:
            results.put((False, exc))

    threading.Thread(
        target=send_request,
        name=f"deepseek-{request_kind.replace(' ', '-')}",
        daemon=True,
    ).start()
    while time.monotonic() < deadline:
        try:
            succeeded, value = results.get(timeout=HEARTBEAT_SECONDS)
        except queue.Empty:
            return_code = process.poll()
            if return_code is not None:
                raise RuntimeError(
                    f"DeepSeek server exited during generation (code={return_code})"
                ) from None
            print(f"Waiting for DeepSeek {request_kind}...", flush=True)
            continue
        if succeeded:
            if not isinstance(value, dict):
                raise TypeError(
                    f"{request_kind} response must be a JSON object, got {type(value).__name__}"
                )
            return value
        if isinstance(value, BaseException):
            raise value
        raise RuntimeError(f"{request_kind} request failed: {value}")
    raise TimeoutError(f"DeepSeek {request_kind} exceeded the end-to-end timeout")


def _assert_chat_template_matches_manual_prompt(
    process: subprocess.Popen,
    port: int,
    deadline: float,
) -> None:
    raw_response = _request_completion(
        process,
        port,
        deadline,
        prompt=DEFAULT_CHAT_PROMPT,
        max_new_tokens=1,
        temperature=0.0,
        top_k=None,
        seed=None,
    )
    chat_response = _request_chat_completion(
        process,
        port,
        deadline,
        content=CHAT_TEMPLATE_CONTENT,
        max_new_tokens=1,
    )

    raw_choice = raw_response["choices"][0]
    chat_choice = chat_response["choices"][0]
    assert chat_response.get("model") == MODEL_ID
    assert chat_choice["message"]["role"] == "assistant"
    assert chat_choice["message"]["content"] == raw_choice["text"]
    assert chat_response["usage"]["prompt_tokens"] == raw_response["usage"]["prompt_tokens"]
    assert chat_response["usage"]["completion_tokens"] == 1


def _stop_process_group(process: subprocess.Popen) -> None:
    """Gracefully stop the server, then force-kill its process group if needed."""
    try:
        # Signal only the uvicorn parent first. Its shutdown hook stops the
        # serving worker through the normal ShutdownCommand protocol, which
        # releases the persistent L3 worker and merges profiling fragments.
        os.kill(process.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    except OSError as exc:
        print(f"WARNING: failed to request graceful server shutdown {process.pid}: {exc}", flush=True)
        return

    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        try:
            # SIGTERM is still sent to the parent only; this gives uvicorn a
            # second chance to run its shutdown lifecycle before escalation.
            os.kill(process.pid, signal.SIGTERM)
        except OSError:
            pass
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print(f"WARNING: process group {process.pid} still alive after SIGKILL", flush=True)
            except OSError as exc:
                print(f"WARNING: failed to reap process group {process.pid}: {exc}", flush=True)
        except OSError as exc:
            print(f"WARNING: failed to reap process group {process.pid}: {exc}", flush=True)
        return
    except OSError as exc:
        print(f"WARNING: failed to wait for process group {process.pid}: {exc}", flush=True)
        return

    # The server parent may exit before a worker child. Give the normal worker
    # shutdown command a short grace period, then kill remaining descendants.
    shutdown_deadline = time.monotonic() + 5
    while time.monotonic() < shutdown_deadline:
        try:
            os.killpg(process.pid, 0)
        except OSError:
            return
        time.sleep(0.2)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except OSError:
        pass


def _print_server_log(log_path: Path) -> None:
    if not log_path.exists():
        return
    try:
        with log_path.open("rb") as log_file:
            log_file.seek(0, os.SEEK_END)
            log_file.seek(max(0, log_file.tell() - 50000))
            content = log_file.read().decode("utf-8", errors="replace")
    except OSError as exc:
        print(f"WARNING: failed to read DeepSeek server log: {exc}", flush=True)
        return
    print("\n--- DeepSeek server log (tail) ---", flush=True)
    print(content, flush=True)


@pytest.mark.parametrize(
    "case",
    MTP_CASES,
    ids=MTP_CASE_IDS,
)
def test_deepseek_v4_http_completion_matches_expected_text(
    tmp_path: Path,
    case: MtpAccuracyCase,
) -> None:
    model_dir_env = os.environ.get("PYPTO_DSV4_MODEL_DIR")
    model_dir = Path(model_dir_env) if model_dir_env else None
    if model_dir is None or not model_dir.is_dir():
        pytest.fail(f"PYPTO_DSV4_MODEL_DIR not set or not a directory: {model_dir}")
    devices = _task_devices()
    port = _unused_local_port()
    enable_prefix_caching = case.enable_prefix_caching
    cache_suffix = "-prefix-cache" if enable_prefix_caching else ""
    log_path = tmp_path / f"deepseek-v4-k{case.num_speculative_tokens}{cache_suffix}-server.log"
    deadline = time.monotonic() + OVERALL_TIMEOUT_SECONDS

    try:
        with log_path.open("w", encoding="utf-8") as server_log:
            process = subprocess.Popen(
                _server_command(
                    model_dir,
                    devices,
                    port,
                    case.num_speculative_tokens,
                    enable_prefix_caching,
                ),
                cwd=ROOT,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
            )
            try:
                _wait_for_health(process, port, deadline)
                run_count = 2 if enable_prefix_caching else 1
                responses = []
                for run_index in range(run_count):
                    response = _request_completion(
                        process,
                        port,
                        deadline,
                        prompt=case.prompt,
                        max_new_tokens=case.max_new_tokens,
                        temperature=case.temperature,
                        top_k=case.top_k,
                        seed=case.seed,
                    )
                    responses.append(response)
                    print(
                        f"DeepSeek K={case.num_speculative_tokens} "
                        f"prefix_cache={enable_prefix_caching} run={run_index} "
                        f"completion: {response}",
                        flush=True,
                    )
                    assert response.get("model") == MODEL_ID
                    choices = response.get("choices")
                    assert isinstance(choices, list) and len(choices) == 1
                    assert choices[0].get("finish_reason") == "length"
                    usage = response.get("usage", {})
                    assert usage.get("completion_tokens") == case.max_new_tokens
                    if case.prompt_tokens is not None:
                        assert usage.get("prompt_tokens") == case.prompt_tokens
                    if case.expected_text is not None:
                        assert choices[0].get("text") == case.expected_text

                if case.validate_chat_template:
                    _assert_chat_template_matches_manual_prompt(process, port, deadline)

                if enable_prefix_caching:
                    prompt_tokens = responses[0].get("usage", {}).get("prompt_tokens", 0)
                    assert prompt_tokens > 1024
                    # The target model's near-tie makes run-to-run text
                    # equality unattainable; cache corruption is guarded by
                    # pinning the set of known-valid continuations.
                    for response in responses:
                        text = response["choices"][0]["text"]
                        assert text in case.acceptable_texts, (
                            f"completion is not a known-valid continuation: {text!r}"
                        )
            finally:
                _stop_process_group(process)
        if enable_prefix_caching:
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            hits = [
                int(value)
                for value in re.findall(r"prefix_cache_hit_tokens=(\d+)", log_text)
            ]
            assert hits and max(hits) >= 128, (
                "Repeated long prompt produced no observable grouped prefix-cache hit"
            )
    except BaseException:
        _print_server_log(log_path)
        raise


def test_completion_http_error_includes_response_body(monkeypatch) -> None:
    error = urllib.error.HTTPError(
        "http://127.0.0.1/completions",
        500,
        "Internal Server Error",
        {},
        io.BytesIO(b"device allocation failed"),
    )

    def raise_http_error(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(LOCAL_URL_OPENER, "open", raise_http_error)

    class RunningProcess:
        @staticmethod
        def poll():
            return None

    with pytest.raises(RuntimeError, match="HTTP 500: device allocation failed"):
        _request_completion(
            RunningProcess(),
            1,
            time.monotonic() + 1,
            prompt="Huawei is",
            max_new_tokens=1,
            temperature=0.0,
            top_k=None,
            seed=None,
        )


def test_server_command_uses_explicit_mtp_depth_and_serving_capacity(tmp_path) -> None:
    command = _server_command(tmp_path, tuple(range(8)), 12345, 3)

    assert "--enable-mtp" not in command
    assert command[command.index("--num-speculative-tokens") + 1] == "3"
    assert command[command.index("--max-num-seqs") + 1] == "8"
    assert command[command.index("--long-prefill-token-threshold") + 1] == "1024"
    assert command[command.index("--ring-dep-pool") + 1] == "16384"
    assert command[command.index("--ring-task-window") + 1] == "16384"
    assert command[command.index("--ring-heap") + 1] == "1073741824"


def test_mtp_matrix_covers_fused_and_standalone_shapes() -> None:
    non_prefix_depths = tuple(
        case.num_speculative_tokens for case in MTP_CASES if not case.enable_prefix_caching
    )
    prefix_cases = tuple(case for case in MTP_CASES if case.enable_prefix_caching)

    assert non_prefix_depths == (1, 3)
    assert len(prefix_cases) == 1
    assert prefix_cases[0].num_speculative_tokens == 1
    assert [case for case in MTP_CASES if case.validate_chat_template] == [MTP_CASES[0]]
    assert (MTP_CASES[0].prompt_tokens, MTP_CASES[0].max_new_tokens) == (64, 128)
    assert (MTP_CASES[0].temperature, MTP_CASES[0].top_k, MTP_CASES[0].seed) == (
        0.8,
        32,
        42,
    )


def test_stop_process_group_suppresses_final_wait_timeout(monkeypatch, capsys) -> None:
    class StuckProcess:
        pid = 123

        @staticmethod
        def wait(timeout):
            raise subprocess.TimeoutExpired("server", timeout)

    monkeypatch.setattr(os, "killpg", lambda *_args: None)
    monkeypatch.setattr(os, "kill", lambda *_args: None)

    _stop_process_group(StuckProcess())

    assert "still alive after SIGKILL" in capsys.readouterr().out


def test_print_server_log_reads_only_tail(tmp_path, capsys) -> None:
    log_path = tmp_path / "server.log"
    log_path.write_bytes(b"excluded-prefix\n" + b"x" * 60000 + b"\nincluded-tail\n")

    _print_server_log(log_path)

    output = capsys.readouterr().out
    assert "excluded-prefix" not in output
    assert "included-tail" in output
