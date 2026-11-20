# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Contract checks for the hardware-independent parts of the daily benchmark."""

import importlib.util
import io
import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/daily_serving_perf.py"
SPEC = importlib.util.spec_from_file_location("daily_serving_perf", SCRIPT)
assert SPEC and SPEC.loader
perf = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(perf)


class WordTokenizer:
    def encode(self, text):
        return text.split()


def test_manifest_and_prompts_match_requested_workload():
    manifest = json.loads(perf.MANIFEST.read_text())
    assert manifest["rounds"] == 2
    assert [
        (item["requests"], item["input_tokens"], item["output_tokens"]) for item in manifest["cases"]
    ] == [
        (16, 8192, 1024),
        (1, 131072, 1024),
        (4, 131072, 1024),
    ]
    tokenizer = WordTokenizer()
    first = perf.build_prompt(tokenizer, 8192, 42)
    assert first == perf.build_prompt(tokenizer, 8192, 42)
    assert first != perf.build_prompt(tokenizer, 8192, 43)
    assert len(tokenizer.encode(first)) == 8192


def test_server_uses_allocated_devices_and_disables_prefix_cache():
    devices = tuple(range(16, 32))
    command = perf.server_command(Path("/model"), devices, 8232)
    assert command[command.index("--devices") + 1] == ",".join(map(str, devices))
    assert "--no-enable-prefix-caching" in command
    assert command[command.index("--max-model-len") + 1] == "262144"
    assert command[command.index("--speculative-config") + 1] == (
        '{"method":"dspark","num_speculative_tokens":7}'
    )


def test_v41_fixed_workload_and_allocated_topology(monkeypatch):
    manifest = json.loads(perf.MANIFEST.with_name("daily_v41_perf_cases.json").read_text())
    assert (manifest["rounds"], manifest["warmup_rounds"]) == (2, 1)
    case, = manifest["cases"]
    assert (case["requests"], case["input_tokens"], case["output_tokens"]) == (16, 8192, 256)
    assert len(manifest["prompts"]) == len(set(manifest["prompts"])) == 16
    for index, suffix in enumerate(manifest["prompts"]):
        prompt = perf.build_prompt(WordTokenizer(), 8192, 123 + index, suffix="\n\n" + suffix)
        assert len(WordTokenizer().encode(prompt)) == 8192
        assert prompt.endswith(suffix)
    for count in (4, 8):
        devices = tuple(range(8, 8 + count))
        monkeypatch.setenv("TASK_DEVICE", ",".join(map(str, devices)))
        assert perf.task_devices((4, 8)) == devices
        command = perf.server_command(Path("/model"), devices, 8232, v41_a5=True)
        for flag, value in (
            ("--platform", "a5"), ("--tp", "1"), ("--dp", str(count)), ("--ep", str(count)),
            ("--max-num-seqs", str(case["requests"])),
            ("--max-num-batched-tokens", str(case["max_num_batched_tokens"])),
            ("--long-prefill-token-threshold", str(case["max_num_batched_tokens"])),
            ("--max-model-len", str(case["input_tokens"] + case["output_tokens"])),
        ):
            assert command[command.index(flag) + 1] == value
        assert "--enable-chunked-prefill" in command
        assert "--no-enable-prefix-caching" in command
        assert json.loads(command[command.index("--speculative-config") + 1])["num_speculative_tokens"] == 5
    for invalid in ("0,1", "0,0,1,2", "", "0,1,2,three"):
        monkeypatch.setenv("TASK_DEVICE", invalid)
        with pytest.raises(ValueError, match="TASK_DEVICE"):
            perf.task_devices((4, 8))


def test_v41_tpot_excludes_terminal_stream_delay(monkeypatch):
    def request(url, prompt, output_tokens, input_tokens, gate, timeout, model_id):
        assert model_id == "deepseek-v41-daily"
        gate.wait(timeout=1)
        return {
            "ttft_s": 1.0, "last_token_s": 3.0, "latency_s": 9.0,
            "usage": {"prompt_tokens": 8192, "completion_tokens": 256}, "finish_reason": "length",
        }

    monkeypatch.setattr(perf, "request_one", request)
    counters = {"server_id": "server", "draft_tokens": 0, "accepted_tokens": 0}
    monkeypatch.setattr(perf, "speculation_counters", lambda url: counters)
    case = {
        "id": "v41", "input_tokens": 8192, "output_tokens": 256,
        "model": "deepseek-v41-daily", "token_event_tpot": True,
    }
    result = perf.run_round("http://localhost/v1/completions", ["prompt"] * 16, case, 1)
    assert result["status"] == "pass"
    assert len(result["request_results"]) == 16
    assert result["tpot_p50_ms"] == pytest.approx(2000 / 255)


def test_v41_sidecar_preserves_standard_fallback(tmp_path, monkeypatch):
    from pypto_serving.model import tokenizer

    monkeypatch.setattr(tokenizer, "load_tokenizer", lambda path: WordTokenizer())
    monkeypatch.setattr(perf, "revision", lambda path: "source-revision")
    monkeypatch.setenv("TASK_DEVICE", "1,2,4,7")
    monkeypatch.setenv("PYPTO_SRC", str(tmp_path))
    monkeypatch.delenv("PYPTO_V41_DAILY_DP4_PREPACKED_WEIGHTS", raising=False)
    monkeypatch.setenv("PYPTO_V41_PREPACKED_WEIGHTS", "/standard-sidecar")
    monkeypatch.setattr(perf.subprocess, "Popen", lambda command, **kwargs: SimpleNamespace(poll=lambda: None))
    monkeypatch.setattr(perf, "wait_for_health", lambda *args: None)
    monkeypatch.setattr(perf, "stop_server", lambda process: None)
    monkeypatch.setattr(perf, "run_round", lambda *args: {
        "case": "v41", "requests": 16, "status": "pass", "errors": [],
    })
    args = SimpleNamespace(
        profile="v41-a5", model=tmp_path, output_dir=tmp_path / "results",
        startup_timeout=1, request_timeout=1,
    )
    environments = []
    original = perf.subprocess.Popen
    def capture(command, **kwargs):
        environments.append(kwargs["env"])
        return original(command, **kwargs)
    monkeypatch.setattr(perf.subprocess, "Popen", capture)
    assert perf.run(args) == 0
    assert environments[0]["PYPTO_V41_PREPACKED_WEIGHTS"] == "/standard-sidecar"


@pytest.mark.parametrize("warmup_fails", [False, True])
def test_v41_run_separates_warmup_and_preserves_failures(tmp_path, monkeypatch, warmup_fails):
    from pypto_serving.model import tokenizer

    monkeypatch.setattr(tokenizer, "load_tokenizer", lambda path: WordTokenizer())
    monkeypatch.setattr(perf, "revision", lambda path: "source-revision")
    monkeypatch.setenv("TASK_DEVICE", "1,2,4,7")
    monkeypatch.setenv("PYPTO_SRC", str(tmp_path))
    monkeypatch.setenv("PYPTO_V41_DAILY_DP4_PREPACKED_WEIGHTS", "/matching-dp4-sidecar")
    environments = []

    def start(command, **kwargs):
        environments.append(kwargs["env"])
        return SimpleNamespace(poll=lambda: None)

    monkeypatch.setattr(perf.subprocess, "Popen", start)
    monkeypatch.setattr(perf, "wait_for_health", lambda *args: None)
    closed = []
    monkeypatch.setattr(perf, "stop_server", lambda process: closed.append(process))

    def run_round(url, prompts, case, timeout):
        assert len(prompts) == 16
        return {
            "case": case["id"], "requests": 16,
            "status": "fail" if warmup_fails else "pass",
            "errors": ["warmup failure"] if warmup_fails else [],
        }

    monkeypatch.setattr(perf, "run_round", run_round)
    output = tmp_path / "results"
    args = SimpleNamespace(
        profile="v41-a5", model=tmp_path, output_dir=output, startup_timeout=1, request_timeout=1,
    )
    assert perf.run(args) == int(warmup_fails)
    result = json.loads((output / "results.json").read_text())
    assert len(result["warmups"]) == 1
    assert len(result["rounds"]) == (0 if warmup_fails else 2)
    assert (output / "inputs.json").is_file() and (output / "report.md").is_file()
    assert environments[0]["PYPTO_DSV41_DP_SIZE"] == "4"
    assert environments[0]["PYPTO_V41_PREPACKED_WEIGHTS"] == "/matching-dp4-sidecar"
    assert len(closed) == 1


def test_stream_requires_terminal_usage_and_done():
    def event(payload):
        return f"data: {json.dumps(payload)}\n\n".encode()

    stream = io.BytesIO(
        event({"choices": [{"text": "hello", "finish_reason": None}], "usage": None})
        + event({"choices": [{"text": "", "finish_reason": "length"}], "usage": None})
        + event({"choices": [], "usage": {"prompt_tokens": 512, "completion_tokens": 512}})
        + b"data: [DONE]\n\n"
    )
    result = perf.read_stream(stream, time.monotonic() - 1)
    assert result["finish_reason"] == "length"
    assert result["usage"] == {"prompt_tokens": 512, "completion_tokens": 512}
    assert 0 < result["ttft_s"] <= result["latency_s"]
    with pytest.raises(ValueError, match="incomplete stream"):
        perf.read_stream(io.BytesIO(event({"choices": [{"text": "x"}]})), time.monotonic())


def test_speculation_counters_sum_replicas(monkeypatch):
    class Response(io.BytesIO):
        pass

    snapshot = {
        "schema_version": 1,
        "server_id": "server-1",
        "replicas": [
            {"counters": {"draft_tokens": 7, "accepted_tokens": 5}},
            {"counters": {"draft_tokens": 3, "accepted_tokens": 1}},
        ],
    }

    def fake_open(url, timeout):
        assert url == "http://127.0.0.1:8232/metrics/json"
        assert timeout == 10
        return Response(json.dumps(snapshot).encode())

    monkeypatch.setattr(perf.HTTP, "open", fake_open)
    counters = perf.speculation_counters("http://127.0.0.1:8232/v1/completions")
    assert counters == {"server_id": "server-1", "draft_tokens": 10, "accepted_tokens": 6}


def test_round_rejects_bad_usage_without_publishing_partial_throughput(monkeypatch):
    class Response(io.BytesIO):
        status = 200

    bad_usage = False

    def fake_open(request, timeout):
        assert timeout == 10
        body = json.loads(request.data)
        assert body["stream"] is True and body["temperature"] == 0
        count = 3 if bad_usage else 4
        chunks = [
            {"choices": [{"text": "word", "finish_reason": None}], "usage": None},
            {"choices": [{"text": "", "finish_reason": "length"}], "usage": None},
            {"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": count}},
        ]
        stream = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
        return Response(stream.encode())

    monkeypatch.setattr(perf.HTTP, "open", fake_open)
    snapshots = iter(
        [
            {"server_id": "server-1", "draft_tokens": 0, "accepted_tokens": 0},
            {"server_id": "server-1", "draft_tokens": 28, "accepted_tokens": 14},
            {"server_id": "server-1", "draft_tokens": 28, "accepted_tokens": 14},
            {"server_id": "server-1", "draft_tokens": 56, "accepted_tokens": 21},
        ]
    )
    monkeypatch.setattr(perf, "speculation_counters", lambda url: next(snapshots))
    case = {"id": "test", "input_tokens": 4, "output_tokens": 4}
    result = perf.run_round("http://127.0.0.1:1/v1/completions", ["a b c d"] * 4, case, 10)
    assert result["status"] == "pass"
    assert result["output_tokens_per_s"] > 0
    assert result["total_tokens_per_s"] == 2 * result["output_tokens_per_s"]
    assert (result["draft_tokens"], result["accepted_tokens"], result["acceptance_rate"]) == (28, 14, 0.5)
    assert {item["request"] for item in result["request_results"]} == set(range(4))
    result["round"] = 1
    report = perf.render_report(
        {"rounds": [result], "serving_commit": "serving", "pypto_lib_commit": "lib", "pypto_commit": "pypto"}
    )
    assert "| Total tok/s | Acceptance (%) |" in report
    assert "| 50.00 |" in report

    bad_usage = True
    result = perf.run_round("http://127.0.0.1:1/v1/completions", ["a b c d"] * 4, case, 10)
    assert result["status"] == "fail"
    assert len(result["errors"]) == 4
    assert "output_tokens_per_s" not in result
    assert "total_tokens_per_s" not in result
    assert "acceptance_rate" not in result
