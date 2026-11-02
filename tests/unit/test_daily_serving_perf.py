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
