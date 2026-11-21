# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate eight-lane trace association and reject misleading partial traces."""

import copy
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[4] / ".agents/skills/profile-dsv4-lowlat-strace/scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import pytest

from lowlat_trace import analyze
from lowlat_profile import server_command


def save(directory, name, value):
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def evidence(tmp_path):
    run = dict(
        status="PASS",
        source_unchanged=True,
        server_exit=-15,
        run_id="cpu-fixture",
        prompt_ids=list(range(128)),
        output_tokens=26,
        devices=list(range(8)),
        compile_cache=False,
        device_strace=False,
    )
    save(tmp_path, "run-summary.json", run)
    response = dict(
        text="same output",
        usage=dict(prompt_tokens=128, completion_tokens=26),
        finish_reason="length",
        tpot_ms=5,
    )
    for name in ("warmup-responses.json", "baseline-responses.json", "responses.json"):
        save(tmp_path, name, response)
    events = []
    base = 1000000

    def span(name, ts, dur, args=None, cat="lowlat"):
        e = dict(
            ph="X", name=name, ts=base + ts, dur=dur, args=copy.deepcopy(args or {}), cat=cat, pid=500, tid=1
        )
        events.append(e)
        return e

    for category in ("scheduler", "serving", "worker", "executor"):
        span(category, 0, 500, cat=category)
    span("lowlat.prefill.submit", 10, 2, {"runtime_run_id": 7, "success": True}, "kernel")
    specs = [
        (8, 0, 2, 100, 103, 220, 223, 12),
        (9, 1, 2, 110, 224, 310, 314, 12),
        (10, 0, 1, 315, 318, 400, 404, 1),
    ]
    cursor, epoch, remaining = 128, 100, 25
    for rid, (identity, frame, cycles, submit, retire, capture, done, emitted) in enumerate(specs, 1):
        ra = dict(
            reservation_id=rid,
            request_id="request",
            generation=2,
            cycles=cycles,
            frame_id=frame,
            max_output_tokens=min(cycles * 6, remaining),
        )
        span("lowlat.native.dispatch", submit - 1, 4, ra)
        span(
            "lowlat.native.submit",
            submit,
            2,
            dict(runtime_run_id=identity, cycles=cycles, frame_id=frame),
            "kernel",
        )
        span("lowlat.native.retire", retire, done - retire, ra)
        remaining -= emitted
        span(
            "lowlat.native.capture",
            capture,
            2,
            dict(
                frame_id=frame,
                physical_input_start=cursor,
                physical_input_end=cursor + 6 * cycles,
                epoch=epoch,
                next_epoch=epoch + cycles,
                advances=[6] * cycles,
                accepted_drafts=[5] * cycles,
                emitted_ids=[123] * emitted,
                emitted=emitted,
                remaining_output_tokens=remaining,
            ),
        )
        cursor += cycles * 6
        epoch += cycles
        span("lowlat.native.release", done - 2, 1, {"frame_id": frame})
    save(tmp_path, "serving-trace/trace.json", {"traceEvents": events})
    lines = [f"[chip_process pid={100 + d} dev={d}] ready" for d in range(8)]
    for identity, ts, dur in ((7, 15, 60), (8, 115, 80), (9, 200, 80), (10, 330, 60)):
        hid = (
            "1111111111111111"
            if identity == 7
            else "3333333333333333"
            if identity == 10
            else "2222222222222222"
        )
        for device in range(8):
            prefix = f"[STRACE] v=1 pid={100 + device} tid={100 + device} inv={identity} hid={hid}"
            start = (base + ts) * 1000
            lines.append(
                f"{prefix} depth=0 name=chip.run ts={start} dur={dur * 1000} run_id={identity} dispatch_id={identity} slot_id=0 generation=1 run_epoch={identity}"
            )
            for name, offset, duration in (
                ("bind", 1, 3),
                ("runner_run", 6, dur - 12),
                ("validate", dur - 5, 3),
            ):
                lines.append(
                    f"{prefix} depth=1 name=chip.run.{name} ts={start + offset * 1000} dur={duration * 1000} run_id={identity}"
                )
    (tmp_path / "server.log").write_text("\n".join(lines))
    return events


def test_analyzer_joins_overlapping_two_cycle_parents_and_truncated_tail_without_npu(tmp_path):
    evidence(tmp_path)
    valid = analyze(tmp_path)
    assert valid["valid"] and valid["host_lanes"] == 8 and valid["post_prefill_tokens"] == 25
    assert valid["cycles_seen"] == [1, 2] and valid["frames"] == [0, 1] and valid["max_accepted_parents"] == 2
    summary = json.loads((tmp_path / "profile-summary.json").read_text())
    assert summary["parent_count"] == 3 and summary["cycles"] == 5
    assert summary["parents"][-1]["physical_input_end"] == 158
    merged = json.loads((tmp_path / "serving-strace-swimlane.json").read_text())["traceEvents"]
    roots = [e for e in merged if e.get("cat") == "strace.host" and e["args"]["depth"] == 0]
    assert len(roots) == 32 and {e["tid"] for e in roots} == set(range(8))
    assert all(e["ts"] >= 0 for e in merged if "ts" in e)
    decode = json.loads((tmp_path / "decode-only-swimlane.json").read_text())
    native_roots = [e for e in decode["traceEvents"] if e.get("cat") == "strace.host"
                    and e["args"]["depth"] == 0]
    assert len(native_roots) == 24
    assert {e["tid"] for e in native_roots} == set(range(8))
    assert {e["args"]["runtime_run_id"] for e in native_roots} == {8, 9, 10}
    assert not any("prefill" in e.get("name", "") for e in decode["traceEvents"])
    assert decode["metadata"]["device_timing_available"] is False
    assert all(e["ts"] >= 0 for e in decode["traceEvents"] if "ts" in e)
    # Reprocessing must deterministically preserve event/rank identities.
    assert analyze(tmp_path) == valid


def test_short_warmup_is_not_treated_as_a_long_context_performance_reference(tmp_path):
    evidence(tmp_path)
    run = json.loads((tmp_path / "run-summary.json").read_text())
    run.update(reference_response_files=["warmup-responses.json"], reference_matches_workload=False,
               warmup_prompt_tokens=16, warmup_output_tokens=25)
    save(tmp_path, "run-summary.json", run)
    response = dict(text="different short warmup", usage=dict(prompt_tokens=16, completion_tokens=25),
                    finish_reason="length", tpot_ms=1)
    save(tmp_path, "warmup-responses.json", response)
    assert analyze(tmp_path)["valid"]
    summary = json.loads((tmp_path / "profile-summary.json").read_text())
    assert summary["baseline_tpot_ms"] is None and summary["instrumentation_delta_percent"] is None
    response["usage"]["completion_tokens"] = 24
    save(tmp_path, "warmup-responses.json", response)
    with pytest.raises(ValueError, match="token counts"):
        analyze(tmp_path)


@pytest.mark.parametrize(
    "damage",
    [
        "missing_rank",
        "duplicate_rank",
        "wrong_run_id",
        "capture_before_publish",
        "wrong_retirement",
        "budget",
        "changed_output",
        "source",
        "failed_operation",
        "torn",
    ],
)
def test_invalid_or_misattributed_evidence_never_leaves_a_success_receipt(tmp_path, damage):
    events = evidence(tmp_path)
    analyze(tmp_path)
    if damage in ("missing_rank", "duplicate_rank", "torn"):
        lines = (tmp_path / "server.log").read_text().splitlines()
        if damage == "missing_rank":
            lines = [l for l in lines if not ("pid=107 " in l and "inv=8 " in l)]
        elif damage == "duplicate_rank":
            lines += [l.replace("inv=8 ", "inv=88 ") for l in lines if "pid=107 " in l and "inv=8 " in l]
        else:
            lines += ["[STRACE] v=1 pid=100 torn-record"]
        (tmp_path / "server.log").write_text("\n".join(lines))
    elif damage in ("changed_output", "source"):
        name = "responses.json" if damage == "changed_output" else "run-summary.json"
        value = json.loads((tmp_path / name).read_text())
        value["text" if damage == "changed_output" else "source_unchanged"] = (
            "different" if damage == "changed_output" else False
        )
        save(tmp_path, name, value)
    else:
        events = copy.deepcopy(events)
        if damage == "wrong_run_id":
            next(e for e in events if e["name"] == "lowlat.native.submit")["args"]["runtime_run_id"] = 800
        elif damage == "wrong_retirement":
            next(e for e in events if e["name"] == "lowlat.native.retire")["args"]["reservation_id"] = 999
        elif damage == "budget":
            next(e for e in events if e["name"] == "lowlat.native.capture")["args"][
                "remaining_output_tokens"
            ] = 99
        elif damage == "failed_operation":
            events[0]["args"]["success"] = False
        else:
            capture = next(e for e in events if e["name"] == "lowlat.native.capture")
            capture["ts"] = 1000180
        save(tmp_path, "serving-trace/trace.json", {"traceEvents": events})
    with pytest.raises(ValueError):
        analyze(tmp_path)
    assert json.loads((tmp_path / "skill-profile-validation.json").read_text())["valid"] is False


def test_launch_command_uses_qualified_single_batch_contract_without_compile_cache(tmp_path):
    from types import SimpleNamespace

    args = SimpleNamespace(model_dir=tmp_path, artifact_dir=tmp_path)
    cmd = server_command(args, list(range(8)), 1234)
    assert cmd[cmd.index("--tp") + 1] == "8" and cmd[cmd.index("--max-num-seqs") + 1] == "1"
    assert cmd[cmd.index("--speculative-config") + 1] == '{"method":"dspark","num_speculative_tokens":5}'
    assert "--lowlat-native-pipeline" not in cmd and "--use-compile-cache" not in cmd
    assert cmd[cmd.index("--ring-heap") + 1].split(",")[0] == "1073741824"


@pytest.mark.parametrize(
    "path,status,body,expected",
    [
        ("/start_profile", 200, b"", {}),
        ("/stop_profile", 200, b"", {}),
        ("/empty", 204, b"", {}),
        ("/health", 200, b'{"status":"ok"}', {"status": "ok"}),
    ],
)
def test_profile_http_client_accepts_empty_control_responses(path, status, body, expected):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading
    from lowlat_profile import request_json

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_POST = do_GET

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert (
            request_json(
                f"http://127.0.0.1:{server.server_port}", path, {} if "profile" in path else None, timeout=3
            )
            == expected
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(3)


def test_skill_entrypoint_runs_from_an_unrelated_working_directory(tmp_path):
    import subprocess

    result = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "lowlat_profile.py"), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "run" in result.stdout and "analyze" in result.stdout


def test_fingerprint_tracks_the_installed_serving_sources_after_script_move():
    import pypto_serving
    from lowlat_profile import source_fingerprint

    sources = source_fingerprint()
    root = Path(pypto_serving.__file__).resolve().parent
    import hashlib

    assert (
        sources["serving/model/deepseek_lowlat/npu_runner.py"]
        == hashlib.sha256((root / "model/deepseek_lowlat/npu_runner.py").read_bytes()).hexdigest()
    )


def test_one_unprofiled_reference_is_sufficient_for_long_context_profile(tmp_path):
    evidence(tmp_path)
    run = json.loads((tmp_path / "run-summary.json").read_text())
    run["reference_response_files"] = ["warmup-responses.json"]
    save(tmp_path, "run-summary.json", run)
    (tmp_path / "baseline-responses.json").unlink()
    assert analyze(tmp_path)["valid"] is True


def test_long_context_command_forwards_explicit_memory_limit(tmp_path):
    from types import SimpleNamespace

    args = SimpleNamespace(
        model_dir=tmp_path,
        artifact_dir=tmp_path,
        max_model_len=1048576,
        prefill_chunk_size=512,
        npu_memory_utilization=0.93,
    )
    command = server_command(args, list(range(8)), 1234)
    assert command[command.index("--npu-memory-utilization") + 1] == "0.93"
    assert command[command.index("--max-model-len") + 1] == "1048576"


@pytest.mark.parametrize("benchmark", [True, False])
def test_long_context_runs_short_warmup_and_one_formal_request(tmp_path, monkeypatch, benchmark):
    from types import SimpleNamespace
    import lowlat_profile as profile
    import pypto_serving.model.tokenizer as tokenizer_module

    monkeypatch.delenv("TASK_DEVICE", raising=False)
    monkeypatch.setattr(
        tokenizer_module,
        "load_tokenizer",
        lambda path: SimpleNamespace(encode=lambda prompt: [1] * (850000 if prompt == "formal" else 16)),
    )
    monkeypatch.setattr(profile, "source_fingerprint", lambda: {"source": "frozen"})
    calls = []

    def response(base, args, count):
        calls.append((args.prompt, count, args.output_tokens))
        return dict(
            text="output",
            usage=dict(prompt_tokens=count, completion_tokens=args.output_tokens),
            finish_reason="length",
            tpot_ms=5.0,
            ttft_ms=20.0,
            wall_ms=200.0,
        )

    monkeypatch.setattr(profile, "completion", response)

    controls = []

    def request(base, path, *args, **kwargs):
        controls.append(path)
        assert path in ("/health", "/start_profile", "/stop_profile")
        return {"status": "ok"}

    monkeypatch.setattr(profile, "request_json", request)

    class Process:
        pid = 123
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            self.returncode = -15

    process = Process()

    def launch(command, **kwargs):
        assert kwargs["env"]["PYPTO_RUNTIME_LOG"] == ("error" if benchmark else "v9")
        assert kwargs["env"]["PYPTO_SIMPLER_LOG_LEVEL"] == ("ERROR" if benchmark else "INFO")
        assert command[command.index("--npu-memory-utilization") + 1] == "0.93"
        return process

    monkeypatch.setattr(profile.subprocess, "Popen", launch)
    monkeypatch.setattr(profile.os, "killpg", lambda *a: None)
    analyzed = []
    import lowlat_trace
    monkeypatch.setattr(lowlat_trace, "analyze", lambda path: analyzed.append(path))
    args = SimpleNamespace(
        devices="0,1,2,3,4,5,6,7",
        model_dir=tmp_path,
        artifact_dir=tmp_path / "run",
        run_id="benchmark-test",
        prompt="formal",
        prompt_file=None,
        output_tokens=512,
        warmup_count=1,
        max_model_len=1048576,
        prefill_chunk_size=512,
        prefix_cache_mode="off",
        benchmark_only=benchmark,
        warmup_prompt="short",
        npu_memory_utilization=0.93,
        startup_timeout=1,
        request_timeout=1,
    )
    profile.run(args)
    assert calls == [("short", 16, 64), ("formal", 850000, 512)]
    receipt = json.loads((args.artifact_dir / "run-summary.json").read_text())
    label = "benchmark" if benchmark else "profile"
    assert receipt["status"] == "PASS" and receipt["mode"] == label
    assert receipt["reference_matches_workload"] is False
    assert receipt["warmup_output_tokens"] == 64
    assert receipt["measurements"][label]["usage"] == dict(prompt_tokens=850000, completion_tokens=512)
    assert controls == (["/health"] if benchmark else ["/health", "/start_profile", "/stop_profile"])
    assert analyzed == ([] if benchmark else [args.artifact_dir])
    assert not (args.artifact_dir / "skill-profile-validation.json").exists()


@pytest.mark.parametrize("benchmark", [False, True])
def test_runtime_log_threshold_survives_audit_import_before_cli_setup(tmp_path, monkeypatch, benchmark):
    import subprocess
    from types import SimpleNamespace
    import lowlat_profile as profile
    import pypto_serving.model.tokenizer as tokenizer_module

    monkeypatch.delenv("TASK_DEVICE", raising=False)
    # A caller's opposite logging preference must not invalidate the run mode.
    monkeypatch.setenv("PYPTO_SIMPLER_LOG_LEVEL", "INFO" if benchmark else "ERROR")
    monkeypatch.setattr(tokenizer_module, "load_tokenizer", lambda path: SimpleNamespace(encode=lambda text: [1]*256))
    monkeypatch.setattr(profile, "source_fingerprint", lambda: {})
    args = SimpleNamespace(devices="0,1,2,3,4,5,6,7", model_dir=tmp_path,
        artifact_dir=tmp_path/"run", run_id="log-order", prompt="prompt", prompt_file=None,
        output_tokens=64, warmup_count=2, max_model_len=16384, prefill_chunk_size=512,
        prefix_cache_mode="off" if benchmark else "hot", benchmark_only=benchmark,
        audit_prefix_state=not benchmark, warmup_prompt=None, npu_memory_utilization=0.93,
        startup_timeout=1, request_timeout=1)
    captured = {}

    def launch(command, **kwargs):
        captured.update(command=command, env=kwargs["env"])
        raise RuntimeError("launch intercepted before acquiring devices")

    with monkeypatch.context() as local:
        local.setattr(profile.subprocess, "Popen", launch)
        with pytest.raises(RuntimeError, match="launch intercepted"):
            profile.run(args)
    # Exercise the real audit import and real CLI/worker logging setup in a
    # fresh interpreter; no Worker or NPU is initialized by this probe.
    code = (
        "import runpy; "
        f"runpy.run_path({str(SCRIPT_DIR/'lowlat_prefix_audit.py')!r}, run_name='audit_log_probe'); "
        "from pypto_serving.serving.utils.env import configure_runtime_logging; "
        "from simpler._log import get_current_config; "
        "configure_runtime_logging(); configure_runtime_logging(); "
        f"assert get_current_config() == {40 if benchmark else 20}, get_current_config(); "
        "from simpler.task_interface import _initialize_host_log; "
        "from _task_interface import _host_spans_active; "
        "_initialize_host_log(); "
        f"assert _host_spans_active() is {not benchmark}"
    )
    subprocess.run([sys.executable, "-c", code], env=captured["env"], check=True, capture_output=True, text=True)


def test_analyzer_reports_missing_host_collection_without_inventing_rank_matches(tmp_path):
    evidence(tmp_path)
    (tmp_path/"server.log").write_text("\n".join(
        line for line in (tmp_path/"server.log").read_text().splitlines() if "[STRACE]" not in line))
    with pytest.raises(ValueError, match="no Host STRACE records"):
        analyze(tmp_path)
    assert json.loads((tmp_path/"skill-profile-validation.json").read_text())["valid"] is False
