# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU request and lifecycle regressions; the Engine boundary never launches a worker."""

import asyncio
import contextlib
import importlib.util
import json
import os
from queue import Queue
from pathlib import Path
from types import SimpleNamespace

import pytest
from pypto_serving.config.parallel import ParallelConfig
from pypto_serving.config.types import GenerateConfig, RuntimeConfig
from pypto_serving.serving.engine.async_engine import AsyncLLMEngine, EngineConfig, ReplicaEngineCore
from pypto_serving.serving.server.ipc import ShutdownCommand, StepResult, decode_command, encode_result

CASE = Path(__file__).resolve().parents[4] / "examples"
spec = importlib.util.spec_from_file_location("qwen_program_serving", CASE / "qwen3_program_acceptance.py")
assert spec is not None and spec.loader is not None
consumer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(consumer)


def reference_data():
    rows = [[10 + step + row * 128 for row in range(16)] for step in range(128)]
    return {
        "schema": "qwen3-14b-program-tokens-v1",
        "versions": {"oracle": "synthetic"},
        "prompt_token_ids": [9, 8],
        "first_generated_token_ids": rows[0].copy(),
        "decode_input_token_ids": rows[:-1],
        "decode_output_token_ids": rows[1:],
    }


def test_reference_includes_prefill_and_every_decode_token():
    expected = consumer.expected_sequences(reference_data(), requests=16, max_tokens=128)
    assert expected[15] == list(range(1930, 2058))
    assert len(expected) == 16
    assert all(len(row) == 128 for row in expected)


@pytest.mark.parametrize("fault", ["chain", "first", "length", "boolean"])
def test_reference_rejects_broken_feedback(fault):
    reference = reference_data()
    if fault == "chain":
        reference["decode_input_token_ids"][100] = [99] * 16
    elif fault == "first":
        reference["first_generated_token_ids"][3] = 99
    elif fault == "length":
        reference["decode_output_token_ids"] = reference["decode_output_token_ids"][:-1]
    else:
        reference["decode_output_token_ids"][-1][-1] = True
    with pytest.raises(ValueError):
        consumer.expected_sequences(reference, requests=16, max_tokens=128)


@pytest.fixture
def reference_case(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    files = {
        "config.json": json.dumps({"model_type": "qwen3", "num_hidden_layers": 40, "hidden_size": 5120}),
        "model.safetensors.index.json": json.dumps({"weight_map": {"weight": "weights.safetensors"}}),
        "tokenizer.json": "{}",
        "tokenizer_config.json": "{}",
        "weights.safetensors": "synthetic",
    }
    for name, value in files.items():
        (model / name).write_text(value)
    prompt = tmp_path / "prompt.txt"
    prompt.write_bytes(b"prompt\r\n")
    reference = reference_data()
    reference["prompt_token_ids"] = [9] * 3338
    reference["prompt_sha256"] = consumer.sha256(prompt)
    reference["checkpoint_sha256"] = {name: consumer.sha256(model / name) for name in files}
    args = SimpleNamespace(
        model=model, prompt=prompt, reference=tmp_path / "reference.json", requests=1, max_tokens=3,
        runtime="tensormap_and_ringbuffer",
    )
    return args, reference


def write_reference(args, reference):
    args.reference.write_text(json.dumps(reference))
    args.reference_sha256 = consumer.sha256(args.reference)


def test_pinned_reference_authenticates_model_and_raw_prompt(reference_case):
    args, reference = reference_case
    write_reference(args, reference)
    report = {}
    assert consumer.load_reference(args, report) == ("prompt\r\n", [9] * 3338, [[10, 11, 12]])
    assert report["inputs"]["checkpoint_sha256"] == reference["checkpoint_sha256"]


def test_reference_rejects_short_prompt(reference_case):
    args, reference = reference_case
    reference["prompt_token_ids"] = [9, 8]
    write_reference(args, reference)
    with pytest.raises(ValueError, match="3338-token prompt"):
        consumer.load_reference(args, {})


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("digest", "reference SHA256"),
        ("prompt", "prompt SHA256"),
        ("optional", "every visible checkpoint asset"),
        ("extra_asset", "every visible checkpoint asset"),
        ("optional_bytes", "checkpoint SHA256 mismatch: tokenizer_config.json"),
        ("missing_core", "authenticate model config, index and tokenizer"),
        ("tokenizer", "checkpoint SHA256 mismatch: tokenizer.json"),
        ("shard_bytes", "checkpoint SHA256 mismatch: weights.safetensors"),
        ("shard_index", "every indexed weight shard"),
        ("shard_path", "local safetensors shards"),
        ("model_shape", "40-layer Qwen3-14B"),
        ("prompt_length", "3338-token prompt"),
        ("directory_link", "directory symlink"),
    ],
)
def test_prepare_rejects_unverified_inputs(reference_case, fault, message):
    args, reference = reference_case
    if fault == "prompt":
        args.prompt.write_text("changed prompt")
    elif fault == "optional":
        del reference["checkpoint_sha256"]["tokenizer_config.json"]
    elif fault == "extra_asset":
        (args.model / "extra.json").write_text("{}")
    elif fault == "optional_bytes":
        (args.model / "tokenizer_config.json").write_text("changed")
    elif fault == "missing_core":
        (args.model / "tokenizer.json").unlink()
        del reference["checkpoint_sha256"]["tokenizer.json"]
    elif fault == "tokenizer":
        (args.model / "tokenizer.json").write_text("changed")
    elif fault == "shard_bytes":
        (args.model / "weights.safetensors").write_text("changed")
    elif fault == "shard_index":
        index = args.model / "model.safetensors.index.json"
        index.write_text(json.dumps({"weight_map": {"weight": "unlisted.safetensors"}}))
        reference["checkpoint_sha256"][index.name] = consumer.sha256(index)
    elif fault == "shard_path":
        index = args.model / "model.safetensors.index.json"
        index.write_text(json.dumps({"weight_map": {"weight": "../weights.safetensors"}}))
        reference["checkpoint_sha256"][index.name] = consumer.sha256(index)
    elif fault == "model_shape":
        config = args.model / "config.json"
        config.write_text(json.dumps({"model_type": "qwen3", "num_hidden_layers": 1, "hidden_size": 5120}))
        reference["checkpoint_sha256"][config.name] = consumer.sha256(config)
    elif fault == "prompt_length":
        reference["prompt_token_ids"] = [9] * 4096
    elif fault == "directory_link":
        (args.model / "link").symlink_to(args.model.parent, target_is_directory=True)
    write_reference(args, reference)
    if fault == "digest":
        args.reference_sha256 = "0" * 64
    with pytest.raises(ValueError, match=message):
        consumer.prepare(args, {})


def test_prepare_rejects_wrong_runtime_before_startup(reference_case, monkeypatch):
    from pypto_serving.model.common.executor import utils

    args, reference = reference_case
    write_reference(args, reference)
    args.device = 0
    monkeypatch.setattr(
        utils,
        "build_pypto_run_config",
        lambda **_: SimpleNamespace(distributed_config=SimpleNamespace(runtime="host_build_graph")),
    )
    with pytest.raises(ValueError, match="does not select the requested Program runtime"):
        consumer.prepare(args, {})


@pytest.mark.parametrize("runtime", ["tensormap_and_ringbuffer", "host_build_graph"])
@pytest.mark.parametrize("wrong_prompt", [False, True])
def test_prepare_selects_program_runtime_after_input_authentication(reference_case, monkeypatch, runtime, wrong_prompt):
    from pypto_serving.model.common.executor import utils
    from pypto_serving.model.qwen import npu_executor
    from pypto_serving.model import tokenizer
    from pypto_serving.serving.engine import async_engine

    args, reference = reference_case
    write_reference(args, reference)
    args.device, args.runtime = 0, runtime
    selected = []
    constructed = []

    def run_config(**kwargs):
        selected.append(kwargs["runtime"])
        return SimpleNamespace(distributed_config=SimpleNamespace(runtime=runtime))

    monkeypatch.setattr(utils, "build_pypto_run_config", run_config)
    monkeypatch.setattr(npu_executor, "_find_pypto_lib_qwen14b_dir", lambda: args.model)
    monkeypatch.setattr(consumer, "package_identity", lambda _: {})
    monkeypatch.setattr(consumer, "source_identity", lambda _: {})
    monkeypatch.setattr(tokenizer, "load_tokenizer", lambda _: SimpleNamespace(
        encode=lambda _: [8 if wrong_prompt else 9] * 3338,
    ))
    sentinel = object()

    def make_engine(**kwargs):
        constructed.append(kwargs["config"])
        return sentinel

    monkeypatch.setattr(async_engine, "AsyncLLMEngine", make_engine)
    if wrong_prompt:
        with pytest.raises(ValueError, match="tokenizer input differs"):
            consumer.prepare(args, {})
        assert not constructed
    else:
        engine, prompt, config, expected, count = consumer.prepare(args, {})
        assert engine is sentinel and prompt == "prompt\r\n"
        assert count == 3338 and expected == [[10, 11, 12]]
        assert config.stream and config.ignore_eos
        assert constructed[0].runtime_config.chip_runtime == runtime
        assert constructed[0].max_num_running_reqs == 1
        assert constructed[0].max_num_scheduled_tokens == (80 if runtime == "host_build_graph" else 4096)
    assert selected == [runtime]


class Engine:
    def __init__(self, *, fault=None, stop_result=True):
        self.fault = fault
        self.stop_result = stop_result
        self.events = []
        self.active = set()

    async def start(self):
        self.events.append("start")
        self.all_started = asyncio.Event()
        if self.fault == "start":
            raise RuntimeError("start failed")

    async def stop(self):
        assert not self.active
        self.events.append("stop")
        if self.stop_result == "raise":
            raise RuntimeError("stop failed")
        return self.stop_result

    async def add_request(self, request_id, prompt, config):
        assert prompt == "prompt" and config.stream
        self.active.add(request_id)
        self.events.append(request_id)
        try:
            for index, token in enumerate([10, 20, 30]):
                if len(self.active) == self.parties:
                    self.all_started.set()
                await self.all_started.wait()
                if self.fault == "request" and index == 1:
                    raise RuntimeError("request failed")
                if self.fault == "short" and index == 2:
                    return
                yield SimpleNamespace(
                    token_id=99 if self.fault == "token" and index == 1 else token,
                    completion_tokens=index + (2 if self.fault == "count" else 1),
                    prompt_tokens=3 if self.fault == "prompt_count" else 2,
                    finished=index == 2,
                    finish_reason="error" if self.fault == "terminal" else "FINISHED_LENGTH",
                    token_ids=(10, 20, 99 if self.fault == "history" else 30) if index == 2 else (),
                )
        finally:
            self.active.remove(request_id)


def run(engine, report, requests=1, rounds=1, *, expected=(10, 20, 30), complete_on_mismatch=False):
    engine.parties = requests
    return asyncio.run(asyncio.wait_for(
        consumer.run_requests(
            engine,
            "prompt",
            SimpleNamespace(stream=True),
            [list(expected)] * requests,
            prompt_tokens=2,
            rounds=rounds,
            report=report,
            complete_on_mismatch=complete_on_mismatch,
        ),
        timeout=2,
    ))


def test_real_request_loop_reuses_engine_but_separates_requests_and_rounds():
    engine, report = Engine(), {"requests": []}
    run(engine, report, requests=16, rounds=2)
    assert engine.events[0] == "start" and engine.events[-1] == "stop"
    assert len(set(engine.events[1:-1])) == 32
    assert report["closed"] is True
    assert len(report["requests"]) == 32
    assert all(row["tokens"] == [10, 20, 30] and row["token_gate"] == "passed" for row in report["requests"])
    assert all(row["ttft_seconds"] >= 0 and row["tpot_seconds"] >= 0 for row in report["requests"])


@pytest.mark.parametrize(
    "fault", ["start", "request", "short", "token", "terminal", "history", "count", "prompt_count"]
)
def test_request_failures_stop_engine_and_preserve_partial_outputs(fault):
    engine, report = Engine(fault=fault), {"requests": []}
    message = "token mismatch.*at index 1" if fault == "token" else None
    with pytest.raises((RuntimeError, ValueError), match=message):
        run(engine, report, requests=2)
    assert engine.events[-1] == "stop"
    assert report["closed"] is True
    if fault != "start":
        assert report["requests"]
        assert any(row["token_gate"] != "passed" for row in report["requests"])


def test_diagnostics_complete_every_round_before_failing_the_token_gate():
    engine, report = Engine(), {"requests": []}
    with pytest.raises(ValueError, match="at index 1 .*4 completed requests mismatched"):
        run(engine, report, requests=2, rounds=2, expected=(10, 99, 30), complete_on_mismatch=True)
    assert report["closed"] is True
    assert len(report["requests"]) == 4
    assert {row["round"] for row in report["requests"]} == {0, 1}
    for row in report["requests"]:
        assert row["tokens"] == row["terminal_tokens"] == [10, 20, 30]
        assert row["first_mismatch_index"] == 1
        assert row["token_gate"] == "failed"


@pytest.mark.parametrize("stop_result", [False, "raise"])
def test_stop_failure_never_becomes_success(stop_result):
    report = {"requests": []}
    with pytest.raises(RuntimeError):
        run(Engine(stop_result=stop_result), report)
    assert report["closed"] is False
    assert report["cleanup_error"]


def test_primary_error_survives_failed_stop():
    report = {"requests": []}
    with pytest.raises(RuntimeError, match="request failed"):
        run(Engine(fault="request", stop_result=False), report)
    assert "reaped" in report["cleanup_error"]


def test_output_after_terminal_is_rejected_and_stream_is_closed():
    class ExtraOutput(Engine):
        async def add_request(self, *args):
            async for output in super().add_request(*args):
                yield output
            yield SimpleNamespace(token_id=None)

    report = {"requests": []}
    with pytest.raises(RuntimeError, match="after the terminal response"):
        run(ExtraOutput(), report)
    assert report["closed"]
    assert report["requests"][0]["tokens"] == [10, 20, 30]
    assert report["requests"][0]["token_gate"] != "passed"


class WorkerEndpoint:
    """In-process stand-in for only the process and its IPC endpoint."""

    def __init__(self, *, fail_decode=False):
        self.alive = True
        self.fail_decode = fail_decode
        self.responses = Queue()
        self.active = {}
        self.admitted = []
        self.released = []
        self.commands = []

    def put(self, payload):
        command = decode_command(payload)
        self.commands.append(command)
        if isinstance(command, ShutdownCommand):
            self.alive = False
            return
        for request_id in command.finished_request_ids:
            self.active.pop(request_id)
            self.released.append(request_id)
        for request in command.new_requests:
            assert request.prompt_token_ids == [9, 8]
            self.admitted.append(request.request_id)
            self.active[request.request_id] = None
        tokens = {}
        for prefill in command.prefill_requests:
            assert prefill.chunk_tokens == [9, 8]
            row = int(prefill.request_id.rsplit("-", 1)[-1])
            tokens[prefill.request_id] = [10 + row * 128]
        for decode in command.decode_requests:
            assert decode.last_token == self.active[decode.request_id]
            tokens[decode.request_id] = [decode.last_token + 1]
        error = "synthetic decode error" if self.fail_decode and command.decode_requests else None
        for request_id, (token,) in tokens.items():
            self.active[request_id] = token
        self.responses.put(encode_result(StepResult(tokens, error=error, step_id=command.step_id)))

    def join(self, timeout):
        assert not self.alive

    def is_alive(self):
        return self.alive


def make_engine(*, replicas=1, fail_decode=False):
    workers = []

    def worker_factory(_):
        worker = WorkerEndpoint(fail_decode=fail_decode)
        workers.append(worker)
        return worker, worker, worker.responses, None, SimpleNamespace(wait=lambda timeout: True), [128]

    def core_factory(**kwargs):
        return ReplicaEngineCore(**kwargs, worker_factory=worker_factory)

    tokenizer = SimpleNamespace(
        eos_token_id=None,
        bos_token_id=None,
        encode=lambda _: [9, 8],
        decode=lambda ids, **_: " ".join(map(str, ids)),
    )
    engine = AsyncLLMEngine(
        config=EngineConfig(
            model_id="cpu-contract",
            model_dir="unused",
            parallel_config=ParallelConfig(data_parallel_size=replicas, devices=tuple(range(replicas))),
            executor_cls="PyptoQwen14BExecutor",
            runtime_config=RuntimeConfig(page_size=128, max_seq_len=512),
            enable_prefix_cache=False,
            async_scheduling=False,
        ),
        tokenizer=tokenizer,
        core_factory=core_factory,
    )
    return engine, workers


def assert_routes_drained(engine):
    assert not engine._request_to_replica
    assert engine._route_extra_load == [0] * len(engine._cores)
    for core in engine._cores:
        assert not core._request_contexts
        assert not core.scheduler.has_work()
        assert all(request.status.is_finished for request in core.scheduler.requests.values())
        assert not core._worker_known_req_ids
        assert core.kv_cache_manager.usage == 0


def test_actual_engine_runs_prefill_decode_and_acknowledges_worker_frees():
    engine, workers = make_engine(replicas=2)
    report = {"requests": []}
    expected = consumer.expected_sequences(reference_data(), requests=16, max_tokens=128)
    asyncio.run(asyncio.wait_for(
        consumer.run_requests(
            engine, "prompt", GenerateConfig(max_new_tokens=128, stream=True), expected,
            prompt_tokens=2, rounds=2, report=report,
        ),
        timeout=10,
    ))
    assert report["closed"]
    assert_routes_drained(engine)
    assert [row["tokens"] for row in report["requests"]] == expected * 2
    assert all(row["token_gate"] == "passed" for row in report["requests"])
    assert len(workers) == 2
    assert all(worker.admitted for worker in workers)
    assert set(workers[0].admitted).isdisjoint(workers[1].admitted)
    assert {rid for worker in workers for rid in worker.admitted} == {
        f"qwen-{round_index}-{row}" for round_index in range(2) for row in range(16)
    }
    for worker in workers:
        assert sorted(worker.admitted) == sorted(worker.released)
        assert not worker.active and not worker.is_alive()
        assert isinstance(worker.commands[-1], ShutdownCommand)


def test_actual_engine_releases_failed_decode_and_retains_prefill_token():
    engine, workers = make_engine(fail_decode=True)
    report = {"requests": []}
    with pytest.raises(RuntimeError, match="unexpected serving finish reason: error"):
        asyncio.run(asyncio.wait_for(
            consumer.run_requests(
                engine, "prompt", GenerateConfig(max_new_tokens=3, stream=True), [[10, 11, 12]],
                prompt_tokens=2, rounds=1, report=report,
            ),
            timeout=5,
        ))
    assert report["requests"][0]["tokens"] == [10]
    assert report["closed"]
    assert_routes_drained(engine)
    assert workers[0].released == ["qwen-0-0"]


def test_actual_engine_abort_routes_to_owner_and_awaits_worker_release():
    engine, workers = make_engine(replicas=2)

    async def run():
        await engine.start()
        try:
            async with contextlib.aclosing(
                engine.add_request("qwen-cancel-0", "prompt", GenerateConfig(max_new_tokens=128, stream=True))
            ) as outputs:
                assert (await anext(outputs)).token_id == 10
                replica = engine._request_to_replica["qwen-cancel-0"]
                await engine.abort_request("qwen-cancel-0")
                rest = [output async for output in outputs]
                assert rest[-1].finish_reason == "FINISHED_ABORTED"
            assert_routes_drained(engine)
            assert workers[replica].released == ["qwen-cancel-0"]
            assert not workers[1 - replica].released
        finally:
            assert await engine.stop()

    asyncio.run(asyncio.wait_for(run(), timeout=5))


def test_request_failure_cancels_and_drains_an_incomplete_sibling():
    class FailingCohort(Engine):
        async def add_request(self, request_id, prompt, config):
            self.active.add(request_id)
            try:
                if request_id.endswith("-1"):
                    self.all_started.set()
                    yield await asyncio.Future()
                await self.all_started.wait()
                raise RuntimeError("request failed")
            finally:
                self.active.remove(request_id)

    report = {"requests": []}
    with pytest.raises(RuntimeError, match="request failed"):
        run(FailingCohort(), report, requests=2)
    assert report["requests"][1]["error"] == "CancelledError"
    assert report["closed"]


def test_installed_file_is_not_misattributed_to_enclosing_worktree(tmp_path):
    path = tmp_path / "installed.py"
    path.write_text("loaded bytes")
    result = consumer.source_identity(path)
    assert "checkout_head" not in result
    assert result["sha256"] == consumer.sha256(path)


def test_cli_retains_primary_and_cleanup_failure_and_restores_environment(tmp_path, monkeypatch):

    engine = Engine(fault="request", stop_result=False)
    engine.parties = 1
    monkeypatch.setenv("PYPTO_PROG_BUILD_DIR", "original")
    monkeypatch.setattr(
        consumer, "prepare", lambda *_: (engine, "prompt", SimpleNamespace(stream=True), [[10, 20, 30]], 2)
    )
    output = tmp_path / "result"
    with pytest.raises(RuntimeError, match="request failed"):
        consumer.main(
            [
                "--model",
                str(tmp_path),
                "--reference",
                str(tmp_path),
                "--reference-sha256",
                "0" * 64,
                "--prompt",
                str(tmp_path),
                "--device",
                "0",
                "--requests",
                "1",
                "--output",
                str(output),
            ]
        )
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == "failed"
    assert result["requests"][0]["tokens"] == [10]
    assert result["closed"] is False
    assert "request failed" in result["error"] and "reaped" in result["cleanup_error"]
    assert result["six_cell_acceptance"] == "not_qualified"
    assert os.environ["PYPTO_PROG_BUILD_DIR"] == "original"


def test_cli_retains_prepare_error_even_when_artifact_hashing_fails(tmp_path, monkeypatch):

    def fail_prepare(args, report):
        artifact = args.output / "artifacts" / "partial.bin"
        artifact.parent.mkdir()
        artifact.write_bytes(b"partial")
        raise ValueError("original compile error")

    def fail_hash(path):
        raise OSError("artifact unavailable")

    monkeypatch.delenv("PYPTO_PROG_BUILD_DIR", raising=False)
    monkeypatch.setattr(consumer, "prepare", fail_prepare)
    monkeypatch.setattr(consumer, "sha256", fail_hash)
    output = tmp_path / "result"
    with pytest.raises(ValueError, match="original compile error"):
        consumer.main(
            [
                "--model",
                str(tmp_path),
                "--reference",
                str(tmp_path),
                "--reference-sha256",
                "0" * 64,
                "--prompt",
                str(tmp_path),
                "--device",
                "0",
                "--output",
                str(output),
            ]
        )
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == "failed"
    assert "original compile error" in result["error"]
    assert "artifact unavailable" in result["artifact_error"]
    assert "PYPTO_PROG_BUILD_DIR" not in os.environ


def test_cli_refuses_existing_evidence_before_loading_engine(tmp_path, monkeypatch):
    monkeypatch.setattr(consumer, "prepare", lambda *_: pytest.fail("must not prepare"))
    with pytest.raises(FileExistsError):
        consumer.main(
            [
                "--model",
                str(tmp_path),
                "--reference",
                str(tmp_path),
                "--reference-sha256",
                "0" * 64,
                "--prompt",
                str(tmp_path),
                "--device",
                "0",
                "--output",
                str(tmp_path),
            ]
        )


@pytest.mark.parametrize("runtime", ["tensormap_and_ringbuffer", "host_build_graph"])
def test_cli_records_program_success_and_artifact_identity(tmp_path, monkeypatch, runtime):
    engine = Engine()
    engine.parties = 1

    def prepare(args, report):
        artifact = args.output / "artifacts" / "compiled.bin"
        artifact.parent.mkdir()
        artifact.write_bytes(b"compiled bytes")
        return engine, "prompt", SimpleNamespace(stream=True), [[10, 20, 30]], 2

    monkeypatch.setattr(consumer, "prepare", prepare)
    monkeypatch.delenv("PYPTO_PROG_BUILD_DIR", raising=False)
    output = tmp_path / "result"
    assert consumer.main([
        "--model", str(tmp_path), "--prompt", str(tmp_path), "--reference", str(tmp_path),
        "--reference-sha256", "0" * 64, "--device", "0", "--requests", "1", "--rounds", "1",
        "--max-tokens", "3", "--runtime", runtime, "--output", str(output),
    ]) == 0
    report = json.loads((output / "result.json").read_text())
    assert report["status"] == "token_gate_passed" and report["closed"]
    assert report["execution"] == "program" and report["runtime"] == runtime
    assert report["six_cell_acceptance"] == "not_qualified"
    assert report["requests"][0]["token_gate"] == "passed"
    assert report["artifact_sha256"] == {
        "artifacts/compiled.bin": consumer.sha256(output / "artifacts" / "compiled.bin"),
    }
    assert "error" not in report and "cleanup_error" not in report
    assert "PYPTO_PROG_BUILD_DIR" not in os.environ
