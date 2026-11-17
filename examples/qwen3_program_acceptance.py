# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run full Qwen requests through the existing PyPTO Serving Program engine."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import importlib
import importlib.metadata
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_sequences(reference, *, requests, max_tokens):
    """Join the prefill token to its complete, checked decode feedback chain."""
    if requests not in (1, 16) or not 2 <= max_tokens <= 128:
        raise ValueError("require 1 or 16 requests and 2..128 generated tokens")
    first = reference["first_generated_token_ids"]
    inputs = reference["decode_input_token_ids"]
    outputs = reference["decode_output_token_ids"]
    if len(inputs) != 127 or len(outputs) != 127:
        raise ValueError("reference must contain 127 decode steps")
    if any(
        not isinstance(row, list) or len(row) != 16 or any(type(token) is not int or token < 0 for token in row)
        for row in [first, *inputs, *outputs]
    ):
        raise ValueError("each reference row must contain 16 nonnegative token IDs")
    if inputs != [first, *outputs[:-1]]:
        raise ValueError("reference decode inputs break the autoregressive token chain")
    return [[first[row], *(step[row] for step in outputs[: max_tokens - 1])] for row in range(requests)]


def load_reference(args, report):
    """Verify the external token reference and the input bytes it describes."""
    raw_reference = args.reference.read_bytes()
    reference_hash = hashlib.sha256(raw_reference).hexdigest()
    if reference_hash != args.reference_sha256:
        raise ValueError("reference SHA256 differs from the pinned digest")
    reference = json.loads(raw_reference)
    if reference["schema"] != "qwen3-14b-program-tokens-v1":
        raise ValueError("unsupported Qwen token reference schema")
    expected = expected_sequences(reference, requests=args.requests, max_tokens=args.max_tokens)

    raw_prompt = args.prompt.read_bytes()
    prompt_hash = hashlib.sha256(raw_prompt).hexdigest()
    if prompt_hash != reference["prompt_sha256"]:
        raise ValueError("prompt SHA256 differs from the reference")
    prompt_ids = reference["prompt_token_ids"]
    if not prompt_ids or any(type(token) is not int or token < 0 for token in prompt_ids):
        raise ValueError("reference prompt must contain nonnegative token IDs")
    if len(prompt_ids) != 3338:
        raise ValueError("reference must describe a 3338-token prompt")
    if len(prompt_ids) + args.max_tokens > 4096:
        raise ValueError("reference prompt and generation exceed the 4096-token context")

    # Cover the visible checkpoint tree, including any auxiliary tokenizer files.
    # HF snapshot symlinks to files are common. Directory links would hide files
    # from this walk, so reject them.
    model_files = {}
    for directory, subdirs, filenames in os.walk(args.model, onerror=_raise_walk_error):
        subdirs[:] = sorted(name for name in subdirs if not name.startswith("."))
        for name in subdirs:
            if (Path(directory) / name).is_symlink():
                raise ValueError(f"checkpoint contains a directory symlink: {name}")
        for name in sorted(filenames):
            if not name.startswith("."):
                path = Path(directory) / name
                if not path.is_file():
                    raise ValueError(f"checkpoint asset is not a regular file: {path}")
                model_files[path.relative_to(args.model).as_posix()] = path

    hashes = reference["checkpoint_sha256"]
    if set(model_files) != set(hashes):
        raise ValueError("reference must authenticate every visible checkpoint asset")
    core = {"config.json", "model.safetensors.index.json", "tokenizer.json"}
    if not core.issubset(hashes):
        raise ValueError("reference must authenticate model config, index and tokenizer")
    for name in sorted(core):
        if sha256(model_files[name]) != hashes[name]:
            raise ValueError(f"checkpoint SHA256 mismatch: {name}")
    index = json.loads(model_files["model.safetensors.index.json"].read_bytes())
    shards = set(index["weight_map"].values())
    if not shards or any(Path(name).name != name or not name.endswith(".safetensors") for name in shards):
        raise ValueError("checkpoint index must reference local safetensors shards")
    if not shards.issubset(hashes):
        raise ValueError("reference must authenticate every indexed weight shard")
    model_config = json.loads(model_files["config.json"].read_bytes())
    if (model_config.get("model_type"), model_config.get("num_hidden_layers"), model_config.get("hidden_size")) != (
        "qwen3", 40, 5120
    ):
        raise ValueError("require the real 40-layer Qwen3-14B checkpoint")
    for name in sorted(hashes.keys() - core):
        if sha256(model_files[name]) != hashes[name]:
            raise ValueError(f"checkpoint SHA256 mismatch: {name}")

    report["inputs"] = {
        "reference_sha256": reference_hash,
        "reference_versions": reference["versions"],
        "checkpoint_sha256": hashes,
        "prompt_ids": prompt_ids,
        "prompt_sha256": prompt_hash,
        "expected_tokens": expected,
    }
    return raw_prompt.decode("utf-8"), prompt_ids, expected


def _raise_walk_error(error):
    raise error


async def collect_request(engine, prompt, config, expected, prompt_tokens, row, *, complete_on_mismatch=False):
    started = time.perf_counter()
    first_at = last_at = None
    terminal = False
    outputs = engine.add_request(row["request_id"], prompt, config)
    async with contextlib.aclosing(outputs):
        async for output in outputs:
            now = time.perf_counter()
            if terminal:
                raise RuntimeError("serving emitted output after the terminal response")
            if output.token_id is not None:
                row["tokens"].append(output.token_id)
                first_at = now if first_at is None else first_at
                last_at = now
                index = len(row["tokens"]) - 1
                if index >= len(expected) or output.token_id != expected[index]:
                    row.setdefault("first_mismatch_index", index)
                    if not complete_on_mismatch:
                        raise ValueError(f"token mismatch for {row['request_id']} at index {index}")
            if output.finished:
                terminal = True
                row["finish_reason"] = output.finish_reason
                row["terminal_tokens"] = list(output.token_ids)
                if output.finish_reason != "FINISHED_LENGTH":
                    raise RuntimeError(f"unexpected serving finish reason: {output.finish_reason}")
            if output.prompt_tokens != prompt_tokens or output.completion_tokens != len(row["tokens"]):
                raise ValueError("serving token counts differ from the observed request")
            if terminal and (row["terminal_tokens"] != row["tokens"] or len(row["tokens"]) != len(expected)):
                raise ValueError("terminal token history differs from the observed stream or reference length")
    if not terminal or first_at is None or last_at is None:
        raise RuntimeError("serving stream ended without a terminal response")
    row.update(
        token_gate="failed" if "first_mismatch_index" in row else "passed",
        ttft_seconds=first_at - started,
        tpot_seconds=(last_at - first_at) / (len(expected) - 1),
        end_to_end_seconds=time.perf_counter() - started,
    )


async def run_requests(engine, prompt, config, expected, *, prompt_tokens, rounds, report, complete_on_mismatch=False):
    """Drain or cancel every request before checked engine shutdown."""
    failure = None
    report["closed"] = False
    try:
        started = time.perf_counter()
        report["engine_start_attempted"] = True
        await engine.start()
        report["engine_started"] = True
        report["startup_seconds"] = time.perf_counter() - started
        for round_index in range(rounds):
            tasks = []
            for index, tokens in enumerate(expected):
                row = {
                    "request_id": f"qwen-{round_index}-{index}",
                    "round": round_index,
                    "tokens": [],
                    "token_gate": "not_passed",
                }
                report["requests"].append(row)
                tasks.append(
                    asyncio.create_task(
                        collect_request(
                            engine, prompt, config, tokens, prompt_tokens, row, complete_on_mismatch=complete_on_mismatch
                        )
                    )
                )
            try:
                await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                for task, row in zip(tasks, report["requests"][-len(tasks) :]):
                    if task.cancelled():
                        row["error"] = "CancelledError"
                    elif task.exception() is not None:
                        error = task.exception()
                        row["error"] = f"{type(error).__name__}: {error}"
        mismatched = [row for row in report["requests"] if "first_mismatch_index" in row]
        if mismatched:
            first = min(mismatched, key=lambda row: row["first_mismatch_index"])
            raise ValueError(
                f"token mismatch for {first['request_id']} at index {first['first_mismatch_index']} "
                f"({len(mismatched)} completed requests mismatched)"
            )
    except BaseException as error:
        failure = error
    finally:
        try:
            if await engine.stop() is not True:
                raise RuntimeError("serving workers were not confirmed reaped")
            report["closed"] = True
        except BaseException as error:
            report["cleanup_error"] = f"{type(error).__name__}: {error}"
            if failure is None:
                failure = error
    if failure is not None:
        raise failure


def source_identity(path):
    path = Path(path).resolve(strict=True)
    result = {"path": str(path), "sha256": sha256(path)}
    tracked = subprocess.run(
        ["git", "-C", str(path.parent), "ls-files", "--error-unmatch", "--", path.name],
        check=False,
        capture_output=True,
        text=True,
    )
    if tracked.returncode:
        return result
    revision = subprocess.run(
        ["git", "-C", str(path.parent), "rev-parse", "HEAD"], check=False, capture_output=True, text=True
    )
    if revision.returncode == 0:
        result["checkout_head"] = revision.stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(path.parent), "status", "--porcelain"], check=False, capture_output=True, text=True
        )
        if status.returncode:
            raise RuntimeError("cannot inspect source checkout status")
        result["checkout_status"] = status.stdout
    return result


def package_identity(name):
    distribution = importlib.metadata.distribution(name)
    result: dict[str, Any] = {"version": distribution.version}
    origin = distribution.read_text("direct_url.json")
    if origin is not None:
        result["origin"] = json.loads(origin)
        url = urlparse(result["origin"]["url"])
        if url.scheme == "file":
            source = Path(unquote(url.path)) / "pyproject.toml"
            if source.is_file():
                result["source"] = source_identity(source)
    return result


def prepare(args, report):
    prompt, ids, expected = load_reference(args, report)

    from pypto_serving.model.common.executor.utils import build_pypto_run_config

    run_config = build_pypto_run_config(platform="a2a3", device_ids=[args.device], runtime=args.runtime)
    if run_config.distributed_config.runtime != args.runtime:
        raise ValueError("the installed serving compiler does not select the requested Program runtime")

    from pypto_serving.config.types import GenerateConfig, RuntimeConfig
    from pypto_serving.model.qwen.npu_executor import _find_pypto_lib_qwen14b_dir
    from pypto_serving.model.tokenizer import load_tokenizer
    from pypto_serving.serving.engine.async_engine import AsyncLLMEngine, EngineConfig

    report["loaded_modules"] = {}
    for name in ("simpler", "_task_interface", "pypto", "pypto_serving", "torch"):
        module = importlib.import_module(name)
        report["loaded_modules"][name] = source_identity(module.__file__)
        report["loaded_modules"][name]["version"] = getattr(module, "__version__", None)
    report["packages"] = {name: package_identity(name) for name in ("simpler", "pypto", "pypto-serving")}
    kernel_dir = _find_pypto_lib_qwen14b_dir()
    report["kernel_sources"] = {
        path.relative_to(kernel_dir).as_posix(): source_identity(path) for path in sorted(kernel_dir.rglob("*.py"))
    }
    report["consumer"] = source_identity(__file__)
    report["cann_root"] = os.environ.get("ASCEND_HOME_PATH")
    tokenizer = load_tokenizer(str(args.model))
    if tokenizer.encode(prompt) != ids:
        raise ValueError("serving tokenizer input differs from the frozen prompt IDs")
    budget = 80 if args.runtime == "host_build_graph" else 4096
    runtime = RuntimeConfig(
        page_size=128,
        max_batch_size=16,
        max_seq_len=4096,
        max_new_tokens=args.max_tokens,
        device="cpu",
        kv_dtype="bfloat16",
        weight_dtype="bfloat16",
        chip_runtime=args.runtime,
        max_num_batched_tokens=budget,
        ring_dep_pool=262144,
        ring_task_window=262144,
        ring_heap=4 << 30,
    )
    config = EngineConfig(
        model_id="qwen3-14b-program",
        model_dir=str(args.model),
        platform="a2a3",
        device_id=args.device,
        device_ids=(args.device,),
        executor_cls="PyptoQwen14BExecutor",
        runtime_config=runtime,
        max_num_running_reqs=args.requests,
        max_num_scheduled_tokens=budget,
        long_prefill_token_threshold=min(2048, budget),
        enable_prefix_cache=False,
        enable_chunk_prefill=True,
        async_scheduling=False,
    )
    generation = GenerateConfig(
        max_new_tokens=args.max_tokens, temperature=0.0, top_p=1.0, top_k=None, stream=True, ignore_eos=True
    )
    return AsyncLLMEngine(config=config, tokenizer=tokenizer), prompt, generation, expected, len(ids)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True, help="Pinned SHA256 of the reference JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument(
        "--runtime", choices=("tensormap_and_ringbuffer", "host_build_graph"), default="tensormap_and_ringbuffer"
    )
    parser.add_argument("--requests", type=int, choices=(1, 16), default=16)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument(
        "--complete-on-mismatch", action="store_true", help="Complete all rounds on token mismatch and still fail the gate"
    )
    args = parser.parse_args(argv)
    if not 2 <= args.max_tokens <= 128 or args.rounds < 1 or args.device < 0:
        parser.error("require 2..128 tokens, positive rounds and a nonnegative device")
    args.output.mkdir(parents=True, exist_ok=False)
    args.output = args.output.resolve()
    args.model, args.reference, args.prompt = args.model.resolve(), args.reference.resolve(), args.prompt.resolve()
    budget = 80 if args.runtime == "host_build_graph" else 4096
    report = {
        "status": "running",
        "cell": "program-hbg" if args.runtime == "host_build_graph" else "program-tmr",
        "mode": "program",
        "execution": "program",
        "runtime": args.runtime,
        "requests": [],
        "closed": False,
        "engine_start_attempted": False,
        "engine_started": False,
        "scope": "serving prefill, decode, greedy sampling and full token checks",
        "tensor_numerics": "not_observed",
        "six_cell_acceptance": "not_qualified",
        "request_config": {"requests": args.requests, "max_tokens": args.max_tokens, "rounds": args.rounds},
        "complete_on_mismatch": args.complete_on_mismatch,
        "timing_scope": "host-observed request latency including scheduler time",
        "scheduler": {
            "async_scheduling": False,
            "chunked_prefill": True,
            "max_num_scheduled_tokens": budget,
            "long_prefill_token_threshold": min(2048, budget),
        },
    }
    result_path = args.output / "result.json"
    result_path.write_text(json.dumps(report, indent=2) + "\n")
    previous_build_dir = os.environ.get("PYPTO_PROG_BUILD_DIR")
    os.environ["PYPTO_PROG_BUILD_DIR"] = str(args.output / "artifacts")
    failure = None
    try:
        engine, prompt, generation, expected, prompt_tokens = prepare(args, report)
        asyncio.run(
            run_requests(
                engine,
                prompt,
                generation,
                expected,
                prompt_tokens=prompt_tokens,
                rounds=args.rounds,
                report=report,
                complete_on_mismatch=args.complete_on_mismatch,
            )
        )
        report["status"] = "token_gate_passed"
    except BaseException as error:
        failure = error
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        if previous_build_dir is None:
            os.environ.pop("PYPTO_PROG_BUILD_DIR", None)
        else:
            os.environ["PYPTO_PROG_BUILD_DIR"] = previous_build_dir
        try:
            report["artifact_sha256"] = {
                str(path.relative_to(args.output)): sha256(path)
                for path in sorted((args.output / "artifacts").rglob("*"))
                if path.is_file()
            }
        except BaseException as error:
            report["artifact_error"] = f"{type(error).__name__}: {error}"
            report["status"] = "failed"
            if failure is None:
                failure = error
        try:
            result_path.write_text(json.dumps(report, indent=2) + "\n")
        except BaseException as error:
            if failure is None:
                failure = error
    if failure is not None:
        raise failure
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
