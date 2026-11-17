# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Manually check 16K/1M DSpark chat retrieval on 16 cards, K=7, chunk=128.

Provide raw user text via --prompt-file and three --expected-codes in order.
Input lengths include the chat template; --check-only needs no devices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_deepseek_dspark_accuracy import (  # noqa: E402
    DEFAULT_MODEL_DIR, MODEL_ID, ROOT, _server_command, _task_devices, _wait_for_device_reclaim,
)
from test_deepseek_v4_accuracy import (  # noqa: E402
    _request_json, _stop_process_group, _unused_local_port, _wait_for_health,
)
from pypto_serving.model.tokenizer import load_tokenizer  # noqa: E402


def _validate(response: dict, log: str, target: int, codes: list[str]) -> None:
    usage = response.get("usage", {})
    count = usage.get("completion_tokens")
    if usage.get("prompt_tokens") != target or not isinstance(count, int) or not 1 < count <= 128:
        raise RuntimeError(f"invalid token accounting or no decode: {usage}")
    choices = response.get("choices", [])
    if (len(choices) != 1 or choices[0].get("finish_reason") != "stop"
            or choices[0]["message"]["content"].strip().splitlines() != codes):
        raise RuntimeError(f"expected three ordered code lines and natural EOS: {choices}")
    if not re.search(r"DSpark speculation progress[^\n]*\bverifies=[1-9]\d*\b", log):
        raise RuntimeError("no DSpark verification observed")
    marker = "DSpark prefill chunk: "
    events = [json.loads(line.split(marker, 1)[1]) for line in log.splitlines() if marker in line]
    if (len(events) != 2 * (target // 128)
            or len({(event["request_id"], event["group"]) for event in events}) != 1):
        raise RuntimeError("prefill chunk count or request/partition mismatch")
    for index in range(target // 128):
        started, completed = events[2 * index:2 * index + 2]
        if (started["status"] != "started" or completed != {**started, "status": "completed"}
                or (started["start"], started["logical_tokens"]) != (index * 128, 128)
                or started["physical_tokens"] < 128):
            raise RuntimeError(f"invalid prefill chunk {index}: {started}, {completed}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("16k", "1m"), required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--expected-codes", nargs=3, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    target = {"16k": 16_384, "1m": 1_048_448}[args.case]
    model = Path(os.environ.get("PYPTO_DSV4_DSPARK_MODEL_DIR", str(DEFAULT_MODEL_DIR)))
    tokenizer = load_tokenizer(model)
    content = args.prompt_file.read_text(encoding="utf-8")
    messages = [{"role": "user", "content": content}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    actual = len(tokenizer.encode(prompt))
    if actual != target or re.findall(r"\b[A-Z]+-[0-9]+\b", content) != args.expected_codes:
        parser.error(f"expected {target} input tokens and the three codes once in order; got {actual} tokens")
    print(json.dumps({
        "case": args.case, "prompt_tokens": actual,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    }), flush=True)
    if args.check_only:
        return
    devices = _task_devices()
    if len(devices) != 16:
        parser.error("these acceptance cases require 16 devices")
    port = _unused_local_port()
    command = _server_command(model, devices, port, num_speculative_tokens=7)
    for option, value in {
        "--max-model-len": "1048576", "--max-num-seqs": "1",
        "--ring-heap": "2147483648,2147483648,4294967296,8589934592",
    }.items():
        command[command.index(option) + 1] = value
    command += ["--generate-config", '{"ignore_eos": false}']
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    log_path = args.output_dir / "server.log"
    with log_path.open("w", encoding="utf-8") as server_log:
        process = subprocess.Popen(
            command, cwd=ROOT,
            env={**os.environ, "PYPTO_DSPARK_TRACE_PREFILL": "1", "PYPTO_CACHE": "1"},
            stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True, text=True,
        )
        try:
            startup = int(os.environ.get("PYPTO_DSPARK_SWEEP_STARTUP_TIMEOUT", "2400"))
            _wait_for_health(process, port, time.monotonic() + startup)
            offset = log_path.stat().st_size
            started = time.monotonic()
            timeout = int(os.environ.get("PYPTO_DSPARK_SWEEP_CASE_TIMEOUT", "7200"))
            # Send raw content: chat applies the template once and respects EOS.
            response = _request_json(
                process, port, started + timeout, endpoint="/v1/chat/completions",
                request_kind="precision chat", payload={
                    "model": MODEL_ID, "messages": messages,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "max_tokens": 128, "temperature": 0.0, "top_p": 1.0,
                },
            )
            (args.output_dir / "response.json").write_text(
                json.dumps(response, ensure_ascii=False) + "\n", encoding="utf-8",
            )
            with log_path.open("rb") as evidence:
                evidence.seek(offset)
                _validate(response, evidence.read().decode("utf-8", errors="replace"),
                          target, args.expected_codes)
            print(f"PASS {args.case}: {target // 128} chunks, {time.monotonic() - started:.3f}s", flush=True)
        finally:
            failed = sys.exc_info()[0] is not None
            cleanup_error = None
            for cleanup, resource in ((_stop_process_group, process), (_wait_for_device_reclaim, devices)):
                try:
                    cleanup(resource)
                except BaseException as exc:
                    print(f"WARNING: {cleanup.__name__}: {exc}", file=sys.stderr, flush=True)
                    cleanup_error = cleanup_error or exc
            if cleanup_error is not None and not failed:
                raise cleanup_error


if __name__ == "__main__":
    main()
