# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""DSpark (deepseek_v4_flash_dspark) target-model HTTP generation guard.

Serves the DSpark W8A8 checkpoint on the canonical 16-card TP4/DP4/EP16
topology through the standard HTTP path and checks greedy generation.  The
process/HTTP harness is shared with the DeepSeek V4 MTP guard; what differs is
the server command (page 32, the dspark speculative-config method, the
prefill-tuned ring heap) and the 16-device task contract.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

# The shared HTTP/process harness lives beside this file; put that directory
# on sys.path for the sibling import (pytest does not insert it for us).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_deepseek_v4_accuracy import (  # noqa: E402
    MTP_CASES,
    OVERALL_TIMEOUT_SECONDS,
    _print_server_log,
    _request_completion,
    _stop_process_group,
    _unused_local_port,
    _wait_for_health,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "dsv4-flash-dspark-w8a8"
DEFAULT_MODEL_DIR = Path("/data/models/dsv4-flash-0731-dspark-w8a8")
DSPARK_EP_SIZE = int(os.environ.get("PYPTO_DSPARK_EP_SIZE", "16"))
DSPARK_TP_SIZE = 4
if DSPARK_EP_SIZE not in (4, 8, 16):
    raise ValueError("PYPTO_DSPARK_EP_SIZE must be one of 4, 8, or 16")
DSPARK_DP_SIZE = DSPARK_EP_SIZE // DSPARK_TP_SIZE
# Prefill's rebalanced per-scope-depth ring heap (pypto-lib#1073). Overridable
# for bring-up sweeps against the per-family harness defaults.
DSPARK_RING_HEAP = os.environ.get(
    "PYPTO_DSPARK_RING_HEAP", "2147483648,2147483648,4294967296,8589934592"
)


@dataclass(frozen=True)
class DSparkCase:
    """One greedy generation case over the DSpark target model."""

    case_id: str
    prompt: str
    prompt_tokens: int
    max_new_tokens: int


# Mirror the MTP guard's prompt_tokens=64 / max_new_tokens=128 gate (its K=1
# case): the same Palace Museum prompt at the same lengths, so both variants
# gate the same hardware feature with the same request shape.
_MTP_64_128 = MTP_CASES[0]
GREEDY_CASES = (
    DSparkCase(
        case_id="palace-64-128",
        prompt=_MTP_64_128.prompt,
        prompt_tokens=_MTP_64_128.prompt_tokens,
        max_new_tokens=_MTP_64_128.max_new_tokens,
    ),
)


def _task_devices() -> tuple[int, ...]:
    raw_devices = os.environ.get("TASK_DEVICE", "")
    try:
        devices = tuple(int(value.strip()) for value in raw_devices.split(",") if value.strip())
    except ValueError:
        pytest.fail(
            f"TASK_DEVICE must contain comma-separated integer device IDs, got {raw_devices!r}"
        )
    if (
        len(devices) != DSPARK_EP_SIZE
        or len(set(devices)) != DSPARK_EP_SIZE
        or any(d < 0 for d in devices)
    ):
        pytest.fail(
            "TASK_DEVICE must contain exactly "
            f"{DSPARK_EP_SIZE} unique non-negative device IDs, got {raw_devices!r}"
        )
    return devices


def _server_command(model_dir: Path, devices: tuple[int, ...], port: int) -> list[str]:
    # Keep these serving options aligned with docs/dev/model/deepseek-v4-dspark.md.
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
        str(DSPARK_DP_SIZE),
        "--ep",
        str(DSPARK_EP_SIZE),
        "--tp",
        str(DSPARK_TP_SIZE),
        "--block-size",
        "32",
        "--max-model-len",
        "1024",
        "--max-num-seqs",
        "8",
        "--max-num-batched-tokens",
        "8192",
        "--long-prefill-token-threshold",
        "128",
        "--speculative-config",
        json.dumps({"method": "dspark", "num_speculative_tokens": 0}),
        "--no-enable-prefix-caching",
        "--ring-heap",
        DSPARK_RING_HEAP,
        "--port",
        str(port),
        "--show-startup-logs",
    ]


@pytest.mark.parametrize("case", GREEDY_CASES, ids=[case.case_id for case in GREEDY_CASES])
def test_dspark_http_greedy_generation(tmp_path: Path, case: DSparkCase) -> None:
    model_dir_env = os.environ.get("PYPTO_DSV4_DSPARK_MODEL_DIR")
    model_dir = Path(model_dir_env) if model_dir_env else DEFAULT_MODEL_DIR
    if not model_dir.is_dir():
        pytest.fail(
            "DSpark W8A8 checkpoint not found (set PYPTO_DSV4_DSPARK_MODEL_DIR): "
            f"{model_dir}"
        )
    devices = _task_devices()
    port = _unused_local_port()
    log_path = tmp_path / f"dspark-{case.case_id}-server.log"
    deadline = time.monotonic() + OVERALL_TIMEOUT_SECONDS

    try:
        with log_path.open("w", encoding="utf-8") as server_log:
            process = subprocess.Popen(
                _server_command(model_dir, devices, port),
                cwd=ROOT,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
            )
            try:
                _wait_for_health(process, port, deadline)
                response = _request_completion(
                    process,
                    port,
                    deadline,
                    prompt=case.prompt,
                    max_new_tokens=case.max_new_tokens,
                    model=MODEL_ID,
                )
                print(f"DSpark case={case.case_id} completion: {response}", flush=True)
                assert response.get("model") == MODEL_ID
                choices = response.get("choices")
                assert isinstance(choices, list) and len(choices) == 1
                assert choices[0].get("finish_reason") == "length"
                usage = response.get("usage", {})
                assert usage.get("prompt_tokens") == case.prompt_tokens
                assert usage.get("completion_tokens") == case.max_new_tokens
            finally:
                _stop_process_group(process)
    except BaseException:
        _print_server_log(log_path)
        raise


def test_server_command_pins_the_dspark_contract(tmp_path) -> None:
    command = _server_command(tmp_path, tuple(range(DSPARK_EP_SIZE)), 12345)

    assert command[command.index("--dp") + 1] == str(DSPARK_DP_SIZE)
    assert command[command.index("--ep") + 1] == str(DSPARK_EP_SIZE)
    assert command[command.index("--tp") + 1] == str(DSPARK_TP_SIZE)
    assert command[command.index("--block-size") + 1] == "32"
    assert json.loads(command[command.index("--speculative-config") + 1]) == {
        "method": "dspark",
        "num_speculative_tokens": 0,
    }
    assert "--no-enable-prefix-caching" in command
    assert command[command.index("--ring-heap") + 1] == DSPARK_RING_HEAP
