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
import re
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
    num_speculative_tokens: int = 0


# One DSpark server parks tens of GiB of pooled arenas per card, and the
# driver reclaims a torn-down server's HBM asynchronously -- process exit is
# not the boundary.  A clean card idles around 3 GiB used (driver baseline),
# while any real residual is tens of GiB, so 8 GiB separates them cleanly.
DEVICE_RECLAIMED_HBM_MIB = 8192
_DEVICE_HBM_ROW = re.compile(
    r"^\|\s*\d+\s+(\d+)\s+\|\s*[0-9A-F]{4}:[0-9A-F]{2}:[0-9A-F]{2}\.[0-9A-F]\s+\|"
    r"\s*[\d.]+\s+\d+\s*/\s*\d+\s+(\d+)\s*/\s*\d+\s+\|$",
    re.MULTILINE,
)


def _device_hbm_used_mib() -> dict[int, int]:
    """Per-device used HBM in MiB from ``npu-smi info``; empty when unusable."""
    try:
        completed = subprocess.run(
            ["npu-smi", "info"], capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    return {
        int(device_id): int(used_mib)
        for device_id, used_mib in _DEVICE_HBM_ROW.findall(completed.stdout)
    }


def _wait_for_device_reclaim(devices: tuple[int, ...], *, timeout_s: int = 900) -> None:
    """Block until the previous server's HBM is back near the driver baseline.

    The next case boots a fresh 16-card server that needs nearly the whole
    card (the K=7 boot measured 37 KV slots of headroom on a clean device);
    starting it while any device still holds the previous server's residual
    HBM fails its weight upload with a device OOM long before an assertion
    could name the cause.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        used = _device_hbm_used_mib()
        residual = {
            device: used[device]
            for device in devices
            if device in used and used[device] > DEVICE_RECLAIMED_HBM_MIB
        }
        if not residual:
            if len(used) < len(devices):
                # npu-smi is present but unparseable/unavailable: proceed on
                # the raw teardown timing rather than failing the guard.
                print(
                    "WARNING: could not read per-device HBM usage; skipping "
                    "the post-teardown reclaim wait",
                    flush=True,
                )
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"devices still hold the previous server's HBM after {timeout_s}s: "
                f"{residual} MiB used"
            )
        time.sleep(15)


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
    # Milestone 2: the same prompt at K=7.  Assertions stay contract-only
    # (token accounting plus proof the speculative path really dispatched);
    # acceptance changes speed, never the served contract.
    DSparkCase(
        case_id="palace-64-128-k7",
        prompt=_MTP_64_128.prompt,
        prompt_tokens=_MTP_64_128.prompt_tokens,
        max_new_tokens=_MTP_64_128.max_new_tokens,
        num_speculative_tokens=7,
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


def _server_command(
    model_dir: Path,
    devices: tuple[int, ...],
    port: int,
    *,
    num_speculative_tokens: int = 0,
) -> list[str]:
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
        json.dumps(
            {"method": "dspark", "num_speculative_tokens": num_speculative_tokens}
        ),
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
                _server_command(
                    model_dir,
                    devices,
                    port,
                    num_speculative_tokens=case.num_speculative_tokens,
                ),
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
        # A following case boots a fresh 16-card server that needs nearly
        # the whole card; wait out the previous server's async HBM reclaim.
        _wait_for_device_reclaim(devices)
        if case.num_speculative_tokens:
            # The K=7 run must really have dispatched the drafter/markov
            # chain: the runner logs acceptance progress unconditionally at
            # its first verify step, so the line exists however few (or many)
            # steps a fast or slow run takes.
            log_text = log_path.read_text(encoding="utf-8")
            assert "DSpark speculation progress" in log_text, (
                "no acceptance progress line in the server log"
            )
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

    k7 = _server_command(
        tmp_path, tuple(range(DSPARK_EP_SIZE)), 12345, num_speculative_tokens=7
    )
    assert json.loads(k7[k7.index("--speculative-config") + 1]) == {
        "method": "dspark",
        "num_speculative_tokens": 7,
    }
