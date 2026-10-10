# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate the fixed DSpark B16/8K/256 profile artifact contract."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    root = parser.parse_args().artifact_dir.resolve()

    required_files = (
        "warmup-responses.json",
        "responses.json",
        "performance_summary.json",
        "server.log",
        "serving-trace/trace.json",
        "simpler-swimlane.json",
        "serving-strace-swimlane.json",
        "profile-summary.json",
        "profile-summary.md",
    )
    missing = [name for name in required_files if not (root / name).is_file()]
    if missing:
        raise RuntimeError(f"missing profile artifacts: {missing}")

    for name in ("warmup-responses.json", "responses.json"):
        responses = json.loads((root / name).read_text())
        if len(responses) != 16 or any(
            int(item.get("prompt_tokens", -1)) != 8192
            or int(item.get("completion_tokens", -1)) != 256
            or item.get("finish_reason") != "length"
            for item in responses
        ):
            raise RuntimeError(
                f"{name} does not contain sixteen successful Seq8192/Output256 responses"
            )

    payload = json.loads((root / "serving-strace-swimlane.json").read_text())
    events = payload["traceEvents"]
    spans = [event for event in events if event.get("ph") == "X"]
    categories = Counter(str(event.get("cat", "")) for event in spans)
    for required in ("scheduler", "worker", "executor", "kernel", "serving", "strace.host"):
        if not categories[required]:
            raise RuntimeError(f"missing combined trace category: {required}")

    host = [event for event in spans if event.get("cat") == "strace.host"]
    devices = sorted({int(event["tid"]) for event in host})
    if len(devices) != 16:
        raise RuntimeError(f"expected sixteen Host STRACE lanes, got {devices}")
    decode_roots = [
        event
        for event in host
        if event.get("args", {}).get("callable")
        in {
            "decode.main+verify",
            "decode.main+verify+drafter+markov+state_commit",
        }
        and event.get("args", {}).get("strace_name") in {"chip.run", "simpler_run"}
    ]
    steps = sorted({int(event["args"]["decode_step"]) for event in decode_roots})
    if (
        not steps
        or steps != list(range(1, len(steps) + 1))
        or len(decode_roots) != len(steps) * 16
    ):
        raise RuntimeError(
            f"unexpected fused decode coverage: steps={len(steps)} roots={len(decode_roots)}"
        )
    server_log = (root / "server.log").read_text(errors="replace")
    if "clk=dev" in server_log:
        raise RuntimeError("Device STRACE leaked into the Host-only skill artifact")
    if "DSpark speculation progress" not in server_log:
        raise RuntimeError("server log does not prove that DSpark speculation dispatched")

    validation = {
        "valid": True,
        "workload": (
            "GBS16/DP4/EP16/TP4/active-DP-group-batch4/Seq8192/Output256/"
            "DSpark7/temperature0"
        ),
        "categories": dict(sorted(categories.items())),
        "host_devices": devices,
        "decode_steps": len(steps),
        "decode_rank_roots": len(decode_roots),
    }
    (root / "skill-profile-validation.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
