# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Verify cold-then-hit PC1 metric deltas from three composite snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not {"prefill", "decode"} <= set(value):
        raise ValueError(f"{path} is not a composite P/D metrics snapshot")
    return value


def _counter(snapshot: dict, role: str, name: str) -> int:
    value = snapshot[role].get("counters", {}).get(name, 0)
    if type(value) is not int or value < 0:
        raise ValueError(f"invalid {role} counter {name!r}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True, type=Path)
    parser.add_argument("--after-cold", required=True, type=Path)
    parser.add_argument("--after-hit", required=True, type=Path)
    args = parser.parse_args()
    before = _load(args.before)
    cold = _load(args.after_cold)
    hit = _load(args.after_hit)

    cold_requests = _counter(cold, "decode", "prefix.cold_requests") - _counter(
        before, "decode", "prefix.cold_requests"
    )
    hit_requests = _counter(hit, "decode", "prefix.hit_requests") - _counter(
        cold, "decode", "prefix.hit_requests"
    )
    hit_tokens = _counter(hit, "decode", "prefix.hit_tokens") - _counter(
        cold, "decode", "prefix.hit_tokens"
    )
    cold_bytes = _counter(cold, "prefill", "transfer.bytes") - _counter(
        before, "prefill", "transfer.bytes"
    )
    hit_bytes = _counter(hit, "prefill", "transfer.bytes") - _counter(
        cold, "prefill", "transfer.bytes"
    )
    result = {
        "cold_requests": cold_requests,
        "hit_requests": hit_requests,
        "hit_tokens": hit_tokens,
        "cold_transfer_bytes": cold_bytes,
        "hit_transfer_bytes": hit_bytes,
        "pass": (
            cold_requests == 1
            and hit_requests == 1
            and hit_tokens >= 128
            and 0 < hit_bytes < cold_bytes
        ),
    }
    print(json.dumps(result, sort_keys=True))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
