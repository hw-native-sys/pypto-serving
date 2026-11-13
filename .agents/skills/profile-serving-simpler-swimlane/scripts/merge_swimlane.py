# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Merge serving and native Simpler host swimlanes on their shared monotonic clock."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def merge(serving, simpler):
    events = serving["traceEvents"]
    durations = [e for e in events if e.get("ph") == "X" and e.get("dur", 0) > 0]
    if not durations:
        raise ValueError("Serving trace has no duration events")
    start = min(e["ts"] for e in durations)
    end = max(e["ts"] + e["dur"] for e in durations)
    if simpler.get("unalignedDeviceSpans") or any(
        (e.get("args") or {}).get("clk") == "dev" for e in simpler["traceEvents"]
    ):
        raise ValueError("Expected host-only Simpler trace; device clocks cannot be merged")
    selected = [copy.deepcopy(e) for e in simpler["traceEvents"]
                if e.get("ph") != "M" and "ts" in e
                and e["ts"] <= end and e["ts"] + e.get("dur", 0) >= start]
    if not any(e.get("ph") == "X" and e.get("dur", 0) > 0 for e in selected):
        raise ValueError("No Simpler duration overlaps serving; check the run and clock sources")

    # Window filtering can orphan native cross-thread flows.
    flows = {}
    for e in selected:
        if e.get("ph") in ("s", "t", "f"):
            key = (e.get("cat"), e.get("scope"), str(e.get("id")))
            flows.setdefault(key, set()).add(e["ph"])
    selected = [e for e in selected if e.get("ph") not in ("s", "t", "f")
                or {"s", "f"} <= flows[(e.get("cat"), e.get("scope"), str(e.get("id")))]]
    pids = {e["pid"] for e in selected}
    metadata = [copy.deepcopy(e) for e in simpler["traceEvents"]
                if e.get("ph") == "M" and e.get("pid") in pids]
    ids = {e.get("pid") for e in events + simpler["traceEvents"]}
    base = max((int(pid) for pid in ids if str(pid).isdecimal()), default=0) + 1
    mapping = {pid: base + i for i, pid in enumerate(sorted(pids, key=str))}
    for e in metadata + selected:
        original = e["pid"]
        e["pid"] = mapping[original]
        args = e.setdefault("args", {})
        args["original_pid"] = original
        if e.get("name") == "process_name":
            args["name"] = "Simpler | " + args.get("name", str(original))
        if e.get("ph") != "M":
            e["cat"] = "strace." + (e.get("cat") or "host")
        if "id" in e:
            e["id"] = "simpler:" + str(e["id"])
    return {"traceEvents": events + metadata + selected, "displayTimeUnit": "ms"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serving", required=True, type=Path)
    parser.add_argument("--simpler", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.resolve() in {args.serving.resolve(), args.simpler.resolve()}:
        parser.error("Output must not overwrite either input")
    try:
        serving = json.loads(args.serving.read_text(encoding="utf-8"))
        simpler = json.loads(args.simpler.read_text(encoding="utf-8"))
        combined = merge(serving, simpler)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(combined), encoding="utf-8")
        print(f"Wrote {args.output}: {len(combined['traceEvents'])} events")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Merge failed: {exc}\n")


if __name__ == "__main__":
    main()
