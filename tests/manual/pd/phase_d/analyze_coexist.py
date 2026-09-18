# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate paired results and same-device profiler overlap, without raw addresses."""
import argparse
from decimal import Decimal
import json
from pathlib import Path


def interval(event):
    start = Decimal(str(event["ts"]))
    return start, start + Decimal(str(event["dur"]))


def overlap_proof(compute, communication):
    intersections, examples = [], []
    for kernel in compute:
        a, b = interval(kernel)
        for transfer in communication:
            c, d = interval(transfer)
            start, stop = max(a, c), min(b, d)
            if start < stop:
                intersections.append((start, stop))
                if len(examples) < 3:
                    examples.append(dict(compute_start_us=str(a), transfer_start_us=str(c),
                                         overlap_us=str(stop - start)))
    merged = []
    for start, stop in sorted(intersections):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(stop, merged[-1][1]))
        else:
            merged.append((start, stop))
    return dict(pairs=len(intersections), union_us=str(sum((b - a for a, b in merged), Decimal(0))),
                examples=examples)


def analyze_trace(events):
    compute = [e for e in events if e.get("ph") == "X" and e.get("name") == "aicore_kernel_0_mix_aic"]
    pd = [e for e in events if e.get("ph") == "X" and e.get("name") == "batch_putAicpuKernel"]
    rdma = [e for e in events if e.get("ph") == "X" and e.get("name") == "RDMASend"
            and e.get("args", {}).get("link type") == "ROCE"]
    if not compute or not pd or not rdma:
        raise ValueError("missing compute, PD kernel, or RoCE events")
    compute_streams = {e["args"]["Physic Stream Id"] for e in compute}
    pd_streams = {e["args"]["Physic Stream Id"] for e in pd}
    rdma_streams = {e["args"]["stream id"] for e in rdma}
    if compute_streams & (pd_streams | rdma_streams):
        raise ValueError("compute and PD do not use disjoint device streams")
    kernel_overlap, rdma_overlap = overlap_proof(compute, pd), overlap_proof(compute, rdma)
    if not kernel_overlap["pairs"] or not rdma_overlap["pairs"]:
        raise ValueError("no same-device overlap observed")
    return dict(compute_streams=sorted(compute_streams), pd_streams=sorted(pd_streams),
                rdma_streams=sorted(rdma_streams), compute_tasks=len(compute), pd_tasks=len(pd),
                kernel_overlap=kernel_overlap, rdma_record_overlap=rdma_overlap)


def read_result(path, role):
    events = []
    for line in path.read_text().splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            events.append(item)
    results = [e for e in events if e.get("kind") == "coexist_result" and e.get("role") == role]
    if len(results) != 1 or results[0].get("status") != "ok":
        raise ValueError(f"missing unique successful {role} result")
    if any(e.get("kind") == "coexist_failed_closed" for e in events):
        raise ValueError("failure event in successful log")
    return results[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--sender-log", type=Path, required=True)
    parser.add_argument("--receiver-log", type=Path, required=True)
    args = parser.parse_args()
    sender, receiver = read_result(args.sender_log, "sender"), read_result(args.receiver_log, "receiver")
    for key in ("run_id", "ranks", "rounds"):
        if sender[key] != receiver[key]:
            raise ValueError(f"pair identity mismatch: {key}")
    traces = []
    for path in sorted(args.trace_dir.rglob("msprof_*.json")):
        events = json.loads(path.read_text())
        if isinstance(events, dict):
            events = events["traceEvents"]
        # One PROF directory per chip owner, with its own device-local timeline.
        devices = [entry.name for entry in path.parent.parent.glob("device_*") if entry.is_dir()]
        if len(devices) != 1:
            raise ValueError("ambiguous profiler device identity")
        traces.append(dict(profile=path.parent.parent.name, device=devices[0], **analyze_trace(events)))
    if len(traces) != sender["ranks"] or len({t["device"] for t in traces}) != sender["ranks"]:
        raise ValueError("not all sender devices have independent-stream overlap evidence")
    print(json.dumps(dict(run_id=sender["run_id"], status="ok", sender=sender, receiver=receiver,
                          sender_device_overlap=traces,
                          note="RoCE records are profiler events, not an independent wire-time measurement; "
                               "receiver device streams are not profiled in this run."), indent=2))


if __name__ == "__main__":
    main()
