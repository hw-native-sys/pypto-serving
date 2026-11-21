# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate lowlat parent identities and merge eight Host lanes with serving."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import statistics


REQUIRED_CATEGORIES = {"scheduler", "serving", "worker", "executor", "kernel", "lowlat"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")


def attrs(span):
    return dict(re.findall(r"(\w+)=([^\s]+)", span.attrs))


def end(event):
    return event["ts"]+event.get("dur", 0)


def contains(outer, inner):
    return outer["ts"] <= inner["ts"] and end(inner) <= end(outer)


def native_parents(events, run):
    """Join submit/capture/retirement using frame and explicit reservations."""
    spans = [e for e in events if e.get("ph") == "X"]
    require(not any(e.get("args", {}).get("success") is False for e in spans), "failed profiled operation")
    named = lambda name: sorted((e for e in spans if e["name"] == name), key=lambda e: e["ts"])
    submits, captures = named("lowlat.native.submit"), named("lowlat.native.capture")
    dispatches, retirements = named("lowlat.native.dispatch"), named("lowlat.native.retire")
    releases = named("lowlat.native.release")
    require(submits and len(submits) == len(captures) == len(dispatches) == len(retirements),
            "native submit/capture/dispatch/retirement counts differ")
    by_frame = defaultdict(list)
    for capture in captures:
        by_frame[capture["args"]["frame_id"]].append(capture)
    parents, seen, current_end, previous_epoch = [], set(), len(run["prompt_ids"]), None
    prior_frame_end = {}
    remaining = run["output_tokens"]-1
    for number, submit in enumerate(submits, 1):
        sa = submit["args"]
        require(type(sa.get("runtime_run_id")) is int and sa["runtime_run_id"] > 0,
                "native submit lacks pinned runtime run identity")
        ds = [d for d in dispatches if contains(d, submit)]
        require(len(ds) == 1, "native submit has no unique dispatch reservation")
        da = ds[0]["args"]
        rid = da["reservation_id"]
        require(rid not in seen, "duplicate reservation ID")
        seen.add(rid)
        require(sa["frame_id"] == da["frame_id"] and sa["cycles"] == da["cycles"],
                "frame/cycles differ from reservation")
        frame = sa["frame_id"]
        require(prior_frame_end.get(frame, 0) <= submit["ts"], "frame reused before predecessor retirement")
        require(by_frame[frame], "missing capture for native frame")
        capture = by_frame[frame].pop(0)
        ca = capture["args"]
        retire = [r for r in retirements if r["args"].get("reservation_id") == rid and contains(r, capture)]
        require(len(retire) == 1 and end(submit) <= capture["ts"], "capture/retirement order differs")
        release = [r for r in releases
                   if r["args"].get("frame_id") == frame
                   and capture["ts"] <= r["ts"] <= end(retire[0])]
        require(len(release) == 1, "native frame release is missing or ambiguous")
        require(ca["physical_input_start"] == current_end, "native physical input has a gap")
        require(previous_epoch is None or ca["epoch"] == previous_epoch, "native epoch has a gap")
        advances, accepted, ids = ca["advances"], ca["accepted_drafts"], ca["emitted_ids"]
        cycles = sa["cycles"]
        require(cycles in (1, 2) and len(advances) == len(accepted) == cycles,
                "capture cycle count differs from reservation")
        require(all(type(a) is int and 0 <= a <= 5 and advance == a+1
                    for advance, a in zip(advances, accepted)), "invalid K5 acceptance/advance")
        require(ca["physical_input_end"] == current_end+sum(advances), "physical advance differs")
        require(ca["emitted"] == len(ids) and 0 < len(ids) <= da["max_output_tokens"]
                and all(type(token) is int and 0 <= token < 129280 for token in ids), "invalid emitted tokens")
        remaining -= len(ids)
        require(remaining >= 0 and ca["remaining_output_tokens"] == remaining, "native remaining budget differs")
        require(ca["next_epoch"] == ca["epoch"]+cycles, "next native epoch differs")
        current_end, previous_epoch = ca["physical_input_end"], ca["next_epoch"]
        prior_frame_end[frame] = end(release[0])
        parents.append(dict(step=number, reservation_id=rid, request_id=da["request_id"],
            generation=da["generation"], runtime_run_id=sa["runtime_run_id"], cycles=cycles,
            frame_id=frame, emitted=len(ids), emitted_ids=ids, accepted_drafts=sum(accepted),
            physical_input_start=ca["physical_input_start"], physical_input_end=current_end,
            submit_ts_us=submit["ts"], submit_end_us=end(submit),
            capture_ts_us=capture["ts"], frame_release_ts_us=end(release[0]),
            retire_end_us=end(retire[0])))
    require(remaining == 0, "profile did not emit the complete post-prefill budget")
    require(len({p["request_id"] for p in parents}) == 1 and len({p["generation"] for p in parents}) == 1,
            "profile contains more than one active request/generation")
    require(len({p["runtime_run_id"] for p in parents}) == len(parents), "duplicate native runtime run ID")
    require({p["frame_id"] for p in parents} <= {0, 1}, "invalid output frame")
    live, peak = 0, 0
    for _, change in sorted([(p["submit_ts_us"], 1) for p in parents]
                           +[(p["retire_end_us"], -1) for p in parents]):
        live += change
        peak = max(peak, live)
    require(0 < peak <= 2, f"accepted parent depth exceeds two: {peak}")
    return parents, peak


def analyze(directory):
    directory = Path(directory)
    validation = directory/"skill-profile-validation.json"
    dump(validation, {"valid": False, "reason": "analysis incomplete"})
    try:
        result = _analyze(directory)
        dump(validation, result)
        return result
    except BaseException as error:
        dump(validation, {"valid": False, "reason": str(error)})
        raise


def decode_swimlane(combined, parents, origin):
    """Export the native decode window, retaining measured rank/run identities."""
    identities = {p["runtime_run_id"] for p in parents}
    start = min(p["submit_ts_us"] for p in parents) - origin
    stop = max(p["retire_end_us"] for p in parents) - origin
    # Dispatch starts slightly before submit; keep that first reservation too.
    dispatches = [e["ts"] for e in combined if e.get("name") == "lowlat.native.dispatch"]
    start = min([start, *dispatches])
    selected = []
    for event in combined:
        if event.get("ph") == "M":
            selected.append(dict(event))
            continue
        if "ts" not in event or "prefill" in event.get("name", ""):
            continue
        identity = event.get("args", {}).get("runtime_run_id")
        if identity is not None and identity not in identities:
            continue
        begin, finish = max(start, event["ts"]), min(stop, end(event))
        if finish <= begin:
            continue
        value = {**event, "ts": begin-start}
        if event.get("ph") == "X":
            value["dur"] = finish-begin
            if begin != event["ts"] or finish != end(event):
                value["args"] = {**event.get("args", {}), "clipped_to_decode_window": True}
        selected.append(value)
    return {"traceEvents": selected, "displayTimeUnit": "ms", "metadata": {
        "scope": "native decode only; excludes prefill and its first output token",
        "clock": "shared host monotonic; timestamps in microseconds",
        "origin_us": origin+start, "host_device_lanes": 8, "device_timing_available": False}}


def _analyze(directory):
    # Parsing is CPU-only and uses the pinned runtime's canonical grammar.
    from simpler_setup.tools.strace_timing import (
        count_record_heads, group_invocations, parse_drop_summaries, parse_spans, to_chrome_trace,
    )

    load = lambda name: json.loads((directory/name).read_text())
    run = load("run-summary.json")
    require(run["status"] == "PASS" and run.get("source_unchanged") is True
            and run.get("server_exit") in (0, -15), "profile run/source/shutdown did not pass")
    require(len(run["devices"]) == len(set(run["devices"])) == 8 and run["device_strace"] is False
            and run["compile_cache"] is False, "invalid lowlat profile contract")
    reference_files=run.get("reference_response_files",["warmup-responses.json","baseline-responses.json"])
    require(reference_files in (["warmup-responses.json"],["warmup-responses.json","baseline-responses.json"]),
            "invalid unprofiled reference list")
    responses = [load(name) for name in (*reference_files,"responses.json")]
    matches = run.get("reference_matches_workload", True)
    for index, response in enumerate(responses):
        formal = index == len(responses)-1
        expected_prompt = len(run["prompt_ids"]) if formal or matches else run["warmup_prompt_tokens"]
        expected_output = run["output_tokens"] if formal or matches else run["warmup_output_tokens"]
        require(response["usage"]["completion_tokens"] == expected_output
                and response["usage"]["prompt_tokens"] == expected_prompt
                and response["finish_reason"] == "length", "request token counts differ")
    if matches:
        require(len({r["text"] for r in responses}) == 1, "profiling changed greedy output text")
    events = load("serving-trace/trace.json")["traceEvents"]
    spans = [e for e in events if e.get("ph") == "X"]
    require(REQUIRED_CATEGORIES <= {e.get("cat") for e in spans}, "missing serving trace categories")
    parents, peak = native_parents(events, run)
    submits = [e for e in spans if e["name"] in ("lowlat.native.submit", "lowlat.prefill.submit")]
    by_run = {}
    for event in submits:
        identity = event["args"].get("runtime_run_id")
        require(type(identity) is int and identity > 0 and identity not in by_run,
                "missing or duplicate runtime submit identity")
        by_run[identity] = event
    require(len(submits) > len(parents), "missing formal prefill submission")
    text = (directory/"server.log").read_text(errors="replace")
    pid_to_device = {int(p): int(d) for p, d in re.findall(r"\[chip_process pid=(\d+) dev=(\d+)\] ready", text)}
    require(len(pid_to_device) == 8 and set(pid_to_device.values()) == set(run["devices"]),
            "actual eight-chip PID/device mapping differs from allocation")
    lines = text.replace("[STRACE]", "\n[STRACE]").splitlines()
    raw = list(parse_spans(lines))
    require(raw, "no Host STRACE records; check runtime logging before Worker initialization")
    require(count_record_heads(lines) == len(raw), "torn Host STRACE records")
    require(not any(row["total"] for row in parse_drop_summaries(lines).values()), "dropped Host STRACE records")
    require(not any(s.is_device for s in raw), "Device STRACE is not a Host timing")
    invocations = group_invocations(raw)
    selected, executions = [], defaultdict(list)
    start, stop = min(e["ts"] for e in spans), max(end(e) for e in spans)
    for invocation in invocations:
        root = invocation.root()
        if root is None or invocation.pid not in pid_to_device:
            continue
        identity = int(attrs(root).get("run_id", -1))
        if identity not in by_run:
            require(not (root.ts/1000 < stop and (root.ts+root.dur)/1000 > start
                        and any(name in invocation.by_name()
                                for name in ("chip.run.runner_run", "simpler_run.runner_run"))),
                    "unattributed chip execution overlaps formal profile")
            continue
        require(root.ts/1000 >= start and (root.ts+root.dur)/1000 <= stop,
                "Host execution falls outside formal profile window")
        selected.append(invocation)
        if any(n in invocation.by_name() for n in ("chip.run.runner_run", "simpler_run.runner_run")):
            executions[identity, pid_to_device[invocation.pid]].append(invocation)
    for identity in by_run:
        for device in run["devices"]:
            require(len(executions[identity, device]) == 1,
                    f"expected one completed rank invocation: run={identity} device={device}")
    labels, hid_labels = {}, {}
    for identity, event in by_run.items():
        label = "prefill.cp8" if event["name"] == "lowlat.prefill.submit" else "native."+str(event["args"]["cycles"])+"cycle"
        labels[identity] = label
        for device in run["devices"]:
            inv = executions[identity, device][0]
            key = (inv.pid, inv.hid)
            require(key not in hid_labels or hid_labels[key] == label, "callable digest changes program classification")
            hid_labels[key] = label
    for parent in parents:
        ranks = []
        for device in run["devices"]:
            inv = executions[parent["runtime_run_id"], device][0]
            names = inv.by_name()
            row = {"device": device, "source_pid": inv.pid, "source_invocation": inv.inv,
                   "host_ms": inv.root().dur/1e6}
            for field, aliases in {"bind_ms": ("chip.run.bind", "simpler_run.bind"),
                                   "runner_run_ms": ("chip.run.runner_run", "simpler_run.runner_run"),
                                   "validate_ms": ("chip.run.validate", "simpler_run.validate")}.items():
                present = [names[n] for n in aliases if n in names]
                require(len(present) == 1, f"missing/ambiguous {field} for device {device}")
                row[field] = present[0].dur/1e6
            require((inv.root().ts+inv.root().dur)/1000 <= parent["capture_ts_us"],
                    "capture precedes whole-rank publication")
            ranks.append(row)
        parent["per_rank"] = ranks
        parent["critical_host_ms"] = max(r["host_ms"] for r in ranks)
    dump(directory/"simpler-swimlane.json", to_chrome_trace(selected))
    origin = min([float(e["ts"]) for e in events if "ts" in e]+[i.root().ts/1000 for i in selected])
    host_pid = max(int(e["pid"]) for e in events if "pid" in e)+1
    combined = []
    for event in events:
        event = dict(event)
        if "ts" in event:
            event["ts"] -= origin
        combined.append(event)
    combined.append(dict(ph="M", name="process_name", pid=host_pid, args={"name": "Lowlat Host STRACE: 8 NPUs"}))
    for device in run["devices"]:
        combined.append(dict(ph="M", name="thread_name", pid=host_pid, tid=device, args={"name": f"device {device}"}))
    step_by_run = {p["runtime_run_id"]: p for p in parents}
    for invocation in selected:
        identity = int(attrs(invocation.root())["run_id"])
        parent = step_by_run.get(identity)
        extra = {"runtime_run_id": identity, "program": labels[identity],
                 "source_pid": invocation.pid, "source_invocation": invocation.inv}
        if parent:
            extra.update({k: parent[k] for k in ("step", "reservation_id", "cycles", "frame_id", "emitted")})
        for span in invocation.spans:
            name = span.name
            if span.depth == 0:
                name = labels[identity]+(f" parent {parent['step']}" if parent else "")
            combined.append(dict(ph="X", name=name, cat="strace.host", pid=host_pid,
                tid=pid_to_device[span.pid], ts=span.ts/1000-origin, dur=span.dur/1000,
                args={**extra, "strace_name": span.name, "depth": span.depth, "attrs": span.attrs, "hid": span.hid}))
    parent_pid = host_pid+1
    combined.append(dict(ph="M", name="process_name", pid=parent_pid,
                         args={"name": "Lowlat parent lifetimes (elapsed, not device time)"}))
    for frame in (0, 1):
        combined.append(dict(ph="M", name="thread_name", pid=parent_pid, tid=frame, args={"name": f"frame {frame}"}))
    for parent in parents:
        combined.append(dict(ph="X", name=f"parent {parent['step']}: {parent['cycles']} cycles / {parent['emitted']} tokens",
            cat="lowlat.parent", pid=parent_pid, tid=parent["frame_id"], ts=parent["submit_ts_us"]-origin,
            dur=parent["retire_end_us"]-parent["submit_ts_us"], args={k: v for k, v in parent.items() if k != "per_rank"}))
    dump(directory/"serving-strace-swimlane.json", {"traceEvents": combined, "displayTimeUnit": "ms",
        "metadata": {"clock": "shared host monotonic; timestamps in microseconds", "origin_us": origin,
                     "host_device_lanes": 8, "device_timing_available": False}})
    dump(directory/"decode-only-swimlane.json", decode_swimlane(combined, parents, origin))
    baseline, profiled = responses[-2:]
    summary = dict(run_id=run["run_id"], workload=run, parent_count=len(parents),
        cycles=sum(p["cycles"] for p in parents), post_prefill_tokens=sum(p["emitted"] for p in parents),
        cycle_histogram=dict(Counter(p["cycles"] for p in parents)),
        accepted_drafts=sum(p["accepted_drafts"] for p in parents), parents=parents,
        baseline_tpot_ms=baseline["tpot_ms"] if matches else None, profile_tpot_ms=profiled["tpot_ms"],
        instrumentation_delta_percent=(profiled["tpot_ms"]/baseline["tpot_ms"]-1)*100 if matches else None,
        parent_critical_host_mean_ms=statistics.mean(p["critical_host_ms"] for p in parents),
        host_lanes=8, device_timing_available=False,
        timing_scope="Host runner_run includes dispatch/execution/wait; overlapping parent times are not additive")
    dump(directory/"profile-summary.json", summary)
    (directory/"profile-summary.md").write_text(
        "# Lowlat single-batch Host profile\n\n"
        f"Validated {len(parents)} parents, {summary['cycles']} cycles and {summary['post_prefill_tokens']} post-prefill tokens.\n\n"
        f"Comparable unprofiled TPOT: {summary['baseline_tpot_ms']} ms/token; diagnostic profile: {profiled['tpot_ms']:.3f} ms/token.\n\n"
        "A different short warmup is not a performance or output-equivalence reference.\n\n"
        "Open serving-strace-swimlane.json in Perfetto. All eight Host tracks share the serving monotonic clock. "
        "Open decode-only-swimlane.json for the native decode window. "
        "Parent lifetimes overlap and must not be summed. Host runner_run is not pure device time.\n")
    roots = [e for e in combined if e.get("cat") == "strace.host" and e["args"]["depth"] == 0
             and e["args"]["runtime_run_id"] in step_by_run
             and any(i.pid == e["args"]["source_pid"] and i.inv == e["args"]["source_invocation"]
                     for rows in executions.values() for i in rows)]
    require(len(roots) == len(parents)*8, "combined trace lost native Host roots")
    require({e["tid"] for e in roots} == set(run["devices"]), "combined trace has incomplete lanes")
    return dict(valid=True, scope="B1/K5/M2/rolling Serving + eight Host STRACE lanes",
                parent_count=len(parents), cycles=summary["cycles"], post_prefill_tokens=summary["post_prefill_tokens"],
                host_lanes=8, identity_join="native run_id + rank PID + explicit reservation/frame", source_unchanged=True,
                device_timing_available=False, frames=sorted({p["frame_id"] for p in parents}),
                cycles_seen=sorted({p["cycles"] for p in parents}), max_accepted_parents=peak)
