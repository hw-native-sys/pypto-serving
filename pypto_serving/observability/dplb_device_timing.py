# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Join Serving invocation metadata with native per-rank device timing.

The adapter consumes raw chip-swimlane records and the runtime identity
sidecars written beside them.  It keeps rank-local clock domains separate and
only reports durations after the dispatch identity, semantic scope and capture
coverage have been checked.  The module deliberately does not infer attention
meaning from kernel name substrings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

from pypto_serving.config.types import DispatchInvocation

IDENTITY_FIELDS = (
    "owner_instance_id", "chip_rank", "run_id", "endpoint_dispatch_id",
    "task_slot", "group_index", "group_size", "pipeline_slot", "pipeline_generation",
)


def interval_union_cycles(intervals: list[tuple[int, int]]) -> int:
    """Count occupied device time once when cores execute concurrently."""
    total = 0
    stop = 0
    for start, end in sorted(intervals):
        total += max(0, end - max(stop, start))
        stop = max(stop, end)
    return total


def _positive_int(value: object) -> bool:
    return type(value) is int and value > 0


def _invocation_fields(invocation: DispatchInvocation | Mapping[str, object]) -> dict[str, object]:
    if isinstance(invocation, DispatchInvocation):
        fields = {
            "invocation_id": invocation.invocation_id,
            "serving_step_id": invocation.step_id,
            "execution_slot": invocation.execution_slot,
            "pipeline_generation": invocation.pipeline_generation,
        }
    else:
        fields = {
            "invocation_id": invocation.get("invocation_id"),
            "serving_step_id": invocation.get("serving_step_id", invocation.get("step_id")),
            "execution_slot": invocation.get("execution_slot", 0),
            "pipeline_generation": invocation.get("pipeline_generation", 0),
        }
    if not isinstance(fields["invocation_id"], str) or not fields["invocation_id"]:
        raise ValueError("invocation_id must be a non-empty string")
    if type(fields["serving_step_id"]) is not int or fields["serving_step_id"] < 0:
        raise ValueError("serving_step_id must be a nonnegative integer")
    if type(fields["execution_slot"]) is not int or fields["execution_slot"] < 0:
        raise ValueError("execution_slot must be a nonnegative integer")
    if type(fields["pipeline_generation"]) is not int or fields["pipeline_generation"] < 0:
        raise ValueError("pipeline_generation must be a nonnegative integer")
    return fields


def build_invocation_manifest(
    invocation: DispatchInvocation | Mapping[str, object],
    *,
    scope: str,
    logical_dp_groups: Mapping[str, list[int]],
    captures: list[dict],
    model_layer_id: int | None = None,
    semantic: Mapping[str, object] | None = None,
    rank_to_device: Mapping[str, int] | None = None,
    provenance: Mapping[str, object] | None = None,
) -> dict:
    """Create the join manifest emitted by a real Serving L3 invocation.

    ``captures`` contains runtime-produced records and identity sidecars.  The
    builder only adds Serving and compiler metadata, so it cannot accidentally
    manufacture a rank or device identity that the runtime did not provide.
    """
    if not isinstance(scope, str) or not scope:
        raise ValueError("semantic scope must be a non-empty string")
    if not isinstance(logical_dp_groups, Mapping) or not logical_dp_groups:
        raise ValueError("logical_dp_groups must be a non-empty mapping")
    if not isinstance(captures, list):
        raise ValueError("captures must be a list")
    fields = _invocation_fields(invocation)
    result = {
        "schema_version": 1,
        "invocation_id": fields["invocation_id"],
        "prototype_invocation_id": fields["invocation_id"],
        "serving_step_id": fields["serving_step_id"],
        "execution_slot": fields["execution_slot"],
        "pipeline_generation": fields["pipeline_generation"],
        "scope": scope,
        "model_layer_id": model_layer_id,
        "semantic": dict(semantic or {}),
        "isolated_single_child_invocation": True,
        "logical_dp_groups": {
            str(group): list(members) for group, members in logical_dp_groups.items()
        },
        "captures": captures,
    }
    if rank_to_device is not None:
        result["rank_to_device"] = {str(rank): device for rank, device in rank_to_device.items()}
    if provenance is not None:
        result["provenance"] = dict(provenance)
    return result


def summarize_capture(raw: dict, identity: dict) -> dict:
    """Validate a runtime sidecar/native join before interpreting timestamps."""
    if not isinstance(raw, dict):
        raise ValueError("native capture must be an object")
    metadata = raw.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("capture metadata must be an object")
    collection = metadata.get("collection", {})
    if collection is None:
        collection = {}
    if not isinstance(collection, dict):
        raise ValueError("collection metadata must be an object")
    dispatch = collection.get("dispatch")
    if dispatch is None:
        dispatch = dict(identity)
        dispatch["capture_index"] = identity.get("local_capture_index")
        dispatch["status"] = "identified"
        coverage_state = "unavailable"
    else:
        if not isinstance(dispatch, dict):
            raise ValueError("dispatch identity must be an object")
        coverage_state = "available"
    if raw.get("chip_swimlane_level") not in (1, 2, 3, 4):
        raise ValueError("unsupported capture level")
    if dispatch.get("status") != "identified":
        raise ValueError("native capture has no stated dispatch identity")
    for key in IDENTITY_FIELDS:
        if identity.get(key) is None or dispatch.get(key) != identity[key]:
            raise ValueError(f"dispatch identity mismatch: {key}")
    if dispatch.get("capture_index") != identity.get("local_capture_index"):
        raise ValueError("dispatch identity mismatch: capture_index")
    if not all(_positive_int(dispatch[key]) for key in ("run_id", "endpoint_dispatch_id", "group_size")):
        raise ValueError("invalid dispatch identity")
    for key in ("chip_rank", "capture_index", "task_slot", "group_index", "pipeline_slot"):
        if type(dispatch.get(key)) is not int or dispatch[key] < 0:
            raise ValueError(f"invalid dispatch identity: {key}")
    if dispatch["group_index"] >= dispatch["group_size"] or not _positive_int(dispatch["pipeline_generation"]):
        raise ValueError("invalid dispatch group or generation")
    owner = dispatch["owner_instance_id"]
    if not isinstance(owner, str) or len(owner) != 16 or any(c not in "0123456789abcdef" for c in owner):
        raise ValueError("invalid owner nonce")
    if not any(c != "0" for c in owner):
        raise ValueError("unstated owner nonce")
    frequency = metadata.get("clock_freq_hz")
    epoch = collection.get("run_epoch") or identity.get("run_epoch")
    if not _positive_int(frequency):
        raise ValueError("missing device frequency")
    rows = raw.get("aicore_tasks", [])
    if not isinstance(rows, list):
        raise ValueError("AICore records must be an array")
    if epoch is None:
        row_epochs = {
            row[-1] for row in rows
            if isinstance(row, list) and len(row) == 7 and type(row[-1]) is int
        }
        epoch = next(iter(row_epochs)) if len(row_epochs) == 1 else 0
    if not _positive_int(epoch):
        raise ValueError("missing device run epoch")
    intervals = []
    seen = set()
    invalid_rows = 0
    for row in rows:
        if not isinstance(row, list) or len(row) != 7 or any(type(v) is not int for v in row):
            invalid_rows += 1
            continue
        core, token, registration, start, end, receive_delta, row_epoch = row
        key = (row_epoch, core, registration)
        if row_epoch != epoch or core < 0 or registration < 0 or token < 0 or start <= 0 or end <= start:
            invalid_rows += 1
            continue
        if receive_delta < 0 or key in seen:
            invalid_rows += 1
            continue
        seen.add(key)
        intervals.append((start, end))
    complete = (
        coverage_state == "available"
        and collection.get("processing_complete") is True
        and collection.get("metadata_complete") is True
        and collection.get("verdict") == "published"
        and all(collection.get(key) == 0 for key in (
            "not_received_buffers", "unpublished_loss", "transport_retired",
            "transport_retired_records", "unattributable_handoffs", "cut_failed_queues",
        ))
        and not collection.get("recorded_window", False)
        and bool(intervals) and invalid_rows == 0
    )
    scale = 1_000_000 / frequency
    observed = bool(intervals) and invalid_rows == 0 and (coverage_state == "unavailable" or complete)
    return {
        "status": "complete" if complete else ("observed" if observed else "partial"),
        "coverage_state": coverage_state,
        "identity": dispatch,
        "run_epoch": epoch,
        "clock_domain": "rank_local_device_syscnt",
        "clock_freq_hz": frequency,
        "raw_records": len(rows),
        "valid_records": len(intervals),
        "invalid_records": invalid_rows,
        "collection": collection,
        "observed_busy_union_us": interval_union_cycles(intervals) * scale if intervals else None,
        "observed_envelope_us": (
            max(e for _, e in intervals) - min(s for s, _ in intervals)
        ) * scale if intervals else None,
        "busy_union_us": interval_union_cycles(intervals) * scale if observed else None,
        "envelope_us": (
            max(e for _, e in intervals) - min(s for s, _ in intervals)
        ) * scale if observed else None,
    }


def analyze_invocation(manifest: dict, base_dir: Path) -> dict:
    """Summarize one Serving invocation and retain incomplete rank coverage."""
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("scope"), str):
        raise ValueError("unsupported invocation schema or semantic scope")
    groups = manifest.get("logical_dp_groups")
    captures = manifest.get("captures")
    if not isinstance(groups, dict) or not isinstance(captures, list):
        raise ValueError("invocation manifest requires groups and captures")
    expected = [rank for members in groups.values() for rank in members]
    if (
        not expected
        or any(not members for members in groups.values())
        or any(type(rank) is not int or rank < 0 for rank in expected)
        or len(set(expected)) != len(expected)
    ):
        raise ValueError("logical DP groups must partition distinct nonnegative ranks")
    if manifest.get("isolated_single_child_invocation") is not True:
        raise ValueError("capture must isolate one invocation with one child per rank")
    ranks = {}
    unresolved = []
    candidates = {}
    for capture in captures:
        rank = capture["identity"].get("chip_rank")
        candidates.setdefault(rank, []).append(capture)
    epochs = set()
    for rank in expected:
        entries = candidates.get(rank, [])
        if len(entries) != 1:
            ranks[str(rank)] = {"status": "missing" if not entries else "ambiguous"}
            continue
        entry = entries[0]
        try:
            if entry.get("program_sidecar") is not None:
                marker = json.loads((base_dir / entry["program_sidecar"]).read_text())
                expected_program = entry.get("program")
                if expected_program is not None and marker.get("program") != expected_program:
                    raise ValueError("semantic program sidecar mismatch")
            if entry.get("identity_sidecar") is not None:
                sidecar = json.loads((base_dir / entry["identity_sidecar"]).read_text())
                if sidecar != entry["identity"]:
                    raise ValueError("manifest identity disagrees with the original sidecar")
            raw = json.loads((base_dir / entry["records"]).read_text())
            summary = summarize_capture(raw, entry["identity"])
            epochs.add((entry["identity"]["owner_instance_id"], entry["identity"]["run_id"]))
            ranks[str(rank)] = {"records": entry["records"], **summary}
        except (OSError, ValueError, TypeError, KeyError) as error:
            ranks[str(rank)] = {"status": "unresolved", "error": str(error)}
    for rank in candidates:
        if rank not in expected:
            unresolved.append({"rank": rank, "reason": "unexpected rank"})
    coherent = len(epochs) == 1
    summaries = {}
    for group, members in groups.items():
        complete = coherent and not unresolved and all(
            ranks[str(rank)]["status"] in ("complete", "observed") for rank in members
        )
        summaries[group] = {
            "members": members,
            "status": "complete" if complete and all(
                ranks[str(rank)]["status"] == "complete" for rank in members
            ) else ("observed" if complete else "partial"),
            "complete_ranks": sum(
                ranks[str(rank)]["status"] == "complete" for rank in members
            ),
            "attention_busy_union_max_us": max(
                ranks[str(rank)]["busy_union_us"] for rank in members
            ) if complete else None,
            "attention_envelope_max_us": max(
                ranks[str(rank)]["envelope_us"] for rank in members
            ) if complete else None,
        }
    return {
        "schema_version": 1,
        "invocation_id": manifest.get("invocation_id", manifest.get("prototype_invocation_id")),
        "prototype_invocation_id": manifest.get("prototype_invocation_id"),
        "serving_step_id": manifest.get("serving_step_id"),
        "execution_slot": manifest.get("execution_slot"),
        "pipeline_generation": manifest.get("pipeline_generation"),
        "scope": manifest["scope"],
        "model_layer_id": manifest.get("model_layer_id"),
        "semantic": manifest.get("semantic", {}),
        "status": "complete" if all(group["status"] == "complete" for group in summaries.values()) else (
            "observed" if all(group["status"] in ("complete", "observed") for group in summaries.values())
            else "partial"
        ),
        "one_native_run": coherent,
        "ranks": ranks,
        "logical_dp_groups": summaries,
        "unresolved": unresolved,
        "unavailable_metrics": {
            "full_attention_block_us": None,
            "moe_us": None,
            "communication_us": None,
            "wait_us": None,
            "full_step_us": None,
        },
        "interpretation": (
            "Per-rank interval union and envelope with group duration maximum. "
            "Raw rank-local clocks are retained; communication, MoE and wait "
            "remain unavailable until their semantic boundaries are published."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_invocation(json.loads(args.manifest.read_text()), args.manifest.parent)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"])
    if result["status"] not in ("complete", "observed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
