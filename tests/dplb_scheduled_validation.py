# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate M2 evidence from a bounded, completed Host smoke capture."""


def validate_scheduled_load(events: list[dict], topology: list[dict]) -> dict:
    plans = {}
    outcomes = {}
    boundary = set()
    returned_tokens = proposed_tokens = accepted_tokens = 0
    stages = set()
    for event in events:
        if event["event"] == "capture_end":
            boundary.update(event.get("in_flight_step_ids", []))
        if event["event"] not in {"step_plan", "step_outcome"}:
            continue
        key = event["step_id"]
        target = plans if event["event"] == "step_plan" else outcomes
        if key in target:
            raise RuntimeError("duplicate scheduled step event")
        target[key] = event
        if event["event"] == "step_outcome":
            if key not in plans or plans[key]["dispatch_ordinal"] != event["dispatch_ordinal"]:
                raise RuntimeError("outcome without a matching submitted step")
            if event["status"] != "completed" or event["error"] is not None:
                raise RuntimeError("worker step failed or was discarded")
            plan_rows = plans[key]["requests"]
            if len(plan_rows) != len(event["requests"]):
                raise RuntimeError("step outcome membership mismatch")
            for planned, actual in zip(plan_rows, event["requests"]):
                for field in ("request_id", "request_uid", "admission_epoch", "cache_partition"):
                    if planned[field] != actual[field]:
                        raise RuntimeError("step outcome request identity mismatch")
                returned = actual["returned_tokens"]
                proposed = actual["speculative_proposed_tokens"]
                accepted = actual["speculative_accepted_tokens"]
                if (min(returned, proposed, accepted) < 0 or accepted > proposed
                        or accepted != (max(0, returned - 1) if proposed else 0)):
                    raise RuntimeError("invalid speculative result accounting")
                returned_tokens += returned
                proposed_tokens += proposed
                accepted_tokens += accepted
            continue

        rows = event["requests"]
        if not rows or event["active_count"] != len(rows):
            raise RuntimeError("scheduled active count mismatch")
        if len({row["request_uid"] for row in rows}) != len(rows) or any(
            row["request_uid"] is None for row in rows
        ):
            raise RuntimeError("scheduled request identity is missing or duplicated")
        stages.update(row["stage"] for row in rows)
        if event["prefill_tokens"] != sum(row["prefill_tokens"] for row in rows):
            raise RuntimeError("scheduled prefill token mismatch")
        if event["decode_tokens"] != sum(row["decode_tokens"] for row in rows):
            raise RuntimeError("scheduled decode token mismatch")
        if event["speculative_budget_tokens"] != sum(row["speculative_budget_tokens"] for row in rows):
            raise RuntimeError("scheduled speculative budget mismatch")
        for row in rows:
            if row["logical_length_after"] != row["num_computed_tokens"] + row["num_new_tokens"]:
                raise RuntimeError("scheduled length snapshot mismatch")
            extent = row["row_extent"]
            if extent["valid_rows"] is not None or extent["padded_rows"] is not None:
                raise RuntimeError("Host snapshot claims unobserved runner rows")
        if event["topology"] != topology:
            raise RuntimeError("scheduled topology mismatch")
        group_ids = {group["group_id"] for group in event["groups"]}
        if group_ids != {group["group_id"] for group in topology}:
            raise RuntimeError("scheduled group membership mismatch")
        if any(row["cache_partition"] not in group_ids for row in rows):
            raise RuntimeError("request has no logical DP group")
        for group in event["groups"]:
            members = [row for row in rows if row["cache_partition"] == group["group_id"]]
            if (group["participating_request_count"] != len(members)
                    or group["participating_request_ids"] != [row["request_id"] for row in members]
                    or group["participating_prefill_tokens"] != sum(row["prefill_tokens"] for row in members)
                    or group["participating_decode_tokens"] != sum(row["decode_tokens"] for row in members)):
                raise RuntimeError("scheduled group aggregate mismatch")
    if not plans or not outcomes:
        raise RuntimeError("missing M2 scheduled-load evidence")
    pending = set(plans) - set(outcomes)
    if not pending <= boundary:
        raise RuntimeError("scheduled step lacks outcome or boundary explanation")
    if len({event["dispatch_ordinal"] for event in plans.values()}) != len(plans):
        raise RuntimeError("duplicate dispatch ordinal")
    return dict(
        planned_steps=len(plans), paired_steps=len(outcomes),
        pending_at_capture_end=sorted(pending), stages=sorted(stages),
        worker_returned_tokens=returned_tokens,
        worker_proposed_draft_tokens=proposed_tokens,
        worker_accepted_draft_tokens=accepted_tokens,
    )
