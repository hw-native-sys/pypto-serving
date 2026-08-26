# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate and summarize pypto-serving MoE statistics JSONL output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="MoE statistics JSONL file")
    parser.add_argument(
        "--require-phase",
        action="append",
        default=[],
        help="Phase that must appear; repeat for multiple phases",
    )
    return parser.parse_args()


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"MoE statistics file does not exist: {path}")
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"record on line {line_number} must be an object")
        records.append(record)
    if not records:
        raise ValueError(f"MoE statistics file contains no records: {path}")
    return records


def _validate_record(record: dict[str, Any], index: int) -> tuple[int, int, int]:
    phase = record.get("phase")
    ranks = record.get("ranks")
    local_experts = record.get("local_experts")
    layers = record.get("layers")
    if not isinstance(phase, str) or not phase:
        raise ValueError(f"record {index} has invalid phase")
    if not isinstance(ranks, int) or ranks <= 0:
        raise ValueError(f"record {index} has invalid ranks")
    if not isinstance(local_experts, int) or local_experts <= 0:
        raise ValueError(f"record {index} has invalid local_experts")
    if not isinstance(layers, list) or not layers:
        raise ValueError(f"record {index} has no layers")

    expected_experts = ranks * local_experts
    record_routed_tokens = 0
    record_active_experts = 0
    for layer_index, layer in enumerate(layers):
        if not isinstance(layer, dict):
            raise ValueError(f"record {index} layer {layer_index} must be an object")
        counts = layer.get("expert_token_counts")
        if not isinstance(counts, list) or len(counts) != expected_experts:
            raise ValueError(
                f"record {index} layer {layer_index} expected {expected_experts} expert counts"
            )
        if any(not isinstance(value, int) or value < 0 for value in counts):
            raise ValueError(f"record {index} layer {layer_index} contains invalid expert counts")
        routed_tokens = sum(counts)
        active_experts = sum(value > 0 for value in counts)
        if layer.get("routed_tokens") != routed_tokens:
            raise ValueError(f"record {index} layer {layer_index} routed_tokens is inconsistent")
        if layer.get("active_experts") != active_experts:
            raise ValueError(f"record {index} layer {layer_index} active_experts is inconsistent")
        record_routed_tokens += routed_tokens
        record_active_experts += active_experts
    return len(layers), record_routed_tokens, record_active_experts


def main() -> None:
    args = _parse_args()
    records = _load_records(args.path)
    summaries = [_validate_record(record, index) for index, record in enumerate(records)]
    phases = [record["phase"] for record in records]
    missing_phases = sorted(set(args.require_phase) - set(phases))
    if missing_phases:
        raise ValueError(f"missing required phases: {', '.join(missing_phases)}")

    print(
        json.dumps(
            {
                "path": str(args.path.resolve()),
                "records": len(records),
                "phases": phases,
                "layers_per_record": [summary[0] for summary in summaries],
                "experts_per_layer": sorted(
                    {record["ranks"] * record["local_experts"] for record in records}
                ),
                "routed_tokens_per_record": [summary[1] for summary in summaries],
                "active_experts_per_record": [summary[2] for summary in summaries],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
