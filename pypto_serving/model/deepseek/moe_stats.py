# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""JSONL output for DeepSeek V4 MoE physical-expert load statistics."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import torch


class MoeStatsWriter:
    """Write one self-contained JSON record for each profiled model dispatch."""

    def __init__(self, output_path: str) -> None:
        if not output_path:
            raise ValueError("MoE statistics output path must not be empty")
        self._path = Path(output_path)
        self._lock = threading.Lock()
        self._dispatch_id = 0

    @property
    def path(self) -> Path:
        return self._path

    def write(self, phase: str, rank_local_counts: torch.Tensor) -> None:
        """Append counts shaped ``[ranks, hidden_layers + 1, local_experts]``."""
        counts = rank_local_counts.detach().cpu().to(torch.int64)
        if counts.ndim != 3:
            raise ValueError(f"MoE token counts must be rank-3, got shape {tuple(counts.shape)}")
        ranks, num_layers, local_experts = counts.shape
        global_counts = counts.permute(1, 0, 2).reshape(num_layers, ranks * local_experts)
        layers = []
        for layer_id, layer_counts in enumerate(global_counts):
            values = layer_counts.tolist()
            layers.append(
                {
                    "layer_id": layer_id,
                    "kind": "mtp" if layer_id == num_layers - 1 else "main",
                    "active_experts": int(torch.count_nonzero(layer_counts).item()),
                    "routed_tokens": int(layer_counts.sum().item()),
                    "expert_token_counts": values,
                }
            )
        with self._lock:
            record = {
                "timestamp_ns": time.time_ns(),
                "dispatch_id": self._dispatch_id,
                "phase": phase,
                "ranks": ranks,
                "local_experts": local_experts,
                "layers": layers,
            }
            self._dispatch_id += 1
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
