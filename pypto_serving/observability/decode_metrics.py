# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Opt-in Host decode batch timing, independent of model and transport."""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time
import uuid


class DecodeStepRecorder:
    """One record per completed batch; accepted counts precede scheduler clipping.

    Dispatch-to-reclaim includes Host and asynchronous completion waits. It is
    not device kernel time. No recorder or clock read is needed when disabled.
    """

    def __init__(self, directory: str):
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        self.pid = os.getpid()
        self.worker_id = uuid.uuid4().hex
        self.path = root / f"decode-steps.{self.pid}.jsonl"
        self._stream = self.path.open("x", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls):
        directory = os.environ.get("PYPTO_SERVING_DECODE_METRICS_DIR")
        return cls(directory) if directory else None

    @staticmethod
    def start() -> int:
        return time.perf_counter_ns()

    def record(self, *, step_id, started_ns, request_ids, new_tokens, num_draft_tokens,
               mode, prefill_requests=0):
        ended_ns = time.perf_counter_ns()
        value = dict(
            schema=1, pid=self.pid, worker_id=self.worker_id, step_id=step_id, mode=mode,
            start_ns=started_ns, end_ns=ended_ns,
            step_time_ms=(ended_ns - started_ns) / 1_000_000,
            prefill_requests=prefill_requests,
            requests=[dict(request_id=rid, accepted_tokens=len(new_tokens.get(rid, ())),
                           proposed_drafts=num_draft_tokens.get(rid, 0)) for rid in request_ids],
        )
        with self._lock:
            self._stream.write(json.dumps(value, separators=(",", ":")) + "\n")

    def close(self):
        self._stream.close()
