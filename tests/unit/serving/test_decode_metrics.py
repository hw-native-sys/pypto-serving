# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import json

import pytest

from pypto_serving.observability import decode_metrics
from pypto_serving.observability.decode_metrics import DecodeStepRecorder


def test_records_raw_accepted_counts_and_one_duration_per_batch(monkeypatch, tmp_path):
    recorder = DecodeStepRecorder(str(tmp_path))
    ticks = iter((1_000_000, 6_000_000))
    monkeypatch.setattr(decode_metrics.time, "perf_counter_ns", lambda: next(ticks))
    recorder.record(step_id=4, started_ns=recorder.start(), request_ids=["a", "b"],
                    new_tokens={"a": [1, 2, 3], "b": [7], "prefill": [8]},
                    num_draft_tokens={"a": 7}, mode="sync", prefill_requests=1)
    row = json.loads(recorder.path.read_text())
    assert row["step_time_ms"] == 5
    assert row["requests"] == [dict(request_id="a", accepted_tokens=3, proposed_drafts=7),
                               dict(request_id="b", accepted_tokens=1, proposed_drafts=0)]
    assert row["prefill_requests"] == 1
    assert row["worker_id"] and row["step_id"] == 4
    recorder.close()
    with pytest.raises(FileExistsError):
        DecodeStepRecorder(str(tmp_path))
