# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from types import SimpleNamespace

import pytest
import torch

from pypto_serving.cli.main import build_parser, build_serving_engine_config
from pypto_serving.model.glm53.kernel_api import packed_capacity
from pypto_serving.model.glm53.npu_runner import Glm53ModelRunner
from pypto_serving.model.model_family import detect_model_family


def test_glm_family_and_worker_topology(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type":"glm5_next"}')
    assert detect_model_family({"architectures": ["Glm5NextForConditionalGeneration"]}) == "glm53"
    argv = ["--model", str(tmp_path), "--devices", ",".join(map(str, range(16))),
            "--tp", "16", "--ep", "16", "--max-num-seqs", "1", "--max-model-len", "512"]
    config = build_serving_engine_config(build_parser().parse_args(argv))
    assert config.executor_cls == "PyptoGlm53Executor"
    assert config.parallel_config.worker_group_size == 16
    assert config.parallel_config.num_replicas == 1
    assert not config.enable_prefix_cache
    assert not config.enable_chunk_prefill
    bad = argv + ["--max-num-seqs", "2"]
    with pytest.raises(ValueError, match="max-num-seqs 1"):
        build_serving_engine_config(build_parser().parse_args(bad))


def test_packed_capacity_reserves_inactive_slots():
    assert packed_capacity(512) == 576
    assert packed_capacity(8177) == 8192
    for length in (0, -1, 8178):
        with pytest.raises(ValueError):
            packed_capacity(length)


class RecordingSession:
    def __init__(self):
        self.inputs = []

    def forward(self, sequences):
        self.inputs.append([list(tokens) for tokens in sequences])
        return torch.zeros(16, 32)


def test_runner_keeps_full_prefix_and_releases_request_state():
    runner = Glm53ModelRunner(SimpleNamespace(BATCH=16), {}, None, list(range(16)), "a2a3")
    runner.init_kv_cache("glm", None, SimpleNamespace(max_batch_size=1, max_seq_len=32, page_size=128))
    runner._session = RecordingSession()
    prefill = SimpleNamespace(request_ids=["first"], token_ids=torch.tensor([20, 21]),
                              chunk_starts=[0], chunk_lens=[2], seq_lens=[2])
    result = runner.run_prefill(None, prefill)
    assert result.logits.shape == (1, 32)
    assert runner._session.inputs[-1] == [[20, 21]] + [[0]] * 15
    decode = SimpleNamespace(request_ids=["first"], token_ids=torch.tensor([3]))
    runner.run_decode(None, decode)
    assert runner._session.inputs[-1][0] == [20, 21, 3]
    runner.release_finished_requests(["first"])
    with pytest.raises(ValueError, match="completed prefill"):
        runner.run_decode(None, decode)
    prefill.request_ids = ["second"]
    runner.run_prefill(None, prefill)
    assert runner._session.inputs[-1][0] == [20, 21]


def test_runner_rejects_partial_prefill_before_forward():
    runner = Glm53ModelRunner(SimpleNamespace(BATCH=16), {}, None, list(range(16)), "a2a3")
    batch = SimpleNamespace(request_ids=["r"], chunk_starts=[2])
    with pytest.raises(ValueError, match="complete prompt"):
        runner.run_prefill(None, batch)
