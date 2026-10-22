# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Request limits and EOS policy across API, Host sampling, and DSpark device masks."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pypto_serving.config.types import GenerateConfig, StopTokenMask
from pypto_serving.model.common.executor.sampler import Sampler
from pypto_serving.model.deepseek_dspark.npu_runner import (
    DSPARK_DECODE_BATCH,
    DSPARK_DRAFTER_QUERY_WIDTH,
    DSPARK_GRAMMAR_SEGMENTS,
    DSPARK_GRAMMAR_SEGMENT_WORDS,
    DSPARK_MAX_LOGIT_ROWS,
    DSPARK_RANKS,
    DSPARK_VOCAB_SIZE,
    DSparkModelRunner,
)
from pypto_serving.serving.memory.kv_cache import KvCacheManager
from pypto_serving.serving.sched.scheduler import (
    Request,
    RequestStatus,
    ScheduledRequest,
    Scheduler,
    SchedulerConfig,
    SchedulerOutput,
)
from pypto_serving.serving.server.ipc import NewRequestData
from pypto_serving.serving.server.server import ChatCompletionRequest, ChatMessage, ServingServer
from pypto_serving.serving.server.serving_worker import WorkerProcess


def _chat_request(**fields) -> ChatCompletionRequest:
    return ChatCompletionRequest(messages=[ChatMessage(role="user", content="Go")], **fields)


@pytest.mark.parametrize("fields,expected", [
    ({}, 11),
    ({"max_tokens": 7}, 7),
    ({"max_completion_tokens": 5}, 5),
    ({"max_tokens": 7, "max_completion_tokens": 5}, 5),
    ({"max_tokens": 7, "max_completion_tokens": None}, 7),
])
def test_chat_token_limit_resolution(fields, expected):
    server = ServingServer(SimpleNamespace(), "test-model", GenerateConfig(max_new_tokens=11))
    assert server._resolve_generate_config(_chat_request(**fields)).max_new_tokens == expected


def test_generation_limits_and_eos_defaults():
    server = ServingServer(SimpleNamespace(), "test-model", GenerateConfig(
        max_new_tokens=8, ignore_eos=True,
    ))
    default = server._resolve_generate_config(_chat_request())
    explicit = server._resolve_generate_config(_chat_request(ignore_eos=False, min_tokens=2))
    assert (default.ignore_eos, default.min_tokens) == (True, 0)
    assert (explicit.ignore_eos, explicit.min_tokens) == (False, 2)

    with pytest.raises(ValueError, match="min_tokens 3 exceeds max_tokens 2"):
        GenerateConfig(max_new_tokens=2, min_tokens=3)
    with pytest.raises(ValueError, match="max_new_tokens must be a positive integer"):
        server._resolve_generate_config(_chat_request(max_completion_tokens=0))

    manager = KvCacheManager(num_blocks=4, block_size=2, enable_prefix_cache=False)
    scheduler = Scheduler(SchedulerConfig(max_seq_len=5, enable_prefix_cache=False), manager)
    with pytest.raises(ValueError, match="min_tokens 3 exceeds available generation tokens 2"):
        scheduler.add_request(Request("r", [1, 2, 3], max_new_tokens=4, min_tokens=3))


@pytest.mark.parametrize("ignore_eos,min_tokens", [
    (False, 0), (False, 2), (True, 0), (True, 2),
])
def test_ignore_eos_and_min_tokens_combinations(ignore_eos, min_tokens):
    worker = WorkerProcess.__new__(WorkerProcess)
    worker.executor = SimpleNamespace(supports_device_sampling=True)
    worker._req_cache = {"r": NewRequestData(
        request_id="r", prompt_token_ids=[3], temperature=0.0,
        top_p=1.0, top_k=None, min_tokens=min_tokens, eos_token_id=1,
    )}
    worker._accepted_output_lengths = {"r": 0}
    scheduled = [SimpleNamespace(request_id="r")]
    params = worker._sampling_params("r")
    assert params.forbidden_token_ids == ((1,) if min_tokens else ())
    assert worker._allow_device_sampled_ids(scheduled, phase="decode") is (min_tokens == 0)
    assert Sampler().sample(torch.tensor([0.0, 10.0, 9.0]), params) == (2 if min_tokens else 1)
    worker._accept_output_tokens("r", min_tokens)
    assert worker._sampling_params("r").forbidden_token_ids == ()

    manager = KvCacheManager(num_blocks=4, block_size=2, enable_prefix_cache=False)
    scheduler = Scheduler(SchedulerConfig(enable_prefix_cache=False), manager)
    request = Request(
        request_id="r", prompt_token_ids=[3], max_new_tokens=5,
        min_tokens=min_tokens, eos_token_id=1, ignore_eos=ignore_eos,
        num_computed_tokens=1, status=RequestStatus.RUNNING,
    )
    scheduler.running.append(request)
    scheduler.requests["r"] = request
    step = SchedulerOutput(scheduled_requests=[ScheduledRequest(
        request=request, num_new_tokens=1, is_prefill=False,
    )])
    tokens = [8] * min_tokens + [1, 9]
    outputs = scheduler.update_from_output(step, {"r": tokens})
    assert request.output_token_ids == (tokens if ignore_eos else tokens[:-1])
    assert request.status is (RequestStatus.RUNNING if ignore_eos else RequestStatus.FINISHED_EOS)
    assert [output.num_output_tokens for output in outputs] == list(
        range(1, len(request.output_token_ids) + 1)
    )


def test_dspark_stages_eos_mask_for_prefill_and_k7_decode():
    runner = DSparkModelRunner.__new__(DSparkModelRunner)
    masks = torch.full(
        (DSPARK_RANKS, DSPARK_MAX_LOGIT_ROWS, DSPARK_GRAMMAR_SEGMENTS, DSPARK_GRAMMAR_SEGMENT_WORDS),
        -1, dtype=torch.int16,
    )
    counts = torch.full((DSPARK_RANKS, DSPARK_DECODE_BATCH), DSPARK_DRAFTER_QUERY_WIDTH, dtype=torch.int32)
    runner._prefill_task_arg_slots = [SimpleNamespace(tensors={"grammar_mask": masks})]
    runner._prefill_grammar_rows = [set(), set()]
    runner._decode_task_args = [SimpleNamespace(tensors={"grammar_mask": masks, "valid_draft_counts": counts})]
    runner._decode_grammar_rows = [set(), set()]
    runner._compiled = SimpleNamespace(
        layout=SimpleNamespace(tp_size=4), num_speculative_tokens=DSPARK_DRAFTER_QUERY_WIDTH,
    )
    runner._drafter_state = lambda _: SimpleNamespace(pending_draft_tokens=[])
    inputs = SimpleNamespace(
        buffer_slot=0, request_ids=("r",), groups=(1,), group_ordinals=(5,), sampled_slots=((6, 40),)
    )
    batch = SimpleNamespace(constraint_states={}, stop_token_masks={"r": StopTokenMask((1,), 3)})

    runner._stage_prefill_grammar(batch, inputs)
    assert int(masks[6, 40, 0, 0]) & 2 == 0

    runner._stage_decode_grammar(batch, inputs)
    assert counts[4:8, 5].tolist() == [DSPARK_DRAFTER_QUERY_WIDTH] * 4
    assert [int(masks[6, row, 0, 0]) & 2 for row in range(40, 48)] == [0, 0, 0, 2, 2, 2, 2, 2]

    batch.stop_token_masks["r"] = StopTokenMask((1,), 1)
    runner._stage_decode_grammar(batch, inputs)
    assert [int(masks[6, row, 0, 0]) & 2 for row in range(40, 48)] == [0, 2, 2, 2, 2, 2, 2, 2]

    class Grammar:
        def __init__(self):
            self.rows = np.full((1, DSPARK_VOCAB_SIZE // 32), -1, dtype=np.int32)
            self.rows[0, 0] &= ~4

        def plan_draft_rows(self, drafts):
            assert drafts == []
            return 0, self.rows

    grammar = Grammar()
    batch.constraint_states["r"] = grammar
    runner._stage_decode_grammar(batch, inputs)
    assert int(masks[6, 40, 0, 0]) & 2 == 0
    assert int(masks[6, 40, 0, 0]) & 4 == 0
    assert int(grammar.rows[0, 0]) & 2 == 2

    grammar.rows.fill(0)
    grammar.rows[0, 0] = 2
    runner._stage_decode_grammar(batch, inputs)
    assert int(masks[6, 40, 0, 0]) & 2 == 2
