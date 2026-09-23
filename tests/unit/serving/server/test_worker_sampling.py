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

from pypto_serving.serving.memory.kv_cache import KvCacheManager
from pypto_serving.serving.server.ipc import (
    PLACEHOLDER_TOKEN,
    DecodeRequest,
    NewRequestData,
    PrefillRequest,
)
from pypto_serving.serving.server.serving_worker import WorkerProcess

from ..device_sampling_fakes import (
    _DeviceSamplingExecutor,
    _DeviceTopkExecutor,
    _FailingSampler,
    _model,
    _RoutingSampler,
)


def test_serving_worker_routes_supported_topk_candidates():
    model = _model(max_batch_size=1)
    manager = KvCacheManager()
    executor = _DeviceTopkExecutor(manager, token_id=7)
    sampler = _RoutingSampler(host_token_id=9, candidate_token_id=7)
    worker = WorkerProcess.__new__(WorkerProcess)
    worker._batch_builder = worker._make_decode_batch
    worker._services = None
    worker.executor = executor
    worker.sampler = sampler
    worker.model_record = SimpleNamespace(config=model.config)
    worker._req_cache = {
        "request": NewRequestData(
            request_id="request",
            prompt_token_ids=[1],
            temperature=0.8,
            top_p=1.0,
            top_k=4,
        )
    }

    prefill_tokens: dict[str, list[int]] = {}
    worker._batch_prefill(
        [
            PrefillRequest(
                request_id="request",
                chunk_tokens=[1],
                num_computed_tokens=0,
                block_ids=[0],
            )
        ],
        model,
        prefill_tokens,
    )

    decode_tokens: dict[str, list[int]] = {}
    worker._batch_decode(
        [
            DecodeRequest(
                request_id="request",
                last_token=7,
                seq_len=2,
                block_ids=[0],
            )
        ],
        model,
        decode_tokens,
    )

    assert prefill_tokens == {"request": [7]}
    assert decode_tokens == {"request": [7]}
    assert executor.prefill_allow_topk is True
    assert executor.decode_allow_topk is True
    assert sampler.sample_calls == 0
    assert sampler.candidate_calls == 2


def test_serving_worker_mixed_topk_batch_falls_back_from_stale_candidates():
    model = _model(max_batch_size=2)
    manager = KvCacheManager()
    executor = _DeviceTopkExecutor(
        manager,
        token_id=7,
        always_return_candidates=True,
    )
    sampler = _RoutingSampler(host_token_id=9, candidate_token_id=7)
    worker = WorkerProcess.__new__(WorkerProcess)
    worker._batch_builder = worker._make_decode_batch
    worker._services = None
    worker.executor = executor
    worker.sampler = sampler
    worker.model_record = SimpleNamespace(config=model.config)
    worker._req_cache = {
        "supported": NewRequestData(
            request_id="supported",
            prompt_token_ids=[1],
            temperature=0.8,
            top_p=1.0,
            top_k=4,
        ),
        "unsupported": NewRequestData(
            request_id="unsupported",
            prompt_token_ids=[2],
            temperature=0.8,
            top_p=1.0,
            top_k=8,
        ),
    }

    prefill_tokens: dict[str, list[int]] = {}
    worker._batch_prefill(
        [
            PrefillRequest(
                request_id=request_id,
                chunk_tokens=[token_id],
                num_computed_tokens=0,
                block_ids=[row],
            )
            for row, (request_id, token_id) in enumerate((("supported", 1), ("unsupported", 2)))
        ],
        model,
        prefill_tokens,
    )

    decode_tokens: dict[str, list[int]] = {}
    worker._batch_decode(
        [
            DecodeRequest(
                request_id=request_id,
                last_token=3,
                seq_len=2,
                block_ids=[row],
            )
            for row, request_id in enumerate(("supported", "unsupported"))
        ],
        model,
        decode_tokens,
    )

    assert prefill_tokens == {"supported": [9], "unsupported": [9]}
    assert decode_tokens == {"supported": [9], "unsupported": [9]}
    assert executor.prefill_allow_topk is False
    assert executor.decode_allow_topk is False
    assert sampler.sample_calls == 4
    assert sampler.candidate_calls == 0


def test_serving_worker_skips_decode_host_embedding_when_executor_embeds_on_device():
    model = _model(max_batch_size=1, eos_token_id=0)
    manager = KvCacheManager()
    executor = _DeviceSamplingExecutor(
        manager,
        first_token=3,
        second_token=0,
        return_next_hidden=False,
    )

    def fail_lookup(model, token_ids):
        raise AssertionError("serving worker decode should let the device kernel embed token ids")

    executor.lookup_embeddings = fail_lookup
    worker = WorkerProcess.__new__(WorkerProcess)
    worker._batch_builder = worker._make_decode_batch
    worker._services = None
    worker.executor = executor
    worker.sampler = _FailingSampler()
    worker.model_record = SimpleNamespace(config=model.config)
    worker._req_cache = {
        "decode": NewRequestData(
            request_id="decode",
            prompt_token_ids=[1],
            temperature=0.0,
            top_p=1.0,
            top_k=None,
        )
    }

    # last_token=3 (the one output token), seq_len=2.
    decode_req = DecodeRequest(
        request_id="decode",
        last_token=3,
        seq_len=2,
        block_ids=[0],
    )
    new_tokens: dict[str, list[int]] = {}

    worker._batch_decode([decode_req], model, new_tokens)

    assert new_tokens == {"decode": [0]}
    assert executor.decode_calls == 1
    assert executor.decode_hidden_seen[0] is None


def test_worker_resolves_placeholder_decode_token_from_cache():
    """Under async scheduling the engine sends PLACEHOLDER_TOKEN; the worker must
    substitute the token(s) it last sampled for that request."""
    worker = WorkerProcess.__new__(WorkerProcess)
    worker._batch_builder = worker._make_decode_batch
    worker._services = None
    worker._last_tokens = {}

    # Record two sampled tokens (simulating two prior decode steps).
    worker._record_last_tokens("r", [11])
    worker._record_last_tokens("r", [22])
    assert worker._last_tokens["r"] == [22]

    placeholder = DecodeRequest(
        request_id="r",
        last_token=PLACEHOLDER_TOKEN,
        seq_len=5,
        block_ids=[0],
    )
    # Most recent sampled token (22).
    assert worker._resolve_decode_token(placeholder) == 22

    # A real (non-placeholder) token is passed through untouched.
    explicit = DecodeRequest(request_id="r", last_token=99, seq_len=5, block_ids=[0])
    assert worker._resolve_decode_token(explicit) == 99

    # Cache keeps only the token needed to resolve the next placeholder.
    worker._record_last_tokens("r", [33])
    assert worker._last_tokens["r"] == [33]

    # Missing cache entry on placeholder is a hard error (never silently wrong).
    orphan = DecodeRequest(
        request_id="missing",
        last_token=PLACEHOLDER_TOKEN,
        seq_len=1,
        block_ids=[0],
    )
    with pytest.raises(RuntimeError):
        worker._resolve_decode_token(orphan)
