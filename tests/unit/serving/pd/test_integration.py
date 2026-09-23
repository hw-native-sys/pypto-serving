# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CPU-only lifecycle contracts for the optional PD composition layer."""

import asyncio
import pickle
from types import SimpleNamespace

import pytest

from pypto_serving.config.types import DecodeBatch
from pypto_serving.model.deepseek_dspark.pd_adapter import DSV4_DSPARK_K7_ADAPTER, DSparkPDWorker
from pypto_serving.serving.pd.integration import PDNodeRuntime
from pypto_serving.serving.pd.worker import PDWorkerServices


async def _empty_request(*args, **kwargs):
    yield SimpleNamespace(finished=True)


def _node(events, submit=_empty_request):
    return PDNodeRuntime(
        cache_manager=object(),
        add_request=submit,
        add_prefilled_request=None,
        abort_request=None,
        call_worker=None,
        finish_prefilled_request=lambda request: events.append(("finish", request)),
        hold_request=lambda request: events.append(("hold", request)),
        resume_request=lambda request: events.append(("resume", request)),
    )


def test_prefill_queue_is_registered_before_submission_and_cleaned_on_close():
    async def exercise():
        node = _node([])
        stream = node.add_request("request")
        assert "request" in node.chunks
        with pytest.raises(ValueError, match="already owns"):
            node.add_request("request")
        assert (await anext(stream)).finished
        await stream.aclose()
        assert node.chunks == {}
        assert node.chunk_counters == {}

    asyncio.run(exercise())


def test_prefill_failure_cleans_chunk_state():
    async def fail(*args, **kwargs):
        raise ValueError("admission failed")
        yield

    async def exercise():
        node = _node([], fail)
        stream = node.add_request("request")
        with pytest.raises(ValueError, match="admission failed"):
            await anext(stream)
        assert not node.chunks

    asyncio.run(exercise())


def test_prefill_result_wrapper_publishes_confirmed_immutable_snapshots():
    async def exercise():
        events = []
        node = _node(events)
        stream = node.add_request("request")
        await anext(stream)
        request = SimpleNamespace(request_id="request", num_prompt_tokens=8)
        blocks = {"kv": [1, 2]}
        scheduled = SimpleNamespace(
            is_prefill=True,
            cache_partition=0,
            request=request,
            num_computed_tokens=0,
            num_new_tokens=4,
            block_ids_by_group=blocks,
        )
        consume = node.wrap_results(lambda *args: events.append(("compute_done", "request")))
        consume(SimpleNamespace(scheduled_requests=[scheduled]), {})
        blocks["kv"].append(3)
        chunk = await node.next_prefill_chunk("request")
        assert (chunk.chunk_id, chunk.start_token, chunk.end_token, chunk.final) == (0, 0, 4, False)
        assert chunk.block_ids_by_group == {"kv": (1, 2)}
        assert events == [("compute_done", "request"), ("hold", "request")]
        node.complete_prefill_chunk_transfer("request")
        scheduled.num_computed_tokens = 4
        consume(SimpleNamespace(scheduled_requests=[scheduled]), {"request": [42]})
        final = await node.next_prefill_chunk("request")
        assert (final.chunk_id, final.first_token, final.final) == (1, 42, True)
        node.acknowledge_prefill_handoff("request")
        assert events[-1] == ("finish", "request")
        await stream.aclose()
        assert not node.chunks
        assert not node.chunk_counters

    asyncio.run(exercise())


def test_pd_worker_factory_is_spawn_serializable_without_runtime_resources():
    config = SimpleNamespace(role="decode")
    factory = DSV4_DSPARK_K7_ADAPTER.worker_factory(config)
    restored = pickle.loads(pickle.dumps(factory))
    assert restored.func is DSparkPDWorker
    assert vars(restored.keywords["config"]) == vars(config)


def test_pd_worker_services_wrap_only_cache_initialized_requests():
    batch = DecodeBatch(["adopted", "local"], None, None, None)
    flags = {
        "adopted": SimpleNamespace(initialize_from_cache=True),
        "local": SimpleNamespace(initialize_from_cache=False),
    }
    events = []
    model = SimpleNamespace(
        handle_command=lambda op, payload: payload,
        set_profile_active=events.append,
    )
    services = PDWorkerServices((model,), flags.__getitem__)
    prepared = services.wrap_batch(lambda *args, **kwargs: batch)(
        [SimpleNamespace(request_id=name) for name in flags], None
    )
    assert batch.initial_request_ids == ()
    assert prepared.initial_request_ids == ("adopted",)
    assert services.handle_command("inspect", b"value") == b"value"
    services.set_profile_active(True)
    assert events == [True]
    with pytest.raises(ValueError, match="one model"):
        PDWorkerServices((), flags.__getitem__)
