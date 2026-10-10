# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import asyncio
from dataclasses import replace

import pytest

from pypto_serving.config.types import GenerateConfig
from pypto_serving.model.deepseek_dspark.pd_adapter import DSV4_DSPARK_K7_ADAPTER
from pypto_serving.serving.pd.config import PDConfig, PDRole
from pypto_serving.serving.pd.http_api import (
    AuthorizeRouteHTTP,
    ExecutePrefillHTTP,
    ReservePlacementHTTP,
    capability_compatibility_digest,
)
from pypto_serving.serving.pd.protocol import HandoffKey
from pypto_serving.serving.pd.service import PDServingService
from pypto_serving.serving.reasoning import ParsedToolCall, ToolCallDelta
from pypto_serving.serving.pd.worker_api import (
    OP_POLL_TRANSFER_CHUNK,
    TransferPollRequest,
    TransferPollResponse,
    decode_worker_payload,
    encode_worker_payload,
)

from .test_service import _FakeCore, _OverlapFakeCore, _free_port


def _external_config(role: PDRole, port: int) -> PDConfig:
    return PDConfig(
        role=role,
        node_id="p" if role is PDRole.PREFILL else "d",
        run_id="external-run",
        control_host="127.0.0.1",
        control_advertise_host="127.0.0.1",
        control_port=port,
        transfer_hostname="127.0.0.1",
        model_revision="ds-v4-test",
        model_adapter=DSV4_DSPARK_K7_ADAPTER,
        connect_timeout_seconds=3,
        prepared_request_ttl_seconds=30,
    )


@pytest.mark.parametrize("source_partition", range(4))
def test_external_nodes_start_independently_and_d_streams_directly(source_partition) -> None:
    port = _free_port()

    async def exercise() -> None:
        p_core = _FakeCore(PDRole.PREFILL, source_partition=source_partition)
        d_core = _FakeCore(PDRole.DECODE, capacity_slots=1)
        prefill = PDServingService(
            p_core,
            _external_config(PDRole.PREFILL, port),
        )
        decode = PDServingService(
            d_core,
            _external_config(PDRole.DECODE, port),
        )
        # Neither start waits for a configured peer.
        await prefill.start()
        await decode.start()
        assert prefill.session is None
        assert decode.session is None

        config = GenerateConfig(max_new_tokens=2, stream=True, ignore_eos=True)
        prepared = prefill.prepare_request("request", "prompt", config, tuple(range(33)))
        key = HandoffKey("request", "handoff", 1, 1, 1)
        placement = decode.reserve_placement(
            ReservePlacementHTTP(
                key=key,
                prepared_request_id=prepared.prepared_request_id,
                prepared_digest=prepared.prepared_digest,
                prompt_token_count=33,
                max_new_tokens=2,
                layout_fingerprint=prefill.descriptor().capabilities.layout_fingerprint,
                prefill_node_id="p",
                prefill_endpoint_generation=1,
            )
        )
        with pytest.raises(ValueError, match="different P binding"):
            decode.reserve_placement(
                ReservePlacementHTTP(
                    key=key,
                    prepared_request_id=prepared.prepared_request_id,
                    prepared_digest=prepared.prepared_digest,
                    prompt_token_count=33,
                    max_new_tokens=2,
                    layout_fingerprint=(
                        prefill.descriptor().capabilities.layout_fingerprint
                    ),
                    prefill_node_id="other-p",
                    prefill_endpoint_generation=1,
                )
            )
        with pytest.raises(ValueError, match="control incarnation is stale"):
            decode.reserve_placement(
                ReservePlacementHTTP(
                    key=HandoffKey("request-2", "handoff-2", 1, 1, 2),
                    prepared_request_id="prepared-stale",
                    prepared_digest="e" * 64,
                    prompt_token_count=33,
                    max_new_tokens=2,
                    layout_fingerprint=(
                        prefill.descriptor().capabilities.layout_fingerprint
                    ),
                    prefill_node_id="p",
                    prefill_endpoint_generation=1,
                )
            )
        compatibility_digest = capability_compatibility_digest(
            prefill.descriptor().capabilities
        )
        decode.authorize_route(
            AuthorizeRouteHTTP(
                key=key,
                prepared_request_id=prepared.prepared_request_id,
                prepared_digest=prepared.prepared_digest,
                reservation_id=placement.reservation_id,
                reservation_capability=placement.reservation_capability,
                compatibility_digest=compatibility_digest,
                prefill_node_id="p",
                prefill_endpoint_generation=1,
                decode_node_id="d",
                decode_endpoint_generation=1,
            )
        )
        stream = decode.open_decode_stream(key)
        waiting = await anext(stream)
        assert waiting.event == "waiting"
        execute = asyncio.create_task(
            prefill.execute_prefill(
                ExecutePrefillHTTP(
                    key=key,
                    prepared_request_id=prepared.prepared_request_id,
                    prepared_digest=prepared.prepared_digest,
                    reservation_id=placement.reservation_id,
                    partition=placement.partition,
                    block_ids_by_group=placement.block_ids_by_group,
                    reservation_capability=placement.reservation_capability,
                    compatibility_digest=compatibility_digest,
                    prefill_node_id="p",
                    prefill_endpoint_generation=1,
                    decode_node_id="d",
                    decode_control_host="127.0.0.1",
                    decode_control_port=port,
                    decode_endpoint_generation=1,
                )
            )
        )
        outputs = []
        async for frame in stream:
            assert frame.output is not None
            outputs.append(frame.output)
        result = await execute
        assert result.state == "READY"
        assert [output.token_id for output in outputs] == [101, 102]
        assert [output.output_sequence for output in outputs] == [1, 2]
        assert outputs[-1].finished
        assert p_core.source_released
        assert p_core.manifests[0].rank_mapping == DSV4_DSPARK_K7_ADAPTER.rank_mapping(
            (16, 4), source_partition, placement.partition,
        )
        await asyncio.gather(prefill.close(), decode.close())

    asyncio.run(asyncio.wait_for(exercise(), 15))


async def _open_test_handoff(prefill, decode, port, *, request_id="overlap", prompt_tokens=257):
    config = GenerateConfig(max_new_tokens=2, stream=True, ignore_eos=True)
    prepared = prefill.prepare_request(request_id, "prompt", config, tuple(range(prompt_tokens)))
    key = HandoffKey(request_id, f"handoff-{request_id}", 1, 1, 1)
    placement = decode.reserve_placement(ReservePlacementHTTP(
        key, prepared.prepared_request_id, prepared.prepared_digest, prompt_tokens, 2,
        prefill.descriptor().capabilities.layout_fingerprint, "p", 1,
    ))
    digest = capability_compatibility_digest(prefill.descriptor().capabilities)
    decode.authorize_route(AuthorizeRouteHTTP(
        key, prepared.prepared_request_id, prepared.prepared_digest,
        placement.reservation_id, placement.reservation_capability, digest, "p", 1, "d", 1,
    ))
    stream = decode.open_decode_stream(key)
    assert (await anext(stream)).event == "waiting"
    execute = asyncio.create_task(prefill.execute_prefill(ExecutePrefillHTTP(
        key, prepared.prepared_request_id, prepared.prepared_digest,
        placement.reservation_id, placement.partition, placement.block_ids_by_group,
        placement.reservation_capability, digest, "p", 1, "d", "127.0.0.1", port, 1,
    )))
    return key, stream, execute


def test_decode_node_stream_preserves_tool_call_deltas_and_final_calls():
    class ToolDecodeCore(_FakeCore):
        async def add_adopted_handoff(self, **kwargs):
            async for output in super().add_adopted_handoff(**kwargs):
                if output.finished:
                    yield replace(
                        output,
                        text="",
                        reasoning="Need data",
                        text_delta="",
                        reasoning_delta="",
                        tool_call_deltas=(ToolCallDelta(0, arguments='{"city":"杭州"}'),),
                        tool_calls=(ParsedToolCall("call-1", "lookup", '{"city":"杭州"}'),),
                        finish_reason="FINISHED_EOS",
                    )
                else:
                    yield replace(
                        output,
                        text="",
                        reasoning="Need data",
                        text_delta="",
                        reasoning_delta="Need data",
                        tool_call_deltas=(ToolCallDelta(0, "call-1", "lookup"),),
                    )

    async def exercise():
        port = _free_port()
        prefill = PDServingService(_FakeCore(PDRole.PREFILL), _external_config(PDRole.PREFILL, port))
        decode = PDServingService(ToolDecodeCore(PDRole.DECODE), _external_config(PDRole.DECODE, port))
        await asyncio.gather(prefill.start(), decode.start())
        try:
            _, stream, execute = await _open_test_handoff(prefill, decode, port, request_id="tool-call")
            outputs = [frame.output async for frame in stream]
            assert (await execute).state == "READY"
            assert outputs[0].tool_call_deltas[0].id == "call-1"
            assert outputs[-1].tool_call_deltas[0].arguments == '{"city":"杭州"}'
            assert outputs[-1].tool_calls == (ParsedToolCall("call-1", "lookup", '{"city":"杭州"}'),)
            assert outputs[-1].finish_reason == "FINISHED_EOS"
        finally:
            await asyncio.gather(prefill.close(), decode.close())

    asyncio.run(asyncio.wait_for(exercise(), 15))


@pytest.mark.parametrize("overlap", (False, True))
@pytest.mark.parametrize("source_partition", (0, 1, 3))
def test_external_chunk_transfer_keeps_source_pinned_and_overlaps_when_enabled(overlap, source_partition):
    async def exercise():
        port = _free_port()
        p_core = _OverlapFakeCore(PDRole.PREFILL, source_partition=source_partition)
        d_core = _FakeCore(PDRole.DECODE)
        prefill = PDServingService(p_core, replace(
            _external_config(PDRole.PREFILL, port), enable_chunk_overlap=overlap,
        ))
        decode = PDServingService(d_core, _external_config(PDRole.DECODE, port))
        await asyncio.gather(prefill.start(), decode.start())
        stream = execute = None
        try:
            key, stream, execute = await _open_test_handoff(prefill, decode, port)
            await asyncio.wait_for(p_core.transfer_started.wait(), 3)
            if overlap:
                await asyncio.wait_for(p_core.final_chunk_queued.wait(), 3)
            else:
                await asyncio.sleep(0.03)
                assert not p_core.next_chunk_released.is_set()
                assert not p_core.final_chunk_queued.is_set()
            assert not execute.done()
            assert not p_core.source_released
            assert prefill.source_lifecycle._in_flight[key.request_id].partition == source_partition
            assert decode.query_handoff(key).state == "TRANSFERRING"
            p_core.allow_completion.set()
            outputs = [frame.output async for frame in stream]
            assert (await execute).state == "READY"
            assert outputs[-1].finished
            assert p_core.source_released
            assert not prefill.source_lifecycle._in_flight
            assert p_core.transfer_calls == 2
            assert p_core.manifests[0].rank_mapping == p_core.manifests[1].rank_mapping
            assert prefill.metrics.snapshot()["counters"].get("overlap.completed", 0) == int(overlap)
        finally:
            p_core.allow_completion.set()
            if execute is not None and not execute.done():
                execute.cancel()
                await asyncio.gather(execute, return_exceptions=True)
            if stream is not None:
                await stream.aclose()
            await asyncio.gather(prefill.close(), decode.close())

    asyncio.run(asyncio.wait_for(exercise(), 15))


def test_slow_final_transfer_does_not_block_another_handoff():
    async def exercise():
        port = _free_port()
        core = _FakeCore(PDRole.PREFILL, capacity_slots=4)
        started, release = asyncio.Event(), asyncio.Event()
        original_call = core.call_pd_worker

        async def call(operation, payload=b""):
            if operation == OP_POLL_TRANSFER_CHUNK:
                poll = decode_worker_payload(payload, TransferPollRequest)
                request, _ = core.transfer_jobs[poll.job_id]
                if request.manifest.key.request_id == "slow" and not release.is_set():
                    started.set()
                    return encode_worker_payload(TransferPollResponse(poll.job_id, False))
            return await original_call(operation, payload)

        core.call_pd_worker = call
        prefill = PDServingService(core, _external_config(PDRole.PREFILL, port))
        decode = PDServingService(_FakeCore(PDRole.DECODE, capacity_slots=4), _external_config(PDRole.DECODE, port))
        await asyncio.gather(prefill.start(), decode.start())
        streams, tasks = [], []
        try:
            key, stream, task = await _open_test_handoff(prefill, decode, port, request_id="slow", prompt_tokens=33)
            streams.append(stream)
            tasks.append(task)
            await asyncio.wait_for(started.wait(), 3)
            _, fast_stream, fast_task = await _open_test_handoff(
                prefill, decode, port, request_id="fast", prompt_tokens=33,
            )
            streams.append(fast_stream)
            tasks.append(fast_task)
            fast_outputs = [frame.output async for frame in fast_stream]
            assert (await fast_task).state == "READY"
            assert fast_outputs[-1].finished
            assert not task.done()
            assert key.request_id in prefill.source_lifecycle._in_flight
            release.set()
            assert [frame.output async for frame in stream][-1].finished
            assert (await task).state == "READY"
            assert not prefill.source_lifecycle._in_flight
        finally:
            release.set()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            for stream in streams:
                await stream.aclose()
            await asyncio.gather(prefill.close(), decode.close())

    asyncio.run(asyncio.wait_for(exercise(), 15))


def test_external_control_session_multiplexes_two_handoffs() -> None:
    port = _free_port()

    async def exercise() -> None:
        p_core = _FakeCore(PDRole.PREFILL, capacity_slots=4)
        d_core = _FakeCore(PDRole.DECODE, capacity_slots=4)
        prefill = PDServingService(
            p_core,
            _external_config(PDRole.PREFILL, port),
        )
        decode = PDServingService(
            d_core,
            _external_config(PDRole.DECODE, port),
        )
        await asyncio.gather(prefill.start(), decode.start())

        async def run(index: int):
            request_id = f"request-{index}"
            prepared = prefill.prepare_request(
                request_id,
                "prompt",
                GenerateConfig(max_new_tokens=2, stream=True, ignore_eos=True),
                tuple(range(33)),
            )
            key = HandoffKey(request_id, f"handoff-{index}", 1, 1, 1)
            placement = decode.reserve_placement(
                ReservePlacementHTTP(
                    key=key,
                    prepared_request_id=prepared.prepared_request_id,
                    prepared_digest=prepared.prepared_digest,
                    prompt_token_count=33,
                    max_new_tokens=2,
                    layout_fingerprint=(
                        prefill.descriptor().capabilities.layout_fingerprint
                    ),
                    prefill_node_id="p",
                    prefill_endpoint_generation=1,
                )
            )
            compatibility_digest = capability_compatibility_digest(
                prefill.descriptor().capabilities
            )
            decode.authorize_route(
                AuthorizeRouteHTTP(
                    key,
                    prepared.prepared_request_id,
                    prepared.prepared_digest,
                    placement.reservation_id,
                    placement.reservation_capability,
                    compatibility_digest,
                    "p",
                    1,
                    "d",
                    1,
                )
            )
            stream = decode.open_decode_stream(key)
            assert (await anext(stream)).event == "waiting"
            execute = asyncio.create_task(
                prefill.execute_prefill(
                    ExecutePrefillHTTP(
                        key,
                        prepared.prepared_request_id,
                        prepared.prepared_digest,
                        placement.reservation_id,
                        placement.partition,
                        placement.block_ids_by_group,
                        placement.reservation_capability,
                        compatibility_digest,
                        "p",
                        1,
                        "d",
                        "127.0.0.1",
                        port,
                        1,
                    )
                )
            )
            outputs = [frame.output async for frame in stream]
            result = await execute
            return key, result, outputs

        results = await asyncio.gather(run(1), run(2))
        assert {result[0].request_id for result in results} == {
            "request-1",
            "request-2",
        }
        assert all(result[1].state == "READY" for result in results)
        assert all(result[2][-1].finished for result in results)
        assert p_core.transfer_calls == 2
        await asyncio.gather(prefill.close(), decode.close())

    asyncio.run(asyncio.wait_for(exercise(), 15))
