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
import msgspec
from types import SimpleNamespace

from pypto_serving.config.types import GenerateConfig
from pypto_serving.serving.constraints import ConstraintSpec
from pypto_serving.model.deepseek_dspark.pd_adapter import DSV4_DSPARK_K7_ADAPTER
from pypto_serving.serving.pd.config import PDConfig, PD_SCHEMA_VERSION, PDRole
from pypto_serving.serving.pd.protocol import BeginHandoff, GenerationContext, HandoffKey
from pypto_serving.serving.pd.admission import PDBackpressureError
from pypto_serving.serving.pd.service import PDServingService
from pypto_serving.serving.reasoning import OutputParserSpec, ParsedToolCall, ToolCallDelta
from pypto_serving.serving.pd.worker_api import TransferPollResponse
from pypto_serving.transfer.types import CompletionCertainty

from .test_service import _FakeCore, _OverlapFakeCore, _free_port


def _external_config(role, port):
    return PDConfig(
        role=role, node_id="p" if role is PDRole.PREFILL else "d", run_id="run",
        control_host="127.0.0.1", control_advertise_host="127.0.0.1", control_port=port,
        transfer_hostname="127.0.0.1", model_revision="ds-v4-test",
        model_adapter=DSV4_DSPARK_K7_ADAPTER, connect_timeout_seconds=3, request_timeout_seconds=5,
        observability_enabled=False,
    )


def _context(port, request_id="request"):
    return GenerationContext(HandoffKey(request_id, f"handoff-{request_id}", 1, 1, 1),
                             "a" * 64, "p", 1, "d", 1, "127.0.0.1", port, PD_SCHEMA_VERSION)


async def _open_test_handoff(prefill, decode, port, *, request_id="overlap", prompt_tokens=257):
    context = _context(port, request_id)
    local = decode.register_generation(context)
    async def execute():
        prefill.register_generation(context)
        prepared = prefill.prepare_input(
            "prompt", GenerateConfig(max_new_tokens=2, stream=True, ignore_eos=True), tuple(range(prompt_tokens)),
        )
        return await prefill.prefill(context, prepared)
    task = asyncio.create_task(execute())
    return context.key, decode.decode_outputs(local), task


@pytest.mark.parametrize("source_partition", range(4))
def test_nodes_own_reservation_and_direct_outputs(source_partition):
    async def exercise():
        port = _free_port()
        prefill = PDServingService(_FakeCore(PDRole.PREFILL, source_partition=source_partition),
                                   _external_config(PDRole.PREFILL, port))
        decode = PDServingService(_FakeCore(PDRole.DECODE, capacity_slots=1), _external_config(PDRole.DECODE, port))
        await asyncio.gather(prefill.start(), decode.start())
        try:
            key, stream, task = await _open_test_handoff(prefill, decode, port)
            outputs = [output async for output in stream]
            result = await task
            assert result.state == "READY"
            assert [output.token_id for output in outputs] == [101, 102]
            assert outputs[-1].finished
            assert prefill.core.source_released
            assert decode.query_handoff(key).state == "COMPLETED"
            partition = prefill.core.manifests[0].rank_mapping[0].destination_rank_id // 4
            assert prefill.core.manifests[0].rank_mapping == DSV4_DSPARK_K7_ADAPTER.rank_mapping(
                (16, 4), source_partition, partition,
            )
            assert not prefill._requests and not decode._requests
        finally:
            await asyncio.gather(prefill.close(), decode.close())
    asyncio.run(asyncio.wait_for(exercise(), 15))


@pytest.mark.parametrize("valid,budget", [(True, 2), (False, 2), (False, 1)])
def test_decode_seed_validation_precedes_ready_and_public_output(valid, budget, monkeypatch):
    monkeypatch.setattr("pypto_serving.serving.pd.service.importlib.util.find_spec", lambda _: object())
    async def exercise():
        port = _free_port()
        prefill = PDServingService(_FakeCore(PDRole.PREFILL), _external_config(PDRole.PREFILL, port))
        d_core = _FakeCore(PDRole.DECODE)
        decode = PDServingService(d_core, _external_config(PDRole.DECODE, port))
        events = []
        original_prefill = prefill.core.add_request

        def constrained_prefill(*args, constraint_spec=None, **kwargs):
            assert constraint_spec is not None
            return original_prefill(*args, **kwargs)

        prefill.core.add_request = constrained_prefill
        original_adopt = d_core.add_adopted_handoff

        async def adopt(**kwargs):
            events.append("adopt")
            async for output in original_adopt(**kwargs):
                yield output

        async def validate(spec, initial_tokens):
            assert spec.format_id == "deepseek_v4" and initial_tokens == (101,)
            events.append("validate")
            if not valid:
                raise ValueError("invalid first token")

        d_core.add_adopted_handoff = adopt
        decode.preflight_constraint = validate
        await asyncio.gather(prefill.start(), decode.start())
        context = _context(port, "constrained")
        local = decode.register_generation(context)
        prefill.register_generation(context)
        spec = ConstraintSpec(
            provider_id="xgrammar", format_id="deepseek_v4",
            tools=({"type": "function", "function": {"name": "lookup"}},),
            tool_choice="required", reasoning=False,
        )
        try:
            prepared = prefill.prepare_input(
                "prompt", GenerateConfig(max_new_tokens=budget, ignore_eos=True), tuple(range(257)),
                output_parser_spec=OutputParserSpec("deepseek_v4", "content"), constraint_spec=spec,
            )
            task = asyncio.create_task(prefill.prefill(context, prepared))
            if valid:
                outputs = [output async for output in decode.decode_outputs(local)]
                assert (await task).state == "READY"
                assert outputs[0].token_id == 101 and events == ["validate", "adopt"]
            else:
                result = await task
                assert result.state == "FAILED" and result.error_code == "INVALID_DECODE_SEED"
                assert not result.retryable
                with pytest.raises(RuntimeError, match="INVALID_DECODE_SEED"):
                    _ = [output async for output in decode.decode_outputs(local)]
                assert events == ["validate"]
                assert decode.query_handoff(context.key).state == "ABORTED"
                assert not decode.health_error and not prefill.health_error
                assert not decode._requests and not prefill._requests
                assert prefill.core.kv_cache_manager.group_request_partition(context.key.request_id) is None
        finally:
            await asyncio.gather(prefill.close(), decode.close())

    asyncio.run(asyncio.wait_for(exercise(), 15))


@pytest.mark.parametrize("certainty", (CompletionCertainty.NOT_SUBMITTED, CompletionCertainty.UNKNOWN))
def test_transfer_failure_preserves_certainty_and_shared_page_safety(certainty):
    async def exercise():
        port = _free_port()
        p_core = _FakeCore(PDRole.PREFILL, transfer_certainties=(certainty, certainty))
        prefill = PDServingService(p_core, _external_config(PDRole.PREFILL, port))
        decode = PDServingService(_FakeCore(PDRole.DECODE), _external_config(PDRole.DECODE, port))
        await asyncio.gather(prefill.start(), decode.start())
        try:
            key, stream, task = await _open_test_handoff(prefill, decode, port)
            with pytest.raises(RuntimeError, match="transfer failed"):
                await task
            state = decode.query_handoff(key).state
            assert state == ("QUARANTINED" if certainty is CompletionCertainty.UNKNOWN else "ABORTED")
            assert bool(prefill.health_error) == (certainty is CompletionCertainty.UNKNOWN)
            await stream.aclose()
        finally:
            await asyncio.gather(prefill.close(), decode.close(), return_exceptions=True)
    asyncio.run(asyncio.wait_for(exercise(), 15))


def test_late_generation_after_abort_is_rejected_without_reservation():
    async def exercise():
        service = PDServingService(_FakeCore(PDRole.DECODE), _external_config(PDRole.DECODE, 12345))
        await service.start()
        context = _context(12345)
        status = await service.abort_handoff(context.key, "client cancelled", deterministic=True)
        assert status.state == "ABORTED"
        with pytest.raises(ValueError, match="already used"):
            service.register_generation(context)
        assert not service.core.kv_cache_manager.group_cache_reservations
        await service.close()
    asyncio.run(exercise())


def test_decode_request_rejects_stale_or_duplicate_bindings():
    async def exercise():
        service = PDServingService(_FakeCore(PDRole.DECODE), _external_config(PDRole.DECODE, 12345))
        await service.start()
        context = _context(12345)
        service.register_generation(context)
        with pytest.raises(ValueError, match="already used"):
            service.register_generation(context)
        with pytest.raises(ValueError, match="incarnation"):
            service.register_generation(msgspec.structs.replace(context, decode_endpoint_generation=2))
        await service.abort_handoff(context.key, "test", deterministic=True)
        await service.close()
    asyncio.run(exercise())


def test_reused_session_allows_control_to_arrive_before_decode_http():
    async def exercise():
        port = _free_port()
        prefill = PDServingService(_FakeCore(PDRole.PREFILL), _external_config(PDRole.PREFILL, port))
        decode = PDServingService(_FakeCore(PDRole.DECODE), _external_config(PDRole.DECODE, port))
        await asyncio.gather(prefill.start(), decode.start())
        try:
            _, stream, task = await _open_test_handoff(prefill, decode, port, request_id="first")
            assert [out async for out in stream][-1].finished
            await task
            context = _context(port, "second")
            async def source():
                prefill.register_generation(context)
                return await prefill.prefill(context, prefill.prepare_input(
                    "prompt", GenerateConfig(max_new_tokens=2, stream=True), (1, 2, 3),
                ))
            task = asyncio.create_task(source())
            await asyncio.sleep(0.03)
            assert not task.done()
            local = decode.register_generation(context)
            assert [out async for out in decode.decode_outputs(local)][-1].finished
            assert (await task).state == "READY"
        finally:
            await asyncio.gather(prefill.close(), decode.close())
    asyncio.run(asyncio.wait_for(exercise(), 15))


@pytest.mark.parametrize("mismatch", ("peer", "context"))
def test_reservation_requires_the_matching_http_request_and_source_peer(mismatch):
    async def exercise():
        port = _free_port()
        service = PDServingService(_FakeCore(PDRole.DECODE), _external_config(PDRole.DECODE, port))
        await service.start()
        context = _context(port)
        local = service.register_generation(context)
        source = PDServingService(_FakeCore(PDRole.PREFILL), _external_config(PDRole.PREFILL, port))
        prepared = source.prepare_input("prompt", GenerateConfig(max_new_tokens=2), (1, 2, 3))
        sent = msgspec.structs.replace(context, prefill_endpoint_generation=2) if mismatch == "context" else context
        begin = BeginHandoff(context.key, sent, prepared.continuation, prepared.prefix_match_spec, prepared.digest)
        peer = SimpleNamespace(peer_hello=SimpleNamespace(
            node_id="foreign" if mismatch == "peer" else "p", endpoint_generation=1,
        ))
        try:
            with pytest.raises(ValueError):
                service._reserve_handoff(begin, peer)
            assert not local.accepted.done()
            assert not service.core.kv_cache_manager.group_cache_reservations
        finally:
            await service.abort_handoff(context.key, "test", deterministic=True)
            await service.close()
    asyncio.run(exercise())


def test_unpaired_context_deadline_and_pending_bound_do_not_allocate_pages():
    async def exercise():
        config = replace(_external_config(PDRole.DECODE, 12345), request_timeout_seconds=0.02)
        service = PDServingService(_FakeCore(PDRole.DECODE), config)
        with pytest.raises(TimeoutError):
            await service._wait_generation(_context(12345))
        for index in range(config.max_pending_handoffs):
            service.register_generation(_context(12345, f"request-{index}"))
        with pytest.raises(PDBackpressureError):
            service.register_generation(_context(12345, "over-limit"))
        assert not service.core.kv_cache_manager.group_cache_reservations
    asyncio.run(exercise())


def test_drain_retires_unreserved_waiters_before_owner_release():
    async def exercise():
        port = _free_port()
        service = PDServingService(_FakeCore(PDRole.DECODE), _external_config(PDRole.DECODE, port))
        await service.start()
        context = _context(port)
        local = service.register_generation(context)
        assert not service.lifecycle.idle()
        await service.close()
        assert service._closed and not service._requests
        assert service.query_handoff(context.key).state == "ABORTED"
        assert isinstance(await local.accepted, RuntimeError)
    asyncio.run(exercise())


def test_uncertain_abort_stops_prefill_orchestration_but_retains_recovery_gate():
    async def exercise():
        service = PDServingService(_FakeCore(PDRole.PREFILL), _external_config(PDRole.PREFILL, 12345))
        context = _context(12345)
        entered, stopped = asyncio.Event(), asyncio.Event()
        async def source():
            service.register_generation(context)
            entered.set()
            try:
                await asyncio.Future()
            finally:
                stopped.set()
        task = asyncio.create_task(source())
        await entered.wait()
        status = await service.abort_handoff(context.key, "source unknown", deterministic=False)
        assert status.state == "RECOVERY_REQUIRED"
        assert task.cancelled() and stopped.is_set()
        assert context.key in service._active_keys and not service.lifecycle.idle()
        assert service.health_error and service.core.transfer_calls == 0
    asyncio.run(exercise())


def test_cancel_stops_producer_blocked_on_output_backpressure():
    class SlowCore(_FakeCore):
        async def add_adopted_handoff(self, **kwargs):
            self.kv_cache_manager.adopt_group_cache(kwargs["reservation_id"])
            for _ in range(1000):
                yield replace_token()
    def replace_token():
        from pypto_serving.serving.engine.async_engine import TokenOutput
        return TokenOutput(text="token")
    async def exercise():
        port = _free_port()
        prefill = PDServingService(_FakeCore(PDRole.PREFILL), _external_config(PDRole.PREFILL, port))
        decode = PDServingService(SlowCore(PDRole.DECODE), _external_config(PDRole.DECODE, port))
        await asyncio.gather(prefill.start(), decode.start())
        try:
            key, stream, task = await _open_test_handoff(prefill, decode, port)
            await task
            await asyncio.sleep(0.03)
            status = await decode.abort_handoff(key, "cancel", deterministic=True)
            assert status.state == "ABORTED"
            assert key not in decode._external_handoff_tasks
            assert not decode._requests
            await stream.aclose()
        finally:
            await asyncio.gather(prefill.close(), decode.close())
    asyncio.run(asyncio.wait_for(exercise(), 15))


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
            outputs = [output async for output in stream]
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
            outputs = [output async for output in stream]
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
        original_poll = core.poll_transfer

        def poll(job_id):
            request, _ = core.transfer_jobs[job_id]
            if request.manifest.key.request_id == "slow" and not release.is_set():
                started.set()
                return TransferPollResponse(job_id, False)
            return original_poll(job_id)

        core.poll_transfer = poll
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
            fast_outputs = [output async for output in fast_stream]
            assert (await fast_task).state == "READY"
            assert fast_outputs[-1].finished
            assert not task.done()
            assert key.request_id in prefill.source_lifecycle._in_flight
            release.set()
            assert [output async for output in stream][-1].finished
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


def test_external_control_session_multiplexes_two_handoffs():
    async def exercise():
        port = _free_port()
        prefill = PDServingService(_FakeCore(PDRole.PREFILL, capacity_slots=4), _external_config(PDRole.PREFILL, port))
        decode = PDServingService(_FakeCore(PDRole.DECODE, capacity_slots=4), _external_config(PDRole.DECODE, port))
        await asyncio.gather(prefill.start(), decode.start())
        async def run(index):
            key, stream, task = await _open_test_handoff(prefill, decode, port, request_id=f"request-{index}")
            outputs = [output async for output in stream]
            assert (await task).state == "READY"
            assert outputs[-1].finished
            return key
        try:
            keys = await asyncio.gather(run(1), run(2))
            assert {key.request_id for key in keys} == {"request-1", "request-2"}
            assert prefill.core.transfer_calls == 2
            assert not prefill._requests and not decode._requests
        finally:
            await asyncio.gather(prefill.close(), decode.close())
    asyncio.run(asyncio.wait_for(exercise(), 15))
