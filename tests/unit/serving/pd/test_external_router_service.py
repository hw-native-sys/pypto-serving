# Copyright (c) PyPTO Contributors.
# Licensed under CANN Open Software License Agreement Version 2.0.

import asyncio

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

from .test_service import _FakeCore, _FakeEngine, _free_port


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


def test_external_nodes_start_independently_and_d_streams_directly() -> None:
    port = _free_port()

    async def exercise() -> None:
        p_core = _FakeCore(PDRole.PREFILL)
        d_core = _FakeCore(PDRole.DECODE, capacity_slots=1)
        prefill = PDServingService(
            _FakeEngine(p_core),
            _external_config(PDRole.PREFILL, port),
        )
        decode = PDServingService(
            _FakeEngine(d_core),
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
        await asyncio.gather(prefill.close(), decode.close())

    asyncio.run(asyncio.wait_for(exercise(), 15))


def test_external_control_session_multiplexes_two_handoffs() -> None:
    port = _free_port()

    async def exercise() -> None:
        p_core = _FakeCore(PDRole.PREFILL, capacity_slots=4)
        d_core = _FakeCore(PDRole.DECODE, capacity_slots=4)
        prefill = PDServingService(
            _FakeEngine(p_core),
            _external_config(PDRole.PREFILL, port),
        )
        decode = PDServingService(
            _FakeEngine(d_core),
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
