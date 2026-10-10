# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import asyncio
import socket

import pytest

from pypto_serving.serving.pd.config import PDCapabilities, PDConfig, PDRole
from pypto_serving.model.deepseek_dspark.pd_adapter import (
    DSV4_DSPARK_K7_ADAPTER,
    DSV4_DSPARK_K7_CONTRACT,
)
from pypto_serving.serving.pd.protocol import (
    HandoffKey,
    HandoffStatus,
    QueryHandoff,
    RankRegistration,
    RegionRegistration,
    RegistryAdvertisement,
)
from pypto_serving.serving.pd.session import (
    MultiplexedPDControlSession,
    PDControlAcceptor,
    connect_control_session,
)


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _config(role: PDRole, port: int) -> PDConfig:
    local = "p" if role is PDRole.PREFILL else "d"
    return PDConfig(
        role=role,
        node_id=local,
        run_id="run",
        control_host="127.0.0.1",
        control_port=port,
        control_advertise_host="127.0.0.1",
        transfer_hostname="127.0.0.1",
        model_revision="model",
        model_adapter=DSV4_DSPARK_K7_ADAPTER,
        connect_timeout_seconds=3,
    )


def _capabilities() -> PDCapabilities:
    return PDCapabilities(
        adapter_id=DSV4_DSPARK_K7_CONTRACT.adapter_id,
        contract_version=DSV4_DSPARK_K7_CONTRACT.version,
        contract_digest=DSV4_DSPARK_K7_CONTRACT.digest,
        continuation_schema=DSV4_DSPARK_K7_CONTRACT.continuation_schema,
        model_revision="model",
        registry_fingerprint="f" * 64,
        layout_fingerprint="l" * 64,
        topology=(1, 1),
        logical_groups=DSV4_DSPARK_K7_CONTRACT.logical_groups,
        physical_regions=DSV4_DSPARK_K7_CONTRACT.physical_regions,
    )


def _advertisement() -> RegistryAdvertisement:
    return RegistryAdvertisement(
        model_revision="model",
        topology=(1, 1),
        registry_fingerprint="f" * 64,
        layout_fingerprint="l" * 64,
        ranks=(
            RankRegistration(
                rank_id=0,
                owner_generation=1,
                endpoint_generation=1,
                worker_id="worker",
                regions=tuple(
                    RegionRegistration(component, 1, 64, b"{}")
                    for component in DSV4_DSPARK_K7_CONTRACT.physical_regions
                ),
            ),
        ),
    )


async def _open_pair(port: int):
    decode_config = _config(PDRole.DECODE, port)
    acceptor = PDControlAcceptor(decode_config)
    decode_task = asyncio.create_task(
        acceptor.accept(_capabilities(), expected_peer_node_id="p")
    )
    prefill = await connect_control_session(
        _config(PDRole.PREFILL, port),
        _capabilities(),
        peer_node_id="d",
        peer_host="127.0.0.1",
        peer_port=port,
    )
    decode = await decode_task
    acceptor.close()
    return decode, prefill


def test_loopback_tcp_session_exchanges_registry() -> None:
    port = _free_port()

    async def exercise() -> None:
        decode, prefill = await _open_pair(port)
        d_registry, p_registry = await asyncio.gather(
            decode.exchange_registry(_advertisement()),
            prefill.exchange_registry(_advertisement()),
        )
        assert d_registry.registry_fingerprint == p_registry.registry_fingerprint

        key = HandoffKey("request", "handoff", 1, 1, 1)
        await prefill.send(QueryHandoff(key))
        assert await decode.receive() == QueryHandoff(key)
        await decode.send(HandoffStatus(key, "RESERVED"))
        assert await prefill.receive() == HandoffStatus(key, "RESERVED")
        await asyncio.gather(prefill.close(), decode.close())

    asyncio.run(exercise())


def test_multiplexed_session_dispatches_interleaved_handoffs() -> None:
    port = _free_port()

    async def exercise() -> None:
        decode, prefill = await _open_pair(port)
        await asyncio.gather(
            decode.exchange_registry(_advertisement()),
            prefill.exchange_registry(_advertisement()),
        )
        multiplex = MultiplexedPDControlSession(prefill)
        first = HandoffKey("request-1", "handoff-1", 1, 1, 1)
        second = HandoffKey("request-2", "handoff-2", 1, 1, 1)
        multiplex.open_route(first)
        multiplex.open_route(second)
        multiplex.start()

        await decode.send(HandoffStatus(second, "READY"))
        await decode.send(HandoffStatus(first, "RESERVED"))
        first_reply, second_reply = await asyncio.gather(
            multiplex.receive(first),
            multiplex.receive(second),
        )
        assert first_reply.key == first
        assert second_reply.key == second
        multiplex.close_route(first)
        multiplex.close_route(second)
        await asyncio.gather(multiplex.close(), decode.close())

    asyncio.run(exercise())


def test_multiplexed_session_fails_every_open_route_on_peer_eof() -> None:
    port = _free_port()

    async def exercise() -> None:
        decode, prefill = await _open_pair(port)
        await asyncio.gather(
            decode.exchange_registry(_advertisement()),
            prefill.exchange_registry(_advertisement()),
        )
        multiplex = MultiplexedPDControlSession(prefill)
        first = HandoffKey("request-1", "handoff-1", 1, 1, 1)
        second = HandoffKey("request-2", "handoff-2", 1, 1, 1)
        multiplex.open_route(first)
        multiplex.open_route(second)
        multiplex.start()
        waiting = [
            asyncio.create_task(multiplex.receive(first)),
            asyncio.create_task(multiplex.receive(second)),
        ]

        await decode.close()
        results = await asyncio.gather(*waiting, return_exceptions=True)
        assert all(isinstance(result, RuntimeError) for result in results)
        assert all("session failed" in str(result) for result in results)
        await multiplex.close()

    asyncio.run(exercise())


def test_multiplexed_session_absorbs_late_reply_for_a_closed_route() -> None:
    port = _free_port()

    async def exercise() -> None:
        decode, prefill = await _open_pair(port)
        await asyncio.gather(
            decode.exchange_registry(_advertisement()),
            prefill.exchange_registry(_advertisement()),
        )
        multiplex = MultiplexedPDControlSession(prefill)
        closed = HandoffKey("request-closed", "handoff-closed", 1, 1, 1)
        active = HandoffKey("request-active", "handoff-active", 1, 1, 1)
        multiplex.open_route(closed)
        multiplex.open_route(active)
        multiplex.start()
        multiplex.close_route(closed)

        await decode.send(HandoffStatus(closed, "READY"))
        await decode.send(HandoffStatus(active, "RESERVED"))
        reply = await multiplex.receive(active)
        assert reply == HandoffStatus(active, "RESERVED")

        multiplex.close_route(active)
        await asyncio.gather(multiplex.close(), decode.close())

    asyncio.run(exercise())


def test_multiplexed_session_fails_all_routes_on_unknown_correlation() -> None:
    port = _free_port()

    async def exercise() -> None:
        decode, prefill = await _open_pair(port)
        await asyncio.gather(
            decode.exchange_registry(_advertisement()),
            prefill.exchange_registry(_advertisement()),
        )
        multiplex = MultiplexedPDControlSession(prefill)
        active = HandoffKey("request-active", "handoff-active", 1, 1, 1)
        unknown = HandoffKey("request-unknown", "handoff-unknown", 1, 1, 1)
        multiplex.open_route(active)
        multiplex.start()

        waiting = asyncio.create_task(multiplex.receive(active))
        await decode.send(HandoffStatus(unknown, "READY"))
        with pytest.raises(RuntimeError, match="session failed"):
            await waiting
        assert multiplex.closed
        await asyncio.gather(multiplex.close(), decode.close())

    asyncio.run(exercise())
