# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""PD Host sessions for external-router lazy peers."""

from __future__ import annotations

import asyncio
import contextlib
import socket
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

from .config import PDConfig, PDCapabilities, PDRole
from .protocol import (
    FramedChannel,
    ControlMessage,
    Hello,
    RegistryAdvertisement,
    exchange_and_validate_hello,
    make_hello,
    message_handoff_key,
)


class PDControlSession:
    """Async wrapper around one ordered, authenticated fixed-peer channel."""

    def __init__(
        self,
        sock: socket.socket,
        channel: FramedChannel,
        peer_hello: Hello,
    ) -> None:
        self._socket = sock
        self._channel = channel
        self.peer_hello = peer_hello
        self._send_lock = asyncio.Lock()
        self._receive_lock = asyncio.Lock()
        self._closed = False

    async def send(self, message: ControlMessage) -> None:
        if self._closed:
            raise RuntimeError("PD control session is closed")
        async with self._send_lock:
            await asyncio.to_thread(self._channel.send, message)

    async def receive(self) -> ControlMessage:
        if self._closed:
            raise RuntimeError("PD control session is closed")
        async with self._receive_lock:
            return await asyncio.to_thread(self._channel.receive)

    async def exchange_registry(
        self,
        local: RegistryAdvertisement,
    ) -> RegistryAdvertisement:
        validate_rank_registrations(
            local.ranks,
            expected_count=local.topology[0],
            expected_components=self.peer_hello.capabilities.physical_regions,
        )
        await self.send(local)
        peer = await self.receive()
        if not isinstance(peer, RegistryAdvertisement):
            raise ValueError("PD peer did not advertise its worker registry")
        capabilities = self.peer_hello.capabilities
        if (
            peer.model_revision != capabilities.model_revision
            or peer.topology != capabilities.topology
            or peer.registry_fingerprint != capabilities.registry_fingerprint
            or peer.layout_fingerprint != capabilities.layout_fingerprint
        ):
            raise ValueError("PD peer registry differs from its authenticated hello")
        if len(peer.ranks) != peer.topology[0]:
            raise ValueError("PD peer registry rank count differs from its topology")
        validate_rank_registrations(
            peer.ranks,
            expected_count=peer.topology[0],
            expected_components=capabilities.physical_regions,
        )
        return peer

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(OSError):
            self._socket.shutdown(socket.SHUT_RDWR)
        self._socket.close()

    @property
    def closed(self) -> bool:
        return self._closed


class MultiplexedPDControlSession:
    """One reader/dispatcher with an independent bounded queue per handoff.

    Registry exchange is completed on the raw session before this wrapper is
    started.  Afterwards no caller may use ``PDControlSession.receive``
    directly: the dispatcher is the sole socket reader and correlation is
    always performed with the complete :class:`HandoffKey`.
    """

    def __init__(self, session: PDControlSession, *, queue_size: int = 128) -> None:
        if type(queue_size) is not int or queue_size < 1:
            raise ValueError("PD multiplex queue size must be a positive integer")
        self._session = session
        self._queue_size = queue_size
        self._routes: dict[object, asyncio.Queue[ControlMessage | BaseException]] = {}
        self._closed_keys: deque[object] = deque(maxlen=1024)
        self._reader_task: asyncio.Task | None = None
        self._failure: BaseException | None = None

    @property
    def peer_hello(self) -> Hello:
        return self._session.peer_hello

    @property
    def closed(self) -> bool:
        return self._session.closed

    def start(self) -> None:
        if self._reader_task is not None:
            raise RuntimeError("PD multiplex reader was already started")
        self._reader_task = asyncio.create_task(self._reader())

    def open_route(self, key) -> None:
        if self._failure is not None:
            raise RuntimeError("PD multiplex session has failed") from self._failure
        if self.closed:
            raise RuntimeError("PD multiplex session is closed")
        if key in self._routes:
            raise ValueError("PD handoff route is already open")
        self._routes[key] = asyncio.Queue(maxsize=self._queue_size)

    async def send(self, message: ControlMessage) -> None:
        key = message_handoff_key(message)
        if key is None or key not in self._routes:
            raise ValueError("PD multiplex send requires an open handoff route")
        await self._session.send(message)

    async def receive(self, key) -> ControlMessage:
        try:
            queue = self._routes[key]
        except KeyError as exc:
            raise ValueError("PD multiplex receive requires an open handoff route") from exc
        item = await queue.get()
        if isinstance(item, BaseException):
            raise RuntimeError("PD multiplex control session failed") from item
        return item

    def close_route(self, key) -> None:
        if self._routes.pop(key, None) is not None:
            self._closed_keys.append(key)

    async def close(self) -> None:
        await self._session.close()
        task = self._reader_task
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._reader_task = None

    async def _reader(self) -> None:
        try:
            while True:
                message = await self._session.receive()
                key = message_handoff_key(message)
                if key is None:
                    raise ValueError("uncorrelated message on multiplexed PD session")
                queue = self._routes.get(key)
                if queue is None:
                    if key in self._closed_keys:
                        # A bounded tombstone set absorbs a reply already on the
                        # wire when cancellation closed the local route.
                        continue
                    raise ValueError("unknown handoff correlation on PD session")
                try:
                    queue.put_nowait(message)
                except asyncio.QueueFull as exc:
                    raise RuntimeError("PD handoff reply queue exceeded its bound") from exc
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            self._failure = exc
            for queue in tuple(self._routes.values()):
                with contextlib.suppress(asyncio.QueueFull):
                    queue.put_nowait(exc)
            await self._session.close()


@dataclass(frozen=True)
class PeerSessionKey:
    node_id: str
    endpoint_generation: int
    registry_fingerprint: str


@dataclass(frozen=True)
class PeerSessionHandle:
    key: PeerSessionKey
    session: MultiplexedPDControlSession
    registry: RegistryAdvertisement


class PeerSessionPool:
    """Single-entry Phase E pool with a stable Phase F expansion seam."""

    def __init__(self) -> None:
        self._handle: PeerSessionHandle | None = None

    def lookup(self, node_id: str, endpoint_generation: int) -> PeerSessionHandle | None:
        handle = self._handle
        if (
            handle is None
            or handle.session.closed
            or handle.key.node_id != node_id
            or handle.key.endpoint_generation != endpoint_generation
        ):
            return None
        return handle

    async def replace(self, handle: PeerSessionHandle) -> None:
        previous = self._handle
        self._handle = handle
        if previous is not None and previous.session is not handle.session:
            await previous.session.close()

    async def close(self) -> None:
        handle = self._handle
        self._handle = None
        if handle is not None:
            await handle.session.close()


class PDControlAcceptor:
    """Cancellable listener used by an independently started external D node."""

    def __init__(self, config: PDConfig) -> None:
        if not config.enabled or config.role is not PDRole.DECODE:
            raise ValueError("PDControlAcceptor requires an enabled Decode config")
        self.config = config
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind((config.control_host, config.control_port))
        self._listener.listen(4)
        self._listener.setblocking(False)
        self._closed = False

    async def accept(
        self,
        capabilities: PDCapabilities,
        *,
        expected_peer_node_id: str,
    ) -> PDControlSession:
        if self._closed:
            raise RuntimeError("PD control acceptor is closed")
        loop = asyncio.get_running_loop()
        sock, address = await loop.sock_accept(self._listener)
        try:
            sock.setblocking(True)
            sock.settimeout(self.config.request_timeout_seconds)
            return await _open_pd_socket(
                sock,
                self.config,
                capabilities,
                expected_peer_node_id=expected_peer_node_id,
                expected_peer_role=PDRole.PREFILL,
            )
        except BaseException:
            sock.close()
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._listener.close()


async def connect_control_session(
    config: PDConfig,
    capabilities: PDCapabilities,
    *,
    peer_node_id: str,
    peer_host: str,
    peer_port: int,
) -> PDControlSession:
    """Lazily connect a P node to the Router-selected D control endpoint."""
    if not config.enabled or config.role is not PDRole.PREFILL:
        raise ValueError("PD control connect requires an enabled Prefill config")
    sock = await asyncio.to_thread(
        _connect_endpoint,
        peer_host,
        peer_port,
        config.connect_timeout_seconds,
    )
    sock.settimeout(config.request_timeout_seconds)
    return await _open_pd_socket(
        sock,
        config,
        capabilities,
        expected_peer_node_id=peer_node_id,
        expected_peer_role=PDRole.DECODE,
    )


async def _open_pd_socket(
    sock: socket.socket,
    config: PDConfig,
    capabilities: PDCapabilities,
    *,
    expected_peer_node_id: str,
    expected_peer_role: PDRole,
) -> PDControlSession:
    channel = FramedChannel(
        sock,
        local_node_id=config.node_id,
        peer_node_id=expected_peer_node_id,
    )
    hello = make_hello(
        node_id=config.node_id,
        role=config.role,
        run_id=config.run_id,
        capabilities=capabilities,
        endpoint_generation=config.generation,
        control_incarnation=config.control_incarnation,
    )
    try:
        peer = await asyncio.to_thread(
            exchange_and_validate_hello,
            channel,
            hello,
            expected_peer_node_id=expected_peer_node_id,
            expected_peer_role=expected_peer_role,
        )
    except BaseException:
        sock.close()
        raise
    # Idle serving periods are not request failures. Native transfers carry
    # their own bounded deadlines; closing this socket unblocks a pending read.
    sock.settimeout(None)
    return PDControlSession(sock, channel, peer)


def _connect_endpoint(host: str, port: int, timeout_seconds: float) -> socket.socket:
    deadline = time.monotonic() + timeout_seconds
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        try:
            remaining = max(0.1, deadline - time.monotonic())
            return socket.create_connection(
                (host, port),
                timeout=min(1.0, remaining),
            )
        except OSError as exc:
            last_error = exc
            time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))
    raise TimeoutError("PD fixed peer did not accept the control connection") from last_error


def validate_rank_registrations(
    registrations: Iterable,
    *,
    expected_count: int,
    expected_components: tuple[str, ...],
) -> None:
    """Shared structural validation used before envelopes reach a worker."""
    ranks = tuple(registrations)
    if len(ranks) != expected_count:
        raise ValueError("PD registry owner count differs from its topology")
    for rank_id, registration in enumerate(ranks):
        if registration.rank_id != rank_id:
            raise ValueError("PD registry owners must be rank ordered")
        if registration.owner_generation < 1 or registration.endpoint_generation < 1:
            raise ValueError("PD registry owner generations must be positive")
        if not registration.worker_id:
            raise ValueError("PD registry worker identity must not be empty")
        regions = {region.component_id: region for region in registration.regions}
        if set(regions) != set(expected_components):
            raise ValueError("PD registry owner components differ from capabilities")
        if len(regions) != len(registration.regions):
            raise ValueError("PD registry owner contains duplicate physical regions")
        for region in regions.values():
            if region.lease < 1 or region.extent < 1 or not region.provider_envelope:
                raise ValueError("PD registry region has an invalid lease or envelope")
