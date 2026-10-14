# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Peer-scoped, sender-before-receiver transfer retirement."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import logging

from .config import PDRole
from .protocol import DrainPeer, SenderReleased, SenderReleaseAck

logger = logging.getLogger(__name__)


class TransferLifecycle:
    """Keep the control reader alive until native release has been acknowledged.

    EOF is not release proof. A failed release remains failed and keeps the
    worker's owner handles available for external recovery.
    """

    def __init__(
        self, role: PDRole, *, idle: Callable[[], bool],
        release: Callable[[], Awaitable[None]], timeout: float,
    ) -> None:
        self.role, self.idle, self.release, self.timeout = role, idle, release, timeout
        self.session = None
        self.draining = False
        self._task = None
        self._sender_released = asyncio.Event()
        self._sender_ack = asyncio.Event()

    def bind(self, session):
        if self.draining:
            raise RuntimeError("PD transfer lifecycle is draining")
        self.session = session

    async def receive(self, message):
        if isinstance(message, DrainPeer):
            self.begin()
        elif isinstance(message, SenderReleased) and self.role is PDRole.DECODE:
            self._sender_released.set()
            self.begin()
            if self._task.done() and self._task.exception() is None:
                await self.session.send(SenderReleaseAck())
        elif isinstance(message, SenderReleaseAck) and self.role is PDRole.PREFILL:
            self._sender_ack.set()
        else:
            raise ValueError("unexpected PD lifecycle message")

    def begin(self):
        self.draining = True
        if self._task is None:
            self._task = asyncio.create_task(self._bounded_retire())
            # Observe asynchronous failures even when only the peer requested drain.
            self._task.add_done_callback(lambda task: None if task.cancelled() else task.exception())
        return self._task

    async def close(self):
        await asyncio.shield(self.begin())

    async def _bounded_retire(self):
        try:
            await asyncio.wait_for(self._retire(), self.timeout)
        except Exception:
            logger.exception("PD %s transfer retirement failed; owner handles retained", self.role.value)
            raise

    async def _retire(self):
        if self.session is not None and not self._sender_released.is_set():
            await self.session.send(DrainPeer())
        while not self.idle():
            await asyncio.sleep(0.01)
        if self.role is PDRole.DECODE and self.session is not None:
            await self._sender_released.wait()
        await self.release()
        if self.role is PDRole.DECODE and self.session is not None:
            # Both engines must remain alive until both sides unregister.
            await self.session.send(SenderReleaseAck())
        if self.role is PDRole.PREFILL and self.session is not None:
            await self.session.send(SenderReleased())
            await self._sender_ack.wait()
