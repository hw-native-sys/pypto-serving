# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Verbatim replica forwarding over the Router's shared HTTP lifecycle."""

import asyncio
import logging

from pypto_serving.tools.observe import record
from .http import (Retirement, RouterError, close_requests, error_response, failure,
                   request_headers, response_headers, resolve_session_id, send)
from .routing import NoReplicaAvailable

logger = logging.getLogger(__name__)


class ReplicaProxy:
    def __init__(self, registry, client, start_timeout):
        self.registry, self.client, self.start_timeout = registry, client, start_timeout
        self.active = {}
        self.closing = False

    async def close(self):
        self.closing = True
        await close_requests(self.active, self.start_timeout)

    async def forward(self, request, path):
        from fastapi.responses import StreamingResponse

        owner = asyncio.current_task()
        self.active[owner] = None
        session_id = resolve_session_id(request.headers, b"")
        retirement = None
        try:
            if self.closing:
                raise RouterError(503, "router_closing", "Router is closing")
            body = await request.body()
            session_id = resolve_session_id(request.headers, body)
            try:
                decision = self.registry.select(session_id)
            except NoReplicaAvailable as exc:
                raise RouterError(503, "no_available_backend", str(exc)) from exc
            replica = decision.replica
            record(logger, "route", session=session_id, node=replica.name, policy="least_loaded_affinity",
                   reason=decision.reason, affinity=decision.affinity_hit, path=path)
            retirement = Retirement(self.active, owner, self.registry.acquire(replica.name))
            upstream = retirement.response = await send(self.client, self.registry, replica, path,
                headers=request_headers(request.headers), content=body)
            async def relay():
                try:
                    retirement.begin()
                    async for chunk in upstream.aiter_raw():
                        yield chunk
                finally:
                    await retirement.wait()
            retirement.arm(self.start_timeout)
            return StreamingResponse(relay(), status_code=upstream.status_code,
                headers=response_headers(upstream, session_id,
                    source="upstream" if upstream.status_code >= 400 else None))
        except BaseException as exc:
            if retirement is not None:
                await retirement.wait()
            if not isinstance(exc, Exception):
                raise
            return error_response(failure(exc), session_id)
        finally:
            if self.active.get(owner) is None:
                self.active.pop(owner, None)
