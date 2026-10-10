# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Streaming reverse proxy from the router to one serving replica.

Request bytes are forwarded verbatim: the router never parses or rebuilds the
OpenAI payload, so it needs no tokenizer, chat template, or model. The only
field it reads is the session id, and a ``session_id`` left in the body is
harmless downstream because the serving request models ignore unknown fields.
"""

from __future__ import annotations

import json
import logging
import re
from typing import AsyncIterator

from pypto_serving.router.routing import NoReplicaAvailable, ReplicaRegistry

logger = logging.getLogger(__name__)

SESSION_HEADER = "x-session-id"

# A session id is echoed back in a response header and used as a dictionary key,
# so it must be header-safe: no CR/LF (which Uvicorn's h11 path rejects) and no
# non-latin-1 byte (which raises when Starlette builds the response). Anchored so
# a trailing newline cannot slip through.
_SESSION_ID_RE = re.compile(r"\A[A-Za-z0-9._:-]{1,128}\Z")

# Headers that describe one hop and must not be relayed to the next.
_HOP_BY_HOP = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
})
# The router re-frames the body, so the upstream framing headers no longer
# describe what the client receives.
_DROP_REQUEST = _HOP_BY_HOP | {"host", "content-length"}
_DROP_RESPONSE = _HOP_BY_HOP | {"content-length"}


def resolve_session_id(headers, body: bytes) -> str:
    """Header, then a top-level ``session_id`` in the JSON body, then a new id.

    A client-supplied id is rejected unless it is header-safe and bounded; an
    unusable one is replaced rather than refused, because the id only decides
    routing and a bad one should not fail the request. An absent or unparsable
    body is likewise not an error: the router does not validate the payload,
    the replica does.
    """
    from pypto_serving.router.routing import new_session_id

    supplied = headers.get(SESSION_HEADER)
    if supplied is None:
        try:
            payload = json.loads(body) if body else None
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = None
        if isinstance(payload, dict):
            candidate = payload.get("session_id")
            supplied = candidate if isinstance(candidate, str) else None

    if supplied and _SESSION_ID_RE.match(supplied):
        return supplied
    if supplied:
        logger.warning("ignoring an unusable session id (%d chars); minting a new one", len(supplied))
    return new_session_id()


def _filter_headers(headers, drop: frozenset[str]) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in drop}


class ReplicaProxy:
    """Routes one request to a replica and relays the response back."""

    def __init__(self, registry: ReplicaRegistry, client) -> None:
        self._registry = registry
        self._client = client

    async def forward(self, request, path: str):
        """Proxy one generation request, choosing the replica by session."""
        from fastapi.responses import JSONResponse, StreamingResponse

        body = await request.body()
        session_id = resolve_session_id(request.headers, body)

        try:
            decision = self._registry.select(session_id)
        except NoReplicaAvailable:
            # The session id is returned on every response, this one included:
            # clients rely on it for affinity and may retry with the same id.
            return JSONResponse(
                status_code=503,
                content={"object": "error", "message": "no serving replica is currently routable"},
                headers={SESSION_HEADER: session_id},
            )

        replica = decision.replica
        logger.info("routing session %s to %s (affinity=%s) %s",
                    session_id, replica.name, decision.affinity_hit, path)

        self._registry.acquire(replica.name)
        released = False

        def release_once() -> None:
            nonlocal released
            if not released:
                released = True
                self._registry.release(replica.name)

        try:
            upstream = await self._client.send(
                self._client.build_request(
                    "POST", f"{replica.base_url}{path}", content=body,
                    headers=_filter_headers(request.headers, _DROP_REQUEST),
                ),
                stream=True,
            )
        except Exception as exc:  # noqa: BLE001 - any transport failure is a dead replica
            release_once()
            self._registry.set_ready(replica.name, False)
            logger.warning("replica %s refused the request: %s", replica.name, exc)
            return JSONResponse(
                status_code=502,
                content={"object": "error", "message": f"replica {replica.name} is unreachable"},
                headers={SESSION_HEADER: session_id},
            )

        headers = _filter_headers(upstream.headers, _DROP_RESPONSE)
        # Headers precede the body, so this is the only mechanism that returns
        # the session id on a streaming response without rewriting SSE.
        headers[SESSION_HEADER] = session_id
        return StreamingResponse(
            self._relay(upstream, replica.name, release_once),
            status_code=upstream.status_code,
            headers=headers,
        )

    async def _relay(self, upstream, replica_name: str, release_once) -> AsyncIterator[bytes]:
        """Relay the upstream body, closing it however this generator ends.

        Closing the upstream is what makes the replica see a client disconnect,
        which is serving's only cancellation path: there is no abort route, and
        an unclosed stream would keep generating and pinning KV blocks.
        """
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            try:
                await upstream.aclose()
            except Exception:  # noqa: BLE001 - teardown must not mask the original exit
                logger.debug("closing upstream to %s failed", replica_name, exc_info=True)
            release_once()
