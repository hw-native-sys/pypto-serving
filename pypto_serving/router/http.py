# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared HTTP framing, failures and retirement; no model or cache ownership."""

import asyncio
import json
import logging
import re

logger = logging.getLogger(__name__)
SESSION_HEADER = "x-session-id"
ERROR_SOURCE_HEADER = "x-pypto-error-source"
_SESSION_ID_RE = re.compile(r"\A[A-Za-z0-9._:-]{1,128}\Z")
_HOP_BY_HOP = frozenset({"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
                         "te", "trailer", "trailers", "transfer-encoding", "upgrade"})


def resolve_session_id(headers, body):
    from .routing import new_session_id

    supplied = headers.get(SESSION_HEADER)
    if supplied is None:
        try:
            payload = json.loads(body) if body else None
        except (ValueError, UnicodeDecodeError):
            payload = None
        candidate = payload.get("session_id") if isinstance(payload, dict) else None
        supplied = candidate if isinstance(candidate, str) else None
    if supplied and _SESSION_ID_RE.match(supplied):
        return supplied
    return new_session_id()


def filter_headers(headers, drop):
    connection = next((v for k, v in headers.items() if k.lower() == "connection"), "")
    blocked = _HOP_BY_HOP | set(drop) | {name.strip().lower() for name in connection.split(",")}
    return {k: v for k, v in headers.items() if k.lower() not in blocked}


def request_headers(headers, *, drop=(), extra=None):
    return {**filter_headers(headers, {"host", "content-length", ERROR_SOURCE_HEADER, *drop}),
            **(extra or {})}


def response_headers(response, session_id, *, decoded=False, source=None):
    blocked = {"content-length", ERROR_SOURCE_HEADER}
    if decoded:
        blocked.add("content-encoding")
    result = {**filter_headers(response.headers, blocked), SESSION_HEADER: session_id}
    if source is not None:
        result[ERROR_SOURCE_HEADER] = source
    return result


def is_router_side_failure(exc):
    try:
        import httpx
    except ImportError:
        return False
    return isinstance(exc, httpx.PoolTimeout)


class RouterError(Exception):
    def __init__(self, status, code, message, upstream=None):
        super().__init__(message)
        self.status, self.code, self.upstream = status, code, upstream

    @property
    def source(self):
        return "upstream" if self.upstream is not None else "router"


def failure(exc, *, timeout_code="upstream_timeout", protocol_code="upstream_protocol_error"):
    import httpx

    if isinstance(exc, RouterError):
        return exc
    if is_router_side_failure(exc):
        return RouterError(503, "http_pool_exhausted", "Router HTTP connection pool is busy")
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException)):
        return RouterError(503, timeout_code, str(exc) or "HTTP operation timed out")
    if isinstance(exc, (OSError, httpx.HTTPError)):
        return RouterError(502, "upstream_connection_error", f"Upstream unreachable: {exc}")
    if isinstance(exc, ValueError):
        return RouterError(400, "invalid_request", str(exc))
    return RouterError(503, protocol_code, str(exc))


def error_response(error, session_id):
    from fastapi.responses import JSONResponse, Response

    headers = {SESSION_HEADER: session_id, ERROR_SOURCE_HEADER: error.source}
    if error.upstream is not None:
        return Response(error.upstream.content, status_code=error.status,
            headers=response_headers(error.upstream, session_id, decoded=True, source=error.source))
    if error.code == "capacity_exhausted":
        headers["Retry-After"] = "1"
    return JSONResponse(dict(object="error", code=error.code, message=str(error)),
                        status_code=error.status, headers=headers)


async def send(client, registry, replica, path, *, headers, stream=True, timeout=None, **body):
    import httpx

    options = dict(headers=headers, **body)
    if timeout is not None:
        options["timeout"] = timeout
    try:
        request = client.build_request("POST", replica.base_url + path, **options)
        return await client.send(request, stream=stream)
    except Exception as exc:
        if isinstance(exc, (OSError, httpx.TransportError)) and not is_router_side_failure(exc):
            registry.set_ready(replica.name, False)
        raise


class Retirement:
    """Own one HTTP response close and one count release, including unstarted bodies."""

    def __init__(self, active, owner, release, before_close=None):
        self.active, self.owner, self.release = active, owner, release
        self.before_close = before_close
        self.response = self.task = self.timer = None
        active[owner] = self.retire

    def arm(self, timeout):
        self.timer = asyncio.get_running_loop().call_later(timeout, self.retire)

    def begin(self):
        if self.timer is not None:
            self.timer.cancel()
        if self.task is not None:
            raise RouterError(503, "response_retired", "HTTP response retired before streaming began")

    def retire(self):
        if self.timer is not None:
            self.timer.cancel()
        if self.task is None:
            self.task = asyncio.create_task(self._close())
        return self.task

    async def wait(self):
        task = self.retire()
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await asyncio.shield(task)
            raise

    async def _close(self):
        try:
            if self.before_close is not None:
                await self.before_close()
        except Exception:
            logger.exception("HTTP coordination cleanup failed")
        finally:
            try:
                if self.response is not None:
                    await self.response.aclose()
            except Exception:
                logger.exception("closing upstream HTTP response failed")
            finally:
                self.release()
                self.active.pop(self.owner, None)


async def close_requests(active, timeout):
    snapshot = tuple(active.items())
    pending = tuple(task for task, _ in snapshot if task is not asyncio.current_task() and not task.done())
    for task in pending:
        task.cancel()
    retiring = tuple(retire() for _, retire in snapshot if retire is not None)
    if pending or retiring:
        await asyncio.wait_for(asyncio.gather(*pending, *retiring, return_exceptions=True), timeout + 5)
