# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded HTTP transport; generation responses remain standard JSON/SSE."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
import socket
import urllib.error
import urllib.request

import anyio

from pypto_serving.serving.pd.generation import MAX_NODE_BODY_BYTES
from pypto_serving.serving.pd.http_api import decode_json, encode_json


@dataclass(frozen=True)
class PublicResult:
    body: bytes
    headers: dict[str, str]


class GenerationResponse:
    """Own a stdlib response and drain its worker on cancellation."""

    def __init__(self, response):
        self.response = response
        self.status = response.status
        self.headers = {name.lower(): value for name, value in response.headers.items()}
        self._closed = False

    async def _read(self, method, *args):
        task = asyncio.create_task(asyncio.to_thread(method, *args))
        try:
            return await asyncio.shield(task)
        except BaseException:
            with anyio.CancelScope(shield=True):
                await self.aclose()
                await asyncio.gather(task, return_exceptions=True)
            raise

    async def json(self) -> dict:
        wire = await self._read(self.response.read, MAX_NODE_BODY_BYTES + 1)
        value = decode_json(wire, dict)
        return value

    async def result(self) -> PublicResult:
        wire = await self._read(self.response.read, MAX_NODE_BODY_BYTES + 1)
        decode_json(wire, dict)
        headers = {name: value for name, value in self.headers.items()
                   if name.lower() == "x-pypto-token-ids-sha256"}
        return PublicResult(wire, headers)

    async def frames(self):
        frame = bytearray()
        while True:
            line = await self._read(self.response.readline, MAX_NODE_BODY_BYTES + 1)
            if not line:
                raise EOFError("Decode closed SSE without [DONE]")
            frame.extend(line)
            if len(frame) > MAX_NODE_BODY_BYTES:
                raise ValueError("Decode SSE frame exceeds the bounded size")
            if line in (b"\n", b"\r\n"):
                yield bytes(frame)
                frame.clear()

    async def aclose(self):
        if not self._closed:
            self._closed = True
            with anyio.CancelScope(shield=True):
                await asyncio.to_thread(NodeClient._close_response, self.response)


class NodeClient:
    def __init__(self, base_url: str, timeout_seconds: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def get(self, path: str, response_type):
        wire = await asyncio.to_thread(self._request, "GET", path, None)
        return decode_json(wire, response_type)

    async def post(self, path: str, payload, response_type=None):
        wire = await asyncio.to_thread(self._request, "POST", path, encode_json(payload))
        return decode_json(wire, response_type) if response_type is not None else None

    async def open_generation(self, path: str, body: bytes) -> GenerationResponse:
        task = asyncio.create_task(asyncio.to_thread(self._open, "POST", path, body, True))
        try:
            return GenerationResponse(await asyncio.shield(task))
        except BaseException:
            # Cancellation cannot abandon a thread that later returns a socket.
            with anyio.CancelScope(shield=True):
                result = await asyncio.gather(task, return_exceptions=True)
                if not isinstance(result[0], BaseException):
                    await asyncio.to_thread(self._close_response, result[0])
            raise

    @staticmethod
    def _close_response(response) -> None:
        raw = getattr(getattr(response, "fp", None), "raw", None)
        stream_socket = getattr(raw, "_sock", None)
        if stream_socket is not None:
            with suppress(OSError):
                stream_socket.shutdown(socket.SHUT_RDWR)
        response.close()

    def _request(self, method: str, path: str, body: bytes | None) -> bytes:
        response = self._open(method, path, body)
        try:
            wire = response.read(MAX_NODE_BODY_BYTES + 1)
            decode_json(wire, object)
            return wire
        finally:
            response.close()

    def _open(self, method: str, path: str, body: bytes | None, generation=False):
        request = urllib.request.Request(
            self.base_url + path, data=body, method=method,
            headers={"content-type": "application/json"},
        )
        try:
            return urllib.request.urlopen(request, timeout=self.timeout_seconds)
        except urllib.error.HTTPError as exc:
            if generation:
                return exc
            try:
                detail = exc.read(4096).decode(errors="replace")
            finally:
                exc.close()
            if exc.code in (400, 422):
                raise ValueError(f"node {path} rejected request: {detail}") from exc
            raise RuntimeError(f"node {path} failed with HTTP {exc.code}: {detail}") from exc
