# Copyright (c) PyPTO Contributors.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""Bounded standard-library HTTP client for Router-to-node control calls."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
import urllib.error
import urllib.request

from pypto_serving.serving.pd.http_api import (
    MAX_INTERNAL_BODY_BYTES,
    DecodeStreamFrame,
    decode_json,
    encode_json,
)


class NodeClient:
    def __init__(self, base_url: str, timeout_seconds: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def get(self, path: str, response_type):
        wire = await asyncio.to_thread(self._request, "GET", path, None)
        return decode_json(wire, response_type)

    async def post(self, path: str, payload, response_type=None):
        wire = await asyncio.to_thread(
            self._request,
            "POST",
            path,
            encode_json(payload),
        )
        if response_type is None:
            return None
        return decode_json(wire, response_type)

    async def stream_decode(self, path: str, payload) -> AsyncGenerator[DecodeStreamFrame, None]:
        response = await asyncio.to_thread(
            self._open,
            "POST",
            path,
            encode_json(payload),
        )
        try:
            while True:
                line = await asyncio.to_thread(response.readline, MAX_INTERNAL_BODY_BYTES + 1)
                if not line:
                    raise EOFError("Decode node closed its output stream without a terminal frame")
                if len(line) > MAX_INTERNAL_BODY_BYTES:
                    raise ValueError("Decode stream frame exceeds the bounded size")
                yield decode_json(line.rstrip(b"\n"), DecodeStreamFrame)
        finally:
            response.close()

    def _request(self, method: str, path: str, body: bytes | None) -> bytes:
        response = self._open(method, path, body)
        try:
            wire = response.read(MAX_INTERNAL_BODY_BYTES + 1)
        finally:
            response.close()
        if not wire or len(wire) > MAX_INTERNAL_BODY_BYTES:
            raise ValueError("node returned an invalid bounded response")
        return wire

    def _open(self, method: str, path: str, body: bytes | None):
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            method=method,
            headers={
                "content-type": "application/json",
            },
        )
        try:
            return urllib.request.urlopen(request, timeout=self.timeout_seconds)
        except urllib.error.HTTPError as exc:
            detail = exc.read(4096).decode(errors="replace")
            raise RuntimeError(f"node {path} failed with HTTP {exc.code}: {detail}") from exc
