# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A stand-in serving replica: the OpenAI surface the router uses, and no model.

Run as a script, it is the second replica in a one-card manual check; imported,
it backs the router's end-to-end test. Every answer names the replica that
produced it, which is what lets a caller see where a session actually landed.

It can also fail, in each of the ways a real replica fails, because the router's
recovery paths key on the differences between them:

* ``POST /toggle`` answers 503 while staying up -- a worker died under a live
  HTTP server. The socket stays open, so this is the one failure that does *not*
  look like an exit.
* ``POST /stall`` stops answering ``/health`` at all. A probe times out instead
  of being refused, which is a hung host rather than a dead one.
* ``POST /exit`` ends the process. That is a crash.
* ``stream_crash_after`` / ``stream_stall_after`` break a response that is
  already streaming, after the status line and headers have gone out.
* ``--ready-after`` and ``--exit-at-startup`` make a *relaunched* replica slow or
  doomed, which is what a replacement policy has to survive.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

STREAM_PIECES = ("hello", " there")

# Long enough that a caller reads it as "no answer", short enough to leave no
# thread behind for the rest of a test session.
STALL_SECONDS = 3600.0

# The response to /exit has to reach the caller before the process goes, or a
# test cannot tell "it died" from "it never got the request".
EXIT_DELAY_SECONDS = 0.05


def build_app(name: str, *, ready_after: float = 0.0) -> FastAPI:
    app = FastAPI()
    started = time.monotonic()
    state = {"ready": True, "stalled": False}

    @app.get("/health")
    async def health() -> JSONResponse:
        if state["stalled"]:
            await asyncio.sleep(STALL_SECONDS)
        if not state["ready"] or time.monotonic() - started < ready_after:
            return JSONResponse({"status": "not_ready"}, status_code=503)
        return JSONResponse({"status": "ok"})

    @app.post("/toggle")
    async def toggle() -> JSONResponse:
        state["ready"] = not state["ready"]
        return JSONResponse({"ready": state["ready"]})

    @app.post("/stall")
    async def stall() -> JSONResponse:
        state["stalled"] = True
        return JSONResponse({"stalled": True})

    @app.post("/exit")
    async def exit_now(request: Request) -> JSONResponse:
        payload = await _body(request)
        code = int(payload.get("code", 1))
        # os._exit, not sys.exit: a crash runs no shutdown hook, and a stub that
        # unwound cleanly would exercise the one path a crash never takes.
        asyncio.get_running_loop().call_later(EXIT_DELAY_SECONDS, os._exit, code)
        return JSONResponse({"exiting": code})

    @app.get("/v1/models")
    async def models() -> JSONResponse:
        return JSONResponse({"object": "list", "data": [{"id": name, "object": "model"}]})

    async def handle(request: Request, *, chat: bool):
        payload = await _body(request)
        text = f"[{name}]"

        if payload.get("stream"):
            # Both counts are "after how many chunks", so 0 breaks the stream
            # before anything is sent and 1 breaks it with the client already
            # reading -- the case the relay cannot answer with a status code.
            crash_after = payload.get("stream_crash_after")
            stall_after = payload.get("stream_stall_after")

            async def events():
                for sent, piece in enumerate(STREAM_PIECES):
                    if crash_after is not None and sent >= int(crash_after):
                        os._exit(1)
                    if stall_after is not None and sent >= int(stall_after):
                        # Neither more data nor an end: the stream the client
                        # waits on forever unless someone else bounds it.
                        await asyncio.sleep(STALL_SECONDS)
                    choice = (
                        {"delta": {"role": "assistant", "content": piece}}
                        if chat
                        else {"text": piece}
                    )
                    chunk = {"id": "stub", "served_by": name, "choices": [choice]}
                    yield f"data: {json.dumps(chunk)}\n\n".encode()
                yield b"data: [DONE]\n\n"

            return StreamingResponse(events(), media_type="text/event-stream")

        choice = (
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
            if chat
            else {"index": 0, "text": text, "finish_reason": "stop"}
        )
        return JSONResponse({
            "id": "stub",
            "object": "chat.completion" if chat else "text_completion",
            "created": int(time.time()),
            "model": payload.get("model") or name,
            "served_by": name,
            "choices": [choice],
        })

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await handle(request, chat=False)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await handle(request, chat=True)

    return app


async def _body(request: Request) -> dict:
    """The JSON body, or an empty one. /exit is called with no body at all."""
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001 - an unreadable body is an empty one here
        return {}
    return payload if isinstance(payload, dict) else {}


def main() -> int:
    import uvicorn

    parser = argparse.ArgumentParser(description="Run a stand-in serving replica.")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    # A replica that answers 503 for a while, like one still loading its model.
    parser.add_argument("--ready-after", type=float, default=0.0)
    # A replica that never comes up at all. Relaunching this one is a crash loop,
    # which is what the restart cap exists to stop.
    parser.add_argument("--exit-at-startup", type=int, default=None)
    # A launch wrapper hands its payload through as a trailing argument. The
    # stub ignores it, which is what lets it stand in for a wrapped launch.
    parser.add_argument("ignored", nargs="*", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.exit_at_startup is not None:
        return args.exit_at_startup
    uvicorn.run(
        build_app(args.name, ready_after=args.ready_after),
        host=args.host, port=args.port, log_level="warning",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
