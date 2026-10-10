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
``POST /toggle`` flips readiness so failover can be exercised without killing
the process.
"""

from __future__ import annotations

import argparse
import json
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

STREAM_PIECES = ("hello", " there")


def build_app(name: str) -> FastAPI:
    app = FastAPI()
    state = {"ready": True}

    @app.get("/health")
    async def health() -> JSONResponse:
        if not state["ready"]:
            return JSONResponse({"status": "not_ready"}, status_code=503)
        return JSONResponse({"status": "ok"})

    @app.post("/toggle")
    async def toggle() -> JSONResponse:
        state["ready"] = not state["ready"]
        return JSONResponse({"ready": state["ready"]})

    @app.get("/v1/models")
    async def models() -> JSONResponse:
        return JSONResponse({"object": "list", "data": [{"id": name, "object": "model"}]})

    async def handle(request: Request, *, chat: bool):
        payload = await request.json()
        text = f"[{name}]"

        if payload.get("stream"):
            async def events():
                for piece in STREAM_PIECES:
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


def main() -> int:
    import uvicorn

    parser = argparse.ArgumentParser(description="Run a stand-in serving replica.")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    # A launch wrapper hands its payload through as a trailing argument. The
    # stub ignores it, which is what lets it stand in for a wrapped launch.
    parser.add_argument("ignored", nargs="*", help=argparse.SUPPRESS)
    args = parser.parse_args()
    uvicorn.run(build_app(args.name), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
