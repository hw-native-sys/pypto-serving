# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU-only external routing. Bulk model data never traverses this process."""

import argparse
import asyncio
from contextlib import asynccontextmanager
import json
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
import msgspec

from .config import Config, log_context
from .protocol import Binding, MAX_FRAME, Node, decode
from .routing import Directory

logger = logging.getLogger("pypto_serving.pd.router")


class Router:
    def __init__(self, config, transport=None):
        self.config = config
        self.directory = Directory(config.prefill + config.decode)
        self.client = httpx.AsyncClient(transport=transport, timeout=config.request_timeout,
                                       limits=httpx.Limits(max_connections=config.capacity * 8))
        self.requests = asyncio.Semaphore(config.capacity * len(config.decode))
        self.handoffs = asyncio.Semaphore(config.capacity * len(config.prefill))

    async def refresh(self):
        async def inspect(endpoint):
            try:
                response = await self.client.get(endpoint.url + "/internal/pd/descriptor", timeout=5)
                response.raise_for_status()
                node = decode(response.json(), Node)
                role = "prefill" if endpoint in self.config.prefill else "decode"
                if node.node_id != endpoint.node_id or node.role != role:
                    raise ValueError("descriptor differs from configured role/endpoint")
                self.directory.update(node)
            except (httpx.HTTPError, ValueError):
                self.directory.nodes.pop(endpoint.node_id, None)
        await asyncio.gather(*(inspect(e) for e in self.directory.endpoints.values()))

    async def abort(self, binding, endpoints):
        await asyncio.gather(*(self.client.post(e.url + "/internal/pd/abort",
            json=msgspec.to_builtins(binding), timeout=5) for e in endpoints), return_exceptions=True)

    async def generate(self, request: Request):
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(400, "request must be a JSON object")
        await self.requests.acquire()
        handoff_held = False
        p_task = d_task = response = binding = None
        endpoints = ()
        finished = False
        async def cleanup():
            nonlocal handoff_held
            if binding is not None and not finished:
                # Abort wakes D's pre-header rendezvous before HTTP tasks are joined.
                await self.abort(binding, endpoints)
            tasks = tuple(t for t in (p_task, d_task) if t is not None)
            if tasks:
                try:
                    await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True),
                                           self.config.handoff_timeout)
                except asyncio.TimeoutError:
                    pass  # Node-side leases remain gated until positive native proof.
            if response is not None:
                await response.aclose()
            if handoff_held:
                handoff_held = False
                self.handoffs.release()
            self.requests.release()

        async def ready():
            result = await p_task
            if result.status_code != 200:
                raise HTTPException(result.status_code, result.text[:1000])
            if result.json() != {"request": binding.request, "status": "ready"}:
                raise RuntimeError(f"Prefill rejected handoff: {result.text[:1000]}")

        try:
            await self.handoffs.acquire()
            handoff_held = True
            await self.refresh()
            p, d = self.directory.select(payload.get("model") or None)
            binding = Binding.create(p, d, payload)
            logger.info("route request=%s prefill=%s decode=%s", binding.request, p.node_id, d.node_id)
            endpoints = tuple(self.directory.endpoints[n.node_id] for n in (p, d))
            def headers(peer):
                return {"x-pd-context": msgspec.json.encode(dict(binding=binding, peer=peer)).decode()}
            p_task = asyncio.create_task(self.client.post(endpoints[0].url + request.url.path,
                json=payload, headers=headers(d)))
            d_task = asyncio.create_task(self.client.send(self.client.build_request(
                "POST", endpoints[1].url + request.url.path, json=payload, headers=headers(p)), stream=True))
            done, _ = await asyncio.wait((p_task, d_task), return_when=asyncio.FIRST_COMPLETED,
                                         timeout=self.config.handoff_timeout)
            if not done:
                raise RuntimeError("handoff header deadline exceeded")
            if p_task in done:
                await ready()
            response = await asyncio.wait_for(asyncio.shield(d_task), self.config.handoff_timeout)
            if response.status_code != 200:
                body = await response.aread()
                raise HTTPException(response.status_code, body.decode()[:1000])
            def release_handoff(done):
                nonlocal handoff_held
                if handoff_held:
                    handoff_held = False
                    self.handoffs.release()
            p_task.add_done_callback(release_handoff)
            if not payload.get("stream", False):
                body = await response.aread()
                await ready()
                finished = True
                result = JSONResponse(json.loads(body))
                await cleanup()
                return result

            async def stream():
                nonlocal finished
                frame = []
                size = 0
                try:
                    async for line in response.aiter_lines():
                        size += len(line)
                        if size > MAX_FRAME:
                            raise ValueError("SSE frame exceeds control limit")
                        if line:
                            frame.append(line)
                            continue
                        text = "\n".join(frame)
                        frame, size = [], 0
                        if text.startswith("data: "):
                            data = text[6:]
                            terminal = data == "[DONE]"
                            value = None if terminal else json.loads(data)
                            if terminal or (value and any(c.get("finish_reason") for c in value.get("choices", []))):
                                await ready()
                            if terminal:
                                finished = True
                            yield text + "\n\n"
                    if not finished:
                        raise RuntimeError("Decode stream ended without terminal event")
                except Exception as exc:
                    yield "data: " + json.dumps({"error": {"message": str(exc), "type": "pd_execution_error"}}) + "\n\n"
                    yield "data: [DONE]\n\n"
                finally:
                    task = asyncio.create_task(cleanup())
                    try:
                        await asyncio.shield(task)
                    except asyncio.CancelledError:
                        await asyncio.shield(task)
                        raise
            return StreamingResponse(stream(), media_type="text/event-stream")
        except BaseException:
            await cleanup()
            raise


def create_app(config, transport=None):
    router = Router(config, transport)
    @asynccontextmanager
    async def lifespan(app):
        with log_context(config, "router"):
            try:
                await router.refresh()
                yield
            finally:
                await router.client.aclose()
    app = FastAPI(lifespan=lifespan)
    app.state.router = router
    app.add_api_route("/v1/completions", router.generate, methods=["POST"])
    app.add_api_route("/v1/chat/completions", router.generate, methods=["POST"])
    @app.get("/health")
    async def health():
        await router.refresh()
        try:
            if not router.directory.pairs():
                raise RuntimeError("no compatible pair")
            return {"status": "ok"}
        except RuntimeError:
            return JSONResponse({"status": "unavailable"}, status_code=503)
    @app.get("/v1/models")
    async def models():
        await router.refresh()
        names = sorted({n.contract.model for n in router.directory.nodes.values()})
        return {"object": "list", "data": [{"id": name, "object": "model"} for name in names]}
    @app.exception_handler(RuntimeError)
    async def unavailable(request, exc):
        return JSONResponse({"error": {"message": str(exc)}}, status_code=503)
    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return JSONResponse({"error": {"message": str(exc)}}, status_code=400)
    return app


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pd-config", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8110)
    args = parser.parse_args()
    uvicorn.run(create_app(Config.load(args.pd_config)), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
