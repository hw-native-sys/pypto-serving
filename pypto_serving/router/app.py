# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""OpenAI-compatible public API owned by the external PD Router."""

from __future__ import annotations

import json
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from pypto_serving.serving.pd.admission import PDBackpressureError
from pypto_serving.serving.pd.observability import write_startup_record
from pypto_serving.serving.pd.generation import public_document
from pypto_serving.serving.server.api_types import ChatCompletionRequest, CompletionRequest, validate_chat_request

from .client import NodeClient, PublicResult
from .config import RouterConfig
from .coordinator import DecodeStreamError, RouterCoordinator
from .directory import WorkerDirectory
from .journal import RouterJournal
from .policy import create_route_policy
from .recovery import FixedPairRuntimeManager, RecoveryPhase


def create_router_app(
    config: RouterConfig,
    runtime_manager: FixedPairRuntimeManager | None = None,
) -> FastAPI:
    prefill_clients = tuple(
        NodeClient(url, config.request_timeout_seconds) for url in config.prefill_urls
    )
    decode_clients = tuple(
        NodeClient(url, config.request_timeout_seconds) for url in config.decode_urls
    )
    directory = WorkerDirectory(
        prefill_clients,
        decode_clients,
        config.run_id,
        create_route_policy(config.policy),
        config.control_incarnation,
    )
    journal = RouterJournal(config.journal_path, config.run_id)
    coordinator = RouterCoordinator(
        config,
        directory,
        journal,
        runtime_manager=runtime_manager,
    )
    app = FastAPI(title="PyPTO External PD Router")

    @app.exception_handler(ValueError)
    async def value_error_handler(request, exc: ValueError):  # noqa: ANN001
        return JSONResponse(
            status_code=400,
            content={"object": "error", "message": str(exc)},
        )

    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(request, exc: RuntimeError):  # noqa: ANN001
        return JSONResponse(
            status_code=503,
            content={"object": "error", "message": str(exc)},
        )

    @app.exception_handler(PDBackpressureError)
    async def backpressure_error_handler(request, exc: PDBackpressureError):  # noqa: ANN001
        return JSONResponse(
            status_code=503,
            headers={"Retry-After": "1"},
            content={"object": "error", "message": str(exc)},
        )

    @app.on_event("shutdown")
    async def shutdown() -> None:
        journal.close()

    @app.on_event("startup")
    async def startup() -> None:
        write_startup_record(
            config.log_dir,
            enabled=config.observability_enabled,
            values={
                "process": "router",
                "run_id": config.run_id,
                "provider": config.provider,
                "policy": config.policy,
            },
        )
        await coordinator.reconcile_startup()

    @app.get("/health")
    async def health() -> JSONResponse:
        if coordinator.recovery.phase is not RecoveryPhase.RUNNING:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "error",
                    "reason": coordinator.recovery.phase.value,
                },
            )
        try:
            await directory.refresh()
        except Exception as exc:
            return JSONResponse(
                status_code=503,
                content={"status": "error", "reason": type(exc).__name__},
            )
        return JSONResponse(
            {
                "status": "ok",
                "run_id": config.run_id,
                "control_incarnation": (
                    coordinator.recovery.current.control_incarnation
                ),
                "data_generation": coordinator.recovery.current.data_generation,
            }
        )

    @app.get("/metrics")
    async def metrics() -> JSONResponse:
        return JSONResponse(coordinator.metrics.snapshot())

    @app.get("/recovery")
    async def recovery() -> JSONResponse:
        return JSONResponse(coordinator.recovery.snapshot())

    @app.get("/v1/models")
    async def list_models() -> JSONResponse:
        pair = await directory.select()
        return JSONResponse(await pair.prefill_client.get("/v1/models", dict))

    async def generate(request, kind):
        raw = await _bounded_body(request, config.max_request_bytes)
        if "pd_transfer" in public_document(raw):
            raise ValueError("pd_transfer is reserved for Router assignments")
        if kind == "chat":
            public = ChatCompletionRequest.model_validate_json(raw)
            validate_chat_request(public)
        else:
            public = CompletionRequest.model_validate_json(raw)
        request_id = ("chatcmpl-" if kind == "chat" else "cmpl-") + uuid.uuid4().hex[:8]
        if public.stream:
            outputs = await coordinator.start_stream(kind, raw, request_id)
            return StreamingResponse(
                _forward_stream(outputs), media_type="text/event-stream",
                background=BackgroundTask(outputs.aclose),
            )
        final = None
        async for output in coordinator.generate(kind, raw, request_id, publish_immediately=False):
            final = output
        if not isinstance(final, PublicResult):
            raise RuntimeError("D returned no public JSON result")
        return Response(final.body, media_type="application/json", headers=final.headers)

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await generate(request, "completion")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await generate(request, "chat")

    return app


async def _bounded_body(request: Request, limit: int) -> bytes:
    raw = await request.body()
    if not raw or len(raw) > limit:
        raise ValueError("public request body is empty or exceeds the Router limit")
    return raw


async def _forward_stream(outputs):
    try:
        async for frame in outputs:
            yield frame
    except DecodeStreamError as exc:
        yield exc.frame
        yield b"data: [DONE]\n\n"
    except Exception as exc:
        yield ("data: " + json.dumps({"error": {
            "message": str(exc), "type": "generation_error", "code": 503,
        }}) + "\n\n").encode()
        yield b"data: [DONE]\n\n"
    finally:
        await outputs.aclose()
