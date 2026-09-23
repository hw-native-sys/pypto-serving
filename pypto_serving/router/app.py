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
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from pypto_serving.serving.pd.admission import PDBackpressureError
from pypto_serving.serving.pd.observability import token_ids_sha256, write_startup_record
from pypto_serving.serving.server.api_types import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    ResponseUsage,
    validate_chat_request,
)
from pypto_serving.serving.server.chat_format import (
    chat_delta,
    chat_finish_reason,
    chat_message,
    map_finish_reason,
)

from .client import NodeClient
from .config import RouterConfig
from .coordinator import RouterCoordinator
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
        model = pair.prefill.capabilities.model_revision
        return JSONResponse(
            {
                "object": "list",
                "data": [{"id": model, "object": "model", "owned_by": "pypto"}],
            }
        )

    @app.post("/v1/completions")
    async def completions(request: Request):  # noqa: ANN202
        raw = await _bounded_body(request, config.max_request_bytes)
        public = CompletionRequest.model_validate_json(raw)
        request_id = f"cmpl-{uuid.uuid4().hex[:8]}"
        model = public.model
        if public.stream:
            return StreamingResponse(
                _stream_completion(coordinator, raw, request_id, model),
                media_type="text/event-stream",
            )
        final = None
        async for output in coordinator.generate(
            "completion", raw, request_id, publish_immediately=False
        ):
            final = output
        if final is None:
            raise RuntimeError("Decode completed without output")
        return JSONResponse(
            {
                "id": request_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "text": final.text,
                        "finish_reason": _map_finish_reason(final.finish_reason),
                    }
                ],
                "usage": _usage(final),
            },
            headers={
                "x-pypto-token-ids-sha256": token_ids_sha256(final.token_ids)
            },
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):  # noqa: ANN202
        raw = await _bounded_body(request, config.max_request_bytes)
        public = ChatCompletionRequest.model_validate_json(raw)
        validate_chat_request(public)
        request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        model = public.model
        if public.stream:
            return StreamingResponse(
                _stream_chat(
                    coordinator,
                    raw,
                    request_id,
                    model,
                    parallel_tool_calls=public.parallel_tool_calls,
                ),
                media_type="text/event-stream",
            )
        final = None
        async for output in coordinator.generate(
            "chat", raw, request_id, publish_immediately=False
        ):
            final = output
        if final is None:
            raise RuntimeError("Decode completed without output")
        response = ChatCompletionResponse(
            id=request_id,
            created=int(time.time()),
            model=model,
            choices=[ChatCompletionChoice(
                message=chat_message(
                    final.text,
                    final.reasoning,
                    final.tool_calls,
                    parallel_tool_calls=public.parallel_tool_calls,
                ),
                finish_reason=chat_finish_reason(final),
            )],
            usage=ResponseUsage(**_usage(final)),
        )
        return JSONResponse(
            response.model_dump(),
            headers={
                "x-pypto-token-ids-sha256": token_ids_sha256(final.token_ids)
            },
        )

    return app


async def _bounded_body(request: Request, limit: int) -> bytes:
    raw = await request.body()
    if not raw or len(raw) > limit:
        raise ValueError("public request body is empty or exceeds the Router limit")
    return raw


async def _stream_completion(
    coordinator: RouterCoordinator,
    raw: bytes,
    request_id: str,
    model: str,
):
    previous = ""
    async for output in coordinator.generate("completion", raw, request_id):
        delta = _cumulative_delta(output.text, previous, "content")
        previous = output.text or previous
        finish_reason = _map_finish_reason(output.finish_reason) if output.finished else None
        yield _sse(
            {
                "id": request_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "text": delta, "finish_reason": finish_reason}],
            }
        )
        if output.finished:
            yield _sse(
                {
                    "id": request_id,
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [],
                    "usage": _usage(output),
                }
            )
            yield b"data: [DONE]\n\n"


async def _stream_chat(
    coordinator: RouterCoordinator,
    raw: bytes,
    request_id: str,
    model: str,
    *,
    parallel_tool_calls: bool = True,
):
    try:
        async for chunk in _stream_chat_chunks(
            coordinator,
            raw,
            request_id,
            model,
            parallel_tool_calls=parallel_tool_calls,
        ):
            yield chunk
    except (ValueError, RuntimeError) as exc:
        # Decode/parser errors may occur after SSE headers have been sent.
        # Keep the public stream well formed, as ordinary Serving does.
        yield _sse({"error": {
            "message": str(exc),
            "type": "invalid_model_output" if isinstance(exc, ValueError) else "pd_decode_error",
            "code": 400 if isinstance(exc, ValueError) else 503,
        }})
        yield b"data: [DONE]\n\n"


async def _stream_chat_chunks(
    coordinator: RouterCoordinator,
    raw: bytes,
    request_id: str,
    model: str,
    *,
    parallel_tool_calls: bool,
):
    async for output in coordinator.generate("chat", raw, request_id):
        finish_reason = chat_finish_reason(output) if output.finished else None
        chunk = ChatCompletionResponse(
            id=request_id,
            object="chat.completion.chunk",
            created=int(time.time()),
            model=model,
            choices=[ChatCompletionChoice(
                delta=chat_delta(
                    output.text_delta,
                    output.reasoning_delta,
                    output.tool_call_deltas,
                    parallel_tool_calls=parallel_tool_calls,
                ),
                finish_reason=finish_reason,
            )],
        )
        yield _sse(chunk.model_dump())
        if output.finished:
            usage_chunk = ChatCompletionResponse(
                id=request_id,
                object="chat.completion.chunk",
                created=int(time.time()),
                model=model,
                choices=[],
                usage=ResponseUsage(**_usage(output)),
            )
            yield _sse(usage_chunk.model_dump())
            yield b"data: [DONE]\n\n"


def _sse(value: dict) -> bytes:
    return b"data: " + json.dumps(value, ensure_ascii=False).encode() + b"\n\n"


def _cumulative_delta(current: str, previous: str, field: str) -> str:
    """Validate replay-safe cumulative output and return only its new suffix."""
    if not current.startswith(previous):
        raise RuntimeError(f"Decode {field} output changed an already published prefix")
    return current[len(previous) :]


def _chat_message(output, *, parallel_tool_calls: bool = True) -> dict:
    return chat_message(
        output.text,
        output.reasoning,
        output.tool_calls,
        parallel_tool_calls=parallel_tool_calls,
    ).model_dump()


def _usage(output) -> dict:
    return {
        "prompt_tokens": output.prompt_tokens,
        "completion_tokens": output.completion_tokens,
        "total_tokens": output.prompt_tokens + output.completion_tokens,
    }


def _map_finish_reason(reason: str) -> str:
    return map_finish_reason(reason)
