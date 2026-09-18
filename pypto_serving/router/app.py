# Copyright (c) PyPTO Contributors.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""OpenAI-compatible public API owned by the external PD Router."""

from __future__ import annotations

import json
import time
import uuid
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from pypto_serving.serving.pd.admission import PDBackpressureError
from pypto_serving.serving.pd.metrics import token_ids_sha256

from .client import NodeClient
from .config import RouterConfig
from .coordinator import RouterCoordinator
from .directory import FixedWorkerDirectory
from .journal import RouterJournal
from .recovery import RecoveryPhase


ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]


class CompletionRequest(BaseModel):
    model: str = ""
    prompt: str | list[int] = ""
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    stop: list[str] | None = None
    stream: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage]
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    stop: list[str] | None = None
    stream: bool = False
    reasoning_effort: ReasoningEffort | None = None
    chat_template_kwargs: dict | None = None


def create_router_app(config: RouterConfig) -> FastAPI:
    auth_secret = config.auth_secret()
    config.route_secret()
    p_client = NodeClient(config.prefill_url, auth_secret, config.request_timeout_seconds)
    d_client = NodeClient(config.decode_url, auth_secret, config.request_timeout_seconds)
    directory = FixedWorkerDirectory(
        p_client,
        d_client,
        config.run_id,
        config.control_incarnation,
    )
    journal = RouterJournal(config.journal_path, config.run_id)
    coordinator = RouterCoordinator(config, directory, journal)
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
                "control_incarnation": config.control_incarnation,
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
        model = config.model_id or pair.prefill.capabilities.model_revision
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
        model = public.model or config.model_id
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
        request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        model = public.model or config.model_id
        if public.stream:
            return StreamingResponse(
                _stream_chat(coordinator, raw, request_id, model),
                media_type="text/event-stream",
            )
        final = None
        async for output in coordinator.generate(
            "chat", raw, request_id, publish_immediately=False
        ):
            final = output
        if final is None:
            raise RuntimeError("Decode completed without output")
        return JSONResponse(
            {
                "id": request_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": final.text},
                        "finish_reason": _map_finish_reason(final.finish_reason),
                    }
                ],
                "usage": _usage(final),
            },
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
        delta = output.text[len(previous) :] if output.text else ""
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
):
    previous = ""
    async for output in coordinator.generate("chat", raw, request_id):
        delta = output.text[len(previous) :] if output.text else ""
        previous = output.text or previous
        finish_reason = _map_finish_reason(output.finish_reason) if output.finished else None
        yield _sse(
            {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": delta},
                        "finish_reason": finish_reason,
                    }
                ],
            }
        )
        if output.finished:
            yield _sse(
                {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [],
                    "usage": _usage(output),
                }
            )
            yield b"data: [DONE]\n\n"


def _sse(value: dict) -> bytes:
    return b"data: " + json.dumps(value, ensure_ascii=False).encode() + b"\n\n"


def _usage(output) -> dict:
    return {
        "prompt_tokens": output.prompt_tokens,
        "completion_tokens": output.completion_tokens,
        "total_tokens": output.prompt_tokens + output.completion_tokens,
    }


def _map_finish_reason(reason: str) -> str:
    return {
        "FINISHED_EOS": "eos",
        "FINISHED_LENGTH": "length",
        "FINISHED_STOP": "stop",
    }.get(reason, reason.lower() if reason else "stop")
