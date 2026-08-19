# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid

from pypto_serving.config.types import GenerateConfig
from pypto_serving.serving.engine.async_engine import AsyncLLMEngine, TokenOutput
from pypto_serving.tools.profile import (
    get_profiler,
    merge_profile,
    profile_instant,
    profile_span,
    start_profile as start_sa_profile,
    stop_profile as stop_sa_profile,
)

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, Response, StreamingResponse
    from pydantic import BaseModel
except ImportError as e:
    raise ImportError(
        "Serving requires fastapi and pydantic. Install with: pip install fastapi uvicorn sse-starlette pydantic"
    ) from e


# --- Request/Response Models ---

class _GenerationRequest(BaseModel):
    model: str = ""
    max_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int | None = None
    stop: list[str] | None = None
    stream: bool = False


class CompletionRequest(_GenerationRequest):
    prompt: str = ""


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(_GenerationRequest):
    messages: list[ChatMessage]
    chat_template_kwargs: dict[str, object] | None = None


class CompletionChoice(BaseModel):
    index: int = 0
    text: str = ""
    finish_reason: str | None = None


class ResponseUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: ResponseUsage | None = None


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage | None = None
    delta: ChatMessage | None = None
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ResponseUsage | None = None


# --- Server ---

class ServingServer:
    def __init__(self, async_engine: AsyncLLMEngine, model_id: str) -> None:
        self.engine = async_engine
        self.model_id = model_id
        self.app = FastAPI(title="PyPTO Serving")
        self._profile_lock = asyncio.Lock()
        self._register_exception_handlers()
        self._register_routes()

    def _register_exception_handlers(self) -> None:
        # Surface scheduler/engine rejections (e.g. a prompt longer than
        # max_seq_len) as a clean HTTP 400 instead of an unhandled 500.
        @self.app.exception_handler(ValueError)
        async def _value_error_handler(request, exc: ValueError) -> JSONResponse:  # noqa: ANN001
            return JSONResponse(
                status_code=400,
                content={"object": "error", "message": str(exc)},
            )

    def _register_routes(self) -> None:
        self.app.add_api_route("/health", self._health, methods=["GET"])
        self.app.add_api_route("/v1/models", self._list_models, methods=["GET"])
        self.app.add_api_route("/v1/completions", self._completions, methods=["POST"], response_model=None)
        self.app.add_api_route("/v1/chat/completions", self._chat_completions, methods=["POST"], response_model=None)
        if get_profiler(initially_active=False).enabled:
            self.app.add_api_route("/start_profile", self._start_profile, methods=["POST"])
            self.app.add_api_route("/stop_profile", self._stop_profile, methods=["POST"])

    async def _health(self) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    async def _list_models(self) -> JSONResponse:
        return JSONResponse({
            "object": "list",
            "data": [{"id": self.model_id, "object": "model", "owned_by": "pypto"}],
        })

    async def _start_profile(self) -> Response:
        async with self._profile_lock:
            logger.info("Starting SA profiler...")
            main_started = start_sa_profile()
            try:
                await self.engine.start_profile()
            except Exception:
                if main_started:
                    stop_sa_profile()
                raise
            logger.info("SA profiler started")
        return Response(status_code=200)

    async def _stop_profile(self) -> Response:
        async with self._profile_lock:
            logger.info("Stopping SA profiler...")
            stop_error = None
            try:
                await self.engine.stop_profile()
            except Exception as exc:
                stop_error = exc
            stop_sa_profile()
            try:
                event_count = merge_profile()
            except Exception:
                if stop_error is None:
                    raise
                logger.exception(
                    "Failed to merge SA profile after worker profile stop failed"
                )
            else:
                logger.info("SA profiler stopped; merged %d events", event_count)
            if stop_error is not None:
                raise stop_error
        return Response(status_code=200)

    @staticmethod
    def _build_generate_config(
        request: _GenerationRequest,
        *,
        ignore_eos: bool = False,
    ) -> GenerateConfig:
        return GenerateConfig(
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop=tuple(request.stop or ()),
            stream=request.stream,
            ignore_eos=ignore_eos,
        )

    async def _collect_generation(
        self,
        request_id: str,
        prompt: str,
        config: GenerateConfig,
    ) -> tuple[str, str, ResponseUsage | None]:
        full_text = ""
        finish_reason = ""
        usage = None
        async for output in self.engine.add_request(request_id, prompt, config):
            if output.text:
                full_text = output.text
            if output.finished:
                finish_reason = self._map_finish_reason(output.finish_reason)
                usage = self._response_usage(output)
        return full_text, finish_reason, usage

    async def _completions(self, request: CompletionRequest) -> StreamingResponse | JSONResponse:
        request_id = f"cmpl-{uuid.uuid4().hex[:8]}"
        config = self._build_generate_config(request, ignore_eos=True)

        with profile_span(
            "http.completions",
            cat="request",
            args={"request_id": request_id, "max_tokens": request.max_tokens, "stream": request.stream},
        ):
            if request.stream:
                return StreamingResponse(
                    self._stream_completion(request_id, request.prompt, config, request.model or self.model_id),
                    media_type="text/event-stream",
                )

            full_text, finish_reason, usage = await self._collect_generation(
                request_id,
                request.prompt,
                config,
            )

            response = CompletionResponse(
                id=request_id,
                created=int(time.time()),
                model=request.model or self.model_id,
                choices=[CompletionChoice(text=full_text, finish_reason=finish_reason)],
                usage=usage,
            )
            return JSONResponse(response.model_dump())

    async def _chat_completions(self, request: ChatCompletionRequest) -> StreamingResponse | JSONResponse:
        request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        prompt = self._apply_chat_template(request.messages, request.chat_template_kwargs)
        config = self._build_generate_config(request)

        with profile_span(
            "http.chat_completions",
            cat="request",
            args={"request_id": request_id, "max_tokens": request.max_tokens, "stream": request.stream},
        ):
            if request.stream:
                return StreamingResponse(
                    self._stream_chat_completion(request_id, prompt, config, request.model or self.model_id),
                    media_type="text/event-stream",
                )

            full_text, finish_reason, usage = await self._collect_generation(
                request_id,
                prompt,
                config,
            )

            response = ChatCompletionResponse(
                id=request_id,
                object="chat.completion",
                created=int(time.time()),
                model=request.model or self.model_id,
                choices=[ChatCompletionChoice(
                    message=ChatMessage(role="assistant", content=full_text),
                    finish_reason=finish_reason,
                )],
                usage=usage,
            )
            return JSONResponse(response.model_dump())

    async def _stream_completion(
        self, request_id: str, prompt: str, config: GenerateConfig, model: str
    ):
        async for chunk in self._stream_response(request_id, prompt, config, model, is_chat=False):
            yield chunk

    async def _stream_chat_completion(
        self, request_id: str, prompt: str, config: GenerateConfig, model: str
    ):
        async for chunk in self._stream_response(request_id, prompt, config, model, is_chat=True):
            yield chunk

    async def _stream_response(
        self,
        request_id: str,
        prompt: str,
        config: GenerateConfig,
        model: str,
        *,
        is_chat: bool,
    ):
        span_name = "http.stream_chat_completion" if is_chat else "http.stream_completion"
        finished_event = "http.stream_chat.finished" if is_chat else "http.stream_completion.finished"
        with profile_span(span_name, cat="request", args={"request_id": request_id}):
            prev_text = ""
            async for output in self.engine.add_request(request_id, prompt, config):
                delta = output.text[len(prev_text):] if output.text else ""
                prev_text = output.text or prev_text
                finish_reason = self._map_finish_reason(output.finish_reason) if output.finished else None

                if is_chat:
                    chunk = ChatCompletionResponse(
                        id=request_id,
                        object="chat.completion.chunk",
                        created=int(time.time()),
                        model=model,
                        choices=[ChatCompletionChoice(
                            delta=ChatMessage(role="assistant", content=delta),
                            finish_reason=finish_reason,
                        )],
                    )
                else:
                    chunk = CompletionResponse(
                        id=request_id,
                        created=int(time.time()),
                        model=model,
                        choices=[CompletionChoice(text=delta, finish_reason=finish_reason)],
                    )
                yield f"data: {json.dumps(chunk.model_dump())}\n\n"

                if output.finished:
                    # Terminal usage chunk (OpenAI stream_options.include_usage
                    # shape): empty choices, authoritative counts from the engine.
                    if is_chat:
                        usage_chunk = ChatCompletionResponse(
                            id=request_id,
                            object="chat.completion.chunk",
                            created=int(time.time()),
                            model=model,
                            choices=[],
                            usage=self._response_usage(output),
                        )
                    else:
                        usage_chunk = CompletionResponse(
                            id=request_id,
                            created=int(time.time()),
                            model=model,
                            choices=[],
                            usage=self._response_usage(output),
                        )
                    yield f"data: {json.dumps(usage_chunk.model_dump())}\n\n"

                    profile_instant(
                        finished_event,
                        cat="request",
                        args={"request_id": request_id, "finish_reason": finish_reason},
                    )
                    yield "data: [DONE]\n\n"
                    break

    def _apply_chat_template(
        self, messages: list[ChatMessage], chat_template_kwargs: dict | None = None,
    ) -> str:
        """Apply the model's official chat template, forwarding chat_template_kwargs.

        ``chat_template_kwargs`` (e.g. ``{"enable_thinking": False}`` for Qwen3) is
        passed straight through to ``apply_chat_template``, mirroring vLLM so clients
        control thinking mode per request.
        """
        hf_messages = [{"role": m.role, "content": m.content} for m in messages]
        return self.engine.tokenizer.apply_chat_template(hf_messages, chat_template_kwargs)

    @staticmethod
    def _response_usage(output: TokenOutput) -> ResponseUsage:
        return ResponseUsage(
            prompt_tokens=output.prompt_tokens,
            completion_tokens=output.completion_tokens,
            total_tokens=output.prompt_tokens + output.completion_tokens,
        )

    @staticmethod
    def _map_finish_reason(reason: str) -> str:
        mapping = {
            "FINISHED_EOS": "eos",
            "FINISHED_LENGTH": "length",
            "FINISHED_STOP": "stop",
            "FINISHED_ABORTED": "aborted",
        }
        return mapping.get(reason, "stop")


def create_serving_app(async_engine: AsyncLLMEngine, model_id: str) -> FastAPI:
    server = ServingServer(async_engine, model_id)
    return server.app
