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
import dataclasses
import json
import logging
import time
import uuid

from pypto_serving.config.types import GenerateConfig
from pypto_serving.serving.engine.async_engine import AsyncLLMEngine
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

class CompletionRequest(BaseModel):
    model: str = ""
    prompt: str = ""
    # Sampling fields are optional: omitted fields fall back to the server's
    # default GenerateConfig (from --generate-config, else GenerateConfig()).
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
    chat_template_kwargs: dict | None = None


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
    def __init__(
        self,
        async_engine: AsyncLLMEngine,
        model_id: str,
        generate_config: GenerateConfig,
    ) -> None:
        self.engine = async_engine
        self.model_id = model_id
        # Server-wide generate defaults. Fields the HTTP request omits fall
        # back to this config; explicit per-request fields still win.
        self.generate_config = generate_config
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

    def _resolve_generate_config(self, request: CompletionRequest | ChatCompletionRequest) -> GenerateConfig:
        """Build the per-request config from the server-wide defaults.

        A field the request explicitly sets always wins — including "empty"
        values that clear a server default (``stop: []`` clears the server
        stop strings, ``top_k: null`` disables the server top-k). Fields the
        request omits fall back to ``self.generate_config``.
        """
        defaults = self.generate_config
        provided = request.model_fields_set

        if "stop" in provided:
            stop = tuple(request.stop) if request.stop else ()
        else:
            stop = defaults.stop

        return GenerateConfig(
            max_new_tokens=request.max_tokens
            if "max_tokens" in provided
            else defaults.max_new_tokens,
            temperature=request.temperature
            if "temperature" in provided
            else defaults.temperature,
            top_p=request.top_p if "top_p" in provided else defaults.top_p,
            top_k=request.top_k if "top_k" in provided else defaults.top_k,
            seed=request.seed if "seed" in provided else defaults.seed,
            stop=stop,
            stream=request.stream if "stream" in provided else defaults.stream,
        )

    async def _completions(self, request: CompletionRequest) -> StreamingResponse | JSONResponse:
        request_id = f"cmpl-{uuid.uuid4().hex[:8]}"
        config = dataclasses.replace(self._resolve_generate_config(request), ignore_eos=True)

        with profile_span(
            "http.completions",
            cat="request",
            args={"request_id": request_id, "max_tokens": config.max_new_tokens, "stream": request.stream},
        ):
            if request.stream:
                return StreamingResponse(
                    self._stream_completion(request_id, request.prompt, config, request.model or self.model_id),
                    media_type="text/event-stream",
                )

            full_text = ""
            finish_reason = ""
            usage = None
            async for output in self.engine.add_request(request_id, request.prompt, config):
                if output.text:
                    full_text = output.text
                if output.finished:
                    finish_reason = self._map_finish_reason(output.finish_reason)
                    usage = ResponseUsage(
                        prompt_tokens=output.prompt_tokens,
                        completion_tokens=output.completion_tokens,
                        total_tokens=output.prompt_tokens + output.completion_tokens,
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
        # The OpenAI chat schema has no ignore_eos field, so the server-wide
        # config decides it (the completions endpoint keeps its historic
        # always-ignore-EOS override).
        config = dataclasses.replace(
            self._resolve_generate_config(request),
            ignore_eos=self.generate_config.ignore_eos,
        )

        with profile_span(
            "http.chat_completions",
            cat="request",
            args={"request_id": request_id, "max_tokens": config.max_new_tokens, "stream": request.stream},
        ):
            if request.stream:
                return StreamingResponse(
                    self._stream_chat_completion(request_id, prompt, config, request.model or self.model_id),
                    media_type="text/event-stream",
                )

            full_text = ""
            finish_reason = ""
            usage = None
            async for output in self.engine.add_request(request_id, prompt, config):
                if output.text:
                    full_text = output.text
                if output.finished:
                    finish_reason = self._map_finish_reason(output.finish_reason)
                    usage = ResponseUsage(
                        prompt_tokens=output.prompt_tokens,
                        completion_tokens=output.completion_tokens,
                        total_tokens=output.prompt_tokens + output.completion_tokens,
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
        with profile_span("http.stream_completion", cat="request", args={"request_id": request_id}):
            prev_text = ""
            async for output in self.engine.add_request(request_id, prompt, config):
                delta = output.text[len(prev_text):] if output.text else ""
                prev_text = output.text or prev_text
                finish_reason = self._map_finish_reason(output.finish_reason) if output.finished else None

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
                    usage_chunk = CompletionResponse(
                        id=request_id,
                        created=int(time.time()),
                        model=model,
                        choices=[],
                        usage=ResponseUsage(
                            prompt_tokens=output.prompt_tokens,
                            completion_tokens=output.completion_tokens,
                            total_tokens=output.prompt_tokens + output.completion_tokens,
                        ),
                    )
                    yield f"data: {json.dumps(usage_chunk.model_dump())}\n\n"

                    profile_instant(
                        "http.stream_completion.finished",
                        cat="request",
                        args={"request_id": request_id, "finish_reason": finish_reason},
                    )
                    yield "data: [DONE]\n\n"
                    break

    async def _stream_chat_completion(
        self, request_id: str, prompt: str, config: GenerateConfig, model: str
    ):
        with profile_span("http.stream_chat_completion", cat="request", args={"request_id": request_id}):
            prev_text = ""
            async for output in self.engine.add_request(request_id, prompt, config):
                delta = output.text[len(prev_text):] if output.text else ""
                prev_text = output.text or prev_text
                finish_reason = self._map_finish_reason(output.finish_reason) if output.finished else None

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
                yield f"data: {json.dumps(chunk.model_dump())}\n\n"

                if output.finished:
                    # Terminal usage chunk (OpenAI stream_options.include_usage
                    # shape): empty choices, authoritative counts from the engine.
                    usage_chunk = ChatCompletionResponse(
                        id=request_id,
                        object="chat.completion.chunk",
                        created=int(time.time()),
                        model=model,
                        choices=[],
                        usage=ResponseUsage(
                            prompt_tokens=output.prompt_tokens,
                            completion_tokens=output.completion_tokens,
                            total_tokens=output.prompt_tokens + output.completion_tokens,
                        ),
                    )
                    yield f"data: {json.dumps(usage_chunk.model_dump())}\n\n"

                    profile_instant(
                        "http.stream_chat.finished",
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
        kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
        if chat_template_kwargs:
            kwargs.update(chat_template_kwargs)
        kwargs["tokenize"] = False
        kwargs["add_generation_prompt"] = True
        return self.engine.tokenizer.tokenizer.apply_chat_template(hf_messages, **kwargs)

    @staticmethod
    def _map_finish_reason(reason: str) -> str:
        mapping = {
            "FINISHED_EOS": "eos",
            "FINISHED_LENGTH": "length",
            "FINISHED_STOP": "stop",
            "FINISHED_ABORTED": "aborted",
        }
        return mapping.get(reason, "stop")


def create_serving_app(
    async_engine: AsyncLLMEngine,
    model_id: str,
    generate_config: GenerateConfig,
) -> FastAPI:
    server = ServingServer(async_engine, model_id, generate_config)
    return server.app
