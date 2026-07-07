# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import json
import logging
import time
import uuid

from .async_engine import AsyncLLMEngine
from python.profile import profile_instant, profile_span
from .types import GenerateConfig

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel
except ImportError as e:
    raise ImportError(
        "Serving requires fastapi and pydantic. Install with: pip install fastapi uvicorn sse-starlette pydantic"
    ) from e


# --- Request/Response Models ---

class CompletionRequest(BaseModel):
    model: str = ""
    prompt: str = ""
    max_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int | None = None
    stop: list[str] | None = None
    stream: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage]
    max_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int | None = None
    stop: list[str] | None = None
    stream: bool = False
    chat_template_kwargs: dict | None = None


class CompletionChoice(BaseModel):
    index: int = 0
    text: str = ""
    finish_reason: str | None = None


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]


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


# --- Server ---

class ServingServer:
    def __init__(self, async_engine: AsyncLLMEngine, model_id: str) -> None:
        self.engine = async_engine
        self.model_id = model_id
        self.app = FastAPI(title="PyPTO Serving")
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

    async def _health(self) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    async def _list_models(self) -> JSONResponse:
        return JSONResponse({
            "object": "list",
            "data": [{"id": self.model_id, "object": "model", "owned_by": "pypto"}],
        })

    async def _completions(self, request: CompletionRequest) -> StreamingResponse | JSONResponse:
        request_id = f"cmpl-{uuid.uuid4().hex[:8]}"
        config = GenerateConfig(
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop=tuple(request.stop) if request.stop else (),
            stream=request.stream,
        )

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

            full_text = ""
            finish_reason = ""
            async for output in self.engine.add_request(request_id, request.prompt, config):
                if output.text:
                    full_text = output.text
                if output.finished:
                    finish_reason = self._map_finish_reason(output.finish_reason)

            response = CompletionResponse(
                id=request_id,
                created=int(time.time()),
                model=request.model or self.model_id,
                choices=[CompletionChoice(text=full_text, finish_reason=finish_reason)],
            )
            return JSONResponse(response.model_dump())

    async def _chat_completions(self, request: ChatCompletionRequest) -> StreamingResponse | JSONResponse:
        request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        prompt = self._apply_chat_template(request.messages, request.chat_template_kwargs)
        config = GenerateConfig(
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop=tuple(request.stop) if request.stop else (),
            stream=request.stream,
        )

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

            full_text = ""
            finish_reason = ""
            async for output in self.engine.add_request(request_id, prompt, config):
                if output.text:
                    full_text = output.text
                if output.finished:
                    finish_reason = self._map_finish_reason(output.finish_reason)

            response = ChatCompletionResponse(
                id=request_id,
                object="chat.completion",
                created=int(time.time()),
                model=request.model or self.model_id,
                choices=[ChatCompletionChoice(
                    message=ChatMessage(role="assistant", content=full_text),
                    finish_reason=finish_reason,
                )],
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
            "FINISHED_EOS": "stop",
            "FINISHED_LENGTH": "length",
            "FINISHED_STOP": "stop",
            "FINISHED_ABORTED": "stop",
        }
        return mapping.get(reason, "stop")


def create_serving_app(async_engine: AsyncLLMEngine, model_id: str) -> FastAPI:
    server = ServingServer(async_engine, model_id)
    return server.app
