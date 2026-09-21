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
import contextlib
import dataclasses
import json
import logging
import time
import uuid
from collections.abc import Sequence
from typing import Literal

from pypto_serving.config.types import GenerateConfig
from pypto_serving.serving.engine.async_engine import AsyncLLMEngine, TokenOutput
from pypto_serving.serving.pd.metrics import token_ids_sha256
from pypto_serving.serving.reasoning import OutputParserSpec, ToolCallDelta, supports_tool_calls
from pypto_serving.tools.profile import (
    get_profiler,
    merge_profile,
    profile_instant,
    profile_span,
    start_profile as start_sa_profile,
    stop_profile as stop_sa_profile,
)

logger = logging.getLogger(__name__)

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response, StreamingResponse
    from pydantic import BaseModel, Field, model_serializer
except ImportError as e:
    raise ImportError(
        "Serving requires fastapi and pydantic. Install with: pip install fastapi uvicorn sse-starlette pydantic"
    ) from e


# --- Request/Response Models ---

class CompletionRequest(BaseModel):
    model: str = ""
    prompt: str | list[int] = ""
    # Sampling fields are optional: omitted fields fall back to the server's
    # default GenerateConfig (from --generate-config, else GenerateConfig()).
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    stop: list[str] | None = None
    stream: bool = False


class FunctionDefinition(BaseModel):
    name: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    description: str | None = None
    parameters: dict = Field(default_factory=lambda: {"type": "object", "properties": {}})
    strict: bool | None = None


class ChatTool(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition


class FunctionCall(BaseModel):
    name: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    arguments: str


class ToolCall(BaseModel):
    id: str = Field(min_length=1)
    type: Literal["function"] = "function"
    function: FunctionCall


class ChatMessage(BaseModel):
    role: str
    content: str | None = None
    reasoning: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        data = handler(self)
        for key in ("tool_calls", "tool_call_id"):
            if not data.get(key):
                data.pop(key, None)
        return data


class DeltaFunctionCall(BaseModel):
    name: str | None = None
    arguments: str | None = None


class DeltaToolCall(BaseModel):
    index: int
    id: str | None = None
    type: Literal["function"] | None = None
    function: DeltaFunctionCall

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        data = {key: value for key, value in handler(self).items() if value is not None}
        data["function"] = {key: value for key, value in data["function"].items() if value is not None}
        return data


class ChatDelta(ChatMessage):
    tool_calls: list[DeltaToolCall] | None = None


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
    include_reasoning: bool = True
    tools: list[ChatTool] | None = None
    tool_choice: str | dict | None = None
    parallel_tool_calls: bool = True


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
    delta: ChatDelta | None = None
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

        from pypto_serving.serving.pd.admission import (  # noqa: PLC0415
            PDBackpressureError,
        )

        @self.app.exception_handler(PDBackpressureError)
        async def _pd_backpressure_handler(  # noqa: ANN001
            request,
            exc: PDBackpressureError,
        ) -> JSONResponse:
            return JSONResponse(
                status_code=503,
                headers={"Retry-After": "1"},
                content={"object": "error", "message": str(exc)},
            )

        @self.app.exception_handler(PermissionError)
        async def _permission_error_handler(  # noqa: ANN001
            request,
            exc: PermissionError,
        ) -> JSONResponse:
            return JSONResponse(
                status_code=401,
                content={"object": "error", "message": str(exc)},
            )

    def _register_routes(self) -> None:
        self.app.add_api_route("/health", self._health, methods=["GET"])
        pd_config = getattr(getattr(self.engine, "config", None), "pd_config", None)
        if pd_config is not None and getattr(pd_config, "enabled", False):
            self._register_internal_pd_routes()
            return
        self.app.add_api_route("/v1/models", self._list_models, methods=["GET"])
        self.app.add_api_route("/v1/completions", self._completions, methods=["POST"], response_model=None)
        self.app.add_api_route("/v1/chat/completions", self._chat_completions, methods=["POST"], response_model=None)
        if getattr(self.engine, "metrics", None) is not None:
            self.app.add_api_route("/metrics", self._metrics, methods=["GET"])
            self.app.add_api_route("/metrics/json", self._metrics_json, methods=["GET"])
        if get_profiler(initially_active=False).enabled:
            self.app.add_api_route("/start_profile", self._start_profile, methods=["POST"])
            self.app.add_api_route("/stop_profile", self._stop_profile, methods=["POST"])

    def _register_internal_pd_routes(self) -> None:
        self.app.add_api_route(
            "/internal/pd/descriptor",
            self._pd_descriptor,
            methods=["GET"],
        )
        self.app.add_api_route(
            "/internal/pd/capacity",
            self._pd_capacity,
            methods=["GET"],
        )
        self.app.add_api_route(
            "/internal/pd/metrics",
            self._pd_metrics,
            methods=["GET"],
        )
        self.app.add_api_route(
            "/internal/pd/prepare",
            self._pd_prepare,
            methods=["POST"],
        )
        self.app.add_api_route(
            "/internal/pd/reserve",
            self._pd_reserve,
            methods=["POST"],
        )
        self.app.add_api_route(
            "/internal/pd/authorize",
            self._pd_authorize,
            methods=["POST"],
        )
        self.app.add_api_route(
            "/internal/pd/execute",
            self._pd_execute,
            methods=["POST"],
        )
        self.app.add_api_route(
            "/internal/pd/await-decode",
            self._pd_await_decode,
            methods=["POST"],
        )
        self.app.add_api_route(
            "/internal/pd/query",
            self._pd_query,
            methods=["POST"],
        )
        self.app.add_api_route(
            "/internal/pd/abort",
            self._pd_abort,
            methods=["POST"],
        )
        if get_profiler().enabled:
            self.app.add_api_route(
                "/internal/pd/start-profile",
                self._pd_start_profile,
                methods=["POST"],
            )
            self.app.add_api_route(
                "/internal/pd/stop-profile",
                self._pd_stop_profile,
                methods=["POST"],
            )

    async def _pd_descriptor(self, request: Request) -> Response:
        from pypto_serving.serving.pd.http_api import encode_json  # noqa: PLC0415

        return Response(
            encode_json(self.engine.pd_service.descriptor()),
            media_type="application/json",
        )

    async def _pd_capacity(self, request: Request) -> Response:
        from pypto_serving.serving.pd.http_api import encode_json  # noqa: PLC0415

        return Response(
            encode_json(self.engine.pd_service.capacity_snapshot()),
            media_type="application/json",
        )

    async def _pd_metrics(self, request: Request) -> JSONResponse:
        return JSONResponse(await self.engine.pd_service.metrics_snapshot())

    async def _pd_start_profile(self, request: Request) -> Response:
        return await self._start_profile()

    async def _pd_stop_profile(self, request: Request) -> Response:
        return await self._stop_profile()

    async def _pd_prepare(self, request: Request) -> Response:
        from pypto_serving.serving.pd.http_api import (  # noqa: PLC0415
            PrepareRequestHTTP,
            decode_json,
            encode_json,
        )

        payload = decode_json(await request.body(), PrepareRequestHTTP)
        output_parser_spec = None
        if payload.request_kind == "completion":
            public = CompletionRequest.model_validate_json(payload.request_json)
            prompt, prompt_token_ids = self._completion_prompt(public.prompt)
            if prompt_token_ids is None:
                prompt_token_ids = tuple(
                    int(token) for token in self.engine._tokenize_prompt(prompt)
                )
            config = dataclasses.replace(
                self._resolve_generate_config(public),
                ignore_eos=True,
            )
        elif payload.request_kind == "chat":
            public = ChatCompletionRequest.model_validate_json(payload.request_json)
            prompt = self._apply_chat_template(
                public.messages,
                public.chat_template_kwargs,
                reasoning_effort=public.reasoning_effort,
            )
            prompt_token_ids = tuple(
                int(token) for token in self.engine._tokenize_prompt(prompt)
            )
            config = dataclasses.replace(
                self._resolve_generate_config(public),
                ignore_eos=self.generate_config.ignore_eos,
            )
            output_parser_spec = self._output_parser_spec(public)
        else:
            raise ValueError("unsupported PD public request kind")
        prepared = self.engine.pd_service.prepare_request(
            payload.request_id,
            prompt,
            config,
            prompt_token_ids,
            output_parser_spec=output_parser_spec,
        )
        return Response(encode_json(prepared), media_type="application/json")

    async def _pd_reserve(self, request: Request) -> Response:
        from pypto_serving.serving.pd.http_api import (  # noqa: PLC0415
            PlacementRejection,
            ReservePlacementResult,
            ReservePlacementHTTP,
            decode_json,
            encode_json,
        )

        payload = decode_json(await request.body(), ReservePlacementHTTP)
        outcome = self.engine.pd_service.reserve_placement(payload)
        result = ReservePlacementResult(
            rejection=outcome if isinstance(outcome, PlacementRejection) else None,
            reservation=None if isinstance(outcome, PlacementRejection) else outcome,
        )
        return Response(encode_json(result), media_type="application/json")

    async def _pd_authorize(self, request: Request) -> Response:
        from pypto_serving.serving.pd.http_api import (  # noqa: PLC0415
            AuthorizeRouteHTTP,
            decode_json,
            encode_json,
        )

        payload = decode_json(await request.body(), AuthorizeRouteHTTP)
        self.engine.pd_service.authorize_route(payload)
        return Response(encode_json({"status": "AUTHORIZED"}), media_type="application/json")

    async def _pd_execute(self, request: Request) -> Response:
        from pypto_serving.serving.pd.http_api import (  # noqa: PLC0415
            ExecutePrefillHTTP,
            decode_json,
            encode_json,
        )

        payload = decode_json(await request.body(), ExecutePrefillHTTP)
        result = await self.engine.pd_service.execute_prefill(payload)
        return Response(encode_json(result), media_type="application/json")

    async def _pd_await_decode(self, request: Request) -> StreamingResponse:
        from pypto_serving.serving.pd.http_api import (  # noqa: PLC0415
            HandoffHTTP,
            decode_json,
            encode_json,
        )

        payload = decode_json(await request.body(), HandoffHTTP)
        stream = self.engine.pd_service.open_decode_stream(payload.key)

        async def ndjson():
            async for frame in stream:
                yield encode_json(frame) + b"\n"

        return StreamingResponse(ndjson(), media_type="application/x-ndjson")

    async def _pd_query(self, request: Request) -> Response:
        from pypto_serving.serving.pd.http_api import (  # noqa: PLC0415
            HandoffHTTP,
            decode_json,
            encode_json,
        )

        payload = decode_json(await request.body(), HandoffHTTP)
        return Response(
            encode_json(self.engine.pd_service.query_handoff(payload.key)),
            media_type="application/json",
        )

    async def _pd_abort(self, request: Request) -> Response:
        from pypto_serving.serving.pd.http_api import (  # noqa: PLC0415
            AbortHandoffHTTP,
            decode_json,
            encode_json,
        )

        payload = decode_json(await request.body(), AbortHandoffHTTP)
        status = await self.engine.pd_service.abort_handoff(
            payload.key,
            payload.reason,
            deterministic=payload.deterministic,
        )
        return Response(encode_json(status), media_type="application/json")

    async def _health(self) -> JSONResponse:
        if getattr(self.engine, "pd_health_error", ""):
            return JSONResponse(
                status_code=503,
                content={
                    "status": "error",
                    "reason": "PD_RECOVERY_REQUIRED",
                },
            )
        content = {"status": "ok"}
        pd_config = getattr(getattr(self.engine, "config", None), "pd_config", None)
        if pd_config is not None and getattr(pd_config, "enabled", False):
            content.update(
                {
                    "run_id": pd_config.run_id,
                    "node_id": pd_config.node_id,
                    "role": pd_config.role.value,
                    "generation": pd_config.generation,
                    "control_incarnation": pd_config.control_incarnation,
                }
            )
        return JSONResponse(content)

    async def _list_models(self) -> JSONResponse:
        return JSONResponse({
            "object": "list",
            "data": [{"id": self.model_id, "object": "model", "owned_by": "pypto"}],
        })

    async def _metrics(self) -> Response:
        return Response(
            content=self.engine.metrics.render_prometheus(),
            media_type="text/plain; version=0.0.4",
        )

    async def _metrics_json(self) -> JSONResponse:
        return JSONResponse(self.engine.metrics.snapshot())

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
        prompt, prompt_token_ids = self._completion_prompt(request.prompt)

        with profile_span(
            "http.completions",
            cat="request",
            args={"request_id": request_id, "max_tokens": config.max_new_tokens, "stream": request.stream},
        ):
            if request.stream:
                return StreamingResponse(
                    self._stream_completion(
                        request_id,
                        prompt,
                        config,
                        request.model or self.model_id,
                        prompt_token_ids=prompt_token_ids,
                    ),
                    media_type="text/event-stream",
                )

            full_text = ""
            finish_reason = ""
            usage = None
            final_token_ids: tuple[int, ...] = ()
            async for output in self.engine.add_request(
                request_id,
                prompt,
                config,
                prompt_token_ids=prompt_token_ids,
            ):
                if output.text:
                    full_text = output.text
                if output.finished:
                    final_token_ids = output.token_ids
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
            return JSONResponse(
                response.model_dump(),
                headers={
                    "x-pypto-token-ids-sha256": token_ids_sha256(final_token_ids)
                },
            )

    async def _chat_completions(self, request: ChatCompletionRequest) -> StreamingResponse | JSONResponse:
        request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        output_parser_spec = self._output_parser_spec(request)
        prompt = self._apply_chat_template(
            request.messages,
            request.chat_template_kwargs,
            reasoning_effort=request.reasoning_effort,
            tools=request.tools,
        )
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
                    self._stream_chat_completion(
                        request_id,
                        prompt,
                        config,
                        request.model or self.model_id,
                        output_parser_spec=output_parser_spec,
                        parallel_tool_calls=request.parallel_tool_calls,
                    ),
                    media_type="text/event-stream",
                )

            full_text = ""
            full_reasoning = ""
            tool_calls = ()
            finish_reason = ""
            usage = None
            final_token_ids: tuple[int, ...] = ()
            async for output in self.engine.add_request(
                request_id,
                prompt,
                config,
                output_parser_spec=output_parser_spec,
            ):
                if output.text:
                    full_text = output.text
                if output.reasoning:
                    full_reasoning = output.reasoning
                if output.finished:
                    final_token_ids = output.token_ids
                    finish_reason = self._chat_finish_reason(output)
                    tool_calls = output.tool_calls if request.parallel_tool_calls else output.tool_calls[:1]
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
                    message=ChatMessage(
                        role="assistant",
                        content=full_text or (None if tool_calls else ""),
                        reasoning=full_reasoning or None,
                        tool_calls=[
                            ToolCall(id=call.id, function=FunctionCall(name=call.name, arguments=call.arguments))
                            for call in tool_calls
                        ] or None,
                    ),
                    finish_reason=finish_reason,
                )],
                usage=usage,
            )
            return JSONResponse(
                response.model_dump(),
                headers={
                    "x-pypto-token-ids-sha256": token_ids_sha256(final_token_ids)
                },
            )

    async def _stream_completion(
        self,
        request_id: str,
        prompt: str,
        config: GenerateConfig,
        model: str,
        *,
        prompt_token_ids: Sequence[int] | None = None,
    ):
        with profile_span("http.stream_completion", cat="request", args={"request_id": request_id}):
            prev_text = ""
            async for output in self.engine.add_request(
                request_id,
                prompt,
                config,
                prompt_token_ids=prompt_token_ids,
            ):
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

    def _completion_prompt(
        self,
        prompt: str | list[int],
    ) -> tuple[str, tuple[int, ...] | None]:
        if isinstance(prompt, str):
            return prompt, None
        tokens = tuple(prompt)
        if not tokens or any(type(token) is not int or token < 0 for token in tokens):
            raise ValueError("Prompt token IDs must be nonempty non-negative integers.")
        return "", tokens

    async def _stream_chat_completion(
        self,
        request_id: str,
        prompt: str,
        config: GenerateConfig,
        model: str,
        *,
        output_parser_spec: OutputParserSpec | None = None,
        parallel_tool_calls: bool = True,
    ):
        # Once SSE headers have been sent, request-local parser failures must be
        # reported in the stream. Cancellation still propagates to engine cleanup.
        chunks = self._stream_chat_chunks(
            request_id, prompt, config, model, output_parser_spec=output_parser_spec,
            parallel_tool_calls=parallel_tool_calls,
        )
        try:
            async with contextlib.aclosing(chunks):
                async for chunk in chunks:
                    yield chunk
        except ValueError as exc:
            error = {"error": {"message": str(exc), "type": "invalid_model_output", "code": 400}}
            yield f"data: {json.dumps(error)}\n\n"
            yield "data: [DONE]\n\n"

    async def _stream_chat_chunks(
        self,
        request_id: str,
        prompt: str,
        config: GenerateConfig,
        model: str,
        *,
        output_parser_spec: OutputParserSpec | None,
        parallel_tool_calls: bool,
    ):
        with profile_span("http.stream_chat_completion", cat="request", args={"request_id": request_id}):
            prev_text = ""
            prev_reasoning = ""
            outputs = self.engine.add_request(
                request_id,
                prompt,
                config,
                output_parser_spec=output_parser_spec,
            )
            async with contextlib.aclosing(outputs):
                async for output in outputs:
                    if output_parser_spec is not None:
                        delta = output.text_delta
                        reasoning_delta = output.reasoning_delta
                    else:
                        delta = output.text[len(prev_text):] if output.text else ""
                        prev_text = output.text or prev_text
                        reasoning_delta = output.reasoning[len(prev_reasoning):] if output.reasoning else ""
                        prev_reasoning = output.reasoning or prev_reasoning
                    finish_reason = self._chat_finish_reason(output) if output.finished else None
                    tool_deltas = [
                        self._tool_delta(item) for item in output.tool_call_deltas
                        if parallel_tool_calls or item.index == 0
                    ]

                    chunk = ChatCompletionResponse(
                        id=request_id,
                        object="chat.completion.chunk",
                        created=int(time.time()),
                        model=model,
                        choices=[ChatCompletionChoice(
                            delta=ChatDelta(
                                role="assistant",
                                content=delta or (None if tool_deltas else ""),
                                reasoning=reasoning_delta or None,
                                tool_calls=tool_deltas or None,
                            ),
                            finish_reason=finish_reason,
                        )],
                    )
                    yield f"data: {json.dumps(chunk.model_dump())}\n\n"

                    if output.finished:
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

    @staticmethod
    def _tool_delta(delta: ToolCallDelta) -> DeltaToolCall:
        return DeltaToolCall(
            index=delta.index, id=delta.id, type="function" if delta.id else None,
            function=DeltaFunctionCall(name=delta.name, arguments=delta.arguments or None),
        )

    @classmethod
    def _chat_finish_reason(cls, output: TokenOutput) -> str:
        if (
            output.finish_reason in ("FINISHED_EOS", "FINISHED_STOP")
            and output.tool_calls and all(call.complete for call in output.tool_calls)
        ):
            return "tool_calls"
        return cls._map_finish_reason(output.finish_reason)

    def _apply_chat_template(
        self,
        messages: list[ChatMessage],
        chat_template_kwargs: dict | None = None,
        *,
        reasoning_effort: str | None = None,
        tools: list[ChatTool] | None = None,
    ) -> str:
        """Apply the model's official chat template, forwarding chat_template_kwargs.

        ``chat_template_kwargs`` (e.g. ``{"enable_thinking": False}`` for Qwen3) is
        passed straight through to ``apply_chat_template``, mirroring vLLM so clients
        control thinking mode per request.
        """
        hf_messages = [m.model_dump(exclude_none=True) for m in messages]
        for original, message in zip(messages, hf_messages):
            message["content"] = original.content
        kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
        kwargs.update(self._chat_kwargs(chat_template_kwargs, reasoning_effort))
        if tools:
            kwargs["tools"] = [tool.model_dump(exclude_none=True) for tool in tools]
        kwargs["tokenize"] = False
        kwargs["add_generation_prompt"] = True
        return self.engine.tokenizer.apply_chat_template(hf_messages, **kwargs)

    @staticmethod
    def _chat_kwargs(chat_template_kwargs: dict | None, reasoning_effort: str | None) -> dict:
        kwargs = dict(chat_template_kwargs or {})
        if reasoning_effort is not None:
            if reasoning_effort == "none":
                kwargs["enable_thinking"] = False
                kwargs["thinking"] = False
            else:
                kwargs.setdefault("enable_thinking", True)
            kwargs["reasoning_effort"] = reasoning_effort
        return kwargs

    @staticmethod
    def _validate_chat_request(request: ChatCompletionRequest) -> str:
        if "tools" in (request.chat_template_kwargs or {}):
            raise ValueError("tools must be supplied as a top-level request field")
        choice = request.tool_choice
        if choice is None:
            choice = "auto" if request.tools else "none"
        if choice not in ("none", "auto"):
            raise ValueError("only tool_choice 'auto' and 'none' are supported; constrained tool choice is unavailable")
        if choice == "auto" and not request.tools:
            raise ValueError("tool_choice 'auto' requires non-empty tools")
        names = []
        for tool in request.tools or ():
            if tool.function.strict:
                raise ValueError("strict tool schemas require constrained decoding, which is not supported")
            names.append(tool.function.name)
        if len(set(names)) != len(names):
            raise ValueError("tool function names must be unique")
        for message in request.messages:
            calls = message.tool_calls or ()
            if calls and message.role != "assistant":
                raise ValueError("only assistant messages can contain tool_calls")
            if message.content is None and not (message.role == "assistant" and calls):
                raise ValueError("message content must be text, or null for an assistant tool call")
            if message.role == "tool" and not message.tool_call_id:
                raise ValueError("tool messages require tool_call_id")
            if len({call.id for call in calls}) != len(calls):
                raise ValueError("assistant tool call IDs must be unique")
            for call in calls:
                if not isinstance(json.loads(call.function.arguments), dict):
                    raise ValueError("tool call arguments must encode an object")
        return choice

    def _output_parser_spec(
        self,
        request: ChatCompletionRequest,
    ) -> OutputParserSpec | None:
        """Freeze model-output semantics before generation starts."""
        tool_choice = self._validate_chat_request(request)
        parser_id = getattr(self.engine.tokenizer, "output_parser_id", None)
        has_tool_history = any(m.tool_calls or m.role == "tool" for m in request.messages)
        if (request.tools or has_tool_history) and not supports_tool_calls(parser_id):
            raise ValueError("the model has no tool-call parser")
        if not parser_id:
            return None
        kwargs = self._chat_kwargs(request.chat_template_kwargs, request.reasoning_effort)
        thinking = bool(
            kwargs.get("thinking", False) or kwargs.get("enable_thinking", False)
        ) and kwargs.get("reasoning_effort") != "none"
        return OutputParserSpec(
            parser_id=str(parser_id),
            initial_state="reasoning" if thinking else "content",
            include_reasoning=request.include_reasoning,
            tool_choice=tool_choice,
            tool_names=tuple(tool.function.name for tool in request.tools or ()),
        )

    @staticmethod
    def _map_finish_reason(reason: str) -> str:
        mapping = {
            "FINISHED_EOS": "stop",
            "FINISHED_LENGTH": "length",
            "FINISHED_STOP": "stop",
            "FINISHED_ABORTED": "aborted",
            "error": "error",
        }
        return mapping.get(reason, "stop")


def create_serving_app(
    async_engine: AsyncLLMEngine,
    model_id: str,
    generate_config: GenerateConfig,
) -> FastAPI:
    server = ServingServer(async_engine, model_id, generate_config)
    return server.app
