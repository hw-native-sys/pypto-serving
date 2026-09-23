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
from pypto_serving.config.types import GenerateConfig
from pypto_serving.serving.engine.async_engine import AsyncLLMEngine, TokenOutput
from pypto_serving.observability.tokens import token_ids_sha256
from pypto_serving.serving.reasoning import OutputParserSpec, supports_tool_calls
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
    from .api_types import (
        ChatCompletionChoice,
        ChatCompletionRequest,
        ChatCompletionResponse,
        ChatDelta as ChatDelta,
        ChatMessage,
        ChatTool,
        CompletionChoice,
        CompletionRequest,
        CompletionResponse,
        DeltaFunctionCall as DeltaFunctionCall,
        DeltaToolCall as DeltaToolCall,
        FunctionCall as FunctionCall,
        FunctionDefinition as FunctionDefinition,
        ReasoningEffort as ReasoningEffort,
        ResponseUsage,
        ToolCall as ToolCall,
        validate_chat_request,
    )
    from .chat_format import chat_delta, chat_finish_reason, chat_message, map_finish_reason
except ImportError as e:
    raise ImportError(
        "Serving requires fastapi and pydantic. Install with: pip install fastapi uvicorn sse-starlette pydantic"
    ) from e


# --- Server ---

class ServingServer:
    def __init__(
        self,
        async_engine: AsyncLLMEngine,
        model_id: str,
        generate_config: GenerateConfig,
        *,
        route_factory=None,
    ) -> None:
        self.engine = async_engine
        self.model_id = model_id
        # Server-wide generate defaults. Fields the HTTP request omits fall
        # back to this config; explicit per-request fields still win.
        self.generate_config = generate_config
        self.app = FastAPI(title="PyPTO Serving")
        self._profile_lock = asyncio.Lock()
        self._register_exception_handlers()
        if route_factory is None:
            self._register_routes()
        else:
            route_factory(self)

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
        if getattr(self.engine, "metrics", None) is not None:
            self.app.add_api_route("/metrics", self._metrics, methods=["GET"])
            self.app.add_api_route("/metrics/json", self._metrics_json, methods=["GET"])
        if get_profiler(initially_active=False).enabled:
            self.app.add_api_route("/start_profile", self.start_profile, methods=["POST"])
            self.app.add_api_route("/stop_profile", self.stop_profile, methods=["POST"])

    async def _health(self) -> JSONResponse:
        return JSONResponse({"status": "ok"})

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

    async def start_profile(self) -> Response:
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

    async def stop_profile(self) -> Response:
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

    def prepare_completion(self, request: CompletionRequest):
        """Prepare public completion semantics without starting generation."""
        prompt, tokens = self._completion_prompt(request.prompt)
        config = dataclasses.replace(self._resolve_generate_config(request), ignore_eos=True)
        return prompt, tokens, config, None

    def prepare_chat(self, request: ChatCompletionRequest):
        """Share chat templating, defaults and parser selection across entry points."""
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
        output_parser_spec = self._output_parser_spec(request)

        return prompt, None, config, output_parser_spec

    def resolve_prompt_tokens(self, prompt, tokens):
        return self.engine.resolve_prompt_tokens(prompt, tokens)

    async def _completions(self, request: CompletionRequest) -> StreamingResponse | JSONResponse:
        request_id = f"cmpl-{uuid.uuid4().hex[:8]}"
        prompt, prompt_token_ids, config, _ = self.prepare_completion(request)

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
        prompt, _, config, output_parser_spec = self.prepare_chat(request)

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
                    tool_calls = output.tool_calls
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
                    message=chat_message(
                        full_text,
                        full_reasoning,
                        tool_calls,
                        parallel_tool_calls=request.parallel_tool_calls,
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
                    chunk = ChatCompletionResponse(
                        id=request_id,
                        object="chat.completion.chunk",
                        created=int(time.time()),
                        model=model,
                        choices=[ChatCompletionChoice(
                            delta=chat_delta(
                                delta,
                                reasoning_delta,
                                output.tool_call_deltas,
                                parallel_tool_calls=parallel_tool_calls,
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
    def _chat_finish_reason(output: TokenOutput) -> str:
        return chat_finish_reason(output)

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
        return validate_chat_request(request)

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
        return map_finish_reason(reason)


def create_serving_app(
    async_engine: AsyncLLMEngine,
    model_id: str,
    generate_config: GenerateConfig,
    *,
    route_factory=None,
) -> FastAPI:
    server = ServingServer(async_engine, model_id, generate_config, route_factory=route_factory)
    return server.app
