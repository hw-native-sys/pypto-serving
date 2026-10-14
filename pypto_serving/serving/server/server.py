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
import importlib.util
import logging
import uuid
from pypto_serving.config.types import GenerateConfig
from pypto_serving.model.model_family import read_model_config
from pypto_serving.serving.constraints import ConstraintSpec
from pypto_serving.serving.constraints.provider import XGrammarProvider
from pypto_serving.serving.engine.async_engine import AsyncLLMEngine
from pypto_serving.serving.reasoning import OutputParserSpec, supports_tool_calls
from pypto_serving.tools.profile import (
    get_profiler,
    merge_profile,
    profile_span,
    start_profile as start_sa_profile,
    stop_profile as stop_sa_profile,
)

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, Response, StreamingResponse
    from .api_types import (
        ChatCompletionRequest,
        ChatDelta as ChatDelta,
        ChatMessage,
        ChatTool,
        CompletionRequest,
        DeltaFunctionCall as DeltaFunctionCall,
        DeltaToolCall as DeltaToolCall,
        FunctionCall as FunctionCall,
        FunctionDefinition as FunctionDefinition,
        ReasoningEffort as ReasoningEffort,
        ToolCall as ToolCall,
        validate_chat_request,
    )
    from .responses import output_response
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
        self._constraint_preflight_lock = asyncio.Lock()
        self._constraint_preflight_provider: XGrammarProvider | None = None
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
        output_parser_spec = self._output_parser_spec(request)
        constraint_spec = self._constraint_spec(request, output_parser_spec)
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
        return prompt, None, config, output_parser_spec, constraint_spec

    def resolve_prompt_tokens(self, prompt, tokens):
        return self.engine.resolve_prompt_tokens(prompt, tokens)

    async def _completions(self, request: CompletionRequest) -> StreamingResponse | JSONResponse:
        request_id = f"cmpl-{uuid.uuid4().hex[:8]}"
        prompt, tokens, config, _ = self.prepare_completion(request)
        return await output_response(
            self.engine.add_request(request_id, prompt, config, prompt_token_ids=tokens),
            request_id, request.model or self.model_id, kind="completion", stream=request.stream,
        )

    async def _chat_completions(self, request: ChatCompletionRequest) -> StreamingResponse | JSONResponse:
        request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        prompt, _, config, parser, constraint = self.prepare_chat(request)
        if constraint is not None:
            await self._preflight_constraint(constraint)
        outputs = self.engine.add_request(
            request_id, prompt, config, output_parser_spec=parser,
            **({"constraint_spec": constraint} if constraint else {}),
        )
        return await output_response(
            outputs, request_id, request.model or self.model_id, kind="chat", stream=request.stream,
            parsed=parser is not None, parallel_tool_calls=request.parallel_tool_calls,
        )

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
    def _validate_chat_request(request: ChatCompletionRequest) -> str | dict:
        return validate_chat_request(request)

    def _constraint_spec(
        self, request: ChatCompletionRequest, parser_spec: OutputParserSpec | None,
    ) -> ConstraintSpec | None:
        """Enable structural generation only for vLLM-compatible tool-choice cases."""
        choice = self._validate_chat_request(request)
        if choice == "none" or (
            choice == "auto" and not any(tool.function.strict for tool in request.tools or ())
        ):
            return None
        if parser_spec is None or parser_spec.parser_id != "deepseek_v4":
            raise ValueError("the model has no DeepSeek V4 structural-tool format")
        engine_config = getattr(self.engine, "config", None)
        if getattr(engine_config, "executor_cls", None) != "PyptoDeepSeekV4DSparkExecutor":
            raise ValueError("constrained tool generation currently requires DeepSeek V4 DSpark")
        if importlib.util.find_spec("xgrammar") is None:
            raise ValueError("xgrammar is required for constrained tool generation")
        return ConstraintSpec(
            provider_id="xgrammar",
            format_id="deepseek_v4",
            tools=tuple(tool.model_dump(mode="json", exclude_none=True) for tool in request.tools or ()),
            tool_choice=choice,
            reasoning=parser_spec.initial_state == "reasoning",
            parallel_tool_calls=request.parallel_tool_calls,
        )

    async def _preflight_constraint(
        self, spec: ConstraintSpec, initial_token_ids: tuple[int, ...] = (),
    ) -> None:
        """Reject unsupported schemas before an SSE response or worker batch starts."""
        async with self._constraint_preflight_lock:
            try:
                await asyncio.to_thread(self._compile_constraint_for_preflight, spec, initial_token_ids)
            except (RuntimeError, ValueError) as exc:
                raise ValueError(f"invalid tool constraint: {exc}") from exc

    def _compile_constraint_for_preflight(
        self, spec: ConstraintSpec, initial_token_ids: tuple[int, ...] = (),
    ) -> None:
        if self._constraint_preflight_provider is None:
            model_vocab_size = int(read_model_config(self.engine.config.model_dir)["vocab_size"])
            self._constraint_preflight_provider = XGrammarProvider(
                self.engine.tokenizer, model_vocab_size
            )
        with profile_span("ServingServer.constraint_preflight", cat="constraints"):
            state = self._constraint_preflight_provider.compile(spec)
        try:
            if initial_token_ids:
                state.accept(initial_token_ids)
        finally:
            state.close()

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
            tool_choice="required" if tool_choice == "required" or isinstance(tool_choice, dict) else tool_choice,
            tool_names=tuple(tool.function.name for tool in request.tools or ()),
        )


def create_serving_app(
    async_engine: AsyncLLMEngine,
    model_id: str,
    generate_config: GenerateConfig,
    *,
    route_factory=None,
) -> FastAPI:
    server = ServingServer(async_engine, model_id, generate_config, route_factory=route_factory)
    return server.app
