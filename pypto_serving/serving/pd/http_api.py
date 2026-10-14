# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Generation endpoints and the small PD operational API."""

from __future__ import annotations

import asyncio
import time
from typing import TypeVar

import msgspec
from fastapi import Request
from fastapi.responses import JSONResponse, Response

from pypto_serving.serving.server.responses import output_response

from .config import PDRole
from .admission import PDBackpressureError
from .generation import MAX_NODE_BODY_BYTES, parse_generation
from .protocol import HandoffKey, HandoffStatus, PrefillResult


class HandoffHTTP(msgspec.Struct, frozen=True):
    key: HandoffKey


class AbortHandoffHTTP(msgspec.Struct, frozen=True):
    key: HandoffKey
    reason: str
    deterministic: bool = True


T = TypeVar("T")


def encode_json(value: object) -> bytes:
    wire = msgspec.json.encode(value)
    if not wire or len(wire) > MAX_NODE_BODY_BYTES:
        raise ValueError("PD HTTP body exceeds the bounded size")
    return wire


def decode_json(wire: bytes, type_: type[T]) -> T:
    if not wire or len(wire) > MAX_NODE_BODY_BYTES:
        raise ValueError("invalid PD HTTP body size")
    return msgspec.json.decode(wire, type=type_)


class PDHTTPRoutes:
    """Compose node generation from existing Serving preparation/formatting."""

    def __init__(
        self, app, service, config, *, model_id, prepare_completion, prepare_chat,
        preflight_constraint, resolve_prompt_tokens, start_profile, stop_profile,
        profiling_enabled=False,
    ):
        self.app, self.service, self.config = app, service, config
        self.model_id = model_id
        self.prepare_completion, self.prepare_chat = prepare_completion, prepare_chat
        self.preflight_constraint = preflight_constraint
        service.preflight_constraint = preflight_constraint
        self.resolve_prompt_tokens = resolve_prompt_tokens
        self.start_profile, self.stop_profile = start_profile, stop_profile
        @app.exception_handler(PDBackpressureError)
        async def backpressure(request, exc):
            return JSONResponse(status_code=503, headers={"Retry-After": "1"},
                                content={"object": "error", "message": str(exc)})
        @app.exception_handler(RuntimeError)
        async def unavailable(request, exc):
            return JSONResponse(status_code=503, content={"object": "error", "message": str(exc)})
        app.add_api_route("/health", self._health, methods=["GET"])
        app.add_api_route("/v1/models", self._models, methods=["GET"])
        app.add_api_route("/v1/completions", self._completion, methods=["POST"])
        app.add_api_route("/v1/chat/completions", self._chat, methods=["POST"])
        for path, handler, method in (
            ("descriptor", self._descriptor, "GET"),
            ("capacity", self._capacity, "GET"),
            ("metrics", self._metrics, "GET"),
            ("query", self._query, "POST"),
            ("abort", self._abort, "POST"),
        ):
            app.add_api_route("/internal/pd/" + path, handler, methods=[method])
        if profiling_enabled:
            app.add_api_route("/internal/pd/start-profile", self._start_profile, methods=["POST"])
            app.add_api_route("/internal/pd/stop-profile", self._stop_profile, methods=["POST"])

    async def _completion(self, request: Request):
        return await self._generate(request, "completion")

    async def _chat(self, request: Request):
        return await self._generate(request, "chat")

    async def _generate(self, request: Request, kind: str):
        public, context = parse_generation(await request.body(), kind)
        model = public.model or self.model_id
        context.validate_node(self.config)
        try:
            local = self.service.register_generation(context)
        except PDBackpressureError as exc:
            if self.config.role is PDRole.DECODE:
                raise
            return self._prefill_response(PrefillResult(
                context.key, context.request_fingerprint, "NOT_STARTED", type(exc).__name__,
            ), kind, model, status_code=503)
        if self.config.role is PDRole.PREFILL:
            # Every exception in this block is before scheduling or native work.
            try:
                if kind == "completion":
                    prompt, tokens, config, parser = self.prepare_completion(public)
                    constraint = None
                else:
                    prompt, tokens, config, parser, constraint = self.prepare_chat(public)
                    if constraint is not None:
                        await self.preflight_constraint(constraint)
                tokens = self.resolve_prompt_tokens(prompt, tokens)
                prepared = self.service.prepare_input(
                    prompt, config, tokens, output_parser_spec=parser, constraint_spec=constraint,
                )
            except Exception as exc:
                result = PrefillResult(context.key, context.request_fingerprint, "NOT_STARTED",
                                       type(exc).__name__, False, str(exc)[:512])
                self.service._finish_request(context.key, HandoffStatus(
                    context.key, "FAILED", error_code=result.error_code,
                ))
                return self._prefill_response(result, kind, model, status_code=400)
            result = await self.service.prefill(context, prepared)
            return self._prefill_response(result, kind, model)
        try:
            continuation = await asyncio.wait_for(
                asyncio.shield(local.accepted), timeout=self.config.request_timeout_seconds,
            )
            if isinstance(continuation, BaseException):
                raise continuation
            return await output_response(
                self.service.decode_outputs(local), context.key.request_id, model,
                kind=kind, stream=public.stream,
                parsed=continuation.output_parser_spec is not None,
                parallel_tool_calls=getattr(public, "parallel_tool_calls", True),
            )
        except BaseException:
            if context.key in self.service._requests:
                status = self.service.query_handoff(context.key)
                await self.service.abort_handoff(
                    context.key, "DECODE_RESPONSE_FAILED",
                    deterministic=status.state in ("REGISTERED", "RESERVED", "READY", "IN_USE", "ABORTED"),
                )
            raise

    @staticmethod
    def _prefill_response(result: PrefillResult, kind: str, model: str, *, status_code=200):
        return JSONResponse(status_code=status_code, content={
            "id": result.key.request_id,
            "object": "chat.completion" if kind == "chat" else "text_completion",
            "created": int(time.time()), "model": model, "choices": [],
            "pd_transfer": msgspec.to_builtins(result),
        })

    async def _descriptor(self, request: Request):
        return Response(encode_json(self.service.descriptor()), media_type="application/json")

    async def _capacity(self, request: Request):
        return Response(encode_json(self.service.capacity_snapshot()), media_type="application/json")

    async def _metrics(self, request: Request):
        return JSONResponse(await self.service.metrics_snapshot())

    async def _query(self, request: Request):
        payload = decode_json(await request.body(), HandoffHTTP)
        return Response(encode_json(self.service.query_handoff(payload.key)), media_type="application/json")

    async def _abort(self, request: Request):
        payload = decode_json(await request.body(), AbortHandoffHTTP)
        status = await self.service.abort_handoff(payload.key, payload.reason, deterministic=payload.deterministic)
        return Response(encode_json(status), media_type="application/json")

    async def _start_profile(self, request: Request):
        return await self.start_profile()

    async def _stop_profile(self, request: Request):
        return await self.stop_profile()

    async def _models(self):
        model = self.model_id
        return JSONResponse({"object": "list", "data": [
            {"id": model, "object": "model", "owned_by": "pypto"},
        ]})

    async def _health(self):
        if self.service.health_error:
            return JSONResponse(status_code=503, content={"status": "error", "reason": "PD_RECOVERY_REQUIRED"})
        return JSONResponse({
            "status": "ok", "run_id": self.config.run_id, "node_id": self.config.node_id,
            "role": self.config.role.value, "generation": self.config.generation,
            "control_incarnation": self.config.control_incarnation,
        })
