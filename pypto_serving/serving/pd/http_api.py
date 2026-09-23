# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Address-free HTTP contracts between the external Router and P/D nodes."""

from __future__ import annotations

import hashlib
from typing import TypeVar

import msgspec
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .protocol import (
    CapabilityWire,
    ContinuationMetadata,
    DecodeOutputWire,
    HandoffKey,
    PrefixMatchSpec,
)


MAX_INTERNAL_BODY_BYTES = 4 << 20


class NodeDescriptor(msgspec.Struct, frozen=True):
    node_id: str
    role: str
    run_id: str
    control_host: str
    control_port: int
    owner_generation: int
    endpoint_generation: int
    control_incarnation: int
    capabilities: CapabilityWire
    health: str


class CapacitySnapshot(msgspec.Struct, frozen=True):
    node_id: str
    role: str
    active_handoffs: int
    prepared_requests: int
    reservations: int
    quarantined_reservations: int
    snapshot_sequence: int
    active_limit: int = 1
    queued_handoffs: int = 0
    inflight_transfer_bytes: int = 0
    inflight_transfer_byte_limit: int = 0


class PrepareRequestHTTP(msgspec.Struct, frozen=True):
    request_id: str
    request_kind: str
    request_json: bytes


class PreparedRequest(msgspec.Struct, frozen=True):
    request_id: str
    prepared_request_id: str
    prepared_digest: str
    continuation: ContinuationMetadata
    expires_at_ns: int
    prefix_match_spec: PrefixMatchSpec | None = None


class ReservePlacementHTTP(msgspec.Struct, frozen=True):
    key: HandoffKey
    prepared_request_id: str
    prepared_digest: str
    prompt_token_count: int
    max_new_tokens: int
    layout_fingerprint: str
    prefill_node_id: str
    prefill_endpoint_generation: int
    prefix_match_spec: PrefixMatchSpec | None = None


class PlacementReservation(msgspec.Struct, frozen=True):
    key: HandoffKey
    prepared_request_id: str
    reservation_id: str
    partition: int
    block_ids_by_group: dict[str, tuple[int, ...]]
    prepared_digest: str
    reservation_capability: str
    decode_node_id: str
    decode_control_host: str
    decode_control_port: int
    decode_endpoint_generation: int
    prefix_hit_tokens: int = 0


class PlacementRejection(msgspec.Struct, frozen=True):
    key: HandoffKey
    decode_node_id: str
    decode_endpoint_generation: int
    reason: str
    retryable: bool


class ReservePlacementResult(msgspec.Struct, frozen=True):
    reservation: PlacementReservation | None = None
    rejection: PlacementRejection | None = None

    def __post_init__(self) -> None:
        if (self.reservation is None) == (self.rejection is None):
            raise ValueError("reserve result must contain exactly one outcome")


class AuthorizeRouteHTTP(msgspec.Struct, frozen=True):
    key: HandoffKey
    prepared_request_id: str
    prepared_digest: str
    reservation_id: str
    reservation_capability: str
    compatibility_digest: str
    prefill_node_id: str
    prefill_endpoint_generation: int
    decode_node_id: str
    decode_endpoint_generation: int


class ExecutePrefillHTTP(msgspec.Struct, frozen=True):
    key: HandoffKey
    prepared_request_id: str
    prepared_digest: str
    reservation_id: str
    partition: int
    block_ids_by_group: dict[str, tuple[int, ...]]
    reservation_capability: str
    compatibility_digest: str
    prefill_node_id: str
    prefill_endpoint_generation: int
    decode_node_id: str
    decode_control_host: str
    decode_control_port: int
    decode_endpoint_generation: int
    prefix_hit_tokens: int = 0


class PrefillHandoffResult(msgspec.Struct, frozen=True):
    key: HandoffKey
    reservation_id: str
    manifest_hash: str
    state: str


class HandoffHTTP(msgspec.Struct, frozen=True):
    key: HandoffKey


class AbortHandoffHTTP(msgspec.Struct, frozen=True):
    key: HandoffKey
    reason: str
    deterministic: bool = True


class DecodeStreamFrame(msgspec.Struct, frozen=True):
    event: str
    output: DecodeOutputWire | None = None
    state: str = ""
    error_code: str = ""


T = TypeVar("T")


def encode_json(value: object) -> bytes:
    wire = msgspec.json.encode(value)
    if not wire or len(wire) > MAX_INTERNAL_BODY_BYTES:
        raise ValueError("PD internal HTTP body exceeds the bounded size")
    return wire


def decode_json(wire: bytes, type_: type[T]) -> T:
    if not wire or len(wire) > MAX_INTERNAL_BODY_BYTES:
        raise ValueError("invalid PD internal HTTP body size")
    return msgspec.json.decode(wire, type=type_)


def capability_compatibility_digest(value: CapabilityWire) -> str:
    """Digest only fields that must match across P and D owner registries."""
    contract = (
        value.schema_version,
        value.adapter_id,
        value.contract_version,
        value.contract_digest,
        value.continuation_schema,
        value.model_revision,
        value.layout_fingerprint,
        value.topology,
        value.provider,
        value.logical_groups,
        value.physical_regions,
        value.prefix_cache_mode,
    )
    return hashlib.sha256(msgspec.msgpack.encode(contract)).hexdigest()


class PDHTTPRoutes:
    """Node control endpoints composed from public Serving request capabilities."""

    def __init__(
        self,
        app,
        service,
        config,
        *,
        prepare_completion,
        prepare_chat,
        resolve_prompt_tokens,
        start_profile,
        stop_profile,
        profiling_enabled=False,
    ):
        self.app = app
        self.service = service
        self.config = config
        self.prepare_completion = prepare_completion
        self.prepare_chat = prepare_chat
        self.resolve_prompt_tokens = resolve_prompt_tokens
        self.start_profile = start_profile
        self.stop_profile = stop_profile
        self.profiling_enabled = profiling_enabled
        self.app.add_api_route("/health", self._health, methods=["GET"])
        self._register_internal_pd_routes()
        self._register_exception_handlers()

    def _register_exception_handlers(self):
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
        if self.profiling_enabled:
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

        return Response(
            encode_json(self.service.descriptor()),
            media_type="application/json",
        )

    async def _pd_capacity(self, request: Request) -> Response:

        return Response(
            encode_json(self.service.capacity_snapshot()),
            media_type="application/json",
        )

    async def _pd_metrics(self, request: Request) -> JSONResponse:
        return JSONResponse(await self.service.metrics_snapshot())

    async def _pd_start_profile(self, request: Request) -> Response:
        return await self.start_profile()

    async def _pd_stop_profile(self, request: Request) -> Response:
        return await self.stop_profile()

    async def _pd_prepare(self, request: Request) -> Response:

        payload = decode_json(await request.body(), PrepareRequestHTTP)
        from pypto_serving.serving.server.server import CompletionRequest, ChatCompletionRequest

        if payload.request_kind == "completion":
            public = CompletionRequest.model_validate_json(payload.request_json)
            prompt, tokens, config, output_parser_spec = self.prepare_completion(public)
        elif payload.request_kind == "chat":
            public = ChatCompletionRequest.model_validate_json(payload.request_json)
            prompt, tokens, config, output_parser_spec = self.prepare_chat(public)
        else:
            raise ValueError("unsupported PD public request kind")
        prompt_token_ids = self.resolve_prompt_tokens(prompt, tokens)
        prepared = self.service.prepare_request(
            payload.request_id,
            prompt,
            config,
            prompt_token_ids,
            output_parser_spec=output_parser_spec,
        )
        return Response(encode_json(prepared), media_type="application/json")

    async def _pd_reserve(self, request: Request) -> Response:

        payload = decode_json(await request.body(), ReservePlacementHTTP)
        outcome = self.service.reserve_placement(payload)
        result = ReservePlacementResult(
            rejection=outcome if isinstance(outcome, PlacementRejection) else None,
            reservation=None if isinstance(outcome, PlacementRejection) else outcome,
        )
        return Response(encode_json(result), media_type="application/json")

    async def _pd_authorize(self, request: Request) -> Response:

        payload = decode_json(await request.body(), AuthorizeRouteHTTP)
        self.service.authorize_route(payload)
        return Response(encode_json({"status": "AUTHORIZED"}), media_type="application/json")

    async def _pd_execute(self, request: Request) -> Response:

        payload = decode_json(await request.body(), ExecutePrefillHTTP)
        result = await self.service.execute_prefill(payload)
        return Response(encode_json(result), media_type="application/json")

    async def _pd_await_decode(self, request: Request) -> StreamingResponse:

        payload = decode_json(await request.body(), HandoffHTTP)
        stream = self.service.open_decode_stream(payload.key)

        async def ndjson():
            async for frame in stream:
                yield encode_json(frame) + b"\n"

        return StreamingResponse(ndjson(), media_type="application/x-ndjson")

    async def _pd_query(self, request: Request) -> Response:

        payload = decode_json(await request.body(), HandoffHTTP)
        return Response(
            encode_json(self.service.query_handoff(payload.key)),
            media_type="application/json",
        )

    async def _pd_abort(self, request: Request) -> Response:

        payload = decode_json(await request.body(), AbortHandoffHTTP)
        status = await self.service.abort_handoff(
            payload.key,
            payload.reason,
            deterministic=payload.deterministic,
        )
        return Response(encode_json(status), media_type="application/json")

    async def _health(self) -> JSONResponse:
        if self.service.health_error:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "error",
                    "reason": "PD_RECOVERY_REQUIRED",
                },
            )
        content = {"status": "ok"}
        pd_config = self.config
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
