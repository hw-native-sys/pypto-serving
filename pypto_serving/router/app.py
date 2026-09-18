# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The router application and its command line.

Deliberately separate from ``pypto_serving.cli.main``: that module imports the
engine configuration types, which pull torch at import time. The router runs on
hosts with no NPU stack, so it must not share that import graph.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
from collections.abc import Sequence
from typing import Callable

from pypto_serving.router.config import (
    DEFAULT_AFFINITY_SLACK,
    DEFAULT_CONNECT_TIMEOUT_SECONDS,
    DEFAULT_HEALTH_INTERVAL_SECONDS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_SESSION_TTL_SECONDS,
    RouterConfig,
    load_replica_table,
)
from pypto_serving.router.proxy import SESSION_HEADER, ReplicaProxy
from pypto_serving.router.routing import HealthMonitor, ReplicaRegistry, SessionDirectory

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
except ImportError as e:
    raise ImportError(
        "The serving router requires fastapi. Install with: pip install fastapi uvicorn"
    ) from e


def default_client_factory(config: RouterConfig):
    """Build the upstream HTTP client (imported lazily so tests need no httpx)."""
    try:
        import httpx
    except ImportError as e:
        raise ImportError("The serving router requires httpx. Install with: pip install httpx") from e

    # No read timeout: decode is slow by design and a token can be seconds away.
    timeout = httpx.Timeout(
        config.request_timeout_seconds, connect=config.connect_timeout_seconds, read=None
    )
    return httpx.AsyncClient(timeout=timeout, follow_redirects=False)


class ServingRouter:
    """Session-affine router over a flat table of serving replicas."""

    def __init__(self, config: RouterConfig, *,
                 client_factory: Callable[[RouterConfig], object] = default_client_factory) -> None:
        self.config = config
        self.sessions = SessionDirectory(config.session_ttl_seconds)
        self.registry = ReplicaRegistry(config, self.sessions)
        self.client = client_factory(config)
        self.proxy = ReplicaProxy(self.registry, self.client)
        self.health = HealthMonitor(config, self.registry, self.sessions, self.client)
        self.app = FastAPI(title="PyPTO Serving Router", lifespan=self._lifespan)

        self.app.add_api_route("/health", self._health, methods=["GET"])
        self.app.add_api_route("/v1/models", self._list_models, methods=["GET"])
        for path in ("/v1/completions", "/v1/chat/completions"):
            self.app.add_api_route(
                path, self._make_handler(path), methods=["POST"], response_model=None
            )

    def _make_handler(self, path: str):
        async def handler(request: Request):
            return await self.proxy.forward(request, path)
        handler.__name__ = f"proxy_{path.strip('/').replace('/', '_')}"
        return handler

    @contextlib.asynccontextmanager
    async def _lifespan(self, _app: FastAPI):
        """Run the health poller for the lifetime of the server."""
        await self.health.start()
        try:
            yield
        finally:
            await self.health.stop()
            close = getattr(self.client, "aclose", None)
            if close is not None:
                await close()

    async def _health(self) -> JSONResponse:
        """The router's own view of the replica table."""
        ready = self.registry.ready_count()
        payload = {
            "status": "ok" if ready else "no_replicas",
            "replicas_ready": ready,
            "replicas_total": len(self.registry.states),
            "sessions": len(self.sessions),
            "routed": self.registry.total_routed(),
            "affinity_hits": self.registry.affinity_hits,
            "rejected": self.registry.rejected,
            "replicas": [
                {"name": s.name, "url": s.spec.base_url, "ready": s.ready,
                 "outstanding": s.outstanding, "routed": s.routed}
                for s in self.registry.states
            ],
        }
        return JSONResponse(payload, status_code=200 if ready else 503)

    async def _list_models(self) -> JSONResponse:
        """Answer from any routable replica; every replica serves one model."""
        for state in self.registry.states:
            if not state.ready:
                continue
            try:
                response = await self.client.get(f"{state.spec.base_url}/v1/models")
            except Exception:  # noqa: BLE001 - try the next replica
                logger.debug("listing models via %s failed", state.name, exc_info=True)
                continue
            if response.status_code == 200:
                return JSONResponse(response.json())
        return JSONResponse(
            status_code=503,
            content={"object": "error", "message": "no serving replica is currently routable"},
        )


def create_router_app(
    config: RouterConfig,
    *,
    client_factory: Callable[[RouterConfig], object] = default_client_factory,
) -> FastAPI:
    """Build the router ASGI app.

    ``client_factory`` mirrors ``AsyncLLMEngine``'s ``core_factory`` seam: tests
    inject a fake upstream instead of standing up real replicas.
    """
    return ServingRouter(config, client_factory=client_factory).app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pypto-serving-router",
        description=(
            "Route OpenAI-compatible requests across independent pypto-serving "
            "replicas, keeping each conversation on the replica that holds its KV."
        ),
    )
    parser.add_argument("--replicas", required=True, metavar="PATH",
                        help='JSON file: {"replicas": [{"host": "h", "port": 8000}]}.')
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0).")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000).")
    parser.add_argument("--session-ttl", type=float, default=DEFAULT_SESSION_TTL_SECONDS,
                        help="Seconds a conversation keeps its replica pin after its last "
                             f"request (default: {DEFAULT_SESSION_TTL_SECONDS:g}).")
    parser.add_argument("--affinity-slack", type=int, default=DEFAULT_AFFINITY_SLACK,
                        help="Extra outstanding requests a pinned replica may carry before a "
                             f"session is re-routed (default: {DEFAULT_AFFINITY_SLACK}; "
                             "0 disables affinity).")
    parser.add_argument("--health-interval", type=float, default=DEFAULT_HEALTH_INTERVAL_SECONDS,
                        help=f"Seconds between health probes (default: {DEFAULT_HEALTH_INTERVAL_SECONDS:g}).")
    parser.add_argument("--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
                        help=f"Overall timeout per proxied request (default: {DEFAULT_REQUEST_TIMEOUT_SECONDS:g}s).")
    parser.add_argument("--connect-timeout", type=float, default=DEFAULT_CONNECT_TIMEOUT_SECONDS,
                        help=f"Connect timeout per replica (default: {DEFAULT_CONNECT_TIMEOUT_SECONDS:g}s).")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        config = RouterConfig(
            replicas=load_replica_table(args.replicas),
            session_ttl_seconds=args.session_ttl,
            affinity_slack=args.affinity_slack,
            health_interval_seconds=args.health_interval,
            request_timeout_seconds=args.request_timeout,
            connect_timeout_seconds=args.connect_timeout,
        )
    except ValueError as exc:
        print(f"error: {exc}")
        return 1

    try:
        import uvicorn
    except ImportError as e:
        raise ImportError("The router requires uvicorn. Install with: pip install uvicorn") from e

    print(f"Starting PyPTO serving router on {args.host}:{args.port}")
    for replica in config.replicas:
        print(f"  Replica: {replica.name} -> {replica.base_url}")
    print(f"  Session TTL: {config.session_ttl_seconds:g}s, affinity slack: {config.affinity_slack}")
    print("  Endpoints: /v1/completions, /v1/chat/completions, /v1/models, /health")

    uvicorn.run(create_router_app(config), host=args.host, port=args.port, log_level="info")
    return 0


__all__ = ["SESSION_HEADER", "ServingRouter", "build_parser", "create_router_app", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
