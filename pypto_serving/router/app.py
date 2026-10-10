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
import secrets
from collections.abc import Sequence
from pathlib import Path
from typing import Callable

from pypto_serving.router.config import (
    DEFAULT_AFFINITY_SLACK,
    DEFAULT_CONNECT_TIMEOUT_SECONDS,
    DEFAULT_DRAIN_TIMEOUT_SECONDS,
    DEFAULT_HEALTH_INTERVAL_SECONDS,
    DEFAULT_LAUNCH_TIMEOUT_SECONDS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_SESSION_TTL_SECONDS,
    RouterConfig,
    load_fleet_file,
)
from pypto_serving.router.fleet import FleetManager, PoolExhausted
from pypto_serving.router.launcher import LaunchError
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
                 client_factory: Callable[[RouterConfig], object] = default_client_factory,
                 transport_factory=None,
                 state_path: Path | None = None,
                 admin_token: str | None = None) -> None:
        self.config = config
        self.sessions = SessionDirectory(config.session_ttl_seconds)
        self.registry = ReplicaRegistry(config, self.sessions)
        self.client = client_factory(config)
        self.proxy = ReplicaProxy(self.registry, self.client)
        self.health = HealthMonitor(config, self.registry, self.sessions, self.client)
        self._admin_token = admin_token
        fleet_kwargs = {"state_path": state_path, "probe": self._probe_replica}
        if transport_factory is not None:
            fleet_kwargs["transport_factory"] = transport_factory
        self.fleet = FleetManager(config, self.registry, **fleet_kwargs)
        self.app = FastAPI(title="PyPTO Serving Router", lifespan=self._lifespan)

        self.app.add_api_route("/health", self._health, methods=["GET"])
        self.app.add_api_route("/v1/models", self._list_models, methods=["GET"])
        for path in ("/v1/completions", "/v1/chat/completions"):
            self.app.add_api_route(
                path, self._make_handler(path), methods=["POST"], response_model=None
            )
        # Admin surface. These start processes on other machines, so they are
        # gated on --admin-token when one is configured.
        self.app.add_api_route("/replicas", self._list_replicas, methods=["GET"])
        self.app.add_api_route(
            "/replicas", self._add_replica, methods=["POST"], response_model=None
        )
        self.app.add_api_route(
            "/replicas/{name}", self._remove_replica, methods=["DELETE"], response_model=None
        )

    def _make_handler(self, path: str):
        async def handler(request: Request):
            return await self.proxy.forward(request, path)
        handler.__name__ = f"proxy_{path.strip('/').replace('/', '_')}"
        return handler

    @contextlib.asynccontextmanager
    async def _lifespan(self, _app: FastAPI):
        """Run the health poller, and own the replicas we launch.

        The shutdown half only runs on an orderly exit (SIGTERM/SIGINT through
        uvicorn). A crash never reaches it, which is exactly the intent: a
        replica that survives a router crash is minutes of model load saved,
        and the next run adopts it.
        """
        await self.health.start()
        try:
            await self.fleet.adopt_previous(self._probe_replica)
            await self.fleet.launch_initial()
            yield
        finally:
            await self.fleet.stop_all()
            await self.health.stop()
            close = getattr(self.client, "aclose", None)
            if close is not None:
                await close()

    async def _probe_replica(self, spec) -> bool:
        """Is something already serving at this address?"""
        try:
            response = await self.client.get(
                f"{spec.base_url}/health", timeout=self.config.connect_timeout_seconds
            )
        except Exception:  # noqa: BLE001 - unreachable is the answer, not an error
            return False
        return response.status_code == 200

    def _authorized(self, request: Request) -> bool:
        if self._admin_token is None:
            return True
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        # Constant time: this is the only thing standing in front of a remote
        # process launch.
        return scheme.lower() == "bearer" and secrets.compare_digest(token, self._admin_token)

    @staticmethod
    def _unauthorized() -> JSONResponse:
        return JSONResponse(
            status_code=401,
            content={"object": "error", "message": "admin token required"},
        )

    def _replica_view(self, state) -> dict:
        return {
            "name": state.name,
            "url": state.spec.base_url,
            "ready": state.ready,
            "draining": state.draining,
            "owned": state.owned,
            "outstanding": state.outstanding,
            "routed": state.routed,
        }

    async def _list_replicas(self, request: Request) -> JSONResponse:
        """The fleet, and what capacity is left to grow into."""
        if not self._authorized(request):
            return self._unauthorized()
        return JSONResponse({
            "replicas": [self._replica_view(state) for state in self.registry.states],
            "capacity": self.fleet.capacity(),
        })

    async def _add_replica(self, request: Request) -> JSONResponse:
        """Launch one more replica on the next free declared device."""
        if not self._authorized(request):
            return self._unauthorized()
        try:
            spec = await self.fleet.launch_one()
        except PoolExhausted as exc:
            return JSONResponse(
                status_code=409,
                content={"object": "error", "message": str(exc),
                         "capacity": self.fleet.capacity()},
            )
        except LaunchError as exc:
            return JSONResponse(
                status_code=502,
                content={"object": "error", "message": f"launch failed: {exc}"},
            )
        # 202: the process is starting, and will not serve for minutes.
        return JSONResponse(
            status_code=202,
            content={
                "name": spec.name, "url": spec.base_url, "ready": False,
                "message": "launched; it becomes routable once its model is loaded",
            },
        )

    async def _remove_replica(self, name: str, request: Request) -> JSONResponse:
        """Drain a replica, then stop it and return its device to the pool."""
        if not self._authorized(request):
            return self._unauthorized()
        state = self.registry.state(name)
        if state is None:
            return JSONResponse(
                status_code=404,
                content={"object": "error", "message": f"no replica named {name!r}"},
            )
        if not state.owned:
            return JSONResponse(
                status_code=409,
                content={"object": "error",
                         "message": f"replica {name!r} was not launched by this router"},
            )
        await self.fleet.stop(name)
        return JSONResponse({"name": name, "stopped": True,
                             "capacity": self.fleet.capacity()})

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
                # Without an explicit timeout this inherits the client default
                # read=None, so a replica that accepts the connection and then
                # sends nothing would hang /v1/models forever.
                response = await self.client.get(
                    f"{state.spec.base_url}/v1/models",
                    timeout=self.config.connect_timeout_seconds,
                )
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
    transport_factory=None,
    state_path: Path | None = None,
    admin_token: str | None = None,
) -> FastAPI:
    """Build the router ASGI app.

    ``client_factory`` and ``transport_factory`` are the test seams: the first
    fakes the upstream replicas, the second fakes the thing that would ssh out
    and start one.
    """
    return ServingRouter(
        config,
        client_factory=client_factory,
        transport_factory=transport_factory,
        state_path=state_path,
        admin_token=admin_token,
    ).app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pypto-serving-router",
        description=(
            "Route OpenAI-compatible requests across independent pypto-serving "
            "replicas, keeping each conversation on the replica that holds its KV."
        ),
    )
    parser.add_argument("--replicas", required=True, metavar="PATH",
                        help='JSON file describing existing replicas and/or launchable hosts: '
                             '{"replicas": [...], "hosts": [...]}.')
    parser.add_argument("--initial-replicas", type=int, default=None,
                        help="Replicas to launch from the host pool at startup. Defaults to 1 "
                             "when the file declares hosts, 0 when it does not.")
    parser.add_argument("--max-replicas", type=int, default=None,
                        help="Cap the number of launched replicas below the declared pool. "
                             "On a shared cluster this is what stops the router taking every "
                             "free card.")
    parser.add_argument("--launch-timeout", type=float, default=DEFAULT_LAUNCH_TIMEOUT_SECONDS,
                        help="Seconds a launched replica may take to report ready before it is "
                             f"stopped and its device freed (default: {DEFAULT_LAUNCH_TIMEOUT_SECONDS:g}).")
    parser.add_argument("--drain-timeout", type=float, default=DEFAULT_DRAIN_TIMEOUT_SECONDS,
                        help="Seconds to wait for in-flight requests before stopping a replica "
                             f"(default: {DEFAULT_DRAIN_TIMEOUT_SECONDS:g}).")
    parser.add_argument("--state-file", default=None, metavar="PATH",
                        help="Where to record launched replicas so a restarted router can adopt "
                             "them instead of paying for a fresh model load.")
    parser.add_argument("--admin-token", default=None, metavar="TOKEN",
                        help="Require 'Authorization: Bearer TOKEN' on /replicas. Those routes "
                             "start processes on other machines; without a token they are open "
                             "to anyone who can reach the port.")
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
        replicas, hosts = load_fleet_file(args.replicas)
        # Launching one replica is the useful default, but only where there is
        # somewhere to launch it; a static-only file must not be forced to.
        initial = args.initial_replicas
        if initial is None:
            initial = 1 if hosts else 0
        config = RouterConfig(
            replicas=replicas,
            hosts=hosts,
            session_ttl_seconds=args.session_ttl,
            affinity_slack=args.affinity_slack,
            health_interval_seconds=args.health_interval,
            request_timeout_seconds=args.request_timeout,
            connect_timeout_seconds=args.connect_timeout,
            launch_timeout_seconds=args.launch_timeout,
            drain_timeout_seconds=args.drain_timeout,
            initial_replicas=initial,
            max_replicas=args.max_replicas,
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
        print(f"  Replica (external): {replica.name} -> {replica.base_url}")
    for host in config.hosts:
        where = "local" if host.is_local else host.ssh
        print(
            f"  Host: {host.name} [{where}] devices={list(host.devices)} "
            f"({host.devices_per_replica}/replica -> {host.replica_capacity} slot(s))"
        )
    print(f"  Pool: launching {config.initial_replicas} of {config.pool_ceiling} slot(s) at startup")
    print(f"  Session TTL: {config.session_ttl_seconds:g}s, affinity slack: {config.affinity_slack}")
    print("  Endpoints: /v1/completions, /v1/chat/completions, /v1/models, /health, /replicas")
    if config.hosts and args.admin_token is None:
        print("  WARNING: /replicas is unauthenticated; it can start processes on other hosts")

    app = create_router_app(
        config,
        state_path=Path(args.state_file) if args.state_file else None,
        admin_token=args.admin_token,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


__all__ = ["SESSION_HEADER", "ServingRouter", "build_parser", "create_router_app", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
