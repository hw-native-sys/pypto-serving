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
import os
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
    DEFAULT_REPLACE_AFTER_SECONDS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_SESSION_TTL_SECONDS,
    RouterConfig,
    load_router_config,
)
from pypto_serving.router.fleet import FleetManager, PoolExhausted
from pypto_serving.router.launcher import LaunchError
from pypto_serving.router.proxy import SESSION_HEADER, ReplicaProxy
from pypto_serving.router.routing import (
    HealthMonitor,
    ReplicaRegistry,
    SessionDirectory,
    is_router_side_failure,
)

logger = logging.getLogger(__name__)

# Preferred over --admin-token: argv is world-readable on the router host.
ADMIN_TOKEN_ENV = "PYPTO_ROUTER_ADMIN_TOKEN"


class ProbeNotSent(RuntimeError):
    """A probe that failed in the router, before it could reach the replica.

    Separate from a silent endpoint on purpose: silence is evidence about the
    replica, this is evidence about us.
    """

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
    #
    # pool is set explicitly because it otherwise inherits request_timeout: a
    # request with no free connection would then wait an hour for one rather
    # than failing, which is not a wait any client is asking for.
    timeout = httpx.Timeout(
        config.request_timeout_seconds,
        connect=config.connect_timeout_seconds,
        read=None,
        pool=config.connect_timeout_seconds,
    )
    # Explicit, rather than httpx's default of 100. This is the ceiling on
    # concurrent upstream requests, and a deployment whose clients hold long
    # streams -- a coding agent does -- reaches it with far fewer users than
    # the number suggests.
    max_connections = config.max_connections
    if config.is_pd:
        capacity = (config.pd_options or {}).get("capacity", 8)
        p = sum(r.role == "prefill" for r in config.replicas)
        d = sum(r.role == "decode" for r in config.replicas)
        max_connections = capacity * (d * (p + d + 3) + p) + p + d
    limits = httpx.Limits(max_connections=max_connections,
                          max_keepalive_connections=config.max_keepalive_connections)
    return httpx.AsyncClient(timeout=timeout, limits=limits, follow_redirects=False)


class ServingRouter:
    """Common HTTP entry point for complete replicas or coordinated P/D pools."""

    def __init__(self, config: RouterConfig, *,
                 client_factory: Callable[[RouterConfig], object] = default_client_factory,
                 transport_factory=None,
                 state_path: Path | None = None,
                 admin_token: str | None = None) -> None:
        self.config = config
        self.sessions = SessionDirectory(config.session_ttl_seconds)
        self.registry = ReplicaRegistry(config, self.sessions)
        pd_config = None
        if config.is_pd:
            from pypto_serving.pd.config import Config
            from pypto_serving.pd.coordinator import PDCoordinator

            pd_config = Config.from_router(config)
            if state_path is not None:
                raise ValueError("PD replicas are externally managed; --state-file requires ordinary hosts")
        self.client = client_factory(config)
        # A second client, so probes have a connection pool of their own. They
        # share the proxy's otherwise, and a coding agent's long streams can
        # hold every slot in it: the probes then fail for want of a connection
        # and the fleet reads that as replicas dying. One pool makes router
        # congestion indistinguishable from a dead backend, for every replica
        # at once, because the pool is what they have in common.
        self.probe_client = client_factory(config)
        self.proxy = (PDCoordinator(pd_config, self.registry, self.client, self.probe_client) if config.is_pd
                      else ReplicaProxy(self.registry, self.client))
        self.health = HealthMonitor(config, self.registry, self.sessions, self.probe_client,
                                    probe=self.proxy.probe if config.is_pd else None)
        self._admin_token = admin_token
        fleet_kwargs = {
            "state_path": state_path,
            # Readiness and liveness are different questions, and the fleet
            # needs both: see FleetManager._wait_gone.
            "probe": self._probe_replica,
            "alive_probe": self._endpoint_answers,
        }
        if transport_factory is not None:
            fleet_kwargs["transport_factory"] = transport_factory
        self.fleet = FleetManager(config, self.registry, **fleet_kwargs)
        self.app = FastAPI(title="PyPTO Serving Router", lifespan=self._lifespan)
        self.app.state.router = self

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
        with contextlib.ExitStack() as stack:
            if self.config.is_pd:
                from pypto_serving.pd.config import log_context

                stack.enter_context(log_context(self.proxy.config, "router"))
            try:
                await self.health.start(probe_first=self.config.is_pd)
                if not self.config.is_pd:
                    await self.fleet.adopt_previous()
                    await self.fleet.launch_initial()
                yield
            finally:
                try:
                    if self.config.is_pd:
                        await self.proxy.close()
                    else:
                        await self.fleet.stop_all()
                finally:
                    await self.health.stop()
                    for client in (self.client, self.probe_client):
                        close = getattr(client, "aclose", None)
                        if close is not None:
                            await close()

    async def _health_status(self, spec) -> int | None:
        """The replica's /health status, or None if nothing answered.

        Raises ProbeNotSent when the request never left the router, which is a
        third answer: not "200", not "silent", but "we did not get to ask".
        """
        try:
            response = await self.probe_client.get(
                f"{spec.base_url}/health", timeout=self.config.connect_timeout_seconds
            )
        except Exception as exc:  # noqa: BLE001 - unreachable is the answer, not an error
            if is_router_side_failure(exc):
                raise ProbeNotSent(str(exc)) from exc
            return None
        return response.status_code

    async def _probe_replica(self, spec) -> bool:
        """Is a replica *serving* at this address? Readiness, so 200 only."""
        try:
            return (await self._health_status(spec)) == 200
        except ProbeNotSent:
            # Not ready, because we cannot say it is. This probe decides
            # whether to adopt or to keep waiting, and both are recoverable --
            # unlike the liveness answer below, which frees devices.
            return False

    async def _endpoint_answers(self, spec) -> bool:
        """Is *anything* still answering at this address? Any status counts.

        Liveness, which is a different question. serving answers /health with
        503 whenever its engine is not ready, so a replica whose worker died
        with its HTTP server still up answers 503 while holding its devices.
        Reading that as readiness is correct; reading it as an exit is how those
        devices get handed to the next launch.
        """
        try:
            return (await self._health_status(spec)) is not None
        except ProbeNotSent:
            # Our own pool, not the replica's silence. The caller treats False
            # as "the process is gone" and relaunches onto its cards, so a
            # router that is merely busy would put a second serving process on
            # a device group the first one still holds. Assume it is there.
            return True

    def _authorized(self, request: Request) -> bool:
        if self._admin_token is None:
            return True
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer":
            return False
        # Constant time, and compared as bytes rather than str: compare_digest
        # raises TypeError on any non-ASCII character, and the supplied half is
        # attacker-controlled, so comparing str turns `Bearer <non-ASCII>` into an
        # unhandled 500 on every admin route instead of a 401 -- and a configured
        # token with an accented character locks every caller out the same way.
        return secrets.compare_digest(
            token.encode("utf-8"), self._admin_token.encode("utf-8")
        )

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
            # starting / ready / draining / failed. "ready" alone could not say
            # whether an unroutable replica was still loading or had died.
            "phase": state.phase,
            "role": state.spec.role,
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
            # Checked before the 404: between a dead replica's stop and its
            # replacement's launch there is no registry entry, and answering
            # "no such replica" there would let the one the operator just
            # retired come back a moment later.
            if self.fleet.cancel_replacement(name):
                return JSONResponse({"name": name, "stopped": True,
                                     "message": "its replacement was called off",
                                     "capacity": self.fleet.capacity()})
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
        # Disarmed before the stop, not only once the entry has gone: a
        # replacement sets its flag and *then* drains, which can take the whole
        # drain timeout. A DELETE inside that window joins the retirement
        # already running and is answered "stopped", while the replacement --
        # still armed -- puts the replica back a moment later.
        self.fleet.cancel_replacement(name)
        stopped = await self.fleet.stop(name)
        if not stopped:
            # Out of rotation, but its devices are in an unknown state, so the
            # slot stays reserved rather than being handed to the next launch.
            return JSONResponse(
                status_code=500,
                content={
                    "object": "error",
                    "message": (
                        f"replica {name!r} was removed from routing but could not be "
                        "confirmed stopped; its slot stays reserved"
                    ),
                    "capacity": self.fleet.capacity(),
                },
            )
        return JSONResponse({"name": name, "stopped": True,
                             "capacity": self.fleet.capacity()})

    async def _health(self) -> JSONResponse:
        """The router's own view of the replica table."""
        ready = self.registry.ready_count()
        available = bool(self.proxy.pairs()) if self.config.is_pd else bool(ready)
        payload = {
            "status": "ok" if available else "no_replicas",
            "replicas_ready": ready,
            # Reported here and not only on /replicas, which needs the admin
            # token: how many replicas died is operational, not administrative.
            "replicas_failed": self.registry.failed_count(),
            "replicas_total": len(self.registry.states),
            "sessions": len(self.sessions),
            "routed": self.registry.total_routed(),
            "affinity_hits": self.registry.affinity_hits,
            "rejected": self.registry.rejected,
            "replicas": [
                {"name": s.name, "url": s.spec.base_url, "ready": s.ready,
                 "phase": s.phase, "outstanding": s.outstanding, "routed": s.routed}
                for s in self.registry.states
            ],
        }
        if self.config.is_pd:
            payload["compatible_pairs"] = len(self.proxy.pairs())
        return JSONResponse(payload, status_code=200 if available else 503)

    async def _list_models(self) -> JSONResponse:
        """Answer from any routable replica; every replica serves one model."""
        if self.config.is_pd:
            names = sorted({p.contract.model for p, _ in self.proxy.pairs()})
            return JSONResponse({"object": "list", "data": [{"id": n, "object": "model"} for n in names]})
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
    parser.add_argument("--replace-after", type=float, default=DEFAULT_REPLACE_AFTER_SECONDS,
                        metavar="SECONDS",
                        help="Seconds a replica may stay unroutable before the router retires it "
                             "and launches another on the same device(s) "
                             f"(default: {DEFAULT_REPLACE_AFTER_SECONDS:g}).")
    parser.add_argument("--no-replace-dead", action="store_true",
                        help="Leave a replica that died in place instead of replacing it. Its "
                             "device(s) stay held, which is what you want while investigating "
                             "one.")
    parser.add_argument("--state-file", default=None, metavar="PATH",
                        help="Where to record launched replicas so a restarted router can adopt "
                             "them instead of paying for a fresh model load.")
    parser.add_argument("--admin-token", default=None, metavar="TOKEN",
                        help=f"Require 'Authorization: Bearer TOKEN' on /replicas. Those routes "
                             f"start processes on other machines; without a token they are open "
                             f"to anyone who can reach the port. Prefer {ADMIN_TOKEN_ENV}: a "
                             f"token in argv is readable by any local user through ps.")
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
                        help="Write and connection-pool timeout per proxied request. It is not "
                             "an overall budget: there is no read timeout, because decode is "
                             "slow by design, so a replica that sends headers and then stalls "
                             f"is not bounded by this (default: {DEFAULT_REQUEST_TIMEOUT_SECONDS:g}s).")
    parser.add_argument("--connect-timeout", type=float, default=DEFAULT_CONNECT_TIMEOUT_SECONDS,
                        help=f"Connect timeout per replica (default: {DEFAULT_CONNECT_TIMEOUT_SECONDS:g}s).")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        # Launching one replica is the useful default, but only where there is
        # somewhere to launch it; a static-only file must not be forced to.
        config = load_router_config(
            args.replicas,
            session_ttl_seconds=args.session_ttl,
            affinity_slack=args.affinity_slack,
            health_interval_seconds=args.health_interval,
            request_timeout_seconds=args.request_timeout,
            connect_timeout_seconds=args.connect_timeout,
            launch_timeout_seconds=args.launch_timeout,
            drain_timeout_seconds=args.drain_timeout,
            replace_dead=not args.no_replace_dead,
            replace_after_seconds=args.replace_after,
            initial_replicas=args.initial_replicas,
            max_replicas=args.max_replicas,
        )
    except ValueError as exc:
        print(f"error: {exc}")
        return 1

    # The environment wins: a token on the command line is visible to every local
    # user through ps and /proc/<pid>/cmdline, and this one is the only thing in
    # front of a route that launches processes on other machines.
    #
    # Deliberately not a bare `or`: an env var that is set but empty -- a deploy
    # script expanding a variable that was never defined -- would fall through to
    # --admin-token and, with none given, silently leave those routes open to
    # anyone who can reach the port. Checked before the banner, so the error is
    # not buried under a "Starting ..." line the process never honours.
    env_token = os.environ.get(ADMIN_TOKEN_ENV)
    if env_token is not None and not env_token.strip():
        print(f"error: {ADMIN_TOKEN_ENV} is set but empty; unset it or give it a value")
        return 1
    admin_token = env_token or args.admin_token

    try:
        import uvicorn
    except ImportError as e:
        raise ImportError("The router requires uvicorn. Install with: pip install uvicorn") from e

    print(f"Starting PyPTO serving router on {args.host}:{args.port}")
    for replica in config.replicas:
        print(f"  Replica ({replica.role}, external): {replica.name} -> {replica.base_url}")
    for host in config.hosts:
        where = "local" if host.is_local else host.ssh
        print(
            f"  Host: {host.name} [{where}] devices={list(host.devices)} "
            f"({host.devices_per_replica}/replica -> {host.replica_capacity} slot(s))"
        )
    print(f"  Pool: launching {config.initial_replicas} of {config.pool_ceiling} slot(s) at startup")
    if config.is_pd:
        print("  Routing: round-robin compatible P/D pairs; nodes are managed externally")
    else:
        print(f"  Session TTL: {config.session_ttl_seconds:g}s, affinity slack: {config.affinity_slack}")
    print("  Endpoints: /v1/completions, /v1/chat/completions, /v1/models, /health, /replicas")
    if config.hosts and admin_token is None:
        print("  WARNING: /replicas is unauthenticated; it can start processes on other hosts")
    elif args.admin_token is not None and not os.environ.get(ADMIN_TOKEN_ENV):
        print(f"  NOTE: --admin-token is visible in ps; prefer {ADMIN_TOKEN_ENV}")

    try:
        app = create_router_app(
            config,
            state_path=Path(args.state_file) if args.state_file else None,
            admin_token=admin_token,
        )
    except ValueError as exc:
        print(f"error: {exc}")
        return 1
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


__all__ = ["SESSION_HEADER", "ServingRouter", "build_parser", "create_router_app", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
