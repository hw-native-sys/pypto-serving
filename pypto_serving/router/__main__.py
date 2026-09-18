# Copyright (c) PyPTO Contributors.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""CLI entry point for the external PD Router."""

from __future__ import annotations

import argparse

from .app import create_router_app
from .config import RouterConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pypto-pd-router")
    parser.add_argument("--prefill-url", required=True)
    parser.add_argument("--decode-url", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-id", default="")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--pd-auth-secret-env", default="PYPTO_PD_AUTH_SECRET")
    parser.add_argument("--pd-router-auth-secret-env", default="PYPTO_PD_ROUTER_SECRET")
    parser.add_argument("--route-epoch", type=int, default=1)
    parser.add_argument("--control-incarnation", type=int, default=1)
    parser.add_argument("--request-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--ticket-ttl-seconds", type=float, default=300.0)
    parser.add_argument("--journal-path", required=True)
    parser.add_argument("--max-active-handoffs", type=int, default=4)
    parser.add_argument("--max-pending-handoffs", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = RouterConfig(
        prefill_url=args.prefill_url,
        decode_url=args.decode_url,
        run_id=args.run_id,
        model_id=args.model_id,
        auth_secret_env=args.pd_auth_secret_env,
        route_secret_env=args.pd_router_auth_secret_env,
        route_epoch=args.route_epoch,
        control_incarnation=args.control_incarnation,
        request_timeout_seconds=args.request_timeout_seconds,
        ticket_ttl_seconds=args.ticket_ttl_seconds,
        journal_path=args.journal_path,
        max_active_handoffs=args.max_active_handoffs,
        max_pending_handoffs=args.max_pending_handoffs,
    )
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError("Router serving requires uvicorn") from exc
    uvicorn.run(create_router_app(config), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
