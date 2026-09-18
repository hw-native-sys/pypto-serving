# Copyright (c) PyPTO Contributors.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""CLI entry point for the external PD Router."""

from __future__ import annotations

import argparse

from .app import create_router_app
from .config import RouterConfig
from pypto_serving.serving.pd.config import load_pd_document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pypto-pd-router")
    parser.add_argument("--pd-config", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = RouterConfig.from_document(load_pd_document(args.pd_config))
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError("Router serving requires uvicorn") from exc
    uvicorn.run(create_router_app(config), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
