# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import argparse
from pathlib import Path

from .app import create_app
from .collector import MetricsCollector
from .store import MonitorStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local monitoring dashboard for PyPTO Serving")
    parser.add_argument("--target", default="http://127.0.0.1:8899", help="PyPTO Serving base URL")
    parser.add_argument("--host", default="127.0.0.1", help="Dashboard listen address")
    parser.add_argument("--port", type=int, default=9090, help="Dashboard listen port")
    parser.add_argument("--interval", type=float, default=1.0, help="Metrics polling interval in seconds")
    parser.add_argument("--timeout", type=float, default=2.0, help="Metrics request timeout in seconds")
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("~/.local/state/pypto-serving/monitor.sqlite3").expanduser(),
        help="SQLite history database",
    )
    parser.add_argument("--timezone", default="local", help="IANA timezone or 'local'")
    parser.add_argument("--retention-hours", type=int, default=24, help="Detailed history retention")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError("PyPTO Monitor requires uvicorn") from exc

    store = MonitorStore(
        args.database,
        timezone_name=args.timezone,
        retention_seconds=args.retention_hours * 3600,
    )
    collector = MetricsCollector(
        args.target,
        store,
        interval=args.interval,
        timeout=args.timeout,
    )
    app = create_app(collector, store)
    print(f"Monitoring: {collector.target}")
    print(f"Dashboard: http://{args.host}:{args.port}")
    print(f"Database: {store.path}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
