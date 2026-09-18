# Copyright (c) PyPTO Contributors.
# Licensed under CANN Open Software Agreement Version 2.0.
"""Small default log surface for PD processes."""

from __future__ import annotations

import json
from pathlib import Path
import time


def write_startup_record(log_dir: str, *, enabled: bool, values: dict[str, object]) -> None:
    if not enabled:
        return
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp_ns": time.time_ns(), **values}
    (directory / "startup.json").write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (directory / "pd.log").touch(exist_ok=True)
    (directory / "events.jsonl").touch(exist_ok=True)
