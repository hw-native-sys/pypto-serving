# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
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
