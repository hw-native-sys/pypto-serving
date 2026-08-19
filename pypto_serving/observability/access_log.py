# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import copy
import logging
from urllib.parse import urlparse


class SuccessfulEndpointAccessLogFilter(logging.Filter):
    """Suppress successful Uvicorn access logs for high-frequency endpoints."""

    def __init__(self, excluded_paths: list[str] | None = None) -> None:
        super().__init__()
        self.excluded_paths = set(excluded_paths or [])

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name != "uvicorn.access" or not self.excluded_paths:
            return True

        log_args = record.args
        if not isinstance(log_args, tuple) or len(log_args) < 5:
            return True

        path_with_query = log_args[2]
        status_code = log_args[4]
        if not isinstance(path_with_query, str) or not isinstance(status_code, int):
            return True

        path = urlparse(path_with_query).path
        return path not in self.excluded_paths or not 200 <= status_code < 300


def create_uvicorn_log_config(excluded_paths: list[str]) -> dict:
    """Add a selective endpoint filter to Uvicorn's standard log config."""
    try:
        from uvicorn.config import LOGGING_CONFIG
    except ImportError as exc:
        raise ImportError("Serving mode requires uvicorn. Install with: pip install uvicorn") from exc

    config = copy.deepcopy(LOGGING_CONFIG)
    config.setdefault("filters", {})["successful_endpoint_filter"] = {
        "()": SuccessfulEndpointAccessLogFilter,
        "excluded_paths": excluded_paths,
    }
    access_handler = config["handlers"]["access"]
    access_handler.setdefault("filters", []).append("successful_endpoint_filter")
    return config
