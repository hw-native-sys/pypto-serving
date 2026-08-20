# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import logging

from pypto_serving.observability.access_log import (
    SuccessfulEndpointAccessLogFilter,
    create_uvicorn_log_config,
)


def _access_record(path: str, status_code: int) -> logging.LogRecord:
    return logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg='%s - "%s %s HTTP/%s" %d',
        args=("127.0.0.1:12345", "GET", path, "1.1", status_code),
        exc_info=None,
    )


def test_filter_suppresses_successful_metrics_requests() -> None:
    access_filter = SuccessfulEndpointAccessLogFilter(["/metrics", "/metrics/json"])

    assert access_filter.filter(_access_record("/metrics/json", 200)) is False
    assert access_filter.filter(_access_record("/metrics?format=text", 204)) is False


def test_filter_keeps_failures_and_business_requests() -> None:
    access_filter = SuccessfulEndpointAccessLogFilter(["/metrics", "/metrics/json"])

    assert access_filter.filter(_access_record("/metrics/json", 500)) is True
    assert access_filter.filter(_access_record("/v1/completions", 200)) is True


def test_uvicorn_log_config_attaches_filter_to_access_handler() -> None:
    config = create_uvicorn_log_config(["/metrics/json"])

    assert "successful_endpoint_filter" in config["filters"]
    assert "successful_endpoint_filter" in config["handlers"]["access"]["filters"]
