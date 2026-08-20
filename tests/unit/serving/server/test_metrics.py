# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import asyncio

from pypto_serving.observability import InMemoryStatLogger
from pypto_serving.serving.server.server import ServingServer


class _Engine:
    def __init__(self) -> None:
        self.metrics = InMemoryStatLogger("test-model", [0])


def test_metrics_routes_expose_prometheus_and_json():
    server = ServingServer(_Engine(), model_id="test-model")
    paths = {route.path for route in server.app.routes}

    assert "/metrics" in paths
    assert "/metrics/json" in paths

    prometheus = asyncio.run(server._metrics())
    structured = asyncio.run(server._metrics_json())
    assert prometheus.status_code == 200
    assert b"pypto:num_requests_running" in prometheus.body
    assert b'"schema_version":1' in structured.body
