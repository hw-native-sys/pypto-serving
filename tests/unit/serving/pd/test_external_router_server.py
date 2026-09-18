# Copyright (c) PyPTO Contributors.
# Licensed under CANN Open Software License Agreement Version 2.0.

from types import SimpleNamespace

from pypto_serving.config.types import GenerateConfig
from pypto_serving.serving.server.server import ServingServer


class _ExternalConfig:
    enabled = True


class _Engine:
    config = SimpleNamespace(pd_config=_ExternalConfig())
    pd_health_error = ""


def test_external_node_exposes_internal_api_but_not_public_generation() -> None:
    server = ServingServer(_Engine(), "model", GenerateConfig())
    paths = {route.path for route in server.app.routes}
    assert "/health" in paths
    assert "/internal/pd/descriptor" in paths
    assert "/internal/pd/prepare" in paths
    assert "/internal/pd/await-decode" in paths
    assert "/v1/completions" not in paths
    assert "/v1/chat/completions" not in paths
