# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import asyncio
from types import SimpleNamespace

import pytest

from pypto_serving.config.types import DecodeResult, GenerateConfig
from pypto_serving.observability import InMemoryStatLogger
from pypto_serving.serving.server.ipc import DecodeRequest, StepCommand, decode_result, encode_result
from pypto_serving.serving.server.server import ServingServer
from pypto_serving.serving.server.serving_worker import WorkerProcess


class _Engine:
    def __init__(self) -> None:
        self.metrics = InMemoryStatLogger("test-model", [0])


def test_metrics_routes_expose_prometheus_and_json():
    server = ServingServer(_Engine(), model_id="test-model", generate_config=GenerateConfig())
    paths = {route.path for route in server.app.routes}

    assert "/metrics" in paths
    assert "/metrics/json" in paths

    prometheus = asyncio.run(server._metrics())
    structured = asyncio.run(server._metrics_json())
    assert prometheus.status_code == 200
    assert b"pypto:num_requests_running" in prometheus.body
    assert b"vllm:num_requests_running" in prometheus.body
    assert b"vllm:spec_decode_num_draft_tokens_total" in prometheus.body
    assert prometheus.headers["content-type"].startswith("text/plain; version=0.0.4")
    assert b'"schema_version":1' in structured.body
    assert b"pypto:accepted_tokens_total" in prometheus.body
    assert b'"draft_tokens":0' in structured.body


@pytest.mark.parametrize("reclaim", [False, True])
@pytest.mark.parametrize("counts", [[7, 0], None])
def test_worker_preserves_speculation_metadata_in_serial_and_reclaim_results(reclaim, counts):
    worker = WorkerProcess.__new__(WorkerProcess)
    worker._batch_builder = worker._make_decode_batch
    worker._services = None
    result = DecodeResult(None, None, accepted_token_ids=[[11, 12, 13], [21]], num_draft_tokens=counts)
    worker.executor = SimpleNamespace(
        run_decode=lambda model, batch: result,
        reclaim_prepared_decode=lambda pending: result,
    )
    worker.model_record = SimpleNamespace(runtime_model=object())
    worker._batch_builder = lambda *args, **kwargs: object()
    worker._allow_device_sampled_ids = lambda scheduled: True
    worker._allow_device_topk_sampling = lambda scheduled: False
    worker._last_tokens = {}
    scheduled = [DecodeRequest("spec", 10, 20, []), DecodeRequest("fallback", 20, 30, [])]
    command = StepCommand([], [], scheduled, [], step_id=42)
    if reclaim:
        step = worker._reclaim_pending_decode(SimpleNamespace(cmd=command, pending=object(), scheduled=scheduled))
    else:
        step = worker._execute_step(command)
    wire = decode_result(encode_result(step))
    assert wire.error is None
    assert wire.step_id == 42
    assert wire.new_tokens == {"spec": [11, 12, 13], "fallback": [21]}
    assert wire.num_draft_tokens == ({"spec": 7, "fallback": 0} if counts else {})


def test_legacy_step_result_defaults_to_no_speculation():
    import msgspec

    result = decode_result(msgspec.msgpack.encode({"new_tokens": {"r": [1]}, "step_id": 9}))
    assert result.num_draft_tokens == {}


def test_metric_serialization_preserves_labels_and_distinct_latency_semantics():
    from prometheus_client.parser import text_string_to_metric_families

    model = 'model"with\\slashes\nand-newline'
    logger = InMemoryStatLogger(model, [0])
    logger.start_request(0, "r", arrival_monotonic=0, num_prompt_tokens=4)
    logger.record_output(0, "r", completion_tokens=1, timestamp=1)
    logger.record_output(0, "r", completion_tokens=4, timestamp=4)
    logger.finish_request(0, "r", "finished_length", timestamp=4)
    logger.start_request(0, "single", arrival_monotonic=5, num_prompt_tokens=1)
    logger.record_output(0, "single", completion_tokens=1, timestamp=6)
    logger.finish_request(0, "single", "finished_length", timestamp=6)
    samples = {
        item.name: item for family in text_string_to_metric_families(logger.render_prometheus())
        for item in family.samples if "le" not in item.labels
    }
    assert samples["pypto:generation_tokens_total"].value == 5
    assert samples["pypto:generation_tokens_total"].labels["model_name"] == model
    assert samples["vllm:generation_tokens_total"].value == 5
    assert samples["pypto:inter_token_latency_seconds_count"].value == 3
    assert samples["vllm:inter_token_latency_seconds_count"].value == 1
    assert samples["pypto:request_time_per_output_token_seconds_count"].value == 1
    assert samples["vllm:request_time_per_output_token_seconds_count"].value == 2
