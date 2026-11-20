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
from unittest.mock import Mock

import msgspec
import pytest

from pypto_serving.config.types import GenerateConfig, RuntimeConfig
from pypto_serving.serving.model_extension import RequestExecutionContext
from pypto_serving.serving.engine.async_engine import EngineConfig, ReplicaEngineCore
from pypto_serving.serving.server.ipc import NewRequestData, StepCommand
from pypto_serving.serving.server.serving_worker import WorkerProcess


def test_effective_request_budget_survives_ipc_roundtrip_without_changing_legacy_defaults():
    original = NewRequestData("r", [260]*7, 0.0, 1.0, None, max_output_tokens=9)
    restored = msgspec.msgpack.decode(msgspec.msgpack.encode(original), type=NewRequestData)
    assert restored.max_output_tokens == 9
    legacy = msgspec.msgpack.decode(msgspec.msgpack.encode(
        {"request_id": "old", "prompt_token_ids": [260], "temperature": 0.0,
         "top_p": 1.0, "top_k": None}), type=NewRequestData)
    assert legacy.max_output_tokens is None


def test_worker_releases_previous_generation_before_delivering_typed_context():
    order = []
    worker = WorkerProcess.__new__(WorkerProcess)
    def release(ids):
        order.append(("release", ids))
    def prepare(model, contexts):
        assert contexts == [RequestExecutionContext("r", 7, 9)]
        order.append(("prepare", model))
    worker.executor = SimpleNamespace(uses_request_execution_context=True,
                                      release_finished_requests=release, prepare_request_contexts=prepare)
    worker.model_record = SimpleNamespace(runtime_model=object())
    worker._req_cache = {"r": NewRequestData("r", [260], 0.0, 1.0, None)}
    worker._last_tokens = {}
    replacement = NewRequestData("r", [260]*7, 0.0, 1.0, None, max_output_tokens=9)
    worker._apply_command_lifecycle(StepCommand([replacement], [], [], ["r"]))
    assert [row[0] for row in order] == ["release", "prepare"]
    assert worker._req_cache["r"] is replacement


def test_missing_native_budget_is_rejected_before_registering_request():
    worker = WorkerProcess.__new__(WorkerProcess)
    worker.executor = SimpleNamespace(uses_request_execution_context=True, prepare_request_contexts=Mock())
    worker.model_record = SimpleNamespace(runtime_model=object())
    worker._req_cache = {}
    with pytest.raises(ValueError, match="execution context"):
        worker._apply_command_lifecycle(StepCommand([NewRequestData("r", [260], 0.0, 1.0, None)], [], [], []))
    worker.executor.prepare_request_contexts.assert_not_called()
    assert not worker._req_cache


@pytest.mark.parametrize("prompt,output", [(16315, 64), (8, 2049)])
def test_native_physical_admission_rejects_before_scheduler_or_request_publication(prompt, output):
    core = ReplicaEngineCore.__new__(ReplicaEngineCore)
    core.config = EngineConfig(executor_cls="PyptoDeepSeekV4LowlatExecutor", enable_prefix_cache=False,
                               max_num_running_reqs=1,
                               runtime_config=RuntimeConfig(max_seq_len=16384, num_speculative_tokens=5))
    core.tokenizer = SimpleNamespace(eos_token_id=1)
    core._request_contexts = {}
    core.scheduler = Mock()
    async def execute():
        generator = core.add_request("r", "", GenerateConfig(max_new_tokens=output),
                                      prompt_token_ids=[260]*prompt)
        with pytest.raises(ValueError):
            await generator.__anext__()
    asyncio.run(execute())
    assert not core._request_contexts
    core.scheduler.add_request.assert_not_called()
