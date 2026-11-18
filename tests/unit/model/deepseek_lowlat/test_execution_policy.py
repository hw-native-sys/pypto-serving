# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate policy isolation, arbitrary public reservations and atomic receipt checks."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from pypto_serving.config.decoding import DecodeReservation
from pypto_serving.model.execution_policy import ExecutionPolicy
from pypto_serving.model.deepseek_lowlat.decoding import validate_reservation
from pypto_serving.model.deepseek_lowlat.execution_policy import LowlatExecutionPolicy
from pypto_serving.model.deepseek_lowlat.scheduling import LowlatScheduleState
from pypto_serving.serving.engine.async_engine import EngineConfig
from pypto_serving.serving.sched.scheduler import Request, ScheduledRequest, SchedulerOutput
from pypto_serving.serving.server.ipc import StepResult
from pypto_serving.model.deepseek_lowlat.decoding import NativeDecodeRetirement, encode_retirement


def test_default_policy_has_no_lowlat_behavior_or_model_metadata():
    config = EngineConfig()
    policy = config.execution_policy()
    assert type(policy) is ExecutionPolicy
    assert policy.resolve_async_scheduling() is True
    assert policy.scheduler_options() == {}
    assert policy.request_metadata(object()) == b""
    assert policy.prefill_metadata(object(), is_new=True, cache=object()) == b""
    policy.validate_request_budget(1000000, 512)
    policy.confirm_step(SchedulerOutput(), StepResult({}), object())


def test_shared_reservation_does_not_impose_k5_cycle_or_frame_counts():
    reservation = DecodeReservation(1, "other-model", 2, 3, 9, 256, 20, 7)
    assert reservation.cycles == 3 and reservation.frame_id == 7
    with pytest.raises(ValueError, match="native"):
        validate_reservation(reservation)


def test_malformed_second_native_receipt_cannot_partially_commit_first_request():
    policy = LowlatExecutionPolicy(EngineConfig())
    scheduled = []
    receipts = []
    tokens = {}
    for identity in (1, 2):
        request = Request(str(identity), [1] * 128, max_new_tokens=7)
        request.execution_state = LowlatScheduleState(physical_end=128)
        reservation = DecodeReservation(identity, request.request_id, 1, 1, 6, 256, 6, 0)
        scheduled.append(ScheduledRequest(request, 1, False, decode_reservation=reservation))
        receipts.append(NativeDecodeRetirement(identity, request.request_id, 1, 128, 134, (6,), (5,), 6, 0, 2))
        tokens[request.request_id] = [9] * 6
    bad = receipts[-1]
    receipts[-1] = NativeDecodeRetirement(bad.reservation_id, bad.request_id, 1, 999, 1005, (6,), (5,), 6, 0, 2)
    result = StepResult(tokens, retired_reservation_ids=(1, 2), execution_receipts=tuple(map(encode_retirement, receipts)))
    cache = Mock()
    with pytest.raises(ValueError, match="physical"):
        policy.confirm_step(SchedulerOutput(scheduled_requests=scheduled), result, cache)
    assert all(s.request.execution_state.physical_end == 128 for s in scheduled)
    assert all(s.request.execution_state.raw_output_count == 1 for s in scheduled)
    assert all(s.confirmed_input_end is None for s in scheduled)
    cache.publish_group_page_proof.assert_not_called()


def test_model_state_and_context_extensions_are_absent_from_shared_types():
    from dataclasses import fields
    from pypto_serving.config.types import RuntimeConfig, RequestExecutionContext, PrefillBatch

    assert "confirmed_prefix_cache_namespace" not in {f.name for f in fields(RuntimeConfig)}
    assert "prefix_resume" not in {f.name for f in fields(RequestExecutionContext)}
    assert "cache_checkpoint_plan" not in {f.name for f in fields(PrefillBatch)}
    assert "native_physical_end" not in {f.name for f in fields(Request)}


@pytest.mark.parametrize("kind", ["request", "prefill"])
def test_unsupported_model_metadata_version_fails_before_execution(kind):
    import msgspec
    from pypto_serving.model.deepseek_lowlat.execution_policy import (
        decode_request_metadata,
        decode_prefill_metadata,
    )

    decoder = decode_request_metadata if kind == "request" else decode_prefill_metadata
    with pytest.raises(ValueError, match="version"):
        decoder(msgspec.msgpack.encode({"version": 2}))


def test_worker_delivers_versioned_prefix_metadata_to_real_model_context_before_prefill():
    import msgspec
    from pypto_serving.config.types import RequestExecutionContext
    from pypto_serving.model.deepseek_lowlat.execution_policy import LowlatRequestMetadata
    from pypto_serving.model.deepseek_lowlat.prefix_protocol import PrefixResumeDescriptor
    from pypto_serving.model.deepseek_lowlat.npu_executor import DeepSeekV4LowlatPyptoExecutor
    from pypto_serving.model.deepseek_lowlat.npu_runner import LowlatModelRunner
    from pypto_serving.serving.server.ipc import NewRequestData, StepCommand
    from pypto_serving.serving.server.serving_worker import WorkerProcess

    runner = LowlatModelRunner(None, None)
    executor = DeepSeekV4LowlatPyptoExecutor.__new__(DeepSeekV4LowlatPyptoExecutor)
    executor._runners = {"model": runner}
    worker = WorkerProcess.__new__(WorkerProcess)
    worker.executor = executor
    worker.model_record = SimpleNamespace(
        runtime_model=SimpleNamespace(config=SimpleNamespace(model_id="model"))
    )
    worker._req_cache = {}
    worker._last_tokens = {}
    descriptor = PrefixResumeDescriptor("ns", "hot", 7, 1408, 128, 1, (), (), {})
    registration = NewRequestData(
        "hot",
        [1] * 1536,
        0.0,
        1.0,
        None,
        max_output_tokens=64,
        execution_generation=7,
        execution_metadata=msgspec.msgpack.encode(LowlatRequestMetadata(descriptor)),
    )
    worker._apply_command_lifecycle(StepCommand([registration], [], [], []))
    context = runner.request_contexts["hot"]
    assert isinstance(context, RequestExecutionContext)
    assert context.prefix_resume == descriptor and context.execution_generation == 7
    assert runner.state is None
    # Admission/metadata failures before a device step still release their contexts.
    runner.release_finished_requests(["hot"])
    assert not runner.request_contexts


def test_invalid_request_metadata_does_not_leak_k0_registration_after_error_cleanup():
    import msgspec
    from pypto_serving.config.types import RequestExecutionContext
    from pypto_serving.model.deepseek_lowlat.npu_executor import DeepSeekV4LowlatPyptoExecutor
    from pypto_serving.model.deepseek_lowlat.npu_runner import LowlatModelRunner
    from pypto_serving.serving.server.ipc import NewRequestData

    runner = LowlatModelRunner(None, None)
    runner.register_contexts([RequestExecutionContext("r", 1536, 64)])
    executor = DeepSeekV4LowlatPyptoExecutor.__new__(DeepSeekV4LowlatPyptoExecutor)
    executor._runners = {"model": runner}
    registration = NewRequestData(
        "r",
        [1] * 1536,
        0.0,
        1.0,
        None,
        max_output_tokens=64,
        execution_metadata=msgspec.msgpack.encode({"version": 2}),
    )
    with pytest.raises(ValueError, match="version"):
        executor.prepare_request_metadata(
            SimpleNamespace(config=SimpleNamespace(model_id="model")), [registration]
        )
    executor.release_finished_requests(["r"])
    assert not runner.request_contexts


def test_retirement_model_schema_is_versioned_and_opaque_to_shared_ipc():
    import msgspec
    from dataclasses import fields
    from pypto_serving.config.types import DecodeResult
    from pypto_serving.model.deepseek_lowlat.decoding import decode_retirement
    from pypto_serving.serving.server.ipc import encode_result, decode_result

    payload = b"model-owned-receipt"
    result = StepResult({}, execution_receipts=(payload,))
    assert decode_result(encode_result(result)).execution_receipts == (payload,)
    assert "decode_retirements" not in {f.name for f in fields(DecodeResult)}
    for bad in (b"invalid", msgspec.msgpack.encode({"version": 2, "retirement": {}})):
        with pytest.raises((ValueError, msgspec.DecodeError)):
            decode_retirement(bad)
