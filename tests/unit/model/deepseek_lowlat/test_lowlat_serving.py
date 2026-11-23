# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Lowlat admission, opaque worker metadata, reservations and prefix ownership."""

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import msgspec
import pytest

from pypto_serving.cli.main import build_parser, build_serving_engine_config
from pypto_serving.config.types import GenerateConfig, RuntimeConfig
from pypto_serving.model.deepseek_lowlat.capacity import LowlatCapacityPlan
from pypto_serving.model.deepseek_lowlat.contract import (
    LOWLAT_EXECUTOR,
    build_lowlat_cache_group_specs,
    validate_sampling,
)
from pypto_serving.model.deepseek_lowlat.decoding import (
    DecodeReservation,
    LowlatDecodeExtension,
    NativeDecodeRetirement,
    choose_native_parent,
    encode_decode_extension,
    encode_retirement,
)
from pypto_serving.model.deepseek_lowlat.execution_policy import (
    LowlatExecutionPolicy,
    LowlatPrefillMetadata,
    LowlatRequestContext,
    LowlatRequestMetadata,
    decode_request_metadata,
)
from pypto_serving.model.deepseek_lowlat.prefix import checkpoint_proof
from pypto_serving.model.deepseek_lowlat.scheduling import LowlatScheduleState, LowlatSchedulingPolicy
from pypto_serving.model.execution_policy import ExecutionPolicy
from pypto_serving.serving.engine.async_engine import EngineConfig, ReplicaEngineCore
from pypto_serving.serving.memory.kv_cache import KvCacheManager
from pypto_serving.serving.memory.protocol import CachePublicationProof, GroupPageRange
from pypto_serving.serving.model_extension import RequestExecutionContext
from pypto_serving.serving.sched.scheduler import (
    Request,
    RequestStatus,
    ScheduledRequest,
    Scheduler,
    SchedulerConfig,
    SchedulerOutput,
)
from pypto_serving.serving.server.ipc import (
    DecodeRequest,
    NewRequestData,
    PrefillRequest,
    StepCommand,
    StepResult,
    decode_command,
    decode_result,
    encode_command,
    encode_result,
)
from pypto_serving.serving.server.server import ServingServer
from pypto_serving.serving.server.serving_worker import WorkerProcess


# Public admission and worker metadata.

@pytest.fixture
def cli_args(tmp_path):
    raw = {
        "model_type": "deepseek_v4", "hidden_size": 4096, "num_hidden_layers": 43,
        "num_attention_heads": 64, "vocab_size": 129280, "n_routed_experts": 256,
        "moe_intermediate_size": 2048,
        "quantization_config": {"quant_method": "compressed-tensors"},
        "compress_ratios": [0, 0] + [r for _ in range(20) for r in (4, 128)] + [4],
    }
    (tmp_path / "config.json").write_text(json.dumps(raw))
    return build_parser().parse_args([
        "--model", str(tmp_path), "--devices", "7,6,5,4,3,2,1,0",
        "--dp", "1", "--tp", "8", "--ep", "1", "--lm-head-tp", "8",
        "--max-num-seqs", "1", "--block-size", "128", "--max-model-len", "512",
        "--max-num-batched-tokens", "128", "--long-prefill-token-threshold", "128",
        "--no-enable-prefix-caching",
        "--speculative-config", '{"method":"dspark","num_speculative_tokens":5}',
    ])


def test_dspark_k5_routes_to_stable_lowlat_contract(cli_args):
    config = build_serving_engine_config(cli_args)
    assert config.executor_cls == LOWLAT_EXECUTOR
    assert config.worker_device_ids() == (7, 6, 5, 4, 3, 2, 1, 0)
    assert config.parallel_config.placement_mode == "replica"
    assert config.resolve_async_scheduling() is True
    assert config.executor_kwargs["num_speculative_tokens"] == 5
    assert config.executor_kwargs["cache_initialization"] == "on-demand"
    assert "native_pipeline" not in config.executor_kwargs
    assert "native_launch_depth" not in config.executor_kwargs
    assert config.runtime_config.requires_homogeneous_prefill_decode
    assert any(group.sliding_window is not None for group in config.runtime_config.kv_cache_groups)


def test_dspark_k0_keeps_regular_dspark_executor(cli_args):
    cli_args.speculative_config = {"method": "dspark", "num_speculative_tokens": 0}
    cli_args.devices = ",".join(str(index) for index in range(16))
    cli_args.data_parallel_size = 4
    cli_args.tensor_parallel_size = 4
    cli_args.expert_parallel_size = 16
    cli_args.lm_head_tp = 4
    cli_args.block_size = 32
    config = build_serving_engine_config(cli_args)
    assert config.executor_cls == "PyptoDeepSeekV4DSparkExecutor"
    assert config.runtime_config.num_speculative_tokens == 0


@pytest.mark.parametrize(
    ("name", "value"),
    [("expert_parallel_size", 8), ("tensor_parallel_size", 4),
     ("data_parallel_size", 2), ("lm_head_tp", 4), ("max_num_seqs", 2),
     ("block_size", 32),
     ("long_prefill_token_threshold", 8193), ("pd_role", "decode")],
)
def test_invalid_topologies_fail_before_device_work(cli_args, name, value):
    setattr(cli_args, name, value)
    with pytest.raises(ValueError):
        build_serving_engine_config(cli_args)


@pytest.mark.parametrize("value", [True, 0.0, 2])
def test_rejects_invalid_dspark_speculation(cli_args, value):
    cli_args.speculative_config["num_speculative_tokens"] = value
    with pytest.raises(ValueError, match="num_speculative_tokens"):
        build_serving_engine_config(cli_args)


def test_lowlat_prefix_is_fixed_to_rolling_and_on_demand(cli_args):
    cli_args.enable_prefix_caching = True
    config = build_serving_engine_config(cli_args)
    assert config.enable_prefix_cache is True
    assert config.executor_kwargs["prefix_cache_namespace"]
    assert config.runtime_config.speculative_prefix_cache_replay_tokens == 128
    assert config.runtime_config.kv_cache_groups[0].num_blocks > config.runtime_config.kv_cache_groups[0].max_blocks_per_seq


def test_lowlat_public_parser_has_no_private_options():
    options = {action.option_strings[0] for action in build_parser()._actions if action.option_strings}
    assert "--speculative-config" in options
    assert "--lowlat-native-pipeline" not in options
    assert "--lowlat-native-launch-depth" not in options
    assert "--lowlat-cache-layout" not in options
    assert "--lowlat-cache-initialization" not in options
    assert "--num-speculative-tokens" not in options


@pytest.mark.parametrize(
    "options", [dict(temperature=0.1), dict(min_tokens=1), dict(top_p=0.9), dict(top_k=3)]
)
def test_unsupported_sampling_is_rejected(options):
    with pytest.raises(ValueError, match="greedy"):
        validate_sampling(GenerateConfig(**options))


def test_min_tokens_is_rejected_before_downgrade(cli_args):
    config = build_serving_engine_config(cli_args)
    with pytest.raises(ValueError, match="min_tokens"):
        config.resolve_min_tokens(1)
    with pytest.raises(ValueError, match="structured"):
        config.validate_generate_config(GenerateConfig(), constraint_spec=object())


def test_million_context_cli_uses_rolling_capacity(cli_args):
    cli_args.max_model_len = 1048576
    config = build_serving_engine_config(cli_args)
    specs = {spec.name: spec for spec in config.resolve_runtime_config().kv_cache_groups}
    assert specs["cmp_c4"].num_blocks == 8192
    assert specs["hca_state"].num_blocks == 18


@pytest.mark.parametrize("budget", [6, 11])
def test_lowlat_rejects_token_budget_below_native_parent_minimum(cli_args, budget):
    cli_args.max_num_batched_tokens = budget
    with pytest.raises(ValueError, match="budget>=12"):
        build_serving_engine_config(cli_args)


class AdmissionOnlyEngine:
    def __init__(self):
        self.config = EngineConfig(
            executor_cls=LOWLAT_EXECUTOR, max_num_running_reqs=1, enable_prefix_cache=False
        )
        self.tokenizer = SimpleNamespace(output_parser_id=None)
        self.dispatched = False

    def validate_generate_config(self, config, **kwargs):
        self.config.validate_generate_config(config, **kwargs)

    async def add_request(self, *args, **kwargs):
        self.dispatched = True
        raise AssertionError("invalid input reached inference")
        yield


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/v1/completions", {"prompt": "hi", "temperature": 0.5, "stream": True}),
        ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}], "stream": True}),
        (
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "hi"}], "min_tokens": 1, "stream": True},
        ),
        ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "top_p": 0.5}),
    ],
)
def test_http_rejects_before_sse_headers_or_device_dispatch(path, payload):
    engine = AdmissionOnlyEngine()
    server = ServingServer(engine, "lowlat", GenerateConfig())

    async def request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            return await client.post(path, json=payload)

    response = asyncio.run(request())
    assert response.status_code == 400, response.text
    assert response.headers["content-type"].startswith("application/json")
    assert engine.dispatched is False


@pytest.mark.parametrize("prompt_tokens,outputs", [(16315, 64), (7, 2049)])
def test_native_budget_rejection_precedes_sse_and_inference(prompt_tokens, outputs):
    from types import SimpleNamespace

    from pypto_serving.config.types import RuntimeConfig
    from pypto_serving.serving.engine.async_engine import AsyncLLMEngine

    engine = AsyncLLMEngine.__new__(AsyncLLMEngine)
    engine.config = EngineConfig(executor_cls=LOWLAT_EXECUTOR, max_num_running_reqs=1,
                                 enable_prefix_cache=False,
                                 runtime_config=RuntimeConfig(max_seq_len=16384, num_speculative_tokens=5))
    engine.tokenizer = SimpleNamespace(encode=lambda text: [260]*prompt_tokens, bos_token_id=None)
    engine.dispatched = False
    async def dispatch(*args, **kwargs):
        engine.dispatched = True
        raise AssertionError("invalid native budget reached serving")
        yield
    engine.add_request = dispatch
    server = ServingServer(engine, "lowlat", GenerateConfig())
    async def request():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app),
                                     base_url="http://test", trust_env=False) as client:
            return await client.post("/v1/completions", json=dict(prompt="x", max_tokens=outputs,
                                                                  temperature=0, stream=True))
    response = asyncio.run(request())
    assert response.status_code == 400 and response.headers["content-type"].startswith("application/json")
    assert engine.dispatched is False


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


def test_native_physical_admission_rejects_before_scheduler_or_request_publication():
    prompt, output = 16315, 64
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
    with pytest.raises(ValueError, match="undeclared"):
        policy.confirm_step(SchedulerOutput(), StepResult({}, model_extension=b"opaque"), object())


@pytest.mark.parametrize("kind", ["request", "prefill"])
def test_unsupported_model_metadata_version_fails_before_execution(kind):
    import msgspec

    from pypto_serving.model.deepseek_lowlat.execution_policy import (
        decode_prefill_metadata,
        decode_request_metadata,
    )

    decoder = decode_request_metadata if kind == "request" else decode_prefill_metadata
    with pytest.raises(ValueError, match="version"):
        decoder(msgspec.msgpack.encode({"version": 2}))


def test_worker_delivers_versioned_prefix_metadata_to_real_model_context_before_prefill():
    import msgspec

    from pypto_serving.model.deepseek_lowlat.execution_policy import LowlatRequestMetadata
    from pypto_serving.model.deepseek_lowlat.npu_executor import DeepSeekV4LowlatPyptoExecutor
    from pypto_serving.model.deepseek_lowlat.npu_runner import LowlatModelRunner
    from pypto_serving.model.deepseek_lowlat.prefix_protocol import PrefixResumeDescriptor
    from pypto_serving.serving.model_extension import RequestExecutionContext
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
        model_extension=msgspec.msgpack.encode(LowlatRequestMetadata(descriptor)),
    )
    worker._apply_command_lifecycle(StepCommand([registration], [], [], []))
    context = runner.request_contexts["hot"]
    assert isinstance(context, RequestExecutionContext)
    assert context.prefix_resume == descriptor and context.execution_generation == 7
    assert runner.state is None
    # Admission/metadata failures before a device step still release their contexts.
    runner.release_finished_requests(["hot"])
    assert not runner.request_contexts


def test_invalid_request_metadata_is_cleaned_up_before_model_execution():
    import msgspec

    from pypto_serving.model.deepseek_lowlat.npu_executor import DeepSeekV4LowlatPyptoExecutor
    from pypto_serving.model.deepseek_lowlat.npu_runner import LowlatModelRunner
    from pypto_serving.serving.model_extension import RequestExecutionContext
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
        model_extension=msgspec.msgpack.encode({"version": 2}),
    )
    with pytest.raises(ValueError, match="version"):
        executor.prepare_request_metadata(
            SimpleNamespace(config=SimpleNamespace(model_id="model")), [registration]
        )
    executor.release_finished_requests(["r"])
    assert not runner.request_contexts


def test_retirement_model_schema_is_versioned_and_opaque_to_shared_ipc():
    from dataclasses import fields

    import msgspec

    from pypto_serving.config.types import DecodeResult
    from pypto_serving.model.deepseek_lowlat.decoding import decode_retirement
    from pypto_serving.serving.server.ipc import decode_result, encode_result

    payload = b"model-owned-receipt"
    result = StepResult({}, model_extension=payload)
    assert decode_result(encode_result(result)).model_extension == payload
    assert "decode_retirements" not in {f.name for f in fields(DecodeResult)}
    for bad in (b"invalid", msgspec.msgpack.encode({"version": 2, "retirement": {}})):
        with pytest.raises((ValueError, msgspec.DecodeError)):
            decode_retirement(bad)


# Native scheduler reservations and physical retirement.

@pytest.mark.parametrize(("remaining", "cycles", "capacity"),
                         [(1, 1, 1), (6, 1, 6), (7, 2, 7), (12, 2, 12), (23, 2, 12), (24, 2, 12)])
def test_parent_tail_policy_and_explicit_output_capacity(remaining, cycles, capacity):
    assert choose_native_parent(remaining, []) == (cycles, remaining)
    r = DecodeReservation(1, "r", 7, cycles, capacity, 256, remaining, 0)
    assert choose_native_parent(remaining, [r]) == ((2, 12) if remaining == 24 else None)
    assert choose_native_parent(remaining, [r, replace(r, reservation_id=2, frame_id=1)]) is None


def scheduler(remaining=24, *, eos=None):
    manager = KvCacheManager(enable_prefix_cache=False)
    manager.init_groups(build_lowlat_cache_group_specs(max_seq_len=512), max_batch_size=1)
    config = SchedulerConfig(max_num_running_reqs=1, max_num_scheduled_tokens=128,
        max_seq_len=512, enable_prefix_cache=False, num_speculative_tokens=5,
        async_scheduling=True, execution_policy=LowlatSchedulingPolicy())
    instance = Scheduler(config, manager)
    request = Request("r", [1, 2, 3, 4], remaining+1, temperature=0, eos_token_id=eos)
    instance.add_request(request)
    prefill = instance.schedule()
    instance.advance_after_schedule(prefill)
    instance.update_from_output(prefill, {"r": [9]})
    return instance, request


def accept(instance):
    step = instance.schedule()
    if step.scheduled_requests:
        instance.advance_after_schedule(step)
    return step


def test_two_parents_have_separate_reservations_and_reconcile_actual_variable_counts():
    instance, request = scheduler()
    a, b = accept(instance), accept(instance)
    first, second = a.scheduled_requests[0], b.scheduled_requests[0]
    assert first.model_execution_plan.cycles == second.model_execution_plan.cycles == 2
    assert first.model_execution_plan.frame_id != second.model_execution_plan.frame_id
    assert first.model_execution_plan.generation == request.execution_generation > 0
    assert request.num_output_placeholders == 24
    assert instance.schedule().is_empty
    assert first.block_ids_by_group == second.block_ids_by_group
    # Even the first prefill claimed the effective request budget plus6real slots.
    assert len(first.block_ids_by_group["hca_state"]) == (4+25+6+7)//8
    instance.update_from_output(a, {"r": [10, 11]})
    assert request.num_output_placeholders == 12
    instance.update_from_output(b, {"r": [12, 13]})
    assert request.num_output_placeholders == 0
    assert request.output_token_ids == [9, 10, 11, 12, 13]
    assert request.num_computed_tokens == request.num_prompt_tokens+len(request.output_token_ids)-1
    assert not instance._pending_reservations
    assert instance.config.num_speculative_tokens == 5


def test_23token_budget_does_not_accept_a_second_parent_before_retirement():
    instance, _ = scheduler(23)
    a = accept(instance)
    assert instance.schedule().is_empty
    instance.update_from_output(a, {"r": [10, 11]})
    assert accept(instance).scheduled_requests[0].model_execution_plan.remaining_lower_bound == 21


@pytest.mark.parametrize("terminal", ["eos", "abort"])
def test_eos_or_abort_discards_and_retires_the_already_accepted_successor(terminal):
    instance, request = scheduler(eos=42)
    a, b = accept(instance), accept(instance)
    if terminal == "abort":
        instance.abort_request("r")
    instance.update_from_output(a, {"r": [42, 10]})
    assert request.status == (RequestStatus.FINISHED_EOS if terminal == "eos" else RequestStatus.FINISHED_ABORTED)
    assert len(instance._pending_reservations["r"]) == 1
    before = list(request.output_token_ids)
    instance.update_from_output(b, {"r": [20, 21]})
    assert request.output_token_ids == before
    assert not instance._pending_reservations
    assert a.scheduled_requests[0].group_block_snapshot is None
    assert b.scheduled_requests[0].group_block_snapshot is None


def test_wire_round_trip_preserves_generation_cycle_budget_range_and_frame():
    reservation = DecodeReservation(13, "r", 5, 2, 12, 256, 24, 1)
    dr = DecodeRequest("r", -1, 128, [], model_extension=msgspec.msgpack.encode(vars(reservation)))
    command = StepCommand([], [], [dr], [], 88)
    restored = decode_command(encode_command(command)).decode_requests[0].model_extension
    assert msgspec.msgpack.decode(restored) == vars(reservation)


@pytest.mark.parametrize("field", ["reservation_id", "generation", "physical_input_start", "physical_input_end",
                                   "raw_output_tokens", "remaining_output_tokens", "next_epoch"])
def test_physical_retirement_rejects_a_corrupt_cursor_budget_or_generation(field):
    from pypto_serving.model.deepseek_lowlat.decoding import (
        NativeDecodeRetirement,
        validate_native_retirement,
    )

    reservation = DecodeReservation(1, "r", 7, 2, 12, 256, 24, 0)
    receipt = NativeDecodeRetirement(1, "r", 7, 128, 131, (1, 2), (0, 1), 3, 21, 4)
    validate_native_retirement(receipt, reservation, previous_input_end=128, previous_raw_outputs=1,
                               max_output_tokens=25, actual_token_count=3)
    value = getattr(receipt, field)
    bad = replace(receipt, **{field: 0 if field == "next_epoch" else value+1})
    with pytest.raises(ValueError, match="physical retirement"):
        validate_native_retirement(bad, reservation, previous_input_end=128, previous_raw_outputs=1,
                                   max_output_tokens=25, actual_token_count=3)


def test_publication_uses_actual_committed_range_after_draft_rejection():
    instance, request = scheduler()
    a = accept(instance)
    scheduled = a.scheduled_requests[0]
    scheduled.confirmed_input_end = 6
    observed = []
    original = instance.kv_cache_manager.cache_group_blocks_from_snapshot
    def track(hashes, confirmed_tokens, *args, **kwargs):
        observed.append(confirmed_tokens)
        return original(hashes, confirmed_tokens, *args, **kwargs)
    instance.kv_cache_manager.cache_group_blocks_from_snapshot = track
    instance.update_from_output(a, {"r": [10, 11]})
    assert observed == [6]
    assert request.num_computed_tokens == 6


@pytest.mark.parametrize("budget", [1, 7, 23, 24, 127])
def test_continuous_variable_acceptance_reconciles_two_inflight_parents_through_the_tail(budget):
    from pypto_serving.model.deepseek_lowlat.decoding import (
        NativeDecodeRetirement,
        validate_native_retirement,
    )

    instance, request = scheduler(budget)
    queue = []
    raw_remaining, physical_end, raw_count, epoch = budget, 4, 1, 1
    iteration = 0
    while raw_remaining or queue:
        while len(queue) < 2:
            step = accept(instance)
            if step.is_empty:
                break
            queue.append(step)
        assert queue, (budget, raw_remaining)
        step = queue.pop(0)
        scheduled = step.scheduled_requests[0]
        reservation = scheduled.model_execution_plan
        advances, accepted, emitted = [], [], 0
        for cycle in range(reservation.cycles):
            # Independent cycle acceptance, including0and5; never infer from parent width.
            count = (iteration+cycle*5) % 6
            advances.append(count+1)
            accepted.append(count)
            count_emitted = min(count+1, raw_remaining)
            emitted += count_emitted
            raw_remaining -= count_emitted
            assert count_emitted > 0
        receipt = NativeDecodeRetirement(reservation.reservation_id, "r", request.execution_generation,
            physical_end, physical_end+sum(advances), tuple(advances), tuple(accepted), emitted,
            raw_remaining, epoch+reservation.cycles)
        validate_native_retirement(receipt, reservation, previous_input_end=physical_end,
            previous_raw_outputs=raw_count, max_output_tokens=budget+1, actual_token_count=emitted)
        physical_end, epoch = receipt.physical_input_end, receipt.next_epoch
        raw_count += emitted
        scheduled.confirmed_input_end = physical_end
        instance.update_from_output(step, {"r": list(range(100+iteration, 100+iteration+emitted))})
        assert request.num_output_placeholders == sum(s.scheduled_requests[0].model_execution_plan.max_output_tokens
                                                      for s in queue)
        if not request.status.is_finished:
            assert request.num_computed_tokens == 4+len(request.output_token_ids)+request.num_output_placeholders-1
        iteration += 1
        assert iteration <= budget
    assert len(request.output_token_ids) == budget+1 and raw_count == budget+1
    assert request.status == RequestStatus.FINISHED_LENGTH and not instance._pending_reservations


def test_successor_snapshot_keeps_actual_pages_unavailable_until_retirement():
    instance, request = scheduler(eos=42)
    a, b = accept(instance), accept(instance)
    names = list(a.scheduled_requests[0].block_ids_by_group)
    totals = {s.name: s.num_blocks for s in instance.kv_cache_manager.group_specs}
    instance.update_from_output(a, {"r": [42, 10]})
    assert all(instance.kv_cache_manager.group_num_free_blocks(name, 0) < totals[name] for name in names)
    instance.update_from_output(b, {"r": [20, 21]})
    assert all(instance.kv_cache_manager.group_num_free_blocks(name, 0) == totals[name] for name in names)


def test_same_request_id_cannot_start_a_new_generation_while_old_reservation_is_pending():
    instance, request = scheduler()
    a, b = accept(instance), accept(instance)
    old_generation = request.execution_generation
    instance.abort_request("r")
    with pytest.raises(ValueError, match="still owns a generation"):
        instance.add_request(Request("r", [1, 2, 3, 4], 8))
    instance.update_from_output(a, {"r": [10, 11]})
    instance.update_from_output(b, {"r": [12, 13]})
    next_request = Request("r", [1, 2, 3, 4], 8)
    instance.add_request(next_request)
    assert next_request.execution_generation > old_generation


# Prefix checkpoint, restore, publication and page lifetime.

def prefix_scheduler():
    groups = LowlatCapacityPlan(16384, 512, "rolling", True).cache_specs()
    manager = KvCacheManager(enable_prefix_cache=True)
    manager.init_groups(groups, max_batch_size=1)
    scheduler = Scheduler(
        SchedulerConfig(
            max_num_running_reqs=1,
            max_num_scheduled_tokens=512,
            long_prefill_token_threshold=512,
            max_prefill_tokens_per_request=512,
            max_seq_len=16384,
            num_speculative_tokens=5,
            speculative_prefix_cache_replay_tokens=128,
            enable_prefix_cache=True,
            async_scheduling=True,
            execution_policy=LowlatSchedulingPolicy("model-A"),
        ),
        manager,
    )
    return manager, scheduler


def finish_prompt(manager, scheduler, rid, tokens):
    request = Request(rid, list(tokens), max_new_tokens=1, temperature=0.0)
    scheduler.add_request(request)
    starts = []
    maps = []
    for _ in range(100):
        output = scheduler.schedule()
        assert len(output.scheduled_requests) == 1
        step = output.scheduled_requests[0]
        starts.append(step.num_computed_tokens)
        maps.append(step.block_ids_by_group)
        scheduler.advance_after_schedule(output)
        # Restore/copy must settle before another prefill chunk can be accepted.
        assert not scheduler.schedule().scheduled_requests
        if getattr(step.execution_plan, "checkpoint", None):
            plan = getattr(step.execution_plan, "checkpoint", None)
            proof = checkpoint_proof(plan)
            manager.publish_group_page_proof(
                proof,
                request_id=rid,
                generation=request.execution_generation,
                mapping_version=1,
                snapshot=plan.destination_tables,
                partition=0,
                block_hashes=request.group_block_hashes,
            )
            manager.release_prefix_checkpoint(rid)
            request.execution_state.checkpoint_complete = True
        end = step.num_computed_tokens + step.num_new_tokens
        scheduler.update_from_output(output, {rid: [19]} if end == len(tokens) else {})
        if request.status.is_finished:
            break
    else:
        pytest.fail("Prefix request made no bounded progress")
    assert all(table == maps[0] for table in maps)
    assert all(block.ref_cnt == 0 for pool in manager._group_pools.values() for block in pool.blocks)
    return request, starts, maps


@pytest.mark.parametrize("length", [127, 513, 1536])
def test_cold_hot_checkpoint_restore_with_stable_exclusive_rings(length):
    manager, scheduler = prefix_scheduler()
    manager.checkpoint_writer.prepare = Mock(side_effect=AssertionError("duplicate checkpoint writer"))
    tokens = list(range(length))
    cold, starts, _ = finish_prompt(manager, scheduler, "cold", tokens)
    boundary = max(0, length - 128) // 128 * 128
    assert starts[0] == 0
    if boundary:
        assert boundary in [s for s in starts[1:]]
    hot, hot_starts, hot_maps = finish_prompt(manager, scheduler, "hot", tokens)
    assert hot_starts[0] == boundary
    if boundary:
        assert hot.execution_state.prefix_resume.hit_tokens == boundary
        assert length - boundary >= 128
        source = {p.page_id for p in hot.execution_state.prefix_resume.pages if p.group == "hca_state"}
        destination = set(hot_maps[0]["hca_state"])
        assert not source.intersection(destination)
    manager.checkpoint_writer.prepare.assert_not_called()


def test_resume_and_completion_proofs_survive_real_ipc_roundtrip():
    manager, scheduler = prefix_scheduler()
    tokens = list(range(1536))
    finish_prompt(manager, scheduler, "cold", tokens)
    request = Request("hot", tokens, max_new_tokens=64, temperature=0.0)
    scheduler.add_request(request)
    output = scheduler.schedule()
    step = output.scheduled_requests[0]
    new = NewRequestData(
        "hot",
        tokens,
        0.0,
        1.0,
        None,
        max_output_tokens=64,
        execution_generation=request.execution_generation,
        model_extension=msgspec.msgpack.encode(
            LowlatRequestMetadata(request.execution_state.prefix_resume)
        ),
    )
    prefill = PrefillRequest(
        "hot",
        tokens[step.num_computed_tokens :],
        step.num_computed_tokens,
        [],
        step.block_ids_by_group,
        0,
        model_extension=msgspec.msgpack.encode(
            LowlatPrefillMetadata(manager.group_page_identities(step.block_ids_by_group, 0))
        ),
    )
    cmd = StepCommand([new], [prefill], [], [])
    recovered = decode_command(encode_command(cmd))
    assert recovered == cmd
    descriptor = decode_request_metadata(recovered.new_requests[0].model_extension).prefix_resume
    assert descriptor == request.execution_state.prefix_resume
    context = LowlatRequestContext(
        "hot", 1536, 64, execution_generation=request.execution_generation, prefix_resume=descriptor
    )
    assert context.prefix_resume.hit_tokens == 1408
    # Proof types are preserved by msgspec, rather than accepted as opaque dicts.
    plan = manager.reserve_prefix_checkpoint(
        "hot", generation=request.execution_generation, boundary=1408, snapshot=step.block_ids_by_group
    )
    proof = checkpoint_proof(plan)
    result = decode_result(encode_result(StepResult({}, cache_publications=(proof,))))
    assert isinstance(result.cache_publications[0], CachePublicationProof)
    assert result.cache_publications == (proof,)
    scheduler.abort_request("hot")
    scheduler.discard_scheduled_request(step)
    assert all(b.ref_cnt == 0 for p in manager._group_pools.values() for b in p.blocks)


def test_aborted_restore_releases_pins_and_checkpoint_reservations():
    manager, scheduler = prefix_scheduler()
    tokens = list(range(1536))
    finish_prompt(manager, scheduler, "cold", tokens)
    request = Request("abort", tokens, max_new_tokens=64, temperature=0.0)
    scheduler.add_request(request)
    output = scheduler.schedule()
    assert manager._prefix_resume_pins
    scheduler.abort_request("abort")
    assert not manager._prefix_resume_pins
    assert any(b.ref_cnt > 0 for p in manager._group_pools.values() for b in p.blocks)
    scheduler.discard_scheduled_request(output.scheduled_requests[0])
    assert all(b.ref_cnt == 0 for p in manager._group_pools.values() for b in p.blocks)
    _, starts, _ = finish_prompt(manager, scheduler, "next", tokens)
    assert starts[0] == 1408


def test_registered_page_copies_preserve_logical_tail_across_permuted_ring_ids():
    import ctypes
    from contextlib import contextmanager
    from math import prod
    from types import SimpleNamespace

    import torch

    from pypto_serving.model.deepseek_lowlat.cache import TABLE_GROUPS, TENSOR_GROUPS, LowlatCachePool
    from pypto_serving.model.deepseek_lowlat.kernel_loader import isolated_profile
    from pypto_serving.model.deepseek_lowlat.prefix import PrefixPageCopies
    from pypto_serving.serving.memory.protocol import CachePageIdentity

    class Worker:
        def __init__(self):
            self.memory = {}
            self.calls = []
            self.allocations = set()

        def copy_from(self, destination, source, nbytes, *, src_offset=0, worker_id):
            assert source in self.allocations, "copy_from requires the allocation base"
            ctypes.memset(destination, self.memory[source + src_offset, nbytes], nbytes)
            self.calls.append(("read", worker_id, nbytes))

        def copy_to(self, destination, source, nbytes, *, dst_offset=0, worker_id):
            assert destination in self.allocations, "copy_to requires the allocation base"
            data = ctypes.string_at(source, nbytes)
            assert data == bytes([data[0]]) * nbytes
            self.memory[destination + dst_offset, nbytes] = data[0]
            self.calls.append(("write", worker_id, nbytes))

    @contextmanager
    def context():
        with isolated_profile():
            yield

    rows = {"params": [dict(name=n, shape=[8, 1, 4096], dtype="int32", direction="In") for n in TABLE_GROUPS]}
    programs = SimpleNamespace(single_metadata=rows, context=context)
    worker = Worker()
    specs = LowlatCapacityPlan(16384, 512, "rolling", True).cache_specs()
    cache = LowlatCachePool(programs, SimpleNamespace(worker=worker), specs)
    cache.preserve_contents = True
    source, destination = {}, {}
    for name, spec in cache.specs.items():
        count = spec.max_blocks_per_seq if spec.sliding_window else 12
        source[name] = tuple(reversed(range(count)))
        destination[name] = tuple(range(count, 2 * count)) if spec.sliding_window else source[name]
    with context():
        from models.deepseek_v4_flash_lowlat.prefill_cache import CACHE_LAYOUTS

        layouts = dict(CACHE_LAYOUTS)
    ranges = cache.checkpoint_ranges(1408)
    expected = {}
    for tensor_index, (tensor, (layers, rows, _, tail, dtype)) in enumerate(layouts.items()):
        group = TENSOR_GROUPS[tensor]
        spec = cache.specs[group]
        shards = [
            SimpleNamespace(data_ptr=(rank + 1) * 10**14 + (tensor_index + 1) * 10**10, byte_offset=64)
            for rank in range(8)
        ]
        worker.allocations.update(shard.data_ptr - shard.byte_offset for shard in shards)
        cache.owners[tensor] = SimpleNamespace(shards=shards)
        if not spec.sliding_window:
            continue
        interval = next(r for r in ranges if r.group == group)
        nbytes = rows * prod(tail) * torch.empty((), dtype=dtype).element_size()
        for rank, shard in enumerate(shards):
            for layer in range(layers):
                for logical in range(interval.start, interval.end):
                    value = (logical * 13 + layer * 7 + rank * 5 + tensor_index) % 251 + 1
                    old = source[group][logical % len(source[group])]
                    new = destination[group][logical % len(destination[group])]
                    base = shard.data_ptr + layer * spec.num_blocks * nbytes
                    worker.memory[base + old * nbytes, nbytes] = value
                    expected[base + new * nbytes, nbytes] = value
    copier = PrefixPageCopies()
    copier.copy(cache, source=source, destination=destination, ranges=ranges)
    assert all(worker.memory[key] == value for key, value in expected.items())
    assert copier.bytes_copied == 17842176 * 8
    assert sum(kind == "read" for kind, _, _ in worker.calls) == sum(
        kind == "write" for kind, _, _ in worker.calls
    )
    # Retained contents survive request release; a new allocation epoch invalidates them.
    identity = CachePageIdentity("ori", 0, destination["ori"][10 % len(destination["ori"])], 1)
    cache.page_epochs[identity.group, identity.page_id] = 1
    cache.adopt_page_identities((identity,), retained=True)
    cache.release()
    cache.adopt_page_identities((identity,), retained=True)
    cache.adopt_page_identities((replace_identity(identity, allocation_epoch=2),))
    with pytest.raises(ValueError, match="content/epoch"):
        cache.adopt_page_identities((identity,), retained=True)


def replace_identity(identity, **fields):
    from dataclasses import replace

    return replace(identity, **fields)


def test_checkpoint_destination_lives_until_whole_step_ack_after_abort():
    manager, scheduler = prefix_scheduler()
    request = Request("cold", list(range(1536)), max_new_tokens=64, temperature=0.0)
    scheduler.add_request(request)
    for _ in range(3):
        output = scheduler.schedule()
        step = output.scheduled_requests[0]
        scheduler.advance_after_schedule(output)
        if getattr(step.execution_plan, "checkpoint", None) is not None:
            break
        scheduler.update_from_output(output, {})
    assert getattr(step.execution_plan, "checkpoint", None) is not None
    plan = getattr(step.execution_plan, "checkpoint", None)
    scheduler.abort_request("cold")
    for page in plan.destination_pages:
        assert manager._group_pools[page.group].block_from_local_id(0, page.page_id).ref_cnt > 0
    assert all(not p.hash_to_block for p in manager._group_pools.values())
    scheduler.update_from_output(output, {})
    assert all(b.ref_cnt == 0 for p in manager._group_pools.values() for b in p.blocks)
    assert all(not p.hash_to_block for p in manager._group_pools.values())


def setup_cache():
    specs = LowlatCapacityPlan(16384, 512, "rolling", True).cache_specs()
    cache = KvCacheManager(enable_prefix_cache=True)
    cache.init_groups(specs, max_batch_size=1)
    return cache


def checkpoint(cache, request="cold", boundary=1408):
    tokens = list(range(1536))
    hashes = cache.compute_group_block_hashes(tokens)
    pages = cache.ensure_group_blocks(request, boundary)
    identities = cache.group_page_identities(pages, 0)
    ranges = tuple(
        GroupPageRange(
            s.name,
            max(0, boundary // s.spec.token_capacity - s.sliding_window // s.spec.token_capacity)
            if s.sliding_window
            else 0,
            boundary // s.spec.token_capacity,
        )
        for s in cache.group_specs
    )
    proof = CachePublicationProof(request, 1, 1, boundary, boundary, ranges, identities)
    return hashes, pages, proof


def publish(cache, hashes, pages, proof):
    cache.publish_group_page_proof(
        proof,
        request_id="cold",
        generation=1,
        mapping_version=1,
        snapshot=pages,
        partition=0,
        block_hashes=hashes,
    )


def test_joint_proof_only_publishes_materialized_state_suffix():
    cache = setup_cache()
    hashes, pages, proof = checkpoint(cache)
    publish(cache, hashes, pages, proof)
    for spec in cache.group_specs:
        pool = cache._group_pools[spec.name]
        actual = {i for i, h in enumerate(hashes[spec.name]) if (0, h) in pool.hash_to_block}
        item = next(r for r in proof.complete_logical_pages if r.group == spec.name)
        assert actual == set(range(item.start, item.end))
    cache.release_all_group_requests("cold")
    _, hit, _ = cache.acquire_group_prefix_blocks("hot", hashes, max_cache_hit_tokens=1408)
    assert hit == 1408
    cache.release_all_group_requests("hot")


@pytest.mark.parametrize(
    "corruption", ["missing-group", "overclaim-state", "stale-epoch", "wrong-generation", "wrong-mapping"]
)
def test_invalid_proofs_publish_nothing(corruption):
    cache = setup_cache()
    hashes, pages, proof = checkpoint(cache)
    if corruption == "missing-group":
        proof = replace(proof, complete_logical_pages=proof.complete_logical_pages[:-1])
    elif corruption == "overclaim-state":
        proof = replace(
            proof,
            complete_logical_pages=tuple(
                replace(r, start=0) if r.group == "hca_state" else r for r in proof.complete_logical_pages
            ),
        )
    elif corruption == "stale-epoch":
        proof = replace(
            proof,
            page_identities=(
                replace(proof.page_identities[0], allocation_epoch=99),
                *proof.page_identities[1:],
            ),
        )
    elif corruption == "wrong-generation":
        proof = replace(proof, generation=2)
    else:
        proof = replace(proof, mapping_version=2)
    with pytest.raises(ValueError):
        publish(cache, hashes, pages, proof)
    assert all(not p.hash_to_block for p in cache._group_pools.values())


def test_same_page_id_reassignment_invalidates_old_identity():
    cache = setup_cache()
    pages = cache.ensure_group_blocks("a", 512)
    before = cache.group_page_identities(pages, 0)
    cache.release_all_group_requests("a")
    seen = {}
    for i in range(8):
        current = cache.ensure_group_blocks(str(i), 512)
        for page in cache.group_page_identities(current, 0):
            seen[(page.group, page.page_id)] = page.allocation_epoch
        cache.release_all_group_requests(str(i))
    assert any(seen.get((p.group, p.page_id), 0) > p.allocation_epoch for p in before)


@pytest.mark.parametrize("boundary", [0, True, 127, 1409])
def test_invalid_copy_boundary_does_not_allocate_or_leak_references(boundary):
    cache = setup_cache()
    hashes, pages, proof = checkpoint(cache)
    before = [
        (p.spec.name, tuple((b.block_id, b.ref_cnt, b.allocation_epoch) for b in p.blocks))
        for p in cache._group_pools.values()
    ]
    with pytest.raises(ValueError, match="aligned"):
        cache.reserve_prefix_checkpoint("cold", generation=1, boundary=boundary, snapshot=pages)
    after = [
        (p.spec.name, tuple((b.block_id, b.ref_cnt, b.allocation_epoch) for b in p.blocks))
        for p in cache._group_pools.values()
    ]
    assert after == before
    assert not getattr(cache, "_prefix_checkpoint_leases", {})


def test_invalid_native_receipt_does_not_commit_model_state():
    policy = LowlatExecutionPolicy(EngineConfig())
    request = Request("r", [1] * 128, max_new_tokens=7)
    request.execution_state = LowlatScheduleState(physical_end=128)
    reservation = DecodeReservation(1, "r", 1, 1, 6, 256, 6, 0)
    step = ScheduledRequest(request, 1, False, model_execution_plan=reservation)
    receipt = NativeDecodeRetirement(1, "r", 1, 999, 1005, (6,), (5,), 6, 0, 2)
    result = StepResult({"r": [9] * 6}, model_extension=encode_decode_extension(
        LowlatDecodeExtension((1,), (encode_retirement(receipt),))))
    cache = Mock()
    with pytest.raises(ValueError, match="physical"):
        policy.confirm_step(SchedulerOutput(scheduled_requests=[step]), result, cache)
    assert request.execution_state.physical_end == 128
    assert request.execution_state.raw_output_count == 1 and step.confirmed_input_end is None
    cache.publish_group_page_proof.assert_not_called()
