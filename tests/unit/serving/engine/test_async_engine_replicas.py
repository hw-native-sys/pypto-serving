# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import asyncio

import pytest

from pypto_serving.config.parallel import ParallelConfig
from pypto_serving.config.types import GenerateConfig
from pypto_serving.serving.engine.async_engine import AsyncLLMEngine, EngineConfig, TokenOutput


def test_async_llm_engine_routes_to_least_pending_tokens():
    asyncio.run(_check_async_llm_engine_routes_to_least_pending_tokens())


def test_async_llm_engine_does_not_double_count_admitted_requests():
    asyncio.run(_check_async_llm_engine_does_not_double_count_admitted_requests())


def test_async_llm_engine_cleans_up_failed_startup():
    asyncio.run(_check_async_llm_engine_cleans_up_failed_startup())


def test_async_llm_engine_generates_structured_offline_batch():
    asyncio.run(_check_async_llm_engine_generates_structured_offline_batch())


def test_async_llm_engine_forwards_agent_identity_to_selected_replica():
    asyncio.run(_check_async_llm_engine_forwards_agent_identity_to_selected_replica())


def test_engine_config_resolves_model_runtime_requirements():
    default_runtime = EngineConfig().resolve_runtime_config()
    assert default_runtime.requires_homogeneous_prefill_decode is False

    deepseek_runtime = EngineConfig(
        executor_cls="PyptoDeepSeekV4Executor"
    ).resolve_runtime_config()
    assert deepseek_runtime.requires_homogeneous_prefill_decode is True


def test_agentinfer_enforce_rejects_multi_replica_engine_config():
    with pytest.raises(ValueError, match="single-replica"):
        AsyncLLMEngine(
            EngineConfig(
                agentinfer_mode="enforce",
                parallel_config=ParallelConfig(
                    data_parallel_size=2, tensor_parallel_size=1, devices=(0, 1),
                ),
            ),
            tokenizer=_Tokenizer(),
        )


async def _check_async_llm_engine_routes_to_least_pending_tokens():
    created: list[_FakeCore] = []
    tokenizer = _Tokenizer()

    def factory(**kwargs):
        core = _FakeCore(**kwargs)
        created.append(core)
        return core

    engine = AsyncLLMEngine(
        EngineConfig(
            model_id="model",
            model_dir="/tmp/model",
            parallel_config=ParallelConfig(
                data_parallel_size=2,
                tensor_parallel_size=1,
                devices=(0, 1),
            ),
        ),
        tokenizer=tokenizer,
        core_factory=factory,
    )
    created[0].load = 10
    created[1].load = 1

    outputs = [
        output
        async for output in engine.add_request(
            "req-0",
            "hello world",
            GenerateConfig(max_new_tokens=4),
        )
    ]

    assert [fake.config.worker_device_ids() for fake in created] == [(0,), (1,)]
    assert created[0].requests == []
    assert created[1].requests == ["req-0"]
    assert created[1].prompt_token_ids == [[1, 1]]
    assert tokenizer.encode_calls == 1
    assert outputs == [TokenOutput(finished=True, finish_reason="FINISHED_LENGTH")]


async def _check_async_llm_engine_does_not_double_count_admitted_requests():
    created: list[_BlockingCore] = []

    def factory(**kwargs):
        core = _BlockingCore(**kwargs)
        created.append(core)
        return core

    engine = AsyncLLMEngine(
        EngineConfig(
            model_id="model",
            model_dir="/tmp/model",
            parallel_config=ParallelConfig(
                data_parallel_size=2,
                tensor_parallel_size=1,
                devices=(0, 1),
            ),
        ),
        tokenizer=_Tokenizer(),
        core_factory=factory,
    )
    created[1].load = 5

    first_task = asyncio.create_task(
        _collect_outputs(
            engine.add_request(
                "req-0",
                "long prompt that should not remain route extra load",
                GenerateConfig(max_new_tokens=4),
            )
        )
    )
    await created[0].admitted.wait()

    outputs = [
        output
        async for output in engine.add_request(
            "req-1",
            "short prompt",
            GenerateConfig(max_new_tokens=1),
        )
    ]

    created[0].release.set()
    await first_task

    assert engine._route_extra_load == [0, 0]
    assert created[0].requests == ["req-0", "req-1"]
    assert created[1].requests == []
    assert outputs == [TokenOutput(finished=True, finish_reason="FINISHED_LENGTH")]


async def _check_async_llm_engine_cleans_up_failed_startup():
    created: list[_StartupCore] = []

    def factory(**kwargs):
        core = _StartupCore(should_fail=len(created) == 1, **kwargs)
        created.append(core)
        return core

    engine = AsyncLLMEngine(
        EngineConfig(
            model_id="model",
            model_dir="/tmp/model",
            parallel_config=ParallelConfig(
                data_parallel_size=2,
                tensor_parallel_size=1,
                devices=(0, 1),
            ),
        ),
        tokenizer=_Tokenizer(),
        core_factory=factory,
    )

    with pytest.raises(RuntimeError, match="startup failed"):
        await engine.start()

    assert [core.started for core in created] == [True, True]
    assert [core.stopped for core in created] == [True, True]


async def _check_async_llm_engine_generates_structured_offline_batch():
    engine = AsyncLLMEngine(
        EngineConfig(
            model_id="model",
            model_dir="/tmp/model",
            parallel_config=ParallelConfig(devices=(0,)),
        ),
        tokenizer=_Tokenizer(),
        core_factory=_ResultCore,
    )

    results = await engine.generate_batch(
        ["first", "second"],
        GenerateConfig(max_new_tokens=2, stream=False),
    )

    assert [result.text for result in results] == ["FIRST", "SECOND"]
    assert [result.token_ids for result in results] == [[10, 11], [10, 11]]
    assert [result.finish_reason for result in results] == ["length", "length"]

    with pytest.raises(ValueError, match="stream=False"):
        await engine.generate_result("streaming", GenerateConfig(stream=True))


async def _check_async_llm_engine_forwards_agent_identity_to_selected_replica():
    engine = AsyncLLMEngine(
        EngineConfig(model_id="model", parallel_config=ParallelConfig(devices=(0,))),
        tokenizer=_Tokenizer(), core_factory=_FakeCore,
    )
    for request_id, session_id in (("round-1", "session-a"), ("round-2", "session-a"),
                                   ("other", "session-b")):
        await _collect_outputs(engine.add_request(
            request_id, "hello", GenerateConfig(max_new_tokens=1),
            agent_session_id=session_id,
        ))
    await _collect_outputs(engine.add_request("plain", "hello", GenerateConfig(max_new_tokens=1)))
    assert engine._cores[0].session_ids == ["session-a", "session-a", "session-b", None]
    assert engine._request_to_replica == {}


async def _collect_outputs(outputs):
    return [output async for output in outputs]


class _Tokenizer:
    def __init__(self) -> None:
        self.encode_calls = 0
        self.eos_token_id = 0
        self.bos_token_id = 1

    def encode(self, text: str) -> list[int]:
        self.encode_calls += 1
        return [1 for _ in text.split()]


class _FakeCore:
    def __init__(self, *, config, tokenizer=None, eos_token_id=None, bos_token_id=None) -> None:
        self.config = config
        self.load = 0
        self.requests: list[str] = []
        self.prompt_token_ids: list[list[int] | None] = []
        self.session_ids: list[str | None] = []

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    def pending_token_load(self) -> int:
        return self.load

    async def add_request(
        self, request_id: str, prompt: str, config, *, on_queued=None, prompt_token_ids=None,
        agent_session_id=None,
    ):
        self.requests.append(request_id)
        self.prompt_token_ids.append(list(prompt_token_ids) if prompt_token_ids is not None else None)
        self.session_ids.append(agent_session_id)
        if on_queued is not None:
            on_queued()
        yield TokenOutput(finished=True, finish_reason="FINISHED_LENGTH")

    async def abort_request(self, request_id: str) -> None:
        return None


class _BlockingCore(_FakeCore):
    def __init__(self, *, config, tokenizer=None, eos_token_id=None, bos_token_id=None) -> None:
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
        )
        self.admitted = asyncio.Event()
        self.release = asyncio.Event()

    async def add_request(
        self, request_id: str, prompt: str, config, *, on_queued=None, prompt_token_ids=None
    ):
        self.requests.append(request_id)
        self.prompt_token_ids.append(list(prompt_token_ids) if prompt_token_ids is not None else None)
        self.load = 1
        if on_queued is not None:
            on_queued()
        self.admitted.set()
        if request_id == "req-0":
            await self.release.wait()
        yield TokenOutput(finished=True, finish_reason="FINISHED_LENGTH")


class _ResultCore(_FakeCore):
    async def add_request(
        self, request_id: str, prompt: str, config, *, on_queued=None, prompt_token_ids=None
    ):
        if on_queued is not None:
            on_queued()
        yield TokenOutput(
            text=prompt.upper(),
            finished=True,
            finish_reason="FINISHED_LENGTH",
            prompt_tokens=len(prompt_token_ids),
            completion_tokens=2,
            token_ids=(10, 11),
        )


class _StartupCore(_FakeCore):
    def __init__(
        self,
        *,
        config,
        tokenizer=None,
        eos_token_id=None,
        bos_token_id=None,
        should_fail: bool = False,
    ) -> None:
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
        )
        self.should_fail = should_fail
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True
        if self.should_fail:
            raise RuntimeError("startup failed")

    async def stop(self) -> None:
        self.stopped = True


@pytest.mark.parametrize("bound", [False, True])
@pytest.mark.parametrize("with_parser", [False, True])
def test_metrics_have_one_owner_with_custom_core_fallback(bound, with_parser):
    from unittest.mock import Mock
    from pypto_serving.serving.reasoning import OutputParserSpec

    parser_spec = OutputParserSpec("deepseek_v4", "reasoning") if with_parser else None

    class ParserCore(_ResultCore):
        async def add_request(self, request_id, *args, **kwargs):
            if parser_spec is None:
                assert "output_parser_spec" not in kwargs
            else:
                assert kwargs.pop("output_parser_spec") is parser_spec
            async for output in super().add_request(request_id, *args, **kwargs):
                yield output

    class BoundCore(ParserCore):
        def set_stat_logger(self, logger, index):
            self.logger, self.index = logger, index

        async def add_request(self, request_id, *args, **kwargs):
            async for output in super().add_request(request_id, *args, **kwargs):
                self.logger.record_output(self.index, request_id, completion_tokens=output.completion_tokens)
                self.logger.finish_request(self.index, request_id, output.finish_reason)
                yield output

    async def check():
        engine = AsyncLLMEngine(
            EngineConfig(model_id="model", model_dir="/tmp/model",
                         parallel_config=ParallelConfig(devices=(0,))),
            tokenizer=_Tokenizer(), core_factory=BoundCore if bound else ParserCore,
        )
        engine.metrics.record_output = Mock(wraps=engine.metrics.record_output)
        engine.metrics.finish_request = Mock(wraps=engine.metrics.finish_request)
        await _collect_outputs(engine.add_request(
            "r", "hello", GenerateConfig(max_new_tokens=2), output_parser_spec=parser_spec,
        ))
        assert engine.metrics.record_output.call_count == 1
        assert engine.metrics.finish_request.call_count == 1
        counters = engine.metrics.snapshot()["replicas"][0]["counters"]
        assert counters["generation_tokens"] == 2
        assert counters["requests_finished"] == 1
    asyncio.run(check())
