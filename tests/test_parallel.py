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
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import python.cli.main as cli
from python.core.async_engine import AsyncLLMEngine, EngineConfig, TokenOutput
from python.core.parallel import ParallelConfig, parse_device_ids
from python.core.types import GenerateConfig


def _parse_cli_args(argv: list[str]):
    return cli.build_parser().parse_args(argv)


def test_parallel_config_groups_dp_replicas_into_tp_groups():
    config = ParallelConfig(
        data_parallel_size=2,
        tensor_parallel_size=2,
        devices=(0, 1, 2, 3),
    )

    assert config.replica_device_groups == ((0, 1), (2, 3))
    assert config.for_replica((2, 3)).data_parallel_size == 1
    assert config.for_replica((2, 3)).devices == (2, 3)


def test_parallel_config_rejects_unsupported_modes():
    with pytest.raises(ValueError, match="pipeline_parallel_size"):
        ParallelConfig(pipeline_parallel_size=2)

    with pytest.raises(ValueError, match="expert parallel"):
        ParallelConfig(enable_expert_parallel=True)

    with pytest.raises(ValueError, match="duplicates"):
        ParallelConfig(data_parallel_size=1, tensor_parallel_size=2, devices=(0, 0))


def test_parse_device_ids_uses_default_device():
    assert parse_device_ids(None, default_device=3) == (3,)
    assert parse_device_ids("0, 2,4", default_device=3) == (0, 2, 4)


def test_build_serving_engine_config_uses_parallel_config_for_devices(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    args = _parse_cli_args([
        "--model",
        str(model_dir),
        "--devices",
        "0,1,2,3",
        "--dp",
        "2",
        "--tp",
        "2",
    ])

    config = cli.build_serving_engine_config(args)

    assert config.device_id == 0
    assert config.device_ids == ()
    assert config.worker_device_ids() == (0, 1)
    assert config.parallel_config.data_parallel_size == 2
    assert config.parallel_config.tensor_parallel_size == 2
    assert config.parallel_config.replica_device_groups == ((0, 1), (2, 3))


def test_build_serving_engine_config_rejects_invalid_parallel_topology(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    args = _parse_cli_args([
        "--model",
        str(model_dir),
        "--devices",
        "0,1,2",
        "--dp",
        "2",
        "--tp",
        "2",
    ])

    with pytest.raises(ValueError, match="number of devices"):
        cli.build_serving_engine_config(args)


def test_parser_rejects_unsupported_parallel_flags(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    with pytest.raises(SystemExit):
        _parse_cli_args(["--model", str(model_dir), "--pp", "2"])


def test_async_llm_engine_routes_to_least_pending_tokens():
    asyncio.run(_check_async_llm_engine_routes_to_least_pending_tokens())


def test_async_llm_engine_does_not_double_count_admitted_requests():
    asyncio.run(_check_async_llm_engine_does_not_double_count_admitted_requests())


def test_async_llm_engine_cleans_up_failed_startup():
    asyncio.run(_check_async_llm_engine_cleans_up_failed_startup())


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


async def _collect_outputs(outputs):
    return [output async for output in outputs]


class _Tokenizer:
    def __init__(self) -> None:
        self.encode_calls = 0

    def encode(self, text: str) -> list[int]:
        self.encode_calls += 1
        return [1 for _ in text.split()]


class _FakeCore:
    def __init__(self, *, config, tokenizer=None, eos_token_id=None, bos_token_id=None) -> None:
        self.config = config
        self.load = 0
        self.requests: list[str] = []
        self.prompt_token_ids: list[list[int] | None] = []

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    def pending_token_load(self) -> int:
        return self.load

    async def add_request(self, request_id: str, prompt: str, config, *, on_queued=None, prompt_token_ids=None):
        self.requests.append(request_id)
        self.prompt_token_ids.append(list(prompt_token_ids) if prompt_token_ids is not None else None)
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

    async def add_request(self, request_id: str, prompt: str, config, *, on_queued=None, prompt_token_ids=None):
        self.requests.append(request_id)
        self.prompt_token_ids.append(list(prompt_token_ids) if prompt_token_ids is not None else None)
        self.load = 1
        if on_queued is not None:
            on_queued()
        self.admitted.set()
        if request_id == "req-0":
            await self.release.wait()
        yield TokenOutput(finished=True, finish_reason="FINISHED_LENGTH")


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
