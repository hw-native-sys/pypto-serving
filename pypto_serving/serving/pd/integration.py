# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Optional PD composition root; ordinary Serving never loads this module."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

from pypto_serving.model import builtin_pd_adapter_registry
from pypto_serving.model.model_family import detect_model_family, read_model_config
from pypto_serving.config.types import RuntimeConfig
from pypto_serving.serving.engine.async_engine import AsyncLLMEngine, ReplicaEngineCore
from pypto_serving.serving.sched.scheduler import Scheduler
from pypto_serving.serving.server.serving_worker import spawn_worker
from pypto_serving.serving.server.server import create_serving_app
from pypto_serving.tools.profile import get_profiler

from .adapter import ModelRuntimeFacts
from .config import PDRole, load_pd_document, resolve_pd_config
from .http_api import PDHTTPRoutes
from .service import PDServingService
from .worker import PDWorkerServices


@dataclass(frozen=True)
class PrefillChunkReady:
    """Confirmed P chunk plus its exact scheduler block-table snapshot."""

    request_id: str
    chunk_id: int
    start_token: int
    end_token: int
    final: bool
    first_token: int | None
    block_ids_by_group: dict[str, tuple[int, ...]]
    cache_partition: int


def validate_engine_config(config, pd_config) -> None:
    """Reject configurations that violate the selected model contract."""
    contract = pd_config.model_contract
    if config.executor_cls != contract.executor_cls:
        raise ValueError("PD executor differs from the selected model contract")
    if config.enable_prefix_cache != pd_config.prefix_cache_enabled:
        raise ValueError("PD prefix-cache flag differs from the configured role/profile")
    expected_async_scheduling = contract.async_scheduling_for_role(pd_config.role.value)
    if config.resolve_async_scheduling() != expected_async_scheduling:
        raise ValueError("PD scheduling mode differs from the selected model contract")
    runtime = config.runtime_config or RuntimeConfig()
    expected_local_tokens = contract.local_speculative_tokens(pd_config.role.value)
    if runtime.num_speculative_tokens != expected_local_tokens:
        raise ValueError(
            f"PD {pd_config.role.value} requires local num_speculative_tokens="
            f"{expected_local_tokens} under {contract.adapter_id}"
        )
    executor_tokens = config.executor_kwargs.get("num_speculative_tokens", 0)
    if executor_tokens != expected_local_tokens:
        raise ValueError(
            f"PD {pd_config.role.value} executor requires num_speculative_tokens="
            f"{expected_local_tokens} under {contract.adapter_id}"
        )


class PDNodeRuntime:
    """PD orchestration over explicitly supplied Serving capabilities."""

    def __init__(
        self,
        *,
        cache_manager,
        add_request,
        add_prefilled_request,
        abort_request,
        call_worker,
        finish_prefilled_request,
        hold_request,
        resume_request,
    ):
        self.kv_cache_manager = cache_manager
        self.submit_request = add_request
        self.add_adopted_handoff = add_prefilled_request
        self.abort_request = abort_request
        self.call_pd_worker = call_worker
        self.finish_prefilled_request = finish_prefilled_request
        self.hold_request = hold_request
        self.resume_request = resume_request
        self.chunks = {}
        self.chunk_counters = {}

    def add_request(self, request_id, *args, **kwargs):
        if request_id in self.chunks:
            raise ValueError("request already owns a Prefill chunk stream")
        self.chunks[request_id] = asyncio.Queue()

        async def generate():
            from contextlib import aclosing

            try:
                async with aclosing(self.submit_request(request_id, *args, **kwargs)) as stream:
                    async for output in stream:
                        yield output
            finally:
                self.chunks.pop(request_id, None)
                self.chunk_counters.pop(request_id, None)

        return generate()

    async def next_prefill_chunk(self, request_id):
        return await self.chunks[request_id].get()

    def complete_prefill_chunk_transfer(self, request_id):
        self.resume_request(request_id)

    def acknowledge_prefill_handoff(self, request_id):
        self.finish_prefilled_request(request_id)
        self.chunks.pop(request_id, None)
        self.chunk_counters.pop(request_id, None)

    def wrap_results(self, process):
        def complete(scheduler_output, new_tokens, num_draft_tokens=None):
            process(scheduler_output, new_tokens, num_draft_tokens)
            self.publish_chunks(scheduler_output, new_tokens)

        return complete

    def publish_chunks(self, scheduler_output, new_tokens):
        for scheduled in scheduler_output.scheduled_requests:
            if not scheduled.is_prefill:
                continue
            if scheduled.cache_partition is None:
                raise RuntimeError("PD Prefill chunk has no cache partition")
            request = scheduled.request
            start = scheduled.num_computed_tokens
            end = start + scheduled.num_new_tokens
            final = end >= request.num_prompt_tokens
            token_value = new_tokens.get(request.request_id)
            if isinstance(token_value, list):
                first_token = int(token_value[0]) if token_value else None
            elif token_value is None:
                first_token = None
            else:
                first_token = int(token_value)
            if final and first_token is None:
                raise RuntimeError("terminal PD Prefill did not return the first token")
            chunk_id = self.chunk_counters.get(request.request_id, 0)
            self.chunk_counters[request.request_id] = chunk_id + 1
            self.hold_request(request.request_id)
            try:
                chunk_queue = self.chunks[request.request_id]
            except KeyError as exc:
                raise RuntimeError("PD Prefill request lost its chunk queue") from exc
            chunk_queue.put_nowait(
                PrefillChunkReady(
                    request_id=request.request_id,
                    chunk_id=chunk_id,
                    start_token=start,
                    end_token=end,
                    final=final,
                    first_token=first_token,
                    block_ids_by_group={
                        name: tuple(block_ids) for name, block_ids in scheduled.block_ids_by_group.items()
                    },
                    cache_partition=scheduled.cache_partition,
                )
            )


def prefill_scheduler(*, config, kv_cache_manager):
    return Scheduler(config=replace(config, stop_after_prefill=True), kv_cache_manager=kv_cache_manager)


class PDApplication:
    def __init__(self, config, pd_config, tokenizer, generate_config):
        validate_engine_config(config, pd_config)
        self.engine = AsyncLLMEngine(
            config,
            tokenizer,
            core_factory=partial(
                ReplicaEngineCore,
                scheduler_factory=prefill_scheduler if pd_config.role is PDRole.PREFILL else Scheduler,
                worker_factory=partial(spawn_worker, services_factory=PDWorkerServices),
            ),
        )
        core = self.engine.single_core()
        self.node = PDNodeRuntime(
            cache_manager=core.kv_cache_manager,
            add_request=core.add_request,
            add_prefilled_request=core.add_prefilled_request,
            abort_request=core.abort_request,
            call_worker=core.call_worker,
            finish_prefilled_request=core.finish_prefilled_request,
            hold_request=core.scheduler.hold_request,
            resume_request=core.scheduler.resume_request,
        )
        if pd_config.role is PDRole.PREFILL:
            core.configure_result_handler(self.node.wrap_results)
        self.service = PDServingService(self.node, pd_config, eos_token_id=self.engine.eos_token_id)
        self.app = create_serving_app(
            self.engine,
            config.model_id,
            generate_config,
            route_factory=partial(self.install_routes, config=pd_config),
        )

    def install_routes(self, server, *, config):
        return PDHTTPRoutes(
            server.app,
            self.service,
            config,
            prepare_completion=server.prepare_completion,
            prepare_chat=server.prepare_chat,
            resolve_prompt_tokens=server.resolve_prompt_tokens,
            start_profile=server.start_profile,
            stop_profile=server.stop_profile,
            profiling_enabled=get_profiler(initially_active=False).enabled,
        )

    async def start(self):
        await self.engine.start()
        try:
            await self.service.start()
        except BaseException:
            await self.stop()
            raise

    async def stop(self):
        try:
            await self.service.close()
        finally:
            await self.engine.stop()


def build_pd_launch(args, build_config, *, model_variant):
    if args.prompt:
        raise ValueError("PD nodes serve Router requests; --prompt is only available in ordinary mode")
    if not args.pd_config:
        raise ValueError("--pd-role requires --pd-config")
    config = build_config(args)
    facts = ModelRuntimeFacts(
        model_family=detect_model_family(read_model_config(config.model_dir)),
        model_variant=model_variant,
        num_speculative_tokens=config.runtime_config.num_speculative_tokens,
    )
    adapter = builtin_pd_adapter_registry().select(facts)
    pd = resolve_pd_config(
        load_pd_document(args.pd_config),
        role=PDRole(args.pd_role),
        node_id=args.pd_node_id,
        model_revision=args.served_model_name or Path(args.model).name,
        model_adapter=adapter,
    )
    local_tokens = pd.model_contract.local_speculative_tokens(pd.role.value)
    config = replace(
        config,
        runtime_config=replace(config.runtime_config, num_speculative_tokens=local_tokens),
        executor_kwargs={
            **config.executor_kwargs,
            "num_speculative_tokens": local_tokens,
            "runner_extension_factory": adapter.worker_factory(pd.worker_config()),
        },
        enable_prefix_cache=pd.prefix_cache_enabled,
        async_scheduling=pd.model_contract.async_scheduling_for_role(pd.role.value),
    )
    validate_engine_config(config, pd)
    return config, partial(PDApplication, pd_config=pd)
