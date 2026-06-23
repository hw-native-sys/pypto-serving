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
import logging
import os
import queue
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field, replace
from typing import Callable

from .kv_cache import KvCacheManager
from python.profile import profile_instant, profile_span
from .parallel import ParallelConfig
from .scheduler import Request, RequestStatus, Scheduler, SchedulerConfig, SchedulerOutput
from .types import RuntimeConfig, StepOutput, WorkerCommand
from .serving_worker import spawn_worker

logger = logging.getLogger(__name__)
_DEFAULT_WORKER_INIT_TIMEOUT_SECONDS = 600.0
_DEFAULT_WORKER_STEP_TIMEOUT_SECONDS = 300.0


def _positive_env_timeout_seconds(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        timeout = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive number of seconds") from exc
    if timeout <= 0:
        raise ValueError(f"{name} must be a positive number of seconds")
    return timeout


def _worker_init_timeout_seconds() -> float:
    return _positive_env_timeout_seconds("PYPTO_WORKER_INIT_TIMEOUT", _DEFAULT_WORKER_INIT_TIMEOUT_SECONDS)


def _worker_step_timeout_seconds() -> float:
    return _positive_env_timeout_seconds("PYPTO_WORKER_STEP_TIMEOUT", _DEFAULT_WORKER_STEP_TIMEOUT_SECONDS)


@dataclass
class EngineConfig:
    # Model
    model_id: str = ""
    model_dir: str = ""

    # Device / executor
    platform: str = "a2a3"
    device_id: int = 0
    device_ids: tuple[int, ...] = ()
    parallel_config: ParallelConfig | None = None
    dp_rank: int = 0
    executor_cls: str = "PyptoQwen14BExecutor"
    executor_kwargs: dict = field(default_factory=dict)

    # Runtime
    runtime_config: RuntimeConfig | None = None

    # Scheduler / serving
    max_num_running_reqs: int = 32
    max_num_scheduled_tokens: int = 4096
    long_prefill_token_threshold: int = 2048
    engine_loop_interval: float = 0.001

    # Feature flags
    enable_prefix_cache: bool = True
    enable_chunk_prefill: bool = True

    def worker_device_ids(self) -> tuple[int, ...]:
        """Return the device ids this engine worker should own."""
        if self.device_ids:
            return tuple(int(device) for device in self.device_ids)
        if self.parallel_config is not None and self.parallel_config.data_parallel_size == 1:
            return self.parallel_config.replica_device_groups[0]
        return (int(self.device_id),)


@dataclass
class _RequestContext:
    request: Request
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)


@dataclass
class TokenOutput:
    token_id: int | None = None
    text: str = ""
    finished: bool = False
    finish_reason: str = ""


class DPEngineCore:
    """Engine core for one data-parallel replica.

    A core owns all mutable serving state for one replica: one scheduler, one
    KV cache manager, one worker process, one executor/model runtime, and one
    tensor-parallel device group. Requests assigned to this core are scheduled
    only against this core's local KV cache and worker state.
    """

    def __init__(
        self,
        config: EngineConfig,
        tokenizer=None,
        eos_token_id: int | None = None,
        bos_token_id: int | None = None
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id

        runtime = self.config.runtime_config or RuntimeConfig()
        block_size = runtime.page_size
        max_blocks_per_seq = (runtime.max_seq_len + block_size - 1) // block_size
        num_blocks = runtime.total_kv_pages or runtime.max_batch_size * max_blocks_per_seq
        self.kv_cache_manager = KvCacheManager(
            num_blocks=num_blocks,
            block_size=block_size,
            enable_prefix_cache=self.config.enable_prefix_cache,
        )

        scheduler_config = SchedulerConfig(
            max_num_running_reqs=self.config.max_num_running_reqs,
            max_num_scheduled_tokens=self.config.max_num_scheduled_tokens,
            long_prefill_token_threshold=self.config.long_prefill_token_threshold,
            max_seq_len=runtime.max_seq_len,
            enable_prefix_cache=self.config.enable_prefix_cache,
            enable_chunk_prefill=self.config.enable_chunk_prefill,
        )
        self.scheduler = Scheduler(config=scheduler_config, kv_cache_manager=self.kv_cache_manager)

        self._request_contexts: dict[str, _RequestContext] = {}
        self._running = False
        self._loop_task: asyncio.Task | None = None
        self._request_counter = 0
        self._pending_free_ids: list[str] = []

        self._worker_process = None
        self._input_queue = None
        self._output_queue = None

    async def start(self) -> None:
        """Start worker process and engine loop."""
        try:
            with profile_span("DPEngineCore.start", cat="serving"):
                process, input_q, output_q, ready_event = spawn_worker(self.config)
                self._worker_process = process
                self._input_queue = input_q
                self._output_queue = output_q

                logger.info("Waiting for worker to initialize model...")
                init_timeout = _worker_init_timeout_seconds()
                await asyncio.to_thread(ready_event.wait, timeout=init_timeout)
                if not ready_event.is_set():
                    raise RuntimeError(f"Worker failed to initialize within {init_timeout:g}s")
                logger.info("Worker ready")

            self._running = True
            self._loop_task = asyncio.create_task(self._engine_loop())
            logger.info("DPEngineCore started")
        except BaseException:
            await self.stop()
            raise

    async def stop(self) -> None:
        """Stop engine loop and worker process."""
        self._running = False
        if self._loop_task is not None:
            await self._loop_task
            self._loop_task = None

        if self._input_queue is not None:
            self._input_queue.put(WorkerCommand(type="shutdown"))

        if self._worker_process is not None:
            self._worker_process.join(timeout=30)
            if self._worker_process.is_alive():
                self._worker_process.terminate()
            self._worker_process = None
        logger.info("DPEngineCore stopped")

    def generate_request_id(self) -> str:
        self._request_counter += 1
        return f"serving-req-{self._request_counter}"

    def pending_token_load(self) -> int:
        """Estimate unfinished work for routing new data-parallel requests."""
        load = 0
        for request in self.scheduler.requests.values():
            if request.status.is_finished:
                continue
            prompt_remaining = max(0, request.num_prompt_tokens - request.num_computed_tokens)
            generation_remaining = max(0, request.max_new_tokens - len(request.output_token_ids))
            load += prompt_remaining + generation_remaining
        return load

    async def add_request(
        self,
        request_id: str,
        prompt: str,
        config,
        *,
        on_queued: Callable[[], None] | None = None,
    ) -> AsyncGenerator[TokenOutput, None]:
        """Add a request and yield token outputs as they are generated."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is required for request processing")
        with profile_span(
            "DPEngineCore.add_request",
            cat="serving",
            args={"request_id": request_id, "max_new_tokens": config.max_new_tokens},
        ):
            prompt_token_ids = self.tokenizer.encode(prompt)
            if not prompt_token_ids and self.bos_token_id is not None:
                prompt_token_ids = [self.bos_token_id]
            if not prompt_token_ids:
                raise ValueError("Prompt tokenization produced no tokens.")

            request = Request(
                request_id=request_id,
                prompt_token_ids=prompt_token_ids,
                max_new_tokens=config.max_new_tokens,
                arrival_time=time.time(),
                stop_strings=tuple(config.stop) if config.stop else (),
                eos_token_id=self.eos_token_id,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
            )

            ctx = _RequestContext(request=request)
            self._request_contexts[request_id] = ctx
            self.scheduler.add_request(request)
            if on_queued is not None:
                on_queued()
            profile_instant(
                "request.queued",
                cat="serving",
                args={"request_id": request_id, "prompt_tokens": len(prompt_token_ids)},
            )

        try:
            while True:
                output: TokenOutput = await ctx.queue.get()
                yield output
                if output.finished:
                    break
        finally:
            if request_id in self._request_contexts:
                self._request_contexts.pop(request_id, None)
                self.scheduler.abort_request(request_id)

    async def abort_request(self, request_id: str) -> None:
        self.scheduler.abort_request(request_id)
        ctx = self._request_contexts.pop(request_id, None)
        if ctx is not None:
            await ctx.queue.put(
                TokenOutput(finished=True, finish_reason="FINISHED_ABORTED")
            )

    async def _engine_loop(self) -> None:
        """Main loop: schedule -> send to worker -> receive results -> dispatch."""
        logger.info("Engine loop started")
        while self._running:
            if not self.scheduler.has_work():
                await asyncio.sleep(self.config.engine_loop_interval)
                continue

            with profile_span("scheduler.schedule", cat="scheduler"):
                scheduler_output = self.scheduler.schedule()
            if scheduler_output.is_empty:
                await asyncio.sleep(self.config.engine_loop_interval)
                continue

            finished_ids = self._pending_free_ids.copy()
            self._pending_free_ids.clear()
            with profile_span(
                "scheduler.queue_worker_step",
                cat="scheduler",
                args={"scheduled": len(scheduler_output.scheduled_requests)},
            ):
                self._input_queue.put(
                    WorkerCommand(
                        type="step",
                        scheduler_output=scheduler_output,
                        finished_request_ids=finished_ids or None,
                    )
                )

            try:
                with profile_span("scheduler.wait_worker_output", cat="scheduler"):
                    step_timeout = _worker_step_timeout_seconds()
                    step_output: StepOutput = await asyncio.to_thread(
                        self._output_queue.get, timeout=step_timeout
                    )
            except queue.Empty:
                logger.error(f"Worker response timed out ({step_timeout:g}s)")
                self._handle_step_error(scheduler_output)
                continue

            if step_output.error:
                logger.error(f"Worker returned error: {step_output.error}")
                self._handle_step_error(scheduler_output)
                continue

            with profile_span(
                "scheduler.process_step_output",
                cat="scheduler",
                args={"new_tokens": len(step_output.new_tokens)},
            ):
                self._process_step_output(scheduler_output, step_output)

        logger.info("Engine loop stopped")

    def _process_step_output(
        self, scheduler_output: SchedulerOutput, step_output: StepOutput
    ) -> None:
        """Process worker results: update scheduler state, push tokens to request queues."""
        request_outputs = self.scheduler.update_from_output(
            scheduler_output, step_output.new_tokens
        )

        for req_output in request_outputs:
            ctx = self._request_contexts.get(req_output.request_id)
            if ctx is None:
                continue

            text = ""
            if ctx.request.output_token_ids:
                text = self.tokenizer.decode(ctx.request.output_token_ids)

            if not req_output.finished and ctx.request.stop_strings:
                for stop in ctx.request.stop_strings:
                    if stop and text.endswith(stop):
                        req_output.finished = True
                        req_output.finish_reason = "FINISHED_STOP"
                        self.scheduler.finish_request(
                            req_output.request_id, RequestStatus.FINISHED_STOP
                        )
                        break

            if req_output.finished:
                self._pending_free_ids.append(req_output.request_id)

            token_output = TokenOutput(
                token_id=req_output.new_token_id,
                text=text,
                finished=req_output.finished,
                finish_reason=req_output.finish_reason,
            )
            ctx.queue.put_nowait(token_output)

    def _handle_step_error(self, scheduler_output: SchedulerOutput) -> None:
        """On worker error, abort all requests in the failed batch."""
        for sr in scheduler_output.scheduled_requests:
            ctx = self._request_contexts.get(sr.request.request_id)
            if ctx is not None:
                ctx.queue.put_nowait(
                    TokenOutput(finished=True, finish_reason="error")
                )
            self.scheduler.abort_request(sr.request.request_id)


class AsyncLLMEngineClient:
    """Async serving client that routes requests across DP engine cores.

    The client owns one or more ``DPEngineCore`` instances and exposes the
    server-facing async API: ``start``, ``stop``, ``add_request``,
    ``abort_request``, and ``generate_request_id``. For ``data_parallel_size=1``
    it can still wrap a single core, while for DP>1 it selects a core for each
    request and records request placement so aborts are sent to the correct
    replica. It does not own scheduler, KV cache, model, or worker state itself.
    """

    def __init__(
        self,
        config: EngineConfig,
        tokenizer=None,
        eos_token_id: int | None = None,
        bos_token_id: int | None = None,
        *,
        core_factory: Callable[..., DPEngineCore] = DPEngineCore,
    ) -> None:
        parallel = config.parallel_config
        if parallel is None:
            parallel = ParallelConfig(devices=config.worker_device_ids())
            config = replace(config, parallel_config=parallel)

        self.config = config
        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.parallel_config = parallel
        self._request_counter = 0
        self._route_counter = 0
        self._request_to_replica: dict[str, int] = {}
        self._route_extra_load = [0 for _ in parallel.replica_device_groups]
        self._cores: list[DPEngineCore] = []

        for dp_rank, device_group in enumerate(parallel.replica_device_groups):
            replica_parallel = parallel.for_replica(device_group)
            replica_config = replace(
                config,
                device_id=device_group[0],
                device_ids=device_group,
                parallel_config=replica_parallel,
                dp_rank=dp_rank,
            )
            self._cores.append(
                core_factory(
                    config=replica_config,
                    tokenizer=tokenizer,
                    eos_token_id=eos_token_id,
                    bos_token_id=bos_token_id,
                )
            )

    async def start(self) -> None:
        """Start all DP engine cores in parallel."""
        tasks = [asyncio.create_task(core.start()) for core in self._cores]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self.stop()
            raise

    async def stop(self) -> None:
        """Stop all DP engine cores."""
        for core in reversed(self._cores):
            await core.stop()

    def generate_request_id(self) -> str:
        self._request_counter += 1
        return f"serving-req-{self._request_counter}"

    def pending_token_load(self) -> int:
        return sum(core.pending_token_load() for core in self._cores)

    async def add_request(
        self,
        request_id: str,
        prompt: str,
        config,
    ) -> AsyncGenerator[TokenOutput, None]:
        replica_idx = self._select_replica()
        request_load = self._estimate_request_load(prompt, config)
        self._route_extra_load[replica_idx] += request_load
        self._request_to_replica[request_id] = replica_idx
        route_extra_active = True

        def clear_route_extra_load() -> None:
            nonlocal route_extra_active
            if not route_extra_active:
                return
            self._route_extra_load[replica_idx] = max(
                0,
                self._route_extra_load[replica_idx] - request_load,
            )
            route_extra_active = False

        try:
            core = self._cores[replica_idx]
            async for output in core.add_request(
                request_id,
                prompt,
                config,
                on_queued=clear_route_extra_load,
            ):
                yield output
        finally:
            self._request_to_replica.pop(request_id, None)
            clear_route_extra_load()

    async def abort_request(self, request_id: str) -> None:
        replica_idx = self._request_to_replica.get(request_id)
        if replica_idx is not None:
            await self._cores[replica_idx].abort_request(request_id)
            return
        for core in self._cores:
            await core.abort_request(request_id)

    def _select_replica(self) -> int:
        loads = [
            core.pending_token_load() + self._route_extra_load[idx]
            for idx, core in enumerate(self._cores)
        ]
        replica_count = len(self._cores)
        ordered = [
            (loads[idx], (idx - self._route_counter) % replica_count, idx)
            for idx in range(replica_count)
        ]
        replica_idx = min(ordered)[2]
        self._route_counter = (replica_idx + 1) % replica_count
        return replica_idx

    def _estimate_request_load(self, prompt: str, config) -> int:
        prompt_tokens = 0
        if self.tokenizer is not None:
            prompt_tokens = len(self.tokenizer.encode(prompt))
        return prompt_tokens + int(getattr(config, "max_new_tokens", 0))
