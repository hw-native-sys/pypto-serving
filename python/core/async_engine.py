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
import contextlib
import logging
import os
import queue
import time
from collections.abc import AsyncGenerator, Sequence
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
_DEFAULT_DEEPSEEK_V4_WORKER_STEP_TIMEOUT_SECONDS = 1200.0


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


def _worker_step_timeout_seconds(executor_cls: str = "") -> float:
    default = _DEFAULT_WORKER_STEP_TIMEOUT_SECONDS
    if executor_cls == "PyptoDeepSeekV4Executor":
        default = _DEFAULT_DEEPSEEK_V4_WORKER_STEP_TIMEOUT_SECONDS
    return _positive_env_timeout_seconds("SERVING_WORKER_STEP_TIMEOUT", default)


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
        if self.parallel_config is not None:
            groups = self.parallel_config.replica_device_groups
            if len(groups) == 1:
                return groups[0]
            if 0 <= self.dp_rank < len(groups):
                return groups[self.dp_rank]
            raise ValueError(
                f"dp_rank {self.dp_rank} is outside configured replica groups: "
                f"{len(groups)}"
            )
        if self.device_ids:
            return tuple(int(device) for device in self.device_ids)
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


class ReplicaEngineCore:
    """Engine core for one serving replica.

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
        self._runtime = runtime
        # Block metadata is initialised lazily after the worker reports the
        # actual device-side KV cache page count (computed from remaining
        # NPU memory after model weight upload).
        self.kv_cache_manager = KvCacheManager(
            num_blocks=None,
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
        with profile_span("AsyncLLMEngine.start", cat="serving"):
            process, input_q, output_q, ready_event, num_pages_value = spawn_worker(self.config)
            self._worker_process = process
            self._input_queue = input_q
            self._output_queue = output_q

            logger.info("Waiting for worker to initialize model...")
            try:
                ready = await asyncio.to_thread(ready_event.wait, timeout=600)
                if not ready:
                    raise RuntimeError("Worker failed to initialize within timeout")
            except BaseException:
                await asyncio.to_thread(self._shutdown_worker, timeout=5)
                raise
            logger.info("Worker ready")

            # Synchronise block metadata with the actual device-side KV cache size.
            actual_num_pages = num_pages_value.value
            if actual_num_pages <= 0:
                raise RuntimeError(
                    f"Worker reported invalid KV cache page count: {actual_num_pages}"
                )
            self.kv_cache_manager._init_blocks(actual_num_pages, self._runtime.page_size)
            logger.info(
                "KV cache block pool initialised: num_blocks=%d, block_size=%d",
                actual_num_pages,
                self._runtime.page_size,
            )

        self._running = True
        self._loop_task = asyncio.create_task(self._engine_loop())
        logger.info("ReplicaEngineCore started")

    async def stop(self) -> None:
        """Stop engine loop and worker process."""
        self._running = False
        if self._loop_task is not None:
            await self._loop_task
            self._loop_task = None

        await asyncio.to_thread(self._shutdown_worker, timeout=30)
        logger.info("ReplicaEngineCore stopped")

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
        prompt_token_ids: Sequence[int] | None = None,
    ) -> AsyncGenerator[TokenOutput, None]:
        """Add a request and yield token outputs as they are generated."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is required for request processing")
        with profile_span(
            "ReplicaEngineCore.add_request",
            cat="serving",
            args={"request_id": request_id, "max_new_tokens": config.max_new_tokens},
        ):
            if prompt_token_ids is None:
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
            logger.info(
                "request %s received: prompt=%d tokens, max_new_tokens=%d",
                request_id, len(prompt_token_ids), config.max_new_tokens,
            )
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
                    e2e = time.time() - request.arrival_time
                    n_out = len(request.output_token_ids)
                    logger.info(
                        "request %s finished: prompt=%d out=%d reason=%s e2e=%.2fs (%.1f tok/s)",
                        request_id, len(prompt_token_ids), n_out, output.finish_reason,
                        e2e, (n_out / e2e) if e2e > 0 else 0.0,
                    )
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
                    step_timeout = _worker_step_timeout_seconds(self.config.executor_cls)
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
            request_id = sr.request.request_id
            ctx = self._request_contexts.get(request_id)
            if ctx is not None:
                ctx.queue.put_nowait(
                    TokenOutput(finished=True, finish_reason="error")
                )
            if request_id not in self._pending_free_ids:
                self._pending_free_ids.append(request_id)
            self.scheduler.abort_request(request_id)

    def _shutdown_worker(self, *, timeout: float) -> None:
        input_q = self._input_queue
        process = self._worker_process

        if input_q is not None:
            with contextlib.suppress(Exception):
                input_q.put(WorkerCommand(type="shutdown"))

        if process is not None:
            with contextlib.suppress(Exception):
                process.join(timeout=timeout)
            with contextlib.suppress(Exception):
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=1)

        self._worker_process = None
        self._input_queue = None
        self._output_queue = None


class AsyncLLMEngine:
    """Async serving engine that routes requests across replica cores.

    The engine owns one or more ``ReplicaEngineCore`` instances and exposes the
    server-facing async API: ``start``, ``stop``, ``add_request``,
    ``abort_request``, and ``generate_request_id``. For ``data_parallel_size=1``
    it wraps a single core. For DP>1 it selects a core for each request and
    records request placement so aborts are sent to the correct replica.
    """

    def __init__(
        self,
        config: EngineConfig,
        tokenizer=None,
        eos_token_id: int | None = None,
        bos_token_id: int | None = None,
        *,
        core_factory: Callable[..., ReplicaEngineCore] = ReplicaEngineCore,
    ) -> None:
        parallel = config.parallel_config
        if parallel is None:
            worker_devices = config.worker_device_ids()
            parallel = ParallelConfig(
                tensor_parallel_size=len(worker_devices),
                devices=worker_devices,
            )
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
        self._cores: list[ReplicaEngineCore] = []

        for dp_rank, device_group in enumerate(parallel.replica_device_groups):
            replica_parallel = parallel.for_replica(device_group)
            replica_config = replace(
                config,
                device_id=device_group[0],
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
        await asyncio.gather(*(core.stop() for core in reversed(self._cores)))

    def generate_request_id(self) -> str:
        self._request_counter += 1
        return f"serving-req-{self._request_counter}"

    def pending_token_load(self) -> int:
        return sum(core.pending_token_load() for core in self._cores)

    @property
    def scheduler(self) -> Scheduler:
        return self._single_core().scheduler

    @property
    def kv_cache_manager(self) -> KvCacheManager:
        return self._single_core().kv_cache_manager

    async def add_request(
        self,
        request_id: str,
        prompt: str,
        config,
    ) -> AsyncGenerator[TokenOutput, None]:
        replica_idx = self._select_replica()
        prompt_token_ids = self._tokenize_prompt(prompt)
        request_load = self._estimate_request_load(prompt_token_ids, config)
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
                prompt_token_ids=prompt_token_ids,
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

    def _single_core(self):
        if len(self._cores) != 1:
            raise AttributeError("scheduler and kv_cache_manager are only exposed for single-replica engines")
        return self._cores[0]

    def _tokenize_prompt(self, prompt: str) -> Sequence[int] | None:
        if self.tokenizer is not None:
            prompt_token_ids = self.tokenizer.encode(prompt)
            if not prompt_token_ids and self.bos_token_id is not None:
                prompt_token_ids = [self.bos_token_id]
            if not prompt_token_ids:
                raise ValueError("Prompt tokenization produced no tokens.")
            return prompt_token_ids
        return None

    def _estimate_request_load(self, prompt_token_ids: Sequence[int] | None, config) -> int:
        prompt_tokens = len(prompt_token_ids) if prompt_token_ids is not None else 0
        return prompt_tokens + int(getattr(config, "max_new_tokens", 0))
