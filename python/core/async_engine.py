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
import queue
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

from .kv_cache import KvCacheManager
from python.profile import profile_instant, profile_span
from .scheduler import Request, RequestStatus, Scheduler, SchedulerConfig, SchedulerOutput
from .types import RuntimeConfig, StepOutput, WorkerCommand
from .serving_worker import spawn_worker

logger = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    # Model
    model_id: str = ""
    model_dir: str = ""

    # Device / executor
    platform: str = "a2a3"
    device_id: int = 0
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
    prefix_cache_backend: str = "hash"


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


class AsyncLLMEngine:
    """Async engine with multiprocess worker for NPU execution.

    Architecture:
      Main process: scheduler + API serving + output processing
      Worker process: NPU device + model execution (single-card, extensible to multi-card)
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
            prefix_cache_backend=self.config.prefix_cache_backend,
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
            process, input_q, output_q, ready_event = spawn_worker(self.config)
            self._worker_process = process
            self._input_queue = input_q
            self._output_queue = output_q

            logger.info("Waiting for worker to initialize model...")
            await asyncio.to_thread(ready_event.wait, timeout=600)
            if not ready_event.is_set():
                raise RuntimeError("Worker failed to initialize within timeout")
            logger.info("Worker ready")

        self._running = True
        self._loop_task = asyncio.create_task(self._engine_loop())
        logger.info("AsyncLLMEngine started")

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
        logger.info("AsyncLLMEngine stopped")

    def generate_request_id(self) -> str:
        self._request_counter += 1
        return f"serving-req-{self._request_counter}"

    async def add_request(
        self,
        request_id: str,
        prompt: str,
        config,
    ) -> AsyncGenerator[TokenOutput, None]:
        """Add a request and yield token outputs as they are generated."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is required for request processing")
        with profile_span(
            "AsyncLLMEngine.add_request",
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
                model_id=self.config.model_id,
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
                    step_output: StepOutput = await asyncio.to_thread(
                        self._output_queue.get, timeout=300
                    )
            except queue.Empty:
                logger.error("Worker response timed out (300s)")
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
