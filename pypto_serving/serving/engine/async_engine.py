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
import queue
import time
from collections import deque
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass, field, replace
from typing import Callable

from pypto_serving.config.parallel import ParallelConfig
from pypto_serving.config.types import GenerateConfig, GenerateResult, RuntimeConfig
from pypto_serving.observability import InMemoryStatLogger, IterationStats, SchedulerStats
from pypto_serving.serving.memory.kv_cache import KvCacheManager
from pypto_serving.serving.reasoning import (
    OutputParserSpec, ParsedToolCall, ToolCallDelta, create_output_parser,
)
from pypto_serving.serving.utils.env import (
    worker_init_timeout_seconds,
    worker_step_timeout_seconds,
)
from pypto_serving.serving.utils.gc_utils import freeze_gc_heap
from pypto_serving.serving.sched.scheduler import (
    Request,
    RequestStatus,
    Scheduler,
    SchedulerConfig,
    SchedulerOutput,
)
from pypto_serving.serving.pd.worker_api import MAX_WORKER_PD_PAYLOAD_BYTES
from pypto_serving.serving.server.ipc import (
    PLACEHOLDER_TOKEN,
    DecodeRequest,
    NewRequestData,
    PDWorkerCommand,
    PrefillRequest,
    ProfileCommand,
    ShutdownCommand,
    StepCommand,
    decode_profile_result,
    decode_pd_result,
    decode_result,
    encode_command,
)
from pypto_serving.serving.server.serving_worker import spawn_worker
from pypto_serving.tools.profile import (
    ProfileConfig,
    create_profile_config,
    profile_instant,
    profile_span,
)

logger = logging.getLogger(__name__)


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
    profile_config: ProfileConfig = field(
        default_factory=lambda: create_profile_config(enabled=False)
    )

    # Scheduler / serving
    max_num_running_reqs: int = 32
    max_num_scheduled_tokens: int = 4096
    long_prefill_token_threshold: int = 2048
    engine_loop_interval: float = 0.001

    # Feature flags
    enable_prefix_cache: bool = True
    enable_chunk_prefill: bool = True
    # Async (pipelined) scheduling. None = auto (on). Speculative/MTP decoders
    # are supported: the scheduler optimistically reserves the upper bound of
    # tokens per step and subtracts the shortfall once the worker replies.
    async_scheduling: bool | None = None
    # Fixed 1P1D configuration. ``None`` preserves the exact legacy path.
    pd_config: object | None = None

    def resolve_async_scheduling(self) -> bool:
        """Resolve the async-scheduling flag (default on for all executors)."""
        if self.async_scheduling is not None:
            return self.async_scheduling
        return True

    def validate_pd(self) -> None:
        """Reject configurations that violate the first-version PD contract."""
        from pypto_serving.serving.pd.config import (  # noqa: PLC0415
            PD_DSPARK_SPECULATIVE_TOKENS,
            PDRole,
        )

        config = self.pd_config
        if config is None or not getattr(config, "enabled", False):
            return
        if self.executor_cls != "PyptoDeepSeekV4DSparkExecutor":
            raise ValueError("the first PD version supports the DeepSeek V4 DSpark executor only")
        if self.enable_prefix_cache:
            raise ValueError("the first PD version requires prefix caching to be disabled")
        if self.resolve_async_scheduling():
            raise ValueError("the first PD version requires async scheduling to be disabled")
        runtime = self.runtime_config or RuntimeConfig()
        expected_local_tokens = (
            0 if config.role is PDRole.PREFILL else PD_DSPARK_SPECULATIVE_TOKENS
        )
        if runtime.num_speculative_tokens != expected_local_tokens:
            raise ValueError(
                f"PD {config.role.value} requires local num_speculative_tokens="
                f"{expected_local_tokens} under the K7 product contract"
            )
        executor_tokens = self.executor_kwargs.get("num_speculative_tokens", 0)
        if executor_tokens != expected_local_tokens:
            raise ValueError(
                f"PD {config.role.value} executor requires num_speculative_tokens="
                f"{expected_local_tokens} under the K7 product contract"
            )

    def resolve_runtime_config(self) -> RuntimeConfig:
        """Return runtime settings with executor requirements resolved."""
        runtime = self.runtime_config or RuntimeConfig()
        is_deepseek_v4 = self.executor_cls == "PyptoDeepSeekV4Executor"
        homogeneous_prefill_decode = (
            runtime.requires_homogeneous_prefill_decode or is_deepseek_v4
        )
        if runtime.requires_homogeneous_prefill_decode == homogeneous_prefill_decode:
            return runtime
        return replace(
            runtime,
            requires_homogeneous_prefill_decode=homogeneous_prefill_decode,
        )

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
    # When False (non-streaming), intermediate TokenOutputs are suppressed and
    # only the final one is enqueued — one queue push / one HTTP wake-up per
    # request instead of one per token. Stop-string detection still runs every
    # step; only publishing is deferred (cf. vLLM's FINAL_ONLY output kind).
    stream: bool = True
    # Incremental-detokenization state (avoids re-decoding the full output each
    # step, which would be O(N^2) over a generation). detok_text is the running
    # cumulative text; the offsets bound the per-step decode window.
    detok_text: str = ""
    detok_prefix_offset: int = 0
    detok_read_offset: int = 0
    output_parser: object | None = None
    parser_detok_parts: list[str] = field(default_factory=list)
    parser_detok_prefix_offset: int = 0
    parser_detok_read_offset: int = 0
    parser_token_offset: int = 0
    parser_reasoning_parts: list[str] = field(default_factory=list)
    parser_content_parts: list[str] = field(default_factory=list)


@dataclass
class TokenOutput:
    token_id: int | None = None
    text: str = ""
    reasoning: str = ""
    text_delta: str = ""
    reasoning_delta: str = ""
    finished: bool = False
    finish_reason: str = ""
    # Authoritative token counts from the engine, so the HTTP layer can report
    # OpenAI-style usage without re-tokenizing the prompt or counting deltas.
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Populated on the final output only. Keeping intermediate streaming
    # outputs empty avoids copying the complete token history once per step,
    # while offline callers can still build a structured GenerateResult.
    token_ids: tuple[int, ...] = ()
    tool_call_deltas: tuple[ToolCallDelta, ...] = ()
    tool_calls: tuple[ParsedToolCall, ...] = ()


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
        tokenizer
    ) -> None:
        config.validate_pd()
        runtime = config.resolve_runtime_config()
        self.config = replace(config, runtime_config=runtime)
        self.tokenizer = tokenizer
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
        self._async_scheduling = self.config.resolve_async_scheduling()
        scheduler_config = SchedulerConfig(
            max_num_running_reqs=self.config.max_num_running_reqs,
            max_num_scheduled_tokens=self.config.max_num_scheduled_tokens,
            long_prefill_token_threshold=self.config.long_prefill_token_threshold,
            max_prefill_tokens_per_request=runtime.max_prefill_tokens_per_request,
            prefill_chunk_size_choices=runtime.prefill_chunk_size_choices,
            max_seq_len=runtime.max_seq_len,
            enable_prefix_cache=self.config.enable_prefix_cache,
            enable_chunk_prefill=self.config.enable_chunk_prefill,
            num_speculative_tokens=runtime.num_speculative_tokens,
            supports_chunked_prefill_with_speculation=(
                runtime.supports_chunked_prefill_with_speculation
            ),
            speculative_prefix_cache_replay_tokens=runtime.speculative_prefix_cache_replay_tokens,
            requires_homogeneous_prefill_decode=(
                runtime.requires_homogeneous_prefill_decode
            ),
            async_scheduling=self._async_scheduling,
            prefill_handoff=(
                getattr(getattr(self.config.pd_config, "role", None), "value", None)
                == "prefill"
            ),
        )
        # Validate before worker startup, while the manager's pools are still lazy.
        scheduler_config.validate_cache_groups(runtime.kv_cache_groups)
        self.scheduler = Scheduler(config=scheduler_config, kv_cache_manager=self.kv_cache_manager)

        self._request_contexts: dict[str, _RequestContext] = {}
        self._running = False
        self._loop_task: asyncio.Task | None = None
        self._request_counter = 0
        self._pending_free_ids: list[str] = []

        self._worker_process = None
        self._input_queue = None
        self._output_queue = None
        self._profile_output_queue = None
        self._profile_lock = asyncio.Lock()
        self._pd_command_counter = 0
        # Chunk readiness is request-scoped. A single global queue lets one
        # concurrent handoff consume another request's cache snapshot.
        self._pd_prefill_chunks: dict[str, asyncio.Queue[PrefillChunkReady]] = {}
        self._pd_chunk_counters: dict[str, int] = {}
        # Tracks which request_ids the worker has already received via
        # NewRequestData — prompt tokens are sent exactly once per request.
        self._worker_known_req_ids: set[str] = set()
        # Async (pipelined) scheduling: in-flight steps dispatched to the worker
        # but whose results have not yet been applied. Depth 2 keeps one step
        # executing on the device while the next is scheduled. Each entry is the
        # SchedulerOutput awaiting its StepResult. Sync mode leaves this empty.
        self._batch_queue: deque[tuple[int, SchedulerOutput]] = deque()
        self._max_in_flight = 2 if self._async_scheduling else 1
        self._step_counter = 0
        # step_ids of dispatched steps whose worker StepResult has NOT yet been
        # consumed off _output_queue but whose batch was discarded (error/timeout
        # dropped the pipeline). The worker is FIFO and emits exactly one result
        # per command, so these results are still in transit and must be drained
        # and ignored before the next live result is applied — otherwise a stale
        # result would be misapplied to a later batch.
        self._discard_result_step_ids: set[int] = set()
        # Worker timeouts are env-driven and process-level; resolve once at
        # construction rather than re-reading os.environ every pipelined step.
        self._init_timeout = worker_init_timeout_seconds()
        self._step_timeout = worker_step_timeout_seconds()
        self._stat_logger: InMemoryStatLogger | None = None
        self._engine_index = int(config.dp_rank)

    def set_stat_logger(
        self,
        stat_logger: InMemoryStatLogger,
        engine_index: int,
    ) -> None:
        """Attach the API-process metrics publisher for this replica."""
        self._stat_logger = stat_logger
        self._engine_index = engine_index

    def _record_scheduler_stats(self, output: SchedulerOutput | None = None) -> None:
        stat_logger = getattr(self, "_stat_logger", None)
        if stat_logger is not None:
            stat_logger.record_scheduler(
                getattr(self, "_engine_index", 0),
                SchedulerStats(
                    num_running_reqs=len(self.scheduler.running),
                    num_waiting_reqs=len(self.scheduler.waiting),
                    kv_cache_usage=self.kv_cache_manager.usage,
                    prefix_cache_queries=output.prefix_cache_queries if output else 0,
                    prefix_cache_query_tokens=output.prefix_cache_query_tokens if output else 0,
                    prefix_cache_hits=output.prefix_cache_hits if output else 0,
                    num_preemptions=len(output.preempted_requests) if output else 0,
                ),
            )

    async def start(self) -> None:
        """Start worker process and engine loop."""
        with profile_span("AsyncLLMEngine.start", cat="serving"):
            (
                process,
                input_q,
                output_q,
                profile_output_q,
                ready_event,
                num_pages_value,
            ) = spawn_worker(self.config)
            self._worker_process = process
            self._input_queue = input_q
            self._output_queue = output_q
            self._profile_output_queue = profile_output_q

            logger.info("Waiting for worker to initialize model...")
            try:
                ready = await asyncio.to_thread(ready_event.wait, timeout=self._init_timeout)
                if not ready:
                    raise RuntimeError(
                        f"Worker failed to initialize within {self._init_timeout:g}s timeout; "
                        "set PYPTO_WORKER_INIT_TIMEOUT to allow more time for large checkpoints"
                    )
            except BaseException:
                await asyncio.to_thread(self._shutdown_worker, timeout=5)
                raise
            logger.info("Worker ready")

            # Synchronise block metadata with the actual device-side KV cache
            # size. Grouped runners report the first group's rank-local block
            # count; generic runners report their single-pool page count.
            reported_num_blocks = num_pages_value.value
            self.kv_cache_manager.initialize(
                self._runtime,
                num_blocks=reported_num_blocks,
            )
            if self.kv_cache_manager.has_groups:
                logger.info(
                    "Grouped KV cache pools initialised: %s",
                    ", ".join(
                        f"{name}={self.kv_cache_manager.group_num_blocks(name)}"
                        for name in self.kv_cache_manager.group_names
                    ),
                )
            else:
                logger.info(
                    "KV cache block pool initialised: num_blocks=%d, block_size=%d",
                    reported_num_blocks,
                    self._runtime.page_size,
                )

        # The KV-cache block pool, scheduler tables and tokenizer are now
        # resident. Freeze the engine-process heap so the GC won't rescan them
        # during serving (see gc_utils). Per-process: the worker freezes its
        # own heap separately.
        freeze_gc_heap()

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

    async def start_profile(self) -> None:
        """Start SA profiling in this replica's worker."""
        await self._set_profile_active(True)

    async def stop_profile(self) -> None:
        """Flush and stop SA profiling in this replica's worker."""
        await self._set_profile_active(False)

    async def _set_profile_active(self, active: bool) -> None:
        """Send an ordered worker profile command and wait for its acknowledgement."""
        async with self._profile_lock:
            input_queue = self._input_queue
            output_queue = self._profile_output_queue
            if input_queue is None or output_queue is None:
                raise RuntimeError("Serving worker is not running")

            input_queue.put(encode_command(ProfileCommand(active=active)))
            try:
                raw_result = await asyncio.to_thread(
                    output_queue.get,
                    timeout=self._step_timeout,
                )
            except queue.Empty as exc:
                action = "start" if active else "stop"
                raise RuntimeError(
                    f"Worker profile {action} timed out ({self._step_timeout:g}s)"
                ) from exc

            result = decode_profile_result(raw_result)
            if result.error:
                raise RuntimeError(result.error)
            if result.active != active:
                raise RuntimeError(
                    f"Worker profile state mismatch: requested active={active}, "
                    f"received active={result.active}"
                )

    async def call_pd_worker(self, operation: str, payload: bytes = b"") -> bytes:
        """Run an ordered owner/cache operation in the spawned worker process."""
        if not isinstance(operation, str) or not operation or len(operation) > 128:
            raise ValueError("PD worker operation must be a nonempty bounded string")
        if (
            not isinstance(payload, bytes)
            or len(payload) > MAX_WORKER_PD_PAYLOAD_BYTES
        ):
            raise ValueError(
                "PD worker payload must be bytes no larger than "
                f"{MAX_WORKER_PD_PAYLOAD_BYTES >> 20} MiB"
            )
        if self.config.pd_config is None or not getattr(self.config.pd_config, "enabled", False):
            raise RuntimeError("PD worker control is disabled")
        async with self._profile_lock:
            input_queue = self._input_queue
            output_queue = self._profile_output_queue
            if input_queue is None or output_queue is None:
                raise RuntimeError("Serving worker is not running")
            self._pd_command_counter += 1
            command_id = self._pd_command_counter
            input_queue.put(
                encode_command(
                    PDWorkerCommand(
                        command_id=command_id,
                        operation=operation,
                        payload=payload,
                    )
                )
            )
            try:
                raw_result = await asyncio.to_thread(
                    output_queue.get,
                    timeout=self._step_timeout,
                )
            except queue.Empty as exc:
                raise RuntimeError(
                    f"Worker PD control timed out ({self._step_timeout:g}s)"
                ) from exc
            result = decode_pd_result(raw_result)
            if result.command_id != command_id:
                raise RuntimeError("Worker PD control response ordering mismatch")
            if result.error:
                raise RuntimeError(result.error)
            return result.payload

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
        cache_partition: int | None = None,
        output_parser_spec: OutputParserSpec | None = None,
    ) -> AsyncGenerator[TokenOutput, None]:
        """Add a request and yield token outputs as they are generated."""
        with profile_span(
            "ReplicaEngineCore.add_request",
            cat="serving",
            args={"request_id": request_id, "max_new_tokens": config.max_new_tokens},
        ):
            request = Request(
                request_id=request_id,
                prompt_token_ids=prompt_token_ids,
                max_new_tokens=config.max_new_tokens,
                arrival_time=time.time(),
                stop_strings=tuple(config.stop) if config.stop else (),
                eos_token_id=None if config.ignore_eos else self.tokenizer.eos_token_id,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                seed=config.seed,
                cache_partition=cache_partition,
            )

            if request_id in self._request_contexts:
                raise ValueError(f"request {request_id!r} is already active")
            ctx = _RequestContext(
                request=request,
                stream=getattr(config, "stream", True),
                output_parser=create_output_parser(output_parser_spec, self.tokenizer),
            )
            self._request_contexts[request_id] = ctx
            if self.scheduler.config.prefill_handoff:
                if request_id in self._pd_prefill_chunks:
                    self._request_contexts.pop(request_id, None)
                    raise ValueError(f"request {request_id!r} already has a PD chunk queue")
                self._pd_prefill_chunks[request_id] = asyncio.Queue()
            try:
                self.scheduler.add_request(request)
            except BaseException:
                self._request_contexts.pop(request_id, None)
                self._pd_prefill_chunks.pop(request_id, None)
                raise
            stat_logger = getattr(self, "_stat_logger", None)
            if stat_logger is not None:
                stat_logger.record_queued(getattr(self, "_engine_index", 0), request_id)
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

        finished_normally = False
        try:
            while True:
                queued = await ctx.queue.get()
                if isinstance(queued, BaseException):
                    self._request_contexts.pop(request_id, None)
                    raise queued
                output: TokenOutput = queued
                yield output
                if output.finished:
                    finished_normally = True
                    e2e = time.time() - request.arrival_time
                    n_out = len(request.output_token_ids)
                    logger.info(
                        "request %s finished: prompt=%d out=%d reason=%s e2e=%.2fs (%.1f tok/s)",
                        request_id, len(prompt_token_ids), n_out, output.finish_reason,
                        e2e, (n_out / e2e) if e2e > 0 else 0.0,
                    )
                    break
        finally:
            # Only cancellation/disconnect needs cleanup here. On normal
            # completion the request already finished in the scheduler and
            # _process_step_output already scheduled the worker free, so
            # re-scheduling would double-release: the id may have been drained
            # into a StepCommand before this finally runs, defeating a plain
            # membership check and freeing the same request on the worker twice.
            if not finished_normally and request_id in self._request_contexts:
                self._request_contexts.pop(request_id, None)
                self.scheduler.abort_request(request_id)
                stat_logger = getattr(self, "_stat_logger", None)
                if stat_logger is not None:
                    stat_logger.finish_request(
                        getattr(self, "_engine_index", 0),
                        request_id,
                        "FINISHED_ABORTED",
                    )
                # Aborted/cancelled ids must ride the next StepCommand's
                # finished_request_ids, otherwise they leak in _req_cache /
                # _worker_known_req_ids and pin device resources.
                self._schedule_worker_free(request_id)
                self._pd_prefill_chunks.pop(request_id, None)

    async def abort_request(self, request_id: str) -> None:
        ctx = self._request_contexts.pop(request_id, None)
        if ctx is None:
            # Already finished/cleaned up: nothing pinned to release, and the
            # scheduler no longer tracks it. Avoid scheduling a duplicate free.
            return
        getattr(self, "_pd_prefill_chunks", {}).pop(request_id, None)
        getattr(self, "_pd_chunk_counters", {}).pop(request_id, None)
        self.scheduler.abort_request(request_id)
        self._record_scheduler_stats()
        stat_logger = getattr(self, "_stat_logger", None)
        if stat_logger is not None:
            stat_logger.finish_request(
                getattr(self, "_engine_index", 0),
                request_id,
                "FINISHED_ABORTED",
            )
        await ctx.queue.put(
            TokenOutput(finished=True, finish_reason="FINISHED_ABORTED")
        )
        # See note in add_request's finally block: schedule worker-side cleanup.
        self._schedule_worker_free(request_id)

    async def next_prefill_chunk(self, request_id: str) -> PrefillChunkReady:
        """Wait for one confirmed P chunk that is safe to hand to the connector."""
        if not self.scheduler.config.prefill_handoff:
            raise RuntimeError("this engine is not configured as the PD Prefill role")
        try:
            queue = self._pd_prefill_chunks[request_id]
        except KeyError as exc:
            raise ValueError("request has no PD Prefill chunk queue") from exc
        chunk = await queue.get()
        if chunk.request_id != request_id:
            raise RuntimeError("PD Prefill chunk queue correlation mismatch")
        return chunk

    def complete_prefill_chunk_transfer(self, request_id: str) -> None:
        """Allow the next P chunk after a deterministic non-final completion."""
        self.scheduler.complete_prefill_chunk_transfer(request_id)

    def acknowledge_prefill_handoff(self, request_id: str) -> None:
        """Release P state only after D returned the matching READY fact."""
        self.scheduler.complete_prefill_handoff(request_id)
        self._schedule_worker_free(request_id)
        self._pd_chunk_counters.pop(request_id, None)
        self._pd_prefill_chunks.pop(request_id, None)
        context = self._request_contexts.get(request_id)
        if context is not None:
            context.queue.put_nowait(
                TokenOutput(finished=True, finish_reason="FINISHED_HANDOFF")
            )

    async def add_adopted_handoff(
        self,
        *,
        reservation_id: str,
        request_id: str,
        prompt_token_ids: Sequence[int],
        first_token: int,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int | None = None,
        seed: int | None = None,
        stop_strings: tuple[str, ...] = (),
        eos_token_id: int | None = None,
        stream: bool = True,
    ) -> AsyncGenerator[TokenOutput, None]:
        """Enter D Decode from committed cache; never enqueue a Prefill request."""
        request, first_output = self.scheduler.adopt_handoff(
            reservation_id=reservation_id,
            request_id=request_id,
            prompt_token_ids=list(prompt_token_ids),
            first_token=first_token,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            stop_strings=stop_strings,
            eos_token_id=eos_token_id,
        )
        ctx = _RequestContext(request=request, stream=stream)
        self._request_contexts[request_id] = ctx
        text = self._detokenize_incrementally(ctx)
        ctx.queue.put_nowait(
            TokenOutput(
                token_id=first_token,
                text=text,
                finished=first_output.finished,
                finish_reason=first_output.finish_reason,
                prompt_tokens=request.num_prompt_tokens,
                completion_tokens=1,
                token_ids=(first_token,) if first_output.finished else (),
            )
        )
        finished_normally = False
        try:
            while True:
                queued = await ctx.queue.get()
                if isinstance(queued, BaseException):
                    self._request_contexts.pop(request_id, None)
                    raise queued
                output: TokenOutput = queued
                yield output
                if output.finished:
                    finished_normally = True
                    break
        finally:
            if not finished_normally and request_id in self._request_contexts:
                self._request_contexts.pop(request_id, None)
                self.scheduler.abort_request(request_id)
                self._schedule_worker_free(request_id)

    def _schedule_worker_free(self, request_id: str) -> None:
        """Queue a request id for worker-side release on the next StepCommand.

        Idempotent against ids still queued; combined with the single-owner
        cleanup paths (normal completion vs. abort/cancel) this guarantees each
        request is released on the worker exactly once.
        """
        if request_id not in self._pending_free_ids:
            self._pending_free_ids.append(request_id)

    async def _engine_loop(self) -> None:
        """Pipelined schedule/execute loop.

        Depth is 1 in sync mode (dispatch a step, immediately await its result)
        and 2 in async mode (dispatch step N+1 while step N executes on the
        device, hiding host scheduling/IPC latency behind worker execution).

        Each iteration:
          1. If the in-flight queue has room and there is schedulable work,
             schedule + dispatch a new step (non-blocking). In async mode this
             also advances scheduler state optimistically so the next schedule
             sees consistent counts.
          2. Otherwise (queue full, or nothing new to schedule), block on the
             OLDEST in-flight step's result and apply it. The worker is FIFO, so
             results return in dispatch order.
        """
        logger.info("Engine loop started (async_scheduling=%s)", self._async_scheduling)
        while self._running:
            dispatched = False
            if len(self._batch_queue) < self._max_in_flight and self.scheduler.has_work():
                dispatched = self._try_dispatch_step()

            if self._batch_queue:
                # Block on the oldest in-flight step when the queue is full, or
                # when we could not dispatch anything new this iteration.
                if len(self._batch_queue) >= self._max_in_flight or not dispatched:
                    applied = await self._await_and_apply_oldest()
                    if not applied:
                        continue
            elif not dispatched:
                # Nothing in flight and nothing to dispatch: flush any pending
                # frees (e.g. a just-aborted request) and idle briefly.
                await self._flush_pending_frees()
                await asyncio.sleep(self.config.engine_loop_interval)

        logger.info("Engine loop stopped")

    def _try_dispatch_step(self) -> bool:
        """Schedule one step and dispatch it to the worker without blocking.

        Returns True if a non-empty step was dispatched (and enqueued as
        in-flight), False if the scheduler produced nothing.
        """
        with profile_span("scheduler.schedule", cat="scheduler"):
            scheduler_output = self.scheduler.schedule()
        for request_id, reason in scheduler_output.rejected_requests.items():
            logger.warning("request %s rejected during scheduling: %s", request_id, reason)
            ctx = self._request_contexts.get(request_id)
            if ctx is not None:
                ctx.queue.put_nowait(ValueError(reason))
            if request_id in self._worker_known_req_ids:
                self._schedule_worker_free(request_id)
        self._record_scheduler_stats(scheduler_output)
        # Preempted requests must release their worker-side cache / device slots;
        # queue their ids so the next StepCommand frees them.
        for request in scheduler_output.preempted_requests:
            self._schedule_worker_free(request.request_id)
        if scheduler_output.is_empty:
            return False

        finished_ids = self._pending_free_ids.copy()
        self._pending_free_ids.clear()
        self._step_counter += 1
        with profile_span(
            "scheduler.queue_worker_step",
            cat="scheduler",
            args={"scheduled": len(scheduler_output.scheduled_requests)},
        ):
            step_cmd = self._build_step_command(
                scheduler_output, finished_ids, step_id=self._step_counter
            )
            self._input_queue.put(encode_command(step_cmd))

        stat_logger = getattr(self, "_stat_logger", None)
        if stat_logger is not None:
            scheduled_at = time.monotonic()
            for scheduled in scheduler_output.scheduled_requests:
                stat_logger.record_scheduled(
                    getattr(self, "_engine_index", 0), scheduled.request.request_id, timestamp=scheduled_at,
                )

        # Advance scheduler state optimistically so the NEXT schedule() (which may
        # run before this step's tokens return) sees consistent counts. No-op in
        # sync mode.
        self.scheduler.advance_after_schedule(scheduler_output)

        self._batch_queue.append((self._step_counter, scheduler_output))
        return True

    async def _await_and_apply_oldest(self) -> bool:
        """Block on the oldest in-flight step's result and apply it.

        Returns True on success, False if the step errored/timed out (the batch
        and any dependent in-flight batches are dropped and the failure handled).
        The worker is FIFO and emits exactly one result per command, so the
        oldest dispatched step is the next result off the output queue.
        """
        step_id, scheduler_output = self._batch_queue.popleft()
        try:
            with profile_span("scheduler.wait_worker_output", cat="scheduler"):
                raw_output = await self._get_live_result()
        except queue.Empty:
            logger.error(f"Worker response timed out ({self._step_timeout:g}s)")
            # Timeout: the failed step's own result is still in transit.
            self._handle_step_error(step_id, scheduler_output, result_pending=True)
            return False

        step_result = decode_result(raw_output)
        if step_result.step_id != step_id:
            # Should never happen: the worker is FIFO and stale results are
            # drained in _get_live_result. Treat as a fatal desync rather than
            # silently misapplying tokens. This result is already consumed.
            logger.error(
                "Pipeline desync: expected step_id=%d, got %d; aborting batch",
                step_id, step_result.step_id,
            )
            self._handle_step_error(step_id, scheduler_output, result_pending=False)
            return False
        if step_result.error:
            logger.error(f"Worker returned error: {step_result.error}")
            # This step's result was just consumed; only in-flight steps pend.
            self._handle_step_error(step_id, scheduler_output, result_pending=False)
            return False

        # Unwrap list[int] values back to int | list[int] for update_from_output.
        new_tokens: dict[str, int | list[int]] = {
            req_id: (tokens[0] if len(tokens) == 1 else tokens)
            for req_id, tokens in step_result.new_tokens.items()
        }
        with profile_span(
            "scheduler.process_step_output",
            cat="scheduler",
            args={"new_tokens": len(new_tokens)},
        ):
            self._process_step_output(scheduler_output, new_tokens, step_result.num_draft_tokens)
        return True

    async def _get_live_result(self) -> bytes:
        """Return the next non-discarded StepResult, draining stale ones.

        Results for batches discarded by a prior error/timeout are still in
        transit (the worker had already been sent those commands). Drain and
        drop them so a stale result is never applied to a live batch.
        """
        while True:
            raw = await asyncio.to_thread(self._output_queue.get, timeout=self._step_timeout)
            if not self._discard_result_step_ids:
                return raw
            sid = decode_result(raw).step_id
            if sid in self._discard_result_step_ids:
                self._discard_result_step_ids.discard(sid)
                logger.info("Drained stale result for discarded step_id=%d", sid)
                continue
            return raw

    def _build_step_command(
        self,
        scheduler_output: SchedulerOutput,
        finished_ids: list[str],
        step_id: int = 0,
    ) -> StepCommand:
        """Build a lightweight StepCommand from the scheduler output.

        Prompt tokens for requests that the worker has not yet seen are shipped
        as ``NewRequestData`` entries exactly once; subsequent steps carry only
        per-request deltas (~1 KB total at batch 16).

        In async mode a decode input token for step N+1 may not be sampled yet
        (its step N result has not returned); such tokens are sent as
        ``PLACEHOLDER_TOKEN`` and the worker substitutes from its own cache.
        """
        new_requests: list[NewRequestData] = []
        prefill_requests: list[PrefillRequest] = []
        decode_requests: list[DecodeRequest] = []

        # A preempted request may be restarted in this same scheduler pass.
        # Forget released IDs before building deltas so such a request carries
        # a fresh NewRequestData record for the worker cache.
        for req_id in finished_ids:
            self._worker_known_req_ids.discard(req_id)

        for sr in scheduler_output.scheduled_requests:
            req = sr.request
            req_id = req.request_id

            # Register with worker the first time this request is scheduled.
            if req_id not in self._worker_known_req_ids:
                new_requests.append(NewRequestData(
                    request_id=req_id,
                    prompt_token_ids=list(req.prompt_token_ids),
                    temperature=req.temperature,
                    top_p=req.top_p,
                    top_k=req.top_k,
                    seed=req.seed,
                    pd_adopted=bool(req.pd_reservation_id),
                ))
                self._worker_known_req_ids.add(req_id)

            if sr.is_prefill:
                num_computed = sr.num_computed_tokens
                num_new = sr.num_new_tokens
                chunk_tokens = req.prompt_token_ids[num_computed: num_computed + num_new]
                prefill_requests.append(PrefillRequest(
                    request_id=req_id,
                    chunk_tokens=list(chunk_tokens),
                    num_computed_tokens=num_computed,
                    block_ids=list(sr.block_ids),
                    block_ids_by_group={
                        name: list(block_ids)
                        for name, block_ids in sr.block_ids_by_group.items()
                    },
                    cache_partition=sr.cache_partition,
                ))
            else:
                output_ids = req.output_token_ids
                prompt_ids = req.prompt_token_ids
                # In async mode a request scheduled while a prior token is still
                # in flight (num_output_placeholders > 0) has a stale
                # output_token_ids tail. Send a placeholder so the worker uses the
                # token it last committed (FIFO guarantees it is cached by the
                # time the worker runs this step).
                if req.num_output_placeholders > 0:
                    last_token = PLACEHOLDER_TOKEN
                else:
                    last_token = output_ids[-1] if output_ids else prompt_ids[-1]
                decode_requests.append(DecodeRequest(
                    request_id=req_id,
                    last_token=last_token,
                    # Context length for THIS step, snapshotted at schedule time:
                    # positions already computed plus the token(s) this step adds.
                    # Do not use req.num_tokens — under async scheduling it also
                    # counts in-flight placeholders from *other* steps, which would
                    # inflate seq_len past the KV actually written and shift the
                    # kernel's positions (observed as duplicated/misplaced tokens
                    # with chunked prefill at pipeline depth 2).
                    seq_len=sr.num_computed_tokens + sr.num_new_tokens,
                    block_ids=list(sr.block_ids),
                    block_ids_by_group={
                        name: list(block_ids)
                        for name, block_ids in sr.block_ids_by_group.items()
                    },
                    cache_partition=sr.cache_partition,
                ))

        return StepCommand(
            new_requests=new_requests,
            prefill_requests=prefill_requests,
            decode_requests=decode_requests,
            finished_request_ids=finished_ids,
            step_id=step_id,
        )

    async def _flush_pending_frees(self) -> None:
        """Send a cleanup-only StepCommand when frees are pending but no work is
        schedulable, so an aborted request's worker cache / device slot is not
        pinned until unrelated future work happens to carry the free along.

        The worker tolerates empty prefill/decode batches and replies with an
        empty StepResult, which we drain to keep the request/response queues in
        lock-step with the normal loop.
        """
        if not self._pending_free_ids:
            return
        # Only safe to run its own request/response round-trip when nothing else
        # is in flight; the caller (engine loop) only invokes this with an empty
        # batch queue. Assert to catch a future ordering regression.
        assert not self._batch_queue, "flush must not run with in-flight steps"

        finished_ids = self._pending_free_ids.copy()
        self._pending_free_ids.clear()
        for req_id in finished_ids:
            self._worker_known_req_ids.discard(req_id)

        self._step_counter += 1
        cleanup_cmd = StepCommand(
            new_requests=[],
            prefill_requests=[],
            decode_requests=[],
            finished_request_ids=finished_ids,
            step_id=self._step_counter,
        )
        self._input_queue.put(encode_command(cleanup_cmd))
        try:
            # Drains any stale results from previously-discarded steps first, so
            # the cleanup reply is not confused with a leftover in-transit result.
            raw_output = await self._get_live_result()
        except queue.Empty:
            logger.error(f"Worker cleanup-step timed out ({self._step_timeout:g}s)")
            return
        step_result = decode_result(raw_output)
        if step_result.error:
            logger.error(f"Worker cleanup-step returned error: {step_result.error}")

    def _handle_step_error(
        self,
        failed_step_id: int,
        scheduler_output: SchedulerOutput,
        *,
        result_pending: bool = True,
    ) -> None:
        """On worker error/timeout, abort all requests in the failed batch.

        In async mode any still-in-flight steps were built on the failed step's
        optimistic state and can no longer be reconciled, so their batches are
        discarded too and their requests aborted.

        The worker is FIFO and emits exactly one result per dispatched command,
        so every discarded step still has a StepResult in transit. Their step_ids
        are recorded in ``_discard_result_step_ids`` and drained (ignored) by
        ``_get_live_result`` before any live result is applied, preventing a
        stale result from being misapplied to a later batch. ``result_pending``
        is False when the caller already consumed the failed step's own result
        (worker returned an error / step_id desync); True on timeout, where the
        failed step's result is still coming.
        """
        failed_batches = [scheduler_output]
        if result_pending:
            self._discard_result_step_ids.add(failed_step_id)
        # Any remaining in-flight batches depend on the failed step; drop them.
        # Their results are still in transit and must be drained.
        while self._batch_queue:
            sid, batch = self._batch_queue.popleft()
            self._discard_result_step_ids.add(sid)
            failed_batches.append(batch)

        seen: set[str] = set()
        for batch in failed_batches:
            for sr in batch.scheduled_requests:
                self.scheduler.discard_scheduled_request(sr)
                request_id = sr.request.request_id
                if request_id in seen:
                    continue
                seen.add(request_id)
                ctx = self._request_contexts.pop(request_id, None)
                if ctx is not None:
                    ctx.queue.put_nowait(
                        TokenOutput(finished=True, finish_reason="error")
                    )
                stat_logger = getattr(self, "_stat_logger", None)
                if stat_logger is not None:
                    stat_logger.finish_request(
                        getattr(self, "_engine_index", 0),
                        request_id,
                        "error",
                    )
                self._schedule_worker_free(request_id)
                self.scheduler.abort_request(request_id)
        self._record_scheduler_stats()

    def _process_step_output(
        self,
        scheduler_output: SchedulerOutput,
        new_tokens: dict[str, int | list[int]],
        num_draft_tokens: dict[str, int] | None = None,
    ) -> None:
        """Process worker results: update scheduler state, push tokens to request queues."""
        stat_logger = getattr(self, "_stat_logger", None)
        engine_index = getattr(self, "_engine_index", 0)
        if stat_logger is not None:
            iteration = IterationStats(num_prefill_tokens=scheduler_output.num_prefill_tokens)
            live_request_ids = {
                scheduled.request.request_id for scheduled in scheduler_output.scheduled_requests
                if not scheduled.request.status.is_finished
                and scheduled.request.status is not RequestStatus.PREEMPTED
            }
            # Count verified drafts before EOS/length/stop truncation in the scheduler.
            for request_id, proposed in (num_draft_tokens or {}).items():
                if request_id not in live_request_ids:
                    continue
                if proposed > 0:
                    tokens = new_tokens[request_id]
                    iteration.num_speculative_drafts += 1
                    iteration.num_draft_tokens += proposed
                    accepted = len(tokens) - 1 if isinstance(tokens, list) else 0
                    iteration.num_accepted_tokens += accepted
                    positions = iteration.num_accepted_tokens_per_pos
                    if len(positions) < proposed:
                        positions.extend([0] * (proposed - len(positions)))
                    for pos in range(accepted):
                        positions[pos] += 1
                else:
                    iteration.num_speculative_fallbacks += 1
        request_outputs = self.scheduler.update_from_output(scheduler_output, new_tokens)
        if stat_logger is not None:
            iteration.num_iteration_tokens = scheduler_output.num_prefill_tokens + sum(
                output.new_token_id is not None for output in request_outputs
            )
            stat_logger.record_iteration(engine_index, iteration)
        self._record_scheduler_stats()

        if getattr(getattr(self.scheduler, "config", None), "prefill_handoff", False):
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
                chunk_id = self._pd_chunk_counters.get(request.request_id, 0)
                self._pd_chunk_counters[request.request_id] = chunk_id + 1
                self.scheduler.mark_prefill_chunk_transfer_pending(request.request_id)
                try:
                    chunk_queue = self._pd_prefill_chunks[request.request_id]
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
                            name: tuple(block_ids)
                            for name, block_ids in scheduled.block_ids_by_group.items()
                        },
                        cache_partition=scheduled.cache_partition,
                    )
                )

        for req_output in request_outputs:
            if req_output.handoff_ready:
                # The first token is part of the final handoff metadata. It is
                # published only by the D output path after READY commit.
                continue
            ctx = self._request_contexts.get(req_output.request_id)
            if ctx is None:
                continue

            previous_text = ctx.detok_text
            text = self._detokenize_incrementally(ctx)
            text_delta = (
                text[len(previous_text) :]
                if text.startswith(previous_text)
                else ""
            )

            if not req_output.finished and ctx.request.stop_strings:
                for stop in ctx.request.stop_strings:
                    if stop and text.endswith(stop):
                        req_output.finished = True
                        req_output.finish_reason = "FINISHED_STOP"
                        self.scheduler.finish_request(
                            req_output.request_id, RequestStatus.FINISHED_STOP
                        )
                        self._record_scheduler_stats()
                        break

            if stat_logger is not None and req_output.new_token_id is not None:
                stat_logger.record_output(
                    engine_index,
                    req_output.request_id,
                    completion_tokens=len(ctx.request.output_token_ids),
                )

            if req_output.finished:
                # Flush the authoritative full decode: if generation ends while a
                # multi-token character is incomplete (or a token legitimately
                # decodes to U+FFFD), the incremental path withholds that tail
                # forever. A one-shot full decode at finish guarantees the final
                # text matches the offline baseline instead of being truncated.
                text = self._finalize_detokenization(ctx)
                self._schedule_worker_free(req_output.request_id)

            # Stop detection above operates on the original generated text.
            # Semantic parsing only changes the public presentation channels.
            reasoning = ""
            reasoning_delta = ""
            tool_call_deltas: tuple[ToolCallDelta, ...] = ()
            tool_calls: tuple[ParsedToolCall, ...] = ()
            if ctx.output_parser is not None:
                try:
                    truncated = req_output.finish_reason in ("FINISHED_LENGTH", "FINISHED_ABORTED")
                    if ctx.stream:
                        parser_text_delta = self._detokenize_parser_incrementally(ctx)
                        token_ids = ctx.request.output_token_ids
                        delta_token_ids = token_ids[ctx.parser_token_offset :]
                        ctx.parser_token_offset = len(token_ids)
                        parsed = ctx.output_parser.feed(
                            parser_text_delta,
                            delta_token_ids,
                        )
                        reasoning_delta = parsed.reasoning
                        text_delta = parsed.content
                        tool_call_deltas = parsed.tool_call_deltas

                        if req_output.finished:
                            final_parser_text = self._finalize_parser_detokenization(ctx)
                            if final_parser_text:
                                tail = ctx.output_parser.feed(final_parser_text, ())
                                reasoning_delta += tail.reasoning
                                text_delta += tail.content
                                tool_call_deltas += tail.tool_call_deltas
                            tail = ctx.output_parser.finish(truncated=truncated)
                            reasoning_delta += tail.reasoning
                            text_delta += tail.content
                            tool_call_deltas += tail.tool_call_deltas
                            tool_calls = tail.tool_calls

                        if reasoning_delta:
                            ctx.parser_reasoning_parts.append(reasoning_delta)
                        if text_delta:
                            ctx.parser_content_parts.append(text_delta)
                        if req_output.finished:
                            reasoning = "".join(ctx.parser_reasoning_parts)
                            text = "".join(ctx.parser_content_parts)
                        else:
                            reasoning = ""
                            text = ""
                    elif req_output.finished:
                        raw_text = self.tokenizer.decode(
                            ctx.request.output_token_ids,
                            skip_special_tokens=False,
                        )
                        parsed = ctx.output_parser.parse_complete(
                            raw_text,
                            ctx.request.output_token_ids,
                            truncated=truncated,
                        )
                        text = parsed.content
                        reasoning = parsed.reasoning
                        text_delta = text
                        reasoning_delta = reasoning
                        tool_calls = parsed.tool_calls
                except ValueError as exc:
                    # Parser failures are request-local presentation errors.
                    # Remove ownership before waking the consumer so its
                    # add_request() finally block cannot schedule a duplicate
                    # worker free while this loop explicitly releases state.
                    self._request_contexts.pop(req_output.request_id, None)
                    ctx.queue.put_nowait(exc)
                    self.scheduler.abort_request(req_output.request_id)
                    self._schedule_worker_free(req_output.request_id)
                    if stat_logger is not None:
                        stat_logger.finish_request(engine_index, req_output.request_id, "error")
                    self._record_scheduler_stats()
                    logger.exception(
                        "request %s output parser failed",
                        req_output.request_id,
                    )
                    continue

            if req_output.finished and stat_logger is not None:
                stat_logger.finish_request(
                    engine_index, req_output.request_id, req_output.finish_reason,
                )

            # Non-streaming requests only need the final output: suppress
            # intermediate ones to save a queue push and HTTP-coroutine wake-up
            # per token. Detok + stop detection above still ran this step, so the
            # final text is complete.
            if not ctx.stream and not req_output.finished:
                continue

            token_output = TokenOutput(
                token_id=req_output.new_token_id,
                text=text,
                reasoning=reasoning,
                text_delta=text_delta,
                reasoning_delta=reasoning_delta,
                tool_call_deltas=tool_call_deltas,
                tool_calls=tool_calls,
                finished=req_output.finished,
                finish_reason=req_output.finish_reason,
                prompt_tokens=ctx.request.num_prompt_tokens,
                completion_tokens=len(ctx.request.output_token_ids),
                token_ids=(
                    tuple(ctx.request.output_token_ids)
                    if req_output.finished
                    else ()
                ),
            )
            if req_output.finished:
                # Final output owns the already-scheduled worker release. Drop
                # the context before waking a consumer that may close at once.
                self._request_contexts.pop(req_output.request_id, None)
            ctx.queue.put_nowait(token_output)

    def _detokenize_incrementally(self, ctx: _RequestContext) -> str:
        """Decode only the newly-completed text and append it to the running text.

        O(1) amortized per step (bounded decode window) instead of re-decoding
        the full output_token_ids every step (O(N^2) over a generation).
        Returns the cumulative decoded text so far.
        """
        (
            ctx.detok_text,
            ctx.detok_prefix_offset,
            ctx.detok_read_offset,
            _,
        ) = self._advance_detokenization(
            ctx.request.output_token_ids,
            text=ctx.detok_text,
            prefix_offset=ctx.detok_prefix_offset,
            read_offset=ctx.detok_read_offset,
            skip_special_tokens=True,
        )
        return ctx.detok_text

    def _detokenize_parser_incrementally(self, ctx: _RequestContext) -> str:
        """Return newly released raw text for one stateful output parser."""
        (
            _,
            ctx.parser_detok_prefix_offset,
            ctx.parser_detok_read_offset,
            delta,
        ) = self._advance_detokenization(
            ctx.request.output_token_ids,
            text="",
            prefix_offset=ctx.parser_detok_prefix_offset,
            read_offset=ctx.parser_detok_read_offset,
            skip_special_tokens=False,
        )
        if delta:
            ctx.parser_detok_parts.append(delta)
        return delta

    def _advance_detokenization(
        self,
        output_ids: Sequence[int],
        *,
        text: str,
        prefix_offset: int,
        read_offset: int,
        skip_special_tokens: bool,
    ) -> tuple[str, int, int, str]:
        """Advance one bounded-context detokenizer and return its text delta."""
        if not output_ids:
            return text, prefix_offset, read_offset, ""

        prefix_ids = output_ids[prefix_offset:read_offset]
        new_ids = output_ids[prefix_offset:]

        def decode(token_ids: Sequence[int]) -> str:
            if skip_special_tokens:
                return self.tokenizer.decode(token_ids)
            return self.tokenizer.decode(
                token_ids,
                skip_special_tokens=False,
            )

        prefix_text = decode(prefix_ids) if prefix_ids else ""
        new_text = decode(new_ids)
        if len(new_text) <= len(prefix_text) or new_text.endswith("�"):
            return text, prefix_offset, read_offset, ""

        delta = new_text[len(prefix_text) :]
        text += delta
        read_offset = len(output_ids)
        prefix_offset = max(0, read_offset - 3)
        return text, prefix_offset, read_offset, delta

    def _finalize_detokenization(self, ctx: _RequestContext) -> str:
        """Return the authoritative final text for a finished request.

        The incremental path withholds a trailing U+FFFD (an incomplete
        multi-token character) waiting for a token that never arrives once
        generation stops. A single full decode of the whole output at finish
        matches the offline baseline; O(N) once per request is negligible.
        """
        output_ids = ctx.request.output_token_ids
        if not output_ids:
            return ctx.detok_text
        final_text = self.tokenizer.decode(output_ids)
        ctx.detok_text = final_text
        ctx.detok_read_offset = len(output_ids)
        ctx.detok_prefix_offset = max(0, ctx.detok_read_offset - 3)
        return final_text

    def _finalize_parser_detokenization(self, ctx: _RequestContext) -> str:
        """Flush raw parser text without changing an already streamed prefix."""
        output_ids = ctx.request.output_token_ids
        if not output_ids:
            return ""
        final_text = self.tokenizer.decode(
            output_ids,
            skip_special_tokens=False,
        )
        streamed_text = "".join(ctx.parser_detok_parts)
        if not final_text.startswith(streamed_text):
            raise ValueError(
                "final parser detokenization changed an already emitted prefix"
            )
        delta = final_text[len(streamed_text) :]
        if delta:
            ctx.parser_detok_parts.append(delta)
        ctx.parser_detok_read_offset = len(output_ids)
        ctx.parser_detok_prefix_offset = max(
            0,
            ctx.parser_detok_read_offset - 3,
        )
        return delta

    def _shutdown_worker(self, *, timeout: float) -> None:
        input_q = self._input_queue
        process = self._worker_process

        if input_q is not None:
            with contextlib.suppress(Exception):
                # New protocol: send encoded ShutdownCommand bytes.
                input_q.put(encode_command(ShutdownCommand()))

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
        self._profile_output_queue = None


class AsyncLLMEngine:
    """Async serving engine that routes requests across replica cores.

    The engine owns one or more ``ReplicaEngineCore`` instances and exposes the
    server-facing async API: ``start``, ``stop``, ``add_request``,
    ``abort_request``, and ``generate_request_id``. With one serving replica it
    wraps a single core. With multiple replicas it selects a core for each
    request and records request placement so aborts reach the correct replica.
    """

    def __init__(
        self,
        config: EngineConfig,
        tokenizer,
        *,
        core_factory: Callable[..., ReplicaEngineCore] = ReplicaEngineCore,
    ) -> None:
        config = replace(config, runtime_config=config.resolve_runtime_config())
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
        assert self.tokenizer is not None
        self.eos_token_id = tokenizer.eos_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.parallel_config = parallel
        self._request_counter = 0
        self._route_counter = 0
        self._request_to_replica: dict[str, int] = {}
        self._route_extra_load = [0 for _ in parallel.replica_device_groups]
        self._cores: list[ReplicaEngineCore] = []
        self._core_records_metrics: list[bool] = []
        self.metrics = InMemoryStatLogger(
            config.model_id,
            list(range(len(parallel.replica_device_groups))),
            max_model_len=config.runtime_config.max_seq_len,
            num_speculative_tokens=config.runtime_config.num_speculative_tokens,
        )
        self._pd_service = None

        for dp_rank, device_group in enumerate(parallel.replica_device_groups):
            replica_parallel = parallel.for_replica(device_group)
            replica_config = replace(
                config,
                device_id=device_group[0],
                parallel_config=replica_parallel,
                dp_rank=dp_rank,
            )
            core = core_factory(
                config=replica_config,
                tokenizer=tokenizer,
            )
            records_metrics = callable(getattr(core, "set_stat_logger", None))
            self._core_records_metrics.append(records_metrics)
            if records_metrics:
                core.set_stat_logger(self.metrics, dp_rank)
            self._cores.append(core)

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
        pd_config = self.config.pd_config
        if pd_config is not None and getattr(pd_config, "enabled", False):
            from pypto_serving.serving.pd.service import PDServingService

            service = PDServingService(self, pd_config)
            try:
                await service.start()
            except BaseException:
                await service.close()
                await self.stop()
                raise
            self._pd_service = service

    async def stop(self) -> None:
        """Stop all DP engine cores."""
        if self._pd_service is not None:
            await self._pd_service.close()
            self._pd_service = None
        await asyncio.gather(*(core.stop() for core in reversed(self._cores)))

    async def start_profile(self) -> None:
        """Start SA profiling in every replica worker."""
        results = await asyncio.gather(
            *(core.start_profile() for core in self._cores),
            return_exceptions=True,
        )
        errors = [result for result in results if isinstance(result, BaseException)]
        if errors:
            await asyncio.gather(
                *(core.stop_profile() for core in self._cores),
                return_exceptions=True,
            )
            raise RuntimeError(f"Failed to start profiling in a worker: {errors[0]}")

    async def stop_profile(self) -> None:
        """Flush and stop SA profiling in every replica worker."""
        results = await asyncio.gather(
            *(core.stop_profile() for core in self._cores),
            return_exceptions=True,
        )
        errors = [result for result in results if isinstance(result, BaseException)]
        if errors:
            raise RuntimeError(f"Failed to stop profiling in a worker: {errors[0]}")

    def generate_request_id(self) -> str:
        self._request_counter += 1
        return f"serving-req-{self._request_counter}"

    def pending_token_load(self) -> int:
        return sum(core.pending_token_load() for core in self._cores)

    @property
    def pd_health_error(self) -> str:
        if self._pd_service is None:
            return ""
        return self._pd_service.health_error

    @property
    def pd_service(self):
        """Expose the role-local control plane to the protected internal API."""
        if self._pd_service is None:
            raise RuntimeError("PD serving control plane has not started")
        return self._pd_service

    async def generate_result(
        self,
        prompt: str,
        config: GenerateConfig | None = None,
    ) -> GenerateResult:
        """Generate one non-streaming result through the serving scheduler.

        This is the offline counterpart to :meth:`add_request`. It deliberately
        uses the same worker, scheduler, grouped-cache, and speculative-decode
        paths as online serving, which is required by distributed model
        integrations such as DeepSeek V4.
        """
        generate_config = config or GenerateConfig(stream=False)
        if generate_config.stream:
            raise ValueError("generate_result requires stream=False")

        request_id = self.generate_request_id()
        final_output: TokenOutput | None = None
        async for output in self.add_request(request_id, prompt, generate_config):
            if output.finished:
                final_output = output

        if final_output is None:
            raise RuntimeError(f"Generation for request {request_id!r} ended without a final output")
        if final_output.finish_reason == "error":
            raise RuntimeError(f"Generation failed for request {request_id!r}")

        return GenerateResult(
            text=final_output.text,
            token_ids=list(final_output.token_ids),
            finish_reason=self.normalize_finish_reason(final_output.finish_reason),
        )

    async def generate_batch(
        self,
        prompts: Sequence[str],
        config: GenerateConfig | None = None,
    ) -> list[GenerateResult]:
        """Generate non-streaming results for prompts with continuous batching."""
        generate_config = config or GenerateConfig(stream=False)
        if generate_config.stream:
            raise ValueError("generate_batch requires stream=False")
        tasks = [
            asyncio.create_task(self.generate_result(prompt, generate_config))
            for prompt in prompts
        ]
        try:
            return list(await asyncio.gather(*tasks))
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

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
        *,
        prompt_token_ids: Sequence[int] | None = None,
        output_parser_spec: OutputParserSpec | None = None,
    ) -> AsyncGenerator[TokenOutput, None]:
        arrival_monotonic = time.monotonic()
        pd_config = self.config.pd_config
        if pd_config is not None and getattr(pd_config, "enabled", False):
            raise ValueError("send generation requests through the external PD Router")

        replica_idx = self._select_replica()
        prompt_token_ids = self._resolve_prompt_tokens(prompt, prompt_token_ids)
        self.metrics.start_request(
            replica_idx,
            request_id,
            arrival_monotonic=arrival_monotonic,
            num_prompt_tokens=len(prompt_token_ids),
            max_tokens=config.max_new_tokens,
        )
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

        terminal_output_seen = False
        try:
            core = self._cores[replica_idx]
            if output_parser_spec is None:
                outputs = core.add_request(
                    request_id,
                    prompt,
                    config,
                    on_queued=clear_route_extra_load,
                    prompt_token_ids=prompt_token_ids,
                )
            else:
                outputs = core.add_request(
                    request_id,
                    prompt,
                    config,
                    on_queued=clear_route_extra_load,
                    prompt_token_ids=prompt_token_ids,
                    output_parser_spec=output_parser_spec,
                )
            async with contextlib.aclosing(outputs):
                async for output in outputs:
                    if not self._core_records_metrics[replica_idx]:
                        self.metrics.record_output(
                            replica_idx, request_id, completion_tokens=output.completion_tokens,
                        )
                    if output.finished:
                        terminal_output_seen = True
                        if not self._core_records_metrics[replica_idx]:
                            self.metrics.finish_request(replica_idx, request_id, output.finish_reason)
                    yield output
        except (asyncio.CancelledError, GeneratorExit):
            if not terminal_output_seen:
                self.metrics.finish_request(
                    replica_idx,
                    request_id,
                    "FINISHED_ABORTED",
                )
            raise
        except BaseException:
            if not terminal_output_seen:
                self.metrics.finish_request(replica_idx, request_id, "error")
            raise
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
        prompt_token_ids = self.tokenizer.encode(prompt)
        if not prompt_token_ids and self.tokenizer.bos_token_id is not None:
            prompt_token_ids = [self.tokenizer.bos_token_id]
        if not prompt_token_ids:
            raise ValueError("Prompt tokenization produced no tokens.")
        return prompt_token_ids

    def _resolve_prompt_tokens(
        self,
        prompt: str,
        prompt_token_ids: Sequence[int] | None,
    ) -> tuple[int, ...]:
        if prompt_token_ids is None:
            return tuple(int(token) for token in self._tokenize_prompt(prompt))
        tokens = tuple(prompt_token_ids)
        if not tokens or any(type(token) is not int or token < 0 for token in tokens):
            raise ValueError("Prompt token IDs must be nonempty non-negative integers.")
        vocab_size = getattr(self.tokenizer, "vocab_size", None)
        if isinstance(vocab_size, int) and any(token >= vocab_size for token in tokens):
            raise ValueError("Prompt token ID is outside the tokenizer vocabulary.")
        return tokens

    def _estimate_request_load(self, prompt_token_ids: Sequence[int] | None, config) -> int:
        prompt_tokens = len(prompt_token_ids) if prompt_token_ids is not None else 0
        return prompt_tokens + int(getattr(config, "max_new_tokens", 0))

    @staticmethod
    def normalize_finish_reason(reason: str) -> str:
        """Map scheduler status names to the public GenerateResult values."""
        return {
            "FINISHED_EOS": "eos",
            "FINISHED_LENGTH": "length",
            "FINISHED_STOP": "stop",
            "FINISHED_ABORTED": "aborted",
        }.get(reason, reason.lower())
