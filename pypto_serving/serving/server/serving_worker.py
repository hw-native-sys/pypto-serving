# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import sys
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from pypto_serving.config.types import (
    DecodeBatch,
    DecodeResult,
    SamplingParams,
)
from pypto_serving.serving.utils.gc_utils import freeze_gc_heap
from pypto_serving.serving.server.ipc import (
    PLACEHOLDER_TOKEN,
    DecodeRequest,
    NewRequestData,
    PrefillRequest,
    ProfileCommand,
    ProfileResult,
    ShutdownCommand,
    StepCommand,
    StepResult,
    decode_command,
    encode_profile_result,
    encode_result,
)
from pypto_serving.serving.utils.prefill import pack_prefill_batch
from pypto_serving.tools.profile import configure_profiler, get_profiler, profile_span

if TYPE_CHECKING:
    from pypto_serving.serving.engine.async_engine import EngineConfig

logger = logging.getLogger(__name__)

_DECODE_PIPELINE_SLOTS = 2
_MISSING_REQUEST = object()


@dataclass(frozen=True)
class _PreparedDecodeWork:
    """Backend snapshot prepared without resolving prior-step output tokens."""

    prepared: object | None
    buffer_slot: int
    batch: DecodeBatch | None = None
    error: str | None = None


@dataclass(frozen=True)
class _PendingDecodeOutput:
    """Submitted decode ticket completed and consumed by the output lane."""

    cmd: StepCommand
    scheduled: tuple[DecodeRequest, ...]
    pending: object
    buffer_slot: int


@dataclass(frozen=True)
class _CompletedStepOutput:
    """Already materialized fallback-path result."""

    result: StepResult
    buffer_slot: int | None = None


@dataclass(frozen=True)
class _DecodeCommandFailure:
    """Malformed input forwarded through the ordered device/output lanes."""

    result: StepResult


@dataclass(frozen=True)
class _ProfileBarrier:
    """FIFO barrier that keeps profiler control ordered with output reclaim."""

    command: ProfileCommand
    completed: threading.Event


class WorkerProcess:
    """Dedicated process that owns a single NPU device and executes model inference.

    Architecture (single-card, extensible to multi-card by spawning multiple workers):
      Main Process  --[input_queue]--> WorkerProcess --[output_queue]--> Main Process
    """

    def __init__(
        self,
        config: EngineConfig,
        input_queue: mp.Queue,
        output_queue: mp.Queue,
        profile_output_queue: mp.Queue | None = None,
    ):
        self.config = config
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.profile_output_queue = profile_output_queue

        self.executor = None
        self.sampler = None
        self.model_record = None
        self._page_size: int = 64
        # Request cache: prompt tokens + sampling params registered once per request.
        # Populated by StepCommand.new_requests; entries removed when the request finishes.
        self._req_cache: dict[str, NewRequestData] = {}
        # Latest sampled token per request. Under async scheduling the engine sends
        # PLACEHOLDER_TOKEN for a decode input it hasn't sampled yet; the worker
        # substitutes from here. Entries cleared when a request is released.
        self._last_tokens: dict[str, list[int]] = {}

    def init_device_and_model(self) -> int:
        from pypto_serving.config.types import ModelRecord
        from pypto_serving.model.common.executor.sampler import Sampler
        from pypto_serving.model.model_loader import ModelLoader

        device_ids = self.config.worker_device_ids()
        device_label = ",".join(str(device_id) for device_id in device_ids)
        pypto_build_dir = self._configure_pypto_build_dir(device_ids)
        configure_profiler(
            self.config.profile_config,
            process_name=f"serving-worker-{device_label}",
            initially_active=False,
        )
        with profile_span(
            "WorkerProcess.init_device_and_model",
            cat="worker",
            args={
                "model_id": self.config.model_id,
                "device_id": self.config.device_id,
                "device_ids": list(device_ids),
                "dp_rank": self.config.dp_rank,
                "pypto_build_dir": str(pypto_build_dir),
            },
        ):
            logger.info(
                f"Worker initializing: platform={self.config.platform}, "
                f"devices={list(device_ids)}, dp_rank={self.config.dp_rank}, "
                f"pypto_build_dir={pypto_build_dir}"
            )

            self.sampler = Sampler()

            executor_cls = self._resolve_executor_cls()
            self.executor = executor_cls(
                platform=self.config.platform,
                device_ids=device_ids,
                pypto_build_dir=str(pypto_build_dir),
                **self.config.executor_kwargs,
            )

            loaded = ModelLoader().load(
                model_id=self.config.model_id,
                model_dir=self.config.model_dir,
                runtime_config=self.config.runtime_config,
            )

            self.model_record = ModelRecord(
                config=loaded.config,
                runtime=loaded.runtime_model.runtime,
                tokenizer=loaded.tokenizer,
                layer_specs=loaded.layer_specs,
                runtime_model=loaded.runtime_model,
            )

            self._page_size = loaded.runtime_model.runtime.page_size

            register_model = getattr(self.executor, "register_model", None)
            if callable(register_model):
                num_pages = register_model(self.config.model_id, self.model_record)
            else:
                raise RuntimeError("Executor has no register_model method")

            logger.info("Worker model loaded and ready")
            return num_pages

    def _resolve_executor_cls(self):
        if self.config.executor_cls == "PyptoQwen14BExecutor":
            from pypto_serving.model.qwen.npu_executor import Qwen314BPyptoExecutor

            return Qwen314BPyptoExecutor
        if self.config.executor_cls == "PyptoDeepSeekV4Executor":
            from pypto_serving.model.deepseek.npu_executor import DeepSeekV4PyptoExecutor

            return DeepSeekV4PyptoExecutor
        from pypto_serving.model.common.executor.executor import ModelExecutor

        return ModelExecutor

    def _configure_pypto_build_dir(self, device_ids: tuple[int, ...]) -> Path:
        """Give each worker process an isolated PyPTO build base."""
        base = Path(os.environ.get("PYPTO_PROG_BUILD_DIR") or "build_output")
        device_label = "_".join(str(device_id) for device_id in device_ids)
        worker_dir = base / f"serving_dp{self.config.dp_rank}_d{device_label}"
        os.environ["PYPTO_PROG_BUILD_DIR"] = str(worker_dir)
        return worker_dir

    def busy_loop(self) -> None:
        logger.info("Worker entering busy loop")
        config = getattr(self, "config", None)
        executor = getattr(self, "executor", None)
        async_enabled = bool(config is not None and config.resolve_async_scheduling())
        if async_enabled and executor is not None and executor.supports_async_decode_prepare:
            self._pipelined_busy_loop()
        else:
            self._serial_busy_loop()
        logger.info("Worker exiting")

    def _serial_busy_loop(self) -> None:
        """Original one-command-at-a-time worker path."""
        while True:
            try:
                raw: bytes = self.input_queue.get()
            except Exception:
                break

            try:
                cmd = decode_command(raw)
            except Exception as e:
                logger.error(f"Worker failed to decode command: {e}", exc_info=True)
                self.output_queue.put(encode_result(StepResult(new_tokens={}, error=str(e))))
                continue

            if isinstance(cmd, ShutdownCommand):
                logger.info("Worker received shutdown command")
                break

            if isinstance(cmd, ProfileCommand):
                self._handle_profile_command(cmd)
                continue

            self._handle_step_command(cmd)

    def _pipelined_busy_loop(self) -> None:
        """Run prepare, device dispatch, and host reclaim as three FIFO stages.

        The device lane submits steady fused-MTP work to PyPTO. It hands the
        asynchronous handle and ping-ponged output ticket to the reclaim lane,
        then immediately consumes the next prepared command.
        Prefill and lifecycle-heavy restart commands retain the serial fallback.
        """
        work_queue: queue.Queue[
            tuple[object, _PreparedDecodeWork | None, dict[str, object]]
        ] = queue.Queue(maxsize=_DECODE_PIPELINE_SLOTS)
        output_work_queue: queue.Queue[object] = queue.Queue()
        slot_ownership = [threading.Semaphore(1) for _ in range(_DECODE_PIPELINE_SLOTS)]
        output_thread = threading.Thread(
            target=self._output_reclaim_loop,
            args=(output_work_queue, slot_ownership),
            name="pypto-output",
        )
        device_thread = threading.Thread(
            target=self._device_execution_loop,
            args=(work_queue, output_work_queue),
            name="pypto-device",
        )
        output_thread.start()
        device_thread.start()
        try:
            while True:
                try:
                    raw: bytes = self.input_queue.get()
                except Exception:
                    work_queue.put((ShutdownCommand(), None, {}))
                    break
                try:
                    cmd = decode_command(raw)
                except Exception as exc:
                    logger.error("Worker failed to decode command: %s", exc, exc_info=True)
                    work_queue.put(
                        (
                            _DecodeCommandFailure(
                                StepResult(new_tokens={}, error=str(exc))
                            ),
                            None,
                            {},
                        )
                    )
                    continue
                release_cache_entries: dict[str, object] = {}
                if isinstance(cmd, StepCommand):
                    # Snapshot the entries this command intends to release before
                    # publishing newer registrations. Lifecycle release remains
                    # on the FIFO device lane; the snapshots prevent it from
                    # deleting a later registration that reused the same ID.
                    release_cache_entries = {
                        request_id: self._req_cache.get(request_id, _MISSING_REQUEST)
                        for request_id in cmd.finished_request_ids
                    }
                    for new_request in cmd.new_requests:
                        self._req_cache[new_request.request_id] = new_request
                owned_slot = None
                if (
                    isinstance(cmd, StepCommand)
                    and cmd.decode_requests
                    and not cmd.prefill_requests
                    and self.executor.supports_device_decode_embedding
                ):
                    owned_slot = cmd.step_id % _DECODE_PIPELINE_SLOTS
                    slot_ownership[owned_slot].acquire()
                try:
                    prepared = (
                        self._prepare_step_command(cmd, buffer_slot=owned_slot)
                        if isinstance(cmd, StepCommand)
                        else None
                    )
                except Exception as exc:
                    logger.error("Worker decode preparation failed: %s", exc, exc_info=True)
                    prepared = _PreparedDecodeWork(
                        prepared=None,
                        buffer_slot=owned_slot if owned_slot is not None else 0,
                        batch=None,
                        error=str(exc),
                    )
                work_queue.put((cmd, prepared, release_cache_entries))
                if isinstance(cmd, ShutdownCommand):
                    break
        finally:
            device_thread.join()
            output_thread.join()

    def _device_execution_loop(
        self,
        work_queue: queue.Queue[
            tuple[object, _PreparedDecodeWork | None, dict[str, object]]
        ],
        output_work_queue: queue.Queue[object],
    ) -> None:
        """Dispatch FIFO device work without reclaiming host outputs."""
        while True:
            cmd, prepared, release_cache_entries = work_queue.get()
            if isinstance(cmd, ShutdownCommand):
                output_work_queue.put(cmd)
                return
            if isinstance(cmd, _DecodeCommandFailure):
                output_work_queue.put(_CompletedStepOutput(cmd.result))
                continue
            if isinstance(cmd, ProfileCommand):
                barrier = _ProfileBarrier(command=cmd, completed=threading.Event())
                output_work_queue.put(barrier)
                barrier.completed.wait()
                continue
            assert isinstance(cmd, StepCommand)
            try:
                self._apply_command_lifecycle(cmd, release_cache_entries)
            except Exception as exc:
                logger.error("Worker command lifecycle failed: %s", exc, exc_info=True)
                output_work_queue.put(
                    _CompletedStepOutput(
                        StepResult(new_tokens={}, error=str(exc), step_id=cmd.step_id),
                        buffer_slot=prepared.buffer_slot if prepared is not None else None,
                    )
                )
                continue
            if self._can_split_decode_output(cmd, prepared):
                assert prepared is not None and prepared.batch is not None
                try:
                    with profile_span(
                        "WorkerProcess.dispatch_decode",
                        cat="worker",
                        args={"step_id": cmd.step_id, "buffer_slot": prepared.buffer_slot},
                    ):
                        dispatch_batch = self._late_bind_prepared_decode_batch(
                            prepared,
                            tuple(cmd.decode_requests),
                        )
                        pending = self.executor.dispatch_prepared_decode(
                            self.model_record.runtime_model,
                            dispatch_batch,
                            prepared.prepared,
                        )
                    output_work_queue.put(
                        _PendingDecodeOutput(
                            cmd=cmd,
                            scheduled=tuple(cmd.decode_requests),
                            pending=pending,
                            buffer_slot=prepared.buffer_slot,
                        )
                    )
                except Exception as exc:
                    logger.error("Worker decode dispatch failed: %s", exc, exc_info=True)
                    output_work_queue.put(
                        _CompletedStepOutput(
                            StepResult(new_tokens={}, error=str(exc), step_id=cmd.step_id),
                            buffer_slot=prepared.buffer_slot,
                        )
                    )
                continue
            try:
                result = self._execute_step(cmd, prepared_decode=prepared)
            except Exception as exc:
                logger.error("Worker step failed: %s", exc, exc_info=True)
                result = StepResult(new_tokens={}, error=str(exc), step_id=cmd.step_id)
            output_work_queue.put(
                _CompletedStepOutput(
                    result,
                    buffer_slot=prepared.buffer_slot if prepared is not None else None,
                )
            )

    def _output_reclaim_loop(
        self,
        output_work_queue: queue.Queue[object],
        slot_ownership: list[threading.Semaphore],
    ) -> None:
        """Reclaim completed outputs and publish StepResults in dispatch order."""
        while True:
            work = output_work_queue.get()
            if isinstance(work, ShutdownCommand):
                return
            if isinstance(work, _ProfileBarrier):
                try:
                    self._handle_profile_command(work.command)
                finally:
                    work.completed.set()
                continue
            if isinstance(work, _CompletedStepOutput):
                result = work.result
                buffer_slot = work.buffer_slot
            elif isinstance(work, _PendingDecodeOutput):
                result = self._reclaim_pending_decode(work)
                buffer_slot = work.buffer_slot
            else:
                raise TypeError(f"unexpected output work item: {type(work).__name__}")
            if buffer_slot is not None:
                slot_ownership[buffer_slot].release()
            self.output_queue.put(encode_result(result))

    def _can_split_decode_output(
        self,
        cmd: StepCommand,
        prepared: _PreparedDecodeWork | None,
    ) -> bool:
        """Return whether a command is safe for independent host reclaim."""
        return bool(
            prepared is not None
            and prepared.error is None
            and prepared.prepared is not None
            and prepared.batch is not None
            and cmd.decode_requests
            and not cmd.prefill_requests
            and not cmd.new_requests
            and bool(getattr(self.executor, "supports_async_decode_reclaim", False))
        )

    def _reclaim_pending_decode(self, work: _PendingDecodeOutput) -> StepResult:
        """Materialize tokens and lifecycle updates off the device lane."""
        try:
            with profile_span(
                "WorkerProcess.reclaim_decode",
                cat="worker",
                args={"step_id": work.cmd.step_id},
            ):
                decode_result = self.executor.reclaim_prepared_decode(work.pending)
                new_tokens: dict[str, list[int]] = {}
                self._consume_decode_result(work.scheduled, decode_result, new_tokens)
                for req_id, tokens in new_tokens.items():
                    if tokens:
                        self._record_last_tokens(req_id, tokens)
            return StepResult(new_tokens=new_tokens, step_id=work.cmd.step_id)
        except Exception as exc:
            logger.error("Worker decode reclaim failed: %s", exc, exc_info=True)
            return StepResult(new_tokens={}, error=str(exc), step_id=work.cmd.step_id)

    def _late_bind_prepared_decode_batch(
        self,
        prepared: _PreparedDecodeWork,
        scheduled: tuple[DecodeRequest, ...],
    ) -> DecodeBatch:
        """Patch a prior token only when the active executor requests it.

        DeepSeek fused MTP finalizes persistent state during prefill, so both
        cold and steady descriptors return the early-prepared batch unchanged.
        """
        assert prepared.batch is not None
        if not self.executor.prepared_decode_requires_token(prepared.prepared):
            return prepared.batch
        tokens = [self._resolve_decode_token(request) for request in scheduled]
        token_ids = torch.tensor(
            tokens,
            dtype=torch.long,
            device=self.model_record.runtime_model.runtime.device,
        ).unsqueeze(1)
        return replace(prepared.batch, token_ids=token_ids)

    def _handle_profile_command(self, cmd: ProfileCommand) -> None:
        """Apply a profile command and acknowledge it after the file is flushed."""
        profiler = get_profiler(initially_active=False)
        error = None
        try:
            if cmd.active:
                profiler.start()
            else:
                profiler.stop()
            if profiler.active != cmd.active:
                error = "SA profiling is not configured in the worker process"
        except Exception as exc:
            error = str(exc)
            logger.error("Worker profile command failed: %s", exc, exc_info=True)

        if self.profile_output_queue is not None:
            self.profile_output_queue.put(
                encode_profile_result(ProfileResult(active=profiler.active, error=error))
            )

    def _handle_step_command(self, cmd: StepCommand) -> None:
        """Handle a StepCommand and push an encoded StepResult.

        The whole body is guarded: an exception during request registration or
        device-resource release (steps 1-2) would otherwise propagate out of the
        busy loop and crash the worker. Any failure is reported back to the
        engine as an error result so the loop keeps serving.
        """
        self.output_queue.put(encode_result(self._run_step_command(cmd, None)))

    def _run_step_command(
        self,
        cmd: StepCommand,
        prepared: _PreparedDecodeWork | None,
    ) -> StepResult:
        """Apply lifecycle deltas and execute one FIFO command."""
        try:
            self._apply_command_lifecycle(cmd)
            return self._execute_step(cmd, prepared_decode=prepared)
        except Exception as e:
            logger.error(f"Worker step failed: {e}", exc_info=True)
            return StepResult(new_tokens={}, error=str(e), step_id=cmd.step_id)

    def _apply_command_lifecycle(
        self,
        cmd: StepCommand,
        expected_cache_entries: dict[str, object] | None = None,
    ) -> None:
        """Apply request release and registration on the FIFO device lane."""
        self._release_finished_request_state(
            cmd.finished_request_ids,
            expected_cache_entries=expected_cache_entries,
        )
        for new_request in cmd.new_requests:
            self._req_cache[new_request.request_id] = new_request

    def _release_finished_request_state(
        self,
        request_ids: list[str],
        *,
        expected_cache_entries: dict[str, object] | None = None,
    ) -> None:
        """Release executor and worker mirrors after older device work is safe."""
        if not request_ids:
            return
        release_finished = getattr(self.executor, "release_finished_requests", None)
        if callable(release_finished):
            release_finished(request_ids)
        for req_id in request_ids:
            if expected_cache_entries is not None:
                expected = expected_cache_entries.get(req_id, _MISSING_REQUEST)
                current = self._req_cache.get(req_id, _MISSING_REQUEST)
                if current is not expected:
                    continue
            self._req_cache.pop(req_id, None)
            self._last_tokens.pop(req_id, None)
            release_sampler = getattr(self.sampler, "release_requests", None)
            if callable(release_sampler):
                release_sampler([req_id])

    def _execute_step(
        self,
        cmd: StepCommand,
        prepared_decode: _PreparedDecodeWork | None = None,
    ) -> StepResult:
        """Execute one step using the lightweight IPC protocol."""
        runtime_model = self.model_record.runtime_model
        new_tokens: dict[str, list[int]] = {}

        with profile_span(
            "WorkerProcess.execute_step",
            cat="worker",
            args={"prefill": len(cmd.prefill_requests), "decode": len(cmd.decode_requests)},
        ):
            if cmd.prefill_requests:
                max_prefill_batch = self.executor.max_prefill_batch_size
                if max_prefill_batch is None:
                    self._batch_prefill(cmd.prefill_requests, runtime_model, new_tokens)
                else:
                    if max_prefill_batch <= 0:
                        raise ValueError("executor max_prefill_batch_size must be positive")
                    for chunk in self._partitioned_prefill_chunks(
                        cmd.prefill_requests,
                        max_prefill_batch,
                        self.executor.max_prefill_batch_size_per_partition,
                    ):
                        self._batch_prefill(chunk, runtime_model, new_tokens)
            if cmd.decode_requests:
                self._batch_decode(
                    cmd.decode_requests,
                    runtime_model,
                    new_tokens,
                    prepared_decode=prepared_decode,
                )

        # Retain the tokens just sampled so a following pipelined decode step can
        # resolve its PLACEHOLDER_TOKEN input from the worker cache.
        for req_id, tokens in new_tokens.items():
            if tokens:
                self._record_last_tokens(req_id, tokens)

        return StepResult(new_tokens=new_tokens, step_id=cmd.step_id)

    def _record_last_tokens(self, request_id: str, tokens: list[int]) -> None:
        """Remember the latest sampled token for async placeholder resolution."""
        recent = self._last_tokens.get(request_id, [])
        recent.extend(int(t) for t in tokens)
        self._last_tokens[request_id] = recent[-1:]

    @staticmethod
    def _partitioned_prefill_chunks(
        scheduled: list,
        max_batch: int,
        max_per_partition: int = 1,
    ) -> list[list]:
        """Pack requests within global and per-cache-partition limits."""
        if max_batch <= 0:
            raise ValueError("prefill max_batch must be positive")
        if max_per_partition <= 0:
            raise ValueError("prefill max_per_partition must be positive")
        pending = list(scheduled)
        chunks: list[list] = []
        while pending:
            chunk = []
            deferred = []
            partition_counts: dict[int, int] = {}
            for item in pending:
                partition = item.cache_partition
                can_add = len(chunk) < max_batch and (
                    partition is None
                    or partition_counts.get(partition, 0) < max_per_partition
                )
                if can_add:
                    chunk.append(item)
                    if partition is not None:
                        partition_counts[partition] = partition_counts.get(partition, 0) + 1
                else:
                    deferred.append(item)
            if not chunk:
                raise RuntimeError("unable to form a prefill dispatch chunk")
            chunks.append(chunk)
            pending = deferred
        return chunks

    def _batch_prefill(
        self,
        scheduled: list[PrefillRequest],
        runtime_model,
        new_tokens: dict[str, list[int]],
    ) -> None:
        with profile_span(
            "WorkerProcess.batch_prefill",
            cat="worker",
            args={"batch_size": len(scheduled), "request_ids": [pr.request_id for pr in scheduled]},
        ):
            device = runtime_model.runtime.device
            chunk_tokens_list = [pr.chunk_tokens for pr in scheduled]
            seq_lens = [pr.num_computed_tokens + len(pr.chunk_tokens) for pr in scheduled]
            chunk_starts = [pr.num_computed_tokens for pr in scheduled]
            block_ids_list = [pr.block_ids for pr in scheduled]
            allow_device_greedy_sampling = self._allow_device_sampled_ids(scheduled)
            allow_device_topk_sampling = self._allow_device_topk_sampling(scheduled)
            embedding_lookup = None
            if not self.executor.supports_device_embedding:
                embedding_lookup = lambda token_ids: self.executor.lookup_embeddings(
                    runtime_model, token_ids
                )

            prefill_result = self.executor.run_prefill(
                runtime_model,
                pack_prefill_batch(
                    request_ids=[pr.request_id for pr in scheduled],
                    token_chunks=chunk_tokens_list,
                    seq_lens=seq_lens,
                    chunk_starts=chunk_starts,
                    device=device,
                    embedding_lookup=embedding_lookup,
                    allow_device_greedy_sampling=allow_device_greedy_sampling,
                    allow_device_topk_sampling=allow_device_topk_sampling,
                    block_ids=block_ids_list,
                    block_ids_by_group=[pr.block_ids_by_group for pr in scheduled],
                    cache_partitions=[pr.cache_partition for pr in scheduled],
                ),
            )

            # Sample only for requests whose prefill chunk completes the prompt.
            completed_request_ids: list[str] = []
            completed_token_ids: list[int] = []
            completed_sampling_params: list[SamplingParams] = []
            for i, pr in enumerate(scheduled):
                cached = self._req_cache[pr.request_id]
                # num_prompt_tokens is len(prompt_token_ids), which we have in cache.
                will_be_computed = pr.num_computed_tokens + len(pr.chunk_tokens)
                if will_be_computed >= len(cached.prompt_token_ids):
                    logits = (
                        prefill_result.logits[i]
                        if prefill_result.logits.dim() > 1
                        else prefill_result.logits
                    )
                    params = SamplingParams(
                        temperature=cached.temperature,
                        top_p=cached.top_p,
                        top_k=cached.top_k,
                        seed=cached.seed,
                    )
                    token_id = self._sample_result_row(
                        prefill_result,
                        logits,
                        params,
                        pr.request_id,
                        i,
                        allow_device_greedy_sampling,
                        allow_device_topk_sampling=allow_device_topk_sampling,
                    )
                    new_tokens[pr.request_id] = [token_id]
                    completed_request_ids.append(pr.request_id)
                    completed_token_ids.append(token_id)
                    completed_sampling_params.append(params)

            # MTP prefill needs the first sampled output token to build its
            # shifted input. Keep that work in the terminal-prefill command so
            # the first decode only consumes already-initialized device state.
            if completed_request_ids:
                self.executor.finalize_prefill(
                    runtime_model,
                    completed_request_ids,
                    completed_token_ids,
                    completed_sampling_params,
                )

    def _resolve_decode_token(self, dr: DecodeRequest) -> int:
        """Return the decode input token, substituting from cache on placeholder.

        Sync scheduling sends the real ``last_token``. Async scheduling may send
        ``PLACEHOLDER_TOKEN`` when the step was built before the prior token was
        sampled; the worker then uses the most recent token it sampled.
        """
        if dr.last_token != PLACEHOLDER_TOKEN:
            return dr.last_token
        recent = self._last_tokens.get(dr.request_id)
        if not recent:
            raise RuntimeError(
                f"No cached token to resolve placeholder decode input for {dr.request_id!r}"
            )
        return recent[-1]

    def _prepare_step_command(
        self,
        cmd: StepCommand,
        *,
        buffer_slot: int | None = None,
    ) -> _PreparedDecodeWork | None:
        """Prepare decode-only metadata without touching request output state."""
        if cmd.prefill_requests or not cmd.decode_requests:
            return None
        if not self.executor.supports_device_decode_embedding:
            # Placeholder tokens cannot be embedded correctly until the prior
            # device result is available. Keep this executor on the serial path.
            return None
        runtime_model = self.model_record.runtime_model
        if buffer_slot is None:
            buffer_slot = cmd.step_id % _DECODE_PIPELINE_SLOTS
        allow_device_greedy_sampling = self._allow_device_sampled_ids(cmd.decode_requests)
        allow_device_topk_sampling = self._allow_device_topk_sampling(cmd.decode_requests)
        # Tokens remain placeholders during early preparation. Executors with a
        # host-token dependency may patch them on the device lane; fused MTP
        # consumes persistent state finalized by the terminal-prefill command.
        batch = self._make_decode_batch(
            cmd.decode_requests,
            runtime_model,
            resolve_tokens=False,
            allow_device_greedy_sampling=allow_device_greedy_sampling,
            allow_device_topk_sampling=allow_device_topk_sampling,
        )
        with profile_span(
            "WorkerProcess.prepare_decode",
            cat="worker",
            args={"step_id": cmd.step_id, "buffer_slot": buffer_slot},
        ):
            prepared = self.executor.prepare_decode(
                runtime_model,
                batch,
                buffer_slot=buffer_slot,
            )
        return _PreparedDecodeWork(prepared=prepared, buffer_slot=buffer_slot, batch=batch)

    def _make_decode_batch(
        self,
        scheduled: list[DecodeRequest],
        runtime_model,
        *,
        resolve_tokens: bool,
        allow_device_greedy_sampling: bool,
        allow_device_topk_sampling: bool,
    ) -> DecodeBatch:
        """Build a decode batch snapshot, optionally resolving prior output tokens."""
        device = runtime_model.runtime.device
        decode_tokens = (
            [self._resolve_decode_token(dr) for dr in scheduled]
            if resolve_tokens
            else [0] * len(scheduled)
        )
        decode_token_tensor = torch.tensor(decode_tokens, dtype=torch.long, device=device)
        if self.executor.supports_device_decode_embedding:
            decode_embeddings = None
        else:
            decode_embeddings = self.executor.lookup_embeddings(runtime_model, decode_token_tensor)
        return DecodeBatch(
            request_ids=[dr.request_id for dr in scheduled],
            token_ids=decode_token_tensor.unsqueeze(1),
            hidden_states=decode_embeddings,
            seq_lens=torch.tensor(
                [dr.seq_len for dr in scheduled], dtype=torch.int32, device=device
            ),
            allow_device_greedy_sampling=allow_device_greedy_sampling,
            allow_device_topk_sampling=allow_device_topk_sampling,
            sampling_params=[
                SamplingParams(
                    temperature=self._req_cache[dr.request_id].temperature,
                    top_p=self._req_cache[dr.request_id].top_p,
                    top_k=self._req_cache[dr.request_id].top_k,
                    seed=self._req_cache[dr.request_id].seed,
                )
                for dr in scheduled
            ],
            block_ids=[dr.block_ids for dr in scheduled],
            block_ids_by_group=[dr.block_ids_by_group for dr in scheduled],
            cache_partitions=[dr.cache_partition for dr in scheduled],
        )

    def _batch_decode(
        self,
        scheduled: list[DecodeRequest],
        runtime_model,
        new_tokens: dict[str, list[int]],
        prepared_decode: _PreparedDecodeWork | None = None,
    ) -> None:
        with profile_span(
            "WorkerProcess.batch_decode",
            cat="worker",
            args={"batch_size": len(scheduled), "request_ids": [dr.request_id for dr in scheduled]},
        ):
            allow_device_greedy_sampling = self._allow_device_sampled_ids(scheduled)
            allow_device_topk_sampling = self._allow_device_topk_sampling(scheduled)

            batch = self._make_decode_batch(
                scheduled,
                runtime_model,
                resolve_tokens=True,
                allow_device_greedy_sampling=allow_device_greedy_sampling,
                allow_device_topk_sampling=allow_device_topk_sampling,
            )
            if prepared_decode is None:
                decode_result = self.executor.run_decode(runtime_model, batch)
            else:
                if prepared_decode.error is not None:
                    raise RuntimeError(prepared_decode.error)
                decode_result = self.executor.run_prepared_decode(
                    runtime_model,
                    batch,
                    prepared_decode.prepared,
                )
            self._consume_decode_result(scheduled, decode_result, new_tokens)

    def _consume_decode_result(
        self,
        scheduled: tuple[DecodeRequest, ...] | list[DecodeRequest],
        decode_result: DecodeResult,
        new_tokens: dict[str, list[int]],
    ) -> None:
        """Convert a runner result to tokens on either serial or reclaim lane."""
        if decode_result.accepted_token_ids is not None:
            for i, dr in enumerate(scheduled):
                new_tokens[dr.request_id] = list(decode_result.accepted_token_ids[i])
            return
        allow_device_greedy_sampling = self._allow_device_sampled_ids(list(scheduled))
        allow_device_topk_sampling = self._allow_device_topk_sampling(list(scheduled))
        for i, dr in enumerate(scheduled):
            cached = self._req_cache[dr.request_id]
            logits = None
            if decode_result.logits is not None:
                logits = (
                    decode_result.logits[i]
                    if decode_result.logits.dim() > 1
                    else decode_result.logits
                )
            params = SamplingParams(
                temperature=cached.temperature,
                top_p=cached.top_p,
                top_k=cached.top_k,
                seed=cached.seed,
            )
            token_id = self._sample_result_row(
                decode_result,
                logits,
                params,
                dr.request_id,
                i,
                allow_device_greedy_sampling,
                allow_device_topk_sampling=allow_device_topk_sampling,
            )
            new_tokens[dr.request_id] = [token_id]

    def close(self) -> None:
        """Release executor-owned runtime and device resources."""
        executor = self.executor
        self.executor = None
        if executor is None:
            return

        close = getattr(executor, "close", None)
        if callable(close):
            close()

    def _sample_result_row(
        self,
        result,
        logits: torch.Tensor | None,
        params: SamplingParams,
        request_id: str,
        row_idx: int,
        allow_device_sampled: bool,
        allow_device_topk_sampling: bool,
    ) -> int:
        """Return a sampled token from executor output, falling back to host sampling."""
        sampled = getattr(result, "sampled_token_ids", None)
        if allow_device_sampled and sampled is not None:
            flat = sampled.view(-1)
            if flat.numel() <= row_idx:
                raise ValueError(
                    f"sampled_token_ids has {flat.numel()} rows, expected row {row_idx}"
                )
            return int(flat[row_idx].item())
        candidates = getattr(result, "sampling_candidates", None)
        if allow_device_topk_sampling and candidates is not None:
            return self.sampler.sample_from_candidates(
                candidates,
                row_idx,
                params,
                request_id,
            )
        return self.sampler.sample(logits, params, request_id)

    def _allow_device_topk_sampling(self, scheduled: list) -> bool:
        """Return whether a scheduled batch can use executor top-k candidates."""
        max_device_topk = self.executor.device_topk_sampling_k
        cached_requests = [self._req_cache[item.request_id] for item in scheduled]
        return (
            max_device_topk > 0
            and all(request.temperature > 0.0 for request in cached_requests)
            and all(request.top_k is not None for request in cached_requests)
            and all(request.top_k > 0 for request in cached_requests)
            and all(request.top_k <= max_device_topk for request in cached_requests)
        )

    def _allow_device_sampled_ids(self, scheduled: list) -> bool:
        """Return whether the executor can perform every request's sampling policy."""
        if not self.executor.supports_device_sampling:
            return False
        requests = [self._req_cache[item.request_id] for item in scheduled]
        return all(
            request.temperature <= 0.0
            or (
                getattr(self.executor, "supports_device_stochastic_sampling", False)
                and request.top_p >= 1.0
            )
            for request in requests
        )


def _worker_entry(
    config: EngineConfig,
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    ready_event,
    num_pages_value,
    profile_output_queue: mp.Queue | None = None,
):
    """Entry point for the worker subprocess."""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Spawned workers do not inherit the parent's logging config; configure a
    # stderr handler so per-stage progress logs (weight load, preflight) are
    # visible alongside kernel/perf output.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )
    for _n in ("simpler_setup", "pypto", "simpler"):
        logging.getLogger(_n).setLevel(logging.WARNING)

    worker = WorkerProcess(config, input_queue, output_queue, profile_output_queue)
    try:
        num_pages = worker.init_device_and_model()
        num_pages_value.value = num_pages
        # Model weights, compiled kernels and KV-cache objects are now resident.
        # Freeze them so the GC won't rescan them during decode (avoids
        # multi-ms gen2 pauses landing mid-step). Must happen in this process:
        # gc.freeze() does not cross the spawn boundary.
        freeze_gc_heap()
        ready_event.set()
        worker.busy_loop()
    except Exception as e:
        logger.error(f"Worker process failed: {e}", exc_info=True)
        ready_event.set()
    finally:
        try:
            worker.close()
        except Exception:
            logger.exception("Worker process cleanup failed")
        get_profiler(initially_active=False).stop()


def spawn_worker(config: EngineConfig):
    """Spawn a worker process and return its process, queues, and ready state.

    ``num_pages_value`` is a shared ``multiprocessing.Value('i')`` that the
    worker writes after ``init_device_and_model()`` completes.  The main
    process reads it to synchronise the ``KvCacheManager`` block metadata with
    the actual device-side KV cache size. Profile acknowledgements use a
    dedicated output queue so they cannot be mistaken for inference results.
    """
    ctx = mp.get_context("spawn")
    input_queue = ctx.Queue()
    output_queue = ctx.Queue()
    profile_output_queue = ctx.Queue()
    ready_event = ctx.Event()
    num_pages_value = ctx.Value("i", 0)

    process = ctx.Process(
        target=_worker_entry,
        args=(
            config,
            input_queue,
            output_queue,
            ready_event,
            num_pages_value,
            profile_output_queue,
        ),
        daemon=False,
    )
    process.start()
    return (
        process,
        input_queue,
        output_queue,
        profile_output_queue,
        ready_event,
        num_pages_value,
    )
