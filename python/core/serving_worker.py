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

import torch

from typing import TYPE_CHECKING

from python.profile import get_profiler, profile_span
from .types import (
    DecodeBatch,
    PrefillBatch,
    SamplingParams,
    StepOutput,
    WorkerCommand,
)

if TYPE_CHECKING:
    from .async_engine import EngineConfig

logger = logging.getLogger(__name__)


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
    ):
        self.config = config
        self.input_queue = input_queue
        self.output_queue = output_queue

        self.executor = None
        self.sampler = None
        self.model_record = None
        self._page_size: int = 64

    def init_device_and_model(self) -> None:
        from .model_loader import ModelLoader
        from .sampler import Sampler
        from .types import ModelRecord

        device_ids = self.config.worker_device_ids()
        device_label = ",".join(str(device_id) for device_id in device_ids)
        if mp.current_process().name != "MainProcess":
            get_profiler(process_name=f"serving-worker-{device_label}")
        with profile_span(
            "WorkerProcess.init_device_and_model",
            cat="worker",
            args={
                "model_id": self.config.model_id,
                "device_id": self.config.device_id,
                "device_ids": list(device_ids),
                "dp_rank": self.config.dp_rank,
            },
        ):
            logger.info(
                f"Worker initializing: platform={self.config.platform}, "
                f"devices={list(device_ids)}, dp_rank={self.config.dp_rank}"
            )

            self.sampler = Sampler()

            executor_cls = self._resolve_executor_cls()
            self.executor = executor_cls(
                platform=self.config.platform,
                device_id=self.config.device_id,
                device_ids=device_ids,
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
                register_model(self.config.model_id, self.model_record)

            logger.info("Worker model loaded and ready")

    def _resolve_executor_cls(self):
        if self.config.executor_cls == "PyptoQwen14BExecutor":
            from examples.model.qwen3_14b.runner.npu_executor import Qwen314BPyptoExecutor
            return Qwen314BPyptoExecutor
        if self.config.executor_cls == "PyptoDeepSeekV4Executor":
            from examples.model.deepseek_v4.runner.npu_executor import DeepSeekV4PyptoExecutor
            return DeepSeekV4PyptoExecutor
        from .executor import ModelExecutor
        return ModelExecutor

    def busy_loop(self) -> None:
        logger.info("Worker entering busy loop")
        while True:
            try:
                cmd: WorkerCommand = self.input_queue.get()
            except Exception:
                break

            if cmd.type == "shutdown":
                logger.info("Worker received shutdown command")
                break
            elif cmd.type == "step":
                if cmd.finished_request_ids:
                    release_finished = getattr(self.executor, "release_finished_requests", None)
                    if callable(release_finished):
                        release_finished(cmd.finished_request_ids)

                try:
                    result = self._execute_step(cmd.scheduler_output)
                    self.output_queue.put(result)
                except Exception as e:
                    logger.error(f"Worker step failed: {e}", exc_info=True)
                    self.output_queue.put(StepOutput(new_tokens={}, error=str(e)))
            else:
                logger.warning(f"Worker received unknown command: {cmd.type}")

        logger.info("Worker exiting")

    def _execute_step(self, scheduler_output) -> StepOutput:
        """Execute one batch step (may contain prefill + decode requests)."""
        runtime_model = self.model_record.runtime_model
        new_tokens: dict[str, int] = {}

        prefill_requests = [
            sr for sr in scheduler_output.scheduled_requests if sr.is_prefill
        ]
        decode_requests = [
            sr for sr in scheduler_output.scheduled_requests if not sr.is_prefill
        ]

        with profile_span(
            "WorkerProcess.execute_step",
            cat="worker",
            args={"prefill": len(prefill_requests), "decode": len(decode_requests)},
        ):
            with self.executor.session():
                if prefill_requests:
                    self._batch_prefill(prefill_requests, runtime_model, new_tokens)
                if decode_requests:
                    self._batch_decode(decode_requests, runtime_model, new_tokens)

        return StepOutput(new_tokens=new_tokens)

    def _batch_prefill(
        self, scheduled: list, runtime_model, new_tokens: dict[str, int]
    ) -> None:
        with profile_span(
            "WorkerProcess.batch_prefill",
            cat="worker",
            args={
                "batch_size": len(scheduled),
                "request_ids": [sr.request.request_id for sr in scheduled],
            },
        ):
            device = runtime_model.runtime.device
            batch_size = len(scheduled)

            chunk_tokens_list = []
            positions_list = []
            seq_lens = []
            block_ids_list = []

            for sr in scheduled:
                request = sr.request
                num_computed = sr.num_computed_tokens
                num_new = sr.num_new_tokens

                chunk_tokens = request.prompt_token_ids[num_computed:num_computed + num_new]
                chunk_tokens_list.append(chunk_tokens)

                positions = range(num_computed, num_computed + num_new)
                positions_list.append(positions)

                seq_lens.append(num_computed + num_new)
                block_ids_list.append(sr.block_ids)

            max_chunk = max(len(t) for t in chunk_tokens_list)
            token_tensor = torch.zeros((batch_size, max_chunk), dtype=torch.long, device=device)
            embeddings = torch.zeros(
                (batch_size, max_chunk, self.model_record.config.hidden_size),
                dtype=runtime_model.embed_tokens.dtype,
                device=device,
            )
            positions_tensor = torch.full((batch_size, max_chunk), -1, dtype=torch.long, device=device)

            for i, tokens in enumerate(chunk_tokens_list):
                row = torch.tensor(tokens, dtype=torch.long, device=device)
                token_tensor[i, : len(tokens)] = row
                embeddings[i, : len(tokens), :] = self.executor.lookup_embeddings(
                    runtime_model, row
                )
                positions_tensor[i, : len(tokens)] = torch.tensor(
                    list(positions_list[i]), dtype=torch.long, device=device
                )

            group_block_ids_list = [sr.block_ids_by_group for sr in scheduled]
            prefill_result = self.executor.run_prefill(
                runtime_model,
                PrefillBatch(
                    request_ids=[sr.request.request_id for sr in scheduled],
                    token_ids=token_tensor,
                    input_embeddings=embeddings,
                    seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
                    positions=positions_tensor,
                    block_ids=block_ids_list,
                    block_ids_by_group=group_block_ids_list,
                ),
            )

            for i, sr in enumerate(scheduled):
                request = sr.request
                will_be_computed = sr.num_computed_tokens + sr.num_new_tokens
                if will_be_computed >= request.num_prompt_tokens:
                    logits = (
                        prefill_result.logits[i]
                        if prefill_result.logits.dim() > 1
                        else prefill_result.logits
                    )
                    params = SamplingParams(
                        temperature=request.temperature,
                        top_p=request.top_p,
                        top_k=request.top_k,
                    )
                    token_id = self.sampler.sample(logits, params)
                    new_tokens[request.request_id] = token_id

    def _batch_decode(
        self, scheduled: list, runtime_model, new_tokens: dict[str, int]
    ) -> None:
        with profile_span(
            "WorkerProcess.batch_decode",
            cat="worker",
            args={
                "batch_size": len(scheduled),
                "request_ids": [sr.request.request_id for sr in scheduled],
            },
        ):
            device = runtime_model.runtime.device

            decode_tokens = []
            block_ids_list = []
            seq_lens = []

            for sr in scheduled:
                request = sr.request
                last_token = (
                    request.output_token_ids[-1]
                    if request.output_token_ids
                    else request.prompt_token_ids[-1]
                )
                decode_tokens.append(last_token)
                block_ids_list.append(sr.block_ids)
                seq_lens.append(request.num_tokens)

            decode_token_tensor = torch.tensor(decode_tokens, dtype=torch.long, device=device)
            decode_embeddings = self.executor.lookup_embeddings(runtime_model, decode_token_tensor)

            group_block_ids_list = [sr.block_ids_by_group for sr in scheduled]
            decode_result = self.executor.run_decode(
                runtime_model,
                DecodeBatch(
                    request_ids=[sr.request.request_id for sr in scheduled],
                    token_ids=decode_token_tensor.unsqueeze(1),
                    hidden_states=decode_embeddings,
                    seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
                    block_ids=block_ids_list,
                    block_ids_by_group=group_block_ids_list,
                ),
            )

            for i, sr in enumerate(scheduled):
                request = sr.request
                logits = (
                    decode_result.logits[i]
                    if decode_result.logits.dim() > 1
                    else decode_result.logits
                )
                params = SamplingParams(
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                )
                token_id = self.sampler.sample(logits, params)
                new_tokens[request.request_id] = token_id

def _worker_entry(
    config: EngineConfig,
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    ready_event,
):
    """Entry point for the worker subprocess."""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    worker = WorkerProcess(config, input_queue, output_queue)
    try:
        worker.init_device_and_model()
        ready_event.set()
        worker.busy_loop()
    except Exception as e:
        logger.error(f"Worker process failed: {e}", exc_info=True)
        ready_event.set()


def spawn_worker(config: EngineConfig) -> tuple[mp.Process, mp.Queue, mp.Queue, mp.Event]:
    """Spawn a worker process and return (process, input_queue, output_queue, ready_event)."""
    ctx = mp.get_context("spawn")
    input_queue = ctx.Queue()
    output_queue = ctx.Queue()
    ready_event = ctx.Event()

    process = ctx.Process(
        target=_worker_entry,
        args=(config, input_queue, output_queue, ready_event),
        daemon=False,
    )
    process.start()
    return process, input_queue, output_queue, ready_event
