# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import itertools
from collections.abc import Iterator

import torch

from pypto_serving.config.types import (
    DecodeBatch,
    GenerateConfig,
    GenerateResult,
    ModelRecord,
    PrefillBatch,
    RequestState,
    RuntimeConfig,
)
from pypto_serving.model.common.executor.executor import ModelExecutor
from pypto_serving.model.common.executor.sampler import Sampler
from pypto_serving.model.model_loader import ModelLoader
from pypto_serving.serving.memory.kv_cache import KvCacheManager
from pypto_serving.tools.profile import profile_span


class LLMEngine:
    """High-level model registry and text generation coordinator."""

    def __init__(
        self,
        model_loader: ModelLoader | None = None,
        kv_cache_manager: KvCacheManager | None = None,
        executor: ModelExecutor | None = None,
        sampler: Sampler | None = None,
    ) -> None:
        """Create an engine from pluggable loader, cache, executor, and sampler."""
        self._model_loader = model_loader or ModelLoader()
        self._kv_cache_manager = kv_cache_manager or KvCacheManager()
        if executor is None:
            raise ValueError("LLMEngine requires a ModelExecutor instance.")
        self._executor = executor
        self._sampler = sampler or Sampler()
        self._models: dict[str, ModelRecord] = {}
        self._request_counter = itertools.count()

    def init_model(
        self,
        model_id: str,
        model_dir: str,
        runtime_config: RuntimeConfig | None = None,
        model_format: str | None = None,
        **loader_options: object,
    ) -> None:
        """Load a model, register its KV cache, and notify the executor."""
        with profile_span("LLMEngine.init_model", cat="engine", args={"model_id": model_id}):
            loaded = self._model_loader.load(
                model_id=model_id,
                model_dir=model_dir,
                runtime_config=runtime_config,
                model_format=model_format,
                **loader_options,
            )
            config = loaded.config
            runtime = loaded.runtime_model.runtime
            self._kv_cache_manager.register_model(model_id, config, runtime)
            self._models[model_id] = ModelRecord(
                config=config,
                runtime=runtime,
                tokenizer=loaded.tokenizer,
                layer_specs=loaded.layer_specs,
                runtime_model=loaded.runtime_model,
            )
            register_model = getattr(self._executor, "register_model", None)
            if callable(register_model):
                register_model(model_id, self._models[model_id])

    def generate(self, model_id: str, prompt: str, config: GenerateConfig | None = None) -> str | Iterator[str]:
        """Generate text for one prompt, optionally returning a text stream."""
        generate_config = config or GenerateConfig()
        if generate_config.stream:
            return self._generate_stream(model_id, prompt, generate_config)
        return self._generate_result(model_id, prompt, generate_config).text

    def _generate_non_stream(self, model_id: str, prompt: str, config: GenerateConfig) -> str:
        """Generate non-streaming text for one prompt."""
        return self._generate_result(model_id, prompt, config).text

    def generate_batch(
        self,
        model_id: str,
        prompts: list[str] | tuple[str, ...],
        config: GenerateConfig | None = None,
    ) -> list[GenerateResult]:
        """Generate non-streaming completions for a batch of prompts."""
        generate_config = config or GenerateConfig()
        if generate_config.stream:
            raise ValueError("generate_batch requires stream=False")
        with profile_span(
            "LLMEngine.generate_batch",
            cat="engine",
            args={
                "model_id": model_id,
                "batch_size": len(prompts),
                "max_new_tokens": generate_config.max_new_tokens,
            },
        ):
            return self._generate_batch_impl(model_id, prompts, generate_config)

    def _generate_batch_impl(
        self,
        model_id: str,
        prompts: list[str] | tuple[str, ...],
        generate_config: GenerateConfig,
    ) -> list[GenerateResult]:
        if not prompts:
            return []
        if model_id not in self._models:
            raise KeyError(f"Model {model_id} is not initialized.")
        record = self._models[model_id]
        if len(prompts) > record.runtime.max_batch_size:
            max_batch_size = record.runtime.max_batch_size
            raise ValueError(
                f"batch has {len(prompts)} prompts, but runtime max_batch_size is {max_batch_size}"
            )

        runtime_model = record.runtime_model
        tokenizer = record.tokenizer
        prompt_token_ids = [tokenizer.encode(prompt) for prompt in prompts]
        for token_ids in prompt_token_ids:
            if not token_ids and record.config.bos_token_id is not None:
                token_ids.append(record.config.bos_token_id)
            if not token_ids:
                raise ValueError("Prompt tokenization produced no tokens.")

        self._executor.validate_generate_batch(record, len(prompts), generate_config)

        requests: list[RequestState] = []
        allocations = []
        try:
            for prompt, token_ids in zip(prompts, prompt_token_ids, strict=True):
                request_id = f"req-{next(self._request_counter)}"
                alloc_len = self._executor.prompt_allocation_length(
                    record,
                    len(token_ids),
                    generate_config,
                )
                alloc = self._kv_cache_manager.allocate_for_prompt(model_id, request_id, alloc_len)
                allocations.append(alloc)
                requests.append(
                    RequestState(
                        request_id=request_id,
                        model_id=model_id,
                        prompt=prompt,
                        prompt_token_ids=token_ids,
                        max_new_tokens=generate_config.max_new_tokens,
                        stop_strings=generate_config.stop,
                        eos_token_id=record.config.eos_token_id,
                        seq_len=len(token_ids),
                        num_prompt_tokens=len(token_ids),
                        kv_allocation=alloc,
                    )
                )

            max_prompt_len = max(len(token_ids) for token_ids in prompt_token_ids)
            allow_device_greedy_sampling = (
                generate_config.temperature <= 0.0
                and self._executor.supports_device_sampling
                and self._executor.supports_device_embedding
            )
            token_tensor = torch.zeros(
                (len(prompt_token_ids), max_prompt_len),
                dtype=torch.long,
                device=runtime_model.runtime.device,
            )
            embeddings = None
            if not self._executor.supports_device_embedding:
                embeddings = torch.zeros(
                    (len(prompt_token_ids), max_prompt_len, record.config.hidden_size),
                    dtype=runtime_model.embed_tokens.dtype,
                    device=runtime_model.runtime.device,
                )
            for batch_idx, token_ids in enumerate(prompt_token_ids):
                row_tokens = torch.tensor(token_ids, dtype=torch.long, device=runtime_model.runtime.device)
                token_tensor[batch_idx, : len(token_ids)] = row_tokens
                if embeddings is not None:
                    embeddings[batch_idx, : len(token_ids), :] = self._executor.lookup_embeddings(
                        runtime_model,
                        row_tokens,
                    )

            prefill_batch = PrefillBatch(
                request_ids=[request.request_id for request in requests],
                token_ids=token_tensor,
                input_embeddings=embeddings,
                seq_lens=torch.tensor(
                    [len(token_ids) for token_ids in prompt_token_ids],
                    dtype=torch.int32,
                    device=runtime_model.runtime.device,
                ),
                allow_device_greedy_sampling=allow_device_greedy_sampling,
                kv_allocations=allocations,
            )
            prefill_token_budget = record.runtime.max_num_batched_tokens
            if prefill_token_budget <= 0:
                raise ValueError("max_num_batched_tokens must be positive")
            long_prefill_threshold = record.runtime.long_prefill_token_threshold
            if long_prefill_threshold > 0:
                prefill_token_budget = min(prefill_token_budget, long_prefill_threshold)
            total_prefill_tokens = sum(len(token_ids) for token_ids in prompt_token_ids)
            batch_fits_budget = total_prefill_tokens <= prefill_token_budget
            if batch_fits_budget:
                fast_path_result = self._executor.try_generate_batch(
                    record,
                    requests,
                    prefill_batch,
                    generate_config,
                )
                if fast_path_result is not None:
                    return fast_path_result

            with self._executor.session():
                if batch_fits_budget:
                    prefill_result = self._executor.run_prefill(runtime_model, prefill_batch)
                    prefill_logits = prefill_result.logits
                    prefill_sampled_token_ids = (
                        prefill_result.sampled_token_ids
                        if prefill_batch.allow_device_greedy_sampling
                        else None
                    )
                else:
                    prefill_logits, prefill_sampled_token_ids = self._run_prefill_in_chunks(
                        runtime_model,
                        prefill_batch,
                        prefill_token_budget,
                    )

                sampling_params = self._sampler.from_generate_config(generate_config)
                current_tokens = self._sample_batch_rows(
                    prefill_logits,
                    sampling_params,
                    len(requests),
                    prefill_sampled_token_ids,
                )
                active_indices = list(range(len(requests)))
                finish_reasons = ["length"] * len(requests)

                for _ in range(generate_config.max_new_tokens):
                    next_active: list[int] = []
                    decode_tokens: list[int] = []
                    for request_idx in active_indices:
                        request = requests[request_idx]
                        current_token = current_tokens[request_idx]
                        request.generated_token_ids.append(current_token)
                        request.output_text = tokenizer.decode(request.generated_token_ids)

                        if record.config.eos_token_id is not None and current_token == record.config.eos_token_id:
                            finish_reasons[request_idx] = "eos"
                            continue
                        if any(stop and request.output_text.endswith(stop) for stop in generate_config.stop):
                            finish_reasons[request_idx] = "stop"
                            continue
                        if len(request.generated_token_ids) >= generate_config.max_new_tokens:
                            finish_reasons[request_idx] = "length"
                            continue

                        alloc = request.kv_allocation
                        if alloc is None:
                            raise RuntimeError("Request is missing KV allocation.")
                        self._kv_cache_manager.ensure_one_more_slot(alloc)
                        request.seq_len += 1
                        next_active.append(request_idx)
                        decode_tokens.append(current_token)

                    if not next_active:
                        break

                    decode_token_tensor = torch.tensor(
                        decode_tokens,
                        dtype=torch.long,
                        device=runtime_model.runtime.device,
                    )
                    decode_embeddings = self._decode_embeddings_from_cache_or_lookup(
                        runtime_model,
                        decode_token_tensor,
                    )
                    active_allocations = []
                    for idx in next_active:
                        alloc = requests[idx].kv_allocation
                        if alloc is None:
                            raise RuntimeError("Request is missing KV allocation.")
                        active_allocations.append(alloc)
                    decode_result = self._executor.run_decode(
                        runtime_model,
                        DecodeBatch(
                            request_ids=[requests[idx].request_id for idx in next_active],
                            token_ids=decode_token_tensor.unsqueeze(1),
                            hidden_states=decode_embeddings,
                            seq_lens=torch.tensor(
                                [requests[idx].seq_len for idx in next_active],
                                dtype=torch.int32,
                                device=runtime_model.runtime.device,
                            ),
                            allow_device_greedy_sampling=allow_device_greedy_sampling,
                            kv_allocations=active_allocations,
                        ),
                    )
                    decoded_tokens = self._sample_batch_rows(
                        decode_result.logits,
                        sampling_params,
                        len(next_active),
                        decode_result.sampled_token_ids if allow_device_greedy_sampling else None,
                    )
                    for row_idx, request_idx in enumerate(next_active):
                        current_tokens[request_idx] = decoded_tokens[row_idx]
                    active_indices = next_active
        finally:
            for alloc in allocations:
                self._kv_cache_manager.free(alloc)

        return [
            GenerateResult(
                text=request.output_text,
                token_ids=list(request.generated_token_ids),
                finish_reason=finish_reasons[request_idx],
            )
            for request_idx, request in enumerate(requests)
        ]

    def _generate_stream(self, model_id: str, prompt: str, config: GenerateConfig) -> Iterator[str]:
        """Yield decoded text deltas for one streaming prompt."""
        if model_id not in self._models:
            raise KeyError(f"Model {model_id} is not initialized.")
        record = self._models[model_id]
        runtime_model = record.runtime_model
        tokenizer = record.tokenizer
        prompt_token_ids = tokenizer.encode(prompt)
        if not prompt_token_ids and record.config.bos_token_id is not None:
            prompt_token_ids = [record.config.bos_token_id]
        if not prompt_token_ids:
            raise ValueError("Prompt tokenization produced no tokens.")

        request_id = f"req-{next(self._request_counter)}"
        alloc = self._kv_cache_manager.allocate_for_prompt(model_id, request_id, len(prompt_token_ids))
        request = RequestState(
            request_id=request_id,
            model_id=model_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=config.max_new_tokens,
            stop_strings=config.stop,
            eos_token_id=record.config.eos_token_id,
            seq_len=len(prompt_token_ids),
            num_prompt_tokens=len(prompt_token_ids),
            kv_allocation=alloc,
        )

        try:
            token_tensor = torch.tensor(prompt_token_ids, dtype=torch.long, device=runtime_model.runtime.device)
            embeddings = None
            if not self._executor.supports_device_embedding:
                embeddings = self._executor.lookup_embeddings(runtime_model, token_tensor).unsqueeze(0)

            with self._executor.session():
                prefill_result = self._executor.run_prefill(
                    runtime_model,
                    PrefillBatch(
                        request_ids=[request.request_id],
                        token_ids=token_tensor.unsqueeze(0),
                        input_embeddings=embeddings,
                        seq_lens=torch.tensor(
                            [len(prompt_token_ids)],
                            dtype=torch.int32,
                            device=runtime_model.runtime.device,
                        ),
                        kv_allocations=[alloc],
                    ),
                )

                logits = self._select_batch_row(prefill_result.logits, 0)
                generated: list[int] = []
                emitted_text = ""
                sampling_params = self._sampler.from_generate_config(config)
                current_token = self._sampler.sample(logits, sampling_params)

                for _ in range(config.max_new_tokens):
                    generated.append(current_token)
                    text = tokenizer.decode(generated)
                    delta = text[len(emitted_text) :]
                    emitted_text = text
                    if delta:
                        yield delta
                    if self._should_stop(record, config, generated, emitted_text, current_token):
                        break

                    self._kv_cache_manager.ensure_one_more_slot(alloc)
                    request.seq_len += 1
                    decode_token = torch.tensor([current_token], dtype=torch.long, device=runtime_model.runtime.device)
                    decode_embeddings = self._decode_embeddings_from_cache_or_lookup(
                        runtime_model,
                        decode_token,
                    )
                    decode_result = self._executor.run_decode(
                        runtime_model,
                        DecodeBatch(
                            request_ids=[request.request_id],
                            token_ids=decode_token.unsqueeze(0),
                            hidden_states=decode_embeddings,
                            seq_lens=torch.tensor(
                                [request.seq_len],
                                dtype=torch.int32,
                                device=runtime_model.runtime.device,
                            ),
                            kv_allocations=[alloc],
                        ),
                    )
                    logits = self._select_batch_row(decode_result.logits, 0)
                    current_token = self._sampler.sample(logits, sampling_params)
        finally:
            self._kv_cache_manager.free(alloc)

    def generate_result(self, model_id: str, prompt: str, config: GenerateConfig | None = None) -> GenerateResult:
        """Generate a structured non-streaming result for one prompt."""
        generate_config = config or GenerateConfig()
        if generate_config.stream:
            raise ValueError("generate_result requires stream=False")
        return self._generate_result(model_id, prompt, generate_config)

    def _generate_result(self, model_id: str, prompt: str, config: GenerateConfig) -> GenerateResult:
        """Generate one result by reusing the batch path."""
        return self.generate_batch(model_id, [prompt], config)[0]

    def _run_prefill_in_chunks(
        self,
        runtime_model,
        batch: PrefillBatch,
        token_budget: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run prefill calls whose combined chunk size does not exceed the budget."""
        if token_budget <= 0:
            raise ValueError("max_num_batched_tokens must be positive")

        row_count = len(batch.request_ids)
        prompt_lengths = [int(batch.seq_lens[row].item()) for row in range(row_count)]
        computed_tokens = [0] * row_count
        final_logits: list[torch.Tensor | None] = [None] * row_count
        final_sampled_ids: list[torch.Tensor | None] = [None] * row_count
        next_request_idx = 0

        while any(computed_tokens[row] < prompt_lengths[row] for row in range(row_count)):
            active_rows: list[int] = []
            for offset in range(row_count):
                row = (next_request_idx + offset) % row_count
                if computed_tokens[row] < prompt_lengths[row]:
                    active_rows.append(row)
            selected_rows = active_rows[:1]
            per_request_budget = token_budget
            chunk_lengths = [
                min(prompt_lengths[row] - computed_tokens[row], per_request_budget)
                for row in selected_rows
            ]
            max_chunk_len = max(chunk_lengths)

            chunk_token_ids = batch.token_ids.new_zeros((len(selected_rows), max_chunk_len))
            chunk_embeddings = None
            if batch.input_embeddings is not None:
                chunk_embeddings = batch.input_embeddings.new_zeros(
                    (len(selected_rows), max_chunk_len, *batch.input_embeddings.shape[2:])
                )
            chunk_positions = torch.full(
                (len(selected_rows), max_chunk_len),
                -1,
                dtype=torch.long,
                device=batch.token_ids.device,
            )
            chunk_seq_lens = batch.seq_lens.new_empty((len(selected_rows),))

            for chunk_row, (request_row, chunk_len) in enumerate(
                zip(selected_rows, chunk_lengths, strict=True)
            ):
                chunk_start = computed_tokens[request_row]
                chunk_end = chunk_start + chunk_len
                chunk_token_ids[chunk_row, :chunk_len] = batch.token_ids[
                    request_row, chunk_start:chunk_end
                ]
                if chunk_embeddings is not None and batch.input_embeddings is not None:
                    chunk_embeddings[chunk_row, :chunk_len] = batch.input_embeddings[
                        request_row, chunk_start:chunk_end
                    ]
                chunk_positions[chunk_row, :chunk_len] = torch.arange(
                    chunk_start,
                    chunk_end,
                    dtype=torch.long,
                    device=batch.token_ids.device,
                )
                chunk_seq_lens[chunk_row] = chunk_end

            prefill_result = self._executor.run_prefill(
                runtime_model,
                PrefillBatch(
                    request_ids=[batch.request_ids[row] for row in selected_rows],
                    token_ids=chunk_token_ids,
                    input_embeddings=chunk_embeddings,
                    seq_lens=chunk_seq_lens,
                    allow_device_greedy_sampling=(
                        batch.allow_device_greedy_sampling
                        and any(
                            computed_tokens[row] + chunk_len == prompt_lengths[row]
                            for row, chunk_len in zip(
                                selected_rows,
                                chunk_lengths,
                                strict=True,
                            )
                        )
                    ),
                    kv_allocations=[batch.kv_allocations[row] for row in selected_rows],
                    positions=chunk_positions,
                    block_ids=(
                        [batch.block_ids[row] for row in selected_rows]
                        if batch.block_ids
                        else []
                    ),
                ),
            )

            for chunk_row, (request_row, chunk_len) in enumerate(
                zip(selected_rows, chunk_lengths, strict=True)
            ):
                computed_tokens[request_row] += chunk_len
                if computed_tokens[request_row] != prompt_lengths[request_row]:
                    continue
                final_logits[request_row] = self._select_batch_row(
                    prefill_result.logits,
                    chunk_row,
                ).clone()
                sampled_ids = (
                    prefill_result.sampled_token_ids
                    if batch.allow_device_greedy_sampling
                    else None
                )
                if sampled_ids is not None:
                    if sampled_ids.dim() == 0:
                        sampled_id = sampled_ids
                    elif sampled_ids.dim() == 1:
                        sampled_id = sampled_ids[chunk_row]
                    else:
                        sampled_id = sampled_ids[chunk_row].reshape(-1)[0]
                    final_sampled_ids[request_row] = sampled_id.clone()

            next_request_idx = (selected_rows[-1] + 1) % row_count

        if any(logits is None for logits in final_logits):
            raise RuntimeError("prefill did not produce final logits for every request")
        logits = torch.stack([row for row in final_logits if row is not None])
        sampled_ids = None
        if all(sampled_id is not None for sampled_id in final_sampled_ids):
            sampled_ids = torch.stack(
                [sampled_id for sampled_id in final_sampled_ids if sampled_id is not None]
            )
        return logits, sampled_ids

    def _sample_batch_rows(
        self,
        logits: torch.Tensor | None,
        sampling_params,
        row_count: int,
        sampled_token_ids: torch.Tensor | None = None,
    ) -> list[int]:
        """Return sampled token IDs, preferring executor-provided device samples."""
        if sampled_token_ids is not None:
            flat_ids = sampled_token_ids.view(-1)
            if flat_ids.numel() < row_count:
                raise ValueError(
                    f"sampled_token_ids has {flat_ids.numel()} rows, expected at least {row_count}"
                )
            return [int(flat_ids[idx].item()) for idx in range(row_count)]
        return [
            self._sampler.sample(
                self._select_batch_row(logits, row_idx),
                sampling_params,
            )
            for row_idx in range(row_count)
        ]

    def _decode_embeddings_from_cache_or_lookup(
        self,
        runtime_model,
        decode_token_tensor: torch.Tensor,
    ) -> torch.Tensor | None:
        """Build decode hidden states only when the executor consumes them."""
        if self._executor.supports_device_embedding:
            return None
        return self._executor.lookup_embeddings(runtime_model, decode_token_tensor)

    @staticmethod
    def _select_batch_row(tensor: torch.Tensor, row_idx: int) -> torch.Tensor:
        """Return row ``row_idx`` from a batch tensor or the tensor itself."""
        return tensor[row_idx] if tensor.dim() > 1 else tensor

    @staticmethod
    def _should_stop(
        record: ModelRecord,
        config: GenerateConfig,
        generated: list[int],
        emitted_text: str,
        current_token: int,
    ) -> bool:
        """Return whether generation should stop for one request."""
        if record.config.eos_token_id is not None and current_token == record.config.eos_token_id:
            return True
        if len(generated) >= config.max_new_tokens:
            return True
        return any(stop and emitted_text.endswith(stop) for stop in config.stop)
