# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Batched GLM chunked prefill and decode with request-local resident caches."""

from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import threading
from typing import TYPE_CHECKING

import torch
from pypto.runtime import StackedDeviceTensor

from pypto_serving.config.types import DecodeResult, PrefillResult
from pypto_serving.model.common.runner.buffer_set import shared_empty
from pypto_serving.model.common.runner.l3_dispatch import L3DispatchMixin
from pypto_serving.model.common.runner.model_runner import ModelRunner
from pypto_serving.model.glm53.kernel_api import build_glm53_cache_group_specs, validate_serving_limits
from pypto_serving.serving.memory.kv_cache import KvCacheManager
from pypto_serving.model.glm53.task_args import (
    forward_input_task_args,
    forward_task_args,
    input_task_args,
    resident_input_names,
)
from pypto_serving.model.glm53.weight_loader import load_kernel_weights

if TYPE_CHECKING:
    from pypto.runtime import RunConfig


@dataclass
class _RequestState:
    slot: int
    committed_length: int
    prompt_length: int


@dataclass(frozen=True)
class _PreparedDecode:
    request_ids: tuple[str, ...]
    seq_lens: tuple[int, ...]
    buffer_slot: int
    cache_groups: tuple[dict[str, list[int]], ...]


@dataclass
class _PendingForward:
    dispatch: object
    phase: str
    buffer_slot: int
    epoch: int
    slots: tuple[int, ...] = ()
    sequences: tuple[tuple[int, ...], ...] = ()
    states: tuple[tuple[str, _RequestState], ...] = ()
    logits: torch.Tensor | None = None
    error: BaseException | None = None
    reclaimed: bool = False
    committed: bool = False
    lock: object = field(default_factory=threading.RLock)


class Glm53ModelRunner(L3DispatchMixin, ModelRunner):
    """Own GLM weights, request caches and incremental L3 execution."""

    def __init__(
        self,
        api,
        programs,
        checkpoint,
        device_ids,
        platform,
        *,
        host_num_threads=None,
        enable_mtp=False,
        prepared_weights=None,
    ):
        if host_num_threads is not None and (type(host_num_threads) is not int or host_num_threads <= 0):
            raise ValueError("host_num_threads must be a positive integer or None")
        super().__init__()
        self._init_l3_dispatch(stacked=True)
        self._api, self._programs, self._checkpoint = api, programs, checkpoint
        self._prepared_weights = prepared_weights
        self._device_ids, self._platform = list(device_ids), platform
        self._execution_options = dict(
            host_num_threads=host_num_threads,
            enable_mtp=enable_mtp,
        )
        self.runtime = None
        self._close_error: BaseException | None = None
        self._input_args = None
        self._forward_args = {}
        self._forward_input_args = {}
        self._input_args_by_slot = []
        self._pending_forwards = {}
        self._state_lock = threading.RLock()
        self._previous_host_num_threads = None
        self._requests: dict[str, _RequestState] = {}
        self._free_slots: list[int] = []
        self._prefix_seed_slots = set()

    def _shared_l3_worker(self):
        if self._close_error is not None:
            raise RuntimeError("GLM cleanup failed; retry close before using the runner") from self._close_error
        if self._l3_worker is None:
            raise RuntimeError("open the GLM runner before dispatching programs")
        return self._l3_worker

    def init_kv_cache(self, model_id, config, runtime, *, num_pages=None):
        """Report the physical attention pool used by the grouped allocator."""
        validate_serving_limits(runtime.max_batch_size, runtime.max_seq_len, runtime.max_num_batched_tokens)
        self._max_batch_size = runtime.max_batch_size
        self._max_seq_len = runtime.max_seq_len
        self._max_num_batched_tokens = runtime.max_num_batched_tokens
        self._free_slots = list(range(self._max_batch_size))
        groups = build_glm53_cache_group_specs(
            runtime.max_seq_len, runtime.max_num_batched_tokens, tp_size=len(self._device_ids),
        )
        self._cache_group_num_blocks = {group.name: group.num_blocks for group in groups}
        return groups[0].num_blocks

    def preflight(self, record):
        from pypto.ir import DistributedConfig
        from pypto.runtime import RunConfig

        options = {
            name: getattr(record.runtime, name) for name in ("ring_heap", "ring_task_window", "ring_dep_pool")
        }
        if options["ring_heap"] is None:
            options["ring_heap"] = 2 * 1024**3
        config = RunConfig(
            platform=self._platform,
            distributed_config=DistributedConfig(device_ids=self._device_ids, num_sub_workers=0),
            **options,
        )
        self.open(config)

    def _alloc_kv_cache_tensor(self, shape, dtype):
        return self._shared_l3_worker().alloc_tensor(shape, dtype, worker_id=0)

    def _free_kv_cache_tensor(self, tensor):
        self._shared_l3_worker().free_tensor(tensor)

    def _check_request_ids(self, request_ids):
        if not request_ids or len(request_ids) > self._max_batch_size:
            raise ValueError("GLM batch request count exceeds max_num_seqs or is empty")
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("GLM batch request ids must be unique")

    def _forward_chunks(self, request_ids, chunks, states, *, cache_groups):
        if any(not tokens for tokens in chunks) or sum(map(len, chunks)) > self._max_num_batched_tokens:
            raise ValueError("GLM new-token batch is empty or exceeds max_num_batched_tokens")
        if any(
            state.committed_length + len(tokens) > self._max_seq_len
            for state, tokens in zip(states, chunks, strict=True)
        ):
            raise ValueError("GLM request context exceeds max_model_len")
        new_slots = [
            state.slot
            for request_id, state in zip(request_ids, states, strict=True)
            if request_id not in self._requests
        ]
        if new_slots:
            self.reset_requests(new_slots)
        for state in states:
            if state.slot in new_slots:
                self._seq_lens[state.slot] = state.committed_length
                if state.committed_length:
                    self._prefix_seed_slots.add(state.slot)
        logits = self.forward_chunks(
            chunks,
            [state.slot for state in states],
            [state.committed_length for state in states],
            phase="prefill", cache_groups=cache_groups,
        )
        if logits.ndim != 2 or logits.shape[0] != len(request_ids):
            raise RuntimeError("GLM chunk forward must return one logits row per request")
        for request_id, tokens, state in zip(request_ids, chunks, states, strict=True):
            state.committed_length += len(tokens)
            self._requests[request_id] = state
        self._free_slots = [slot for slot in self._free_slots if slot not in new_slots]
        return logits

    def run_prefill(self, model, batch):
        self._check_request_ids(batch.request_ids)
        count = len(batch.request_ids)
        fields = (
            batch.chunk_starts,
            batch.chunk_lens,
            batch.chunk_offsets,
            batch.seq_lens,
            batch.prompt_lens,
        )
        if any(len(values) != count for values in fields):
            raise ValueError("GLM prefill metadata must contain one entry per request")
        if len(batch.block_ids_by_group) != count:
            raise ValueError("GLM prefill needs one grouped cache table per request")
        flat_tokens = batch.token_ids.reshape(-1).tolist()
        chunks, states = [], []
        offset, next_free = 0, 0
        for index, request_id in enumerate(batch.request_ids):
            length, start = int(batch.chunk_lens[index]), int(batch.chunk_starts[index])
            prompt_length, seq_len = int(batch.prompt_lens[index]), int(batch.seq_lens[index])
            if batch.chunk_offsets[index] != offset:
                raise ValueError("GLM prefill chunks must have contiguous offsets")
            if length <= 0:
                raise ValueError("GLM prefill requires nonempty token chunks")
            if seq_len != start + length or not 0 <= start <= seq_len <= prompt_length <= self._max_seq_len:
                raise ValueError("GLM prefill sequence length or prompt length is inconsistent")
            state = self._requests.get(request_id)
            if state is None:
                if next_free >= len(self._free_slots):
                    raise ValueError("GLM request cache slots are exhausted")
                state = _RequestState(self._free_slots[next_free], start, prompt_length)
                next_free += 1
            if state.committed_length != start or state.prompt_length != prompt_length:
                raise ValueError("GLM prefill chunk does not continue the committed request position")
            chunks.append(flat_tokens[offset : offset + length])
            states.append(state)
            offset += length
        if offset != len(flat_tokens):
            raise ValueError("GLM packed prefill tokens do not match chunk lengths")
        logits = self._forward_chunks(
            batch.request_ids, chunks, states, cache_groups=batch.block_ids_by_group,
        )
        return PrefillResult(last_hidden=None, logits=logits)

    def run_decode(self, model, batch):
        return self.run_prepared_decode(model, batch, self.prepare_decode(model, batch, buffer_slot=0))

    @property
    def supports_async_decode_reclaim(self):
        return True

    def prepare_decode(self, model, batch, *, buffer_slot):
        """Freeze metadata without reading live request state or placeholder embeddings."""
        self._check_request_ids(batch.request_ids)
        lengths = tuple(int(value) for value in batch.seq_lens.reshape(-1).tolist())
        if len(lengths) != len(batch.request_ids) or buffer_slot not in (0, 1):
            raise ValueError("GLM prepared decode needs one length per request and buffer slot zero or one")
        if len(batch.block_ids_by_group) != len(batch.request_ids):
            raise ValueError("GLM decode needs one grouped cache table per request")
        groups = tuple({name: list(ids) for name, ids in tables.items()} for tables in batch.block_ids_by_group)
        return _PreparedDecode(tuple(batch.request_ids), lengths, buffer_slot, groups)

    def run_prepared_decode(self, model, batch, prepared):
        return self.reclaim_prepared_decode(self.dispatch_prepared_decode(model, batch, prepared))

    def dispatch_prepared_decode(self, model, batch, prepared):
        """Late-bind real tokens and submit one packed decode without waiting."""
        if not isinstance(prepared, _PreparedDecode):
            raise TypeError("GLM prepared decode has an unexpected type")
        if tuple(batch.request_ids) != prepared.request_ids:
            raise ValueError("GLM prepared decode request order changed")
        count = len(batch.request_ids)
        tokens = batch.token_ids.reshape(-1).tolist()
        lengths = batch.seq_lens.reshape(-1).tolist()
        if len(tokens) != count or tuple(lengths) != prepared.seq_lens:
            raise ValueError("GLM decode requires one token and sequence length per request")
        with self._state_lock:
            if prepared.buffer_slot in self._pending_forwards:
                raise RuntimeError("GLM input/output buffer is still owned by an unreclaimed forward")
            if count > self._max_num_batched_tokens:
                raise ValueError("GLM decode batch exceeds max_num_batched_tokens")
            states = []
            for request_id, length in zip(batch.request_ids, lengths, strict=True):
                state = self._requests.get(request_id)
                if state is None or state.committed_length < state.prompt_length:
                    raise ValueError("GLM decode requires a completed prefill for this request")
                if length != state.committed_length + 1:
                    raise ValueError("GLM decode sequence length does not continue the committed request position")
                states.append(state)
            sequences = [[int(token)] for token in tokens]
            inputs, slots = self._prepare_chunk_inputs(
                sequences, [state.slot for state in states], [state.committed_length for state in states],
                cache_groups=prepared.cache_groups,
            )
            try:
                pending = self._dispatch_step(inputs, phase="decode", buffer_slot=prepared.buffer_slot)
            except BaseException:
                self._invalid_slots.update(slots)
                raise
            pending.slots = tuple(slots)
            pending.sequences = tuple(tuple(tokens) for tokens in sequences)
            pending.states = tuple(zip(batch.request_ids, states, strict=True))
            return pending

    def reclaim_prepared_decode(self, pending):
        if not isinstance(pending, _PendingForward) or pending.phase != "decode":
            raise TypeError("GLM pending decode has an unexpected type")
        logits = self._reclaim_chunks(pending)
        return DecodeResult(hidden_states=None, logits=logits)

    def release_finished_requests(self, request_ids):
        released = [request_id for request_id in dict.fromkeys(request_ids) if request_id in self._requests]
        if not released:
            return
        slots = [self._requests[request_id].slot for request_id in released]
        self.reset_requests(slots)
        for request_id in released:
            self._requests.pop(request_id)
        self._free_slots = sorted(self._free_slots + slots)

    def _initialize_execution(
        self,
        checkpoint,
        programs: dict,
        device_ids: list[int],
        *,
        platform: str = "a2a3",
        run_config: RunConfig | None = None,
        host_num_threads: int | None = None,
        enable_mtp: bool = False,
    ):
        api = self._api
        if host_num_threads is not None and (type(host_num_threads) is not int or host_num_threads <= 0):
            raise ValueError("host_num_threads must be a positive integer or None")
        self._host_num_threads = host_num_threads
        self._previous_host_num_threads = None
        self.checkpoint = checkpoint
        self.auxiliary_programs = programs.get("_auxiliary", {})
        self.programs = {name: value for name, value in programs.items() if not name.startswith("_")}
        programs = self.programs
        self.enable_mtp = bool(enable_mtp)
        self._mtp_lengths = torch.zeros(api.BATCH, dtype=torch.int32)
        self._mtp_ready = set()
        self._mtp_logits = torch.empty((api.BATCH, api.FLASH.vocab_size), dtype=torch.float32)
        self._last_chunk_sequences = None
        self._last_chunk_slots = None
        self.mtp_stats = {"proposed": 0, "accepted": 0, "rejected": 0, "rounds": 0}
        self.device_ids = list(device_ids)
        self.platform = platform
        self.run_config = run_config
        self.runtime = None
        self._device_lm_head = None
        self._device_mtp_head = None
        self._device_final_norm = None
        self._device_logits = None
        self._host_logits_shards = None
        self._mtp_prepare_program = None
        self._mtp_head_program = None
        self.calls = 0
        self.guard_events: list[str] = []
        self._seq_lens = torch.zeros(api.BATCH, dtype=torch.int32)
        self.cache_pages = api.CacheTables(api.BATCH, api.MAX_SEQ_LEN)
        self._local_cache = KvCacheManager(enable_prefix_cache=False)
        self._local_cache.init_groups(build_glm53_cache_group_specs(
            api.MAX_SEQ_LEN, api.CAPACITY, tp_size=api.TP_SIZE, cache_tokens=api.CACHE_TOKENS,
        ), max_batch_size=api.BATCH)
        self._prefix_seed_slots.clear()
        self._invalid_slots: set[int] = set()
        self._reset_program = None
        self._dispatch_error: BaseException | None = None
        self._collective_dispatches = {"dsa_attn": 0, "moe_ffn": 0}
        self._next_forward_epoch = {"prefill": 0, "decode": 0}
        self.main_dispatches = 0
        self._pending_forwards = {}
        if set(programs) != {"prefill", "decode"}:
            raise ValueError("GLM requires separate packed prefill and decode programs")
        required_auxiliary = {"reset"}
        if self.enable_mtp:
            required_auxiliary.update(("mtp_prepare", "mtp_head", "dsa_attn", "moe_ffn", "head"))
        missing = required_auxiliary - self.auxiliary_programs.keys()
        if missing:
            raise ValueError(f"GLM compiled kernels lack auxiliary entries: {sorted(missing)}")
        self.arity = {
            key: api._l3_arg_arity(api.production_program_entry(key))
            for key in ("dsa_attn", "moe_ffn", "head")
        } if self.enable_mtp else {}
        if len(device_ids) != api.RANKS or len(set(device_ids)) != api.RANKS:
            raise ValueError(f"GLM needs {api.RANKS} distinct devices")
        for phase, program in programs.items():
            names = api.forward_parameter_names(phase)
            layouts = api.forward_tensor_layouts(phase)
            params = program.compiled._get_metadata()[0]
            if len(params) != len(names) + 1 or params[-1].shape is not None:
                raise ValueError(f"compiled {phase} signature differs from the packed GLM contract")
            for name, info in zip(names, params, strict=False):
                shape, _ = layouts[name]
                expected = (api.RANKS, *shape)
                actual = tuple(info.shape) if info.shape is not None else ()
                # HOST selects a whole logical shard; packed layouts belong to L2.
                if getattr(info, "layout", None) not in (None, "ND"):
                    raise ValueError(f"compiled {phase} tensor {name} must use logical HOST shard transport")
                if (name == "ws_x_hc_pair" and actual != expected
                        or len(actual) != len(expected) or actual[0] != api.RANKS
                        or any(got != -1 and got != want for got, want in zip(actual, expected, strict=False))):
                    raise ValueError(f"compiled {phase} tensor {name} differs from the GLM geometry")

    def open(self, run_config: RunConfig | None = None):
        api = self._api
        from pypto.ir import DistributedConfig
        from pypto.runtime import DistributedWorker, RunConfig

        if self._close_error is not None:
            raise RuntimeError("GLM cleanup failed; retry close before reopening the runner") from self._close_error
        if self.runtime is not None or self._l3_worker is not None:
            raise RuntimeError("the GLM runner is already open")
        self._initialize_execution(
            self._checkpoint, self._programs, self._device_ids,
            platform=self._platform, run_config=run_config, **self._execution_options,
        )
        if self.run_config is not None and self.run_config.enable_chip_swimlane:
            raise ValueError("incremental runners cannot replay two-pass swimlane dispatches")
        checkpoint_start = time.monotonic()
        weights = self._prepared_weights or load_kernel_weights(
            api, self.checkpoint, tp_size=api.TP_SIZE, ep_size=api.EP_SIZE,
        )
        config = self.run_config or RunConfig(
            platform=self.platform, ring_heap=2 * 1024**3,
            distributed_config=DistributedConfig(device_ids=self.device_ids, num_sub_workers=0),
        )
        self.runtime_started_at = time.monotonic()
        self.checkpoint_setup_seconds = self.runtime_started_at - checkpoint_start
        print("GLM_RESIDENT_BYTES_PER_RANK", weights.bytes_per_rank, flush=True)
        inputs = api.prepare_step_inputs(weights, [[] for _ in range(api.BATCH)], cache_pages=self.cache_pages)
        self._input_args_by_slot = [input_task_args(inputs)] if self.enable_mtp else []
        for args in self._input_args_by_slot:
            args.allocate_host_shared(None)
            args.stage(inputs)
        self._forward_input_args = {
            (slot, phase): forward_input_task_args(api, phase, inputs)
            for slot in range(2) for phase in ("prefill", "decode")
        }
        for args in self._forward_input_args.values():
            args.allocate_host_shared(None)
        self._input_args = self._input_args_by_slot[0] if self._input_args_by_slot else None
        self._head_rows_by_slot = [
            shared_empty((api.RANKS, api.LOGIT_ROWS, api.D), torch.bfloat16, name=f"glm_head_rows_{slot}")
            for slot in range(2)
        ]
        self._logits_by_slot = [
            shared_empty((api.RANKS, api.BATCH, api.FP32_HEAD_VOCAB_PAD), torch.float32,
                         name=f"glm_logits_{slot}")
            for slot in range(2)
        ]
        runtime = DistributedWorker(
            [entry.compiled for entry in (*self.programs.values(), *self.auxiliary_programs.values())],
            config=config, persistent=True, reset_persistent_windows=False,
        )
        self._l3_worker = runtime
        self._l3_run_config = config
        self.runtime, self.config = runtime, config
        try:
            layouts = api.forward_tensor_layouts("prefill")
            allocation_layouts = dict(api.CACHE_LAYOUTS)
            mutable_inputs = set(inputs) - set(api.CONSTANT_STEP_INPUTS) - set(api._PARAM_TO_STACK)
            resident_inputs = resident_input_names(
                api, inputs, enable_mtp=self.enable_mtp,
            )
            allocation_layouts.update({name: (tuple(value.shape), value.dtype) for name, value in inputs.items()
                                       if name in resident_inputs})
            allocation_layouts.update({api._STACK_TO_PARAM.get(name, name): value
                                       for name, value in weights.tensor_layouts.items()})
            if self.enable_mtp:
                allocation_layouts.update(api.WORKSPACES)
                allocation_layouts.update(api.OUTPUTS)
            allocation_layouts.update({name: value for name, value in layouts.items()
                                       if self.enable_mtp or name not in mutable_inputs | {"head_hidden"}})
            for name in ("final_norm", "lm_head_weight", "logits", "ws_x_hc_a", "ws_x_hc_b"):
                allocation_layouts.pop(name, None)
            per_rank = [{} for _ in self.device_ids]
            self.per_rank = per_rank
            for rank, tensors in enumerate(per_rank):
                for name, (shape, dtype) in allocation_layouts.items():
                    tensors[name] = runtime.alloc_tensor(shape, dtype, init=inputs.get(name), worker_id=rank)
                if self.enable_mtp:
                    stream_shape = (api.CAPACITY, api.HC_MULT, api.D)
                    stream_bytes = math.prod(stream_shape) * 2
                    target_side = api.NUM_LAYERS % 2
                    for name, side in (("ws_x_hc_a", target_side), ("ws_x_hc_b", 1 - target_side)):
                        tensors[name] = runtime.view_tensor(
                            tensors["ws_x_hc_pair"], stream_shape,
                            byte_offset=side * stream_bytes, worker_id=rank,
                        )

            def upload(rank):
                handles = {name: per_rank[rank][api._STACK_TO_PARAM.get(name, name)]
                           for name in weights.tensor_layouts}
                for name, data, offset in weights.iter_rank_uploads(rank):
                    runtime.copy_to(handles[name].data_ptr, data.data_ptr(), data.numel() * data.element_size(),
                                    dst_offset=offset, worker_id=rank)
                print("GLM_RANK_WEIGHTS_READY", rank, flush=True)

            with ThreadPoolExecutor(max_workers=api.RANKS) as executor:
                list(executor.map(upload, range(api.RANKS)))
            head = weights.lm_head()
            for rank, tensors in enumerate(per_rank):
                shard = torch.zeros((api.FP32_HEAD_VOCAB_PAD, api.D), dtype=torch.float32)
                shard[:api.LOCAL_VOCAB].copy_(head[rank * api.LOCAL_VOCAB:(rank + 1) * api.LOCAL_VOCAB])
                tensors["lm_head_weight"] = runtime.alloc_tensor(shard.shape, shard.dtype, init=shard,
                                                                worker_id=rank)
                tensors["final_norm"] = tensors["norm.final_norm"]
                if self.enable_mtp:
                    tensors["logits"] = runtime.alloc_tensor((api.BATCH, api.FP32_HEAD_VOCAB_PAD),
                                                            torch.float32, worker_id=rank)
                    shard.zero_()
                    shard[:api.LOCAL_VOCAB].copy_(weights.mtp_head_shard(rank))
                    tensors["mtp_head_weight"] = runtime.alloc_tensor(shard.shape, shard.dtype, init=shard,
                                                                     worker_id=rank)
            print("GLM_DEVICE_LM_HEAD_BYTES_PER_RANK", api.FP32_HEAD_VOCAB_PAD * api.D * 4, flush=True)

            def stack(tensors, name):
                return StackedDeviceTensor([rank[name] for rank in tensors],
                                           (api.RANKS, *tensors[0][name].shape), tuple(range(api.RANKS)))

            stacked = {name: stack(per_rank, name) for name in per_rank[0]}
            self.stacked = stacked
            self._reset_program = self.auxiliary_programs["reset"]
            self._mtp_prepare_program = self.auxiliary_programs.get("mtp_prepare")
            self._mtp_head_program = self.auxiliary_programs.get("mtp_head")
            self._run_l3(self._reset_program, *(stacked[name] for name in (
                "mla_latent_cache", "index_raw_cache", "index_pool_cache", "index_pool_scale",
                "kda_conv_state", "kda_recurrent_state")),
                -1, config=config)
            self._phase_tensors = []
            for slot in range(2):
                phases = {}
                for phase in ("prefill", "decode"):
                    phase_tensors = dict(self._forward_input_args[slot, phase].tensors)
                    phase_tensors["head_hidden"] = self._head_rows_by_slot[slot]
                    phase_tensors["logits"] = self._logits_by_slot[slot]
                    for name, (shape, _) in api.forward_tensor_layouts(phase).items():
                        if name in phase_tensors:
                            continue
                        shards = [runtime.view_tensor(rank[name], shape, worker_id=rank_id)
                                  for rank_id, rank in enumerate(per_rank)]
                        phase_tensors[name] = StackedDeviceTensor(
                            shards, (api.RANKS, *shape), tuple(range(api.RANKS)),
                        )
                    phases[phase] = phase_tensors
                    self._forward_args[slot, phase] = forward_task_args(
                        api, phase, phase_tensors,
                    )
                self._phase_tensors.append(phases)
            if self.enable_mtp:
                mtp_layouts = {
                    "mtp_cached_hidden": ((api.BATCH, api.D), torch.bfloat16, 0),
                    "mtp_cached_pools": ((api.BATCH, api.KPOOL_SELECT_K), torch.int32, -1),
                    "mtp_cached_valid": ((api.BATCH, api.KPOOL_SELECT_K), torch.int32, 0),
                    "mtp_selected_pools": ((api.CAPACITY, api.KPOOL_SELECT_K), torch.int32, -1),
                    "mtp_selected_valid": ((api.CAPACITY, api.KPOOL_SELECT_K), torch.int32, 0),
                }
                for name, (shape, dtype, fill) in mtp_layouts.items():
                    value = torch.full(shape, fill, dtype=dtype)
                    for rank in range(api.RANKS):
                        per_rank[rank][name] = runtime.alloc_tensor(shape, dtype, init=value, worker_id=rank)
                    stacked[name] = stack(per_rank, name)
            self._device_lm_head = stacked.get("lm_head_weight")
            self._device_final_norm = stacked.get("final_norm")
            self._device_mtp_head = stacked.get("mtp_head_weight")
            self._device_logits = stacked.get("logits")
            self._host_logits_shards = self._logits_by_slot[0]
            self.head_rows = self._head_rows_by_slot[0]
            self.weights = weights
            self._seq_lens.zero_()
            self._invalid_slots.clear()
            self._dispatch_error = None
            if self._host_num_threads is not None:
                self._previous_host_num_threads = torch.get_num_threads()
                torch.set_num_threads(self._host_num_threads)
        except BaseException:
            self._close_execution()
            raise
        if hasattr(self, "_max_batch_size"):
            self._requests.clear()
            self._free_slots = list(range(self._max_batch_size))
        return self

    def _restore_host_num_threads(self) -> None:
        previous, self._previous_host_num_threads = self._previous_host_num_threads, None
        if previous is not None:
            torch.set_num_threads(previous)

    def _require_healthy_runtime(self) -> None:
        """Require an open runtime with valid per-program communication state."""
        if self._close_error is not None:
            raise RuntimeError("GLM cleanup failed; retry close before using the runner") from self._close_error
        if self.runtime is None:
            raise RuntimeError("open the GLM runner before accessing request state")
        if self._dispatch_error is not None:
            raise RuntimeError("a dispatch failed; close and reopen the GLM runner") from self._dispatch_error

    def reset_requests(self, request_slots: list[int]) -> None:
        """Clear private working state and drop request-owned table mappings."""
        api = self._api
        self._require_healthy_runtime()
        self._drain_forwards()
        slots = list(request_slots)
        if len(set(slots)) != len(slots) or any(slot < 0 or slot >= api.BATCH for slot in slots):
            raise ValueError(f"request slots must be unique integers in 0..{api.BATCH - 1}")
        if not slots:
            return
        # Mark requested slots invalid through the distributed clear.
        self._invalid_slots.update(slots)
        slot_mask = sum(1 << slot for slot in slots)
        self._run_l3(
            self._reset_program,
            self.stacked["mla_latent_cache"],
            self.stacked["index_raw_cache"],
            self.stacked["index_pool_cache"],
            self.stacked["index_pool_scale"],
            self.stacked["kda_conv_state"],
            self.stacked["kda_recurrent_state"],
            slot_mask,
            config=self.config,
        )
        self._seq_lens[slots] = 0
        self.cache_pages.table[slots] = -1
        self._prefix_seed_slots.difference_update(slots)
        for slot in slots:
            self._local_cache.release_all_group_requests(str(slot))
        self._mtp_lengths[slots] = 0
        self._mtp_ready.difference_update(slots)
        if self.enable_mtp:
            for name, width, dtype, fill in (
                ("mtp_cached_hidden", api.D, torch.bfloat16, 0),
                ("mtp_cached_pools", api.KPOOL_SELECT_K, torch.int32, -1),
                ("mtp_cached_valid", api.KPOOL_SELECT_K, torch.int32, 0),
            ):
                row = torch.full((1, width), fill, dtype=dtype)
                for rank in range(api.RANKS):
                    for slot in slots:
                        self.runtime.copy_to(
                            self.per_rank[rank][name].data_ptr,
                            row.data_ptr(),
                            row.numel() * row.element_size(),
                            dst_offset=slot * row.numel() * row.element_size(),
                            worker_id=rank,
                        )
        self._invalid_slots.difference_update(slots)

    def _prepare_chunk_inputs(
        self, token_sequences, request_slots, kv_seq_lens=None, *, visual_features=None, cache_groups=None,
    ):
        api = self._api
        self._require_healthy_runtime()
        slots = list(request_slots)
        if len(token_sequences) != len(slots) or not slots:
            raise ValueError("each nonempty query sequence needs one request slot")
        if len(set(slots)) != len(slots) or any(slot < 0 or slot >= api.BATCH for slot in slots):
            raise ValueError(f"request slots must be unique integers in 0..{api.BATCH - 1}")
        if any(not tokens for tokens in token_sequences):
            raise ValueError("active chunk sequences must contain at least one token")
        if self._invalid_slots.intersection(slots):
            raise RuntimeError("reset request slots after a failed dispatch before reusing their cache state")
        if any(set(slots).intersection(pending.slots) for pending in self._pending_forwards.values()):
            raise RuntimeError("GLM request has an unreclaimed forward")
        previous = self._seq_lens[slots].tolist()
        if kv_seq_lens is not None and list(kv_seq_lens) != previous:
            raise ValueError("chunk prefix lengths disagree with committed request cache state")
        sequences = [[] for _ in range(api.BATCH)]
        for slot, tokens in zip(slots, token_sequences, strict=True):
            sequences[slot] = list(tokens)
        if visual_features is not None and set(visual_features) - set(slots):
            raise ValueError("visual features must belong to active request slots")
        new_lengths = self._seq_lens + torch.tensor([len(tokens) for tokens in sequences], dtype=torch.int32)
        state_rows = self._bind_cache_tables(slots, new_lengths, cache_groups)
        inputs = api.prepare_step_inputs(
            self.weights, sequences, kv_seq_lens=self._seq_lens, visual_features=visual_features,
            cache_pages=self.cache_pages, kda_state_rows=state_rows,
        )
        return inputs, slots

    def _bind_cache_tables(self, slots, new_lengths, cache_groups=None):
        """Lower scheduler pages; standalone generation uses the same allocator locally."""
        api = self._api
        if cache_groups is None:
            cache_groups = [self._local_cache.ensure_group_blocks(str(slot), int(new_lengths[slot]))
                            for slot in slots]
        state_rows = torch.full((api.BATCH, api.KDA_STATE_COLUMNS), -1, dtype=torch.int32)
        state_rows[:, :2] = torch.arange(api.BATCH, dtype=torch.int32).unsqueeze(1)
        for slot, groups in zip(slots, cache_groups, strict=True):
            if set(groups) != {"attention", "kda_state"}:
                raise ValueError("GLM cache tables must contain attention and kda_state groups")
            end = int(new_lengths[slot])
            start = int(self._seq_lens[slot])
            attention, states = groups["attention"], groups["kda_state"]
            if len(attention) != (end + api.BLOCK_SIZE - 1) // api.BLOCK_SIZE:
                raise ValueError("GLM attention table does not cover the request context")
            if len(states) != min((end + api.BLOCK_SIZE - 1) // api.BLOCK_SIZE, api.KDA_STATE_BLOCKS):
                raise ValueError("GLM state table does not cover the request's rolling checkpoints")
            self.cache_pages.table[slot].fill_(-1)
            self.cache_pages.table[slot, :len(attention)] = torch.tensor(attention, dtype=torch.int32)
            if slot in self._prefix_seed_slots:
                logical = (start - 1) // api.BLOCK_SIZE
                state_rows[slot, 0] = api.BATCH + states[logical % api.KDA_STATE_BLOCKS]
            for chunk in range((end - start + api.CHUNK_TILE - 1) // api.CHUNK_TILE):
                endpoint = min(start + (chunk + 1) * api.CHUNK_TILE, end)
                if endpoint % api.BLOCK_SIZE == 0:
                    logical = endpoint // api.BLOCK_SIZE - 1
                    state_rows[slot, chunk + 2] = api.BATCH + states[logical % api.KDA_STATE_BLOCKS]
        return state_rows

    def forward_chunks(
        self, token_sequences, request_slots, kv_seq_lens=None, *, visual_features=None, phase="prefill",
        cache_groups=None,
    ):
        """Run one complete prefill or decode and commit its device cache lengths."""
        if phase not in ("prefill", "decode"):
            raise ValueError("GLM forward phase must be prefill or decode")
        if phase == "decode" and any(len(tokens) != 1 for tokens in token_sequences):
            raise ValueError("GLM decode requires exactly one query token per active request")
        self._drain_forwards()
        inputs, slots = self._prepare_chunk_inputs(
            token_sequences, request_slots, kv_seq_lens, visual_features=visual_features, cache_groups=cache_groups,
        )
        try:
            pending = self._dispatch_step(inputs, phase=phase, buffer_slot=0)
            try:
                logits = self._reclaim_step(pending)
            finally:
                if pending.error is None:
                    self._pending_forwards.pop(pending.buffer_slot, None)
        except BaseException:
            self._invalid_slots.update(slots)
            raise
        for slot, tokens in zip(slots, token_sequences, strict=True):
            self._seq_lens[slot] += len(tokens)
        self._prefix_seed_slots.difference_update(slots)
        self._last_chunk_sequences = [list(tokens) for tokens in token_sequences]
        self._last_chunk_slots = slots
        return logits[slots]

    def update_mtp(self, next_token_ids: list[int]) -> torch.Tensor:
        """Refresh the MTP prefix with true target feedback and one shifted token per query row."""
        if not self.enable_mtp or self._last_chunk_sequences is None:
            raise RuntimeError("MTP needs an enabled runner and a completed target chunk")
        slots = self._last_chunk_slots
        sequences = self._last_chunk_sequences
        if len(next_token_ids) != len(slots):
            raise ValueError("MTP needs one next token per target request")
        expected = [int(self._seq_lens[slot]) - len(tokens) for slot, tokens in zip(slots, sequences)]
        if self._mtp_lengths[slots].tolist() != expected:
            raise RuntimeError("MTP cache must be refreshed after every target chunk")
        shifted = [tokens[1:] + [int(next_token)] for tokens, next_token in zip(sequences, next_token_ids)]
        logits = self._run_mtp(shifted, slots, recycle=False)
        self._mtp_ready.update(slots)
        self._last_chunk_sequences = None
        return logits

    def propose_mtp(self, request_slots: list[int], count: int) -> dict[int, list[int]]:
        """Draft with one MTP-owned TopK and roll speculative cache visibility back afterward."""
        slots = list(request_slots)
        if not self.enable_mtp or count not in range(1, self._api.MTP_SPEC_TOKENS + 1):
            raise ValueError(f"MTP proposal count must be in 1..{self._api.MTP_SPEC_TOKENS}")
        if len(set(slots)) != len(slots) or not slots or not set(slots) <= self._mtp_ready:
            raise ValueError("MTP proposal slots need fresh target feedback")
        if bool((self._mtp_lengths[slots] + count - 1 > self._api.MAX_SEQ_LEN).any()):
            raise ValueError("MTP proposal exceeds the context limit")
        base_lengths = self._mtp_lengths.clone()
        proposed = {slot: [] for slot in slots}
        for step in range(count):
            next_ids = self._mtp_logits[slots].argmax(dim=-1).tolist()
            for slot, token in zip(slots, next_ids):
                proposed[slot].append(int(token))
            if step + 1 < count:
                self._run_mtp([[int(token)] for token in next_ids], slots, recycle=True)
        # The target verifier consumes only confirmed input tokens. Its KDA states
        # never advance through a rejected draft. Uncommitted MTP payloads remain
        # owned but are invisible until a later true-feedback write replaces them.
        self._mtp_lengths.copy_(base_lengths)
        self._mtp_ready.difference_update(slots)
        self.mtp_stats["rounds"] += 1
        self.mtp_stats["proposed"] += count * len(slots)
        return proposed

    def _run_mtp(self, token_sequences, slots, *, recycle):
        api = self._api
        self._require_healthy_runtime()
        self._drain_forwards()
        sequences = [[] for _ in range(api.BATCH)]
        for slot, tokens in zip(slots, token_sequences):
            sequences[slot] = list(tokens)
        lengths = self._mtp_lengths + torch.tensor([len(tokens) for tokens in sequences], dtype=torch.int32)
        self._bind_cache_tables(slots, lengths)
        inputs = api.prepare_step_inputs(
            self.weights,
            sequences,
            kv_seq_lens=self._mtp_lengths,
            cache_pages=self.cache_pages,
            mtp=True,
        )
        for slot in slots:
            if int(self._mtp_lengths[slot]) == 0:
                row = int(inputs["query_start_loc"][slot])
                inputs["embedding_rows"][row].zero_()
        runtime, config, stacked = self.runtime, self.config, self.stacked
        try:
            if inputs:
                if self._input_args is None:
                    raise RuntimeError("open the GLM runner before staging input tensors")
                args = self._input_args_by_slot[0] if self._input_args_by_slot else self._input_args
                args.stage(inputs)
                inputs = args.tensors
            for rank in range(api.RANKS):
                for name, value in inputs.items():
                    runtime.copy_to(
                        self.per_rank[rank][name].data_ptr,
                        value.data_ptr(),
                        value.numel() * value.element_size(),
                        worker_id=rank,
                    )
            feedback_norm = stacked["mtp.final_norm"] if recycle else self._device_final_norm
            self._run_l3(
                self._mtp_prepare_program,
                stacked["ws_x_hc_a"],
                stacked["embedding_rows"],
                feedback_norm,
                stacked["mtp.enorm"],
                stacked["mtp.hnorm"],
                stacked["mtp.eh_proj"],
                stacked["request_ids"],
                stacked["mtp_cached_hidden"],
                stacked["mtp_cached_pools"],
                stacked["mtp_cached_valid"],
                stacked["mtp_selected_pools"],
                stacked["mtp_selected_valid"],
                stacked["ws_x_hc_b"],
                int(recycle),
                config=config,
            )
            for program, overrides in (
                ("dsa_attn", {"x_hc": "ws_x_hc_b", "out_x_hc_mid": "ws_x_hc_a"}),
                ("moe_ffn", {"x_hc_mid": "ws_x_hc_a", "out_x_hc_out": "ws_x_hc_b"}),
                ("head", {"x_hc": "ws_x_hc_b"}),
            ):
                tensor_names, scalar_names = self.arity[program]
                index = self._collective_dispatches.get(program, 0)
                epoch = index + 1
                if epoch > (2**31 - 1) // 2:
                    raise OverflowError("collective epochs exhausted; reopen the runner")
                scalars = dict(
                    bb_ord=api.NUM_LAYERS,
                    dsa_ord=api.MLA_STACK_LAYERS - 1,
                    moe_ord=api.MOE_STACK_LAYERS - 1,
                    attn_epoch=epoch,
                    moe_epoch=epoch,
                    ffn_epoch=epoch,
                    skip_topk=int(recycle),
                )
                arguments = []
                for name in tensor_names:
                    allocation = api._alloc_name_of(name, overrides)
                    if name in ("selected_pools", "selected_valid"):
                        allocation = "mtp_" + name
                    arguments.append(stacked[allocation])
                arguments.extend(scalars[name] for name in scalar_names)
                self._run_l3(self.auxiliary_programs[program], *arguments, config=config)
                if program in self._collective_dispatches:
                    self._collective_dispatches[program] = epoch
            self._run_l3(
                self._mtp_head_program,
                stacked["head_hidden"],
                stacked["mtp.final_norm"],
                self._device_mtp_head,
                stacked["mtp_selected_pools"],
                stacked["mtp_selected_valid"],
                stacked["query_start_loc"],
                stacked["mtp_cached_hidden"],
                stacked["mtp_cached_pools"],
                stacked["mtp_cached_valid"],
                self._device_logits,
                config=config,
            )
            runtime.copy_stacked_from(self._device_logits, self._host_logits_shards)
            logits = torch.cat(
                tuple((shard[:, : api.LOCAL_VOCAB] for shard in self._host_logits_shards.unbind(0))), dim=1
            )
            if not torch.isfinite(logits[slots]).all():
                raise FloatingPointError("MTP produced nonfinite logits")
        except BaseException as error:
            self._dispatch_error = error
            self._invalid_slots.update(slots)
            raise
        self._mtp_lengths.copy_(lengths)
        self._mtp_logits[slots] = logits[slots]
        return logits[slots]

    def _dispatch_step(self, inputs, *, phase, buffer_slot):
        """Stage one slot and submit exactly one complete model program."""
        api = self._api
        self._require_healthy_runtime()
        if phase not in ("prefill", "decode") or buffer_slot not in (0, 1):
            raise ValueError("GLM dispatch needs a phase and buffer slot zero or one")
        with self._state_lock:
            if buffer_slot in self._pending_forwards:
                raise RuntimeError("GLM input/output buffer is still owned by an unreclaimed forward")
            epoch = api.forward_epoch_guard(self._next_forward_epoch[phase])
            query_names = set(api.forward_query_tensor_names())
            query_capacity = api.forward_geometry(phase)["query_capacity"]
            host_inputs = self._forward_input_args[buffer_slot, phase]
            try:
                values = {}
                for name, destination in host_inputs.tensors.items():
                    value = inputs[name]
                    if name in query_names:
                        value = value[:query_capacity]
                    values[name] = value.unsqueeze(0).expand_as(destination)
                host_inputs.stage(values)
                self.last_dispatch_started_at = time.monotonic()
                print("GLM_DEVICE_START", self.main_dispatches, flush=True)
                args = self._forward_args[buffer_slot, phase].build(use_cache=True)
                dispatch = self._submit_l3(self.programs[phase], *args, epoch, config=self.config)
            except BaseException as error:
                self._poison_forwards(error)
                raise
            pending = _PendingForward(dispatch, phase, buffer_slot, epoch)
            self._pending_forwards[buffer_slot] = pending
            self._next_forward_epoch[phase] += 1
            self.main_dispatches += 1
            return pending

    def _reclaim_step(self, pending):
        """Wait for ordered host output staging and validate this dispatch slot."""
        with pending.lock:
            if pending.reclaimed:
                if pending.error is not None:
                    raise pending.error
                return pending.logits
            api = self._api
            try:
                pending.dispatch.wait()
                if self._dispatch_error is not None:
                    raise RuntimeError("a GLM forward failed; close and reopen the runner") from self._dispatch_error
                head_rows = self._head_rows_by_slot[pending.buffer_slot]
                if not bool(torch.isfinite(head_rows).all()):
                    self.guard_events.append(f"nonfinite head_hidden at step {pending.epoch}")
                    raise FloatingPointError(f"nonfinite head_hidden at step {pending.epoch}")
                shards = self._logits_by_slot[pending.buffer_slot]
                logits = torch.cat(tuple(shard[:, :api.LOCAL_VOCAB] for shard in shards.unbind(0)), dim=1)
                if not bool(torch.isfinite(logits[:api.BATCH]).all()):
                    self.guard_events.append(f"nonfinite logits at step {pending.epoch}")
                    raise FloatingPointError(f"nonfinite last-token logits at step {pending.epoch}")
                pending.logits = logits[:api.BATCH].clone()
                self.calls += 1
            except BaseException as error:
                pending.error = error
                self._poison_forwards(error)
                raise
            finally:
                pending.reclaimed = True
            return pending.logits

    def _poison_forwards(self, error):
        self._dispatch_error = error
        for pending in tuple(self._pending_forwards.values()):
            self._invalid_slots.update(pending.slots)
            try:
                pending.dispatch.wait()
            except BaseException:
                pass

    def _reclaim_chunks(self, pending):
        with pending.lock:
            logits = self._reclaim_step(pending)
            if not pending.committed:
                with self._state_lock:
                    for slot, tokens in zip(pending.slots, pending.sequences, strict=True):
                        self._seq_lens[slot] += len(tokens)
                    for (_, state), tokens in zip(pending.states, pending.sequences, strict=True):
                        state.committed_length += len(tokens)
                    self._last_chunk_sequences = [list(tokens) for tokens in pending.sequences]
                    self._last_chunk_slots = list(pending.slots)
                    self._pending_forwards.pop(pending.buffer_slot, None)
                    pending.committed = True
            return logits[list(pending.slots)]

    def _drain_forwards(self):
        self._require_healthy_runtime()
        for pending in tuple(self._pending_forwards.values()):
            self._reclaim_chunks(pending)

    def _close_execution(self):
        worker = self._l3_worker
        try:
            if worker is not None:
                worker.close()
        except BaseException as error:
            # Native close may need another attempt. Keep the worker and every
            # shared argument alive until the underlying process tree is reaped.
            self._close_error = error
            self._restore_host_num_threads()
            raise
        self._l3_worker = None
        self.runtime = None
        self._close_error = None
        self._reset_program = None
        self._device_lm_head = None
        self._device_mtp_head = None
        self._device_final_norm = None
        self._device_logits = None
        self._host_logits_shards = None
        self._pending_forwards.clear()
        self._phase_tensors = []
        self._slot_per_rank = []
        self.per_rank = []
        self.stacked = {}
        self._head_rows_by_slot = []
        self._logits_by_slot = []
        try:
            input_args = self._input_args_by_slot or ([self._input_args] if self._input_args is not None else [])
            for args in input_args:
                args.close()
            self._input_args_by_slot.clear()
            self._input_args = None
            for args in self._forward_input_args.values():
                args.close()
            self._forward_input_args.clear()
            for args in self._forward_args.values():
                args.close()
            self._forward_args.clear()
            self._reset_l3_dispatch()
        finally:
            self._restore_host_num_threads()

    def close(self):
        self._close_execution()
        self._requests.clear()
        self._free_slots.clear()
