# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Synchronous chunk prefill and T1 decode over shared resident TP8 owners."""

from dataclasses import replace
import importlib
import json
import logging
from math import ceil, prod
import time

import torch

from pypto_serving.config.types import DecodeBatch, DecodeResult, PrefillBatch, PrefillResult
from pypto_serving.model.common.runner.model_runner import ModelRunner

from .profiling import operation
from .bindings import (
    DTYPES,
    HOST_PARAMETERS,
    TensorArena,
    borrow_view,
    frame_shape,
    parameter_rows,
)
from .cache import LowlatCachePool, TENSOR_GROUPS
from .state import LowlatRequestState
from .transfers import DecodeTransferSlots
from .prefill_transfers import PrefillTransferSlots
from .weight_loader import LowlatWeightStore
from .weight_spec import WEIGHT_NAMES


logger = logging.getLogger(__name__)
_PREFIX = "models.deepseek_v4_flash_lowlat"


class LowlatModelRunner(ModelRunner):
    def __init__(self, programs, contract, *, cache_initialization="on-demand"):
        super().__init__()
        self.programs, self.contract = programs, contract
        self.worker = self.arena = self.cache = self.weights = None
        self.state = None
        self.generation = 0
        self.request_contexts = {}
        self.cache_publications = []
        self.prefix_copies = None
        self.prefix_namespace = ""
        self.decode_bindings = {}
        self.prefill_bindings = {}
        self.prefill_input = None
        self.prefill_weights = {}
        self.prefill_transfers = None
        self.run_config = None
        self._closed = False
        self.transfers = None
        if cache_initialization not in ("on-demand", "request"):
            raise ValueError("unknown lowlat cache initialization policy")
        self.cache_initialization = cache_initialization

    def init_kv_cache(self, model_id, config, runtime, *, num_pages=None):
        if num_pages is not None or runtime.total_kv_pages is not None:
            raise ValueError("lowlat uses a complete grouped B1 pool, not total_kv_pages")
        base = self.programs.run_config
        self.chunk_capacity = (
            ceil(min(runtime.max_prefill_tokens_per_request or 128, runtime.max_num_batched_tokens, 8192) / 8)
            * 8
        )
        self.host_slots = {"num_tokens_per_owner": torch.ones((8,), dtype=torch.int32).share_memory_()}
        with self.programs.context():
            shared_input = importlib.import_module(_PREFIX + ".prefill_host_inputs").SharedInput
            self.prefill_input = shared_input(self.chunk_capacity)
        self.transfers = DecodeTransferSlots(self.programs.single_metadata)
        self.prefill_transfers = PrefillTransferSlots(self.programs, self.chunk_capacity)
        self.cache = LowlatCachePool(self.programs, None, runtime.kv_cache_groups)
        cache_hosts = self.cache.prepare_host_templates(self.chunk_capacity)
        self.prefix_namespace = getattr(self.programs,"prefix_namespace", "")
        self.cache.preserve_contents = bool(self.prefix_namespace)
        if self.prefix_namespace:
            from .prefix import PrefixPageCopies

            self.prefix_copies = PrefixPageCopies()
        additional_hosts = self._prepare_additional_hosts()
        if self.prefix_copies is not None:
            additional_hosts = (*additional_hosts,self.prefix_copies.host)
        self.run_config = replace(
            base,
            codegen_only=False,
            ring_heap=runtime.ring_heap,
            ring_task_window=runtime.ring_task_window,
            ring_dep_pool=runtime.ring_dep_pool,
        )
        with self._worker_factory_context():
            self.worker = self.programs.single.compiled.prepare(
                self.run_config,
                extra_compiled=[self.programs.prefill.compiled, *self._additional_compiled()],
                persistent=True,
                reset_persistent_windows=False,
                inherited_host_tensors=[*self.host_slots.values(), self.prefill_input.storage,
                                        *self.transfers.inherited, *cache_hosts,
                                        *self.prefill_transfers.inherited,
                                        *additional_hosts],
            )
        self.arena = TensorArena(self.worker)
        self.cache.arena = self.arena
        self.weights = LowlatWeightStore(self.programs.model, self.programs, self.arena)
        frames = sum(
            prod(frame_shape(name, row["shape"], self.chunk_capacity)[1:])
            * torch.empty((), dtype=DTYPES[row["dtype"]]).element_size()
            for profile, metadata in (("single", self.programs.single_metadata),
                                      ("prefill", self.programs.prefill_metadata))
            for name, row in parameter_rows(metadata).items()
            if name not in WEIGHT_NAMES
            and name not in TENSOR_GROUPS
            and name not in HOST_PARAMETERS
            and not (profile == "prefill" and name in ("wo_a", "wo_b"))
            and not (profile == "prefill" and name in self.prefill_transfers.names)
            and row["shape"] is not None
        )
        frames += self._additional_bytes_per_rank()
        frames += self.prefill_transfers.bytes_per_rank
        self._check_budget(
            runtime, self.weights.planned_bytes_per_rank() + self.cache.bytes_per_rank() + frames
        )
        # Reserve the complete fixed B1 cache before the lengthy checkpoint
        # upload, then verify the remaining weight/scratch budget against the
        # actual allocations. Admission must not leave cache capacity unclaimed.
        self.cache.allocate()
        self._check_budget(runtime, self.weights.planned_bytes_per_rank() + frames)
        self.weights.load()
        self.prefill_weights = self.weights.prefill_bindings()
        self._allocate_frames(runtime)
        self.transfers.bind(
            self.worker, self.arena, self.decode_bindings,
            {**self.weights.owners, **self.cache.owners, **self.decode_bindings},
            self.programs.single_metadata, self.programs.single.compiled,
        )
        self._allocate_additional_frames()
        self._check_budget(runtime, 0)
        return self.cache.counts["ori"]

    def register_contexts(self, contexts):
        from pypto_serving.config.types import RequestExecutionContext

        if len(contexts)>1 or self.state is not None:
            raise ValueError("Lowlat accepts one new request context while idle")
        for context in contexts:
            if (not isinstance(context,RequestExecutionContext) or context.initial_output_tokens
                    or context.request_id in self.request_contexts):
                raise ValueError("Lowlat requires a fresh admitted request context")
            self.request_contexts[context.request_id]=context

    def _worker_factory_context(self):
        from contextlib import nullcontext

        return nullcontext()

    def _prepare_additional_hosts(self):
        return ()

    def _additional_compiled(self):
        return ()

    def _additional_bytes_per_rank(self):
        return 0

    def _allocate_additional_frames(self):
        pass

    def _on_request_started(self, generation):
        pass

    def _after_prefill(self, batch, result):
        pass

    def _preflight_input_extent(self):
        return 2

    def _check_budget(self, runtime, additional):
        snapshots = []
        for rank in range(8):
            free, total = self.worker.device_memory_info(rank)
            available = int(total * runtime.npu_memory_utilization) - (total - free)
            snapshots.append({"rank": rank, "free": free, "total": total, "budget": available})
        logger.info(
            "lowlat memory budget: additional=%d, tightest=%d bytes",
            additional,
            min(item["budget"] for item in snapshots),
        )
        history = getattr(self, "memory_history", [])
        history.append({"additional_bytes_required": additional, "ranks": snapshots})
        self.memory_history = history
        (self.programs.output_dir / "lowlat-memory.json").write_text(json.dumps(history, indent=2) + "\n")
        for item in snapshots:
            if additional > item["budget"]:
                raise RuntimeError(
                    f"lowlat rank {item['rank']}: required additional {additional} bytes, budget {item['budget']}"
                )

    def _allocate_frames(self, runtime):
        for profile, metadata, destination in (
            ("single", self.programs.single_metadata, self.decode_bindings),
            ("prefill", self.programs.prefill_metadata, self.prefill_bindings),
        ):
            for name, row in parameter_rows(metadata).items():
                if (name in self.weights.owners or name in TENSOR_GROUPS or row["shape"] is None
                        or profile == "prefill" and (name in ("wo_a", "wo_b")
                            or name in self.prefill_transfers.names)):
                    continue
                if name in HOST_PARAMETERS:
                    destination[name] = self.host_slots[name]
                    continue
                shape = frame_shape(name, row["shape"], self.chunk_capacity)
                destination[name] = self.arena.allocate(tuple(shape), DTYPES[row["dtype"]])
        self.prefill_bindings.update(self.prefill_transfers.bind(self.arena))
        with self.programs.context():
            utils = importlib.import_module(_PREFIX + ".utils")
            cfg = importlib.import_module(_PREFIX + ".config").FLASH
            ropes = [utils.build_rope_tables(cfg, ratio, max_seq_len=self.programs.context_capacity) for ratio in (0, 4)]
        self.rope_cos = torch.stack([pair[0] for pair in ropes])
        self.rope_sin = torch.stack([pair[1] for pair in ropes])
        for name, host in (("freqs_cos", self.rope_cos), ("freqs_sin", self.rope_sin)):
            self.arena.copy(self.decode_bindings[name], host.unsqueeze(0).expand(8, *host.shape))
        for name, fill in (
            ("sampling_temperatures", 0.0),
            ("sampling_top_ks", 1),
            ("sampling_seeds", 0),
            ("sampling_positions", 0),
        ):
            owner = self.decode_bindings[name]
            self.arena.copy(owner, torch.full(tuple(owner.shape), fill, dtype=owner.dtype))
        owner = self.prefill_bindings["grammar_mask"]
        self.arena.copy(owner, torch.full(tuple(owner.shape), -1, dtype=owner.dtype))
        with self.programs.context():
            adapter = importlib.import_module(_PREFIX + ".prefill_fwd")
            controls = {**adapter.prepare_projection_bindings(), **adapter.prepare_causal_bindings()}
        for name, host in controls.items():
            self.arena.copy(self.prefill_bindings[name], host)
        owner = self.prefill_bindings["echo_diagnostics"]
        self.arena.copy(owner, torch.zeros(tuple(owner.shape), dtype=owner.dtype))
        # The selected prefill plan disables ECHO and causal/projection sharing.
        self.prefill_bindings.update(echo_epoch=1, echo_first_chunk=1)

    def _stage(self, destination, inputs):
        result = dict(destination)
        packed = {}
        if destination is self.prefill_bindings:
            packed = self.prefill_transfers.stage(inputs, revision=(self.state.generation,
                tuple(table._version for table in self.cache._tables.values())))
            result.update(packed)
        for name, host in inputs.items():
            if name not in destination or name in packed:
                continue
            owner = destination[name]
            value = (
                owner
                if tuple(host.shape) == tuple(owner.shape)
                else borrow_view(self.worker, owner, host.shape)
            )
            self.arena.copy(value, host)
            result[name] = value
        return result

    def _read_sample(self, owner):
        host = self.transfers.readback
        self.worker.copy_from(host.data_ptr(), owner.shards[0].data_ptr, 4, worker_id=0)
        token = int(host[0])
        if not 0 <= token < 129280:
            raise RuntimeError(f"lowlat sampled invalid token ID: {token}")
        return token

    def _request(self, request_ids, prompt_len=None):
        if self._closed or (self.transfers is not None and self.transfers.poisoned):
            raise RuntimeError("lowlat Worker is closed or has poisoned transfers")
        if len(request_ids) != 1:
            raise ValueError("lowlat M0 accepts exactly one active request")
        if self.state is None:
            if prompt_len is None:
                raise ValueError("decode cannot start without a completed prefill")
            generation = self.generation + 1
            self.cache.begin_generation(generation)
            if self.cache_initialization == "request":
                try:
                    # This executor owns every physical pool buffer exclusively;
                    # initialized pages still need a real scheduler claim.
                    self.cache.initialize_private_pool(generation)
                except Exception:
                    self.transfers.poison()
                    raise
            try:
                self._on_request_started(generation)
            except Exception:
                self.transfers.poison()
                raise
            self.generation = generation
            self.state = LowlatRequestState(request_ids[0], generation, prompt_len)
            context=self.request_contexts.get(request_ids[0])
            if context is not None and getattr(context,"prefix_resume",None) is not None:
                from .prefix import validate_resume

                try:
                    hit=validate_resume(context.prefix_resume,namespace=self.prefix_namespace,
                                        context=context,cache=self.cache)
                    self.state.commit_prefill(hit)
                except Exception:
                    self.transfers.poison()
                    raise
        if self.state.request_id != request_ids[0]:
            raise ValueError("lowlat cannot interleave active requests")
        return self.state

    @operation("lowlat.prefill", category="lowlat")
    def run_prefill(self, model, batch):
        state = self._request(batch.request_ids, batch.prompt_lens[0])
        start, length = batch.chunk_starts[0], batch.chunk_lens[0]
        state.check_prefill(start, length)
        if length > self.chunk_capacity or batch.constraint_states:
            raise ValueError("unsupported lowlat prefill chunk or constraints")
        terminal = start + length == state.prompt_len
        from .submissions import PrefillSubmissions

        from .execution_policy import decode_prefill_metadata

        metadata=decode_prefill_metadata(batch.execution_metadata.get(state.request_id,b""))
        with PrefillSubmissions(self.worker, state, self.transfers) as submissions:
            if self.prefix_namespace and metadata.page_identities:
                self.cache.adopt_page_identities(metadata.page_identities)
            tables, counts = self.cache.tables_for(batch.block_ids_by_group[0], input_end=start + length)
            context=self.request_contexts.get(state.request_id)
            resume=getattr(context,"prefix_resume",None)
            if resume is not None and start==resume.hit_tokens:
                before=self.prefix_copies.bytes_copied
                self.prefix_copies.copy(self.cache,source=resume.tables,
                    destination=batch.block_ids_by_group[0],ranges=resume.valid_ranges)
                logger.info("LOWLAT_PREFIX_RESTORE %s",json.dumps(dict(request_id=state.request_id,
                    hit_tokens=resume.hit_tokens,replay_tokens=resume.replay_tokens,
                    copy_bytes=self.prefix_copies.bytes_copied-before,mapping_version=1)))
            offset = batch.chunk_offsets[0]
            tokens = batch.token_ids[offset : offset + length].to(device="cpu", dtype=torch.int64)
            with self.programs.context():
                helper = importlib.import_module(_PREFIX + ".prefill_metadata")
                adapter = importlib.import_module(_PREFIX + ".prefill_fwd")
                inputs = helper.prepare_prefill_metadata(
                    tokens,
                    start_pos=start,
                    tables=tables,
                    caches=self.cache.owners,
                    rope_cos=self.rope_cos,
                    rope_sin=self.rope_sin,
                    max_logit_rows=self.prefill_bindings["sampled_ids"].shape[1],
                    terminal=terminal,
                    block_counts=counts,
                )
                hidden = self.prefill_input.fill(self.weights.lookup_embeddings(tokens))
                inputs["x_hc"] = hidden
                inputs["input_ids"] = inputs.pop("input_ids_full")
                padded = hidden.shape[1]
                self.prefill_input.begin_copy()
                try:
                    scratch = self._stage(self.prefill_bindings, inputs)
                finally:
                    self.prefill_input.end_copy()
                for name, shape in (
                    ("attn_stage", hidden.shape),
                    ("x_out", (8, padded, 4096)),
                    ("dspark_target_hidden", (8, padded // 8, 12288)),
                ):
                    scratch[name] = borrow_view(self.worker, self.prefill_bindings[name], shape)
                scratch["input_ids_full"] = scratch["input_ids"]
                # The chunk adapter checks this host layout but does not dispatch it.
                scratch["local_request_ids"] = inputs["local_request_ids"]
                scratch["target_hidden"] = scratch.pop("dspark_target_hidden")
                # Each name has exactly one owner category at the final ABI boundary.
                result_bindings = adapter.prepare_prefill_bindings(
                    self.worker, self.cache.owners, self.prefill_weights, {}, scratch
                )
                result = adapter.submit_prefill(
                    submissions,
                    {"chunk": self.programs.prefill.compiled},
                    {"chunk": self.programs.prefill_metadata},
                    result_bindings,
                    group_tokens=length,
                    terminal=terminal,
                    config=self.run_config,
                )
            submissions.ensure_published()
            self._after_prefill(batch, result)
            sampled = (
                torch.tensor([self._read_sample(result["sampled_ids"])], dtype=torch.int64) if terminal else None
            )
            plan=metadata.checkpoint_plan
            if plan is not None:
                from .prefix import checkpoint_proof

                plan.__post_init__()
                if (not self.prefix_namespace or plan.request_id!=state.request_id
                        or plan.generation!=context.execution_generation or plan.mapping_version!=1
                        or plan.boundary!=start+length
                        or plan.complete_logical_pages!=self.cache.checkpoint_ranges(plan.boundary)
                        or {k:tuple(v) for k,v in batch.block_ids_by_group[0].items()}!=plan.source_tables
                        or set(plan.source_pages)!=set(metadata.page_identities)):
                    raise ValueError("Checkpoint copy plan differs from the confirmed prefill")
                self.cache.adopt_page_identities(plan.destination_pages)
                self.prefix_copies.copy(self.cache,source=plan.source_tables,
                    destination=plan.destination_tables,ranges=plan.complete_logical_pages)
                self.cache_publications.append(checkpoint_proof(plan))
                logger.info("LOWLAT_PREFIX_CHECKPOINT %s",json.dumps(dict(request_id=state.request_id,
                    boundary=plan.boundary,mapping_version=1)))
            state.commit_prefill(start + length)
            return PrefillResult(last_hidden=None, logits=None, sampled_token_ids=sampled)

    def finalize_prefill(self, request_ids, sampled_token_ids, sampling_params=None):
        state = self._request(request_ids)
        if state.prefilled_tokens != state.prompt_len or len(sampled_token_ids) != 1:
            raise ValueError("lowlat finalize requires one completed prompt")
        state.last_sampled_token = int(sampled_token_ids[0])
        state.phase = "decode"

    def run_decode(self, model, batch):
        started = time.perf_counter()
        state = self._request(batch.request_ids)
        end = int(batch.seq_lens.reshape(-1)[0])
        token = int(batch.token_ids.reshape(-1)[0])
        state.check_decode(end - 1, token)
        if batch.constraint_states or any(
            p.temperature != 0 or p.forbidden_token_ids for p in batch.sampling_params
        ):
            raise ValueError("lowlat M0 supports unconstrained greedy decode only")
        self.transfers.refresh(
            {**self.weights.owners, **self.cache.owners, **self.decode_bindings},
            self.programs.single_metadata,
        )
        self.transfers.acquire(state.generation, self.programs.single.compiled)
        try:
            tables, counts = self.cache.tables_for(batch.block_ids_by_group[0], input_end=end)
            arguments = self.transfers.stage(tables, counts, token=token, end=end)
            prepared = time.perf_counter()
            handle = self.worker.submit(self.programs.single.compiled, *arguments, config=self.run_config)
            state.pending_handles.append(handle)
            handle.result()
            completed = time.perf_counter()
            sampled = self._read_sample(self.decode_bindings["sampled_ids"])
            result = DecodeResult(
                hidden_states=None, logits=None, sampled_token_ids=torch.tensor([sampled], dtype=torch.int64)
            )
        except Exception:
            self.transfers.poison()
            raise
        state.pending_handles.remove(handle)
        state.kv_committed_tokens = end
        state.last_sampled_token = sampled
        self.transfers.complete()
        captured = time.perf_counter()
        logger.info(
            "LOWLAT_DECODE_METRIC %s",
            json.dumps(
                {
                    "position": end - 1,
                    "prepare_ms": (prepared - started) * 1000,
                    "submit_wait_ms": (completed - prepared) * 1000,
                    "capture_ms": (captured - completed) * 1000,
                    "total_ms": (captured - started) * 1000,
                }
            ),
        )
        return result

    def preflight(self, record):
        # Before publishing the pools to the scheduler, reserve private warmup
        # pages from them. The next owner reinitializes those pages completely.
        ids = ["__lowlat_warmup__"]
        groups = {
            s.name: list(range(ceil(self._preflight_input_extent() / s.spec.token_capacity))) for s in record.runtime.kv_cache_groups
        }
        batch = PrefillBatch(
            ids,
            torch.zeros(1, dtype=torch.int64),
            None,
            [1],
            [1],
            [0],
            [0],
            prompt_lens=[1],
            block_ids_by_group=[groups],
        )
        result = self.run_prefill(record.runtime_model, batch)
        token = int(result.sampled_token_ids[0])
        self.finalize_prefill(ids, [token])
        self.run_decode(
            record.runtime_model,
            DecodeBatch(ids, torch.tensor([token]), None, torch.tensor([2]), block_ids_by_group=[groups]),
        )
        self.release_finished_requests(ids)
        logger.info("lowlat preflight: shared-owner prefill and decode complete")

    def release_finished_requests(self, request_ids):
        if self.state is None or self.state.request_id not in request_ids:
            for request_id in request_ids:
                getattr(self,"request_contexts",{}).pop(request_id,None)
            return
        self.state.phase = "draining"
        for handle in self.state.pending_handles:
            handle.result()
        self.state.pending_handles.clear()
        self.cache.release()
        if self.transfers is not None and not self.transfers.poisoned:
            self.transfers.invalidate_request()
        self.request_contexts.pop(self.state.request_id,None)
        self.state = None

    def close(self):
        if self._closed:
            return
        self._closed = True
        # Worker shutdown establishes completion even when a failed dispatch
        # prevents individual frees; never reuse its buffers after failure.
        try:
            if self.state is not None:
                self.release_finished_requests([self.state.request_id])
            if self.arena is not None:
                self.arena.close()
        except Exception:
            logger.exception("lowlat buffer teardown failed; shutting down its Worker")
        finally:
            if self.worker is not None:
                try:
                    self.worker.close()
                except Exception:
                    logger.exception("lowlat Worker teardown failed")

    def close_kv_cache(self):
        self.close()

    def _alloc_kv_cache_tensor(self, shape, dtype):
        raise RuntimeError("lowlat requires the grouped cache pool")

    def _free_kv_cache_tensor(self, tensor):
        raise RuntimeError("lowlat cache owners belong to its TensorArena")
