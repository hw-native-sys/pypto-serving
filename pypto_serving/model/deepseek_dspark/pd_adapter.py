# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek V4 DSpark K7 PD adapter."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
import logging

from pypto_serving.config.types import DecodeBatch
from pypto_serving.model.deepseek.transfer_layout import BlockCopy, COMPONENTS, ComponentLayout, DSV4Registry
from pypto_serving.serving.pd.adapter import ModelRuntimeFacts
from pypto_serving.serving.pd.connector import DecodeConnector
from pypto_serving.serving.pd.contracts import ModelPDContract, TransferComponent
from pypto_serving.serving.pd.planner import ChunkTransferPlanner
from pypto_serving.serving.pd.protocol import (
    ChunkManifest,
    ContinuationMetadata,
    RankMapping,
    continuation_metadata_hash,
    make_prefix_match_spec,
)


DSV4_DSPARK_K7_CONTRACT = ModelPDContract(
    adapter_id="deepseek-v4-dspark-k7",
    version=6,
    model_family="deepseek_v4",
    model_variant="dspark",
    transfer_granularity="chunk-after-prefill",
    continuation_schema="deepseek-v4-dspark-k7/v2",
    components=(
        TransferComponent("ori", "ori", "ori"),
        TransferComponent("hca_cmp", "cmp_c128", "hca_cmp"),
        TransferComponent("csa_cmp", "cmp_c4", "csa_cmp"),
        TransferComponent("idx_k", "idx", "index"),
        TransferComponent("idx_scale", "idx", "index"),
        TransferComponent("hca_state", "hca_state", "hca_state", True),
        TransferComponent("csa_state", "csa_state", "csa_state", True),
        TransferComponent("csa_inner_state", "csa_inner_state", "csa_inner_state", True),
    ),
    executor_cls="PyptoDeepSeekV4DSparkExecutor",
    prefill_speculative_tokens=0,
    decode_speculative_tokens=7,
    supported_prefix_cache_modes=("disabled", "d_only", "independent"),
    prefill_async_scheduling=False,
    decode_async_scheduling=True,
)


class DeepSeekV4DSparkK7Adapter:
    contract = DSV4_DSPARK_K7_CONTRACT

    @staticmethod
    def rank_mapping(
        topology: tuple[int, ...], source_partition: int, destination_partition: int,
    ) -> tuple[RankMapping, ...]:
        if len(topology) != 2:
            raise ValueError("DSpark mapping requires rank count and TP group size")
        ranks, tp_size = topology
        if ranks <= 0 or tp_size <= 0 or ranks % tp_size:
            raise ValueError("invalid DSpark mapping topology")
        for partition in (source_partition, destination_partition):
            if type(partition) is not int or not 0 <= partition < ranks // tp_size:
                raise ValueError("cache partition is outside the DSpark topology")
        # DSpark replicates cache within a TP group. Preserve the matching
        # slot on each side without requiring equal group/rank numbers.
        return tuple(
            RankMapping(source_partition * tp_size + slot, destination_partition * tp_size + slot)
            for slot in range(tp_size)
        )

    def worker_factory(self, config):
        return partial(DSparkPDWorker, config=config)

    def matches(self, facts: ModelRuntimeFacts) -> bool:
        return (
            facts.model_family == self.contract.model_family
            and facts.model_variant == self.contract.model_variant
            and facts.num_speculative_tokens == 7
        )

    def build_registry(self, bundle) -> DSV4Registry:
        return DSV4Registry(
            model_revision=bundle.model_revision,
            topology=bundle.topology,
            components=tuple(
                ComponentLayout(
                    component_id=component.component_id,
                    dtype=component.dtype,
                    item_bytes=component.item_bytes,
                    layers=component.layers,
                    blocks_per_layer=component.blocks_per_layer,
                    block_tokens=component.block_tokens,
                    token_stride_bytes=component.token_stride_bytes,
                )
                for component in bundle.components
            ),
        )

    def make_planner(self, registry, group_specs) -> ChunkTransferPlanner:
        return ChunkTransferPlanner(registry, group_specs, self.contract)

    def make_decode_connector(
        self, cache_manager, capabilities, registry, destination_ranks
    ) -> DecodeConnector:
        return DecodeConnector(
            cache_manager,
            capabilities,
            registry,
            destination_ranks,
            contract=self.contract,
            rank_mapping=self.rank_mapping,
        )

    @staticmethod
    def validate_generate_config(config) -> None:
        if config.temperature != 0.0 or config.top_p != 1.0 or config.top_k is not None:
            raise ValueError("DeepSeek V4 DSpark PD supports greedy sampling only")

    @staticmethod
    def build_continuation(
        *,
        config,
        prompt_token_ids,
        eos_token_id: int | None,
        output_parser_spec=None,
    ) -> ContinuationMetadata:
        return ContinuationMetadata(
            prompt_token_ids=tuple(int(token) for token in prompt_token_ids),
            max_new_tokens=int(config.max_new_tokens),
            temperature=float(config.temperature),
            top_p=float(config.top_p),
            top_k=config.top_k,
            seed=config.seed,
            stop_strings=tuple(config.stop) if config.stop else (),
            eos_token_id=None if config.ignore_eos else eos_token_id,
            stream=bool(getattr(config, "stream", True)),
            output_parser_spec=output_parser_spec,
        )

    def build_prefix_match_spec(self, prompt_token_ids, cache_manager):
        hashes = cache_manager.compute_group_block_hashes(
            list(prompt_token_ids),
            group_names=self.contract.prefix_cache_groups,
        )
        return make_prefix_match_spec(
            token_count=len(prompt_token_ids),
            alignment=cache_manager.group_prefix_cache_alignment,
            contract_digest=self.contract.digest,
            group_block_hashes=hashes,
        )

    @staticmethod
    def build_manifest(
        *,
        key,
        plan,
        chunk,
        continuation: ContinuationMetadata | None,
        prepared_digest: str,
    ) -> ChunkManifest:
        metadata_hash = continuation_metadata_hash(continuation) if continuation is not None else ""
        return ChunkManifest(
            key=key,
            chunk_id=chunk.chunk_id,
            start_token=plan.start_token,
            end_token=plan.end_token,
            final=plan.final,
            manifest_hash=plan.manifest_hash,
            rank_mapping=plan.rank_mapping,
            expected_units=plan.expected_units,
            copies_by_destination_rank=plan.copies_by_destination_rank,
            source_prefix_hit_tokens=plan.source_prefix_hit_tokens,
            first_token=chunk.first_token,
            metadata_hash=metadata_hash,
            continuation=continuation,
            prepared_digest=prepared_digest,
        )

    @staticmethod
    def validate_continuation(continuation: ContinuationMetadata) -> None:
        if not continuation.prompt_token_ids:
            raise ValueError("DeepSeek V4 DSpark continuation requires prompt tokens")
        if continuation.max_new_tokens <= 0:
            raise ValueError("DeepSeek V4 DSpark continuation requires Decode tokens")
        if continuation.temperature != 0.0 or continuation.top_p != 1.0 or continuation.top_k is not None:
            raise ValueError("DeepSeek V4 DSpark continuation must use greedy sampling")

    @staticmethod
    def adopt_decode(
        core,
        *,
        reservation_id: str,
        request_id: str,
        first_token: int,
        continuation: ContinuationMetadata,
    ):
        return core.add_adopted_handoff(
            reservation_id=reservation_id,
            request_id=request_id,
            prompt_token_ids=continuation.prompt_token_ids,
            first_token=first_token,
            max_new_tokens=continuation.max_new_tokens,
            temperature=continuation.temperature,
            top_p=continuation.top_p,
            top_k=continuation.top_k,
            seed=continuation.seed,
            stop_strings=continuation.stop_strings,
            eos_token_id=continuation.eos_token_id,
            stream=continuation.stream,
            output_parser_spec=continuation.output_parser_spec,
        )


DSV4_DSPARK_K7_ADAPTER = DeepSeekV4DSparkK7Adapter()
BUILTIN_PD_ADAPTERS = (DSV4_DSPARK_K7_ADAPTER,)


logger = logging.getLogger(__name__)


class DSparkPDWorker:
    """Per-worker model semantics using explicit resident-cache/state capabilities."""

    rank_mapping = staticmethod(DeepSeekV4DSparkK7Adapter.rank_mapping)

    def __init__(
        self,
        *,
        config,
        compiled,
        worker,
        cache,
        draft_states,
        reserve_state,
        initialize_device_state,
        metrics,
    ):
        from pypto_serving.serving.pd.worker import PDWorkerRuntime

        self.compiled = compiled
        self.ranks = compiled.layout.ranks
        self.speculative = compiled.num_speculative_tokens > 0
        self.worker = worker
        self.cache = cache
        self.draft_states = draft_states
        self.reserve_state = reserve_state
        self.initialize_device_state = initialize_device_state
        self.metrics = metrics
        self.component_ids = tuple(COMPONENTS)
        self.transfer = PDWorkerRuntime(config, self)

    def regions(self):
        cache = self.cache()
        return {name: cache[tensor] for name, (tensor, _, _) in COMPONENTS.items()}

    @staticmethod
    def copies(records):
        return tuple(
            BlockCopy(c.component_id, c.layer, c.source_block, c.destination_block, c.valid_tokens)
            for c in records
        )

    def worker_options(self):
        return self.transfer.worker_options()

    def worker_ready(self):
        self.transfer.worker_ready()

    def handle_command(self, operation, payload):
        return self.transfer.handle_pd_command(operation, payload)

    def set_profile_active(self, active):
        self.transfer.set_transfer_profile_active(active)

    def close(self):
        self.transfer.close()

    def wrap_prepare(self, prepare):
        def build(batch, *, buffer_slot):
            bootstrap = tuple(
                request_id for request_id in batch.initial_request_ids if request_id not in self.draft_states
            )
            self.initialize_drafter(batch)
            return replace(prepare(batch, buffer_slot=buffer_slot), bootstrap_request_ids=bootstrap)

        return build

    def wrap_dispatch(self, dispatch):
        def submit(batch, inputs):
            self.finalize_device(batch)
            return dispatch(batch, inputs)

        return submit

    def registry(self, rank: int, *, model_revision: str):
        """Describe the eight resident cache regions without exporting addresses."""
        from pypto_serving.model.deepseek.transfer_layout import DSV4Registry  # noqa: PLC0415

        cache = self.cache()
        if cache is None:
            raise RuntimeError("resident cache must exist before exporting its transfer layout")
        all_layers = tuple(layer.layer_id for layer in self.compiled.layer_plan)
        csa_layers = tuple(layer.layer_id for layer in self.compiled.layer_plan if layer.compress_ratio == 4)
        hca_layers = tuple(
            layer.layer_id for layer in self.compiled.layer_plan if layer.compress_ratio == 128
        )
        return DSV4Registry.from_device_cache(
            cache,
            rank=rank,
            model_revision=model_revision,
            # The second dimension is the replicated-cache TP group.  A
            # reservation selects one group, so only those four rank owners
            # participate in a request transfer even though all 16 owners are
            # registered at process startup.
            topology=(self.compiled.layout.ranks, self.compiled.layout.tp_size),
            layer_mapping={
                "ori": all_layers,
                "hca_cmp": hca_layers,
                "csa_cmp": csa_layers,
                "idx_k": csa_layers,
                "idx_scale": csa_layers,
                "hca_state": hca_layers,
                "csa_state": csa_layers,
                "csa_inner_state": csa_layers,
            },
        )

    def initialize_drafter(
        self,
        batch: DecodeBatch,
        assignment=None,
    ) -> None:
        """Create Decode-local K7 state for committed target-cache handoffs.

        Prefill-local prompt-tail/drafter state is deliberately not transferred.
        The first D target step therefore runs without proposals and bootstraps
        the drafter from its verified hidden row.  Normal K7 requests still
        require ``finalize_prefill`` to have seeded their state.
        """
        if not self.speculative or not batch.initial_request_ids:
            return
        if not set(batch.initial_request_ids) <= set(batch.request_ids):
            raise ValueError("PD adoption names a request outside the Decode batch")
        groups = (
            assignment.groups
            if assignment is not None
            else tuple(int(group) for group in batch.cache_partitions)
        )
        if len(groups) != len(batch.request_ids):
            raise ValueError("PD adoption requires one cache partition per Decode request")
        lengths = batch.seq_lens[: len(batch.request_ids)].detach().cpu().tolist()
        for index, request_id in enumerate(batch.request_ids):
            if request_id not in batch.initial_request_ids:
                continue
            group = groups[index]
            state = self.draft_states.get(request_id)
            if state is None:
                seq_len = int(lengths[index])
                if seq_len < 1:
                    raise ValueError("PD-adopted Decode sequence length must be positive")
                prompt_len = seq_len - 1
                state = self.reserve_state(
                    request_id,
                    group=group,
                    prompt_len=prompt_len,
                    defer_device_clear=True,
                )
                state.committed_count = prompt_len
                logger.info(
                    "Initialized K7 drafter state for PD-adopted request %s (group=%d, prompt_len=%d)",
                    request_id,
                    group,
                    prompt_len,
                )
            elif state.group != group:
                raise RuntimeError(
                    f"PD-adopted request {request_id!r} changed cache partition from {state.group} to {group}"
                )

    def finalize_device(self, batch: DecodeBatch) -> None:
        """Publish D-local K7 state after the real first token is late-bound.

        Async preparation deliberately builds Decode plans with placeholder
        tokens.  A remotely prefetched request therefore reserves its local
        drafter lease during early preparation, but must wait until execution
        binds the real P-sampled token before publishing persistent device
        state.  The initial state carries no proposals: the first fused target
        step is the correctness anchor and produces the next K7 window.
        """
        if not self.speculative or not batch.initial_request_ids:
            return
        if not set(batch.initial_request_ids) <= set(batch.request_ids):
            raise ValueError("PD adoption names a request outside the Decode batch")
        if batch.token_ids.shape[0] < len(batch.request_ids):
            raise ValueError("PD adoption requires one late-bound token per request")
        for index, request_id in enumerate(batch.request_ids):
            if request_id not in batch.initial_request_ids:
                continue
            state = self.draft_states[batch.request_ids[index]]
            if state.device_state_initialized:
                continue
            state.current_token_id = int(batch.token_ids[index].reshape(-1)[-1].item())
            self.initialize_device_state(state)
