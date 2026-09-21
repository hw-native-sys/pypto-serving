# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek V4 DSpark K7 PD adapter."""

from pypto_serving.model.deepseek.transfer_layout import ComponentLayout, DSV4Registry
from pypto_serving.serving.pd.adapter import ModelRuntimeFacts
from pypto_serving.serving.pd.connector import DecodeConnector
from pypto_serving.serving.pd.contracts import ModelPDContract, TransferComponent
from pypto_serving.serving.pd.planner import ChunkTransferPlanner
from pypto_serving.serving.pd.protocol import (
    ChunkManifest,
    ContinuationMetadata,
    continuation_metadata_hash,
    make_prefix_match_spec,
)


DSV4_DSPARK_K7_CONTRACT = ModelPDContract(
    adapter_id="deepseek-v4-dspark-k7",
    version=3,
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
        TransferComponent(
            "csa_inner_state", "csa_inner_state", "csa_inner_state", True
        ),
    ),
    executor_cls="PyptoDeepSeekV4DSparkExecutor",
    prefill_speculative_tokens=0,
    decode_speculative_tokens=7,
    # ``independent`` is a protocol-level profile reserved for PC2/H8.  Do not
    # advertise it until P_hit > D_hit source backfill is implemented.
    supported_prefix_cache_modes=("disabled", "d_only"),
)


class DeepSeekV4DSparkK7Adapter:
    contract = DSV4_DSPARK_K7_CONTRACT

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
        metadata_hash = (
            continuation_metadata_hash(continuation)
            if continuation is not None
            else ""
        )
        return ChunkManifest(
            key=key,
            chunk_id=chunk.chunk_id,
            start_token=chunk.start_token,
            end_token=chunk.end_token,
            final=chunk.final,
            manifest_hash=plan.manifest_hash,
            expected_units=plan.expected_units,
            copies_by_rank=plan.copies_by_rank,
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
        if (
            continuation.temperature != 0.0
            or continuation.top_p != 1.0
            or continuation.top_k is not None
        ):
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
