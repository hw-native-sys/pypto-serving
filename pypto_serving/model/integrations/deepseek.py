# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

from pypto_serving.config.launch_options import LaunchOptions
from pypto_serving.config.parallel import ParallelConfig
from pypto_serving.config.types import PREFILL_CHUNK_SIZE_CHOICES
from pypto_serving.model.integrations.base import IntegrationPlan, ModelIntegration, Speculation
from pypto_serving.serving.sampling.capabilities import ModelCapabilities, SamplingCapabilities


class DeepSeekV4Integration(ModelIntegration):
    integration_id = "deepseek_v4"

    def resolve(
        self, options: LaunchOptions, metadata: dict[str, object],
        devices: tuple[int, ...], speculation: Speculation,
    ) -> IntegrationPlan:
        from pypto_serving.model.deepseek.npu_executor import load_deepseek_v4_serving_contract
        from pypto_serving.model.deepseek.npu_runner import (
            build_deepseek_v4_cache_group_specs, deepseek_v4_decode_layout,
        )

        if speculation.method not in ("", "mtp"):
            raise ValueError("DeepSeek V4 --speculative-config requires method='mtp' or method='dspark'")
        if speculation.method == "mtp":
            if not speculation.explicit_tokens:
                raise ValueError("DeepSeek V4 method='mtp' requires num_speculative_tokens")
            if speculation.num_tokens <= 0:
                raise ValueError("DeepSeek V4 method='mtp' num_speculative_tokens must be positive")
        chunk = options.scheduler.long_prefill_token_threshold
        if chunk not in PREFILL_CHUNK_SIZE_CHOICES:
            choices = ", ".join(str(size) for size in PREFILL_CHUNK_SIZE_CHOICES)
            raise ValueError(f"--long-prefill-token-threshold must be one of ({choices}) for DeepSeek V4, got {chunk}")
        quantization = metadata.get("quantization_config") or {}
        if not isinstance(quantization, dict) or quantization.get("quant_method") != "compressed-tensors":
            raise ValueError("DeepSeekV4 serving requires the quantized W8A8 compressed-tensors checkpoint")
        placement = ParallelConfig(
            data_parallel_size=options.parallel.data_parallel_size,
            tensor_parallel_size=options.parallel.tensor_parallel_size,
            expert_parallel_size=options.parallel.expert_parallel_size,
            enable_expert_parallel=options.parallel.expert_parallel_size > 1,
            devices=devices,
            data_parallel_routing=options.parallel.data_parallel_routing,
            placement_mode="overlapped",
        )
        if (placement.data_parallel_size, placement.expert_parallel_size,
            placement.tensor_parallel_size, len(devices)) != (8, 8, 1, 8):
            raise ValueError("DeepSeekV4 serving requires --dp 8 --ep 8 with --tp 1 and exactly 8 NPU device ids")
        if options.scheduler.block_size != 128:
            raise ValueError("DeepSeekV4 kernels require --block-size 128")
        layout = deepseek_v4_decode_layout(speculation.num_tokens)
        max_global_batch = layout.ranks * layout.decode_batch
        if options.scheduler.max_num_seqs > max_global_batch:
            raise ValueError(f"DeepSeekV4 decode kernels support at most --max-num-seqs {max_global_batch}")
        max_model_len = layout.decode_csa_state_table_blocks * layout.c4_state_block_size
        if options.scheduler.max_model_len > max_model_len:
            raise ValueError(f"DeepSeekV4 pypto-lib decode CSA state tables support at most --max-model-len {max_model_len}")

        compress_ratios = metadata.get("compress_ratios")
        if not isinstance(compress_ratios, list):
            compress_ratios = None
        cache_groups = build_deepseek_v4_cache_group_specs(
            int(metadata.get("num_hidden_layers", 43)), compress_ratios,
            decode_batch=layout.decode_batch,
            enable_mtp=speculation.num_tokens == 1,
            max_seq_len=options.scheduler.max_model_len,
        )
        contract = load_deepseek_v4_serving_contract()
        runtime_values = {
            "num_speculative_tokens": speculation.num_tokens,
            "kv_cache_groups": cache_groups,
            "max_prefill_tokens_per_request": int(contract.max_prefill_tokens_per_request),
            "prefill_chunk_size_choices": PREFILL_CHUNK_SIZE_CHOICES,
            "requires_homogeneous_prefill_decode": bool(contract.requires_homogeneous_prefill_decode),
        }
        return IntegrationPlan(
            self.integration_id, placement, runtime_values,
            {"compile_kernels": True, "num_speculative_tokens": speculation.num_tokens},
            enable_prefix_cache=options.scheduler.enable_prefix_caching and speculation.num_tokens <= 1,
            capabilities=ModelCapabilities(sampling=SamplingCapabilities(
                honor_min_tokens=speculation.num_tokens == 0,
            )),
        )

    def executor_class(self) -> type:
        from pypto_serving.model.deepseek.npu_executor import DeepSeekV4PyptoExecutor

        return DeepSeekV4PyptoExecutor
