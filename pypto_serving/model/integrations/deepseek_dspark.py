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
from pypto_serving.model.integrations.base import IntegrationPlan, ModelIntegration, Speculation
from pypto_serving.serving.sampling.capabilities import ModelCapabilities, SamplingCapabilities


class DeepSeekV4DSparkIntegration(ModelIntegration):
    integration_id = "deepseek_v4_dspark"

    def resolve(
        self, options: LaunchOptions, metadata: dict[str, object],
        devices: tuple[int, ...], speculation: Speculation,
    ) -> IntegrationPlan:
        from pypto_serving.model.deepseek_dspark.npu_runner import (
            DSPARK_DECODE_BATCH, DSPARK_MAX_SEQ_LEN, DSPARK_PREFILL_MAX_TOKENS,
            DSPARK_SLIDING_WINDOW, DSPARK_SPECULATIVE_TOKENS, DSPARK_TP_SIZE,
            build_dspark_cache_group_specs,
        )

        if speculation.method != "dspark":
            raise ValueError("DSpark serving requires method='dspark'")
        if speculation.num_tokens not in (0, DSPARK_SPECULATIVE_TOKENS):
            raise ValueError(
                "DeepSeek V4 method='dspark' speculation is fixed at "
                f"num_speculative_tokens {DSPARK_SPECULATIVE_TOKENS}; use 0 for target-only serving"
            )
        ranks = options.parallel.expert_parallel_size
        partitions = ranks // DSPARK_TP_SIZE
        if options.parallel.tensor_parallel_size != DSPARK_TP_SIZE:
            raise ValueError(f"DSpark serving requires --tp {DSPARK_TP_SIZE}")
        if ranks not in (2 * DSPARK_TP_SIZE, 4 * DSPARK_TP_SIZE):
            raise ValueError("DSpark serving requires --ep 8 or --ep 16 (TP4, EP=ranks)")
        if options.parallel.data_parallel_size != partitions:
            raise ValueError(f"DSpark serving requires --dp {partitions} with --ep {ranks}")
        if len(devices) != ranks:
            raise ValueError(f"DSpark serving requires exactly {ranks} NPU device ids")
        if options.scheduler.block_size != 32:
            raise ValueError("DSpark kernels require --block-size 32")
        max_global_batch = partitions * DSPARK_DECODE_BATCH
        if options.scheduler.max_num_seqs > max_global_batch:
            raise ValueError(f"DSpark decode kernels support at most --max-num-seqs {max_global_batch}")
        if options.scheduler.max_model_len > DSPARK_MAX_SEQ_LEN:
            raise ValueError(f"DSpark decode cache tables support at most --max-model-len {DSPARK_MAX_SEQ_LEN}")

        # The requested TP4/DP axes describe kernel layout; the worker group owns
        # all ranks as one overlapped placement with EP spanning the group.
        placement = ParallelConfig(
            data_parallel_size=1, tensor_parallel_size=1,
            expert_parallel_size=ranks, enable_expert_parallel=True,
            devices=devices, data_parallel_routing=options.parallel.data_parallel_routing,
            placement_mode="overlapped",
        )
        max_prefill = min(DSPARK_PREFILL_MAX_TOKENS, options.scheduler.max_num_batched_tokens)
        if options.scheduler.enable_chunked_prefill and options.scheduler.long_prefill_token_threshold > 0:
            max_prefill = min(max_prefill, options.scheduler.long_prefill_token_threshold)
        compress_ratios = metadata.get("compress_ratios")
        if not isinstance(compress_ratios, list):
            compress_ratios = None
        cache_groups = build_dspark_cache_group_specs(
            int(metadata.get("num_hidden_layers", 43)), compress_ratios,
            max_seq_len=options.scheduler.max_model_len,
            max_prefill_tokens=max_prefill,
            partitions=partitions,
        )
        runtime_values = {
            "num_speculative_tokens": speculation.num_tokens,
            "max_prefill_tokens_per_request": max_prefill,
            "kv_cache_groups": cache_groups,
            "speculative_prefix_cache_replay_tokens": (
                DSPARK_SLIDING_WINDOW if speculation.num_tokens else 0
            ),
        }
        return IntegrationPlan(
            self.integration_id, placement, runtime_values,
            {"compile_kernels": True, "num_speculative_tokens": speculation.num_tokens,
             "cache_ranks": ranks},
            enable_prefix_cache=options.scheduler.enable_prefix_caching,
            capabilities=ModelCapabilities(
                sampling=SamplingCapabilities(prefill_stop_mask="device", decode_stop_mask="device"),
                constrained_tool_format="deepseek_v4",
            ),
        )

    def executor_class(self) -> type:
        from pypto_serving.model.deepseek_dspark.npu_executor import DeepSeekV4DSparkPyptoExecutor

        return DeepSeekV4DSparkPyptoExecutor
