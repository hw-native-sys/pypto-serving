# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU admission and fixed M0 topology for the low-latency Flash kernels."""

from dataclasses import dataclass

from .capacity import CACHE_GROUPS as CACHE_GROUPS, LowlatCapacityPlan, MODEL_CONTEXT


LOWLAT_EXECUTOR = "PyptoDeepSeekV4LowlatExecutor"


@dataclass(frozen=True)
class LowlatServingContract:
    target_ranks: int = 8
    target_tp: int = 8
    lm_head_tp: int = 8
    compiled_context: int = 16384
    max_prefill_tokens: int = 8192
    num_layers: int = 43
    num_speculative_tokens: int = 5

    def validate_checkpoint(self, raw):
        expected = {
            "hidden_size": 4096,
            "num_hidden_layers": 43,
            "num_attention_heads": 64,
            "vocab_size": 129280,
            "n_routed_experts": 256,
            "moe_intermediate_size": 2048,
        }
        for key, value in expected.items():
            if raw.get(key) != value:
                raise ValueError(f"lowlat requires {key}={value}, got {raw.get(key)!r}")
        if raw.get("max_position_embeddings", MODEL_CONTEXT) < self.compiled_context:
            raise ValueError("lowlat compiled context exceeds checkpoint context")
        ratios = raw.get("compress_ratios", [])[:43]
        if ratios != [0, 0] + [ratio for _ in range(20) for ratio in (4, 128)] + [4]:
            raise ValueError("lowlat requires the Flash 2 SWA / 21 CSA / 20 HCA layer order")
        if (raw.get("quantization_config") or {}).get("quant_method") != "compressed-tensors":
            raise ValueError("lowlat requires the W8A8 compressed-tensors checkpoint")

    def validate_cli(self, args, devices, raw):
        self.validate_checkpoint(raw)
        if (
            args.data_parallel_size,
            args.tensor_parallel_size,
            args.expert_parallel_size,
            args.lm_head_tp,
        ) != (1, 8, 1, 8):
            raise ValueError("lowlat M0 requires --dp 1 --tp 8 --ep 1 --lm-head-tp 8")
        if len(devices) != 8 or len(set(devices)) != 8 or any(d < 0 for d in devices):
            raise ValueError("lowlat requires eight distinct physical device IDs")
        if args.max_num_seqs != 1 or args.block_size != 128:
            raise ValueError("lowlat M0 requires --max-num-seqs 1 --block-size 128")
        if getattr(args, "pd_role", None):
            raise ValueError("lowlat M0 does not support disaggregated prefill/decode")
        if self.num_speculative_tokens == 5 and (args.max_model_len < 8 or args.max_num_batched_tokens < 6):
            raise ValueError("native K5 requires context>=8 and a scheduled token budget>=6")
        # K5 lowlat always uses the rolling cache contract.  The layout and
        # initialization policy are implementation details, not CLI options.
        LowlatCapacityPlan(args.max_model_len, args.long_prefill_token_threshold, "rolling")
        if not 1 <= args.long_prefill_token_threshold <= self.max_prefill_tokens:
            raise ValueError("lowlat prefill chunk must be in [1, 8192]")
        if args.max_num_batched_tokens <= 0:
            raise ValueError("lowlat requires a positive token budget")


def validate_sampling(config, *, constraint_spec=None):
    """Fail before admission, including before min_tokens can be downgraded."""
    if (
        config.temperature != 0
        or config.top_p != 1
        or config.top_k not in (None, -1, 1)
        or config.min_tokens != 0
    ):
        raise ValueError("lowlat M0 supports greedy sampling only, with min_tokens=0")
    if constraint_spec is not None:
        raise ValueError("lowlat M0 does not support structured generation constraints")


def build_lowlat_cache_group_specs(*, max_seq_len, max_prefill_tokens=None, speculative_width=0,
                                   layout="full", prefix_cache=False):
    """Keep physical pool capacity independent of the active modulo extent."""
    if speculative_width != 0:
        raise ValueError("invalid lowlat cache capacity")
    return LowlatCapacityPlan(max_seq_len, 128 if max_prefill_tokens is None else max_prefill_tokens, layout, prefix_cache).cache_specs()
