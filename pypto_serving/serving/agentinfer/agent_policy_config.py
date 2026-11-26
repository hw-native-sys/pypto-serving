# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Local calibration and capacity controls for the adaptive admission policy."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from enum import Enum


class ProgressTTLMode(str, Enum):
    OFF = "off"
    ON = "on"
    AUTO = "auto"


class ProgressTTLResumeOrder(str, Enum):
    FCFS = "fcfs"
    MRU = "mru"


@dataclass(frozen=True)
class ProgressTTLConfig:
    mode: ProgressTTLMode = ProgressTTLMode.ON
    resume_order: ProgressTTLResumeOrder = ProgressTTLResumeOrder.MRU
    resume_reclaim_acting_programs: bool = True
    resume_capacity_ratio: float = 1.0
    pause_capacity_ratio: float = 1.0
    decode_buffer_tokens: int = 100

    # The longest continuous service segment and its projected KV growth.
    target_max_segment_rounds: int = 14
    use_fixed_input_token_growth: bool = False
    fixed_input_token_growth_per_round: int = 1024
    enable_batch_gain_admission: bool = True

    # Idle retention and the calibrated cost of computing a cold prefix.
    ttl_min_seconds: float = 0.05
    ttl_max_seconds: float = 32.0
    paused_program_ttl_seconds: float = 1800.0
    ttl_max_cache_miss_impact_ratio: float = 1.0
    ttl_prefill_model_intercept_seconds: float = 0.042935
    ttl_prefill_model_linear_seconds_per_1k_tokens: float = 0.080027
    ttl_prefill_model_quadratic_seconds_per_1k_tokens_squared: float = 0.00220962
    ttl_decode_throughput_alpha: float = 0.15
    decode_step_fixed_seconds: float = 0.012112
    decode_step_seconds_per_request: float = 0.0006939
    decode_step_seconds_per_context_token: float = 2.2516e-7
    shared_prefix_freshness_kv_turnovers: float = 2.0
    shared_prefix_freshness_warmup_seconds: float = 100.0
    auto_disable_utility_seconds: float = 5.0
    auto_enable_utility_seconds: float = 20.0

    # Bounded queue latency and optional access to reserved headroom.
    force_resume_timeout_min_seconds: float = 30.0
    force_resume_timeout_max_seconds: float = 300.0
    force_resume_timeout_scale: float = 3.0
    resume_wait_relax_seconds: float | None = None
    resume_wait_extra_headroom_tokens: int = 0
    resume_wait_growth_reserve_fraction: float = 0.0
    resume_wait_preserve_cached_blocks: bool = False
    privileged_ttl_seconds: float = 5.0
    privileged_max_context_tokens: int = 262_144

    def __post_init__(self) -> None:
        positive_fields = {
            "decode_step_fixed_seconds",
            "force_resume_timeout_scale",
            "pause_capacity_ratio",
            "privileged_max_context_tokens",
            "resume_capacity_ratio",
            "shared_prefix_freshness_kv_turnovers",
            "target_max_segment_rounds",
        }
        fractions = {
            "pause_capacity_ratio",
            "resume_capacity_ratio",
            "resume_wait_growth_reserve_fraction",
            "ttl_decode_throughput_alpha",
            "ttl_max_cache_miss_impact_ratio",
        }
        for option in fields(self):
            key, default = option.name, option.default
            value = getattr(self, key)
            if value is None and key == "resume_wait_relax_seconds":
                continue
            if isinstance(default, (Enum, bool)) or type(default) is int:
                if type(value) is not type(default):
                    raise ValueError(f"invalid type for {key}")
            elif isinstance(value, bool) or not isinstance(value, (float, int)):
                raise ValueError(f"{key} must be a number")
            if isinstance(value, (Enum, bool)):
                continue
            if not math.isfinite(value) or value < 0 or (key in positive_fields and value == 0):
                raise ValueError(f"invalid value for {key}")
            if key in fractions and value > 1:
                raise ValueError(f"{key} must be at most 1")
        for lower, upper in (
            ("ttl_min_seconds", "ttl_max_seconds"),
            ("force_resume_timeout_min_seconds", "force_resume_timeout_max_seconds"),
            ("auto_disable_utility_seconds", "auto_enable_utility_seconds"),
        ):
            if getattr(self, lower) > getattr(self, upper):
                raise ValueError(f"{lower} must not exceed {upper}")
        if self.resume_wait_relax_seconds is None:
            for key in (
                "resume_wait_extra_headroom_tokens",
                "resume_wait_growth_reserve_fraction",
                "resume_wait_preserve_cached_blocks",
            ):
                if getattr(self, key):
                    raise ValueError(f"{key} requires resume_wait_relax_seconds")
