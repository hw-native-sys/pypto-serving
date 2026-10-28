# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Explicit, replica-local AgentInfer admission ahead of the native scheduler."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, replace

from pypto_serving.serving.memory.kv_cache import KVCacheCapacityObservation, KvCacheManager
from pypto_serving.serving.sched.scheduler import Request, ScheduledRequest, Scheduler
from pypto_serving.serving.agentinfer_core.scheduling.backend import BackendInfo, BackendPoolInfo, DpRankInfo
from pypto_serving.serving.agentinfer_core.scheduling.identity import AgentIdentity
from pypto_serving.serving.agentinfer_core.scheduling.lifecycle import ProgramLifecycle
from pypto_serving.serving.agentinfer_core.scheduling.progress_ttl import (
    ProgressTTLConfig, ProgressTTLMode, build_progress_ttl_strategy,
)
from pypto_serving.serving.agentinfer_core.scheduling.runtime import ProgramScheduler

logger = logging.getLogger(__name__)

MAX_RETAINED_REQUESTS = 1024
MAX_ADMISSION_WAIT_SECONDS = 300.0
FAST_ADMISSION_SCHEDULE_INTERVAL_SECONDS = 0.02
PRESSURE_ADMISSION_SCHEDULE_INTERVAL_SECONDS = 1.0


@dataclass(frozen=True)
class AgentInferOptions:
    """Admission bounds; None keeps the capacity-derived resume ratio."""

    resume_capacity_ratio: float | None = None
    resume_wait_relax_seconds: float | None = None
    resume_wait_growth_reserve_fraction: float = 0.0
    resume_wait_extra_headroom_tokens: int = 0
    resume_wait_preserve_cached_blocks: bool = False
    force_resume_timeout_min_seconds: float = 30.0
    force_resume_timeout_max_seconds: float = 300.0

    def __post_init__(self) -> None:
        for name in (
            "resume_capacity_ratio", "force_resume_timeout_min_seconds", "force_resume_timeout_max_seconds",
        ):
            value = getattr(self, name)
            if value is None and name == "resume_capacity_ratio":
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"agentinfer_config.{name} must be a finite number")
        if self.resume_capacity_ratio is not None and not 0 < self.resume_capacity_ratio <= 1:
            raise ValueError("agentinfer_config.resume_capacity_ratio must be in (0, 1]")
        if self.resume_wait_relax_seconds is not None and (
            isinstance(self.resume_wait_relax_seconds, bool)
            or not isinstance(self.resume_wait_relax_seconds, (int, float))
            or not math.isfinite(self.resume_wait_relax_seconds)
            or self.resume_wait_relax_seconds < 0
        ):
            raise ValueError("agentinfer_config.resume_wait_relax_seconds must be finite and non-negative")
        fraction = self.resume_wait_growth_reserve_fraction
        if (isinstance(fraction, bool) or not isinstance(fraction, (int, float))
                or not math.isfinite(fraction) or not 0 <= fraction <= 1):
            raise ValueError("agentinfer_config.resume_wait_growth_reserve_fraction must be between 0 and 1")
        if fraction and self.resume_wait_relax_seconds is None:
            raise ValueError("agentinfer_config.resume_wait_growth_reserve_fraction requires resume_wait_relax_seconds")
        if (isinstance(self.resume_wait_extra_headroom_tokens, bool)
                or not isinstance(self.resume_wait_extra_headroom_tokens, int)
                or self.resume_wait_extra_headroom_tokens < 0):
            raise ValueError("agentinfer_config.resume_wait_extra_headroom_tokens must be a non-negative integer")
        if self.resume_wait_extra_headroom_tokens and self.resume_wait_relax_seconds is None:
            raise ValueError("agentinfer_config.resume_wait_extra_headroom_tokens requires resume_wait_relax_seconds")
        if not isinstance(self.resume_wait_preserve_cached_blocks, bool):
            raise ValueError("agentinfer_config.resume_wait_preserve_cached_blocks must be a boolean")
        if self.resume_wait_preserve_cached_blocks and self.resume_wait_relax_seconds is None:
            raise ValueError("agentinfer_config.resume_wait_preserve_cached_blocks requires resume_wait_relax_seconds")
        if not 0 <= self.force_resume_timeout_min_seconds <= self.force_resume_timeout_max_seconds:
            raise ValueError("agentinfer_config force-resume bounds must satisfy 0 <= min <= max")

    @classmethod
    def from_mapping(cls, values: dict) -> AgentInferOptions:
        if not isinstance(values, dict):
            raise ValueError("--agentinfer-config must be a JSON object")
        unknown = values.keys() - cls.__dataclass_fields__.keys()
        if unknown:
            raise ValueError(f"Unknown agentinfer_config fields: {', '.join(sorted(unknown))}")
        return cls(**values)


@dataclass
class _Retained:
    request: Request
    arrived_at: float


class AgentInferAdmission:
    """Own AgentInfer bindings while the native scheduler owns admitted work."""

    def __init__(
        self, dp_rank: int, scheduler: Scheduler, kv_cache_manager: KvCacheManager,
        options: AgentInferOptions | None = None,
    ) -> None:
        self._backend_types = (BackendInfo, BackendPoolInfo, DpRankInfo)
        self._identity_type = AgentIdentity
        self._unknown_lifecycle = ProgramLifecycle.UNKNOWN
        self._lifecycle_type = ProgramLifecycle
        self.dp_rank = dp_rank
        self.native = scheduler
        self.kv_cache_manager = kv_cache_manager
        self.grouped = kv_cache_manager.has_groups
        self._max_seq_len = scheduler.config.max_seq_len
        if self.grouped and (
            scheduler.config.num_speculative_tokens != 1
            or not scheduler.config.enable_prefix_cache
            or not any(group.is_eagle_group for group in kv_cache_manager.group_specs)
        ):
            raise ValueError("Grouped AgentInfer admission requires DeepSeek V4 MTP K=1 prefix caching")
        self.options = options or AgentInferOptions()
        self._sessions: dict[str, str] = {}
        self._retained: dict[str, _Retained] = {}
        self._prefix_observed: set[str] = set()
        self._prefix_refresh_pending = False
        # Keep the Qwen NPU calibration separate from the grouped-cache path.
        # DeepSeek uses the upstream defaults until device timing is available.
        qwen_cost = dict(
            ttl_prefill_model_intercept_seconds=0.0135,
            ttl_prefill_model_linear_seconds_per_1k_tokens=0.351,
            ttl_prefill_model_quadratic_seconds_per_1k_tokens_squared=0.173,
            decode_step_fixed_seconds=0.0279,
            decode_step_seconds_per_request=0.0,
            decode_step_seconds_per_context_token=8.6e-7,
        ) if not self.grouped else {}
        config = ProgressTTLConfig(
            mode=ProgressTTLMode.ON,
            **qwen_cost,
            resume_wait_relax_seconds=self.options.resume_wait_relax_seconds,
            resume_wait_growth_reserve_fraction=self.options.resume_wait_growth_reserve_fraction,
            resume_wait_extra_headroom_tokens=self.options.resume_wait_extra_headroom_tokens,
            resume_wait_preserve_cached_blocks=self.options.resume_wait_preserve_cached_blocks,
            force_resume_timeout_min_seconds=self.options.force_resume_timeout_min_seconds,
            force_resume_timeout_max_seconds=self.options.force_resume_timeout_max_seconds,
        )
        full_batch_tokens = scheduler.config.max_num_running_reqs * scheduler.config.max_seq_len
        capacity_tokens = self._observe_capacity().total_tokens
        # Leave room for one returning session's context while admitting background work.
        # A pool smaller than two contexts cannot provide this guarantee.
        reserve_tokens = (
            scheduler.config.max_seq_len
            if 2 * scheduler.config.max_seq_len <= capacity_tokens < 2 * full_batch_tokens
            else 0
        )
        if reserve_tokens:
            config = replace(
                config,
                resume_capacity_ratio=(capacity_tokens - reserve_tokens) / capacity_tokens,
                shared_prefix_freshness_warmup_seconds=0.0,
            )
        if self.options.resume_capacity_ratio is not None:
            config = replace(config, resume_capacity_ratio=self.options.resume_capacity_ratio)
        components = build_progress_ttl_strategy(
            config, capacity_cost=self._capacity_cost if self.grouped else None,
        )
        schedule_interval = (
            FAST_ADMISSION_SCHEDULE_INTERVAL_SECONDS
            if reserve_tokens or capacity_tokens >= 2 * full_batch_tokens
            else PRESSURE_ADMISSION_SCHEDULE_INTERVAL_SECONDS
        )
        self.policy = ProgramScheduler(
            components.strategy, components.initial_factors, self._backend_pool(),
            schedule_interval_seconds=schedule_interval,
        )
        logger.info(
            "AgentInfer admission configured: capacity_tokens=%d reserve_tokens=%d schedule_interval=%.3fs "
            "resume_capacity_ratio=%.4f force_resume_min=%.3fs force_resume_max=%.3fs",
            capacity_tokens, capacity_tokens - math.floor(capacity_tokens * config.resume_capacity_ratio),
            schedule_interval, config.resume_capacity_ratio,
            config.force_resume_timeout_min_seconds, config.force_resume_timeout_max_seconds,
        )

    @property
    def retained_request_ids(self) -> tuple[str, ...]:
        return tuple(self._retained)

    def _capacity_cost(self, tokens: int) -> int:
        if not self.grouped:
            return tokens
        return self.kv_cache_manager.group_capacity_cost_tokens(
            min(tokens, self._max_seq_len), self._max_seq_len,
        )

    def _observe_capacity(self) -> KVCacheCapacityObservation:
        if self.grouped:
            return self.kv_cache_manager.observe_group_capacity(self._max_seq_len)
        return self.kv_cache_manager.observe_capacity()

    def _observe_uncached_free_head(self) -> int | None:
        if not self.options.resume_wait_preserve_cached_blocks:
            return None
        if self.grouped:
            return self.kv_cache_manager.observe_group_uncached_free_head_tokens(self._max_seq_len)
        return self.kv_cache_manager.observe_uncached_free_head_tokens()

    def _backend_pool(self):
        BackendInfo, BackendPoolInfo, DpRankInfo = self._backend_types
        capacity = self._observe_capacity()
        # Match upstream waiting demand: current contexts, including tracked
        # Programs, without reserving every queued request's future decode budget.
        native_waiting = sum(self._capacity_cost(request.num_tokens) for request in self.native.waiting)
        now = time.monotonic()
        return BackendPoolInfo(
            backends=(BackendInfo(
                backend_id=f"pypto-replica-{self.dp_rank}",
                backend_url=f"embedded://pypto/{self.dp_rank}",
                healthy=True,
                dp_ranks=(DpRankInfo(
                    dp_rank=self.dp_rank,
                    healthy=True,
                    schedulable=True,
                    running_requests=len(self.native.running),
                    waiting_requests=len(self.native.waiting),
                    total_hbm_kv_tokens=capacity.total_tokens,
                    used_hbm_kv_tokens=capacity.used_tokens,
                    waiting_hbm_kv_tokens=native_waiting,
                    uncached_free_head_hbm_kv_tokens=self._observe_uncached_free_head(),
                ),),
                observed_at_monotonic_s=now,
            ),),
            observed_at_monotonic_s=now,
        )

    def arrive(self, request: Request, identity: str | AgentIdentity) -> bool:
        """Return whether the caller should immediately enqueue the request."""
        if request.request_id in self._sessions:
            raise ValueError(f"duplicate AgentInfer request: {request.request_id}")
        if len(self._retained) >= MAX_RETAINED_REQUESTS:
            logger.warning("AgentInfer admission queue full; using native queue for %s", request.request_id)
            return True
        metadata = (
            identity if isinstance(identity, AgentIdentity)
            else self._identity_type(program_id=identity, session_id=identity)
        )
        self._sessions[request.request_id] = metadata.program_id
        request.observe_prefix_cache = True
        try:
            admitted = self.policy.on_request_arrival(
                request.request_id,
                metadata,
                request.num_prompt_tokens,
                self._backend_pool(),
                request,
            )
        except BaseException:
            self._sessions.pop(request.request_id, None)
            self.policy.cancel_request(request.request_id)
            raise
        if not admitted:
            self._retained[request.request_id] = _Retained(request, time.monotonic())
            self._observe_retained_prefix(request.request_id, request)
        return admitted

    def poll(self) -> list[Request]:
        """Advance TTL checks even when the native scheduler is idle."""
        now = time.monotonic()
        self.policy.run_due_lightweight_checks(now)
        if self._prefix_refresh_pending:
            self._refresh_retained_prefix_cache()
            self._prefix_refresh_pending = False
        if self.policy.needs_schedule_cycle(now):
            self.policy.schedule_cycle(self._backend_pool())
        admitted = []
        for entry, _ in self.policy.consume_admitted_requests():
            retained = self._retained.pop(entry.request_id, None)
            if retained is not None:
                admitted.append(retained.request)
                logger.info(
                    "AgentInfer admitted request %s after %.3fs",
                    entry.request_id, now - retained.arrived_at,
                )
        for request_id, retained in tuple(self._retained.items()):
            if now - retained.arrived_at >= MAX_ADMISSION_WAIT_SECONDS:
                self.policy.cancel_request(request_id)
                self._retained.pop(request_id, None)
                self._sessions.pop(request_id, None)
                admitted.append(retained.request)
                logger.warning("AgentInfer admission timeout for %s; using native queue", request_id)
        return admitted

    def _refresh_retained_prefix_cache(self) -> None:
        """Mirror the upstream completion-triggered lookup without acquiring KV references."""
        for request_id, request in self.policy.prefix_cache_candidates():
            self._observe_retained_prefix(request_id, request)

    def _observe_retained_prefix(self, request_id: str, request: Request) -> None:
        try:
            if self.grouped:
                hashes = self.kv_cache_manager.compute_group_block_hashes(request.prompt_token_ids)
                prefix = self.kv_cache_manager.observe_group_prefix(
                    hashes,
                    max_cache_hit_tokens=max(
                        0, request.num_prompt_tokens
                        - max(1, self.native.config.speculative_prefix_cache_replay_tokens),
                    ),
                )
            else:
                prefix = self.kv_cache_manager.observe_prefix(request.prompt_token_ids)
        except Exception:
            logger.exception("AgentInfer retained-prefix observation failed for %s", request_id)
            return
        # Upstream uses all immediately reusable tokens for this segment estimate,
        # without attributing residual self KV to a different Program.
        self.policy.on_prefix_cache_observation(request_id, prefix.cached_tokens)

    def native_scheduled(self, scheduled: ScheduledRequest) -> None:
        request_id = scheduled.request.request_id
        if request_id not in self._sessions or request_id in self._prefix_observed or not scheduled.is_prefill:
            return
        self._prefix_observed.add(request_id)
        hit = min(scheduled.request.num_prompt_tokens, scheduled.num_computed_tokens)
        prefix = scheduled.request.prefix_cache_observation
        if self.grouped:
            self.policy.on_prefix_cache_observation(
                request_id, hit, hbm_cached_prefix_tokens=hit,
            )
        elif prefix is not None and prefix.cached_tokens == hit:
            self.policy.on_prefix_cache_observation(
                request_id, hit, shared_prefix_tokens=prefix.shared_tokens,
                hbm_cached_prefix_tokens=hit, hbm_hit_ref_zero_tokens=prefix.reclaimable_tokens,
            )
        else:
            self.policy.on_prefix_cache_observation(
                request_id, hit, hbm_cached_prefix_tokens=hit,
            )

    def observe_lifecycle(self, session_id: str, request_id: str, lifecycle: str) -> bool:
        """Forward a parsed response or client-confirmed fact to the original policy."""
        released = self.policy.on_response_completion(session_id, request_id, self._lifecycle_type(lifecycle))
        logger.info(
            "AgentInfer lifecycle session=%s request=%s lifecycle=%s released=%s",
            session_id, request_id, lifecycle, released,
        )
        return released

    def progress(self, request: Request) -> None:
        if request.request_id in self._sessions:
            self.policy.on_stream_output(request.request_id, len(request.output_token_ids))

    def complete(self, request: Request) -> None:
        session_id = self._sessions.pop(request.request_id, None)
        if session_id is None:
            return
        self._retained.pop(request.request_id, None)
        self._prefix_observed.discard(request.request_id)
        self.policy.on_stream_output(request.request_id, len(request.output_token_ids))
        self.policy.on_request_completion(
            request.request_id, request.num_prompt_tokens + len(request.output_token_ids),
        )
        self.policy.on_response_completion(session_id, request.request_id, self._unknown_lifecycle)
        self._prefix_refresh_pending = True

    def cancel(self, request_id: str) -> bool:
        """Return True when the request was retained and never reached native scheduling."""
        retained = self._retained.pop(request_id, None) is not None
        self._prefix_observed.discard(request_id)
        if self._sessions.pop(request_id, None) is not None:
            self.policy.cancel_request(request_id)
        return retained

    def close(self) -> None:
        for request_id in tuple(self._sessions):
            self.cancel(request_id)
