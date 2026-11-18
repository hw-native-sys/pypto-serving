# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Admission and confirmed-step semantics for lowlat, independent of engine transport."""

from dataclasses import dataclass

import msgspec

from pypto_serving.config.types import RequestExecutionContext
from pypto_serving.model.execution_policy import ExecutionPolicy
from pypto_serving.serving.memory.protocol import CachePageIdentity, GroupCacheCopyPlan
from .contract import validate_sampling
from .prefix_protocol import PrefixResumeDescriptor
from .speculation import validate_native_capacity


@dataclass(frozen=True)
class LowlatRequestContext(RequestExecutionContext):
    prefix_resume: PrefixResumeDescriptor | None = None


@dataclass(frozen=True)
class LowlatRequestMetadata:
    prefix_resume: PrefixResumeDescriptor | None = None
    version: int = 1


@dataclass(frozen=True)
class LowlatPrefillMetadata:
    page_identities: tuple[CachePageIdentity, ...] = ()
    checkpoint_plan: GroupCacheCopyPlan | None = None
    version: int = 1


def decode_prefill_metadata(payload):
    metadata = (
        msgspec.msgpack.decode(payload, type=LowlatPrefillMetadata) if payload else LowlatPrefillMetadata()
    )
    if metadata.version != 1:
        raise ValueError("Unsupported lowlat prefill metadata version")
    return metadata


def decode_request_metadata(payload):
    metadata = (
        msgspec.msgpack.decode(payload, type=LowlatRequestMetadata) if payload else LowlatRequestMetadata()
    )
    if metadata.version != 1:
        raise ValueError("Unsupported lowlat request metadata version")
    return metadata


class LowlatExecutionPolicy(ExecutionPolicy):
    requires_token_budget = True

    def resolve_async_scheduling(self):
        if self.config.resolve_runtime_config().num_speculative_tokens != 5 or self.config.async_scheduling is False:
            raise ValueError("lowlat requires K5 and async scheduling")
        return True

    def validate_min_tokens(self, value):
        if value:
            raise ValueError("lowlat M0 does not support min_tokens")

    def validate_generate_config(self, config, *, constraint_spec=None):
        if self.config.max_num_running_reqs != 1:
            raise ValueError("lowlat requires B1")
        validate_sampling(config, constraint_spec=constraint_spec)

    def validate_request_budget(self, prompt_tokens, output_tokens):
        runtime = self.config.resolve_runtime_config()
        effective = min(output_tokens, runtime.max_seq_len - prompt_tokens)
        validate_native_capacity(
            prompt_tokens=prompt_tokens,
            max_output_tokens=effective,
            physical_capacity=runtime.max_seq_len,
        )

    def scheduler_options(self):
        from .scheduling import LowlatSchedulingPolicy

        return dict(
            execution_policy=LowlatSchedulingPolicy(
                self.config.executor_kwargs.get("prefix_cache_namespace", "")
            )
        )

    def request_metadata(self, request):
        resume = request.execution_state.prefix_resume if request.execution_state is not None else None
        return msgspec.msgpack.encode(LowlatRequestMetadata(resume)) if resume is not None else b""

    def prefill_metadata(self, scheduled, *, is_new, cache):
        prefix = self.config.executor_kwargs.get("prefix_cache_namespace", "")
        if not prefix:
            return b""
        pages = (
            cache.group_page_identities(scheduled.block_ids_by_group, scheduled.cache_partition)
            if (is_new or getattr(scheduled.execution_plan, "checkpoint", None) is not None)
            else ()
        )
        return msgspec.msgpack.encode(
            LowlatPrefillMetadata(pages, getattr(scheduled.execution_plan, "checkpoint", None))
        )

    def confirm_step(self, output, result, cache):
        from .decoding import decode_retirement, validate_native_retirement

        expected = [s for s in output.scheduled_requests if s.decode_reservation is not None]
        if result.retired_reservation_ids != tuple(
            s.decode_reservation.reservation_id for s in expected
        ) or len(result.execution_receipts) != len(expected):
            raise ValueError("native physical retirement receipt count/ID differs")
        updates = []
        for scheduled, payload in zip(expected, result.execution_receipts, strict=True):
            receipt = decode_retirement(payload)
            request = scheduled.request
            validate_native_retirement(
                receipt,
                scheduled.decode_reservation,
                previous_input_end=request.execution_state.physical_end,
                previous_raw_outputs=request.execution_state.raw_output_count,
                max_output_tokens=request.max_new_tokens,
                actual_token_count=len(result.new_tokens.get(request.request_id, [])),
            )
            updates.append((scheduled, receipt))
        expected_copies = {
            s.request.request_id: s
            for s in output.scheduled_requests
            if getattr(s.execution_plan, "checkpoint", None) is not None
        }
        proofs = {p.request_id: p for p in result.cache_publications}
        if len(proofs) != len(result.cache_publications) or set(expected_copies) != set(proofs):
            raise ValueError("Checkpoint completion receipts do not match scheduled copies")
        publications = []
        for request_id, scheduled in expected_copies.items():
            if scheduled.request.status.is_finished or scheduled.request.status.name == "PREEMPTED":
                continue
            plan = getattr(scheduled.execution_plan, "checkpoint", None)
            proof = proofs[request_id]
            if (
                proof.page_identities != plan.destination_pages
                or proof.complete_logical_pages != plan.complete_logical_pages
            ):
                raise ValueError("Checkpoint receipt differs from reserved copy plan")
            kwargs = dict(
                request_id=request_id,
                generation=scheduled.request.execution_generation,
                mapping_version=plan.mapping_version,
                snapshot=plan.destination_tables,
                partition=scheduled.cache_partition,
                block_hashes=scheduled.request.group_block_hashes,
            )
            cache.publish_group_page_proof(proof, **kwargs, validate_only=True)
            publications.append((scheduled, proof, kwargs))
        # No model cursor or cache hash is changed until every receipt is valid.
        for scheduled, proof, kwargs in publications:
            cache.publish_group_page_proof(proof, **kwargs)
            cache.release_prefix_checkpoint(scheduled.request.request_id)
            scheduled.request.execution_state.checkpoint_complete = True
        for scheduled, receipt in updates:
            scheduled.confirmed_input_end = receipt.physical_input_end
            scheduled.request.execution_state.physical_end = receipt.physical_input_end
            scheduled.request.execution_state.raw_output_count += receipt.raw_output_tokens
