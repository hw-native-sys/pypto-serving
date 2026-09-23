# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import asyncio
import socket

from pypto_serving.serving.engine.async_engine import TokenOutput
from pypto_serving.serving.pd.integration import PrefillChunkReady
from pypto_serving.serving.pd.config import PDRole
from pypto_serving.serving.pd.protocol import TransferResult
from pypto_serving.serving.pd.worker_api import (
    OP_INSTALL_PEER,
    OP_PREPARE_REGISTRY,
    OP_POLL_TRANSFER_CHUNK,
    OP_SUBMIT_TRANSFER_CHUNK,
    ComponentGeometry,
    TransferChunkRequest,
    TransferPollRequest,
    TransferPollResponse,
    TransferSubmission,
    WorkerRegistryBundle,
    decode_worker_payload,
    encode_worker_payload,
)
from pypto_serving.transfer.types import CompletionCertainty

from .helpers import make_cache_manager, make_rank_registrations, make_registry


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _bundle(manager) -> WorkerRegistryBundle:
    registry = make_registry(manager)
    return WorkerRegistryBundle(
        model_revision=registry.model_revision,
        topology=registry.topology,
        registry_fingerprint=registry.fingerprint,
        layout_fingerprint=registry.layout_fingerprint,
        components=tuple(
            ComponentGeometry(
                component_id=component.component_id,
                dtype=component.dtype,
                item_bytes=component.item_bytes,
                layers=component.layers,
                blocks_per_layer=component.blocks_per_layer,
                block_tokens=component.block_tokens,
                token_stride_bytes=component.token_stride_bytes,
                extent=component.extent,
            )
            for component in registry.components
        ),
        ranks=make_rank_registrations(registry),
    )


class _FakeCore:
    def __init__(
        self,
        role: PDRole,
        *,
        capacity_slots: int = 2,
        transfer_certainties: tuple[CompletionCertainty, ...] = (),
        source_partition: int | None = None,
    ) -> None:
        self.role = role
        self.kv_cache_manager = make_cache_manager(capacity_slots=capacity_slots)
        self.bundle = _bundle(self.kv_cache_manager)
        self.chunk_queues = {}
        self.prefill_acks = {}
        self.source_released = False
        self.transfer_certainties = list(transfer_certainties)
        self.transfer_calls = 0
        self.source_partition = source_partition
        self.transfer_jobs = {}
        self.manifests = []

    async def call_pd_worker(self, operation: str, payload: bytes = b"") -> bytes:
        if operation == OP_PREPARE_REGISTRY:
            return encode_worker_payload(self.bundle)
        if operation == OP_INSTALL_PEER:
            assert payload
            return b""
        if operation == OP_SUBMIT_TRANSFER_CHUNK:
            request = decode_worker_payload(payload, TransferChunkRequest)
            self.transfer_calls += 1
            certainty = (
                self.transfer_certainties.pop(0)
                if self.transfer_certainties
                else CompletionCertainty.COMPLETED
            )
            job_id = f"{request.manifest.key.handoff_id}-{request.manifest.chunk_id}-{request.attempt_ordinal}"
            self.transfer_jobs[job_id] = request, certainty
            self.manifests.append(request.manifest)
            return encode_worker_payload(TransferSubmission(job_id))
        if operation == OP_POLL_TRANSFER_CHUNK:
            poll = decode_worker_payload(payload, TransferPollRequest)
            request, certainty = self.transfer_jobs[poll.job_id]
            source_by_destination = {
                pair.destination_rank_id: pair.source_rank_id for pair in request.manifest.rank_mapping
            }
            return encode_worker_payload(
                TransferPollResponse(
                    poll.job_id,
                    True,
                    tuple(
                        TransferResult(
                            key=request.manifest.key,
                            chunk_id=request.manifest.chunk_id,
                            source_rank_id=source_by_destination[unit.destination_rank_id],
                            destination_rank_id=unit.destination_rank_id,
                            component_id=unit.component_id,
                            attempt_id=(
                                f"a{request.attempt_ordinal}-"
                                f"{unit.destination_rank_id}-{unit.component_id}"
                            ),
                            certainty=certainty.value,
                        )
                        for unit in request.manifest.expected_units
                    )
                )
            )
        raise AssertionError(operation)

    async def add_request(
        self,
        request_id,
        _prompt,
        _config,
        *,
        prompt_token_ids,
        cache_partition=None,
    ):
        blocks = self.kv_cache_manager.ensure_group_blocks(
            request_id,
            len(prompt_token_ids),
            partition=self.source_partition if cache_partition is None else cache_partition,
        )
        cache_partition = self.kv_cache_manager.group_request_partition(request_id)
        chunk_queue = self.chunk_queues.setdefault(request_id, asyncio.Queue())
        prefill_ack = self.prefill_acks.setdefault(request_id, asyncio.Event())
        await chunk_queue.put(
            PrefillChunkReady(
                request_id=request_id,
                chunk_id=0,
                start_token=0,
                end_token=len(prompt_token_ids),
                final=True,
                first_token=101,
                block_ids_by_group={name: tuple(ids) for name, ids in blocks.items()},
                cache_partition=cache_partition,
            )
        )
        await prefill_ack.wait()
        yield TokenOutput(finished=True, finish_reason="FINISHED_PREFILL")

    async def next_prefill_chunk(self, request_id):
        chunk = await self.chunk_queues[request_id].get()
        assert chunk.request_id == request_id
        return chunk

    def complete_prefill_chunk_transfer(self, _request_id: str) -> None:
        raise AssertionError("single final chunk should not take the non-final path")

    def acknowledge_prefill_handoff(self, request_id: str) -> None:
        self.kv_cache_manager.release_all_group_requests(request_id)
        self.source_released = True
        self.prefill_acks[request_id].set()

    async def abort_request(self, request_id: str) -> None:
        self.kv_cache_manager.release_all_group_requests(request_id)
        event = self.prefill_acks.get(request_id)
        if event is not None:
            event.set()

    async def add_adopted_handoff(
        self,
        *,
        reservation_id,
        request_id,
        prompt_token_ids,
        first_token,
        max_new_tokens,
        **_kwargs,
    ):
        reservation = self.kv_cache_manager.adopt_group_cache(reservation_id)
        assert reservation.request_id == request_id
        yield TokenOutput(
            token_id=first_token,
            text="first",
            prompt_tokens=len(prompt_token_ids),
            completion_tokens=1,
        )
        self.kv_cache_manager.release_group_cache(reservation_id)
        yield TokenOutput(
            token_id=102,
            text="first second",
            finished=True,
            finish_reason="FINISHED_LENGTH",
            prompt_tokens=len(prompt_token_ids),
            completion_tokens=max_new_tokens,
            token_ids=(first_token, 102),
        )


class _OverlapFakeCore(_FakeCore):
    """Hold the first native result until the test explicitly releases it."""

    def __init__(self, role: PDRole, **kwargs) -> None:
        super().__init__(role, **kwargs)
        self.next_chunk_released = asyncio.Event()
        self.final_chunk_queued = asyncio.Event()
        self.transfer_started = asyncio.Event()
        self.allow_completion = asyncio.Event()
        self.poll_calls = 0

    async def add_request(
        self, request_id, _prompt, _config, *, prompt_token_ids, cache_partition=None,
    ):
        blocks = self.kv_cache_manager.ensure_group_blocks(
            request_id, len(prompt_token_ids),
            partition=self.source_partition if cache_partition is None else cache_partition,
        )
        partition = self.kv_cache_manager.group_request_partition(request_id)
        tables = {name: tuple(ids) for name, ids in blocks.items()}
        queue = self.chunk_queues.setdefault(request_id, asyncio.Queue())
        ack = self.prefill_acks.setdefault(request_id, asyncio.Event())
        await queue.put(PrefillChunkReady(request_id, 0, 0, 128, False, None, tables, partition))
        await self.next_chunk_released.wait()
        await queue.put(PrefillChunkReady(
            request_id, 1, 128, len(prompt_token_ids), True, 101, tables, partition,
        ))
        self.final_chunk_queued.set()
        await ack.wait()
        yield TokenOutput(finished=True, finish_reason="FINISHED_PREFILL")

    def complete_prefill_chunk_transfer(self, _request_id: str) -> None:
        self.next_chunk_released.set()

    async def call_pd_worker(self, operation: str, payload: bytes = b"") -> bytes:
        if operation == OP_POLL_TRANSFER_CHUNK:
            self.poll_calls += 1
            poll = decode_worker_payload(payload, TransferPollRequest)
            request, _ = self.transfer_jobs[poll.job_id]
            if not request.manifest.final and not self.allow_completion.is_set():
                return encode_worker_payload(TransferPollResponse(poll.job_id, False))
        result = await super().call_pd_worker(operation, payload)
        if operation == OP_SUBMIT_TRANSFER_CHUNK:
            self.transfer_started.set()
        return result
