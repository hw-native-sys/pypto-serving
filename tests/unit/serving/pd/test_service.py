# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import asyncio
import json
import socket
from types import SimpleNamespace

import pytest

from pypto_serving.serving.engine.async_engine import PrefillChunkReady, TokenOutput
from pypto_serving.serving.memory.kv_cache import GroupReservationState
from pypto_serving.serving.pd.config import PDConfig, PDRole
from pypto_serving.serving.pd.protocol import TransferResult
from pypto_serving.serving.pd.service import PDServingService
from pypto_serving.serving.pd.worker_api import (
    OP_INSTALL_PEER,
    OP_PREPARE_REGISTRY,
    OP_POLL_TRANSFER_CHUNK,
    OP_SUBMIT_TRANSFER_CHUNK,
    OP_TRANSFER_CHUNK,
    ComponentGeometry,
    TransferChunkRequest,
    TransferChunkResponse,
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
    ) -> None:
        self.role = role
        self.kv_cache_manager = make_cache_manager(capacity_slots=capacity_slots)
        self.bundle = _bundle(self.kv_cache_manager)
        self.chunk_queues = {}
        self.prefill_acks = {}
        self.source_released = False
        self.transfer_certainties = list(transfer_certainties)
        self.transfer_calls = 0

    async def call_pd_worker(self, operation: str, payload: bytes = b"") -> bytes:
        if operation == OP_PREPARE_REGISTRY:
            return encode_worker_payload(self.bundle)
        if operation == OP_INSTALL_PEER:
            assert payload
            return b""
        if operation == OP_TRANSFER_CHUNK:
            request = decode_worker_payload(payload, TransferChunkRequest)
            self.transfer_calls += 1
            certainty = (
                self.transfer_certainties.pop(0)
                if self.transfer_certainties
                else CompletionCertainty.COMPLETED
            )
            return encode_worker_payload(
                TransferChunkResponse(
                    tuple(
                        TransferResult(
                            key=request.manifest.key,
                            chunk_id=request.manifest.chunk_id,
                            rank_id=unit.rank_id,
                            component_id=unit.component_id,
                            attempt_id=(
                                f"a{request.attempt_ordinal}-"
                                f"{unit.rank_id}-{unit.component_id}"
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
        cache_partition,
    ):
        blocks = self.kv_cache_manager.ensure_group_blocks(
            request_id,
            len(prompt_token_ids),
            partition=cache_partition,
        )
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
        yield TokenOutput(finished=True, finish_reason="FINISHED_HANDOFF")

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


class _FakeEngine:
    eos_token_id = 2

    def __init__(self, core: _FakeCore) -> None:
        self.core = core

    def _single_core(self):
        return self.core


def _config(
    role: PDRole,
    port: int,
    *,
    journal_path: str = "",
    enable_chunk_overlap: bool = False,
) -> PDConfig:
    return PDConfig(
        role=role,
        node_id="p" if role is PDRole.PREFILL else "d",
        run_id="run",
        control_host="127.0.0.1",
        control_port=port,
        control_advertise_host="127.0.0.1",
        transfer_hostname="127.0.0.1",
        model_revision="ds-v4-test",
        connect_timeout_seconds=3,
        enable_chunk_overlap=enable_chunk_overlap,
        journal_path=journal_path,
    )


class _OverlapFakeCore(_FakeCore):
    def __init__(self, role: PDRole, **kwargs) -> None:
        super().__init__(role, **kwargs)
        self.next_chunk_released = asyncio.Event()
        self.final_chunk_queued = asyncio.Event()
        self.submitted_request = None
        self.poll_calls = 0

    async def add_request(
        self,
        request_id,
        _prompt,
        _config,
        *,
        prompt_token_ids,
        cache_partition,
    ):
        blocks = self.kv_cache_manager.ensure_group_blocks(
            request_id,
            len(prompt_token_ids),
            partition=cache_partition,
        )
        tables = {name: tuple(ids) for name, ids in blocks.items()}
        queue = self.chunk_queues.setdefault(request_id, asyncio.Queue())
        ack = self.prefill_acks.setdefault(request_id, asyncio.Event())
        await queue.put(
            PrefillChunkReady(
                request_id,
                0,
                0,
                32,
                False,
                None,
                tables,
                cache_partition,
            )
        )
        await self.next_chunk_released.wait()
        await queue.put(
            PrefillChunkReady(
                request_id,
                1,
                32,
                len(prompt_token_ids),
                True,
                101,
                tables,
                cache_partition,
            )
        )
        self.final_chunk_queued.set()
        await ack.wait()
        yield TokenOutput(finished=True, finish_reason="FINISHED_HANDOFF")

    def complete_prefill_chunk_transfer(self, _request_id: str) -> None:
        self.next_chunk_released.set()

    async def call_pd_worker(self, operation: str, payload: bytes = b"") -> bytes:
        if operation == OP_SUBMIT_TRANSFER_CHUNK:
            self.submitted_request = decode_worker_payload(payload, TransferChunkRequest)
            return encode_worker_payload(TransferSubmission("job-1"))
        if operation == OP_POLL_TRANSFER_CHUNK:
            poll = decode_worker_payload(payload, TransferPollRequest)
            assert poll.job_id == "job-1"
            self.poll_calls += 1
            assert self.final_chunk_queued.is_set()
            request = self.submitted_request
            return encode_worker_payload(
                TransferPollResponse(
                    "job-1",
                    True,
                    tuple(
                        TransferResult(
                            request.manifest.key,
                            request.manifest.chunk_id,
                            unit.rank_id,
                            unit.component_id,
                            f"overlap-{unit.rank_id}-{unit.component_id}",
                            CompletionCertainty.COMPLETED.value,
                        )
                        for unit in request.manifest.expected_units
                    ),
                )
            )
        return await super().call_pd_worker(operation, payload)


def _legacy_cpu_end_to_end_prefill_transfer_commit_and_decode(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("TEST_PD_SECRET", "0123456789abcdef0123456789abcdef")
    port = _free_port()

    async def exercise() -> None:
        # K7 Decode reserves memory for its drafter and can therefore expose a
        # smaller target-cache arena than target-only Prefill.
        p_core = _FakeCore(
            PDRole.PREFILL,
            capacity_slots=2,
            transfer_certainties=(CompletionCertainty.NOT_SUBMITTED,),
        )
        d_core = _FakeCore(PDRole.DECODE, capacity_slots=1)
        assert p_core.bundle.registry_fingerprint != d_core.bundle.registry_fingerprint
        assert p_core.bundle.layout_fingerprint == d_core.bundle.layout_fingerprint
        prefill = PDServingService(
            _FakeEngine(p_core),
            _config(
                PDRole.PREFILL,
                port,
                journal_path=str(tmp_path / "p.jsonl"),
            ),
        )
        decode = PDServingService(
            _FakeEngine(d_core),
            _config(
                PDRole.DECODE,
                port,
                journal_path=str(tmp_path / "d.jsonl"),
            ),
        )
        decode_start = asyncio.create_task(decode.start())
        await asyncio.sleep(0.05)
        await asyncio.gather(prefill.start(), decode_start)
        config = SimpleNamespace(
            max_new_tokens=2,
            temperature=0.0,
            top_p=1.0,
            top_k=None,
            seed=None,
            stop=(),
            ignore_eos=True,
            stream=True,
        )
        outputs = [
            output
            async for output in prefill.generate(
                "request",
                "prompt",
                config,
                tuple(range(33)),
            )
        ]
        assert [output.token_id for output in outputs] == [101, 102]
        assert outputs[-1].finished
        assert outputs[-1].token_ids == (101, 102)
        assert p_core.source_released
        assert p_core.transfer_calls == 2
        d_reservations = d_core.kv_cache_manager.group_cache_reservations
        # Released allocator records are retired into the connector's bounded
        # terminal tombstone index; they must not grow for process lifetime.
        assert d_reservations == ()
        await asyncio.gather(prefill.close(), decode.close())

    asyncio.run(exercise())
    for role in ("p", "d"):
        events = [
            json.loads(line)["event"]
            for line in (tmp_path / f"{role}.jsonl").read_text().splitlines()
        ]
        assert "HANDOFF_CREATED" in events
        assert "HANDOFF_COMPLETED" in events
        assert events[-1] == "SERVICE_STOPPED"


def _legacy_unknown_transfer_quarantines_d_and_fails_both_services_closed(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("TEST_PD_SECRET", "0123456789abcdef0123456789abcdef")
    port = _free_port()

    async def exercise() -> None:
        p_core = _FakeCore(
            PDRole.PREFILL,
            transfer_certainties=(CompletionCertainty.UNKNOWN,),
        )
        d_core = _FakeCore(PDRole.DECODE)
        prefill = PDServingService(
            _FakeEngine(p_core),
            _config(
                PDRole.PREFILL,
                port,
                journal_path=str(tmp_path / "unknown-p.jsonl"),
            ),
        )
        decode = PDServingService(
            _FakeEngine(d_core),
            _config(
                PDRole.DECODE,
                port,
                journal_path=str(tmp_path / "unknown-d.jsonl"),
            ),
        )
        decode_start = asyncio.create_task(decode.start())
        await asyncio.sleep(0.05)
        await asyncio.gather(prefill.start(), decode_start)
        config = SimpleNamespace(
            max_new_tokens=2,
            temperature=0.0,
            top_p=1.0,
            top_k=None,
            seed=None,
            stop=(),
            ignore_eos=True,
            stream=True,
        )
        with pytest.raises(RuntimeError, match="UNKNOWN"):
            async for _ in prefill.generate(
                "unknown-request",
                "prompt",
                config,
                tuple(range(33)),
            ):
                pass
        assert prefill.health_error
        assert decode.health_error
        reservations = d_core.kv_cache_manager.group_cache_reservations
        assert len(reservations) == 1
        assert reservations[0].state is GroupReservationState.QUARANTINED
        await asyncio.gather(prefill.close(), decode.close())

    asyncio.run(exercise())
    for role in ("p", "d"):
        events = [
            json.loads(line)["event"]
            for line in (tmp_path / f"unknown-{role}.jsonl").read_text().splitlines()
        ]
        assert "RECOVERY_REQUIRED" in events


def _legacy_closed_chunk_transfer_overlaps_next_prefill_chunk(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TEST_PD_SECRET", "0123456789abcdef0123456789abcdef")
    port = _free_port()

    async def exercise() -> None:
        p_core = _OverlapFakeCore(PDRole.PREFILL, capacity_slots=2)
        d_core = _FakeCore(PDRole.DECODE, capacity_slots=2)
        prefill = PDServingService(
            _FakeEngine(p_core),
            _config(
                PDRole.PREFILL,
                port,
                journal_path=str(tmp_path / "overlap-p.jsonl"),
                enable_chunk_overlap=True,
            ),
        )
        decode = PDServingService(
            _FakeEngine(d_core),
            _config(
                PDRole.DECODE,
                port,
                journal_path=str(tmp_path / "overlap-d.jsonl"),
            ),
        )
        decode_start = asyncio.create_task(decode.start())
        await asyncio.sleep(0.05)
        await asyncio.gather(prefill.start(), decode_start)
        config = SimpleNamespace(
            max_new_tokens=2,
            temperature=0.0,
            top_p=1.0,
            top_k=None,
            seed=None,
            stop=(),
            ignore_eos=True,
            stream=True,
        )
        outputs = [
            output
            async for output in prefill.generate(
                "overlap-request",
                "prompt",
                config,
                tuple(range(33)),
            )
        ]
        assert outputs[-1].finished
        assert p_core.next_chunk_released.is_set()
        assert p_core.final_chunk_queued.is_set()
        assert p_core.poll_calls == 1
        assert prefill.metrics.snapshot()["counters"]["overlap.completed"] == 1
        await asyncio.gather(prefill.close(), decode.close())

    asyncio.run(exercise())
