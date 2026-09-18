# Copyright (c) PyPTO Contributors.
# Licensed under CANN Open Software License Agreement Version 2.0.

import asyncio
import json
import time

import pytest

from pypto_serving.model.deepseek_dspark.pd_adapter import DSV4_DSPARK_K7_CONTRACT
from pypto_serving.router.config import RouterConfig
from pypto_serving.router.coordinator import RouterCoordinator
from pypto_serving.router.directory import WorkerDirectory
from pypto_serving.router.journal import RouterJournal
from pypto_serving.router.policy import RoundRobinRoutePolicy
from pypto_serving.serving.pd.config import PDCapabilities, PDRole
from pypto_serving.serving.pd.http_api import (
    DecodeStreamFrame,
    CapacitySnapshot,
    NodeDescriptor,
    PlacementRejection,
    PlacementReservation,
    PrefillHandoffResult,
    PreparedRequest,
    ReservePlacementResult,
)
from pypto_serving.serving.pd.protocol import (
    CapabilityWire,
    ContinuationMetadata,
    DecodeOutputWire,
    HandoffKey,
)


def _descriptor(role: PDRole, node_id: str = "") -> NodeDescriptor:
    node_id = node_id or role.value
    capabilities = PDCapabilities(
        adapter_id=DSV4_DSPARK_K7_CONTRACT.adapter_id,
        contract_version=DSV4_DSPARK_K7_CONTRACT.version,
        contract_digest=DSV4_DSPARK_K7_CONTRACT.digest,
        continuation_schema=DSV4_DSPARK_K7_CONTRACT.continuation_schema,
        model_revision="model",
        registry_fingerprint=("a" if role is PDRole.PREFILL else "b") * 64,
        layout_fingerprint="c" * 64,
        topology=(16, 4),
        logical_groups=DSV4_DSPARK_K7_CONTRACT.logical_groups,
        physical_regions=DSV4_DSPARK_K7_CONTRACT.physical_regions,
    )
    return NodeDescriptor(
        node_id=node_id,
        role=role.value,
        run_id="run",
        control_host="decode-control",
        control_port=29831,
        owner_generation=1,
        endpoint_generation=1,
        control_incarnation=1,
        capabilities=CapabilityWire.from_capabilities(capabilities),
        health="READY",
    )


def _capacity(role: PDRole, node_id: str = "") -> CapacitySnapshot:
    node_id = node_id or role.value
    return CapacitySnapshot(
        node_id=node_id,
        role=role.value,
        active_handoffs=0,
        prepared_requests=0,
        reservations=0,
        quarantined_reservations=0,
        snapshot_sequence=1,
        active_limit=4,
    )


class _PrefillClient:
    def __init__(self, waiting: asyncio.Event, node_id: str = "prefill") -> None:
        self.waiting = waiting
        self.node_id = node_id
        self.execute_calls = 0
        self.abort_calls = 0
        self.query_calls = 0

    async def get(self, path, _type):
        if path == "/internal/pd/descriptor":
            return _descriptor(PDRole.PREFILL, self.node_id)
        if path == "/internal/pd/capacity":
            return _capacity(PDRole.PREFILL, self.node_id)
        raise AssertionError(path)

    async def post(self, path, payload, response_type=None):
        if path == "/internal/pd/prepare":
            continuation = ContinuationMetadata(
                prompt_token_ids=(1, 2, 3),
                max_new_tokens=2,
                temperature=0.0,
                top_p=1.0,
                top_k=None,
                seed=None,
                stop_strings=(),
                eos_token_id=None,
            )
            return PreparedRequest(
                request_id=payload.request_id,
                prepared_request_id="prepared",
                prepared_digest="d" * 64,
                continuation=continuation,
                expires_at_ns=time.time_ns() + 10_000_000_000,
            )
        if path == "/internal/pd/execute":
            assert self.waiting.is_set()
            self.execute_calls += 1
            return PrefillHandoffResult(
                payload.key,
                payload.reservation_id,
                "m" * 64,
                "READY",
            )
        if path == "/internal/pd/abort":
            self.abort_calls += 1
            return None
        if path == "/internal/pd/query":
            self.query_calls += 1
            return None
        raise AssertionError(path)


class _DecodeClient:
    def __init__(self, waiting: asyncio.Event, node_id: str = "decode") -> None:
        self.waiting = waiting
        self.node_id = node_id
        self.reservation = None
        self.abort_calls = 0
        self.query_calls = 0

    async def get(self, path, _type):
        if path == "/internal/pd/descriptor":
            return _descriptor(PDRole.DECODE, self.node_id)
        if path == "/internal/pd/capacity":
            return _capacity(PDRole.DECODE, self.node_id)
        raise AssertionError(path)

    async def post(self, path, payload, response_type=None):
        if path == "/internal/pd/reserve":
            self.reservation = PlacementReservation(
                key=payload.key,
                prepared_request_id=payload.prepared_request_id,
                reservation_id="reservation",
                partition=0,
                block_ids_by_group={"ori": (1,)},
                prepared_digest=payload.prepared_digest,
                reservation_capability="capability",
                decode_node_id=self.node_id,
                decode_control_host="decode-control",
                decode_control_port=29831,
                decode_endpoint_generation=1,
            )
            return ReservePlacementResult(reservation=self.reservation)
        if path == "/internal/pd/authorize":
            assert payload.reservation_id == self.reservation.reservation_id
            return None
        if path == "/internal/pd/abort":
            self.abort_calls += 1
            return None
        if path == "/internal/pd/query":
            self.query_calls += 1
            return None
        raise AssertionError(path)

    async def stream_decode(self, _path, payload):
        self.waiting.set()
        yield DecodeStreamFrame(event="waiting", state="AUTHORIZED")
        for sequence in (1, 2):
            finished = sequence == 2
            yield DecodeStreamFrame(
                event="finished" if finished else "output",
                state="COMPLETED" if finished else "IN_USE",
                output=DecodeOutputWire(
                    key=payload.key,
                    token_id=100 + sequence,
                    text="hello" if finished else "hel",
                    finished=finished,
                    finish_reason="FINISHED_LENGTH" if finished else "",
                    prompt_tokens=3,
                    completion_tokens=sequence,
                    token_ids=(101, 102) if finished else (),
                    output_sequence=sequence,
                ),
            )


class _GapDecodeClient(_DecodeClient):
    async def stream_decode(self, _path, payload):
        self.waiting.set()
        yield DecodeStreamFrame(event="waiting", state="AUTHORIZED")
        yield DecodeStreamFrame(
            event="output",
            state="IN_USE",
            output=DecodeOutputWire(
                key=payload.key,
                token_id=102,
                text="gap",
                finished=False,
                finish_reason="",
                prompt_tokens=3,
                completion_tokens=1,
                output_sequence=2,
            ),
        )


class _EOFDecodeClient(_DecodeClient):
    async def stream_decode(self, _path, payload):
        self.waiting.set()
        yield DecodeStreamFrame(event="waiting", state="AUTHORIZED")


class _ReserveFailureDecodeClient(_DecodeClient):
    async def post(self, path, payload, response_type=None):
        if path == "/internal/pd/reserve":
            raise RuntimeError("D capacity unavailable")
        return await super().post(path, payload, response_type)


class _RejectDecodeClient(_DecodeClient):
    def __init__(self, waiting: asyncio.Event, node_id: str = "decode-reject") -> None:
        super().__init__(waiting, node_id)
        self.reserve_calls = 0

    async def post(self, path, payload, response_type=None):
        if path == "/internal/pd/reserve":
            self.reserve_calls += 1
            return ReservePlacementResult(
                rejection=PlacementRejection(
                    key=payload.key,
                    decode_node_id=self.node_id,
                    decode_endpoint_generation=1,
                    reason="CAPACITY_EXHAUSTED",
                    retryable=True,
                )
            )
        return await super().post(path, payload, response_type)


def _coordinator(monkeypatch, tmp_path, p_client, d_client):
    config = RouterConfig(
        prefill_urls=("http://prefill",),
        decode_urls=("http://decode",),
        run_id="run",
        policy="round_robin",
        provider="mooncake",
        journal_path=str(tmp_path / "router.jsonl"),
        log_dir=str(tmp_path / "logs"),
    )
    journal = RouterJournal(config.journal_path, config.run_id)
    coordinator = RouterCoordinator(
        config,
        WorkerDirectory(
            (p_client,),
            (d_client,),
            "run",
            RoundRobinRoutePolicy(),
        ),
        journal,
    )
    return coordinator, journal


def test_router_owns_route_and_opens_d_stream_before_p_execute(
    monkeypatch, tmp_path
) -> None:
    async def exercise() -> None:
        waiting = asyncio.Event()
        p_client = _PrefillClient(waiting)
        d_client = _DecodeClient(waiting)
        coordinator, journal = _coordinator(
            monkeypatch, tmp_path, p_client, d_client
        )
        outputs = [
            output
            async for output in coordinator.generate(
                "completion",
                b'{"prompt":"hello","max_tokens":2}',
                "request",
            )
        ]
        assert p_client.execute_calls == 1
        assert [output.output_sequence for output in outputs] == [1, 2]
        assert outputs[-1].finished
        assert journal.unresolved == ()
        gauges = coordinator.metrics.snapshot()["gauges"]
        assert gauges["handoffs.active"] == 0
        assert gauges["handoffs.active_peak"] == 1
        journal.close()

    asyncio.run(exercise())


def test_directory_rejects_a_stale_router_control_incarnation() -> None:
    async def exercise() -> None:
        waiting = asyncio.Event()
        directory = WorkerDirectory(
            (_PrefillClient(waiting),),
            (_DecodeClient(waiting),),
            "run",
            RoundRobinRoutePolicy(),
            control_incarnation=2,
        )
        with pytest.raises(RuntimeError, match="no eligible P/D pool"):
            await directory.refresh()

    asyncio.run(exercise())


def test_router_fails_closed_and_aborts_both_nodes_on_output_gap(
    monkeypatch, tmp_path
) -> None:
    async def exercise() -> None:
        waiting = asyncio.Event()
        p_client = _PrefillClient(waiting)
        d_client = _GapDecodeClient(waiting)
        coordinator, journal = _coordinator(
            monkeypatch, tmp_path, p_client, d_client
        )

        with pytest.raises(RuntimeError, match="stale or discontinuous"):
            _ = [
                output
                async for output in coordinator.generate(
                    "completion",
                    b'{"prompt":"hello","max_tokens":2}',
                    "request-gap",
                )
            ]
        assert p_client.abort_calls == 1
        assert d_client.abort_calls == 1
        assert journal.unresolved == ()
        with open(journal.path, encoding="utf-8") as stream:
            events = [json.loads(line)["event"] for line in stream]
        assert events[-1] == "RECOVERY_REQUIRED"
        journal.close()

    asyncio.run(exercise())


def test_non_streaming_failure_remains_eligible_for_new_generation_reprefill(
    monkeypatch, tmp_path
) -> None:
    async def exercise() -> None:
        waiting = asyncio.Event()
        coordinator, journal = _coordinator(
            monkeypatch,
            tmp_path,
            _PrefillClient(waiting),
            _GapDecodeClient(waiting),
        )
        with pytest.raises(RuntimeError, match="stale or discontinuous"):
            _ = [
                output
                async for output in coordinator.generate(
                    "completion",
                    b'{"prompt":"hello","max_tokens":2}',
                    "request-replayable",
                    publish_immediately=False,
                )
            ]
        assert not coordinator._replayable["request-replayable"].output_published
        journal.close()

    asyncio.run(exercise())


def test_router_client_cancellation_records_recovery_and_aborts_pair(
    monkeypatch, tmp_path
) -> None:
    async def exercise() -> None:
        waiting = asyncio.Event()
        p_client = _PrefillClient(waiting)
        d_client = _DecodeClient(waiting)
        coordinator, journal = _coordinator(
            monkeypatch, tmp_path, p_client, d_client
        )

        outputs = coordinator.generate(
            "completion",
            b'{"prompt":"hello","max_tokens":2}',
            "request-cancel",
        )
        first = await anext(outputs)
        assert first.output_sequence == 1
        await outputs.aclose()

        assert p_client.abort_calls == 1
        assert d_client.abort_calls == 1
        assert journal.unresolved == ()
        with open(journal.path, encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream]
        assert records[-1]["event"] == "RECOVERY_REQUIRED"
        assert records[-1]["error_code"] == "GeneratorExit"
        journal.close()

    asyncio.run(exercise())


def test_router_decode_eof_is_recovery_required_and_aborts_pair(
    monkeypatch, tmp_path
) -> None:
    async def exercise() -> None:
        waiting = asyncio.Event()
        p_client = _PrefillClient(waiting)
        d_client = _EOFDecodeClient(waiting)
        coordinator, journal = _coordinator(
            monkeypatch, tmp_path, p_client, d_client
        )

        with pytest.raises(EOFError, match="without a terminal output"):
            _ = [
                output
                async for output in coordinator.generate(
                    "completion",
                    b'{"prompt":"hello","max_tokens":2}',
                    "request-eof",
                )
            ]
        assert p_client.abort_calls == 1
        assert d_client.abort_calls == 1
        with open(journal.path, encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream]
        assert records[-1]["event"] == "RECOVERY_REQUIRED"
        assert records[-1]["error_code"] == "EOFError"
        journal.close()

    asyncio.run(exercise())


def test_router_reservation_failure_is_definite_and_aborts_without_prefill(
    monkeypatch, tmp_path
) -> None:
    async def exercise() -> None:
        waiting = asyncio.Event()
        p_client = _PrefillClient(waiting)
        d_client = _ReserveFailureDecodeClient(waiting)
        coordinator, journal = _coordinator(
            monkeypatch, tmp_path, p_client, d_client
        )

        with pytest.raises(RuntimeError, match="capacity unavailable"):
            _ = [
                output
                async for output in coordinator.generate(
                    "completion",
                    b'{"prompt":"hello","max_tokens":2}',
                    "request-reserve-failure",
                )
            ]
        assert p_client.execute_calls == 0
        assert p_client.abort_calls == 1
        assert d_client.abort_calls == 1
        with open(journal.path, encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream]
        assert records[-1]["event"] == "HANDOFF_FAILED"
        assert records[-1]["error_code"] == "RuntimeError"
        journal.close()

    asyncio.run(exercise())


def test_router_reselects_decode_after_retryable_reservation_rejection(
    tmp_path,
) -> None:
    async def exercise() -> None:
        waiting = asyncio.Event()
        p_client = _PrefillClient(waiting, "p1")
        rejected = _RejectDecodeClient(waiting, "d1")
        accepted = _DecodeClient(waiting, "d2")
        config = RouterConfig(
            prefill_urls=("http://p1",),
            decode_urls=("http://d1", "http://d2"),
            run_id="run",
            policy="round_robin",
            provider="mooncake",
            journal_path=str(tmp_path / "router-retry.jsonl"),
            log_dir=str(tmp_path / "logs"),
        )
        journal = RouterJournal(config.journal_path, config.run_id)
        coordinator = RouterCoordinator(
            config,
            WorkerDirectory(
                (p_client,),
                (rejected, accepted),
                "run",
                RoundRobinRoutePolicy(),
            ),
            journal,
        )

        outputs = [
            output
            async for output in coordinator.generate(
                "completion",
                b'{"prompt":"hello","max_tokens":2}',
                "request-retry",
            )
        ]

        assert outputs[-1].finished
        assert rejected.reserve_calls == 1
        assert p_client.execute_calls == 1
        with open(journal.path, encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream]
        rejection = next(
            record for record in records if record["event"] == "RESERVATION_REJECTED"
        )
        reserved = next(
            record for record in records if record["event"] == "HANDOFF_RESERVED"
        )
        assert (rejection["prefill_node_id"], rejection["decode_node_id"]) == (
            "p1",
            "d1",
        )
        assert (reserved["prefill_node_id"], reserved["decode_node_id"]) == (
            "p1",
            "d2",
        )
        assert journal.unresolved == ()
        journal.close()

    asyncio.run(exercise())


def test_router_restart_queries_both_owners_and_fences_unresolved_handoff(
    monkeypatch, tmp_path
) -> None:
    async def exercise() -> None:
        waiting = asyncio.Event()
        p_client = _PrefillClient(waiting)
        d_client = _DecodeClient(waiting)
        coordinator, journal = _coordinator(
            monkeypatch, tmp_path, p_client, d_client
        )
        key = HandoffKey("request", "handoff", 1, 1, 1)
        journal.append(
            "HANDOFF_CREATED",
            key,
            prefill_node_id="prefill",
            decode_node_id="decode",
            prefill_endpoint_generation=1,
            decode_endpoint_generation=1,
        )

        await coordinator.reconcile_startup()

        assert journal.unresolved == ()
        with open(journal.path, encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream]
        assert records[-1]["event"] == "RECOVERY_REQUIRED"
        assert records[-1]["error_code"] == "STATUS_STATUS"
        journal.close()

    asyncio.run(exercise())


def test_router_restart_queries_only_the_journaled_multi_node_owners(
    tmp_path,
) -> None:
    async def exercise() -> None:
        waiting = asyncio.Event()
        p1 = _PrefillClient(waiting, "p1")
        p2 = _PrefillClient(waiting, "p2")
        d1 = _DecodeClient(waiting, "d1")
        d2 = _DecodeClient(waiting, "d2")
        config = RouterConfig(
            prefill_urls=("http://p1", "http://p2"),
            decode_urls=("http://d1", "http://d2"),
            run_id="run",
            policy="round_robin",
            provider="mooncake",
            journal_path=str(tmp_path / "router-recovery.jsonl"),
            log_dir=str(tmp_path / "logs"),
        )
        journal = RouterJournal(config.journal_path, config.run_id)
        coordinator = RouterCoordinator(
            config,
            WorkerDirectory(
                (p1, p2),
                (d1, d2),
                "run",
                RoundRobinRoutePolicy(),
            ),
            journal,
        )
        key = HandoffKey("request-bound", "handoff-bound", 1, 1, 1)
        journal.append(
            "HANDOFF_CREATED",
            key,
            prefill_node_id="p2",
            decode_node_id="d2",
            prefill_endpoint_generation=1,
            decode_endpoint_generation=1,
        )

        await coordinator.reconcile_startup()

        assert (p1.query_calls, p2.query_calls) == (0, 1)
        assert (d1.query_calls, d2.query_calls) == (0, 1)
        assert journal.unresolved == ()
        journal.close()

    asyncio.run(exercise())


def test_router_journal_reopens_with_the_latest_reservation_attempt_binding(
    tmp_path,
) -> None:
    path = tmp_path / "router-reopen.jsonl"
    key = HandoffKey("request-reopen", "handoff-reopen", 1, 1, 1)
    journal = RouterJournal(str(path), "run")
    journal.append(
        "HANDOFF_CREATED",
        key,
        prefill_node_id="p1",
        decode_node_id="d1",
        prefill_endpoint_generation=1,
        decode_endpoint_generation=1,
    )
    journal.append(
        "RESERVATION_ATTEMPT",
        key,
        prefill_node_id="p1",
        decode_node_id="d2",
        prefill_endpoint_generation=1,
        decode_endpoint_generation=3,
    )
    journal.close()

    reopened = RouterJournal(str(path), "run")
    assert len(reopened.unresolved_bindings) == 1
    binding = reopened.unresolved_bindings[0]
    assert (
        binding.key,
        binding.prefill_node_id,
        binding.decode_node_id,
        binding.prefill_endpoint_generation,
        binding.decode_endpoint_generation,
    ) == (
        key,
        "p1",
        "d2",
        1,
        3,
    )
    reopened.close()
