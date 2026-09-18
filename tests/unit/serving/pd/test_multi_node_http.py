# Copyright (c) PyPTO Contributors.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""F7 real-HTTP control-plane test with five lightweight node processes."""

from __future__ import annotations

import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import multiprocessing
import queue
import time

from pypto_serving.router.client import NodeClient
from pypto_serving.router.config import RouterConfig
from pypto_serving.router.coordinator import RouterCoordinator
from pypto_serving.router.directory import WorkerDirectory
from pypto_serving.router.journal import RouterJournal
from pypto_serving.router.policy import RoundRobinRoutePolicy
from pypto_serving.serving.pd.config import PDCapabilities, PDRole
from pypto_serving.serving.pd.http_api import (
    CapacitySnapshot,
    DecodeStreamFrame,
    ExecutePrefillHTTP,
    HandoffHTTP,
    NodeDescriptor,
    PlacementReservation,
    PrefillHandoffResult,
    PrepareRequestHTTP,
    PreparedRequest,
    ReservePlacementHTTP,
    ReservePlacementResult,
    decode_json,
    encode_json,
)
from pypto_serving.serving.pd.protocol import (
    CapabilityWire,
    ContinuationMetadata,
    DecodeOutputWire,
)


def _capabilities(node_id: str) -> CapabilityWire:
    return CapabilityWire.from_capabilities(
        PDCapabilities(
            adapter_id="fake.host.adapter",
            contract_version=1,
            contract_digest="a" * 64,
            continuation_schema="fake-v1",
            model_revision="fake-model",
            registry_fingerprint=(node_id[0] or "x") * 64,
            layout_fingerprint="b" * 64,
            topology=(1,),
            logical_groups=("kv",),
            physical_regions=("kv",),
        )
    )


def _serve_fake_node(node_id, role_value, ready, calls, stop) -> None:
    role = PDRole(role_value)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format, *args) -> None:
            return

        def _send(self, value) -> None:
            wire = encode_json(value)
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(wire)))
            self.end_headers()
            self.wfile.write(wire)

        def _body(self) -> bytes:
            length = int(self.headers.get("content-length", "0"))
            return self.rfile.read(length)

        def do_GET(self) -> None:  # noqa: N802
            port = self.server.server_address[1]
            if self.path == "/internal/pd/descriptor":
                self._send(
                    NodeDescriptor(
                        node_id=node_id,
                        role=role.value,
                        run_id="run",
                        control_host="127.0.0.1",
                        control_port=port,
                        owner_generation=1,
                        endpoint_generation=1,
                        control_incarnation=1,
                        capabilities=_capabilities(node_id),
                        health="READY",
                    )
                )
                return
            if self.path == "/internal/pd/capacity":
                self._send(
                    CapacitySnapshot(
                        node_id=node_id,
                        role=role.value,
                        active_handoffs=0,
                        prepared_requests=0,
                        reservations=0,
                        quarantined_reservations=0,
                        snapshot_sequence=1,
                        active_limit=8,
                    )
                )
                return
            self.send_error(404)

        def do_POST(self) -> None:  # noqa: N802
            body = self._body()
            port = self.server.server_address[1]
            if self.path == "/internal/pd/prepare":
                request = decode_json(body, PrepareRequestHTTP)
                calls.put((request.request_id, "prefill", node_id))
                self._send(
                    PreparedRequest(
                        request_id=request.request_id,
                        prepared_request_id=f"{node_id}-{request.request_id}",
                        prepared_digest="c" * 64,
                        continuation=ContinuationMetadata(
                            prompt_token_ids=(1, 2, 3),
                            max_new_tokens=1,
                            temperature=0.0,
                            top_p=1.0,
                            top_k=None,
                            seed=None,
                            stop_strings=(),
                            eos_token_id=None,
                        ),
                        expires_at_ns=time.time_ns() + 10_000_000_000,
                    )
                )
                return
            if self.path == "/internal/pd/reserve":
                request = decode_json(body, ReservePlacementHTTP)
                calls.put((request.key.request_id, "decode", node_id))
                self._send(
                    ReservePlacementResult(
                        reservation=PlacementReservation(
                            key=request.key,
                            prepared_request_id=request.prepared_request_id,
                            reservation_id=f"{node_id}-{request.key.handoff_id}",
                            partition=0,
                            block_ids_by_group={"kv": (1,)},
                            prepared_digest=request.prepared_digest,
                            reservation_capability="fake-capability",
                            decode_node_id=node_id,
                            decode_control_host="127.0.0.1",
                            decode_control_port=port,
                            decode_endpoint_generation=1,
                        )
                    )
                )
                return
            if self.path == "/internal/pd/authorize":
                self._send({"status": "authorized"})
                return
            if self.path == "/internal/pd/execute":
                request = decode_json(body, ExecutePrefillHTTP)
                self._send(
                    PrefillHandoffResult(
                        key=request.key,
                        reservation_id=request.reservation_id,
                        manifest_hash="d" * 64,
                        state="READY",
                    )
                )
                return
            if self.path == "/internal/pd/await-decode":
                request = decode_json(body, HandoffHTTP)
                frames = (
                    DecodeStreamFrame(event="waiting", state="AUTHORIZED"),
                    DecodeStreamFrame(
                        event="finished",
                        state="COMPLETED",
                        output=DecodeOutputWire(
                            key=request.key,
                            token_id=101,
                            text="ok",
                            finished=True,
                            finish_reason="FINISHED_LENGTH",
                            prompt_tokens=3,
                            completion_tokens=1,
                            token_ids=(101,),
                            output_sequence=1,
                        ),
                    ),
                )
                self.send_response(200)
                self.send_header("content-type", "application/x-ndjson")
                self.end_headers()
                for frame in frames:
                    self.wfile.write(encode_json(frame) + b"\n")
                    self.wfile.flush()
                return
            if self.path in ("/internal/pd/abort", "/internal/pd/query"):
                self._send({"status": "ok"})
                return
            self.send_error(404)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.timeout = 0.1
    ready.put((node_id, server.server_address[1]))
    try:
        while not stop.is_set():
            server.handle_request()
    finally:
        server.server_close()


def test_two_prefill_three_decode_processes_use_formal_http_and_journal(
    tmp_path,
) -> None:
    context = multiprocessing.get_context("spawn")
    ready = context.Queue()
    calls = context.Queue()
    stop = context.Event()
    nodes = (
        ("p1", PDRole.PREFILL),
        ("p2", PDRole.PREFILL),
        ("d1", PDRole.DECODE),
        ("d2", PDRole.DECODE),
        ("d3", PDRole.DECODE),
    )
    processes = [
        context.Process(
            target=_serve_fake_node,
            args=(node_id, role.value, ready, calls, stop),
        )
        for node_id, role in nodes
    ]
    for process in processes:
        process.start()

    journal = None
    try:
        ports = dict(ready.get(timeout=10) for _ in processes)
        prefill_urls = tuple(f"http://127.0.0.1:{ports[node]}" for node in ("p1", "p2"))
        decode_urls = tuple(
            f"http://127.0.0.1:{ports[node]}" for node in ("d1", "d2", "d3")
        )
        config = RouterConfig(
            prefill_urls=prefill_urls,
            decode_urls=decode_urls,
            run_id="run",
            policy="round_robin",
            provider="mooncake",
            journal_path=str(tmp_path / "router.jsonl"),
            log_dir=str(tmp_path / "logs"),
            request_timeout_seconds=5,
        )
        journal = RouterJournal(config.journal_path, config.run_id)
        coordinator = RouterCoordinator(
            config,
            WorkerDirectory(
                tuple(NodeClient(url, 5) for url in prefill_urls),
                tuple(NodeClient(url, 5) for url in decode_urls),
                "run",
                RoundRobinRoutePolicy(),
                control_incarnation=1,
            ),
            journal,
        )

        async def exercise() -> None:
            for index in range(6):
                outputs = [
                    output
                    async for output in coordinator.generate(
                        "completion",
                        b'{"prompt":"hello","max_tokens":1}',
                        f"request-{index}",
                    )
                ]
                assert outputs[-1].finished

        asyncio.run(exercise())

        routed = {}
        for _ in range(12):
            request_id, role, node_id = calls.get(timeout=5)
            routed.setdefault(request_id, {})[role] = node_id
        ordered = [routed[f"request-{index}"] for index in range(6)]
        assert [item["prefill"] for item in ordered] == [
            "p1", "p2", "p1", "p2", "p1", "p2"
        ]
        assert [item["decode"] for item in ordered] == [
            "d1", "d2", "d3", "d1", "d2", "d3"
        ]

        with open(journal.path, encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream]
        bindings = [
            record for record in records if record["event"] == "HANDOFF_RESERVED"
        ]
        assert [record["prefill_node_id"] for record in bindings] == [
            "p1", "p2", "p1", "p2", "p1", "p2"
        ]
        assert [record["decode_node_id"] for record in bindings] == [
            "d1", "d2", "d3", "d1", "d2", "d3"
        ]
        assert journal.unresolved == ()
    finally:
        if journal is not None:
            journal.close()
        stop.set()
        for process in processes:
            process.join(timeout=5)
        for process in processes:
            assert not process.is_alive()
            assert process.exitcode == 0
        while True:
            try:
                calls.get_nowait()
            except queue.Empty:
                break
