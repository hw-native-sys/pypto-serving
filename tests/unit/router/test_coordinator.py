# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host transaction tests over public node JSON/SSE, without a model runtime."""

import asyncio
import json
from types import SimpleNamespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

import anyio
import msgspec
import pytest

from pypto_serving.router.client import NodeClient, PublicResult
from pypto_serving.router.config import RouterConfig
from pypto_serving.router.coordinator import RouterCoordinator, sse_document
from pypto_serving.router.directory import WorkerDirectory
from pypto_serving.router.journal import RouterJournal
from pypto_serving.router.policy import RoundRobinRoutePolicy
from pypto_serving.router.recovery import RecoveryPhase
from pypto_serving.serving.pd.config import PDCapabilities, PDRole
from pypto_serving.serving.pd.protocol import CapacitySnapshot, NodeDescriptor
from pypto_serving.serving.pd.generation import parse_generation
from pypto_serving.serving.pd.protocol import CapabilityWire, HandoffKey, HandoffStatus, PrefillResult


def _descriptor(role, node_id="", *, generation=1, control_incarnation=1):
    node_id = node_id or role.value
    capabilities = PDCapabilities(
        adapter_id="host.adapter", contract_version=1, contract_digest="a" * 64,
        continuation_schema="host-v1", model_revision="model", registry_fingerprint="r" * 64,
        layout_fingerprint="l" * 64, topology=(1,), logical_groups=("kv",), physical_regions=("kv",),
    )
    return NodeDescriptor(node_id, role.value, "run", "localhost", 12345, generation, generation,
                          control_incarnation, CapabilityWire.from_capabilities(capabilities), "READY")


def _frame(request_id, *, text="", finish=None, usage=None):
    value = {"id": request_id, "object": "text_completion", "choices": [
        {"index": 0, "text": text, "finish_reason": finish},
    ]}
    if usage is not None:
        value["choices"] = []
        value["usage"] = usage
    return ("data: " + json.dumps(value) + "\n\n").encode()


class _Response:
    status = 200
    headers = {"content-type": "text/event-stream"}
    def __init__(self, value=None, frames=()):
        self.value, self.values = value, frames
        self.closed = False
        self.pause = None

    async def json(self):
        return self.value

    async def result(self):
        return PublicResult(msgspec.json.encode(self.value), {"x-pypto-token-ids-sha256": "hash"})

    async def frames(self):
        for index, frame in enumerate(self.values):
            if self.pause and index == 1:
                await self.pause.wait()
            if isinstance(frame, BaseException):
                raise frame
            yield frame

    async def aclose(self):
        self.closed = True


class _Client:
    def __init__(self, role, peer_seen, *, node_id="", generation=1, control_incarnation=1):
        self.role, self.peer_seen = role, peer_seen
        self.descriptor = _descriptor(role, node_id, generation=generation, control_incarnation=control_incarnation)
        self.node_id = self.descriptor.node_id
        self.calls, self.aborts, self.responses, self.query_calls = [], [], [], 0
        self.reject = None
        self.open_error = None
        self.source_pause = None
        self.frames_error = None
        self.output_pause = None
        self.bad_binding = False
        self.safe_abort = True
        self.abort_pause = None

    async def get(self, path, type_):
        if path.endswith("/descriptor"):
            return self.descriptor
        if path.endswith("/capacity"):
            return CapacitySnapshot(self.node_id, self.role.value, 0, 0, 0, 0, 1, 8)
        raise AssertionError(path)

    async def post(self, path, payload, type_=None):
        if path.endswith("/abort"):
            if self.abort_pause:
                await self.abort_pause.wait()
            await asyncio.sleep(0)
            self.aborts.append(payload)
            return HandoffStatus(payload.key, "ABORTED" if self.safe_abort else "RECOVERY_REQUIRED")
        if path.endswith("/query"):
            self.query_calls += 1
            return HandoffStatus(payload.key, "TRANSFERRING")
        raise AssertionError("obsolete generation HTTP call: " + path)

    async def open_generation(self, path, body):
        kind = "chat" if "chat" in path else "completion"
        public, context = parse_generation(body, kind)
        self.calls.append((path, json.loads(body), context))
        if self.open_error:
            raise self.open_error
        if self.role is PDRole.PREFILL:
            await self.peer_seen.wait()
            if self.source_pause:
                await self.source_pause.wait()
            result = PrefillResult(context.key, "b" * 64 if self.bad_binding else context.request_fingerprint,
                                   "NOT_STARTED" if self.reject else "READY",
                                   self.reject[0] if self.reject else "", self.reject[1] if self.reject else False)
            response = _Response({"pd_transfer": msgspec.to_builtins(result)})
        else:
            self.peer_seen.set()
            await asyncio.sleep(0)
            usage = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
            values = [
                _frame(context.key.request_id, text="first"),
                _frame(context.key.request_id, text=" second", finish="length"),
                _frame(context.key.request_id, usage=usage), b"data: [DONE]\n\n",
            ]
            if self.frames_error:
                values = values[:1] + [self.frames_error]
            if public.stream:
                response = _Response(frames=values)
                response.pause = self.output_pause
            else:
                response = _Response({"id": context.key.request_id, "object": "text_completion",
                                      "choices": [{"text": "first second", "finish_reason": "length"}], "usage": usage})
                if self.frames_error:
                    async def fail():
                        raise self.frames_error
                    response.result = fail
        self.responses.append(response)
        return response


def _coordinator(tmp_path, p, d, runtime_manager=None):
    config = RouterConfig(("http://prefill",), ("http://decode",), "run", "round_robin", "mooncake",
                          str(tmp_path / "router.jsonl"), str(tmp_path / "logs"))
    journal = RouterJournal(config.journal_path, config.run_id)
    return RouterCoordinator(
        config, WorkerDirectory((p,), (d,), "run", RoundRobinRoutePolicy()),
        journal, runtime_manager,
    ), journal


def _clients():
    seen = asyncio.Event()
    return _Client(PDRole.PREFILL, seen), _Client(PDRole.DECODE, seen)


@pytest.mark.parametrize("stream", (False, True))
def test_exactly_two_parallel_posts_preserve_public_semantics_and_terminal_cleanup(tmp_path, stream):
    async def exercise():
        p, d = _clients()
        coordinator, journal = _coordinator(tmp_path, p, d)
        raw = msgspec.json.encode({"prompt": [1, 2, 3], "stream": stream,
                                  "max_tokens": 9, "stop": [], "top_k": None})
        output = [out async for out in coordinator.generate("completion", raw, "request")]
        assert len(p.calls) == len(d.calls) == 1
        assert p.calls[0][0] == d.calls[0][0] == "/v1/completions"
        assert p.calls[0][1] == d.calls[0][1]
        for key in ("top_k", "stop", "max_tokens"):
            assert p.calls[0][1][key] == json.loads(raw)[key]
        assert "temperature" not in p.calls[0][1]
        assert all(response.closed for client in (p, d) for response in client.responses)
        assert coordinator.admission.count == coordinator.requests.count == 0
        assert not journal.unresolved
        if stream:
            assert output[-1] == b"data: [DONE]\n\n"
            assert "".join(value["choices"][0]["text"] for value in map(sse_document, output)
                           if value and value["choices"]) == "first second"
        else:
            assert json.loads(output[0].body)["choices"][0]["text"] == "first second"
            assert output[0].headers["x-pypto-token-ids-sha256"] == "hash"
        journal.close()
    asyncio.run(exercise())


def test_prepared_headers_and_output_do_not_wait_for_source_cleanup(tmp_path):
    async def exercise():
        p, d = _clients()
        p.source_pause, d.output_pause = asyncio.Event(), asyncio.Event()
        coordinator, journal = _coordinator(tmp_path, p, d)
        raw = b'{"prompt":"hello","stream":true}'
        stream = await asyncio.wait_for(coordinator.start_stream("completion", raw, "request"), 1)
        assert len(p.calls) == len(d.calls) == 1
        assert coordinator.admission.count == coordinator.requests.count == 1
        assert b"first" in await asyncio.wait_for(anext(stream), 1)
        assert coordinator.admission.count == 1
        p.source_pause.set()
        await asyncio.sleep(0.01)
        assert coordinator.admission.count == 0 and coordinator.requests.count == 1
        await stream.aclose()
        assert coordinator.requests.count == 0
        assert all(abort.deterministic for client in (p, d) for abort in client.aborts)
        assert coordinator.recovery.phase is RecoveryPhase.RUNNING
        journal.close()
    asyncio.run(exercise())


def test_decode_terminal_waits_for_source_ready_and_budget_cleanup(tmp_path):
    async def exercise():
        p, d = _clients()
        p.source_pause = asyncio.Event()
        coordinator, journal = _coordinator(tmp_path, p, d)
        stream = await coordinator.start_stream("completion", b'{"prompt":"hi","stream":true}', "request")
        assert b"first" in await anext(stream)
        terminal = asyncio.create_task(anext(stream))
        await asyncio.sleep(0.01)
        assert not terminal.done()
        assert coordinator.admission.count == coordinator.requests.count == 1
        p.source_pause.set()
        assert sse_document(await terminal)["choices"][0]["finish_reason"] == "length"
        assert coordinator.admission.count == coordinator.requests.count == 0
        assert all(response.closed for client in (p, d) for response in client.responses)
        assert not journal.unresolved
        await stream.aclose()
        journal.close()
    asyncio.run(exercise())


def test_cancel_after_early_decode_output_waits_for_source_confirmation(tmp_path):
    async def exercise():
        p, d = _clients()
        p.source_pause = asyncio.Event()
        coordinator, journal = _coordinator(tmp_path, p, d)
        stream = await coordinator.start_stream("completion", b'{"prompt":"hi","stream":true}', "request")
        assert b"first" in await anext(stream)
        closing = asyncio.create_task(stream.aclose())
        await asyncio.sleep(0.01)
        assert not closing.done() and not p.aborts and not d.aborts
        assert coordinator.admission.count == coordinator.requests.count == 1
        p.source_pause.set()
        await asyncio.wait_for(closing, 1)
        assert coordinator.requests.count == coordinator.admission.count == 0
        assert all(abort.deterministic for client in (p, d) for abort in client.aborts)
        assert coordinator.recovery.phase is RecoveryPhase.RUNNING
        journal.close()
    asyncio.run(exercise())


def test_source_failure_interrupts_a_waiting_decode_read(tmp_path):
    async def exercise():
        p, d = _clients()
        p.source_pause, d.output_pause = asyncio.Event(), asyncio.Event()
        p.bad_binding = True
        coordinator, journal = _coordinator(tmp_path, p, d)
        stream = await coordinator.start_stream("completion", b'{"prompt":"hi","stream":true}', "request")
        assert b"first" in await anext(stream)
        next_output = asyncio.create_task(anext(stream))
        await asyncio.sleep(0.01)
        assert not next_output.done()
        p.source_pause.set()
        with pytest.raises(RuntimeError, match="mismatched"):
            await asyncio.wait_for(next_output, 1)
        assert coordinator.recovery.phase is RecoveryPhase.RECOVERY_REQUIRED
        assert coordinator.requests.count == coordinator.admission.count == 0
        assert all(response.closed for client in (p, d) for response in client.responses)
        journal.close()
    asyncio.run(exercise())


def test_prepare_rejection_aborts_before_draining_a_real_pending_http_open(tmp_path):
    released = threading.Event()
    async def exercise():
        seen = asyncio.Event()
        loop = asyncio.get_running_loop()
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/v1/completions":
                    loop.call_soon_threadsafe(seen.set)
                    assert released.wait(3), "D needs abort before its headers can complete"
                    status, value = 503, {"error": "aborted"}
                else:
                    assert self.path == "/internal/pd/abort"
                    released.set()
                    status, value = 200, {"key": body["key"], "state": "ABORTED"}
                wire = msgspec.json.encode(value)
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(wire)))
                self.end_headers()
                self.wfile.write(wire)
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        p = _Client(PDRole.PREFILL, seen)
        p.reject = ("ValueError", False)
        d = _Client(PDRole.DECODE, seen)
        real = NodeClient(f"http://127.0.0.1:{server.server_port}", 5)
        d.open_generation, d.post = real.open_generation, real.post
        coordinator, journal = _coordinator(tmp_path, p, d)
        try:
            with pytest.raises(ValueError):
                await asyncio.wait_for(coordinator.start_stream(
                    "completion", b'{"prompt":"hi","stream":true}', "request"), 1)
            assert released.is_set()
            assert coordinator.admission.count == coordinator.requests.count == 0
            assert coordinator.recovery.phase is RecoveryPhase.RUNNING
            assert not journal.unresolved
        finally:
            released.set()
            journal.close()
            await asyncio.to_thread(server.shutdown)
            server.server_close()
            thread.join(2)
    asyncio.run(exercise())


@pytest.mark.parametrize("failure", ("binding", "eof", "socket", "bad_abort"))
def test_unknown_failure_is_not_a_safe_release_or_replay(tmp_path, failure):
    async def exercise():
        p, d = _clients()
        if failure == "binding":
            p.bad_binding = True
        elif failure == "socket":
            p.open_error = OSError("lost source response")
        else:
            d.frames_error = EOFError("lost Decode stream")
            if failure == "bad_abort":
                d.safe_abort = False
        coordinator, journal = _coordinator(tmp_path, p, d)
        with pytest.raises((RuntimeError, EOFError, OSError)):
            [out async for out in coordinator.generate("completion", b'{"prompt":"hi","stream":true}', "request")]
        assert p.aborts and d.aborts
        assert coordinator.recovery.phase is RecoveryPhase.RECOVERY_REQUIRED
        assert coordinator.admission.count == coordinator.requests.count == 0
        assert all(response.closed for client in (p, d) for response in client.responses)
        assert coordinator._replayable["request"].output_published == (failure in ("eof", "bad_abort"))
        journal.close()
    asyncio.run(exercise())


@pytest.mark.parametrize("bad_binding", (False, True))
def test_not_started_proof_is_trusted_only_for_the_bound_request(tmp_path, bad_binding):
    async def exercise():
        p, d = _clients()
        p.reject, p.bad_binding = ("INVALID_INPUT", False), bad_binding
        coordinator, journal = _coordinator(tmp_path, p, d)
        with pytest.raises(RuntimeError):
            await coordinator.start_stream("completion", b'{"prompt":"hi","stream":true}', "request")
        assert all(abort.deterministic == (not bad_binding) for client in (p, d) for abort in client.aborts)
        assert coordinator.recovery.phase is (RecoveryPhase.RECOVERY_REQUIRED if bad_binding else RecoveryPhase.RUNNING)
        journal.close()
    asyncio.run(exercise())


def test_reselection_uses_fresh_attempt_only_after_confirmed_no_compute(tmp_path):
    async def exercise():
        p, d = _clients()
        other = _Client(PDRole.DECODE, d.peer_seen, node_id="decode-other")
        p.reject = ("CAPACITY_EXHAUSTED", True)
        original = p.open_generation
        async def prefill(path, body):
            context = parse_generation(body, "completion")[1]
            p.reject = ("CAPACITY_EXHAUSTED", True) if context.decode_node_id == "decode" else None
            return await original(path, body)
        p.open_generation = prefill
        coordinator, journal = _coordinator(tmp_path, p, d)
        coordinator.directory.decode_clients = (d, other)
        outputs = [out async for out in coordinator.generate("completion", b'{"prompt":"hi"}', "request")]
        assert outputs
        assert len(p.calls) == 2 and len(d.calls) == len(other.calls) == 1
        assert p.calls[0][2].key != p.calls[1][2].key
        assert d.aborts[0].deterministic
        assert coordinator.recovery.phase is RecoveryPhase.RUNNING
        assert not journal.unresolved
        journal.close()
    asyncio.run(exercise())


def test_anyio_cancellation_cannot_interrupt_abort_confirmation(tmp_path):
    async def exercise():
        p, d = _clients()
        coordinator, journal = _coordinator(tmp_path, p, d)
        stream = await coordinator.start_stream("completion", b'{"prompt":"hi","stream":true}', "request")
        await anext(stream)
        with anyio.CancelScope() as scope:
            scope.cancel()
            await stream.aclose()
        assert p.aborts and d.aborts
        assert coordinator.requests.count == 0
        assert coordinator.recovery.phase is RecoveryPhase.RUNNING
        journal.close()
    asyncio.run(exercise())


def test_finish_is_published_after_socket_journal_and_admission_cleanup(tmp_path):
    async def exercise():
        p, d = _clients()
        coordinator, journal = _coordinator(tmp_path, p, d)
        stream = await coordinator.start_stream("completion", b'{"prompt":"hi","stream":true}', "request")
        assert b"first" in await anext(stream)
        terminal = await anext(stream)
        assert sse_document(terminal)["choices"][0]["finish_reason"] == "length"
        assert all(response.closed for client in (p, d) for response in client.responses)
        assert not journal.unresolved and coordinator.requests.count == 0
        await stream.aclose()
        assert not p.aborts and not d.aborts
        journal.close()
    asyncio.run(exercise())


def test_restart_reconciles_only_journaled_owners_and_generation(tmp_path):
    async def exercise():
        p, d = _clients()
        coordinator, journal = _coordinator(tmp_path, p, d)
        key = HandoffKey("interrupted", "handoff", 1, 1, 1)
        journal.append("HANDOFF_CREATED", key, prefill_node_id=p.node_id, decode_node_id=d.node_id,
                       prefill_endpoint_generation=1, decode_endpoint_generation=1)
        await coordinator.reconcile_startup()
        assert p.query_calls == d.query_calls == 1
        assert coordinator.recovery.phase is RecoveryPhase.RECOVERY_REQUIRED
        assert not journal.unresolved
        journal.close()
    asyncio.run(exercise())


def test_automatic_recovery_still_requires_death_barrier_and_unpublished_output(tmp_path):
    async def exercise():
        p, d = _clients()
        p.open_error = OSError("old runtime lost")
        events = []
        coordinator, journal = _coordinator(tmp_path, p, d)
        async def retire(current, reason):
            events.append("retire")
        async def dead(current):
            events.append("dead")
            return True
        async def restart(next_generation):
            events.append("restart")
            p2, d2 = _clients()
            p2.descriptor = _descriptor(PDRole.PREFILL, generation=2, control_incarnation=2)
            d2.descriptor = _descriptor(PDRole.DECODE, generation=2, control_incarnation=2)
            coordinator.directory.prefill_clients, coordinator.directory.decode_clients = (p2,), (d2,)
        async def validate(expected):
            events.append("ready")
            return expected.data_generation == 2
        coordinator.runtime_manager = SimpleNamespace(
            retire=retire, confirm_dead=dead, restart=restart, validate_ready=validate,
        )
        outputs = [out async for out in coordinator.generate("completion", b'{"prompt":"hi"}', "request",
                                                            publish_immediately=False)]
        assert json.loads(outputs[-1].body)["choices"][0]["text"] == "first second"
        assert events == ["retire", "dead", "restart", "ready"]
        assert coordinator.recovery.current.data_generation == 2
        assert coordinator.recovery.phase is RecoveryPhase.RUNNING
        journal.close()
    asyncio.run(exercise())
