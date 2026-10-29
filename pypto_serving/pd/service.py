# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The single P-push handoff path. Cache ownership remains in Serving."""

import asyncio
from collections import OrderedDict
from contextlib import aclosing
from dataclasses import dataclass, field, replace
import logging

import msgspec

from pypto_serving.config.types import GenerateConfig
from pypto_serving.serving.constraints import ConstraintSpec
from pypto_serving.serving.reasoning import OutputParserSpec
from pypto_serving.tools.observe import record, span
from .planner import Geometry, SourcePins, page_indices, physical, plan_chunk, valid_from
from .protocol import (Binding, Completion, Continuation, Node, Placement, Plan, Topology,
                       decode, digest, receive, send)
from .session import PeerPool, Session
from .transfer import TransferError

logger = logging.getLogger(__name__)


class NodeUnavailable(RuntimeError):
    """This node cannot admit new work; existing retirement remains allowed."""


@dataclass
class Request:
    binding: Binding
    http: asyncio.Event = field(default_factory=asyncio.Event)
    prepared: asyncio.Event = field(default_factory=asyncio.Event)
    transferred: asyncio.Event = field(default_factory=asyncio.Event)
    source_done: asyncio.Event = field(default_factory=asyncio.Event)
    cancelled: bool = False
    continuation: object = None
    reservation: object = None
    pending: object = None
    error: Exception | None = None
    chunk: int = 0
    end: int = 0
    p_hit: int = 0
    task: object = None
    response_closed: bool = False
    mapping: tuple = ()
    normalized_digest: str = ""
    state_task: object = None
    state_slots: dict = field(default_factory=dict)
    admitted: bool = False
    worker_release_ack: asyncio.Event = field(default_factory=asyncio.Event)


class Service:
    def __init__(self, config, endpoint, role, adapter, core, progress):
        self.config, self.endpoint, self.role = config, endpoint, role
        self.adapter, self.core, self.progress = adapter, core, progress
        self.cache = core.kv_cache_manager
        self.requests, self.terminal = {}, OrderedDict()
        self.peers = PeerPool(len(config.prefill) + len(config.decode))
        self.pins = SourcePins(self.cache)
        self.chunks = {}
        self.node = None
        self.registry = None
        self.gated = False
        self.closing = False
        self.listener = None
        self.connections = set()
        self.preflight = None

    async def worker(self, operation, value=None):
        payload = msgspec.msgpack.encode(value) if value is not None else b""
        with span(logger, "pd.worker." + operation):
            result = msgspec.msgpack.decode(await self.core.call_worker(operation, payload))
        if operation == "submit":
            if "error" in result:
                raise TransferError(result["error"], uncertain=result["uncertain"])
            return result["job"]
        return result

    def observe(self, name, entry, **fields):
        record(logger, name, request=entry.binding.request, node_id=self.node.node_id,
               boot=self.node.boot, registry=self.node.registry, **fields)

    async def start(self, boot, model):
        self.registry = await self.worker("registry")
        self.geometry = tuple(decode(g, Geometry) for g in self.registry["geometry"])
        contract = self.adapter.contract(model, self.geometry)
        topology = decode(self.registry["topology"], Topology)
        self.node = Node(self.endpoint.node_id, self.role, boot, contract, digest(self.registry),
                         topology, self.config.provider, self.cache.enable_prefix_cache)
        if self.role == "decode":
            self.listener = await asyncio.start_server(self._accept, "0.0.0.0", self.endpoint.control_port)
        logger.info("ready node=%s role=%s boot=%s prefix=%s", self.node.node_id, self.role,
                    boot, self.node.prefix)

    def is_ready(self):
        return (self.node is not None and not self.gated and not self.closing
                and self.core.is_ready())

    def status(self, request):
        """Copy current facts without advancing execution or querying a native owner."""
        entry = self.requests.get(request)
        reservation = self.cache.group_cache_reservation(request) if self.role == "decode" else None
        node = self.node
        result = dict(node=dict(node_id=self.endpoint.node_id, role=self.role,
                                boot=node.boot if node is not None else None,
                                registry=node.registry if node is not None else None,
                                ready=self.is_ready()), request=request, entry_present=entry is not None,
                      terminal_binding_present=request in self.terminal, execution=None, handoff=None,
                      reservation=None, source_pin=request in self.pins.active if self.role == "prefill" else None,
                      worker=None)
        if self.role == "decode":
            result["reservation"] = dict(present=reservation is not None, state=None)
        if reservation is not None:
            result["reservation"].update(state=reservation.state.value,
                reservation_id=reservation.reservation_id, request_id=reservation.request_id,
                write_authorized=reservation.write_authorized, prompt_tokens=reservation.prompt_token_count,
                token_capacity=reservation.token_capacity, prefix_hit_tokens=reservation.prefix_hit_tokens,
                published_block_counts=dict(reservation.published_block_counts),
                publication_valid_from=dict(reservation.publication_valid_from))
        if entry is not None:
            error = entry.error
            uncertain = isinstance(error, TransferError) and error.uncertain
            result["execution"] = dict(cancelled=entry.cancelled, response_closed=entry.response_closed,
                error=dict(type=type(error).__name__, uncertain=uncertain) if error is not None else None,
                source_stop_confirmed=entry.source_done.is_set() if self.role == "decode" else None)
            pending = entry.pending
            result["handoff"] = dict(confirmed_chunks=entry.chunk, confirmed_end=entry.end,
                confirmation_source="owner_completion" if self.role == "prefill" else "sender_completion",
                uncertain=uncertain, pending_plan=(dict(plan_digest=digest(pending), chunk=pending.chunk,
                    start=pending.start, end=pending.end, final=pending.final) if pending is not None else None))
            result["worker"] = dict(state_reservation_confirmed=bool(entry.state_slots)
                if self.role == "decode" else None, state_release_confirmed=entry.worker_release_ack.is_set()
                if self.role == "decode" else None)
        return result

    def entry(self, binding, *, create=False):
        if self.node is None:
            raise NodeUnavailable("PD node is unavailable")
        binding.check(self.node, self.role)
        if binding.request in self.terminal:
            raise ValueError("request already terminal")
        if create and not self.is_ready():
            raise NodeUnavailable("PD node is unavailable")
        entry = self.requests.get(binding.request)
        if entry is None:
            if not create:
                raise ValueError("unknown PD request")
            if len(self.requests) >= self.config.capacity:
                raise ValueError("PD request capacity exhausted")
            entry = self.requests[binding.request] = Request(binding)
        if entry.binding != binding:
            raise ValueError("request binding changed")
        return entry

    def register_http(self, binding, payload):
        if digest(payload) != binding.input_digest:
            raise ValueError("HTTP input differs from routed request")
        entry = self.entry(binding, create=True)
        if entry.cancelled:
            raise ValueError("request cancelled")
        if entry.http.is_set():
            raise ValueError("duplicate generation POST")
        entry.http.set()
        return entry

    def remember_terminal(self, binding):
        previous = self.terminal.get(binding.request)
        if previous is not None and previous != binding:
            raise ValueError("terminal binding mismatch")
        self.terminal[binding.request] = binding
        self.terminal.move_to_end(binding.request)
        while len(self.terminal) > self.config.capacity * 8:
            self.terminal.popitem(last=False)

    async def cancel_binding(self, binding):
        binding.check(self.node, self.role)
        entry = self.requests.get(binding.request)
        if entry is None:
            # An abort can overtake both the generation POST and control begin.
            # Keep a bounded tombstone without allocating an active request slot.
            self.remember_terminal(binding)
        else:
            if entry.binding != binding:
                raise ValueError("abort binding mismatch")
            await self.abort(entry)

    def finish(self, entry):
        if self.requests.pop(entry.binding.request, None) is None:
            return
        self.remember_terminal(entry.binding)
        self.observe("finished", entry, cancelled=entry.cancelled, error=entry.error,
                     p_hit=entry.p_hit, d_hit=entry.reservation.prefix_hit_tokens
                     if entry.reservation is not None else 0)

    async def connect(self, node):
        endpoint = self.config.endpoint("decode", node.node_id)
        reader, writer = await asyncio.open_connection(endpoint.host, endpoint.control_port)
        try:
            await send(writer, dict(node=self.node, registry=self.registry))
            hello = await receive(reader)
            peer = decode(hello["node"], Node)
            if (peer != node or not self.node.compatible(peer) or digest(hello["registry"]) != peer.registry
                    or decode(hello["registry"]["topology"], Topology) != peer.topology):
                raise ValueError("peer identity/contract changed before handshake")
            await self.worker("install", hello["registry"])
            session = Session(reader, writer, self.config.capacity * 2)
            session.geometry = tuple(decode(g, Geometry) for g in hello["registry"]["geometry"])
            return session
        except BaseException:
            writer.close()
            await writer.wait_closed()
            raise

    async def _accept(self, reader, writer):
        self.connections.add(writer)
        tasks = set()
        write_lock = asyncio.Lock()
        slots = asyncio.Semaphore(self.config.capacity * 2)
        try:
            hello = await asyncio.wait_for(receive(reader), self.config.handoff_timeout)
            peer = decode(hello["node"], Node)
            self.config.endpoint("prefill", peer.node_id)
            if not peer.compatible(self.node):
                raise ValueError("incompatible Prefill peer")
            geometry = tuple(decode(g, Geometry) for g in hello["registry"]["geometry"])
            if (digest(hello["registry"]) != peer.registry
                    or decode(hello["registry"]["topology"], Topology) != peer.topology):
                raise ValueError("peer registry fingerprint mismatch")
            await send(writer, dict(node=self.node, registry=self.registry))
            async def dispatch(message):
                try:
                    async with slots:
                        value = await self.control(peer, geometry, message["op"], message["payload"])
                    reply = dict(id=message["id"], result=value)
                except Exception as exc:
                    reply = dict(id=message["id"], error=str(exc))
                async with write_lock:
                    await send(writer, reply)
            while True:
                message = await receive(reader)
                if len(tasks) >= self.config.capacity * 2:
                    await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                task = asyncio.create_task(dispatch(message))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
        except (EOFError, asyncio.IncompleteReadError, ConnectionError):
            pass
        except Exception:
            logger.exception("PD control connection failed")
        finally:
            await asyncio.gather(*tasks, return_exceptions=True)
            # A connection close cannot prove that a remote native write stopped.
            for entry in tuple(self.requests.values()):
                if 'peer' in locals() and entry.binding.prefill == peer.key and not entry.source_done.is_set():
                    self.fail(entry, TransferError("peer control lost", uncertain=True))
            writer.close()
            await writer.wait_closed()
            self.connections.discard(writer)

    async def control(self, peer, geometry, operation, payload):
        binding = decode(payload["binding"], Binding)
        if binding.prefill != peer.key:
            raise ValueError("control sender differs from binding")
        if operation == "abort" and payload.get("stopped") is True:
            binding.check(self.node, self.role)
            if binding.request not in self.requests:
                # Never admitted (including a capacity rejection): no write
                # could be authorized, and abort must not allocate a new slot.
                self.remember_terminal(binding)
                return {}
        entry = self.entry(binding, create=operation == "begin")
        request = binding.request
        if operation == "begin":
            if entry.continuation is not None or entry.cancelled:
                raise ValueError("handoff already begun")
            http = asyncio.create_task(entry.http.wait())
            stopped = asyncio.create_task(entry.source_done.wait())
            try:
                done, _ = await asyncio.wait((http, stopped), timeout=self.config.handoff_timeout,
                                             return_when=asyncio.FIRST_COMPLETED)
                if not done:
                    raise TimeoutError("generation POST did not arrive")
                if entry.cancelled or not entry.http.is_set():
                    raise ValueError("request cancelled before generation POST")
            finally:
                for task in (http, stopped):
                    task.cancel()
                await asyncio.gather(http, stopped, return_exceptions=True)
            continuation = decode(payload["continuation"], Continuation)
            config = GenerateConfig(**continuation.config)
            self.adapter.validate(config)
            prefix = self.adapter.prefix_spec(continuation.tokens, self.cache)
            if prefix != continuation.prefix or not continuation.tokens or config.max_new_tokens <= 0:
                raise ValueError("continuation Prefix specification mismatch")
            if len(continuation.tokens) + config.max_new_tokens > self.core.config.runtime_config.max_seq_len:
                raise ValueError("request exceeds model context capacity")
            entry.continuation = continuation
            entry.normalized_digest = digest(continuation)
            try:
                if not self.is_ready():
                    raise NodeUnavailable("PD node is unavailable")
                reservation = self.cache.reserve_group_cache(
                    request, request, len(continuation.tokens) + config.max_new_tokens,
                    prompt_token_count=len(continuation.tokens),
                    prefix_block_hashes={k: [bytes.fromhex(h) for h in hashes]
                                         for k, hashes in prefix.hashes.items()},
                    shareable_group_names=tuple(prefix.hashes))
                entry.reservation = reservation
                entry.state_task = asyncio.create_task(self.worker("reserve_state", dict(
                    request=request, partition=reservation.partition, prompt_len=len(continuation.tokens))))
                entry.state_slots = await asyncio.shield(entry.state_task)
                if entry.cancelled:
                    raise ValueError("request cancelled during state reservation")
                entry.prepared.set()
                self.observe("reservation", entry, state=reservation.state.value,
                             partition=reservation.partition, d_hit=reservation.prefix_hit_tokens)
                return Placement(request, reservation.prefix_hit_tokens, reservation.partition,
                                 reservation.block_ids_by_group, entry.state_slots)
            except Exception as exc:
                self.fail(entry, exc)
                raise
        if operation == "plan":
            if not self.is_ready():
                raise NodeUnavailable("PD node is unavailable")
            if entry.cancelled:
                raise ValueError("cancelled request cannot authorize new writes")
            plan = decode(payload["plan"], Plan)
            p_hit = payload["p_hit"]
            if (type(p_hit) is not int or not 0 <= p_hit <= plan.end
                    or p_hit % entry.continuation.prefix.alignment or (entry.chunk and p_hit != entry.p_hit)):
                raise ValueError("invalid source Prefix boundary")
            self.validate_plan(entry, plan, geometry, p_hit, peer.topology)
            self.cache.authorize_group_cache_write(request)
            entry.pending = plan
            entry.p_hit = p_hit
            entry.mapping = plan.mapping
            plan_digest = digest(plan)
            self.observe("plan.authorized", entry, plan_digest=plan_digest, chunk=plan.chunk,
                         start=plan.start, end=plan.end, final=plan.final)
            return plan_digest
        if operation == "complete":
            plan = entry.pending
            if plan is None:
                raise ValueError("no authorized chunk")
            completion = decode(payload["completion"], Completion)
            completion.check(plan)
            entry.pending = None
            entry.chunk += 1
            entry.end = plan.end
            self.observe("completion", entry, plan_digest=completion.plan_digest, end=entry.end)
            if plan.final:
                self.cache.commit_group_cache(request, valid_from=valid_from(
                    self.cache.group_specs, self.node.contract.components,
                    entry.p_hit, entry.reservation.prefix_hit_tokens))
                self.observe("commit", entry, plan_digest=completion.plan_digest, end=entry.end,
                             state=self.cache.group_cache_reservation(request).state.value)
                try:
                    if entry.continuation.constraint is not None:
                        await self.preflight(ConstraintSpec.from_wire(entry.continuation.constraint),
                                             (plan.first_token,))
                    entry.first_token = plan.first_token
                    entry.transferred.set()
                except Exception as exc:
                    self.fail(entry, exc)
                    return dict(error=str(exc))
            return {}
        if operation == "source_done":
            entry.source_done.set()
            self.observe("source_done", entry)
            if entry.cancelled or entry.error is not None:
                await self.abort(entry, stopped=True)
            return {}
        if operation == "abort":
            stopped = payload["stopped"]
            if not stopped:
                self.fail(entry, TransferError("sender cannot prove native stop", uncertain=True))
            await self.abort(entry, stopped=stopped)
            return {}
        raise ValueError("unknown handoff operation")

    def validate_plan(self, entry, plan, geometry, p_hit, source_topology):
        if (entry.reservation is None or entry.pending is not None or plan.binding != entry.binding
                or plan.continuation_digest != entry.normalized_digest
                or plan.chunk != entry.chunk or plan.start != entry.end
                or not plan.start < plan.end <= len(entry.continuation.tokens)
                or plan.final != (plan.end == len(entry.continuation.tokens))
                or (plan.final and plan.first_token is None)):
            raise ValueError("invalid chunk sequence or final boundary")
        topology = self.node.topology
        self.adapter.validate_mapping(source_topology, topology, entry.reservation, plan.mapping)
        if entry.mapping and entry.mapping != plan.mapping:
            raise ValueError("rank mapping changed during a request")
        expected = {(rank, c.name) for rank in topology.groups[entry.reservation.partition]
                    for c in self.node.contract.components if plan.final or not c.final}
        if set(plan.units) != expected or len(plan.units) != len(expected):
            raise ValueError("missing required rank/region completion units")
        geo = {g.name: g for g in self.geometry}
        sources = {g.name: g for g in geometry}
        component_group = {c.name: c.group for c in self.node.contract.components}
        occupied = set()
        for copy in plan.copies:
            if ((copy.destination_rank, copy.component) not in expected
                    or not 0 <= copy.source_rank < source_topology.ranks):
                raise ValueError("copy outside rank/component mapping")
            if (copy.source_rank, copy.destination_rank) not in plan.mapping:
                raise ValueError("copy outside request rank mapping")
            g, src = geo[copy.component], sources[copy.component]
            if (g.layout != src.layout or min(copy.source_offset, copy.destination_offset) < 0
                    or copy.length <= 0 or copy.length % g.page_bytes
                    or copy.destination_offset % g.page_bytes or copy.source_offset % src.page_bytes
                    or copy.destination_offset + copy.length > g.extent
                    or copy.source_offset + copy.length > src.extent):
                raise ValueError("copy outside compatible physical layout")
            group_name = component_group[copy.component]
            writable = (entry.reservation.writable_block_ids_by_group[group_name]
                        if group_name in entry.reservation.writable_block_ids_by_group
                        else (entry.state_slots[group_name],))
            for page in range(copy.destination_offset // g.page_bytes,
                              (copy.destination_offset + copy.length) // g.page_bytes):
                key = copy.destination_rank, copy.component, page
                if page % g.blocks not in writable or key in occupied:
                    raise ValueError("write aliases shared Prefix or another write")
                occupied.add(key)
        required = set()
        groups = {group.name: group for group in self.cache.group_specs}
        for _, rank in plan.mapping:
            for component in self.node.contract.components:
                if component.final and not plan.final:
                    continue
                g = geo[component.name]
                if component.group in groups:
                    group = groups[component.group]
                    table = entry.reservation.block_ids_by_group[component.group]
                    blocks = (physical(group, table, logical) for logical, _ in page_indices(
                        group, component, plan.start, plan.end, plan.final,
                        table, p_hit, entry.reservation.prefix_hit_tokens))
                else:
                    blocks = (entry.state_slots[component.group],)
                for block in blocks:
                    required.update((rank, component.name, layer * g.blocks + block)
                                    for layer in range(len(g.layers)))
        if occupied != required:
            raise ValueError("copy plan does not cover the required destination pages")

    def fail(self, entry, error):
        entry.error = error
        entry.prepared.set()
        entry.transferred.set()
        if isinstance(error, TransferError) and error.uncertain:
            self.gated = True
            if entry.reservation is not None:
                self.cache.quarantine_group_cache(entry.binding.request)
        self.observe("request.error", entry, error=type(error).__name__,
                     uncertain=isinstance(error, TransferError) and error.uncertain)

    async def abort(self, entry, stopped=False):
        entry.cancelled = True
        if self.role == "prefill":
            return
        if stopped:
            entry.source_done.set()
        entry.prepared.set()
        entry.transferred.set()
        reservation = self.cache.group_cache_reservation(entry.binding.request)
        if reservation is None or not reservation.write_authorized:
            stopped = True
            entry.source_done.set()
        if reservation is not None:
            if (reservation.write_authorized and not stopped
                    and (entry.pending is not None or entry.end < reservation.prompt_token_count)):
                # Wait for the sender's definitive completion/stop, not an HTTP disconnect.
                return
            if entry.binding.request in self.core.scheduler.requests:
                await self.core.abort_request(entry.binding.request)
            else:
                self.cache.release_group_cache(entry.binding.request, confirmed_stopped=stopped)
                self.cache.retire_group_cache_reservation(entry.binding.request)
        if not entry.admitted:
            await self.discard_state(entry)
        if (entry.response_closed or not entry.http.is_set()) and entry.source_done.is_set():
            self.finish(entry)

    async def discard_state(self, entry):
        if entry.state_task is not None:
            # Even a failed/lost reserve reply may have allocated a local slot.
            # Drain it before the idempotent discard; cancellation must not race reuse.
            await asyncio.shield(asyncio.gather(entry.state_task, return_exceptions=True))
            # Normal Decode already released the lease in its FIFO StepCommand.
            # Early first-token termination/cancellation still needs a barrier.
            if not entry.worker_release_ack.is_set():
                await self.worker("discard_state", entry.binding.request)
                entry.worker_release_ack.set()
            entry.state_task = None

    async def prefill(self, entry, continuation, peer):
        entry.continuation = continuation
        entry.normalized_digest = digest(continuation)
        request = entry.binding.request
        config = GenerateConfig(**continuation.config)
        self.adapter.validate(config)
        queue = self.chunks[request] = asyncio.Queue(maxsize=2)
        async def consume():
            async for _ in self.core.add_request(request, "", config, prompt_token_ids=continuation.tokens,
                                                  constraint_spec=(ConstraintSpec.from_wire(continuation.constraint)
                                                                   if continuation.constraint else None)):
                pass
        drain = None
        inflight = None
        submitted = False
        async with self.peers.use(peer.key, lambda: self.connect(peer)) as session:
            async def call(op, **payload):
                return await asyncio.wait_for(session.call(op, dict(binding=entry.binding, **payload)),
                                              self.config.handoff_timeout)
            try:
                placement = decode(await call("begin", continuation=continuation), Placement)
                peer_geometry = session.geometry
                drain = asyncio.create_task(consume())
                p_hit = None
                while True:
                    chunk_task = asyncio.create_task(queue.get())
                    done, _ = await asyncio.wait((chunk_task, drain), timeout=self.config.handoff_timeout,
                                                 return_when=asyncio.FIRST_COMPLETED)
                    if chunk_task not in done:
                        chunk_task.cancel()
                        await asyncio.gather(chunk_task, return_exceptions=True)
                        if drain in done:
                            await drain
                        raise RuntimeError("Prefill ended before its next confirmed chunk")
                    chunk = chunk_task.result()
                    if p_hit is None:
                        p_hit = chunk.start
                        entry.p_hit = p_hit
                    if inflight is not None:
                        await inflight
                    if entry.cancelled:
                        raise ValueError("request cancelled")
                    if chunk.final:
                        chunk = replace(chunk, state_slots=await self.worker("source_state", request))
                    plan = plan_chunk(entry.binding, chunk, placement, self.node.contract,
                                      self.cache.group_specs, self.geometry, peer_geometry, p_hit,
                                      self.adapter.mapping(self.node.topology, peer.topology,
                                                           chunk.partition, placement.partition),
                                      entry.normalized_digest)
                    self.pins.retain(chunk)
                    entry.pending = plan
                    submitted = False
                    if await call("plan", plan=plan, p_hit=p_hit) != digest(plan):
                        raise ValueError("destination acknowledged a different plan")
                    submitted = True  # Lost ordered-submit replies are uncertain.
                    job = await self.worker("submit", plan)
                    async def settle(plan=plan, job=job):
                        try:
                            completion = await self.progress.wait(job)
                        except TransferError as exc:
                            if not exc.uncertain:
                                self.pins.release(request)
                                await self.worker("forget", job)
                            raise
                        completion.check(plan)
                        entry.chunk, entry.end = plan.chunk + 1, plan.end
                        self.pins.release(request)
                        await self.worker("forget", job)
                        self.observe("completion", entry, plan_digest=job, end=plan.end)
                        response = await call("complete", completion=completion)
                        if response.get("error"):
                            raise ValueError(response["error"])
                        entry.pending = None
                    inflight = asyncio.create_task(settle())
                    if chunk.final:
                        await inflight
                        break
                    self.core.scheduler.resume_request(request)
                self.core.finish_prefilled_request(request)
                await drain
                await call("source_done")
            except BaseException as exc:
                if inflight is not None and not inflight.done():
                    try:
                        await asyncio.shield(inflight)
                    except Exception:
                        pass
                uncertain = (request in self.pins.active and submitted
                             and not (isinstance(exc, TransferError) and not exc.uncertain))
                if uncertain:
                    self.fail(entry, TransferError("source completion unknown", uncertain=True))
                else:
                    self.fail(entry, exc if isinstance(exc, Exception) else ValueError("request cancelled"))
                    if request in self.pins.active:
                        self.pins.release(request)
                    await self.core.abort_request(request)
                    if drain is not None:
                        await drain
                try:
                    await call("abort", stopped=not uncertain)
                except Exception:
                    logger.exception("sender stop confirmation failed request=%s", request)
                    self.gated = True
                raise
            finally:
                if request not in self.pins.active:
                    self.chunks.pop(request, None)
                    self.finish(entry)

    async def outputs(self, request, *args, **kwargs):
        entry = self.requests[request]
        terminal = False
        try:
            try:
                await asyncio.wait_for(entry.transferred.wait(), self.config.request_timeout)
            except asyncio.TimeoutError as exc:
                raise TimeoutError("Prefill data did not arrive within request_timeout") from exc
            if entry.error is not None:
                raise entry.error
            if entry.cancelled:
                raise ValueError("request cancelled")
            continuation = entry.continuation
            config = GenerateConfig(**continuation.config)
            entry.admitted = True
            outputs = self.core.add_prefilled_request(
                reservation_id=request, request_id=request, prompt_token_ids=continuation.tokens,
                first_token=entry.first_token, max_new_tokens=config.max_new_tokens,
                min_tokens=config.min_tokens, ignore_eos=config.ignore_eos,
                temperature=config.temperature, top_p=config.top_p, top_k=config.top_k, seed=config.seed,
                stop_strings=tuple(config.stop), eos_token_id=self.core.tokenizer.eos_token_id,
                stream=config.stream, output_parser_spec=(OutputParserSpec(**continuation.parser)
                                                         if continuation.parser else None),
                constraint_spec=(ConstraintSpec.from_wire(continuation.constraint)
                                 if continuation.constraint else None),
                worker_release_ack=entry.worker_release_ack)
            async with aclosing(outputs):
                async for output in outputs:
                    if output.finished:
                        await asyncio.wait_for(entry.source_done.wait(), self.config.handoff_timeout)
                        terminal = True
                        # The client may disconnect as soon as it sees the
                        # terminal event. Retire before publishing it, so the
                        # Router cannot admit a replacement into a live slot.
                        await outputs.aclose()
                        await self.close_output(entry, terminal)
                    yield output
        finally:
            await self.close_output(entry, terminal)

    async def close_output(self, entry, terminal):
        # D uses its existing task field to own retirement independently of the
        # HTTP cancellation scope. P uses that field for its Prefill task.
        if entry.task is None:
            entry.task = asyncio.create_task(self.retire_output(entry, terminal))
        await asyncio.shield(entry.task)

    async def retire_output(self, entry, terminal):
        entry.admitted = False
        entry.response_closed = True
        if not entry.source_done.is_set():
            await self.abort(entry)
            # Keep the request and lease until the sender proves native stop.
        elif entry.error is None or not isinstance(entry.error, TransferError) or not entry.error.uncertain:
            if not terminal:
                await self.abort(entry, stopped=True)
            reservation = self.cache.group_cache_reservation(entry.binding.request)
            if reservation is not None:
                self.cache.retire_group_cache_reservation(entry.binding.request)
            await self.discard_state(entry)
            self.finish(entry)

    async def close(self):
        self.closing = True
        if self.requests or self.pins.active or self.gated:
            raise RuntimeError("node has active/uncertain handoffs; retire it through external orchestration")
        if self.role == "prefill":
            await self.worker("release")
            # Session close tells D that every sender owner is unregistered.
            await self.peers.close()
        else:
            if self.connections:
                raise RuntimeError("stop Prefill senders before Decode receivers")
            await self.worker("release")
            self.listener.close()
            await self.listener.wait_closed()
