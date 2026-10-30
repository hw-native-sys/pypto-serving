# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Optional composition; Serving exposes capabilities, not PD business logic."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field, replace
from functools import partial
import json
import logging
import multiprocessing
import os
from types import SimpleNamespace
import uuid

from fastapi import Request as HTTPRequest
from fastapi.responses import JSONResponse, StreamingResponse
import msgspec

from pypto_serving.serving.engine.async_engine import AsyncLLMEngine, ReplicaEngineCore
from pypto_serving.serving.reasoning import OutputParserSpec
from pypto_serving.serving.sched.scheduler import Scheduler
from pypto_serving.serving.server.server import ChatCompletionRequest, CompletionRequest, ServingServer
from pypto_serving.serving.server.serving_worker import spawn_worker
from .adapter import select_adapter
from .config import Config, log_context
from .protocol import Binding, Continuation, Node, decode
from .service import NodeUnavailable, Service
from .worker import Progress, Services, WorkerSettings


@dataclass(frozen=True)
class Chunk:
    request: str
    chunk: int
    start: int
    end: int
    final: bool
    first_token: int | None
    blocks: dict[str, tuple[int, ...]]
    partition: int
    state_slots: dict[str, int] = field(default_factory=dict)


def prefill_scheduler(*, config, kv_cache_manager):
    return Scheduler(config=replace(config, stop_after_prefill=True), kv_cache_manager=kv_cache_manager)


class PrefillCore(ReplicaEngineCore):
    publish = None

    def _process_step_output(self, output, tokens, drafts=None):
        super()._process_step_output(output, tokens, drafts)
        self.publish(output, tokens)


class Server(ServingServer):
    def __init__(self, service, engine, generate):
        self.service, self.real_engine = service, engine
        facade = SimpleNamespace(config=engine.config, tokenizer=engine.tokenizer, metrics=engine.metrics,
                                 add_request=service.outputs, start_profile=engine.start_profile,
                                 stop_profile=engine.stop_profile)
        super().__init__(facade, engine.config.model_id, generate)
        service.preflight = self._preflight_constraint

    def _register_exception_handlers(self):
        super()._register_exception_handlers()
        self.app.add_exception_handler(NodeUnavailable, self.unavailable)

    @staticmethod
    async def unavailable(request, exc):
        return JSONResponse(dict(object="error", code="pd_node_unavailable", message=str(exc)), status_code=503)

    def _register_routes(self):
        super()._register_routes()
        self.app.add_api_route("/internal/pd/descriptor", self.descriptor, methods=["GET"])
        self.app.add_api_route("/internal/pd/abort", self.abort, methods=["POST"])
        self.app.add_api_route("/internal/pd/requests/{request_id}", self.status, methods=["GET"])

    async def status(self, request_id: str):
        return JSONResponse(self.service.status(request_id))

    async def _health(self):
        healthy = self.service.is_ready()
        return JSONResponse(dict(status="ok" if healthy else "unavailable"), status_code=200 if healthy else 503)

    async def descriptor(self):
        node = self.service.node
        if node is None:
            return JSONResponse(dict(error="node initializing"), status_code=503)
        return JSONResponse(msgspec.to_builtins(msgspec.structs.replace(
            node, healthy=self.service.is_ready())))

    async def abort(self, request: HTTPRequest):
        binding = decode(await request.json(), Binding)
        await self.service.cancel_binding(binding)
        return JSONResponse(dict(accepted=True))

    async def _completions(self, request: HTTPRequest):
        return await self.handle(request, "completion")

    async def _chat_completions(self, request: HTTPRequest):
        return await self.handle(request, "chat")

    async def handle(self, request, kind):
        if not self.service.is_ready():
            raise NodeUnavailable("PD node is unavailable")
        raw = await request.json()
        context = json.loads(request.headers.get("x-pd-context", "{}"))
        if set(context) != {"binding", "peer"}:
            raise ValueError("generation requires Router PD context")
        binding, peer = decode(context["binding"], Binding), decode(context["peer"], Node)
        if not peer.compatible(self.service.node) or getattr(binding, peer.role) != peer.key:
            raise ValueError("HTTP peer differs from route binding")
        public = (CompletionRequest if kind == "completion" else ChatCompletionRequest).model_validate(raw)
        entry = self.service.register_http(binding, raw)
        if self.service.role == "prefill":
            try:
                if kind == "chat":
                    prompt, config, parser, constraint = await self.prepare_chat(public)
                else:
                    prompt, config = self.prepare_completion(public)
                    parser = constraint = None
                self.service.adapter.validate(config)
                tokens = tuple(self.real_engine._tokenize_prompt(prompt))
                continuation = Continuation(tokens, asdict(config), asdict(parser) if parser else None,
                                            constraint.to_wire() if constraint else None,
                                            self.service.adapter.prefix_spec(tokens, self.service.cache))
                entry.task = asyncio.create_task(self.service.prefill(entry, continuation, peer))
                try:
                    await asyncio.shield(entry.task)
                except asyncio.CancelledError:
                    entry.cancelled = True
                    await asyncio.shield(entry.task)
                    raise
                return JSONResponse(dict(request=binding.request, status="ready"))
            except BaseException as exc:
                if isinstance(exc, asyncio.CancelledError):
                    entry.cancelled = True
                if entry.error is None:
                    self.service.fail(entry, exc)
                if (entry.task is None or entry.task.done()) and binding.request not in self.service.pins.active:
                    self.service.finish(entry)
                raise
        try:
            await asyncio.wait_for(entry.prepared.wait(), self.service.config.handoff_timeout)
            if entry.error is not None:
                raise entry.error
            if entry.cancelled:
                raise ValueError("request cancelled")
            continuation = entry.continuation
            from pypto_serving.config.types import GenerateConfig
            config = GenerateConfig(**continuation.config)
            if kind == "chat":
                response = await self.chat_response(binding.request, public, "", config,
                    OutputParserSpec(**continuation.parser) if continuation.parser else None, None)
            else:
                response = await self.completion_response(binding.request, public, config)
            if isinstance(response, StreamingResponse):
                response.body_iterator = self.stream(response.body_iterator)
            return response
        except BaseException:
            entry.response_closed = True
            await self.service.abort(entry)
            raise

    @staticmethod
    async def stream(iterator):
        from contextlib import aclosing
        try:
            async with aclosing(iterator):
                async for chunk in iterator:
                    yield chunk
        except Exception as exc:
            yield "data: " + json.dumps({"error": {"message": str(exc), "type": "pd_execution_error"}}) + "\n\n"
            yield "data: [DONE]\n\n"


class Application:
    def __init__(self, engine_config, config, role, endpoint, tokenizer, generate):
        self.boot = uuid.uuid4().hex
        self.adapter = select_adapter(engine_config.executor_cls)
        if engine_config.runtime_config.num_speculative_tokens != 7:
            raise ValueError("PD deployment requires the K7 model configuration")
        self.config = config
        progress = channel = None
        if role == "prefill":
            parent, channel = multiprocessing.get_context("spawn").Pipe(duplex=True)
            progress = Progress(parent, config.handoff_timeout)
        self.channel, self.progress = channel, progress
        settings = WorkerSettings(self.boot, endpoint.transfer_hostname or endpoint.host,
                                  config.provider, config.capacity, len(config.prefill) + len(config.decode))
        tokens = 0 if role == "prefill" else 7
        engine_config = replace(engine_config,
            runtime_config=replace(engine_config.runtime_config, num_speculative_tokens=tokens),
            executor_kwargs={**engine_config.executor_kwargs, "num_speculative_tokens": tokens,
                "execution_role": role,
                "capture_prefill_tail": role == "prefill",
                "runner_extension_factory": partial(self.adapter.worker_factory,
                                                      settings=settings, progress_channel=channel)},
            async_scheduling=role == "decode")
        self.engine = AsyncLLMEngine(engine_config, tokenizer, core_factory=partial(
            PrefillCore if role == "prefill" else ReplicaEngineCore,
            scheduler_factory=prefill_scheduler if role == "prefill" else Scheduler,
            worker_factory=partial(spawn_worker, services_factory=Services)))
        self.core = self.engine.single_core()
        self.service = Service(config, endpoint, role, self.adapter, self.core, progress)
        if role == "prefill":
            self.core.publish = self.wrap_results()
        self.server = Server(self.service, self.engine, generate)
        self.app = self.server.app
        self.app.router.lifespan_context = self.lifespan

    @asynccontextmanager
    async def lifespan(self, app):
        await self.start()
        try:
            yield
        finally:
            await self.stop()

    def wrap_results(self):
        counters = {}
        def complete(output, tokens, drafts=None):
            for scheduled in output.scheduled_requests:
                if not scheduled.is_prefill:
                    continue
                request = scheduled.request
                rid = request.request_id
                end = scheduled.num_computed_tokens + scheduled.num_new_tokens
                final = end >= request.num_prompt_tokens
                first = tokens.get(rid)
                if isinstance(first, list):
                    first = first[0] if first else None
                if final and first is None:
                    raise RuntimeError("final Prefill has no sampled first token")
                if scheduled.cache_partition is None:
                    raise RuntimeError("Prefill has no cache partition")
                self.core.scheduler.hold_request(rid)
                chunk = counters.get(rid, 0)
                counters[rid] = chunk + 1
                if final:
                    counters.pop(rid)
                self.service.chunks[rid].put_nowait(Chunk(rid, chunk, scheduled.num_computed_tokens,
                    end, final, first, {k: tuple(v) for k, v in scheduled.block_ids_by_group.items()},
                    scheduled.cache_partition))
        return complete

    async def start(self):
        self.logs = log_context(self.config, self.service.role, self.service.endpoint.node_id)
        self.logs.__enter__()
        await self.engine.start()
        if self.channel is not None:
            self.channel.close()
        await self.service.start(self.boot, self.core.config.model_id)

    async def stop(self):
        # Unsafe retirement raises before registered allocations can be freed.
        await self.service.close()
        await self.engine.stop()
        if self.progress is not None:
            await self.progress.close()
        self.logs.__exit__(None, None, None)


def run(args, engine_config, generate):
    import uvicorn
    from pypto_serving.tools.profile import configure_profiler
    from pypto_serving.serving.utils.env import configure_runtime_logging
    from inspect import signature
    from .transfer import provider_class
    from pypto_serving.model.tokenizer import load_tokenizer
    if args.prompt or not args.pd_config:
        raise ValueError("PD requires --pd-config and online serving")
    config = Config.load(args.pd_config)
    if not hasattr(os, "pidfd_open"):
        raise ValueError("PD model nodes require Linux Python with os.pidfd_open")
    from pypto.runtime.distributed_runner import DistributedWorker
    if ("chip_service_factories" not in signature(DistributedWorker).parameters
            or not all(hasattr(DistributedWorker, name)
                       for name in ("retain_service_tensor", "release_service_tensor"))):
        raise ValueError("PyPTO lacks the required owner-service and tensor-retain interfaces")
    provider_class(config.provider).check_available()
    logging.basicConfig(level=logging.INFO)
    configure_runtime_logging()
    configure_profiler(engine_config.profile_config, process_name="pd-api", initially_active=False)
    endpoint = config.endpoint(args.pd_role, args.pd_node_id)
    if args.port != endpoint.port:
        raise ValueError("HTTP --port must equal this node's configured port")
    application = Application(engine_config, config, args.pd_role, endpoint,
                              load_tokenizer(engine_config.model_dir), generate)
    uvicorn.run(application.app, host=args.host, port=args.port)
