# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Optional engine lifecycle seam; ordinary serving imports no copy/runtime backend."""

import asyncio
from dataclasses import replace
from functools import partial
from mmap import PAGESIZE
import logging
import multiprocessing as mp
from pathlib import Path
import threading
import time

from .backend import SimplerBackend, endpoints
from .cache import TieredCache
from .native import build_copy_library
from .pool import allocate_pool
from .trace import OffloadTrace

logger = logging.getLogger(__name__)


def initialize_loaded(executor):
    extensions = executor.runtime_extensions()
    if len(extensions) != 1:
        raise RuntimeError("cache offload requires one configured model extension")
    extensions[0].model_ready()


class Session:
    def __init__(self, core):
        self.core = core
        config = core.config
        if (config.executor_cls != "PyptoDeepSeekV4DSparkExecutor"
                or config.runtime_config.kv_offload_backend != "simpler"
                or not config.enable_prefix_cache):
            raise ValueError("DDR offload requires DSpark prefix caching and the simpler backend")
        capacity = config.runtime_config.kv_offload_bytes // PAGESIZE * PAGESIZE
        if capacity <= 0:
            raise ValueError("DDR pool is smaller than one physical page")
        context = mp.get_context("spawn")
        self.owners = endpoints(context, len(config.worker_device_ids()))
        self.metadata = context.Queue(maxsize=1)
        self.cache = None
        self._worker_started = False
        self.trace = None
        self._loop = None
        self._notify_lock = threading.Lock()
        self._notified = False
        from .simpler import descriptor_wire
        self.copy_library = build_copy_library()
        try:
            self.pool = allocate_pool(capacity)
        except BaseException:
            self.copy_library.cleanup()
            raise
        self.pool_wire = descriptor_wire(self.pool.to_descriptor())
        self.pool_wire["copy_library"] = str(Path(self.copy_library.name) / "libcache_copy.so")

    def configure(self, config):
        from pypto_serving.model.deepseek_dspark.cache_offload import CacheOffloadExtension
        kwargs = dict(config.executor_kwargs)
        previous = kwargs.get("runner_extension_factory")
        kwargs["runner_extension_factory"] = partial(CacheOffloadExtension,
            endpoints=self.owners, metadata=self.metadata, pool=self.pool_wire, previous=previous)
        return replace(config, executor_kwargs=kwargs)

    def worker_starting(self):
        self._worker_started = True

    async def start(self):
        for owner in self.owners:
            owner.child.close()
        geometry = await asyncio.to_thread(self.metadata.get, timeout=30)
        manager = self.core.kv_cache_manager
        if (geometry["ranks"] != len(self.owners) or geometry["tp"] <= 0
                or manager.group_partition_count * geometry["tp"] != len(self.owners)
                or set(geometry["groups"]) != set(manager.group_names)):
            raise RuntimeError("offload worker/cache-manager geometry mismatch")
        self.trace = OffloadTrace.from_env()
        backend = SimplerBackend(self.owners, geometry, self.pool.nbytes, trace=self.trace)
        self._loop = asyncio.get_running_loop()
        self.cache = TieredCache(manager, backend, self.pool.nbytes, trace=self.trace, notify=self._notify)
        backend.set_notify(self._notify)
        manager.tiering = self.cache
        if self.trace is not None:
            self.trace.emit("ready", capacity_bytes=self.pool.nbytes, ranks=geometry["ranks"],
                            tp=geometry["tp"], groups=geometry["groups"])
            self.trace.snapshot(self.cache, force=True)
        logger.info("kv_offload_ready backend=simpler capacity_bytes=%d ranks=%d tp=%d",
                    self.cache.tier.capacity, geometry["ranks"], geometry["tp"])

    def _notify(self):
        with self._notify_lock:
            if self._loop is None or self._notified:
                return
            self._notified = True
            self._loop.call_soon_threadsafe(self._consume)

    def _consume(self):
        with self._notify_lock:
            self._notified = False
            if self._loop is None:
                return
        try:
            self.cache.progress()
        except Exception as error:
            # Wake clients and remove readiness without claiming native stop.
            self.core._running = False
            for context in self.core._request_contexts.values():
                context.queue.put_nowait(RuntimeError(f"DDR cache progress failed: {error}"))
            logger.exception("DDR cache completion failed; retained allocations require confirmed retirement")

    async def close(self):
        try:
            await self._close()
        finally:
            if self.cache is not None:
                self.cache.backend.set_notify(None)
            with self._notify_lock:
                self._loop = None
            if self.trace is not None:
                if self.cache is not None:
                    self.trace.snapshot(self.cache, force=True)
                self.trace.close()

    async def _close(self):
        if self.cache is None:
            if self._worker_started:
                await asyncio.gather(*(asyncio.to_thread(owner.call, "stop") for owner in self.owners
                                       if not owner.stopped.value))
            self.pool.close()
            self.copy_library.cleanup()
            return
        self.cache.begin_close()
        deadline = time.monotonic() + 120
        previous = None
        while not self.cache.close_ready():
            tier = self.cache.tier
            progress = tier.used, len(tier.inflight), len(tier.retiring), len(self.cache.loads)
            now = time.monotonic()
            if progress != previous:
                previous, deadline = progress, now + 120
            elif now >= deadline:
                raise RuntimeError("DDR drain incomplete; owner allocations remain retained")
            await asyncio.sleep(.002)
        try:
            await asyncio.to_thread(self.cache.backend.close)
        except BaseException as error:
            self.cache.tier.error = error
            raise
        self.pool.close()
        self.copy_library.cleanup()
        self.cache.tier.finish_close()
        self.core.kv_cache_manager.tiering = None
        if self.trace is not None:
            self.trace.emit("shutdown", resources_closed=True)
        logger.info("kv_offload_closed counters=%s", self.cache.tier.counters)
