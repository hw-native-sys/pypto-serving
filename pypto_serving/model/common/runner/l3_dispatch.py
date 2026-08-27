# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared L3 worker handle, static-upload cache, and unified dispatch.

Both the DeepSeek and Qwen runners mix :class:`L3DispatchMixin` in to share the
resolve → ``worker.run`` → free-transients dispatch path. The two runners differ
only in parameters already expressible here:

* ``stacked`` -- multi-rank (DeepSeek, ``alloc_stacked_tensor``) vs single-rank
  (Qwen, ``alloc_tensor``); set via ``_init_l3_dispatch``.
* ``dispatch_args`` -- a static launch-arg prefix (Qwen only); prepended
  unconditionally (DeepSeek's empty tuple is a no-op).
* per-dispatch transient uploads -- always tracked in ``uploaded`` and freed in
  the ``finally`` block; an empty list is a no-op (Qwen has no transients today,
  but gains the escape hatch).

The worker itself (fork, persistence flags, inherited host tensors) is
runner-specific, so ``_shared_l3_worker`` stays abstract.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from pypto_serving.tools.profile import profile_span

from .buffer_set import resolve_l3_arg

__all__ = ["L3DispatchMixin", "PendingL3Dispatch"]


@dataclass
class PendingL3Dispatch:
    """Own one asynchronous PyPTO dispatch until completion is reclaimed."""

    worker: Any
    handle: Any
    uploaded: tuple[Any, ...]
    name: str
    _released: bool = False
    _error: BaseException | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def wait(self) -> None:
        """Wait for completion and release per-dispatch device uploads once."""
        with self._lock:
            if self._released:
                if self._error is not None:
                    raise self._error
                return
            error: BaseException | None = None
            try:
                with profile_span(
                    f"{self.name}.worker_wait",
                    cat="kernel",
                    level="kernel",
                ):
                    self.handle.result()
            except BaseException as exc:
                error = exc
            for tensor in self.uploaded:
                try:
                    self.worker.free_tensor(tensor)
                except BaseException as exc:
                    if error is None:
                        error = exc
            self._error = error
            self._released = True
            if error is not None:
                raise error


class L3DispatchMixin:
    """Owns the shared L3 worker, the static-tensor upload cache, and dispatch."""

    # Attributes provisioned by ``_init_l3_dispatch`` (declared for type checkers).
    _l3_worker: Any | None
    _l3_static_tensors: dict[tuple, Any]
    _l3_stacked: bool

    def _init_l3_dispatch(self, *, stacked: bool) -> None:
        """Initialize the shared dispatch state. Call from the runner ``__init__``."""
        self._l3_worker = None
        self._l3_static_tensors: dict[tuple, Any] = {}
        self._l3_stacked = stacked

    def _shared_l3_worker(self) -> Any:
        """Return the persistent ``DistributedWorker`` (runner-specific fork logic)."""
        raise NotImplementedError

    def _l3_run_config(self, callable_spec: Any) -> Any | None:
        """Return an optional per-dispatch PyPTO ``RunConfig``."""
        return None

    def _run_l3(self, callable_spec: Any, *args: Any) -> None:
        """Dispatch one L3 program: resolve args, run, free per-dispatch uploads.

        ``callable_spec`` is an :class:`~pypto_serving.model.common.compiler.l3_callable.L3Callable`
        (``compiled``, ``name``, ``aicpu_thread_num``, optional ``block_dim``,
        optional ``dispatch_args`` prefix).
        """
        span_args: dict[str, Any] = {"aicpu_thread_num": callable_spec.aicpu_thread_num}
        if callable_spec.block_dim is not None:
            span_args["block_dim"] = callable_spec.block_dim
        with profile_span(callable_spec.name, cat="kernel", level="kernel", args=span_args):
            worker = self._shared_l3_worker()
            uploaded: list[Any] = []
            try:
                l3_args = tuple(callable_spec.dispatch_args) + tuple(
                    resolve_l3_arg(
                        worker,
                        arg,
                        self._l3_static_tensors,
                        uploaded=uploaded,
                        cache_keys=None,
                        stacked=self._l3_stacked,
                    )
                    for arg in args
                )
                with profile_span(
                    f"{callable_spec.name}.worker_run",
                    cat="kernel",
                    level="kernel",
                    args=dict(span_args),
                ):
                    run_config = self._l3_run_config(callable_spec)
                    if run_config is None:
                        worker.run(callable_spec.compiled, *l3_args)
                    else:
                        worker.run(callable_spec.compiled, *l3_args, config=run_config)
            finally:
                for tensor in uploaded:
                    worker.free_tensor(tensor)

    def _submit_l3(self, callable_spec: Any, *args: Any) -> PendingL3Dispatch:
        """Submit one L3 program and transfer argument ownership to its handle.

        The async counterpart of ``_run_l3``: resolve the args the same way, but
        hand the per-dispatch uploads to the pending handle so they are freed
        only after ``wait()`` reclaims the outputs.
        """
        span_args: dict[str, Any] = {"aicpu_thread_num": callable_spec.aicpu_thread_num}
        if callable_spec.block_dim is not None:
            span_args["block_dim"] = callable_spec.block_dim
        worker = self._shared_l3_worker()
        uploaded: list[Any] = []
        try:
            l3_args = tuple(callable_spec.dispatch_args) + tuple(
                resolve_l3_arg(
                    worker,
                    arg,
                    self._l3_static_tensors,
                    uploaded=uploaded,
                    cache_keys=None,
                    stacked=self._l3_stacked,
                )
                for arg in args
            )
            with profile_span(
                f"{callable_spec.name}.worker_submit",
                cat="kernel",
                level="kernel",
                args=dict(span_args),
            ):
                run_config = self._l3_run_config(callable_spec)
                if run_config is None:
                    handle = worker.submit(callable_spec.compiled, *l3_args)
                else:
                    handle = worker.submit(callable_spec.compiled, *l3_args, config=run_config)
        except BaseException:
            for tensor in uploaded:
                worker.free_tensor(tensor)
            raise
        return PendingL3Dispatch(
            worker=worker,
            handle=handle,
            uploaded=tuple(uploaded),
            name=callable_spec.name,
        )

    def _reset_l3_dispatch(self) -> None:
        """Drop the static-upload cache (call on worker reset / teardown)."""
        self._l3_static_tensors.clear()
