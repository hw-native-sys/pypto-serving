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

The mixin also owns the per-dispatch simpler ring sizing: instead of the old
process-wide ``PTO2_RING_*`` envs (which resized rings for every worker in the
process, staging pools included), ``_configure_l3_rings`` installs a pypto
``RunConfig`` whose ``ring_*`` fields are forwarded to ``CallConfig.runtime_env``
on each ``worker.run`` / ``worker.submit`` — scoping the sizing to L3 dispatches
only. Runners call it from ``init_kv_cache`` (before the first dispatch) with
the values straight from ``RuntimeConfig``.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import shutil
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pypto_serving.config.types import RuntimeConfig
from pypto_serving.tools.profile import profile_span

from .buffer_set import resolve_l3_arg

__all__ = ["L3DispatchMixin", "PendingL3Dispatch"]

logger = logging.getLogger(__name__)


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
    _l3_run_config: Any | None
    _chip_swimlane_level: int
    _chip_capture_dir: Path | None
    _chip_capture_errors: list[str]

    def _init_l3_dispatch(self, *, stacked: bool) -> None:
        """Initialize the shared dispatch state. Call from the runner ``__init__``."""
        self._l3_worker = None
        self._l3_static_tensors: dict[tuple, Any] = {}
        self._l3_stacked = stacked
        self._l3_run_config = None
        self._chip_swimlane_level = 0
        self._chip_capture_dir = None
        self._chip_capture_index = 0
        self._chip_capture_errors = []

    def set_chip_swimlane_level(self, level: int, output_dir: Path | None = None) -> list[str]:
        """Set chip capture level and return errors from the session being stopped."""
        if isinstance(level, bool) or not isinstance(level, int) or not 0 <= level <= 4:
            raise ValueError("chip swimlane level must be an integer from 0 to 4")
        if level and output_dir is None:
            raise ValueError("chip swimlane capture requires an output directory")
        if level:
            self._chip_capture_dir = Path(output_dir) / f"worker-{os.getpid()}-{uuid.uuid4().hex[:8]}"
            self._chip_capture_index = 0
            self._chip_capture_errors = []
        else:
            self._chip_capture_dir = None
        self._chip_swimlane_level = level
        errors = self._chip_capture_errors if not level else []
        if not level:
            self._chip_capture_errors = []
        return errors

    def _chip_record_state(self, callable_spec: Any) -> dict[Path, tuple[int, int, int]] | None:
        if self._chip_capture_dir is None:
            return {}
        from pypto.runtime._artifact_runtime import runtime_output_directory  # noqa: PLC0415

        source = runtime_output_directory(callable_spec.compiled) / "dfx_outputs"
        records = {}
        try:
            for record in source.glob("rank*/d*/chip_swimlane_records.json"):
                stat = record.stat()
                records[record] = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
        except OSError as exc:
            message = f"Failed to inspect chip profiling records for {callable_spec.name}: {exc}"
            self._chip_capture_errors.append(message)
            logger.exception(message)
            return None
        return records

    def _save_chip_capture(
        self, callable_spec: Any, previous_records: dict[Path, tuple[int, int, int]] | None,
    ) -> None:
        if self._chip_capture_dir is None or previous_records is None:
            return
        from pypto.runtime._artifact_runtime import runtime_output_directory  # noqa: PLC0415

        source = runtime_output_directory(callable_spec.compiled) / "dfx_outputs"
        records = self._chip_record_state(callable_spec)
        if records is None:
            return
        if not any(state != previous_records.get(path) for path, state in records.items()):
            message = f"Chip profiling produced no new records for {callable_spec.name} at {source}"
            self._chip_capture_errors.append(message)
            logger.error(message)
            return
        self._chip_capture_index += 1
        destination = self._chip_capture_dir / f"{self._chip_capture_index:06d}_{callable_spec.name}"
        temporary = destination.with_name(f".{destination.name}.tmp")
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, temporary)
            temporary.rename(destination)
        except OSError as exc:
            shutil.rmtree(temporary, ignore_errors=True)
            message = f"Failed to archive chip profiling artifacts for {callable_spec.name}: {exc}"
            self._chip_capture_errors.append(message)
            logger.exception(message)

    def _profiled_run_config(self, config: Any) -> Any:
        level = self._chip_swimlane_level
        if not level:
            return config
        if config is None:
            from pypto.runtime import RunConfig  # noqa: PLC0415

            return RunConfig(enable_chip_swimlane=level)
        return dataclasses.replace(config, enable_chip_swimlane=level)

    def _configure_l3_rings(self, runtime: RuntimeConfig) -> None:
        """Install the per-dispatch simpler ring sizing for L3 dispatches.

        Replaces the process-wide ``PTO2_RING_DEP_POOL`` / ``PTO2_RING_TASK_WINDOW``
        / ``PTO2_RING_HEAP`` envs, which sized rings for every worker in the
        process; the values here scope to L3 dispatches alone via pypto's
        per-dispatch ``RunConfig`` (``CallConfig.runtime_env``). Each value is
        a scalar (broadcast to all scope-depth rings) or a list of exactly 4
        ints sizing rings 0..3; an unset field stays ``None`` so the pypto
        runtime's own default applies. When every field is unset no RunConfig
        is passed at all, keeping the dispatch baseline unchanged.
        """
        if all(
            value is None
            for value in (runtime.ring_dep_pool, runtime.ring_task_window, runtime.ring_heap)
        ):
            self._l3_run_config = None
            return
        from pypto.runtime import RunConfig  # noqa: PLC0415

        self._l3_run_config = RunConfig(
            ring_dep_pool=runtime.ring_dep_pool,
            ring_task_window=runtime.ring_task_window,
            ring_heap=runtime.ring_heap,
        )

    def _shared_l3_worker(self) -> Any:
        """Return the persistent ``DistributedWorker`` (runner-specific fork logic)."""
        raise NotImplementedError

    def _run_l3(self, callable_spec: Any, *args: Any, config: Any = None) -> None:
        """Dispatch one L3 program: resolve args, run, free per-dispatch uploads.

        ``callable_spec`` is an :class:`~pypto_serving.model.common.compiler.l3_callable.L3Callable`
        (``compiled``, ``name``, ``aicpu_thread_num``, optional ``block_dim``,
        optional ``dispatch_args`` prefix).  ``config`` overrides the runner's
        default ``RunConfig`` for this dispatch alone -- a runner that hosts
        programs with different ring requirements passes each program's config
        here instead of resizing process-wide state.
        """
        span_args: dict[str, Any] = {"aicpu_thread_num": callable_spec.aicpu_thread_num}
        if callable_spec.block_dim is not None:
            span_args["block_dim"] = callable_spec.block_dim
        run_config = self._profiled_run_config(self._l3_run_config if config is None else config)
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
                    previous_records = self._chip_record_state(callable_spec)
                    worker.run(
                        callable_spec.compiled, *l3_args, config=run_config,
                    )
                self._save_chip_capture(callable_spec, previous_records)
            finally:
                for tensor in uploaded:
                    worker.free_tensor(tensor)

    def _submit_l3(
        self,
        callable_spec: Any,
        *args: Any,
        config: Any = None,
    ) -> PendingL3Dispatch:
        """Submit one L3 program and transfer argument ownership to its handle.

        The async counterpart of ``_run_l3``: resolve the args the same way, but
        hand the per-dispatch uploads to the pending handle so they are freed
        only after ``wait()`` reclaims the outputs.
        """
        span_args: dict[str, Any] = {"aicpu_thread_num": callable_spec.aicpu_thread_num}
        if callable_spec.block_dim is not None:
            span_args["block_dim"] = callable_spec.block_dim
        run_config = self._profiled_run_config(self._l3_run_config if config is None else config)
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
                previous_records = self._chip_record_state(callable_spec)
                handle = worker.submit(
                    callable_spec.compiled, *l3_args, config=run_config,
                )
                if self._chip_swimlane_level:
                    handle.result()
                    self._save_chip_capture(callable_spec, previous_records)
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
