# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Semantic request tapes with asynchronous export; never execution or release authority."""

import asyncio
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import inspect
from itertools import count
import json
import os
from pathlib import Path
from queue import Empty, Full, Queue
import re
import socket
import sys
import threading
import time
import uuid

import msgspec

HEADER = "x-pypto-request-profile"
QUEUE_EVENTS = 262144
FILE_BYTES = 1 << 30
current = ContextVar("request_profile", default=None)
_sessions = {}
_aliases = {}
_jobs = {}
_process = uuid.uuid4().hex
_guard = threading.RLock()
_CURRENT = object()
_lanes = threading.local()
_batch_ids = count()


def lane():
    value = getattr(_lanes, "name", None)
    if value is None:
        value = _lanes.name = f"{threading.current_thread().name}:{threading.get_ident()}"
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    if task is not None:
        task_lane = getattr(task, "_request_profile_lane", None)
        if task_lane is None:
            task_lane = task._request_profile_lane = f"{value}/{task.get_name()}:{id(task)}"
        return task_lane
    return value


class Capture:
    def __init__(self, capture, directory, node, role):
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", capture):
            raise ValueError("invalid request capture ID")
        self.id, self.node, self.role = capture, node, role
        self.directory = Path(directory)
        directory_error = None
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            directory_error = "directory_" + type(error).__name__
        self.key = f"{socket.gethostname()}:{_process}:{os.getpid()}"
        self.path = self.directory / f"trace.{os.getpid()}.{_process}.jsonl"
        self.queue = Queue(maxsize=QUEUE_EVENTS)
        self.references = Counter()
        self.active = 0
        self.dropped = 0
        self.failure = directory_error
        self.closed = False
        self._sentinel = False
        self.clock_end = None
        self.environment = dict(python=sys.version, serving_source=str(Path(__file__).resolve().parents[3]),
            sources={name: os.environ.get(name) for name in ("PYPTO_ROOT", "SIMPLER_ROOT", "PYPTO_LIB_ROOT")})
        self.clock = dict(wall_ns=str(time.time_ns()), monotonic_ns=str(time.monotonic_ns()),
                          perf_ns=str(time.perf_counter_ns()))
        self._writer = threading.Thread(target=self._write, name="request-profile-writer", daemon=True)
        self._writer.start()
        self.emit("process_name", ph="M", tid="0", args=dict(name=f"{node}/{role}/{os.getpid()}"))
        self.emit("clock.anchor", args=self.clock)
        with _guard:
            for key, previous in tuple(_sessions.items()):
                if previous.closed and not previous.pending() and not previous._writer.is_alive():
                    del _sessions[key]
            _sessions[capture] = self

    def wire(self, roots=()):
        return dict(capture=self.id, directory=str(self.directory), node=self.node,
                    role=self.role, roots=list(roots))

    def emit(self, name, *, ph="i", ts=None, tid=None, **fields):
        if self.closed or self.failure:
            return
        try:
            self.queue.put_nowait((name, ph, time.time_ns() // 1000 if ts is None else ts,
                                   tid or lane(), fields))
        except Full:
            self.dropped += 1
        except Exception as error:
            self.failure = type(error).__name__

    def _write(self):
        count = size = 0
        jobs = {}
        try:
            encoder = msgspec.json.Encoder(enc_hook=lambda value: dict(type=type(value).__name__))
            with self.path.open("xb", buffering=1 << 20) as stream:
                while True:
                    try:
                        event = self.queue.get(timeout=.25)
                    except Empty:
                        stream.flush()
                        continue
                    if event is None:
                        break
                    name, phase, stamp, thread, fields = event
                    record = dict(name=name, ph=phase, ts=stamp, pid=self.key, tid=thread, **fields)
                    if phase == "i":
                        record["s"] = "t"
                    records = [record]
                    args = fields.get("args", {})
                    if "job" in args and "monotonic_ns" in args and name.endswith((".submit", ".complete")):
                        kind, phase = name.rsplit(".", 1)
                        key = kind, args["job"]
                        if phase == "submit":
                            jobs[key] = record
                        else:
                            first = jobs.pop(key, None)
                            if first is not None:
                                records.append(dict(name=kind + ".lifetime", ph="X", pid=self.key,
                                    tid=first["tid"], ts=first["ts"],
                                    dur=max(0, args["monotonic_ns"]-first["args"]["monotonic_ns"])/1000,
                                    cat="async-job", args=args))
                    for record in records:
                        line = encoder.encode(record) + b"\n"
                        if size + len(line) > FILE_BYTES:
                            self.dropped += 1
                            continue
                        size += len(line)
                        stream.write(line)
                        count += 1
                stream.flush()
        except Exception as error:
            self.failure = type(error).__name__
        finally:
            try:
                with _guard:
                    status = dict(process=self.key, capture=self.id, node=self.node, role=self.role,
                        events=count, bytes=size, dropped=self.dropped, error=self.failure, environment=self.environment,
                        complete=not self.failure and not self.dropped, clock=self.clock, clock_end=self.clock_end,
                        unresolved=dict(self.references), active_scopes=self.active)
                self.path.with_suffix(".status.json").write_text(json.dumps(status))
            except Exception as error:
                self.failure = "status_write_" + type(error).__name__

    def pending(self):
        with _guard:
            return sum(self.references.values()) + self.active

    def close(self, *, incomplete=False):
        if self.closed and self._sentinel:
            self._writer.join(5)
            return not self._writer.is_alive()
        if self.pending() and not incomplete:
            return False
        self.clock_end = dict(wall_ns=str(time.time_ns()), monotonic_ns=str(time.monotonic_ns()))
        adjustment = (int(self.clock_end["wall_ns"])-int(self.clock["wall_ns"])
                      -int(self.clock_end["monotonic_ns"])+int(self.clock["monotonic_ns"]))
        self.clock_end.update(adjustment_ns=adjustment, clock_adjusted=abs(adjustment)>1_000_000)
        self.emit("clock.end", args=self.clock_end)
        self.closed = True
        if not self._writer.is_alive():
            return True
        try:
            self.queue.put(None, timeout=2)
            self._sentinel = True
        except Full:
            self.failure = "writer_queue_full_at_close"
            return False
        self._writer.join(5)
        if self._writer.is_alive():
            self.failure = "writer_did_not_close"
            return False
        return True


def ensure(tag):
    if not tag:
        return None
    with _guard:
        capture = _sessions.get(tag["capture"])
        if capture is None:
            capture = Capture(tag["capture"], tag["directory"], tag["node"], tag["role"])
        return capture


def context(request_ids=()):
    if not _sessions:
        return None
    tags = [_aliases[r] for r in request_ids if r in _aliases]
    if not tags:
        return None
    result = dict(tags[0])
    result["roots"] = sorted({root for tag in tags for root in tag["roots"]})
    return result


def bind(request_id, tag=None, *, owner="engine"):
    tag = tag or current.get()
    if not tag:
        return
    capture = ensure(tag)
    with _guard:
        if request_id not in _aliases:
            _aliases[request_id] = tag
        capture.references["request:" + owner + ":" + request_id] = 1
    event("request.bind", request_id=request_id, tag=tag)


def release(request_id, *, owner="engine"):
    with _guard:
        tag = _aliases.get(request_id)
        if tag:
            capture = ensure(tag)
            event("request.resources_retired", request_id=request_id, owner=owner, tag=tag)
            capture.references.pop("request:" + owner + ":" + request_id, None)
            if not any(key.startswith("request:") and key.endswith(":" + request_id)
                       for key in capture.references):
                _aliases.pop(request_id, None)


def event(name, *, tag=_CURRENT, **fields):
    tag = current.get() if tag is _CURRENT else tag
    if tag:
        capture = ensure(tag)
        capture.emit(name, args=dict(roots=tag["roots"], **fields))


def track(kind, identity, tag=_CURRENT, *, record=True):
    tag = current.get() if tag is _CURRENT else tag
    if tag:
        key = kind, str(identity)
        with _guard:
            if key not in _jobs:
                _jobs[key] = (tag, record)
                ensure(tag).references[str(key)] += 1
                if record:
                    event(kind + ".submit", tag=tag, job=str(identity), monotonic_ns=time.monotonic_ns())
            else:
                _jobs[key] = (combine((_jobs[key][0], tag)), _jobs[key][1] or record)
    return tag


def finish(kind, identity, *, error=None):
    key = kind, str(identity)
    with _guard:
        job = _jobs.pop(key, None)
        if job:
            tag, record = job
            if record:
                event(kind + ".complete", tag=tag, job=str(identity), error=error, monotonic_ns=time.monotonic_ns())
            ensure(tag).references.pop(str(key), None)


@contextmanager
def scope(tag, name=None, *, cat="request", **fields):
    if not tag:
        token = current.set(None) if current.get() is not None else None
        try:
            yield
        finally:
            if token is not None:
                current.reset(token)
        return
    capture = ensure(tag)
    token = current.set(tag)
    start, wall, tid = time.perf_counter_ns(), time.time_ns(), lane()
    with _guard:
        capture.active += 1
    outcome = "completed"
    try:
        yield
    except BaseException as error:
        outcome = type(error).__name__
        raise
    finally:
        if name:
            capture.emit(name, ph="X", ts=wall // 1000, tid=tid,
                dur=max(0, time.perf_counter_ns()-start) / 1000,
                cat=cat, args=dict(roots=tag["roots"], batch=tag.get("batch"), outcome=outcome,
                                         wall_end_ns=str(time.time_ns()), **fields))
        current.reset(token)
        with _guard:
            capture.active -= 1


def duration(name, dur_us, *, cat="request", args=None, ts_us=None):
    tag = current.get()
    if not tag:
        return False
    capture = ensure(tag)
    # Existing span producers pass perf_counter timestamps, never epoch timestamps.
    wall = time.time_ns() // 1000
    start = wall-dur_us if ts_us is None else wall+(ts_us-time.perf_counter_ns()/1000)
    capture.emit(name, ph="X", ts=start, dur=max(0, dur_us), cat=cat,
                 args=dict(roots=tag["roots"], **(args or {})))
    return True


def traced(name, resolve=None):
    def decorate(function):
        def tag(args, kwargs):
            if not enabled():
                return None
            try:
                return resolve(*args, **kwargs) if resolve is not None else current.get()
            except Exception:
                for capture in tuple(_sessions.values()):
                    if not capture.closed:
                        capture.failure = "context_resolution_error"
                return None
        if inspect.iscoroutinefunction(function):
            @wraps(function)
            async def call(*args, **kwargs):
                value = tag(args, kwargs)
                if not value:
                    if current.get() is None:
                        return await function(*args, **kwargs)
                    with scope(None):
                        return await function(*args, **kwargs)
                with scope(value, name):
                    return await function(*args, **kwargs)
        else:
            @wraps(function)
            def call(*args, **kwargs):
                value = tag(args, kwargs)
                if not value:
                    if current.get() is None:
                        return function(*args, **kwargs)
                    with scope(None):
                        return function(*args, **kwargs)
                with scope(value, name):
                    return function(*args, **kwargs)
        return call
    return decorate


def command_context(command):
    tag = getattr(command, "request_profile", None)
    return part(tag, "compute")


def part(tag, purpose):
    if not tag or "parts" not in tag:
        return tag
    roots = tag["parts"].get(purpose, ())
    return dict(tag, roots=roots) if roots else None


def subset(request_ids):
    tag = current.get()
    if not tag or "members" not in tag:
        return context(request_ids)
    roots = sorted({root for request in request_ids for root in tag["members"].get(request, ())})
    return dict(tag, roots=roots) if roots else None


def batch(step_id, prefill=(), decode=(), register=(), release=()):
    if not enabled():
        return None
    groups = dict(prefill=tuple(prefill), decode=tuple(decode), register=tuple(register), release=tuple(release))
    tags = {name: context(ids) for name, ids in groups.items()}
    tag = combine(tags.values())
    if tag:
        tag["batch"] = f"{_process}:{next(_batch_ids)}"
        tag["step_id"] = step_id
        tag["members"] = {request: _aliases[request]["roots"] for ids in groups.values()
                          for request in ids if request in _aliases}
        tag["parts"] = {name: value["roots"] if value else [] for name, value in tags.items()}
        tag["parts"]["compute"] = sorted(set(tag["parts"]["prefill"] + tag["parts"]["decode"]))
    return tag


def batch_pickup(command):
    tag = getattr(command, "request_profile", None)
    if tag and "parts" in tag:
        event("worker.batch.members", tag=tag, batch=tag["batch"], members=tag["members"], parts=tag["parts"])


def close(capture_id):
    capture = _sessions.get(capture_id)
    return capture.close() if capture is not None else True


def combine(tags):
    tags = [tag for tag in tags if tag]
    if not tags:
        return None
    result = dict(tags[0])
    result["roots"] = sorted({root for tag in tags if tag["capture"] == result["capture"]
                              for root in tag["roots"]})
    return result


def incoming(value):
    """Resolve a peer marker against a locally prepared session, never a peer path."""
    if not isinstance(value, dict):
        return None
    capture = _sessions.get(value.get("capture"))
    roots = value.get("roots")
    if capture is None or capture.closed or not isinstance(roots, list) or not roots:
        return None
    if len(roots) > 256 or any(not isinstance(r, str) or not re.fullmatch("[0-9a-f]{32}", r) for r in roots):
        return None
    return capture.wire(roots)


def marker():
    tag = current.get()
    return dict(capture=tag["capture"], roots=tag["roots"]) if tag else None


def invoke(tag, name, function, *args, **kwargs):
    with scope(tag, name):
        return function(*args, **kwargs)


def job_context(kind, identity):
    with _guard:
        job = _jobs.get((kind, str(identity)))
        return job[0] if job else None


def enabled():
    return any(not session.closed for session in _sessions.values())


def native_interval(name, start_ns, end_ns, **fields):
    tag = current.get()
    if tag:
        wall, mono = time.time_ns(), time.monotonic_ns()
        capture = ensure(tag)
        capture.emit(name, ph="X", cat="native-transfer", ts=(wall+start_ns-mono)/1000,
            dur=max(0, end_ns-start_ns)/1000, args=dict(roots=tag["roots"],
                clock_domain="linux_steady_clock_monotonic", original_start_ns=str(start_ns),
                original_end_ns=str(end_ns), anchor_wall_ns=str(wall), anchor_monotonic_ns=str(mono), **fields))


def incomplete(reason, tag=None):
    tag = tag or current.get()
    if tag:
        capture = ensure(tag)
        event("observation.incomplete", tag=tag, reason=reason)
        capture.failure = reason


def _after_fork():
    # Runtime owners may be forked lazily. No writer/lock/context from a parent is reusable there.
    global _process, _guard, _lanes
    _sessions.clear()
    _aliases.clear()
    _jobs.clear()
    _process, _guard, _lanes = uuid.uuid4().hex, threading.RLock(), threading.local()
    current.set(None)



if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)
