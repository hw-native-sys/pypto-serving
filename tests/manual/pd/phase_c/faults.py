# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Test-only channel fault injection around the production owner service."""
from dataclasses import replace
import socket
import time

from pypto_serving.transfer.errors import ErrorCode, TransferError, TransferFailure
from pypto_serving.transfer.events import TransferEvent
from pypto_serving.transfer.types import CompletionCertainty, OwnerRef, Stage, TransferAttemptRef


class DeadPeerFence:
    """Caller has confirmed peer owner death before admitting any native write."""
    def wait(self, timeout):
        raise TransferFailure(TransferError(ErrorCode.OWNER_LOST, CompletionCertainty.NOT_SUBMITTED))


class LostReplyFactory:
    """Drop a successful WRITE reply in the child, after the real native write returned."""

    def __init__(self, factory):
        self.factory = factory

    def __call__(self, context):
        import pypto_serving.transfer.owner as owner_module
        original_send = owner_module._send
        def send(channel, reply):
            if reply.get("operation") == "write" and reply.get("ok"):
                print('{"kind":"fault_injected","fault":"lost_successful_native_reply"}', flush=True)
                channel.shutdown(socket.SHUT_RDWR)
                raise ConnectionError("test-only lost owner reply")
            return original_send(channel, reply)
        owner_module._send = send
        return self.factory(context)


class InflightFactory:
    """Observe a real native invocation; repeat identical writes to widen the fault window."""

    def __init__(self, factory, channel):
        self.factory, self.channel = factory, channel

    def __call__(self, context):
        import pypto_serving.transfer.owner as module
        original = module.MooncakeTransferProvider
        channel = self.channel

        class ObservedEngine:
            def __init__(self, engine):
                self.engine = engine
            def __getattr__(self, name):
                return getattr(self.engine, name)
            def batch_transfer_sync_write(self, endpoint, sources, destinations, lengths):
                # Only existing, registered ranges; repeated payloads are identical.
                sources, destinations, lengths = sources * 16384, destinations * 16384, lengths * 16384
                module._send(channel, dict(kind="native_enter", monotonic_ns=time.monotonic_ns(),
                                           descriptors=len(lengths), bytes=sum(lengths)))
                try:
                    return self.engine.batch_transfer_sync_write(endpoint, sources, destinations, lengths)
                finally:
                    module._send(channel, dict(kind="native_return", monotonic_ns=time.monotonic_ns()))

        class ObservedProvider(original):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._engine = ObservedEngine(self._engine)
        module.MooncakeTransferProvider = ObservedProvider
        return self.factory(context)


def decode_event(wire):
    ref = wire["attempt"]
    attempt = TransferAttemptRef(**dict(ref, source=OwnerRef(**ref["source"]),
                                      destination=OwnerRef(**ref["destination"])))
    error = wire["error"]
    return TransferEvent(**dict(wire, attempt=attempt, stage=Stage(wire["stage"]),
                                certainty=CompletionCertainty(wire["certainty"]),
                                error=TransferError(**dict(error, code=ErrorCode(error["code"]),
                                                          certainty=CompletionCertainty(error["certainty"])))))


def confirm_group(supervisor, message, expected):
    received = tuple(OwnerRef(**owner) for owner in message["owners"])
    if set(received) != set(expected) or len(received) != len(expected):
        raise ValueError("peer death acknowledgement does not cover the recovery group")
    for owner in received:
        supervisor.confirm_peer_dead(owner)
    fresh = replace(supervisor.owner, generation=supervisor.owner.generation + 1,
                    endpoint_generation=supervisor.owner.endpoint_generation + 1)
    supervisor.permit_rebuild(fresh)


def previous_generation_task(task):
    """Build a consistent old-generation envelope, not a malformed mixed-generation task."""
    def old(owner):
        if owner.generation < 2 or owner.endpoint_generation < 2:
            raise ValueError("previous generation requires a rebuilt owner")
        return replace(owner, generation=owner.generation - 1,
                       endpoint_generation=owner.endpoint_generation - 1)
    source, destination = old(task.attempt.source), old(task.attempt.destination)
    attempt = replace(task.attempt, source=source, destination=destination,
                      data_generation=task.attempt.data_generation - 1)
    segments = tuple(replace(segment, source=replace(segment.source, owner=source),
                             destination=replace(segment.destination, owner=destination))
                     for segment in task.segments)
    return replace(task, attempt=attempt, segments=segments)
