# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Correlated lowlat spans; disabled profiling preserves the original calls."""

from functools import wraps

from pypto_serving.tools.profile import is_enabled, profile_span
from .decoding import decode_reservation_payload


def handle_fields(handle):
    # The pinned runtime emits the native run ID on every chip.run root.
    # Read it before result() retires the underlying native handle.
    native = getattr(handle, "_native_handle", None)
    values = {"runtime_run_id": getattr(native, "_run_id", None),
              "python_dispatch_id": getattr(handle, "_dispatch_id", None)}
    return {name: value for name, value in values.items() if type(value) is int}


def reservation_fields(value):
    reservation = getattr(value, "reservation", None)
    if reservation is None:
        reservation = getattr(getattr(value, "prepared", None), "reservation", None)
    if reservation is None:
        reservations = getattr(value, "model_extensions", ())
        if isinstance(reservations, tuple) and len(reservations) == 1:
            payload = reservations[0]
            if payload:
                reservation = decode_reservation_payload(payload)
    if reservation is None:
        return {}
    return {name: item for name, item in vars(reservation).items()
            if type(item) in (int, str)}


def packet_fields(packet):
    return {"emitted_ids": list(packet.token_ids), "emitted": len(packet.token_ids),
            "advances": list(packet.advances), "accepted_drafts": list(packet.accepted),
            "epoch": packet.controls[0][0], "next_epoch": packet.state.epoch,
            "physical_input_start": packet.controls[0][2]+1,
            "physical_input_end": packet.state.anchor+1,
            "remaining_output_tokens": packet.state.remaining}


def operation(name, *, category="lowlat"):
    """Trace the actual call, retaining its return value and exception behavior.

    Attributes are collected only while recording. Native submit spans carry
    the runtime identity; captures carry immutable output snapshots. Neither
    operation introduces a wait or reads device tensors for instrumentation.
    """
    def decorate(function):
        @wraps(function)
        def call(owner, *args, **kwargs):
            if not is_enabled("verbose"):
                return function(owner, *args, **kwargs)
            fields = {}
            state = getattr(owner, "state", None)
            for key in ("request_id", "generation"):
                value = getattr(state, key, None)
                if type(value) in (int, str):
                    fields[key] = value
            for value in args:
                fields.update(reservation_fields(value))
            for key in ("generation", "cycles"):
                if key in kwargs:
                    fields[key] = kwargs[key]
            if name.startswith("lowlat.native.") and args and type(args[0]) is int:
                fields["frame_id"] = args[0]
                frames = getattr(owner, "frames", ())
                if isinstance(frames, list) and 0 <= args[0] < len(frames):
                    fields.update(handle_fields(frames[args[0]].handle))
            fields.update(handle_fields(getattr(owner, "handle", None)))
            with profile_span(name, cat=category, level="verbose", args=fields):
                try:
                    result = function(owner, *args, **kwargs)
                    fields.update(handle_fields(getattr(result, "handle", result)))
                    fields.update(reservation_fields(result))
                    if name == "lowlat.native.capture":
                        fields.update(packet_fields(result))
                    fields["success"] = True
                    return result
                except BaseException:
                    fields["success"] = False
                    raise
        return call
    return decorate
