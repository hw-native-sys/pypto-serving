# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""K5 parent selection and physical retirement validation."""


from dataclasses import dataclass
import msgspec


@dataclass(frozen=True)
class NativeDecodeRetirement:
    reservation_id: int
    request_id: str
    generation: int
    physical_input_start: int
    physical_input_end: int
    advances: tuple[int, ...]
    accepted_drafts: tuple[int, ...]
    raw_output_tokens: int
    remaining_output_tokens: int
    next_epoch: int


@dataclass(frozen=True)
class RetirementEnvelope:
    retirement: NativeDecodeRetirement
    version: int = 1


def encode_retirement(receipt):
    return msgspec.msgpack.encode(RetirementEnvelope(receipt))


def decode_retirement(payload):
    envelope = msgspec.msgpack.decode(payload, type=RetirementEnvelope)
    if envelope.version != 1:
        raise ValueError("Unsupported lowlat retirement version")
    return envelope.retirement


def choose_native_parent(remaining, pending):
    """Return cycles and conservative remaining, or no legal admission.

    Pending reservations own their maximum output until physical retirement.
    A successor is admitted only behind a two-cycle parent with24real tokens
    still available. Algorithm K remains5; each parent is a separate step.
    """
    if type(remaining) is not int or remaining < 0 or len(pending) > 2:
        raise ValueError("invalid native remaining/pending accounting")
    if not remaining or len(pending) == 2:
        return None
    if pending:
        first = pending[0]
        if first.cycles != 2 or remaining < 24:
            return None
        available = remaining - first.max_output_tokens
        return (2, available) if available >= 12 else None
    return (2 if remaining >= 7 else 1, remaining)


def validate_native_retirement(
    receipt, reservation, *, previous_input_end, previous_raw_outputs, max_output_tokens, actual_token_count
):
    """Validate physical publication independently of optimistic scheduler cursors."""
    if (
        receipt.reservation_id != reservation.reservation_id
        or receipt.request_id != reservation.request_id
        or receipt.generation != reservation.generation
        or receipt.physical_input_start != previous_input_end
        or not receipt.physical_input_start < receipt.physical_input_end <= reservation.physical_input_end
        or len(receipt.advances) != reservation.cycles
        or len(receipt.accepted_drafts) != reservation.cycles
        or any(type(a) is not int or not 0 <= a <= 5 for a in receipt.accepted_drafts)
        or any(
            advance != accepted + 1 for advance, accepted in zip(receipt.advances, receipt.accepted_drafts)
        )
        or receipt.physical_input_end - receipt.physical_input_start != sum(receipt.advances)
        or receipt.raw_output_tokens != actual_token_count
        or not 1 <= actual_token_count <= reservation.max_output_tokens
        or receipt.remaining_output_tokens != max_output_tokens - previous_raw_outputs - actual_token_count
        or receipt.remaining_output_tokens < 0
        or receipt.next_epoch <= reservation.cycles
    ):
        raise ValueError("native physical retirement does not close its reservation")


def validate_reservation(reservation):
    reservation.__post_init__()
    if (
        reservation.cycles not in (1, 2)
        or reservation.frame_id not in (0, 1)
        or (reservation.cycles == 2 and reservation.remaining_lower_bound < 7)
        or reservation.max_output_tokens != min(6 * reservation.cycles, reservation.remaining_lower_bound)
    ):
        raise ValueError("invalid native decode reservation")
