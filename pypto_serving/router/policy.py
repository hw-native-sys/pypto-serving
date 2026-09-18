# Copyright (c) PyPTO Contributors.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""Allowlisted, replaceable Router placement policies."""

from __future__ import annotations

import asyncio
from typing import Protocol


class RoutePolicy(Protocol):
    async def select(
        self,
        compatible_pairs,
        *,
        prefill_node_id: str = "",
        excluded_decode_node_ids: frozenset[str] = frozenset(),
    ): ...


class RoundRobinRoutePolicy:
    """Select compatible P/D nodes using independent serialized cursors."""

    def __init__(self) -> None:
        self._prefill_cursor = 0
        self._decode_cursor = 0
        self._lock = asyncio.Lock()

    async def select(
        self,
        compatible_pairs,
        *,
        prefill_node_id: str = "",
        excluded_decode_node_ids: frozenset[str] = frozenset(),
    ):
        pairs = tuple(
            (prefill, decode)
            for prefill, decode in compatible_pairs
            if (
                not prefill_node_id
                or prefill.descriptor.node_id == prefill_node_id
            )
            and decode.descriptor.node_id not in excluded_decode_node_ids
        )
        if not pairs:
            raise RuntimeError("route policy has no eligible compatible P/D pair")
        async with self._lock:
            prefills = tuple(
                dict.fromkeys(prefill.descriptor.node_id for prefill, _ in pairs)
            )
            selected_prefill_id = (
                prefill_node_id
                if prefill_node_id
                else prefills[self._prefill_cursor % len(prefills)]
            )
            if not prefill_node_id:
                self._prefill_cursor += 1
            decode_pairs = tuple(
                pair for pair in pairs if pair[0].descriptor.node_id == selected_prefill_id
            )
            selected = decode_pairs[self._decode_cursor % len(decode_pairs)]
            self._decode_cursor += 1
            return selected


_POLICY_FACTORIES = {
    "round_robin": RoundRobinRoutePolicy,
}


def create_route_policy(name: str) -> RoutePolicy:
    try:
        factory = _POLICY_FACTORIES[name]
    except KeyError as exc:
        raise ValueError(f"unknown PD route policy {name!r}") from exc
    return factory()
