# Copyright (c) PyPTO Contributors.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""Formal fixed-pair directory and compatibility gate."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from pypto_serving.serving.pd.config import PDRole
from pypto_serving.serving.pd.http_api import NodeDescriptor

from .client import NodeClient


@dataclass(frozen=True)
class SelectedPair:
    prefill: NodeDescriptor
    decode: NodeDescriptor
    prefill_client: NodeClient
    decode_client: NodeClient


class FixedWorkerDirectory:
    """Phase E directory seam; Phase F may replace selection without changing RPCs."""

    def __init__(
        self,
        prefill_client: NodeClient,
        decode_client: NodeClient,
        run_id: str,
        control_incarnation: int | None = None,
    ) -> None:
        self.prefill_client = prefill_client
        self.decode_client = decode_client
        self.run_id = run_id
        self.control_incarnation = control_incarnation
        self._pair: SelectedPair | None = None

    async def refresh(self) -> SelectedPair:
        prefill, decode = await asyncio.gather(
            self.prefill_client.get("/internal/pd/descriptor", NodeDescriptor),
            self.decode_client.get("/internal/pd/descriptor", NodeDescriptor),
        )
        self._validate(prefill, decode)
        self._pair = SelectedPair(
            prefill=prefill,
            decode=decode,
            prefill_client=self.prefill_client,
            decode_client=self.decode_client,
        )
        return self._pair

    async def select(self) -> SelectedPair:
        # Descriptors are cheap control-plane facts. Refreshing per request also
        # makes an endpoint-generation change fail before any reservation.
        return await self.refresh()

    def _validate(self, prefill: NodeDescriptor, decode: NodeDescriptor) -> None:
        if prefill.role != PDRole.PREFILL.value or decode.role != PDRole.DECODE.value:
            raise ValueError("Router endpoints do not form a P/D pair")
        if prefill.run_id != self.run_id or decode.run_id != self.run_id:
            raise ValueError("Router and P/D run_id differ")
        if prefill.node_id == decode.node_id:
            raise ValueError("Router P/D node identities must differ")
        if prefill.control_incarnation != decode.control_incarnation:
            raise ValueError("P/D control incarnations differ")
        if (
            self.control_incarnation is not None
            and prefill.control_incarnation != self.control_incarnation
        ):
            raise ValueError("Router and P/D control incarnations differ")
        if prefill.health != "READY" or decode.health != "READY":
            raise RuntimeError("P/D pair is not healthy")
        p_caps = prefill.capabilities
        d_caps = decode.capabilities
        comparable = (
            "schema_version",
            "adapter_id",
            "contract_version",
            "contract_digest",
            "continuation_schema",
            "model_revision",
            "layout_fingerprint",
            "topology",
            "provider",
            "logical_groups",
            "physical_regions",
        )
        if any(getattr(p_caps, name) != getattr(d_caps, name) for name in comparable):
            raise ValueError("P/D capability or cache-layout contract differs")
