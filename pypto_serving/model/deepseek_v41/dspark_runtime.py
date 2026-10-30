# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Serving-side lifecycle for the V4.1 DSpark composite stages.

The lib owns the device implementation and the ordering of the composites.  This
module only owns request state, stage ordering, and the ABI hand-off.  It is
deliberately independent of PyPTO so host tests can exercise the contract
without allocating an NPU worker.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Protocol, Sequence


DSPARK_SPECULATIVE_TOKENS = 5
DSPARK_ROWS_PER_REQUEST = DSPARK_SPECULATIVE_TOKENS + 1

DECODE_STAGE_ORDER = (
    "prepare_target",
    "prepare_target_decode",
    "target_decode",
    "sample_target_chunk_0",
    "sample_target_chunk_1",
    "sample_target_chunk_2",
    "sample_target_chunk_3",
    "sample_target_chunk_4",
    "merge_target_chunks",
    "accept_target",
    "copy_target_hidden",
    "prepare_drafter_geometry",
    "dspark_drafter",
    "markov_sample",
    "commit_drafts",
)

PREFILL_STAGE_ORDER = (
    "bootstrap_inputs",
    "prepare_prefill_drafter_geometry",
    "dspark_drafter",
    "markov_sample",
    "publish_bootstrap",
)


class DSparkDispatch(Protocol):
    """The narrow resource boundary used by the lifecycle adapter."""

    def dispatch_dspark(self, stage: str, arguments: Mapping[str, Any]) -> None: ...


@dataclass(frozen=True)
class DSparkLoweredInputs:
    """Complete argument bundle for one lib-owned DSpark orchestration.

    Serving owns the lifetime of these tensors and the request-to-rank mapping;
    the lib owns the stage order and all model-side composition.  Keeping this
    bundle explicit prevents a partially populated batch from silently falling
    back to ordinary decode.
    """

    state: Any
    workspace: Mapping[str, Any]
    target_model: Mapping[str, Any] = field(default_factory=dict)
    draft_model: Mapping[str, Any] = field(default_factory=dict)
    markov_model: Mapping[str, Any] = field(default_factory=dict)
    cache_tables: Mapping[str, Any] = field(default_factory=dict)
    rope_tables: Mapping[str, Any] = field(default_factory=dict)
    attention_epoch: int = 1
    moe_epoch: int = 1
    logits_epoch: int = 1
    outputs: Mapping[str, Any] = field(default_factory=dict)
    request_ranks: Sequence[int] | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DSparkLoweredInputs":
        required = ("state", "workspace")
        missing = [name for name in required if name not in value]
        if missing:
            raise ValueError("DSpark lowered input is missing: " + ", ".join(missing))
        values = {name: value[name] for name in required}
        if not isinstance(values["workspace"], Mapping):
            raise TypeError("DSpark workspace must be a mapping")
        for name in ("target_model", "draft_model", "markov_model", "cache_tables", "rope_tables"):
            mapping = value.get(name, {})
            if not isinstance(mapping, Mapping):
                raise TypeError(f"DSpark {name} must be a mapping")
            values[name] = mapping
        for name in ("attention_epoch", "moe_epoch", "logits_epoch"):
            epoch = value.get(name, 1)
            if type(epoch) is not int or epoch <= 0:
                raise ValueError(f"DSpark {name} must be a positive integer")
            values[name] = epoch
        outputs = value.get("outputs", {})
        if not isinstance(outputs, Mapping):
            raise TypeError("DSpark outputs must be a mapping")
        values["outputs"] = outputs
        values["request_ranks"] = value.get("request_ranks")
        return cls(**values)


def coerce_dspark_lowered(value: Any) -> DSparkLoweredInputs:
    if isinstance(value, DSparkLoweredInputs):
        return value
    if isinstance(value, Mapping):
        return DSparkLoweredInputs.from_mapping(value)
    raise TypeError("DSpark lowered input must be a mapping or DSparkLoweredInputs")


@dataclass(frozen=True)
class DSparkPrefillDispatch:
    """Already-lowered DSpark bootstrap arguments carried by a prefill batch.

    The scheduler owns tensor allocation and argument lowering.  This value
    only gives the serving runner a typed hand-off; it never constructs a
    drafter or Markov operator from smaller serving-side operations.
    """

    request_id: str
    slot: int
    generation: int
    prompt_length: int
    anchor: int
    stages: Sequence[tuple[str, Mapping[str, Any]]]
    lowered: DSparkLoweredInputs | None = None

    @classmethod
    def from_batch(cls, batch: Any) -> "DSparkPrefillDispatch":
        result = _dispatch_from_batch(
            cls, batch,
            required=("request_id", "slot", "generation", "prompt_length", "anchor", "stages"),
            optional=("lowered",),
        )
        if result.lowered is not None:
            result = replace(result, lowered=coerce_dspark_lowered(result.lowered))
        return result


@dataclass(frozen=True)
class DSparkDecodeDispatch:
    """Already-lowered arguments for one target/accept/draft/Markov round."""

    request_ids: Sequence[str]
    stages: Sequence[tuple[str, Mapping[str, Any]]]
    accepted_counts: Sequence[int] | None = None
    last_sampled: Sequence[int] | None = None
    draft_ids: Sequence[Sequence[int]] | None = None
    lowered: DSparkLoweredInputs | None = None
    request_ranks: Sequence[int] | None = None

    @classmethod
    def from_batch(cls, batch: Any) -> "DSparkDecodeDispatch":
        result = _dispatch_from_batch(
            cls, batch, required=("request_ids", "stages"),
            optional=("accepted_counts", "last_sampled", "draft_ids", "lowered", "request_ranks"),
        )
        if result.lowered is not None:
            result = replace(result, lowered=coerce_dspark_lowered(result.lowered))
        return result


def _dispatch_from_batch(
    cls, batch: Any, *, required: tuple[str, ...], optional: tuple[str, ...] = (),
):
    payload = getattr(batch, "dspark_stages", None)
    if isinstance(payload, cls):
        return payload
    if not isinstance(payload, Mapping):
        raise TypeError(
            "DSpark batch hook requires dspark_stages to be a mapping or its dispatch value"
        )
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"DSpark batch hook is missing fields: {', '.join(missing)}")
    values = {field: payload[field] for field in required}
    values.update({field: payload.get(field) for field in optional})
    return cls(**values)


@dataclass
class DSparkRequestState:
    """Persistent state mirrored by the scheduler for one request slot."""

    slot: int
    generation: int
    prompt_length: int
    position_limit: int
    anchor: int = 0
    committed: int = 0
    draft_ids: tuple[int, ...] = ()
    valid: bool = False

    def bootstrap(self, prompt_length: int, anchor: int) -> None:
        if prompt_length <= 0 or prompt_length > self.position_limit:
            raise ValueError("DSpark prompt length is outside the position limit")
        # Prefill has no previously emitted token: its next-token anchor is
        # allowed at the prompt boundary. Recurrent decode anchors are inside
        # the committed stream and advance from this position.
        if not 0 <= anchor <= prompt_length:
            raise ValueError("DSpark bootstrap anchor must not exceed the prompt boundary")
        self.prompt_length = prompt_length
        self.anchor = anchor
        self.committed = prompt_length
        self.draft_ids = ()
        self.valid = True

    def accept(self, accepted_count: int, last_sampled: int) -> None:
        if not self.valid:
            raise RuntimeError("cannot accept tokens for an uninitialized DSpark state")
        self.validate_accept(accepted_count, last_sampled)
        self.anchor += accepted_count
        self.committed += accepted_count
        self.draft_ids = ()

    def validate_accept(self, accepted_count: int, last_sampled: int) -> None:
        """Validate the next acceptance without changing request state."""
        if not isinstance(accepted_count, int) or not 1 <= accepted_count <= DSPARK_ROWS_PER_REQUEST:
            raise ValueError("DSpark accepted count must contain the target token")
        next_anchor = self.anchor + accepted_count
        if next_anchor >= self.position_limit:
            # There is no next draft anchor at the position limit.
            return
        if not isinstance(last_sampled, int) or isinstance(last_sampled, bool) or last_sampled < 0:
            raise ValueError("DSpark last sampled token must be a nonnegative token ID")

    def commit(self, draft_ids: Sequence[int]) -> None:
        if not self.valid:
            raise RuntimeError("cannot commit drafts for an uninitialized DSpark state")
        if len(draft_ids) != DSPARK_SPECULATIVE_TOKENS:
            raise ValueError("DSpark must commit exactly five draft tokens")
        if self.anchor + DSPARK_ROWS_PER_REQUEST >= self.position_limit:
            self.draft_ids = ()
            return
        self.draft_ids = tuple(int(token) for token in draft_ids)


@dataclass
class DSparkStateTable:
    """Stable request-to-slot mapping across prefill and recurrent decode."""

    position_limit: int
    requests: dict[str, DSparkRequestState] = field(default_factory=dict)

    def reserve(self, request_id: str, *, slot: int, generation: int, prompt_length: int) -> DSparkRequestState:
        if not request_id or request_id in self.requests:
            raise ValueError("DSpark request IDs must be nonempty and unique")
        state = DSparkRequestState(slot, generation, prompt_length, self.position_limit)
        self.requests[request_id] = state
        return state

    def get(self, request_id: str) -> DSparkRequestState:
        try:
            return self.requests[request_id]
        except KeyError as exc:
            raise RuntimeError(f"DSpark state is missing for request {request_id!r}") from exc

    def release(self, request_ids: Sequence[str]) -> None:
        for request_id in request_ids:
            self.requests.pop(request_id, None)


class DSparkStageRunner:
    """Run the exact lib stage sequence through the resource dispatch seam."""

    def __init__(self, dispatch: DSparkDispatch):
        self.dispatch = dispatch

    def run(self, stages: Sequence[tuple[str, Mapping[str, Any]]], *, phase: str) -> None:
        if phase not in ("prefill", "decode"):
            raise ValueError("DSpark phase must be 'prefill' or 'decode'")
        expected = PREFILL_STAGE_ORDER if phase == "prefill" else DECODE_STAGE_ORDER
        names = tuple(name for name, _ in stages)
        if names != expected:
            raise ValueError(f"DSpark {phase} stages must follow the lib order: {expected}")
        for name, arguments in stages:
            self.dispatch.dispatch_dspark(name, arguments)


class V41DSparkRuntime:
    """Own state and dispatch ordering for one V4.1 resident worker."""

    def __init__(self, resources: DSparkDispatch, *, max_seq_len: int):
        if not isinstance(max_seq_len, int) or max_seq_len <= 0:
            raise ValueError("DSpark position limit must be positive")
        self.states = DSparkStateTable(max_seq_len)
        self.stages = DSparkStageRunner(resources)

    def bootstrap(self, request_id: str, *, slot: int, generation: int,
                  prompt_length: int, anchor: int,
                  stages: Sequence[tuple[str, Mapping[str, Any]]]) -> DSparkRequestState:
        state = self.states.reserve(
            request_id, slot=slot, generation=generation, prompt_length=prompt_length,
        )
        try:
            state.bootstrap(prompt_length, anchor)
            self.stages.run(stages, phase="prefill")
        except BaseException:
            self.states.release((request_id,))
            raise
        return state

    def record_bootstrap(self, request_id: str, *, slot: int, generation: int,
                         prompt_length: int, anchor: int,
                         draft_ids: Sequence[int]) -> DSparkRequestState:
        """Commit outputs already produced by the lib bootstrap composite.

        The device orchestration path has its own runner and therefore must not
        dispatch the stage list a second time merely to update host bookkeeping.
        """
        state = self.states.reserve(
            request_id, slot=slot, generation=generation, prompt_length=prompt_length,
        )
        try:
            state.bootstrap(prompt_length, anchor)
            state.commit(draft_ids)
        except BaseException:
            self.states.release((request_id,))
            raise
        return state

    def decode(self, request_ids: Sequence[str],
               stages: Sequence[tuple[str, Mapping[str, Any]]], *,
               accepted_counts: Sequence[int] | None = None,
               last_sampled: Sequence[int] | None = None,
               draft_ids: Sequence[Sequence[int]] | None = None) -> None:
        for request_id in request_ids:
            self.states.get(request_id)
        if accepted_counts is not None or last_sampled is not None:
            if accepted_counts is None or last_sampled is None:
                raise ValueError("accepted_counts and last_sampled must be supplied together")
            if len(accepted_counts) != len(request_ids) or len(last_sampled) != len(request_ids):
                raise ValueError("DSpark acceptance state must match request order")
        if draft_ids is not None:
            if len(draft_ids) != len(request_ids):
                raise ValueError("DSpark draft state must match request order")
            if any(len(drafts) != DSPARK_SPECULATIVE_TOKENS for drafts in draft_ids):
                raise ValueError("DSpark must commit exactly five draft tokens")
        self.stages.run(stages, phase="decode")
        self.apply_decode_outputs(
            request_ids,
            accepted_counts=accepted_counts,
            last_sampled=last_sampled,
            draft_ids=draft_ids,
        )

    def apply_decode_outputs(
        self,
        request_ids: Sequence[str],
        *,
        accepted_counts: Sequence[int] | None = None,
        last_sampled: Sequence[int] | None = None,
        draft_ids: Sequence[Sequence[int]] | None = None,
    ) -> None:
        """Apply device-produced acceptance and draft outputs after dispatch.

        Serving's resident worker may need to dispatch the stages directly in
        order to retain device completion ownership.  Keeping the state update
        separate from :meth:`decode` lets that path reuse the same validation
        rules without dispatching the stage sequence twice.
        """
        if accepted_counts is not None or last_sampled is not None:
            if accepted_counts is None or last_sampled is None:
                raise ValueError("accepted_counts and last_sampled must be supplied together")
            if len(accepted_counts) != len(request_ids) or len(last_sampled) != len(request_ids):
                raise ValueError("DSpark acceptance state must match request order")
            for request_id, count, token in zip(request_ids, accepted_counts, last_sampled, strict=True):
                self.states.get(request_id).validate_accept(int(count), int(token))
        if accepted_counts is not None or last_sampled is not None:
            for request_id, count, token in zip(request_ids, accepted_counts, last_sampled, strict=True):
                self.states.get(request_id).accept(int(count), int(token))
        if draft_ids is not None:
            for request_id, drafts in zip(request_ids, draft_ids, strict=True):
                self.states.get(request_id).commit(drafts)

    def release(self, request_ids: Sequence[str]) -> None:
        self.states.release(request_ids)
