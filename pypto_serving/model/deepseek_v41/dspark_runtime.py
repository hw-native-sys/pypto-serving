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

# These names are the serving-to-lib handoff, rather than an implementation
# detail of any one runner.  Keep the contract explicit so a provider cannot
# accidentally dispatch a stage with only a subset of the six-row buffers.
_STATE_KEYS = ("slot_ids", "generations", "tokens", "meta")
_PREFILL_WORKSPACE_KEYS = (
    "prompt_lengths", "position_limits", "sampled_rows", "prefill_sampled_ids",
    "next_prefill_tokens", "first_query_positions", "prefill_positions",
    "prefill_request_ids", "prefill_counts", "context_slots", "context_rope_cos",
    "context_rope_sin", "query_rope_cos", "query_rope_sin", "prefill_target_hidden",
    "zero_sampled", "zero_last_sampled", "draft_head_hidden", "draft_ids",
    "confidences",
)
_DECODE_WORKSPACE_KEYS = (
    "input_ids", "target_positions", "active_widths", "x_hc", "pre_mix",
    "position_ids", "request_ids", "window_slots", "window_indices", "c2a_slots",
    "c1a_slots", "query_starts", "target_hidden_padded", "target_hidden",
    "target_sampled_ids", "accepted_ids", "accepted_counts", "context_valid",
    "context_positions", "num_sampled", "last_sampled", "anchor_ids",
    "first_query_positions", "context_slots", "context_rope_cos", "context_rope_sin",
    "query_rope_cos", "query_rope_sin", "draft_head_hidden", "draft_ids",
    "confidences", "zero_prefill_tokens", "committed",
    "chunk_0", "chunk_1", "chunk_2", "chunk_3", "chunk_4",
)
_PREFILL_CACHE_KEYS = ("draft",)
_DECODE_CACHE_KEYS = ("target_window", "target_c2a", "target_c1a", "draft")
_ROPE_KEYS = ("draft_cos", "draft_sin")

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


class DSparkLoweringProvider(Protocol):
    """Serving-owned lowering boundary for complete DSpark ABI payloads.

    The provider allocates and binds all state, cache, workspace, weight and
    output mappings required by the lib composite.  It must not return a
    partially populated stage list: the runner validates the returned typed
    dispatch before invoking the lib orchestrator.
    """

    def lower_prefill(
        self, *, model: Any, batch: Any, request_ids: Sequence[str],
        sampled_token_ids: Sequence[int],
    ) -> "DSparkPrefillDispatch": ...

    def lower_decode(
        self, *, model: Any, batch: Any,
    ) -> "DSparkDecodeDispatch": ...


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

    def validate(self, *, phase: str) -> None:
        """Validate the complete serving/lib resource handoff before dispatch.

        This checks names and container shape only.  Tensor dtype, topology and
        device ownership remain the responsibility of the compiled lib ABI and
        the resident resource allocator.  In particular, this method never
        creates a placeholder tensor for a missing binding.
        """
        if phase not in ("prefill", "decode"):
            raise ValueError("DSpark phase must be 'prefill' or 'decode'")

        def has_field(container, name):
            if isinstance(container, Mapping):
                return name in container
            return hasattr(container, name)

        missing_state = [name for name in _STATE_KEYS if not has_field(self.state, name)]
        if missing_state:
            raise ValueError("DSpark state is missing: " + ", ".join(missing_state))
        workspace_keys = _PREFILL_WORKSPACE_KEYS if phase == "prefill" else _DECODE_WORKSPACE_KEYS
        missing_workspace = [name for name in workspace_keys if name not in self.workspace]
        if missing_workspace:
            raise ValueError(
                f"DSpark {phase} workspace is missing: " + ", ".join(missing_workspace)
            )
        cache_keys = _PREFILL_CACHE_KEYS if phase == "prefill" else _DECODE_CACHE_KEYS
        missing_cache = [name for name in cache_keys if name not in self.cache_tables]
        if missing_cache:
            raise ValueError(
                f"DSpark {phase} cache tables are missing: " + ", ".join(missing_cache)
            )
        missing_rope = [name for name in _ROPE_KEYS if name not in self.rope_tables]
        if missing_rope:
            raise ValueError("DSpark rope tables are missing: " + ", ".join(missing_rope))
        if phase == "decode" and not self.target_model:
            raise ValueError("DSpark decode target model mapping is empty")
        if not self.draft_model:
            raise ValueError("DSpark draft model mapping is empty")
        if not self.markov_model:
            raise ValueError("DSpark Markov model mapping is empty")


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
        # Validate device-produced acceptance values before dispatching any
        # stage.  A malformed sample must not leave a partially executed
        # target/drafter chain or mutate persistent request state.
        if accepted_counts is not None:
            for request_id, count, token in zip(request_ids, accepted_counts, last_sampled, strict=True):
                self.states.get(request_id).validate_accept(int(count), int(token))
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
