# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""V4-style runner lifecycle for explicit V4.1 composite bindings."""
from .composite import BuildOptions, CompositeBindings, LayerState
from .input_preparation import lookup_token_embeddings
import torch
from pypto_serving.config.types import PrefillResult, DecodeResult
from .execution_plan import V41ExecutionPlan
from .metadata import prefill_requests, decode_requests
from .request_state import RequestLedger
from threading import RLock


class V41ModelRunner:
    """Own one collective session; missing entries fail before resource creation.

    This follows the shared runner lifecycle but deliberately does not inherit
    its generic K/V allocator: V4.1 pools include index and compressor state.
    """
    def __init__(self, plan: V41ExecutionPlan, bindings: CompositeBindings, *, device_ids, runtime,
                 build_options: BuildOptions = BuildOptions()):
        self.plan, self.bindings = plan, bindings
        self.device_ids = tuple(device_ids)
        if len(self.device_ids) != plan.placement.ep_size or len(set(self.device_ids)) != len(self.device_ids):
            raise ValueError("one distinct physical device is required for each logical EP rank")
        if any(type(i) is not int or i < 0 for i in self.device_ids):
            raise ValueError("device IDs must be nonnegative integers")
        bindings.require(plan.layers, plan.placement)
        self.runtime = runtime
        self.build_options = build_options
        self.resources = None
        self.num_pages = None
        self.closed = False
        self.failed = False
        self.ledger = None
        self._lock = RLock()
        self._plans = None

    def preflight(self):
        if self.failed:
            raise RuntimeError("runner initialization failed; close the session before retrying")
        if self.closed:
            raise RuntimeError("runner is closed")
        if self.resources is not None:
            return self.num_pages
        try:
            resources, pages = self.bindings.allocate(
                self.plan, self.device_ids, self.runtime, self.build_options)
            self.resources = resources
            if resources is None or type(pages) is not int or pages <= 0:
                raise ValueError("composite allocator must return resources and a positive page capacity")
            self.bindings.wait(resources)
            self.num_pages = pages
        except Exception:
            self.failed = True
            self.close()
            raise
        return self.num_pages

    def close(self):
        if self.closed:
            return
        if self.resources is not None:
            # Do not free or reuse storage if completion itself fails.
            self.bindings.wait(self.resources)
            self.bindings.close(self.resources)
            self.resources = None
        self.closed = True

    def _request_ledger(self):
        self.preflight()
        if self.ledger is None:
            self.ledger = RequestLedger(max_requests=self.runtime.max_batch_size,
                                        max_seq_len=self.runtime.max_seq_len)
        return self.ledger

    def _reset_request(self, key, owner):
        self.bindings.reset_request(self.resources, key, owner)
        self.bindings.wait(self.resources)

    def run_prefill(self, model, batch):
        with self._lock:
            ledger = self._request_ledger()
            requests = prefill_requests(batch, model.config, self.runtime, self.bindings.cache_groups)
            step = ledger.begin_prefill(requests)
            return self._run_transaction(step, batch.input_embeddings)

    def _run_transaction(self, step, embeddings):
        try:
            result = self._execute_step(step, embeddings)
            self.bindings.wait(self.resources)
            self.ledger.commit(step)
            return result
        except Exception:
            try:
                self.bindings.wait(self.resources)
            except Exception:
                self.failed = True
                self.ledger.poisoned = True
                # Keep pending state and buffers owned: completion is unknown.
                raise
            self.ledger.abort(step, self._reset_request)
            raise

    def _rank_plans(self):
        if self._plans is None:
            self._plans = tuple(self.plan.for_rank(rank) for rank in range(self.plan.placement.ep_size))
        return self._plans

    def lookup_embeddings(self, token_ids):
        # A complete TP vocabulary exists in each DP group; read it once on host.
        plans = self._rank_plans()[:self.plan.placement.tp_size]
        return lookup_token_embeddings([plan.weights for plan in plans], token_ids)

    def _check_state(self, state):
        if not isinstance(state, LayerState) or state.residual is None or state.pre_mix is None:
            raise ValueError("composite must return both residual and delayed pre_mix")
        if state.layout != self.bindings.output_layout:
            raise ValueError("composite output token layout differs from the next layer input")
        return state

    def _run_layers(self, step, embeddings):
        config = self.plan.weights.config
        if embeddings is None:
            embeddings = self.lookup_embeddings(torch.tensor(step.token_ids, dtype=torch.int64))
        if (not isinstance(embeddings, torch.Tensor) or embeddings.device.type != "cpu"
                or embeddings.dtype != torch.bfloat16
                or tuple(embeddings.shape) != (len(step.token_ids), config.hidden_size)):
            raise ValueError("input embeddings must be packed CPU BF16 [active_tokens, hidden_size]")
        plans = self._rank_plans()
        state = self.bindings.initialize(embeddings, step, self.resources)
        self.bindings.wait(self.resources)
        self._check_state(state)
        for layer in self.plan.layers:
            # The adapter chooses bounded staging or residency; payloads stay packed.
            weights = self.bindings.prepare_weights(plans, layer, self.resources)
            self.bindings.wait(self.resources)
            state = self.bindings.entries[(step.phase, layer.mode)](
                layer, state, step, self.resources, weights)
            self.bindings.wait(self.resources)
            self._check_state(state)
        return state

    def _execute_step(self, step, embeddings):
        state = self._run_layers(step, embeddings)
        return self._finish_step(state, step)

    def _finish_step(self, state, step):
        logits = self.bindings.output(state, step, self.resources)
        self.bindings.wait(self.resources)
        shape = (len(step.requests), self.plan.weights.config.vocab_size)
        if (not isinstance(logits, torch.Tensor) or logits.device.type != "cpu"
                or tuple(logits.shape) != shape or logits.dtype not in (
                    torch.float32, torch.bfloat16, torch.float16)):
            raise ValueError("output composite must return CPU logits [requests, vocabulary] in request order")
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("output composite returned non-finite logits")
        # Shared workers may sample after the next dispatch reuses device output
        # scratch. Keep returned host logits independent of adapter-owned buffers.
        owned_logits = logits.detach().clone()
        if step.phase == "prefill":
            return PrefillResult(last_hidden=None, logits=owned_logits)
        return DecodeResult(hidden_states=None, logits=owned_logits)

    def run_decode(self, model, batch):
        with self._lock:
            ledger = self._request_ledger()
            requests = decode_requests(batch, model.config, self.runtime, self.bindings.cache_groups)
            step = ledger.begin_decode(requests)
            return self._run_transaction(step, batch.hidden_states)

    def release_finished_requests(self, request_ids):
        with self._lock:
            if self.ledger is not None:
                self.bindings.wait(self.resources)
                self.ledger.release(request_ids, self._reset_request)
