# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Request-local Qwen3.8 DeltaNet/KV state and asynchronous whole-network dispatch."""

from __future__ import annotations

import ctypes
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from pypto_serving.config.types import (
    DecodeBatch, DecodeResult, ModelConfig, ModelRecord, PrefillBatch, PrefillResult,
    RuntimeConfig, RuntimeModel,
)
from pypto_serving.model.common.runner.buffer_set import StaticDeviceTensor, resolve_l3_arg
from pypto_serving.model.common.runner.l3_dispatch import L3DispatchMixin, PendingL3Dispatch
from pypto_serving.model.common.runner.model_runner import ModelRunner
from pypto_serving.model.qwen3_5.task_args import (
    Qwen35HostSlot, STATE_SPECS, allocate_host_slots, allocate_zero_states,
)
from pypto_serving.model.qwen3_5.weight_spec import MAX_SEQUENCE

if TYPE_CHECKING:
    from pypto_serving.model.qwen3_5.npu_executor import Qwen35Compiled


@dataclass(eq=False)
class _Prepared:
    request_id: str
    slot: Qwen35HostSlot
    position: int
    stage: str


@dataclass(eq=False)
class _Pending:
    prepared: _Prepared
    dispatch: PendingL3Dispatch


class Qwen35ModelRunner(L3DispatchMixin, ModelRunner):
    """Own device-resident banks and one request's contiguous state handles."""

    prefill_tail_requires_reclaim_before_dispatch = True

    @property
    def supports_async_prefill_reclaim(self) -> bool:
        return True

    @property
    def supports_async_decode_reclaim(self) -> bool:
        return True

    def __init__(self, *, compiled: Qwen35Compiled, device_id: int) -> None:
        super().__init__()
        self._compiled = compiled
        self._device_id = device_id
        self._init_l3_dispatch(stacked=False)
        self._prefill_slots: tuple[Qwen35HostSlot, ...] = ()
        self._decode_slots: tuple[Qwen35HostSlot, ...] = ()
        self._zero_states: tuple[torch.Tensor, ...] = ()
        self._static_args: tuple[Any, ...] = ()
        self._request_states: dict[str, tuple[Any, ...]] = {}
        self._pending: list[_Pending] = []
        self._reservations: list[_Prepared] = []

    def _ensure_l3_shared_buffers(self) -> None:
        if self._prefill_slots:
            return
        # All host-shared allocations, including 23 large banks, precede the worker fork.
        self._prefill_slots = allocate_host_slots()
        self._decode_slots = allocate_host_slots()
        self._zero_states = allocate_zero_states()
        self._static_args = tuple(StaticDeviceTensor(tensor) for tensor in (
            *self._compiled.weights, self._compiled.cos, self._compiled.sin,
        ))

    def _shared_l3_worker(self) -> Any:
        worker = self._l3_worker
        if worker is None:
            if not self._prefill_slots or not self._zero_states:
                raise RuntimeError("Qwen3.8 shared buffers must exist before worker fork")
            from pypto.runtime import DistributedWorker

            worker = DistributedWorker([
                self._compiled.prefill.compiled, self._compiled.decode.compiled,
            ])
            self._l3_worker = worker
        return worker

    def init_kv_cache(self, model_id: str, config: ModelConfig, runtime: RuntimeConfig,
                      *, num_pages: int | None = None) -> int:
        """Bridge one scheduler page to request-local contiguous NPU state."""
        del model_id, config, num_pages
        if runtime.max_batch_size != 1 or runtime.max_seq_len > MAX_SEQUENCE or runtime.page_size != 128:
            raise ValueError("Qwen3.8 requires one 128-position scheduler page")
        self._ensure_l3_shared_buffers()
        self._configure_l3_rings(runtime)
        worker = self._shared_l3_worker()
        for arg in self._static_args:
            resolve_l3_arg(worker, arg, self._l3_static_tensors, stacked=False)
        return 1

    def preflight(self, record: ModelRecord) -> None:
        del record
        if len(self._l3_static_tensors) != 25:
            raise RuntimeError("Qwen3.8 weights and rotary tables were not uploaded")

    def _alloc_kv_cache_tensor(self, shape: tuple[int, ...], dtype: torch.dtype):
        return self._shared_l3_worker().alloc_tensor(shape, dtype)

    def _free_kv_cache_tensor(self, tensor: Any) -> None:
        self._shared_l3_worker().free_tensor(tensor)

    def _new_request_state(self, request_id: str) -> tuple[Any, ...]:
        if request_id in self._request_states:
            raise ValueError(f"Qwen3.8 state already exists for {request_id}")
        if self._request_states:
            raise ValueError("Qwen3.8 supports one active request")
        worker = self._shared_l3_worker()
        states: list[Any] = []
        try:
            for (shape, dtype), zero in zip(STATE_SPECS, self._zero_states, strict=True):
                states.append(worker.alloc_tensor(shape, dtype, init=zero))
        except BaseException:
            for tensor in states:
                worker.free_tensor(tensor)
            raise
        result = tuple(states)
        self._request_states[request_id] = result
        return result

    def release_finished_requests(self, request_ids: Iterable[str]) -> None:
        for request_id in request_ids:
            error: BaseException | None = None
            for pending in tuple(self._pending):
                if pending.prepared.request_id == request_id:
                    try:
                        pending.dispatch.wait()
                    except BaseException as exc:
                        error = exc
                    finally:
                        pending.prepared.slot.busy = False
                        self._pending.remove(pending)
            for prepared in tuple(self._reservations):
                if prepared.request_id == request_id:
                    prepared.slot.busy = False
                    self._reservations.remove(prepared)
            states = self._request_states.pop(request_id, ())
            for tensor in states:
                try:
                    self._shared_l3_worker().free_tensor(tensor)
                except BaseException as exc:
                    error = error or exc
            if error is not None:
                raise error

    @staticmethod
    def _slot(slots: tuple[Qwen35HostSlot, ...], index: int) -> Qwen35HostSlot:
        if not 0 <= index < len(slots):
            raise ValueError(f"invalid Qwen3.8 buffer slot {index}")
        slot = slots[index]
        if slot.busy:
            raise RuntimeError(f"Qwen3.8 buffer slot {index} is pending reclaim")
        slot.busy = True
        return slot

    def prepare_prefill(self, model: RuntimeModel, batch: PrefillBatch, *, buffer_slot: int) -> _Prepared:
        del model
        if len(batch.request_ids) != 1 or len(batch.seq_lens) != 1:
            raise ValueError("Qwen3.8 prefill accepts exactly one request")
        count = int(batch.seq_lens[0])
        if (not 1 <= count <= MAX_SEQUENCE or int(batch.chunk_starts[0]) != 0
                or int(batch.chunk_lens[0]) != count or batch.token_ids.numel() != count):
            raise ValueError("Qwen3.8 requires an unchunked prompt within 128 positions")
        slot = self._slot(self._prefill_slots, buffer_slot)
        try:
            slot.token_ids.zero_()
            slot.token_ids[:count].copy_(batch.token_ids.reshape(-1).to(dtype=torch.int64, device="cpu"))
            prepared = _Prepared(batch.request_ids[0], slot, count, "prefill")
            self._reservations.append(prepared)
            return prepared
        except BaseException:
            slot.busy = False
            raise

    def dispatch_prepared_prefill(
        self, model: RuntimeModel, batch: PrefillBatch, prepared: _Prepared,
    ) -> _Pending:
        del model, batch
        if prepared.stage != "prefill" or not prepared.slot.busy or prepared not in self._reservations:
            raise ValueError("invalid prepared Qwen3.8 prefill")
        try:
            states = self._new_request_state(prepared.request_id)
            dispatch = self._submit_l3(
                self._compiled.prefill, prepared.slot.token_ids,
                ctypes.c_int64(prepared.position), *self._static_args, *states,
                prepared.slot.logits,
            )
        except BaseException:
            prepared.slot.busy = False
            self._reservations.remove(prepared)
            if prepared.request_id in self._request_states:
                self.release_finished_requests([prepared.request_id])
            raise
        self._reservations.remove(prepared)
        pending = _Pending(prepared, dispatch)
        self._pending.append(pending)
        return pending

    def reclaim_prepared_prefill(self, pending: _Pending) -> PrefillResult:
        try:
            pending.dispatch.wait()
            return PrefillResult(last_hidden=None, logits=pending.prepared.slot.logits.clone())
        except BaseException:
            self.release_finished_requests([pending.prepared.request_id])
            raise
        finally:
            pending.prepared.slot.busy = False
            if pending in self._pending:
                self._pending.remove(pending)

    def run_prefill_in_slot(self, model: RuntimeModel, batch: PrefillBatch,
                            *, buffer_slot: int) -> PrefillResult:
        prepared = self.prepare_prefill(model, batch, buffer_slot=buffer_slot)
        return self.reclaim_prepared_prefill(self.dispatch_prepared_prefill(model, batch, prepared))

    def run_prefill(self, model: RuntimeModel, batch: PrefillBatch) -> PrefillResult:
        return self.run_prefill_in_slot(model, batch, buffer_slot=0)

    def prepare_decode(self, model: RuntimeModel, batch: DecodeBatch,
                       *, buffer_slot: int) -> _Prepared:
        del model
        if len(batch.request_ids) != 1 or batch.seq_lens.numel() != 1:
            raise ValueError("Qwen3.8 decode accepts exactly one request")
        position = int(batch.seq_lens.reshape(-1)[0].item()) - 1
        if not 0 <= position < MAX_SEQUENCE:
            raise ValueError("Qwen3.8 decode position exceeds 128")
        slot = self._slot(self._decode_slots, buffer_slot)
        prepared = _Prepared(batch.request_ids[0], slot, position, "decode")
        self._reservations.append(prepared)
        return prepared

    def dispatch_prepared_decode(
        self, model: RuntimeModel, batch: DecodeBatch, prepared: _Prepared,
    ) -> _Pending:
        del model
        if (prepared.stage != "decode" or not prepared.slot.busy or prepared not in self._reservations
                or batch.request_ids != [prepared.request_id]):
            raise ValueError("invalid prepared Qwen3.8 decode")
        states = self._request_states.get(prepared.request_id)
        if states is None:
            prepared.slot.busy = False
            self._reservations.remove(prepared)
            raise ValueError(f"missing Qwen3.8 state for {prepared.request_id}")
        if batch.token_ids.numel() != 1:
            prepared.slot.busy = False
            self._reservations.remove(prepared)
            raise ValueError("Qwen3.8 decode requires one token")
        try:
            prepared.slot.token_ids.zero_()
            prepared.slot.token_ids[prepared.position] = int(batch.token_ids.reshape(-1)[0].item())
            dispatch = self._submit_l3(
                self._compiled.decode, prepared.slot.token_ids,
                ctypes.c_int64(prepared.position), *self._static_args, *states,
                prepared.slot.logits,
            )
        except BaseException:
            prepared.slot.busy = False
            self._reservations.remove(prepared)
            raise
        self._reservations.remove(prepared)
        pending = _Pending(prepared, dispatch)
        self._pending.append(pending)
        return pending

    def reclaim_prepared_decode(self, pending: _Pending) -> DecodeResult:
        try:
            pending.dispatch.wait()
            return DecodeResult(hidden_states=None, logits=pending.prepared.slot.logits.clone())
        except BaseException:
            self.release_finished_requests([pending.prepared.request_id])
            raise
        finally:
            pending.prepared.slot.busy = False
            if pending in self._pending:
                self._pending.remove(pending)

    def run_prepared_decode(self, model: RuntimeModel, batch: DecodeBatch,
                            prepared: _Prepared) -> DecodeResult:
        return self.reclaim_prepared_decode(self.dispatch_prepared_decode(model, batch, prepared))

    def run_decode(self, model: RuntimeModel, batch: DecodeBatch) -> DecodeResult:
        return self.run_prepared_decode(model, batch, self.prepare_decode(model, batch, buffer_slot=0))

    def close(self) -> None:
        error: BaseException | None = None
        try:
            for pending in tuple(self._pending):
                try:
                    pending.dispatch.wait()
                except BaseException as exc:
                    error = error or exc
                finally:
                    pending.prepared.slot.busy = False
                    self._pending.remove(pending)
            for prepared in self._reservations:
                prepared.slot.busy = False
            self._reservations.clear()
            try:
                self.release_finished_requests(tuple(self._request_states))
            except BaseException as exc:
                error = error or exc
            if self._l3_worker is not None:
                for tensor in self._l3_static_tensors.values():
                    try:
                        self._l3_worker.free_tensor(tensor)
                    except BaseException as exc:
                        error = error or exc
                self._l3_worker.close()
        finally:
            self._reset_l3_dispatch()
            self._l3_worker = None
        if error is not None:
            raise error
