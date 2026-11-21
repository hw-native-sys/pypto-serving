# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise the real prefill runner's pending-handle and capture boundaries."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from pypto_serving.config.types import PrefillBatch
from pypto_serving.model.deepseek_lowlat import npu_runner
from pypto_serving.model.deepseek_lowlat.npu_runner import LowlatModelRunner
from pypto_serving.model.deepseek_lowlat.state import LowlatRequestState


def fixture(monkeypatch, phase=None):
    programs = SimpleNamespace(single=SimpleNamespace(compiled=object()), single_metadata={},
                               prefill=SimpleNamespace(compiled=object()), prefill_metadata={},
                               context=nullcontext)
    instance = LowlatModelRunner(programs, None)
    instance.state = LowlatRequestState("r", 1, 8)
    instance.generation, instance.chunk_capacity = 1, 8
    instance.run_config = object()
    instance.rope_cos, instance.rope_sin = None, None
    instance.cache = SimpleNamespace(tables_for=Mock(return_value=({}, object())), release=Mock(), owners={})
    instance.worker = Mock()
    handle = Mock()
    instance.worker.submit.return_value = handle
    instance.transfers = Mock(poisoned=False)
    instance.transfers.poison.side_effect = lambda: setattr(instance.transfers, "poisoned", True)
    instance.prefill_bindings = {name: SimpleNamespace(shape=(8, 1)) for name in
        ("sampled_ids", "attn_stage", "x_out", "dspark_target_hidden", "input_ids")}
    instance.weights = SimpleNamespace(lookup_embeddings=Mock(), prefill_bindings=Mock(return_value={}))
    instance.prefill_input = Mock()
    instance.prefill_input.fill.return_value = torch.zeros((8, 8, 4, 4096), dtype=torch.float32)
    instance.arena = Mock()
    instance._stage = Mock(return_value={"input_ids": object()})
    instance._read_sample = Mock(return_value=42)
    helper = SimpleNamespace(prepare_prefill_metadata=lambda *a, **k: {
        "input_ids_full": torch.zeros((8, 8), dtype=torch.int64),
        "local_request_ids": torch.zeros((8, 1), dtype=torch.int32)})
    def submit(worker, *args, **kwargs):
        if phase != "no_submit":
            accepted = worker.submit(programs.prefill.compiled)
            if phase != "no_wait":
                accepted.result()
        return {"sampled_ids": object()}
    adapter = SimpleNamespace(prepare_prefill_bindings=lambda *a: {}, submit_prefill=submit)
    monkeypatch.setattr(npu_runner.importlib, "import_module", lambda name: helper if name.endswith("prefill_metadata") else adapter)
    monkeypatch.setattr(npu_runner, "borrow_view", lambda *a: object())
    if phase == "cache_init":
        instance.cache.tables_for.side_effect = RuntimeError(phase)
    elif phase == "copy":
        instance._stage.side_effect = RuntimeError(phase)
    elif phase == "submit":
        instance.worker.submit.side_effect = RuntimeError(phase)
    elif phase == "result":
        handle.result.side_effect = RuntimeError(phase)
    elif phase == "capture":
        instance._read_sample.side_effect = RuntimeError(phase)
    batch = PrefillBatch(["r"], torch.arange(8), None, [8], [8], [0], [0],
                         prompt_lens=[8], block_ids_by_group=[{}])
    return instance, batch, handle


def test_prefill_publishes_and_captures_before_committing_cursor_and_releasing_handle(monkeypatch):
    instance, batch, handle = fixture(monkeypatch)
    def complete():
        assert instance.state.pending_handles == [handle]
        assert instance.state.prefilled_tokens == 0
    handle.result.side_effect = complete
    def capture(owner):
        assert instance.state.pending_handles == [handle]
        assert instance.state.prefilled_tokens == 0
        return 42
    instance._read_sample.side_effect = capture
    result = instance.run_prefill(None, batch)
    assert result.sampled_token_ids.tolist() == [42]
    assert instance.state.prefilled_tokens == instance.state.kv_committed_tokens == 8
    assert instance.state.pending_handles == []
    instance.transfers.poison.assert_not_called()
    instance.prefill_input.begin_copy.assert_called_once()
    instance.prefill_input.end_copy.assert_called_once()
    staged = instance._stage.call_args.args[1]
    assert staged["input_ids"].shape == (8, 8)
    assert "input_ids_full" not in staged
    instance.arena.copy.assert_not_called()
    instance.weights.prefill_bindings.assert_not_called()


@pytest.mark.parametrize("phase", ["cache_init", "copy", "submit", "result", "capture", "no_submit", "no_wait"])
def test_prefill_failure_poison_retains_accepted_handle_and_never_publishes_cursor(monkeypatch, phase):
    instance, batch, handle = fixture(monkeypatch, phase)
    with pytest.raises(RuntimeError):
        instance.run_prefill(None, batch)
    assert instance.state.prefilled_tokens == instance.state.kv_committed_tokens == 0
    instance.transfers.poison.assert_called_once()
    assert instance.transfers.poisoned
    expected = [handle] if phase in ("result", "capture", "no_wait") else []
    assert instance.state.pending_handles == expected
    if expected:
        def drain():
            assert instance.state.pending_handles == [handle]
            instance.cache.release.assert_not_called()
        handle.result.side_effect = drain
    instance.release_finished_requests(["r"])
    instance.cache.release.assert_called_once()
    with pytest.raises(RuntimeError):
        instance._request(["next"], 1)


def test_stage_recovery_keeps_both_transfers_poisoned_if_decode_recovery_fails(monkeypatch):
    instance, batch, _ = fixture(monkeypatch, "copy")
    instance.prefill_transfers = Mock(busy=False, poisoned=True)
    instance.prefill_transfers.recover_after_stage_failure.side_effect = (
        lambda: setattr(instance.prefill_transfers, "poisoned", False)
    )
    instance.prefill_transfers.poison.side_effect = lambda: setattr(
        instance.prefill_transfers, "poisoned", True
    )
    instance.transfers.busy = False
    instance.transfers.recover_after_prefill_failure.side_effect = RuntimeError("decode recovery failed")

    with pytest.raises(RuntimeError, match="decode recovery failed"):
        instance.run_prefill(None, batch)

    instance.prefill_transfers.recover_after_stage_failure.assert_called_once()
    instance.prefill_transfers.poison.assert_called_once()
    assert instance.prefill_transfers.poisoned
    instance.transfers.poison.assert_called()
    assert instance.transfers.poisoned
    assert not instance.state.pending_handles


def test_stage_recovery_skips_poison_only_after_both_transfers_reset(monkeypatch):
    instance, batch, _ = fixture(monkeypatch, "copy")
    instance.prefill_transfers = Mock(busy=False, poisoned=True)
    instance.prefill_transfers.recover_after_stage_failure.side_effect = (
        lambda: setattr(instance.prefill_transfers, "poisoned", False)
    )
    instance.transfers.busy = False
    instance.transfers.recover_after_prefill_failure.side_effect = (
        lambda: setattr(instance.transfers, "poisoned", False)
    )

    with pytest.raises(RuntimeError, match="copy"):
        instance.run_prefill(None, batch)

    instance.prefill_transfers.recover_after_stage_failure.assert_called_once()
    instance.transfers.recover_after_prefill_failure.assert_called_once()
    instance.transfers.poison.assert_not_called()
    assert not instance.prefill_transfers.poisoned
    assert not instance.transfers.poisoned
