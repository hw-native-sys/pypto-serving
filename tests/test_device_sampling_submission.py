# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
QWEN = ROOT / "pypto-lib" / "models" / "qwen3" / "14b"
REAL_VOCAB = 151936
PADDED_VOCAB = 152064

pytestmark = pytest.mark.skipif(
    not QWEN.is_dir(),
    reason="pypto-lib submodule is not checked out",
)


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_device_sampling_is_limited_by_runtime_vocab_size() -> None:
    dispatch = _source(ROOT / "examples" / "model" / "qwen3_14b" / "runner" / "qwen3_l3_dispatch.py")
    executor = _source(ROOT / "examples" / "model" / "qwen3_14b" / "runner" / "npu_executor.py")
    runner = _source(ROOT / "examples" / "model" / "qwen3_14b" / "runner" / "npu_runner.py")
    config = _source(QWEN / "config.py")
    greedy = _source(QWEN / "greedy_sample.py")

    assert "valid_vocab_size" not in dispatch
    assert "REAL_VOCAB = 151936" in config
    assert "REAL_VOCAB" in executor
    assert "lm_head_weight[:1].expand" in executor
    assert "valid_vocab_size" not in runner
    assert "valid_vocab_size" not in greedy
    assert "REAL_NUM_FULL_VOCAB_CHUNKS" in greedy
    assert "REAL_VOCAB_TAIL" in greedy
    assert "token_id >= pl.cast(REAL_VOCAB" in greedy


def test_device_greedy_tie_break_matches_host_argmax() -> None:
    greedy = _source(QWEN / "greedy_sample.py")
    decode = _source(QWEN / "decode_layer.py")

    for source in (greedy, decode):
        assert "local_token = pl.cast(0, pl.INT32)" in source
        assert "scan_c = (" in source
        assert "scan_t = (" in source
        assert "local_token = pl.cast(scan_t, pl.INT32)" in source
        assert "if val == best_val:" in source


def test_prefill_keeps_sampling_in_standalone_device_kernel() -> None:
    prefill = _source(QWEN / "prefill_fwd.py")
    runner = _source(ROOT / "examples" / "model" / "qwen3_14b" / "runner" / "npu_runner.py")

    assert "_greedy_sample_inline" not in prefill
    assert "_token_embed_inline" not in prefill
    assert "compiled.greedy_sample" in runner
    assert "_maybe_run_sample_embed(" in runner


def _device_greedy_argmax_with_clamp(logits):
    best_val = max(logits)
    token_id = 0
    for idx in range(len(logits) - 1, -1, -1):
        if logits[idx] == best_val:
            token_id = idx
    return 0 if token_id >= REAL_VOCAB else token_id


def test_padded_vocab_device_argmax_numeric_semantics() -> None:
    torch = pytest.importorskip("torch")

    logits = torch.full((3, PADDED_VOCAB), -1000.0)

    # Real-token tie: host torch.argmax chooses the smallest real token id.
    logits[0, 7] = 5.0
    logits[0, 42] = 5.0

    # Padded rows are expected to duplicate token 0 exactly, so the full-VOCAB
    # device scan must still match host argmax over REAL_VOCAB.
    logits[1, 0] = 6.0
    logits[1, REAL_VOCAB:] = logits[1, 0]

    # Defensive clamp case: if a padded row ever wins outright, the device path
    # maps it back to token 0 instead of returning an invalid token id.
    logits[2, 5] = 1.0
    logits[2, REAL_VOCAB + 3] = 9.0

    device_ids = torch.tensor(
        [_device_greedy_argmax_with_clamp(row.tolist()) for row in logits],
        dtype=torch.int64,
    )

    assert torch.equal(device_ids[:2], torch.argmax(logits[:2, :REAL_VOCAB], dim=-1))
    assert device_ids.tolist() == [7, 0, 0]
