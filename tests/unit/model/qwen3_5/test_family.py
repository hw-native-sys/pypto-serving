# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Real-index and positive-routing checks for the Qwen3.8 text family."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from pypto_serving.model.model_family import detect_model_family
from pypto_serving.model.model_loader import ModelLoader, Qwen35DirectoryLoader
from pypto_serving.model.qwen3_5.weight_spec import (
    QKV,
    WEIGHT_ORDER,
    _adapt_tensor,
    _stage_bank,
    validate_qwen35_weight_index,
)


def _real_checkpoint() -> tuple[Path, dict, dict]:
    if "MODEL_ROOT" not in os.environ:
        pytest.skip("MODEL_ROOT is required for real Qwen3.8 checkpoint checks")
    root = Path(os.environ["MODEL_ROOT"])
    if not (root / "model.safetensors.index.json").exists():
        pytest.skip("pinned Qwen3.8 checkpoint is unavailable")
    config = json.loads((root / "config.json").read_text())
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    return root, config, index


def test_positive_family_and_unknown_rejection() -> None:
    _, config, _ = _real_checkpoint()
    assert detect_model_family(config) == "qwen3_5"
    with pytest.raises(ValueError, match="Unsupported model family"):
        detect_model_family({"model_type": "unknown_candidate"})


def test_real_index_coverage_and_mutations() -> None:
    _, config, index = _real_checkpoint()
    coverage = validate_qwen35_weight_index(index, config)
    assert tuple(map(len, coverage.values())) == (851, 333, 15)
    assert set().union(*map(set, coverage.values())) == set(index)
    for mutation, name in (("missing", "model.language_model.embed_tokens.weight"),
                           ("extra", "model.language_model.unexpected.weight")):
        altered = dict(index)
        if mutation == "missing":
            altered.pop(name)
        else:
            altered[name] = "bogus"
        with pytest.raises((KeyError, ValueError), match=name):
            validate_qwen35_weight_index(altered, config)


def test_loader_route_and_bank_layout() -> None:
    root, config, index = _real_checkpoint()
    assert ModelLoader()._select_loader(type("Request", (), {"model_dir": str(root), "model_format": None})()) \
        .__class__ is Qwen35DirectoryLoader
    assert len(WEIGHT_ORDER) == 23 and QKV == 10240
    conv = torch.arange(3 * 4, dtype=torch.bfloat16).reshape(3, 1, 4)
    assert _adapt_tensor(conv, "linear_attn.conv1d.weight", (4, 3)).shape == (4, 3)
    assert validate_qwen35_weight_index(index, config)["consumed_text"][0] == "lm_head.weight"


def test_bank_padding_is_explicit_and_row_ordered() -> None:
    names = []

    def source(name):
        names.append(name)
        return torch.tensor([1, 2], dtype=torch.bfloat16)

    bank = _stage_bank([0, 4], "linear_attn.A_log", (1, 4), source)
    assert bank.shape == (2, 1, 4)
    assert bank.tolist() == [[[1, 2, 0, 0]], [[1, 2, 0, 0]]]
    assert names == [
        "model.language_model.layers.0.linear_attn.A_log",
        "model.language_model.layers.4.linear_attn.A_log",
    ]
