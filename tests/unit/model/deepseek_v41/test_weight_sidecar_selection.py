# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Select the prepacked V4.1 bank by page-cache residency or explicit override."""

import pytest
import torch
import safetensors

from pypto_serving.model.deepseek_v41 import weight_loader


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("residency", [None, 0.0, 0.949])
def test_cold_default_sidecar_falls_back_before_mapping(tmp_path, monkeypatch, world, residency):
    (tmp_path / f"pypto-deepseek-v41-decode-r{world}.safetensors").write_bytes(b"sidecar")
    monkeypatch.delenv("PYPTO_V41_PREPACKED_WEIGHTS", raising=False)
    monkeypatch.setattr(weight_loader, "_sample_file_page_cache_residency", lambda *_: residency)
    monkeypatch.setattr(safetensors, "safe_open", lambda *_, **__: pytest.fail("cold sidecar was opened"))

    layout = weight_loader.DecodeWeightLayout(2, 1, 1, 1, world=world, tp_size=1)
    assert weight_loader.load_prepacked_bank(tmp_path, layout, ("wq_a",)) is None


@pytest.mark.parametrize("world,residency,forced", [(2, 0.95, False), (2, 0.0, True),
                                                     (4, 0.95, False), (4, 0.0, True),
                                                     (8, 0.95, False), (8, 0.0, True)])
def test_hot_or_explicit_cold_sidecar_is_mapped(tmp_path, monkeypatch, world, residency, forced):
    sidecar = tmp_path / f"pypto-deepseek-v41-decode-r{world}.safetensors"
    sidecar.write_bytes(b"sidecar")
    if forced:
        monkeypatch.setenv("PYPTO_V41_PREPACKED_WEIGHTS", str(sidecar))
    else:
        monkeypatch.delenv("PYPTO_V41_PREPACKED_WEIGHTS", raising=False)
    monkeypatch.setattr(weight_loader, "_sample_file_page_cache_residency", lambda *_: residency)
    monkeypatch.setattr(weight_loader, "source_fingerprint", lambda *_: "fingerprint")
    monkeypatch.setattr(weight_loader, "_ALL", frozenset({"wq_a"}))
    source = torch.arange(12 * world, dtype=torch.uint8).reshape(2, world, 2, 3)
    monkeypatch.setattr(weight_loader, "_map_shared_prepacked_tensors", lambda *_: {"wq_a": source})

    class Reader:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def metadata(self):
            return {"format": weight_loader._FORMAT, "source_fingerprint": "fingerprint"}

        def keys(self):
            return ("wq_a",)

        def get_slice(self, _):
            return self

        def get_shape(self):
            return source.shape

    monkeypatch.setattr(safetensors, "safe_open", lambda *_, **__: Reader())
    layout = weight_loader.DecodeWeightLayout(2, 1, 1, 1, world=world, tp_size=1)
    bank = weight_loader.load_prepacked_bank(tmp_path, layout, ("wq_a",))
    assert bank.prepacked
    assert torch.equal(bank.tensors["wq_a"], source)
