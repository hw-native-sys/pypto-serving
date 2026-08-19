# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Shared weight-staging test helpers: a byte-level fingerprint both model families use.
from __future__ import annotations

import hashlib
from collections.abc import Mapping

import pytest
import torch


def _digest(tensor: torch.Tensor) -> str:
    """Hash the tensor's bytes, not its repr, so a dtype-preserving change still shows."""
    contiguous = tensor.detach().cpu().contiguous()
    return hashlib.sha256(contiguous.view(torch.uint8).numpy().tobytes()).hexdigest()


@pytest.fixture
def fingerprint_tensors():
    """Return a function mapping tensors to ``name -> (shape, dtype, sha256)``.

    The refactor in #163 has to keep the staged output **byte-identical**, because the
    fused kernels treat the stacked layout as an output contract. Comparing shapes and a
    content hash is what turns that from an intention into something a test can fail on;
    comparing ``torch.equal`` alone would miss a dtype change that happens to round-trip.
    """

    def _fingerprint(tensors: Mapping[str, torch.Tensor]) -> dict[str, tuple[tuple[int, ...], str, str]]:
        return {
            name: (tuple(int(dim) for dim in tensor.shape), str(tensor.dtype), _digest(tensor))
            for name, tensor in sorted(tensors.items())
        }

    return _fingerprint
