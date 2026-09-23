# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""NZ fractal reordering for a weight's trailing ``[R, C]``.

NZ is an Ascend tiling scheme, not a DeepSeek concept, so it lives beside the other
family-agnostic weight transforms rather than under a model package. The reference
implementation is pypto-lib's ``models/deepseek_v4_flash_mtp/utils.py::pack_nz`` — this copy
must stay byte-for-byte identical to it, since a kernel compiled from that source is what
reads the bytes this function produces.
"""

import torch

# 32-byte C0 lines, 16-row fractals: the a2a3/a5 NZ tiling pypto's ``pl.NZ`` layout targets.
_NZ_C0_BYTES = 32
_NZ_FRACTAL_ROWS = 16


def pack_nz(logical: torch.Tensor) -> torch.Tensor:
    """Reorder the trailing ``[R, C]`` of a row-major tensor into NZ fractal order.

    ``c0`` is derived from ``logical``'s own dtype, so the caller must cast to the kernel's
    declared dtype before packing — packing in the wrong dtype silently picks the wrong block
    size instead of failing.
    """
    rows, cols = logical.shape[-2:]
    c0 = _NZ_C0_BYTES // logical.element_size()
    if rows % _NZ_FRACTAL_ROWS:
        raise ValueError(f"NZ needs {_NZ_FRACTAL_ROWS}-row fractals, got {rows} rows")
    if cols % c0:
        raise ValueError(f"NZ needs whole C0 lines of {c0} elements, got {cols} cols")
    blocked = logical.reshape(-1, rows // _NZ_FRACTAL_ROWS, _NZ_FRACTAL_ROWS, cols // c0, c0)
    return blocked.permute(0, 3, 1, 2, 4).contiguous().reshape(logical.shape)
