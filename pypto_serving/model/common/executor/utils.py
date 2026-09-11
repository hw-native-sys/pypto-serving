# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from pypto.runtime import RunConfig


def backend_type_for_platform(platform: str):
    """Map a platform name to the PyPTO backend enum used by runtime config."""
    from pypto.backend import BackendType

    if platform.startswith("a5"):
        return BackendType.Ascend950
    return BackendType.Ascend910B


def build_pypto_run_config(
    *,
    platform: str,
    device_ids: Sequence[int],
    pypto_build_dir: str | None = None,
    aicpu_thread_num: int = 4,
    num_sub_workers: int = 0,
) -> RunConfig:
    """Build a PyPTO ``RunConfig`` (with ``DistributedConfig``) for compile/run.

    This carries the executor's platform/device/backend policy plus the
    distributed-compile topology, and is the base config handed to
    :class:`KernelCompiler` (which only layers on per-model knobs via
    ``dataclasses.replace``). ``device_id`` is the first of ``device_ids``.

    ``pypto_build_dir`` is an explicit diagnostic/output request and bypasses
    PyPTO's JIT cache. Ordinary serving leaves it unset; private builds then use
    PyPTO's unique directories. KernelCompiler adds the kernel name to an
    explicitly requested base directory.
    """
    from pypto.ir.distributed_compiled_program import DistributedConfig  # noqa: PLC0415
    from pypto.runtime import RunConfig  # noqa: PLC0415

    device_ids = list(device_ids)
    return RunConfig(
        platform=platform,
        device_id=device_ids[0],
        backend_type=backend_type_for_platform(platform),
        codegen_only=True,
        save_kernels=pypto_build_dir is not None,
        save_kernels_dir=pypto_build_dir,
        distributed_config=DistributedConfig(
            device_ids=device_ids,
            num_sub_workers=num_sub_workers,
            aicpu_thread_num=aicpu_thread_num,
        ),
    )


def rope_tables(max_seq: int, head_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Build cosine and sine RoPE lookup tables for the configured context."""
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
    freqs = torch.outer(torch.arange(max_seq, dtype=torch.float32), inv_freq)
    cos_half = torch.cos(freqs)
    sin_half = torch.sin(freqs)
    return torch.cat([cos_half, cos_half], dim=-1), torch.cat([sin_half, sin_half], dim=-1)


def round_up(value: int, multiple: int) -> int:
    """Round ``value`` up to the nearest multiple of ``multiple``."""
    return ((value + multiple - 1) // multiple) * multiple
