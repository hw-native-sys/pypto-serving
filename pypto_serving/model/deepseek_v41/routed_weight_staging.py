# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Keep routed source pages out of the V4.1 worker fork."""

import ctypes
import gc
import mmap
import os
from collections.abc import Callable, Iterable, Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.multiprocessing.reductions import StorageWeakRef

from pypto_serving.model.common.moe_offload import ROUTED_FIELDS

from .weight_loader import DecodeWeightBank


def _full_page_ranges(spans: Iterable[tuple[int, int]], page_size: int) -> Iterator[tuple[int, int]]:
    """Return page-aligned address and length pairs fully covered by routed bytes."""
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = merged[-1][0], max(end, merged[-1][1])
        else:
            merged.append((start, end))
    for start, end in merged:
        start = (start + page_size - 1) // page_size * page_size
        end = end // page_size * page_size
        if start < end:
            yield start, end - start


def _discard_sidecar_pages(ranges: Iterable[tuple[int, int]]) -> None:
    # The tensors keep the shared read-only mmap alive through their NumPy bases.
    madvise = ctypes.CDLL(None, use_errno=True).madvise
    madvise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
    for address, length in ranges:
        if madvise(address, length, mmap.MADV_DONTNEED) != 0:
            error = ctypes.get_errno()
            raise OSError(error, f"V4.1 sidecar MADV_DONTNEED failed: {os.strerror(error)}")


def _spill_routed_weights(
    tensors: dict[str, torch.Tensor], directory: Path,
) -> dict[str, tuple[str, torch.dtype, torch.Size]]:
    files = {}
    storages = {}
    for name in ROUTED_FIELDS:
        tensor = tensors[name]
        storages[name] = StorageWeakRef(tensor.untyped_storage())
        path = directory / name
        # A descriptor inherited across fork would keep the spill alive after unlink.
        with path.open("wb") as output:
            tensor.view(torch.uint8).numpy().tofile(output)
        files[name] = (str(path), tensor.dtype, tensor.shape)
        del tensors[name], tensor
    gc.collect()
    live = [name for name, storage in storages.items() if not storage.expired()]
    if live:
        raise RuntimeError(f"V4.1 routed weights still own storage before worker creation: {', '.join(live)}")
    return files


@contextmanager
def stage_routed_weights(bank: DecodeWeightBank) -> Iterator[Callable[[], None]]:
    """Prepare for fork and yield a function that restores eager weights in the parent."""
    for name in ROUTED_FIELDS:
        bank.rank_major_uploads.pop(name, None)
    with ExitStack() as cleanup:
        files = {}
        if bank.prepacked:
            # Sidecar fields occupy disjoint byte spans, so rounding inward
            # preserves pages shared with any retained field.
            spans = ((tensor.data_ptr(), tensor.data_ptr() + tensor.nbytes)
                     for name, tensor in bank.tensors.items() if name in ROUTED_FIELDS)
            ranges = tuple(_full_page_ranges(spans, mmap.PAGESIZE))
            _discard_sidecar_pages(ranges)
            cleanup.callback(_discard_sidecar_pages, ranges)
        else:
            directory = Path(cleanup.enter_context(TemporaryDirectory(prefix="pypto-v41-routed-")))
            files = _spill_routed_weights(bank.tensors, directory)

        def restore() -> None:
            for name, (path, dtype, shape) in files.items():
                bank.tensors[name] = torch.from_file(path, shared=False, size=shape.numel(), dtype=dtype).view(shape)

        try:
            yield restore
        finally:
            for name in ROUTED_FIELDS:
                bank.tensors.pop(name, None)
