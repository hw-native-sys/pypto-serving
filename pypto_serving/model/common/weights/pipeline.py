# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Orchestration: read a layer, pack it into its slab slice, release it, move on.

The staging order is what bounds peak host memory, and the two families arrived at opposite
answers for a reason:

* **DeepSeek V4 stages serially.** Packing one layer allocates ~8 GB of intermediates — 256
  routed experts, each stacked and rank-replicated — so N workers multiply that peak and
  contend on memory bandwidth. Serial packing keeps the working set at one layer and lets the
  disk prefetcher run.
* **Qwen stages on a thread pool**, with each worker pinned to one torch thread: its layers are
  small, so the read latency dominates and overlapping it wins.

Neither is the "right" default, so the policy is a parameter rather than a decision made here.
"""

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class StagingPolicy:
    """How many layers are staged at once, and how each worker treats torch's own threads.

    ``workers=1`` is not merely "a pool of one": it takes the path with no pool at all, so a
    family that must not overlap cannot be made to by a scheduling accident.

    ``pin_torch_threads`` sets each worker's torch thread count to 1 for the duration. Without
    it, N staging threads each fan out into torch's own pool and oversubscribe the machine —
    the copies then run slower than serially, which reads as "threading did not help" rather
    than as a configuration error.
    """

    workers: int = 1
    pin_torch_threads: bool = True

    def __post_init__(self) -> None:
        if self.workers < 1:
            raise ValueError(f"staging needs at least one worker, got {self.workers}")


def stage_layers(
    layer_ids: Sequence[int],
    *,
    stage: Callable[[int], None],
    policy: StagingPolicy,
    on_layer_done: Callable[[int], None] | None = None,
) -> None:
    """Run ``stage(layer_id)`` for every layer, serially or on a pool per *policy*.

    ``stage`` is expected to write into a preallocated destination and to drop its own
    intermediates before returning; this function owns only the concurrency. Errors are not
    swallowed — the first one propagates, and the pool is joined on the way out, so a failure
    cannot leave a half-written slab being used as if it were complete.
    """
    if policy.workers == 1:
        for layer_id in layer_ids:
            stage(int(layer_id))
            if on_layer_done is not None:
                on_layer_done(int(layer_id))
        return

    def _run(layer_id: int) -> int:
        if policy.pin_torch_threads:
            previous = torch.get_num_threads()
            torch.set_num_threads(1)
            try:
                stage(int(layer_id))
            finally:
                torch.set_num_threads(previous)
        else:
            stage(int(layer_id))
        return int(layer_id)

    with ThreadPoolExecutor(max_workers=min(policy.workers, len(layer_ids))) as pool:
        for layer_id in pool.map(_run, [int(layer_id) for layer_id in layer_ids]):
            if on_layer_done is not None:
                on_layer_done(layer_id)


def stage_and_release(
    layer_ids: Sequence[int],
    *,
    load: Callable[[int], Mapping[str, torch.Tensor]],
    write: Callable[[int, Mapping[str, torch.Tensor]], None],
    policy: StagingPolicy,
    on_layer_done: Callable[[int], None] | None = None,
) -> None:
    """Load, write and then drop each layer's raw tensors, one layer at a time.

    The release is the point: holding every layer's raw tensors until the end costs a second
    copy of the model at the peak, which is what the eager path did. Here each layer's mapping
    is a local of one call, so it becomes unreachable when that call returns and the allocator
    can reuse the pages — the peak stays at roughly one layer per worker. Nothing is deleted
    explicitly, because a rebind before returning would only look like it was doing that.
    """

    def _stage(layer_id: int) -> None:
        write(layer_id, load(layer_id))

    stage_layers(layer_ids, stage=_stage, policy=policy, on_layer_done=on_layer_done)
