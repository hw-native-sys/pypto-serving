# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Serving-owned composite boundary; no guessed lib signatures or CPU fallback.

An integration supplies complete layer entries (Attention plus FFN), resource
allocation and the model output boundary. Callbacks may enqueue work: wait()
returns only after all ranks have stopped accessing the supplied buffers.
"""
from dataclasses import dataclass
from typing import Callable, Mapping

from .execution_plan import LayerPlan, RankPlacement


class MissingCompositeInterface(NotImplementedError):
    """The selected lib revision cannot execute the requested serving segment."""


@dataclass(frozen=True)
class BuildOptions:
    """Worker-selected compilation settings passed unchanged to the adapter."""
    platform: str = "a5"
    pypto_build_dir: str = "build_output"
    use_compile_cache: bool = False


@dataclass(frozen=True)
class LayerState:
    """Opaque device state carried between composites, never converted on host."""
    residual: object
    pre_mix: object
    layout: str = "tp_replicated"


@dataclass(frozen=True)
class CompositeBindings:
    """Explicit adapter seam for a verified lib revision.

    Entries consume (LayerPlan, LayerState, step, resources, weights) and return
    LayerState. initialize consumes (embeddings, step, resources); output consumes
    (state, step, resources) and returns host logits in original request order.
    The adapter owns device uploads and lib ABI binding. Serving never expands
    packed FP4 or calls the sub-operators of a complete layer.
    allocate consumes (plan, device_ids, runtime, BuildOptions), including the
    worker's build directory and compile-cache choice, and returns (resources,
    num_pages). It also prepares global weights needed by initialize/output.

    No default implementation fabricates results. A supplied adapter must handle
    all DP partitions collectively, including empty partitions, and retain input
    objects until wait() completes. reset_request must clear every persistent
    cache and compressor slot for that request before returning.
    """
    revision: str
    entries: Mapping[tuple[str, str], Callable]
    initialize: Callable
    output: Callable
    allocate: Callable
    prepare_weights: Callable
    reset_request: Callable
    wait: Callable
    close: Callable
    cache_groups: tuple = ()
    input_layout: str = "tp_replicated"
    output_layout: str = "tp_replicated"

    def require(self, layers: tuple[LayerPlan, ...], placement: RankPlacement) -> None:
        if not self.revision:
            raise ValueError("composite bindings must identify the validated lib revision")
        missing = sorted({f"{phase}/{layer.mode}" for phase in ("prefill", "decode")
                          for layer in layers if not callable(self.entries.get((phase, layer.mode)))})
        if missing:
            raise MissingCompositeInterface("missing complete layer composites: " + ", ".join(missing))
        for name in ("initialize", "output", "allocate", "prepare_weights", "reset_request", "wait", "close"):
            if not callable(getattr(self, name)):
                raise MissingCompositeInterface(f"missing composite resource/output operation: {name}")
        if self.input_layout not in ("tp_replicated", "tp_local_token") or self.output_layout != self.input_layout:
            raise MissingCompositeInterface("layer entries must preserve their declared residual/pre_mix layout")
        if (placement.tp_size, placement.dp_size, placement.ep_size) != (4, 2, 8):
            raise ValueError("V4.1 serving currently targets TP4/DP2/EP8")


def load_composite_bindings() -> CompositeBindings:
    """Fail before resource allocation until a complete adapter is implemented.

    This is an intentional integration placeholder, not a discovery heuristic:
    importing a lib module or finding a function does not establish its ABI.
    """
    raise MissingCompositeInterface(
        "V4.1 serving execution requires verified lib composite bindings: "
        "packed-FP4 full-layer prefill/decode for all modes, initial residual/pre_mix, "
        "cache allocation/reset/completion and final HC/Norm/LM head. "
        "Track pypto-lib #1205, #1275 and #1287; no Torch fallback is enabled."
    )
