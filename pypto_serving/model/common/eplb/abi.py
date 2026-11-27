# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Versioned EPLB kernel contract, specialized per model family by an `EplbAbiSpec`."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from .state import Placement


@dataclass(frozen=True)
class EplbAbiSpec:
    """One model family's frozen kernel surface.

    ``entries`` maps every required kernel module to the HOST entry serving
    dispatches from it. ``conditional_entries`` joins it only when the module
    is loaded (for example a speculative-only fused entry). ``geometry`` names
    the module attributes that must agree with the placement. A module listed
    in ``unversioned_modules`` carries the transfer program, which has no
    ``EPLB_ENABLED`` switch of its own.
    """

    version: int
    entries: Mapping[str, str]
    geometry: Mapping[str, Callable[[Placement], int]]
    conditional_entries: Mapping[str, str] = field(default_factory=dict)
    unversioned_modules: tuple[str, ...] = ()


def validate_kernel_contract(modules: Mapping[str, object], placement: Placement, *,
                             spec: EplbAbiSpec) -> None:
    """Check a model family's versioned HOST entry points and frozen geometry."""
    entries = dict(spec.entries)
    for name, entry in spec.conditional_entries.items():
        if name in modules:
            entries[name] = entry
    dimensions = {name: expected(placement) for name, expected in spec.geometry.items()}
    for name, entry in entries.items():
        module = modules.get(name)
        version = getattr(module, "EPLB_ABI_VERSION", None)
        if type(version) is not int or version != spec.version:
            raise RuntimeError(f"EPLB ABI mismatch in {name}: expected version "
                               f"{spec.version}, got {version!r}; update pypto-lib with serving")
        if not callable(getattr(module, entry, None)):
            raise RuntimeError(f"EPLB kernel export {name}.{entry} is missing or not callable")
        if name in spec.unversioned_modules:
            continue
        if getattr(module, "EPLB_ENABLED", None) is not True:
            raise RuntimeError(f"EPLB kernel {name} was imported without EPLB enabled")
        for dimension, expected in dimensions.items():
            actual = getattr(module, dimension, None)
            if type(actual) is not int or actual != expected:
                raise RuntimeError(f"EPLB kernel geometry mismatch in {name}.{dimension}: "
                                   f"expected {expected}, got {actual!r}")
