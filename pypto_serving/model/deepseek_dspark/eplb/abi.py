# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Reject incompatible kernel exports before compiling or allocating expert weights."""

from .state import Placement

EPLB_ABI_VERSION = 1


def validate_kernel_contract(modules: dict[str, object], placement: Placement) -> None:
    """Check versioned HOST entry points and their frozen target-bank geometry."""
    entries = {
        "prefill_fwd": "l3_prefill_fwd_eplb",
        "decode_fwd": "l3_decode_fwd_eplb",
        "eplb_transfer": "make_eplb_transfer",
    }
    if "decode_fwd_dspark" in modules:
        entries["decode_fwd_dspark"] = "l3_decode_fwd_dspark_eplb"
    dimensions = {
        "N_RANKS": placement.ranks,
        "N_EXPERTS_GLOBAL": placement.logical_experts,
        "N_LOCAL": placement.local_slots,
        "EPLB_LAYERS": placement.layers,
        "EPLB_BANK_ROWS": placement.layers * placement.logical_experts,
        "META_SLOTS": (placement.local_slots + 15) // 16 * 16,
    }
    for name, entry in entries.items():
        module = modules.get(name)
        version = getattr(module, "EPLB_ABI_VERSION", None)
        if type(version) is not int or version != EPLB_ABI_VERSION:
            raise RuntimeError(f"DSpark EPLB ABI mismatch in {name}: expected version "
                               f"{EPLB_ABI_VERSION}, got {version!r}; update pypto-lib with serving")
        if not callable(getattr(module, entry, None)):
            raise RuntimeError(f"DSpark EPLB kernel export {name}.{entry} is missing or not callable")
        if name == "eplb_transfer":
            continue
        if getattr(module, "EPLB_ENABLED", None) is not True:
            raise RuntimeError(f"DSpark EPLB kernel {name} was imported without EPLB enabled")
        for dimension, expected in dimensions.items():
            actual = getattr(module, dimension, None)
            if type(actual) is not int or actual != expected:
                raise RuntimeError(f"DSpark EPLB kernel geometry mismatch in {name}.{dimension}: "
                                   f"expected {expected}, got {actual!r}")
