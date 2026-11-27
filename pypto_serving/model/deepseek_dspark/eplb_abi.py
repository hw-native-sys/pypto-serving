# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""DSpark's EPLB kernel surface: entries, versions, and the frozen target-bank geometry."""

from pypto_serving.model.common.eplb.abi import EplbAbiSpec
from pypto_serving.model.common.eplb.abi import validate_kernel_contract as validate_contract
from pypto_serving.model.common.eplb.state import Placement

EPLB_ABI_VERSION = 1

DSPARK_EPLB_ABI = EplbAbiSpec(
    version=EPLB_ABI_VERSION,
    entries={
        "prefill_fwd": "l3_prefill_fwd_eplb",
        "decode_fwd": "l3_decode_fwd_eplb",
        "eplb": "make_eplb_transfer",
    },
    # The fused speculative target adds its own entry; it lives in the same
    # modules dict only when the speculative stack is compiled.
    conditional_entries={"decode_fwd_dspark": "l3_decode_fwd_dspark_eplb"},
    # The transfer program has no EPLB_ENABLED switch of its own.
    unversioned_modules=("eplb",),
    geometry={
        "N_RANKS": lambda placement: placement.ranks,
        "N_EXPERTS_GLOBAL": lambda placement: placement.logical_experts,
        "N_LOCAL": lambda placement: placement.local_slots,
        "EPLB_LAYERS": lambda placement: placement.layers,
        "EPLB_BANK_ROWS": lambda placement: placement.layers * placement.logical_experts,
        "META_SLOTS": lambda placement: (placement.local_slots + 15) // 16 * 16,
    },
)


def validate_kernel_contract(modules: dict[str, object], placement: Placement) -> None:
    """Check the DSpark kernel exports and geometry before compiling any kernel."""
    validate_contract(modules, placement, spec=DSPARK_EPLB_ABI)
