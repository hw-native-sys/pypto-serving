# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Model-family-neutral expert placement, load windows and device maintenance.

A model family supplies three things: a frozen `EplbAbiSpec` for its kernel
exports, an `ExpertBankSpec` for its resident weight tensors, and the executor
and runner wiring that allocates the four EPLB tensors and submits the forward
entries. Everything that plans, validates, or applies a migration lives here.
"""
